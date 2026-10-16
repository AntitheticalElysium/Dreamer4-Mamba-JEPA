"""Mobs at four levels (FROZEN_STATE_CONTROL_PREREG.md E1 cross-fitted + E2, Amendment 1): DECODABILITY of the input, PERSISTENCE
(stay / vacate), correct FUTURE LOCATIONS (arrive) and COHERENT stochastic SAMPLES (object counts). Evaluation only; exact simulator
labels evaluate, never train.

v2 (10-17, after Codex's audit ../20261017_frozen_state_audit/ASSESSMENT.md; no v1 number was ever reported):
- tie-grouped AP (metrics_tied) with an exactly replicated episode bootstrap;
- a READER-MATCHED persistence baseline: the root's tokens moved by the exact camera shift, with a cross-fitted refit reader
  (the transferred-reader version is kept beside it);
- alignment confidence: a unit's camera shift is "confident" iff its tile-mismatch minimum is unique (homogeneous maps tie);
- the sibling-future frequency is a noisy REFERENCE, not a ceiling (Codex's counterexample);
- coherence: all 4 router samples are scored; object counts with fit-only rate-matched thresholds.

Units: every diagnosis root (144 episodes) x each alive future at depth 1, the B4 setting (root's 2-frame context, the recorded
first action). States: EXPECTED next state, 4 router SAMPLES, COPY (root state), PERS (root tokens moved by the exact camera shift;
entering cells zero), REAL (encoded real next frame).
Cell classes per mob (exact camera shift via objective_audit.shift_of; sources() maps next cells to root cells):
  stay (static / shifted camera), arrive (a mob in a world cell that had none), enter (came into view), vacated, empty.
Scores: refit = ridge lambda 1e-2 on the state's map tokens -> next-cell mob, 5-fold cross-fitting grouped by episode (folds of
consequence_check.py); samples use the reader fitted on sample 0. transfer = the world's Tier B real-state reader.
Usage: location_dynamics.py <label>=<encoder spec>,<predictor ckpt>,<probes2> ...   -> erasure/locdyn2_<label>.pt
       location_dynamics.py --analyse <label> ... [--pairs a:b,...] [--tag T]       -> erasure/locdyn2_summary_<T>.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_encoder as SE                                                   # noqa: E402
import spatial_eval as EV                                                      # noqa: E402
import tierb_eval as TB                                                        # noqa: E402
from consequence_check import folds, FOLDS, auc                                # noqa: E402
from metrics_tied import ap_grouped_boot                                       # noqa: E402

import torch                                                                   # noqa: E402

import objective_audit as OA                                                   # noqa: E402

OUTD = EV.OUT / 'erasure'
B = 4000
MOBS = (('zombie', 1), ('cow', 2))


def mismatches(t0, t1):
    """tile mismatch of every candidate shift, exactly as objective_audit.shift_of computes it"""
    a, b = t0.view(7, 9), t1.view(7, 9)
    out = []
    for dr, dc in OA.SH:
        rs = slice(max(0, -dr), 7 - max(0, dr)); cs = slice(max(0, -dc), 9 - max(0, dc))
        rt = slice(max(0, dr), 7 + min(0, dr)); ct = slice(max(0, dc), 9 + min(0, dc))
        out.append(float((b[rs, cs] != a[rt, ct]).float().mean()))
    return out


@torch.no_grad()
def collect(enc_spec, pred_ck, probe_file):
    enc = EV.PatchState(enc_spec)
    sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
    pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
    probes = torch.load(probe_file, weights_only=False)
    g = torch.Generator(device=EV.DEV).manual_seed(20261071)
    U = {}
    add = lambda k, v: U.setdefault(k, []).append(v)
    rows = (r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False))
    for ri, r in enumerate(rows):
        alive = [f for f in range(5) if not bool(r['future_dead'][f][0])]
        if not alive:
            continue
        ctx = enc(r['context'][-2:])
        exp, smp, _, _ = TB.step(pred, ctx[0:1].expand(4, -1, -1), ctx[1:2].expand(4, -1, -1), r['future_actions'][0:1].repeat(4), g)
        real = enc(torch.stack([r['future_frames'][f][0] for f in alive]))[:, :63].cpu()
        t0, l0, _ = OA.layout(r['root_visible'][None]); t0, l0 = t0[0], l0[0]
        t1, l1, _ = OA.layout(torch.stack([r['future_visible'][f][0] for f in alive]))
        ex = torch.stack([((l1 & b) > 0) for _, b in MOBS], 1).float()
        copy = ctx[1, :63].cpu()
        for i, f in enumerate(alive):
            mm = mismatches(t0, t1[i]); j = min(range(len(mm)), key=lambda k: mm[k])
            dr, dc = OA.SH[j]
            assert (dr, dc) == OA.shift_of(t0, t1[i])
            src = OA.sources(dr, dc)
            others = [k for k in range(len(alive)) if k != i]
            sib = ex[others].mean(0) if others else torch.full((2, 63), float('nan'))
            pers = torch.where((src >= 0)[:, None], copy[src.clamp_min(0)], torch.zeros_like(copy))
            add('seed', int(r['seed'])); add('root', ri); add('future', f); add('moved', (dr, dc) != (0, 0))
            add('confident', sorted(mm)[1] > sorted(mm)[0])
            add('exp', exp[0, :63].cpu().half()); add('smp', smp[:, :63].cpu().half()); add('copy', copy.half()); add('pers', pers.half())
            add('real', real[i].half()); add('src', src); add('srcok', src >= 0)
            add('l0src', torch.where(src >= 0, l0[src.clamp_min(0)], torch.zeros_like(l0))); add('l1', l1[i])
            add('sib_z', sib[0]); add('sib_c', sib[1])
            add('cnt_sib', torch.stack([ex[:, k].sum(-1) for k in range(2)], 1))          # [n_alive, 2] exact sibling counts
    T = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) and k != 'cnt_sib' else (v if k == 'cnt_sib' else torch.tensor(v))) for k, v in U.items()}
    return T, probes


def score_all(T, probes):
    n, D = len(T['seed']), T['exp'].shape[-1]
    fold = folds(T['seed']); fc = fold.repeat_interleave(63)
    S = {}
    for m, b in MOBS:
        y = ((T['l1'] & b) > 0).reshape(-1)
        yf = y.float()[:, None]
        for st in ('exp', 'copy', 'pers', 'real', 'smp'):
            X = (T[st][:, 0] if st == 'smp' else T[st]).reshape(-1, D).float()
            out = torch.empty(len(X)); th = torch.empty(n)
            Xs = T['smp'].reshape(-1, D).float() if st == 'smp' else None
            out4 = torch.empty(n, 4, 63) if st == 'smp' else None
            for k in range(FOLDS):
                tr = fc != k
                W = EV.ridge(X[tr], yf[tr])
                out[fc == k] = EV.apply(W, X[fc == k])[:, 0]
                srt = EV.apply(W, X[tr])[:, 0].sort(descending=True).values          # fit-only rate-matched threshold
                th[fold == k] = float(srt[min(max(int(y[tr].sum()) - 1, 0), len(srt) - 1)])
                if st == 'smp':
                    u = torch.nonzero(fold == k)[:, 0]
                    out4[u] = EV.apply(W, T['smp'][u].reshape(-1, D).float())[:, 0].view(len(u), 4, 63)
            S[f'{m}_{st}_refit'] = out.view(n, 63)
            S[f'{m}_{st}_th'] = th
            if st == 'smp':
                S[f'{m}_smp4_refit'] = out4
            if st in ('exp', 'copy', 'real'):
                S[f'{m}_{st}_transfer'] = EV.apply(probes[m], X)[:, 0].view(n, 63)
        moved = torch.gather(S[f'{m}_copy_transfer'], 1, T['src'].clamp_min(0))
        S[f'{m}_persist_transfer'] = torch.where(T['srcok'], moved, torch.full_like(moved, -1e4))
    return S


def analyse(labels, pairs, tag):
    D = {lb: torch.load(OUTD / f'locdyn2_{lb}.pt', weights_only=False) for lb in labels}
    T0 = D[labels[0]]['units']
    for lb in labels:
        assert torch.equal(D[lb]['units']['seed'], T0['seed']) and torch.equal(D[lb]['units']['future'], T0['future'])
    eps = sorted(set(T0['seed'].tolist())); eo = {e: i for i, e in enumerate(eps)}
    ep_u = torch.tensor([eo[int(s)] for s in T0['seed']])
    ep = ep_u[:, None].expand(-1, 63).reshape(-1)
    g = torch.Generator().manual_seed(20261104)
    draws = torch.randint(len(eps), (B, len(eps)), generator=g)
    mult = torch.zeros(B, len(eps), dtype=torch.long).scatter_add_(1, draws, torch.ones(B, len(eps), dtype=torch.long))
    none = mult[:0]
    moved = T0['moved'][:, None].expand(-1, 63).reshape(-1); conf = T0['confident'][:, None].expand(-1, 63).reshape(-1)
    res = {'units': len(T0['seed']), 'episodes': len(eps), 'confident_alignment_units': int(T0['confident'].sum()),
           'moved_units': int(T0['moved'].sum()), 'ap': 'metrics_tied.ap_grouped (tie-grouped), episode bootstrap exactly replicated'}
    stats = {}
    for m, b in MOBS:
        ex1 = ((T0['l1'] & b) > 0).reshape(-1)
        at = (((T0['l0src'] & b) > 0) & T0['srcok']).reshape(-1)
        srcok = T0['srcok'].reshape(-1)
        cat = {'stay': ex1 & at, 'arrive': ex1 & ~at & srcok, 'enter': ex1 & ~srcok, 'vacated': ~ex1 & at}
        res[f'{m}_counts'] = {k: int(v.sum()) for k, v in cat.items()} | {'stay_shifted': int((cat['stay'] & moved).sum()),
                                                                          'arrive_confident': int((cat['arrive'] & conf).sum())}
        sib = T0[f'sib_{m[0]}'].reshape(-1); has_sib = ~torch.isnan(sib)
        for lb in labels:
            S = D[lb]['scores']
            sc = {k: S[f'{m}_{k}'].reshape(-1) for k in ('exp_refit', 'smp_refit', 'copy_refit', 'pers_refit', 'real_refit',
                                                          'exp_transfer', 'persist_transfer')}
            r = {}
            # 1. decodability of the input / of the real next frame (absolute, reader-matched)
            r['ap_next_cells'] = {k: round(ap_grouped_boot(v, ex1, ep, none)[0], 4) for k, v in sc.items()}
            # E1: relative next-cell linear decodability = AP(expected) / AP(copy), numerator and denominator per draw
            a_e, a_c = ap_grouped_boot(sc['exp_refit'], ex1, ep, mult), ap_grouped_boot(sc['copy_refit'], ex1, ep, mult)
            r['relative_next_cell_decodability'] = round(a_e[0] / a_c[0], 4)
            q = a_e[1] / a_c[1]; stats[(lb, m, 'rel')] = q; q = q[~torch.isnan(q)]
            r['relative_next_cell_decodability_ci95'] = [round(float(q.quantile(.025)), 4), round(float(q.quantile(.975)), 4)]
            stats[(lb, m, 'abs_exp')] = a_e[1]
            # 2. persistence: stay AP and stay-vs-vacated AUC
            for nm, pos in (('stay_static', cat['stay'] & ~moved), ('stay_shifted', cat['stay'] & moved)):
                keep = pos | ~ex1
                r[f'FL_{nm}'] = {k: round(ap_grouped_boot(v[keep], pos[keep], ep[keep], none)[0], 4) for k, v in sc.items()}
                kl = keep & has_sib
                r[f'FL_{nm}']['sibling_freq_reference'] = round(ap_grouped_boot(sib[kl], pos[kl], ep[kl], none)[0], 4) if bool(pos[kl].any()) else None
            vm = cat['stay'] | cat['vacated']
            r['FL_vacate_auc'] = {k: round(auc(v[vm], cat['stay'][vm]), 4) for k, v in sc.items()}
            r['FL_vacate_auc']['sibling_freq_reference'] = round(auc(sib[vm & has_sib], cat['stay'][vm & has_sib]), 4)
            # 3. future locations: arrive AP (all units; confident-alignment units as sensitivity)
            for nm, sub in (('arrive', torch.ones_like(conf)), ('arrive_confident', conf)):
                pos = cat['arrive'] & sub; keep = (pos | ~ex1) & sub
                r[f'FL_{nm}'] = {}
                for k in sc:
                    boot = mult if (nm == 'arrive' and k in ('exp_refit', 'pers_refit', 'persist_transfer')) else none
                    full, bb = ap_grouped_boot(sc[k][keep], pos[keep], ep[keep], boot)
                    r[f'FL_{nm}'][k] = round(full, 4)
                    if len(bb):
                        stats[(lb, m, f'arrive_{k}')] = bb
                kl = keep & has_sib
                r[f'FL_{nm}']['sibling_freq_reference'] = round(ap_grouped_boot(sib[kl], pos[kl], ep[kl], none)[0], 4)
            for base in ('pers_refit', 'persist_transfer'):
                d = stats[(lb, m, 'arrive_exp_refit')] - stats[(lb, m, f'arrive_{base}')]; d = d[~torch.isnan(d)]
                r[f'FL_arrive_exp_minus_{base}_ci95'] = [round(float(d.quantile(.025)), 4), round(float(d.quantile(.975)), 4)]
            # 4. coherent samples: object counts (fit-only rate-matched thresholds), per unit and sample
            n_real = ex1.view(-1, 63).sum(-1).float()
            n_pers = at.view(-1, 63).sum(-1).float()                                   # exact persistence count (no motion)
            cnt = lambda s4, th: (s4 > th.view(-1, *([1] * (s4.dim() - 1)))).flatten(1 if s4.dim() == 2 else 2).sum(-1).float()
            n_smp = cnt(S[f'{m}_smp4_refit'], S[f'{m}_smp_th'])                         # [n,4]
            n_exp = cnt(S[f'{m}_exp_refit'], S[f'{m}_exp_th']); n_dec = cnt(S[f'{m}_real_refit'], S[f'{m}_real_th'])
            sib_lo = torch.tensor([float(c[:, 0 if m == 'zombie' else 1].min()) for c in T0['cnt_sib']])
            sib_hi = torch.tensor([float(c[:, 0 if m == 'zombie' else 1].max()) for c in T0['cnt_sib']])
            sib_sd = torch.tensor([float(c[:, 0 if m == 'zombie' else 1].float().std()) if len(c) > 1 else float('nan') for c in T0['cnt_sib']])
            r['coherence_counts'] = {
                'mean_abs_count_error': {'samples': round(float((n_smp - n_real[:, None]).abs().mean()), 4),
                                         'expected': round(float((n_exp - n_real).abs().mean()), 4),
                                         'decoded_real_reader_floor': round(float((n_dec - n_real).abs().mean()), 4),
                                         'exact_persistence': round(float((n_pers - n_real).abs().mean()), 4)},
                'samples_outside_sibling_count_range': round(float(((n_smp < sib_lo[:, None]) | (n_smp > sib_hi[:, None])).float().mean()), 4),
                'decoded_real_outside_sibling_count_range': round(float(((n_dec < sib_lo) | (n_dec > sib_hi)).float().mean()), 4),
                'mean_sd_over_4_samples': round(float(n_smp.std(1).mean()), 4),
                'mean_sd_over_sibling_futures_exact': round(float(sib_sd[~torch.isnan(sib_sd)].mean()), 4),
                'mean_count': {'samples': round(float(n_smp.mean()), 4), 'exact': round(float(n_real.mean()), 4)}}
            res[f'{lb}_{m}'] = r
        for a_, b_ in pairs:
            for key in ('rel', 'abs_exp'):
                d = stats[(a_, m, key)] - stats[(b_, m, key)]; d = d[~torch.isnan(d)]
                res[f'{a_}_minus_{b_}_{m}_{key}'] = [round(float(d.quantile(.025)), 4), round(float(d.quantile(.975)), 4)]
    print(json.dumps(res, indent=1))
    (OUTD / f'locdyn2_summary_{tag}.json').write_text(json.dumps(res, indent=1))


def main():
    OUTD.mkdir(exist_ok=True)
    if sys.argv[1] == '--analyse':
        args = sys.argv[2:]
        pairs, tag = [], 'all'
        if '--pairs' in args:
            i = args.index('--pairs'); pairs = [tuple(p.split(':')) for p in args[i + 1].split(',')]; args = args[:i] + args[i + 2:]
        if '--tag' in args:
            i = args.index('--tag'); tag = args[i + 1]; args = args[:i] + args[i + 2:]
        return analyse(args, pairs, tag)
    for arg in sys.argv[1:]:
        label, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        T, probes = collect(enc_spec, pred_ck, probe_file)
        S = score_all(T, probes)
        torch.save({'label': label, 'encoder': enc_spec, 'predictor': pred_ck,
                    'units': {k: v for k, v in T.items() if k not in ('exp', 'smp', 'copy', 'pers', 'real')},
                    'scores': S}, OUTD / f'locdyn2_{label}.pt')                                  # full precision (no artificial ties)
        print(json.dumps({label: 'saved', 'units': len(T['seed'])}), flush=True)


if __name__ == '__main__':
    main()
