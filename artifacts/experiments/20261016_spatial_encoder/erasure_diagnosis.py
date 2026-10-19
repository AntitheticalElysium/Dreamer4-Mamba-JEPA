"""Why does the full arm predict background well yet erase mobs at once? (R63; user, 10-17; evaluation only, existing
checkpoints; exact simulator labels evaluate, never train.)

Per checkpoint (encoder + projector + its own co-trained predictor; Raw: frozen Raw@10000 + a Raw-predictor checkpoint):

R  REPRESENTATION, both stages (backbone_ln, deployed), Tier A frames of split_v2.json:
   readers exactly as spatial_eval_v2.a1 (fit frames, seeded 400k-cell subsample, lambda 1e-2, fit-only rate-matched thresholds)
   -> test zombie / cow AP and tile accuracy;
   mob energy: share of map-token variance explained by mob presence WITHIN tile classes (group means tile x mob layer vs tile);
   one-step change energy |z_{t+1} - z_t|^2 per map cell on consecutive Tier A test frames: share and mean per cell at cells whose
   mob layer changed, whose tile changed (mostly camera shifts) and neither (nuisance drift).
P  PREDICTOR, deployed stage, depth 1 in the B4 setting (root's 2-frame context, the recorded first action) for every alive
   (root, future) of the fit and test roots: expected state, 4 router samples, the K candidates (best-of-K = max mob score over
   candidates, an oracle), copy (= the root's own state) and real (= the encoded real next frame):
   P1 transfer reads with the real-state readers vs the exact next mob cells: AP, AUC, per-frame count, precision / recall at
      the threshold, exactly and within one cell; by camera motion (exact tile shift) and action class.
   P2 refit reads: zombie / cow ridge (lambda 1e-2) fitted on the FIT units' expected / sample / copy / real states against the
      exact next cells -> test AP. Expected below copy = the predictor destroyed mob information its input carried.
   P3 background: the real-state tile reader on expected / copy / real (all map cells; cells whose exact tile changed); a tile
      reader refitted on expected states; squared token error relative to copy and to the variance of real next tokens, at
      mob and mob-free cells.
D  IN-DISTRIBUTION (held-out DEV split of the training corpus; never FINAL): 2,048 seeded 4-frame windows, the training
   alignment (z1, z2, a2 -> z3). No simulator labels exist there: labels = Raw@10000's Tier B readers on the real f3 frame
   (pseudo-labels, AP 0.95-0.99 on diagnosis frames). P1 transfer AP and P2 refit AP (episode-disjoint halves).
Usage: erasure_diagnosis.py <label>=<encoder spec>,<predictor ckpt> ... -> artifacts/eda/spatial_encoder_v1/erasure/<label>.json
CORRECTION (10-17, Codex audit ../20261017_frozen_state_audit/ASSESSMENT.md): AP is now metrics_tied.ap_grouped (tie-grouped). The
original spatial_eval_v2.average_precision is order-dependent inside ties, and expected / copy scores repeat across a root's
futures with different labels. Outputs go to erasure_v2/; the original erasure/ outputs (R63) are kept. Per-unit test refit
scores are saved (erasure_v2/<label>_units.pt) for episode-bootstrap intervals.
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
import spatial_eval_v2 as EV2                                                  # noqa: E402
import tierb_eval as TB                                                        # noqa: E402

import torch                                                                   # noqa: E402
from torch.nn import functional as F                                           # noqa: E402

import objective_audit as OA                                                   # noqa: E402
from metrics_tied import ap_grouped                                            # noqa: E402

OUTD = EV.OUT / 'erasure_v2'
AP, AUC = ap_grouped, EV2.auc_tie
rnd = lambda x: None if x is None else round(float(x), 4)
score = lambda W, X: torch.cat([EV.apply(W, X[i:i + 200000].float()) for i in range(0, len(X), 200000)])
dil = lambda m: F.max_pool2d(m.float().reshape(-1, 1, 7, 9), 3, 1, 1).reshape(m.shape) > 0


def readers(Sf, Tf, Lf):
    """= spatial_eval_v2.a1's tile / zombie / cow readers and fit-only rate thresholds"""
    D = Sf.shape[-1]
    sub = torch.randperm(len(Sf) * 63, generator=torch.Generator().manual_seed(1))[:400000]
    Xf = Sf[:, :63].reshape(-1, D)[sub].float()
    R = {'tile': EV.ridge(Xf, F.one_hot(Tf.reshape(-1)[sub], 17).float())}
    for bit, nm in ((1, 'zombie'), (2, 'cow')):
        lf = ((Lf.reshape(-1)[sub] & bit) > 0).float()
        R[nm] = EV.ridge(Xf, lf[:, None])
        srt = EV.apply(R[nm], Xf)[:, 0].sort(descending=True).values
        R[nm + '_th'] = float(srt[min(max(int(round(float(lf.sum()))) - 1, 0), len(srt) - 1)])
    return R


def representation(ft, Tt, Lt, St, R):
    D = St.shape[-1]
    X = St[:, :63].reshape(-1, D).float()
    T, L = torch.cat(Tt).reshape(-1), torch.cat(Lt).reshape(-1)
    out = {'tile_acc': rnd((score(R['tile'], X).argmax(-1) == T).float().mean())}
    for bit, nm in ((1, 'zombie'), (2, 'cow')):
        s = score(R[nm], X)[:, 0]; out[f'{nm}_ap'] = rnd(AP(s, (L & bit) > 0))
    Xd = X.double(); tss = float((Xd - Xd.mean(0)).pow(2).sum())

    def rss(g):
        n = torch.zeros(int(g.max()) + 1, dtype=torch.float64).index_add_(0, g, torch.ones(len(g), dtype=torch.float64))
        s = torch.zeros(len(n), D, dtype=torch.float64).index_add_(0, g, Xd)
        return float(Xd.pow(2).sum() - (s.pow(2).sum(-1) / n.clamp_min(1)).sum())
    r_tile, r_full = rss(T), rss(T * 16 + (L & 15))
    out['var_share_tile'] = rnd(1 - r_tile / tss); out['var_share_mob_within_tile'] = rnd((r_tile - r_full) / tss)
    E = {k: [0., 0] for k in ('mob_change', 'tile_change', 'neither')}
    for x, t, l in zip(ft, Tt, Lt):
        if len(x) < 2:
            continue
        e = (x[1:, :63].float() - x[:-1, :63].float()).pow(2).sum(-1)
        mc = l[1:] != l[:-1]; tc = (t[1:] != t[:-1]) & ~mc; ne = ~mc & ~tc
        for k, m in (('mob_change', mc), ('tile_change', tc), ('neither', ne)):
            E[k][0] += float(e[m].sum()); E[k][1] += int(m.sum())
    tot = sum(v[0] for v in E.values())
    out['change_energy'] = {k: {'energy_share': rnd(v[0] / tot), 'cell_share': rnd(v[1] / sum(w[1] for w in E.values())),
                                'mean_per_cell_rel_neither': rnd((v[0] / max(v[1], 1)) / (E['neither'][0] / max(E['neither'][1], 1)))}
                            for k, v in E.items()}
    return out


def loc(s, th, ex):
    """s [n,63] scores, ex [n,63] bool exact cells"""
    p = s > th
    return {'ap': rnd(AP(s.reshape(-1), ex.reshape(-1))), 'auc': rnd(AUC(s.reshape(-1), ex.reshape(-1))),
            'per_frame': rnd(p.float().sum(-1).mean()),
            'precision': rnd((p & ex).sum() / p.sum().clamp_min(1)), 'recall': rnd((p & ex).sum() / ex.sum().clamp_min(1)),
            'precision_1cell': rnd((p & dil(ex)).sum() / p.sum().clamp_min(1)), 'recall_1cell': rnd((dil(p) & ex).sum() / ex.sum().clamp_min(1))}


@torch.no_grad()
def predictor_units(rows, idx, enc, pred, g):
    """depth-1 units (root, alive future): map tokens of expected / sample 0 / copy / real; candidate tokens; labels"""
    U = {k: [] for k in ('exp', 'smp', 'cand', 'copy', 'real', 'tile0', 'tile1', 'lay1', 'act', 'moved', 'seed')}
    for ri in idx:
        r = rows[ri]
        alive = [f for f in range(5) if not bool(r['future_dead'][f][0])]
        if not alive:
            continue
        ctx = enc(r['context'][-2:])
        a = r['future_actions'][0:1].repeat(4)
        exp, smp, _, cand = TB.step(pred, ctx[0:1].expand(4, -1, -1), ctx[1:2].expand(4, -1, -1), a, g)
        real = enc(torch.stack([r['future_frames'][f][0] for f in alive]))
        t0, _, _ = OA.layout(r['root_visible'][None])
        t1, l1, _ = OA.layout(torch.stack([r['future_visible'][f][0] for f in alive]))
        n = len(alive)
        U['exp'].append(exp[0:1, :63].cpu().expand(n, -1, -1)); U['smp'].append(smp[:, :63].cpu()[None].expand(n, -1, -1, -1))
        U['cand'].append(cand[0:1, :63].cpu().expand(n, -1, -1, -1)); U['copy'].append(ctx[1:2, :63].cpu().expand(n, -1, -1))
        U['real'].append(real[:, :63].cpu()); U['tile0'].append(t0.expand(n, -1)); U['tile1'].append(t1); U['lay1'].append(l1)
        U['act'].append(torch.full((n,), int(r['future_actions'][0]))); U['seed'].append(torch.full((n,), int(r['seed'])))
        U['moved'].append(torch.tensor([OA.shift_of(t0[0], t1[i]) != (0, 0) for i in range(n)]))
    return {k: torch.cat(v) for k, v in U.items()}


def predictor_part(Uf, Ut, R):
    D = Ut['exp'].shape[-1]
    out = {'units_test': len(Ut['exp']), 'units_fit': len(Uf['exp'])}
    flat = lambda x: x.reshape(-1, D)
    for bit, nm in ((1, 'zombie'), (2, 'cow')):
        ex = (Ut['lay1'] & bit) > 0
        th, W = R[nm + '_th'], R[nm]
        sc = {k: score(W, flat(Ut[k]))[:, 0].view(len(ex), 63) for k in ('exp', 'copy', 'real')}
        sc_s = score(W, flat(Ut['smp']))[:, 0].view(len(ex), 4, 63)
        sc_c = score(W, flat(Ut['cand']))[:, 0].view(len(ex), 63, -1)
        o = {k: loc(v, th, ex) for k, v in sc.items()}
        o['sample'] = loc(sc_s.reshape(-1, 63), th, ex[:, None].expand(-1, 4, -1).reshape(-1, 63))
        o['best_of_K'] = loc(sc_c.max(-1).values, th, ex)
        strata = {'moved': Ut['moved'], 'static': ~Ut['moved'], 'noop': Ut['act'] == 0, 'move': (Ut['act'] >= 1) & (Ut['act'] <= 4),
                  'other': Ut['act'] >= 5}
        o['by_stratum'] = {s: {'units': int(m.sum()), 'positives': int(ex[m].sum()),
                               **{k: {kk: loc(v[m], th, ex[m])[kk] for kk in ('ap', 'recall', 'recall_1cell')} for k, v in sc.items()}}
                           for s, m in strata.items() if int(ex[m].sum()) > 0}
        exf = ((Uf['lay1'] & bit) > 0).reshape(-1).float()[:, None]
        o['refit_ap'] = {}
        o['_scores'] = {'label': ex}
        for k, src in (('exp', 'exp'), ('sample0', 'smp'), ('copy', 'copy'), ('real', 'real')):
            xf = Uf[src][:, 0] if src == 'smp' else Uf[src]
            xt = Ut[src][:, 0] if src == 'smp' else Ut[src]
            Wr = EV.ridge(flat(xf).float(), exf)
            st_ = score(Wr, flat(xt))[:, 0]
            o['refit_ap'][k] = rnd(AP(st_, ex.reshape(-1)))
            o['_scores'][k] = st_.view(ex.shape)
        out[nm] = o
    # background
    t1, t0 = Ut['tile1'], Ut['tile0']
    chg = t1 != t0
    bg = {}
    for k in ('exp', 'copy', 'real'):
        p = score(R['tile'], flat(Ut[k])).argmax(-1).view(t1.shape)
        bg[f'tile_acc_{k}'] = rnd((p == t1).float().mean()); bg[f'tile_acc_changed_{k}'] = rnd((p == t1)[chg].float().mean())
    Wt = EV.ridge(flat(Uf['exp']).float(), F.one_hot(Uf['tile1'].reshape(-1), 17).float())
    p = score(Wt, flat(Ut['exp'])).argmax(-1).view(t1.shape)
    bg['tile_acc_exp_refit'] = rnd((p == t1).float().mean()); bg['tile_acc_changed_exp_refit'] = rnd((p == t1)[chg].float().mean())
    bg['changed_cell_share'] = rnd(chg.float().mean())
    mob = (Ut['lay1'] & 3) > 0
    real = Ut['real'].float(); var = (real - real.reshape(-1, D).mean(0)).pow(2).sum(-1)
    for k in ('exp', 'copy'):
        e = (Ut[k].float() - real).pow(2).sum(-1)
        for nm, m in (('mob', mob), ('mob_free', ~mob)):
            bg[f'sq_err_{k}_{nm}_rel_var'] = rnd(e[m].sum() / var[m].sum())
    es = (Ut['smp'].float() - real[:, None]).pow(2).sum(-1).mean(1)
    for nm, m in (('mob', mob), ('mob_free', ~mob)):
        bg[f'sq_err_sample_{nm}_rel_var'] = rnd(es[m].sum() / var[m].sum())
        bg[f'sq_err_exp_{nm}_rel_copy'] = rnd((Ut['exp'].float() - real).pow(2).sum(-1)[m].sum() / (Ut['copy'].float() - real).pow(2).sum(-1)[m].sum())
    out['background'] = bg
    return out


def dev_windows(n=2048, seed=20261101):
    from d4mj.config import load_recipe
    from d4mj.data import load_joint_corpus
    config = load_recipe(EV.ROOT / 'd4mj/recipes/lewm_mamba_raw.json')
    eps, _ = load_joint_corpus([EV.ROOT / 'artifacts/craftax_expert_store_v1', EV.ROOT / 'artifacts/craftax_support_v2'], config)
    dev = [e for e in eps if e.split == 'dev' and e.uniform_eligible and len(e) >= 4]
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor([len(e) - 2 for e in dev], dtype=torch.float64)
    pick = torch.multinomial(w, n, replacement=True, generator=g).tolist()
    fr, ac, ep = [], [], []
    for i in pick:
        s = int(torch.randint(len(dev[i]) - 2, (), generator=g))
        fr.append(dev[i].observations[s:s + 4].clone()); ac.append(dev[i].actions_taken[s:s + 3].clone()); ep.append(i)
    del eps, dev
    return torch.stack(fr), torch.stack(ac), torch.tensor(ep)


@torch.no_grad()
def dev_part(frames, acts, ep, enc, pred, R, lab, g):
    U = {k: [] for k in ('exp', 'copy', 'real')}
    for i in range(0, len(frames), 128):
        z = enc(frames[i:i + 128].flatten(0, 1))
        z = z.view(-1, 4, 81, z.shape[-1])
        exp, _, _, _ = TB.step(pred, z[:, 1], z[:, 2], acts[i:i + 128, 2], g)
        U['exp'].append(exp[:, :63].cpu()); U['copy'].append(z[:, 2, :63].cpu()); U['real'].append(z[:, 3, :63].cpu())
    U = {k: torch.cat(v) for k, v in U.items()}
    D = U['exp'].shape[-1]
    fit = (ep % 2) == 0                                                           # episode-disjoint halves
    out = {'windows': len(frames), 'labels': 'Raw@10000 Tier B readers on the real f3 frame (pseudo-labels)'}
    for nm in ('zombie', 'cow'):
        ex = lab[nm]
        o = {}
        for k in U:
            s = score(R[nm], U[k].reshape(-1, D))[:, 0].view(-1, 63)
            o[k] = {kk: loc(s[~fit], R[nm + '_th'], ex[~fit])[kk] for kk in ('ap', 'per_frame', 'precision', 'recall', 'recall_1cell')}
            Wr = EV.ridge(U[k][fit].reshape(-1, D).float(), ex[fit].reshape(-1).float()[:, None])
            o[k]['refit_ap'] = rnd(AP(score(Wr, U[k][~fit].reshape(-1, D))[:, 0], ex[~fit].reshape(-1)))
        o['positives_test'] = int(ex[~fit].sum())
        out[nm] = o
    return out


@torch.no_grad()
def main():
    OUTD.mkdir(exist_ok=True)
    rows, _, split = EV2.episode_split()
    assert split == json.loads(EV2.SPLIT.read_text())
    frames, acts, ep = dev_windows()
    raw_probes = torch.load(EV.OUT / 'probes2_raw10000.pt', weights_only=False)
    rawenc = EV.PatchState('raw@10000')
    zr = torch.cat([EV.apply(raw_probes['zombie'], rawenc(frames[i:i + 256, 3])[:, :63].reshape(-1, 192).cpu()) for i in range(0, len(frames), 256)])
    cr = torch.cat([EV.apply(raw_probes['cow'], rawenc(frames[i:i + 256, 3])[:, :63].reshape(-1, 192).cpu()) for i in range(0, len(frames), 256)])
    lab = {'zombie': (zr[:, 0] > raw_probes['zombie_threshold']).view(-1, 63), 'cow': (cr[:, 0] > raw_probes['cow_threshold']).view(-1, 63)}
    del rawenc
    cache = {}
    for arg in sys.argv[1:]:
        label, spec = arg.split('=', 1)
        enc_spec, pred_ck = spec.split(',')
        if enc_spec not in cache:                                                 # representation + readers per encoder
            st = EV2.Stages(enc_spec)
            ff, Tf, Lf, _, _, _ = EV2.gather(rows, split['fit'], st)
            ft, Tt, Lt, _, _, _ = EV2.gather(rows, split['test'], st)
            Tfc, Lfc = torch.cat(Tf), torch.cat(Lf)
            res_r, Rs = {}, {}
            for stage in ff:
                Sf, St = torch.cat(ff[stage]), torch.cat(ft[stage])
                Rs[stage] = readers(Sf, Tfc, Lfc)
                res_r[stage] = representation(ft[stage], Tt, Lt, St, Rs[stage])
                del Sf, St
            cache = {enc_spec: (res_r, Rs['deployed'])}
            del ff, ft, st
        res_r, R = cache[enc_spec]
        enc = EV.PatchState(enc_spec)
        sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
        pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
        g = torch.Generator(device=EV.DEV).manual_seed(20261071)
        Uf = predictor_units(rows, split['fit'], enc, pred, g)
        Ut = predictor_units(rows, split['test'], enc, pred, g)
        res = {'label': label, 'encoder': enc_spec, 'predictor': pred_ck, 'device': EV.DEV, 'ap': 'metrics_tied.ap_grouped', 'representation': res_r,
               'predictor_d1': predictor_part(Uf, Ut, R),
               'dev_in_distribution': dev_part(frames, acts, ep, enc, pred, R, lab, g)}
        torch.save({'seed': Ut['seed'], **{m: res['predictor_d1'][m].pop('_scores') for m in ('zombie', 'cow')}}, OUTD / f'{label}_units.pt')
        (OUTD / f'{label}.json').write_text(json.dumps(res, indent=1))
        pz = res['predictor_d1']['zombie']; pc = res['predictor_d1']['cow']
        print(json.dumps({label: {'dep_zAP': res_r['deployed']['zombie_ap'], 'z_exp_ap': pz['exp']['ap'], 'z_refit': pz['refit_ap'],
                                  'c_exp_ap': pc['exp']['ap'], 'c_refit': pc['refit_ap'], 'bg': res['predictor_d1']['background']['tile_acc_exp']}}), flush=True)
        del enc, pred, Uf, Ut


if __name__ == '__main__':
    main()
