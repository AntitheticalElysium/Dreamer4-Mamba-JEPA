"""Tier B (SPATIAL_ENCODER_PREREG.md): the world 'encoder + patch predictor' judged on decision-relevant predictions. Evaluation
only; exact simulator state and real labels score, the probes (fitted per encoder in spatial_eval.py on the FIT roots) read.

World: a frozen encoder spec (spatial_eval.PatchState) + a spatial_encoder.PatchPredictor (K 4 + router) with 2 frames of context.
  expected next state = sum_k router_k * candidate_k (graded: the linear probes then read expected values)
  sampled next state  = one candidate per patch drawn from its router (independent per patch)
Data: the TEST roots of spatial_eval.py's seeded split (diagnosis_futures_v1; probes never saw them) and DV16's 200 roots.
B1 immediate consequence (diagnosis one-step data: last two context frames, 17 actions x 4 keys): real shown health change per
   (root, action, key); predicted change = health_probe(expected next) - health_probe(current). Readings: Pearson r between the
   predicted change and the 4-key mean real change over (root, action); at real attacks (shown health drops >= 1): recall of a
   predicted drop >= 0.5 shown unit and the mean predicted change vs real; false-alarm rate where no key drops.
B2 branching (same data): map cells whose zombie-or-cow presence differs across the 4 keys (real rate = share of keys); the probe's
   mob-read rate in (a) a sampled next state, (b) the expected next state (the regression optimum; ghosts if above the real rate),
   and (c) the router-weighted read probability; mean |(c) - real rate| = calibration.
B3 escape choice (DV16 roots, deeppanel frames): predicted first-step health loss for noop + 4 moves; within-root AUC against the
   real P16 order (R53: one real future 0.710, D 0.711), and the expected-safe of the argmin.
Controls: the probes on the encoder's own states of the REAL next frames (health-change Pearson; mob read rate at cells where
   every key has a mob, and at branching cells) and a copy predictor's read rate at branching cells; they check the readers.
B4 recursion (test roots' 5 recorded futures x 16 steps): 4 sampled rollouts; zombie / cow census (probe reads on the 63 map cells)
   and zombie adjacency by depth vs the exact state.
Usage: tierb_eval.py <name>=<encoder spec>,<predictor checkpoint>,<probes file> ... -> artifacts/eda/spatial_encoder_v1/tierB_<name>.json
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

import torch                                                                   # noqa: E402

sys.path.insert(0, str(SE.ROOT / 'artifacts/experiments/20261015_whiteboard'))
import objective_audit as OA                                                   # noqa: E402
import dv16 as DV                                                              # noqa: E402

ADJ = [22, 30, 32, 40]
DEV = EV.DEV


def mob(probes, z, c):
    """zombie-or-cow read with the probes' rate-matched thresholds"""
    return (z > probes['zombie_threshold']) | (c > probes['cow_threshold'])


def read(probes, states):
    """states [...,81,D] -> health (shown units) [...], zombie / cow scores [...,63]"""
    sh = states.shape[:-2]
    flat = states.reshape(-1, 81, states.shape[-1]).float()
    h = EV.apply(probes['hud'], flat[:, 63:].reshape(len(flat), -1))[:, 0] * 9
    m = flat[:, :63].reshape(-1, flat.shape[-1])
    z = EV.apply(probes['zombie'], m)[:, 0].view(*sh, 63)
    c = EV.apply(probes['cow'], m)[:, 0].view(*sh, 63)
    return h.view(sh), z, c


def auc_within(score, P):
    num = den = 0.
    for i in range(len(P)):
        for a in range(5):
            for b in range(5):
                if P[i, a] > P[i, b]:
                    den += 1; num += float(score[i, a] > score[i, b]) + .5 * float(score[i, a] == score[i, b])
    return num / den


@torch.no_grad()
def step(pred, prev, cur, acts, g=None):
    """prev, cur [N,81,D], acts [N] -> expected [N,81,D], sampled [N,81,D], router probs [N,81,K], candidates [N,81,K,D]"""
    cand, router = pred(prev.to(DEV), cur.to(DEV), acts.to(DEV))
    cand, pi = cand.float(), router.float().softmax(-1)
    exp = (pi[..., None] * cand).sum(-2)
    k = torch.multinomial(pi.reshape(-1, pi.shape[-1]), 1, generator=g).view(pi.shape[:-1])
    samp = cand.gather(2, k[..., None, None].expand(-1, -1, 1, cand.shape[-1]))[:, :, 0]
    return exp, samp, pi, cand


@torch.no_grad()
def main():
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261070))[:400].tolist()
    test = pick[280:]
    roots, _ = DV.load_roots()
    full = {}
    for f in sorted(DV.DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            full[(int(r['seed']), int(r['step']))] = r
    P16 = torch.stack([r['P'] for r in roots])[..., 2]
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        enc = EV.PatchState(enc_spec)
        sd = torch.load(pred_ck, map_location=DEV, weights_only=False)['predictor']
        K = sd['router.weight'].shape[0]                                                   # the arm's own K (naive: 1)
        pred = SE.PatchPredictor(K).to(DEV).eval()
        pred.load_state_dict(sd)
        probes = torch.load(probe_file, weights_only=False)
        g = torch.Generator(device=DEV).manual_seed(20261071)
        res = {'encoder': enc_spec, 'predictor': pred_ck}
        # B1 / B2
        pc, rc, att_pred, att_real, fa = [], [], [], [], []
        units = {'b1_root': [], 'b1_pred': [], 'b1_real': [], 'att_root': [], 'att_pred': []}       # per-unit, for paired bootstraps
        b2 = {'cells': 0, 'real': 0., 'sampled': 0., 'expected': 0., 'weighted': 0., 'abs_err': 0., 'real_next_read': 0., 'copy_read': 0.}
        ctl = {'mob_cells_real_next': 0, 'mob_cells_read_real_next': 0, 'mob_cells_read_expected': 0, 'real_next_dh_pred': [], 'real_dh': []}
        for ri in test:
            r = rows[ri]
            st = enc(r['context'][-2:])                                                      # [2,81,D]
            h_cur, _, _ = read(probes, st[1:2]); h_cur = float(h_cur[0])
            t0, l0, sh0 = OA.layout(r['root_visible'][None])
            exp, samp, pi, cand = step(pred, st[0:1].expand(17, -1, -1), st[1:2].expand(17, -1, -1), torch.arange(17), g)
            h_exp, z_exp, c_exp = read(probes, exp.cpu())
            _, z_s, c_s = read(probes, samp.cpu())
            zc_k, cc_k = [], []
            for k in range(K):
                _, zk, ck = read(probes, cand[:, :, k].cpu()); zc_k.append(zk); cc_k.append(ck)
            zc_k, cc_k = torch.stack(zc_k, -1), torch.stack(cc_k, -1)                        # [17,63,K]
            t1, l1, sh1 = OA.layout(r['onestep_visible'])                                    # [4,17,...]
            # controls: the encoder's own states of the REAL next frames (probes must read them) and a copy predictor
            real_next = enc(r['onestep_frames'].flatten(0, 1)).view(4, 17, 81, -1)
            h_rn, z_rn, c_rn = read(probes, real_next.cpu())
            _, z_cp, c_cp = read(probes, st[1:2].cpu())
            dead = r['onestep_dead'].bool()
            real_dh = (sh1[..., 0] - sh0[0, 0]).float()                                      # shown health change [4,17]
            for a in range(17):
                ok = ~dead[:, a]
                if ok.sum() < 2:
                    continue
                d_pred = float(h_exp[a]) - h_cur
                pc.append(d_pred); rc.append(float(real_dh[ok, a].mean()))
                units['b1_root'].append(ri); units['b1_pred'].append(d_pred); units['b1_real'].append(float(real_dh[ok, a].mean()))
                okk = torch.nonzero(ok)[:, 0]
                ctl['real_next_dh_pred'].append(float(h_rn[okk, a].mean()) - h_cur); ctl['real_dh'].append(float(real_dh[ok, a].mean()))
                allmob = ((l1[okk, a] & 3) > 0).all(0)                                         # mobs present in every key
                if bool(allmob.any()):
                    u = torch.nonzero(allmob)[:, 0]
                    ctl['mob_cells_real_next'] += len(u)
                    ctl['mob_cells_read_real_next'] += float(mob(probes, z_rn[okk][:, a][:, u], c_rn[okk][:, a][:, u]).float().mean(0).sum())
                    ctl['mob_cells_read_expected'] += int(mob(probes, z_exp[a, u], c_exp[a, u]).sum())
                for k in torch.nonzero(ok)[:, 0].tolist():
                    if float(real_dh[k, a]) <= -1:
                        att_pred.append(d_pred); att_real.append(float(real_dh[k, a]))
                        units['att_root'].append(ri); units['att_pred'].append(d_pred)
                if bool((real_dh[ok, a] > -1).all()):
                    fa.append(d_pred <= -0.5)
                mobs = (l1[:, a] & 3) > 0                                                     # zombie or cow present [4,63]
                mk = mobs[ok]
                varying = (mk != mk[0]).any(0)
                if bool(varying.any()):
                    v = torch.nonzero(varying)[:, 0]
                    real_rate = mk[:, v].float().mean(0)
                    reads = lambda zz, cc: mob(probes, zz, cc)
                    b2['cells'] += len(v); b2['real'] += float(real_rate.sum())
                    b2['sampled'] += float(reads(z_s[a, v], c_s[a, v]).float().sum())
                    b2['expected'] += float(reads(z_exp[a, v], c_exp[a, v]).float().sum())
                    okk = torch.nonzero(ok)[:, 0]
                    b2['real_next_read'] += float(reads(z_rn[okk][:, a][:, v], c_rn[okk][:, a][:, v]).float().mean(0).sum())
                    b2['copy_read'] += float(reads(z_cp[0, v], c_cp[0, v]).float().sum())
                    w = (pi[a, v].cpu() * reads(zc_k[a, v], cc_k[a, v]).float()).sum(-1)
                    b2['weighted'] += float(w.sum()); b2['abs_err'] += float((w - real_rate).abs().sum())
        pc, rc = torch.tensor(pc), torch.tensor(rc)
        ap, ar = torch.tensor(att_pred), torch.tensor(att_real)
        res['B1'] = {'pairs': len(pc), 'pearson_pred_vs_real_mean_change': round(float(torch.corrcoef(torch.stack([pc, rc]))[0, 1]), 4),
                     'attacks': len(ap), 'attack_recall': round(float((ap <= -0.5).float().mean()), 4) if len(ap) else None,
                     'attack_mean_pred': round(float(ap.mean()), 3) if len(ap) else None, 'attack_mean_real': round(float(ar.mean()), 3) if len(ar) else None,
                     'false_alarm': round(float(torch.tensor(fa).float().mean()), 4)}
        n = max(b2['cells'], 1)
        m_ = max(ctl['mob_cells_real_next'], 1)
        res['controls'] = {'probe_on_real_next_pearson_dh': round(float(torch.corrcoef(torch.tensor([ctl['real_next_dh_pred'], ctl['real_dh']]))[0, 1]), 4),
                           'certain_mob_cells': ctl['mob_cells_real_next'],
                           'certain_mob_read_rate_real_next': round(ctl['mob_cells_read_real_next'] / m_, 4),
                           'certain_mob_read_rate_expected_pred': round(ctl['mob_cells_read_expected'] / m_, 4),
                           'branching_read_rate_real_next': round(b2['real_next_read'] / n, 4),
                           'branching_read_rate_copy': round(b2['copy_read'] / n, 4)}
        res['B2'] = {'branching_cells': b2['cells'], 'real_rate': round(b2['real'] / n, 4), 'sampled_read_rate': round(b2['sampled'] / n, 4),
                     'expected_read_rate': round(b2['expected'] / n, 4), 'router_weighted_rate': round(b2['weighted'] / n, 4),
                     'calibration_mae': round(b2['abs_err'] / n, 4)}
        print(json.dumps({name: {'B1': res['B1'], 'B2': res['B2'], 'controls': res['controls']}}), flush=True)
        # B3
        risk = torch.zeros(len(roots), 5)
        for i, r in enumerate(roots):
            st = enc(r['frames'][-2:])
            h_cur, _, _ = read(probes, st[1:2])
            exp, _, _, _ = step(pred, st[0:1].expand(5, -1, -1), st[1:2].expand(5, -1, -1), torch.arange(5), g)
            h_exp, _, _ = read(probes, exp.cpu())
            risk[i] = float(h_cur[0]) - h_exp
        P5 = P16[:, :5]
        es = []
        for i in range(len(roots)):
            m = risk[i] == risk[i].min(); es.append(1 - float(P5[i][m].mean()))
        res['B3'] = {'auc5_first_step_risk': round(auc_within(risk, P5), 4), 'expected_safe_5': round(sum(es) / len(es), 4)}
        units['b3_risk'] = risk; units['b3_es'] = torch.tensor(es)
        print(json.dumps({name: {'B3': res['B3']}}), flush=True)
        # B4
        depths = (1, 2, 4, 8, 16)
        acc = {k: torch.zeros(len(depths)) for k in ('z_imag', 'z_real', 'c_imag', 'c_real', 'adj_imag', 'adj_real')}
        cnt = torch.zeros(len(depths))
        for ri in test:
            r = rows[ri]
            st = enc(r['context'][-2:])
            for f in range(5):
                alive = int((~r['future_dead'][f].bool()).long().cumprod(0).sum())
                vis = r['future_visible'][f]
                acts = r['future_actions']
                prev, cur = st[0:1].expand(4, -1, -1), st[1:2].expand(4, -1, -1)
                imag = []
                for d in range(16):
                    _, samp, _, _ = step(pred, prev, cur, acts[d].repeat(4), g)
                    prev, cur = cur, samp
                    imag.append(samp.cpu())
                imag = torch.stack(imag, 1)                                                   # [4,16,81,D]
                _, zi, ci = read(probes, imag)
                _, lr_, _ = OA.layout(vis)
                for di, d in enumerate(depths):
                    if d > alive:
                        continue
                    zr = (lr_[d - 1] & 1) > 0; cr = (lr_[d - 1] & 2) > 0
                    zt, ct = probes['zombie_threshold'], probes['cow_threshold']
                    acc['z_imag'][di] += float((zi[:, d - 1] > zt).float().sum(-1).mean()); acc['z_real'][di] += float(zr.sum())
                    acc['c_imag'][di] += float((ci[:, d - 1] > ct).float().sum(-1).mean()); acc['c_real'][di] += float(cr.sum())
                    acc['adj_imag'][di] += float((zi[:, d - 1][:, ADJ] > zt).any(-1).float().mean()); acc['adj_real'][di] += float(zr[ADJ].any())
                    cnt[di] += 1
        res['B4'] = {k: [round(float(v), 3) for v in (acc[k] / cnt.clamp_min(1))] for k in acc}
        res['B4']['depths'] = list(depths)
        print(json.dumps({name: {'B4': res['B4']}}), flush=True)
        (EV.OUT / f'tierB_{name}.json').write_text(json.dumps(res, indent=1))
        torch.save(units, EV.OUT / f'tierB_units_{name}.pt')
        del enc, pred; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
