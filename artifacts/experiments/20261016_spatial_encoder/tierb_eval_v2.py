"""Tier B, CORRECTED (SPATIAL_ENCODER_PREREG.md Amendment 4). The original tierb_eval.py is kept unchanged.

World: frozen encoder (deployed state) + PatchPredictor (K from its weights), 2 frames of context; read by the DEPLOYED-stage
probes of spatial_eval_v2.py (fit on the fit episodes, thresholds rate-matched on fit only).
Test roots: split_v2.json's episode-disjoint test roots (diagnosis store); DV16's 200 roots for B3. Every unit carries its
episode seed so stage2_analysis_v2.py can resample episodes.
B1 per (root, action): predicted vs 4-key-mean real shown health change; attack events (a key's shown health drops >= 1) with the
   predicted change.
B2 per (root, action) at branching map cells (zombie-or-cow presence differs across the 4 keys): real rate, sampled / expected /
   router-weighted read rates, |weighted - real|; sampled reads at the x0.5 and x2 rate thresholds; controls: the probes on the
   encoder's own states of the REAL next frames at the same cells, and at cells with a mob in every key.
   Reader quality on real next frames (all map cells): zombie / cow average precision and ROC AUC.
B3 per DV16 root: predicted first-step health loss for noop + 4 moves (the analysis takes per-root within-root AUC).
B4 per (root, future): 16-step rollouts (4 router samples), zombie / cow census and zombie adjacency at depths 1, 2, 4, 8, 16
   for imagined states, for the probes on the REAL future frames (decoded-real), and for the exact state.
Usage: tierb_eval_v2.py <name>=<encoder spec>,<predictor checkpoint>,<probes2 file> ...
   -> artifacts/eda/spatial_encoder_v1/tierB2_<name>.json + tierB2_units_<name>.pt
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
import tierb_eval as TB                                                        # noqa: E402  (read, step, mob)

import torch                                                                   # noqa: E402

import objective_audit as OA                                                   # noqa: E402
import dv16 as DV                                                              # noqa: E402

DEPTHS = (1, 2, 4, 8, 16)


@torch.no_grad()
def main():
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    split = json.loads(EV2.SPLIT.read_text())
    test = split['test']
    roots, _ = DV.load_roots()
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        enc = EV.PatchState(enc_spec)
        sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
        K = sd['router.weight'].shape[0]
        pred = SE.PatchPredictor(K).to(EV.DEV).eval(); pred.load_state_dict(sd)
        probes = torch.load(probe_file, weights_only=False)
        g = torch.Generator(device=EV.DEV).manual_seed(20261071)
        U = {'b1': [], 'att': [], 'b2': [], 'b3': [], 'b4': [], 'reader': {'z': [], 'zl': [], 'c': [], 'cl': []}}
        for ri in test:
            r = rows[ri]; seed = int(r['seed'])
            st = enc(r['context'][-2:])
            h_cur = float(TB.read(probes, st[1:2])[0][0])
            t0, l0, sh0 = OA.layout(r['root_visible'][None])
            exp, samp, pi, cand = TB.step(pred, st[0:1].expand(17, -1, -1), st[1:2].expand(17, -1, -1), torch.arange(17), g)
            h_exp, z_exp, c_exp = TB.read(probes, exp.cpu())
            _, z_s, c_s = TB.read(probes, samp.cpu())
            zc_k = torch.stack([TB.read(probes, cand[:, :, k].cpu())[1] for k in range(K)], -1)
            cc_k = torch.stack([TB.read(probes, cand[:, :, k].cpu())[2] for k in range(K)], -1)
            t1, l1, sh1 = OA.layout(r['onestep_visible'])
            dead = r['onestep_dead'].bool()
            real_next = enc(r['onestep_frames'].flatten(0, 1)).view(4, 17, 81, -1)
            h_rn, z_rn, c_rn = TB.read(probes, real_next.cpu())
            U['reader']['z'].append(z_rn.reshape(-1)); U['reader']['zl'].append(((l1 & 1) > 0).reshape(-1))
            U['reader']['c'].append(c_rn.reshape(-1)); U['reader']['cl'].append(((l1 & 2) > 0).reshape(-1))
            real_dh = (sh1[..., 0] - sh0[0, 0]).float()
            for a in range(17):
                ok = ~dead[:, a]
                if ok.sum() < 2:
                    continue
                okk = torch.nonzero(ok)[:, 0]
                d_pred = float(h_exp[a]) - h_cur
                U['b1'].append((seed, ri, a, d_pred, float(real_dh[ok, a].mean()), float(h_rn[okk, a].mean()) - h_cur))
                for k in okk.tolist():
                    if float(real_dh[k, a]) <= -1:
                        U['att'].append((seed, ri, a, d_pred, float(real_dh[k, a])))
                mobs = (l1[:, a] & 3) > 0
                mk = mobs[okk]
                varying = (mk != mk[0]).any(0)
                certain = mk.all(0)
                row = {'seed': seed, 'root': ri, 'action': a, 'n': 0}
                if bool(varying.any()):
                    v = torch.nonzero(varying)[:, 0]
                    rr = mk[:, v].float().mean(0)
                    rd = lambda zz, cc, suf='': (zz > probes['zombie_threshold' + suf]) | (cc > probes['cow_threshold' + suf])
                    w = (pi[a, v].cpu() * rd(zc_k[a, v], cc_k[a, v]).float()).sum(-1)
                    row.update(n=len(v), real=float(rr.sum()), sampled=float(rd(z_s[a, v], c_s[a, v]).float().sum()),
                               expected=float(rd(z_exp[a, v], c_exp[a, v]).float().sum()), weighted=float(w.sum()),
                               abs_err=float((w - rr).abs().sum()),
                               sampled_x05=float(rd(z_s[a, v], c_s[a, v], '_x0.5').float().sum()),
                               sampled_x2=float(rd(z_s[a, v], c_s[a, v], '_x2.0').float().sum()),
                               real_next=float(rd(z_rn[okk][:, a][:, v], c_rn[okk][:, a][:, v]).float().mean(0).sum()))
                if bool(certain.any()):
                    u = torch.nonzero(certain)[:, 0]
                    row.update(n_certain=len(u), certain_real_next=float(TB.mob(probes, z_rn[okk][:, a][:, u], c_rn[okk][:, a][:, u]).float().mean(0).sum()),
                               certain_expected=int(TB.mob(probes, z_exp[a, u], c_exp[a, u]).sum()))
                U['b2'].append(row)
            # B4: rollouts with recorded actions; decoded-real census
            for f in range(5):
                alive = int((~r['future_dead'][f].bool()).long().cumprod(0).sum())
                if alive < 1:
                    continue
                vis = r['future_visible'][f]
                acts = r['future_actions']
                prev, cur = st[0:1].expand(4, -1, -1), st[1:2].expand(4, -1, -1)
                imag = []
                for d in range(16):
                    _, smp, _, _ = TB.step(pred, prev, cur, acts[d].repeat(4), g)
                    prev, cur = cur, smp
                    imag.append(smp.cpu())
                imag = torch.stack(imag, 1)
                _, zi, ci = TB.read(probes, imag)
                real_states = enc(r['future_frames'][f][:alive])
                _, zr_dec, cr_dec = TB.read(probes, real_states.cpu())
                _, lr_, _ = OA.layout(vis)
                zt, ct = probes['zombie_threshold'], probes['cow_threshold']
                for d in DEPTHS:
                    if d > alive:
                        continue
                    zr = (lr_[d - 1] & 1) > 0; cr = (lr_[d - 1] & 2) > 0
                    U['b4'].append((seed, ri, f, d, float((zi[:, d - 1] > zt).float().sum(-1).mean()), float((zr_dec[d - 1] > zt).sum()), float(zr.sum()),
                                    float((ci[:, d - 1] > ct).float().sum(-1).mean()), float((cr_dec[d - 1] > ct).sum()), float(cr.sum()),
                                    float((zi[:, d - 1][:, TB.ADJ] > zt).any(-1).float().mean()), float((zr_dec[d - 1][TB.ADJ] > zt).any()), float(zr[TB.ADJ].any())))
        # B3 on DV16 roots
        for i, r in enumerate(roots):
            st = enc(r['frames'][-2:])
            h_cur = TB.read(probes, st[1:2])[0]
            exp, _, _, _ = TB.step(pred, st[0:1].expand(5, -1, -1), st[1:2].expand(5, -1, -1), torch.arange(5), g)
            U['b3'].append((int(r['seed']), i, (float(h_cur[0]) - TB.read(probes, exp.cpu())[0]).tolist()))
        rz, rzl = torch.cat(U['reader']['z']), torch.cat(U['reader']['zl']); rc, rcl = torch.cat(U['reader']['c']), torch.cat(U['reader']['cl'])
        U['reader_quality'] = {'zombie_ap': EV2.average_precision(rz, rzl), 'zombie_auc': EV2.auc_tie(rz, rzl),
                               'cow_ap': EV2.average_precision(rc, rcl), 'cow_auc': EV2.auc_tie(rc, rcl)}
        del U['reader']
        torch.save(U, EV.OUT / f'tierB2_units_{name}.pt')
        # aggregate summary (the analysis computes the statistics)
        b2 = {k: sum(r_.get(k, 0) for r_ in U['b2']) for k in ('n', 'real', 'sampled', 'expected', 'weighted', 'abs_err', 'sampled_x05', 'sampled_x2', 'real_next', 'n_certain', 'certain_real_next', 'certain_expected')}
        n = max(b2['n'], 1); nc = max(b2['n_certain'], 1)
        att = torch.tensor([a[3] for a in U['att']])
        summ = {'encoder': enc_spec, 'predictor': pred_ck, 'probes': probe_file, 'test_roots': len(test), 'test_episodes': len({int(rows[i]['seed']) for i in test}),
                'B1': {'pairs': len(U['b1']), 'attacks': len(att), 'attack_recall': round(float((att <= -0.5).float().mean()), 4) if len(att) else None},
                'B2': {'branching_cells': b2['n'], 'real_rate': round(b2['real'] / n, 4), 'sampled': round(b2['sampled'] / n, 4), 'expected': round(b2['expected'] / n, 4),
                       'weighted': round(b2['weighted'] / n, 4), 'calibration_mae': round(b2['abs_err'] / n, 4),
                       'sampled_x0.5': round(b2['sampled_x05'] / n, 4), 'sampled_x2': round(b2['sampled_x2'] / n, 4), 'real_next_read': round(b2['real_next'] / n, 4),
                       'certain_cells': b2['n_certain'], 'certain_read_real_next': round(b2['certain_real_next'] / nc, 4), 'certain_read_expected': round(b2['certain_expected'] / nc, 4)},
                'reader_quality_real_next': {k: (round(v, 4) if v is not None else None) for k, v in U['reader_quality'].items()}}
        print(json.dumps({name: summ}), flush=True)
        (EV.OUT / f'tierB2_{name}.json').write_text(json.dumps(summ, indent=1))
        del enc, pred


if __name__ == '__main__':
    main()
