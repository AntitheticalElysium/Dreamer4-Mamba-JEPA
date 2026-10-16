"""Stage 2 verdict, CORRECTED (SPATIAL_ENCODER_PREREG.md Amendments 4 / 4b), from tierb_eval_v2.py's per-unit files.

Paired Spatial vs Raw-predictor per seed. Every interval is a 4,000-resample percentile bootstrap over EPISODE SEEDS (a drawn
episode contributes all its units; diagnosis seeds for B1 / B2 / B4, DV16 episode seeds for B3).
Primary (unchanged): B1 attack recall (predicted change <= -0.5 shown units at real attacks) and B3 = the mean over roots of the
per-root within-root AUC of first-step risk against the real P16 order (noop + 4 moves; roots with a real spread).
Guards (Amendment 4): Spatial must not be worse beyond noise on B2 calibration MAE or on B4 zombie-census absolute error at depth
16 against the decoded-real census ("worse" = the paired interval lies entirely on the worse side).
Verdict: "better predictive world state" iff both primaries improve (interval lower bounds > 0) AND neither guard fails, at both
seeds; "not supported" iff, at both seeds, neither primary improves; otherwise "mixed".
Usage: stage2_analysis_v2.py [--prefix-raw rawpred_s --prefix-sp spatial_s --seeds 1,2] -> stage2_verdict_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402  (paths)
import dv16 as DV                                                               # noqa: E402

B = 4000


def groups(seeds):
    by = {}
    for i, s in enumerate(seeds):
        by.setdefault(s, []).append(i)
    return by


def boot(stat, by, g):
    ids = sorted(by)
    if not ids:
        return [None, None]                                                        # no units: no interval (never counts as a pass)
    vals = []
    for _ in range(B):
        draw = [ids[i] for i in torch.randint(len(ids), (len(ids),), generator=g).tolist()]
        idx = [j for s in draw for j in by[s]]
        vals.append(stat(idx))
    v = torch.tensor(vals, dtype=torch.float64)
    v = v[~torch.isnan(v)]
    if len(v) == 0:
        return [None, None]
    return [round(float(v.quantile(.025)), 4), round(float(v.quantile(.975)), 4)]


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def auc_root(risk, P):
    num = den = 0.
    for a in range(5):
        for b in range(5):
            if P[a] > P[b]:
                den += 1; num += float(risk[a] > risk[b]) + .5 * float(risk[a] == risk[b])
    return num / den if den else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix-raw', default='rawpred_s'); ap.add_argument('--prefix-sp', default='spatial_s')
    ap.add_argument('--seeds', default='1,2'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',')]
    roots, _ = DV.load_roots()
    P5 = torch.stack([r['P'] for r in roots])[..., 2][:, :5]
    g = torch.Generator().manual_seed(20261090)
    pos = lambda ci: ci[0] is not None and ci[0] > 0                                 # interval entirely above 0
    res = {'seeds': seeds}
    for s in seeds:
        sp = torch.load(EV.OUT / f'tierB2_units_{a.prefix_sp}{s}.pt', weights_only=False)
        rw = torch.load(EV.OUT / f'tierB2_units_{a.prefix_raw}{s}.pt', weights_only=False)
        out = {}
        # B1 attack recall
        assert [x[:3] for x in sp['att']] == [x[:3] for x in rw['att']]
        ep = [x[0] for x in sp['att']]; by = groups(ep)
        hs = torch.tensor([x[3] <= -0.5 for x in sp['att']], dtype=torch.float64); hr = torch.tensor([x[3] <= -0.5 for x in rw['att']], dtype=torch.float64)
        f = lambda idx: float(hs[idx].mean() - hr[idx].mean()) if idx else float('nan')
        nz = lambda x: None if x != x else round(x, 4)                               # NaN -> None
        out['B1_attack_recall'] = {'spatial': nz(float(hs.mean())), 'rawpred': nz(float(hr.mean())), 'diff': nz(f(list(range(len(ep))))),
                                   'ci95': boot(f, by, g), 'events': len(ep), 'episodes': len(by)}
        # B1 Pearson (graded)
        assert [x[:3] for x in sp['b1']] == [x[:3] for x in rw['b1']]
        ep1 = [x[0] for x in sp['b1']]; by1 = groups(ep1)
        ps, pr, real = (torch.tensor([x[3] for x in sp['b1']]), torch.tensor([x[3] for x in rw['b1']]), torch.tensor([x[4] for x in sp['b1']]))
        ctl_s, ctl_r = torch.tensor([x[5] for x in sp['b1']]), torch.tensor([x[5] for x in rw['b1']])
        f1 = lambda idx: pearson(ps[idx], real[idx]) - pearson(pr[idx], real[idx])
        out['B1_pearson'] = {'spatial': round(pearson(ps, real), 4), 'rawpred': round(pearson(pr, real), 4), 'diff': round(f1(list(range(len(ep1)))), 4),
                             'ci95': boot(f1, by1, g), 'reader_control_real_next': {'spatial': round(pearson(ctl_s, real), 4), 'rawpred': round(pearson(ctl_r, real), 4)}}
        # B3 per-root AUC, episodes = DV16 seeds
        rs = torch.tensor([x[2] for x in sp['b3']]); rr = torch.tensor([x[2] for x in rw['b3']])
        assert [x[1] for x in sp['b3']] == [x[1] for x in rw['b3']]
        a_s = torch.tensor([auc_root(rs[j], P5[x[1]]) for j, x in enumerate(sp['b3'])]); a_r = torch.tensor([auc_root(rr[j], P5[x[1]]) for j, x in enumerate(rw['b3'])])
        ok = (~torch.isnan(a_s)).nonzero()[:, 0].tolist()
        by3 = groups([sp['b3'][i][0] for i in ok])
        d3 = (a_s - a_r)[ok]
        f3 = lambda idx: float(d3[idx].mean())
        out['B3_auc5_mean_per_root'] = {'spatial': round(float(a_s[ok].mean()), 4), 'rawpred': round(float(a_r[ok].mean()), 4), 'diff': round(f3(list(range(len(ok)))), 4),
                                        'ci95': boot(f3, by3, g), 'roots': len(ok), 'episodes': len(by3)}
        # B2 guard: calibration MAE (ratio estimator over branching cells), paired by (root, action)
        key = lambda r_: (r_['root'], r_['action'])
        ms = {key(r_): r_ for r_ in sp['b2'] if r_['n'] > 0}; mr = {key(r_): r_ for r_ in rw['b2'] if r_['n'] > 0}
        common = sorted(set(ms) & set(mr))
        by2 = groups([ms[k]['seed'] for k in common])
        es = torch.tensor([ms[k]['abs_err'] for k in common]); er = torch.tensor([mr[k]['abs_err'] for k in common]); n2 = torch.tensor([ms[k]['n'] for k in common], dtype=torch.float)
        f2 = lambda idx: float(es[idx].sum() / n2[idx].sum() - er[idx].sum() / n2[idx].sum())
        ci2 = boot(f2, by2, g)
        out['B2_guard_calibration_mae'] = {'spatial': round(float(es.sum() / n2.sum()), 4), 'rawpred': round(float(er.sum() / n2.sum()), 4),
                                           'diff_spatial_minus_raw': round(f2(list(range(len(common)))), 4), 'ci95': ci2, 'fails': pos(ci2)}
        # B4 guard: zombie census abs error at depth 16 vs decoded-real, paired by (root, future)
        k4 = lambda x: (x[1], x[2], x[3])
        s4 = {k4(x): x for x in sp['b4'] if x[3] == 16}; r4 = {k4(x): x for x in rw['b4'] if x[3] == 16}
        c4 = sorted(set(s4) & set(r4))
        by4 = groups([s4[k][0] for k in c4])
        e_s = torch.tensor([abs(s4[k][4] - s4[k][5]) for k in c4]); e_r = torch.tensor([abs(r4[k][4] - r4[k][5]) for k in c4])
        f4 = lambda idx: float(e_s[idx].mean() - e_r[idx].mean())
        ci4 = boot(f4, by4, g)
        out['B4_guard_zombie_census_abs_err_d16_vs_decoded_real'] = {'spatial': round(float(e_s.mean()), 4), 'rawpred': round(float(e_r.mean()), 4),
                                                                      'diff_spatial_minus_raw': round(f4(list(range(len(c4)))), 4), 'ci95': ci4, 'fails': pos(ci4), 'units': len(c4)}
        # descriptive B4 by depth (imagined / decoded-real / exact), both worlds
        for nm, U in (('spatial', sp), ('rawpred', rw)):
            dd = {}
            for d in (1, 2, 4, 8, 16):
                xs = [x for x in U['b4'] if x[3] == d]
                dd[d] = {'zombie_imag': round(sum(x[4] for x in xs) / len(xs), 3), 'zombie_decoded_real': round(sum(x[5] for x in xs) / len(xs), 3),
                         'zombie_exact': round(sum(x[6] for x in xs) / len(xs), 3), 'cow_imag': round(sum(x[7] for x in xs) / len(xs), 3),
                         'cow_decoded_real': round(sum(x[8] for x in xs) / len(xs), 3), 'cow_exact': round(sum(x[9] for x in xs) / len(xs), 3),
                         'adj_imag': round(sum(x[10] for x in xs) / len(xs), 3), 'adj_decoded_real': round(sum(x[11] for x in xs) / len(xs), 3)}
            out[f'B4_by_depth_{nm}'] = dd
        out['reader_quality_real_next'] = {'spatial': sp['reader_quality'], 'rawpred': rw['reader_quality']}
        res[f'seed{s}'] = out
    pos = lambda ci: ci[0] is not None and ci[0] > 0
    prim = lambda s: (pos(res[f'seed{s}']['B1_attack_recall']['ci95']), pos(res[f'seed{s}']['B3_auc5_mean_per_root']['ci95']))
    guards_fail = lambda s: (res[f'seed{s}']['B2_guard_calibration_mae']['fails'] or res[f'seed{s}']['B4_guard_zombie_census_abs_err_d16_vs_decoded_real']['fails']
                             or None in res[f'seed{s}']['B2_guard_calibration_mae']['ci95'] or None in res[f'seed{s}']['B4_guard_zombie_census_abs_err_d16_vs_decoded_real']['ci95'])
    if all(all(prim(s)) and not guards_fail(s) for s in seeds):
        v = 'better predictive world state'
    elif all(not any(prim(s)) for s in seeds):
        v = 'not supported'
    else:
        v = 'mixed (reported as is)'
    res['verdict'] = v
    print(json.dumps(res, indent=1))
    (EV.OUT / f'stage2_verdict_v2{a.tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
