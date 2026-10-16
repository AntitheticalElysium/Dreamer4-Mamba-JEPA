"""Amendment 6 readings, from sleep_strat_eval.py's units (and tierb_eval_v2.py's units for reader-relative predictor fidelity).

Arms per seed s: raw_s (Raw encoder + Raw-predictor s), mop_s (MoP + SIGReg s), full_s (full s); a missing arm is skipped.
Bootstrap: 4,000 resamples of the 40 test episodes (one draw matrix shared by every statistic and pair); percentile 95%.
M1 = zombie false-positive rate in SLEEPING frames of population B (ratio estimator), deployed and backbone_ln stages.
Guard = zombie AP in AWAKE frames of population B (deployed): exact AP of the episode-replicated sample (copies are tied and adjacent).
Reading (full vs MoP, deployed M1): "reduces" iff the interval lies entirely below 0 at both seeds; "does not" iff it does not lie
below 0 at either seed; else "mixed"; "traded, not reduced" if it reduces while the guard interval lies entirely below 0 at a seed.
Level 3 (descriptive): predictor false-positive rate at certain-empty cells after transitions into sleep vs the reader on the real
next frames; Tier B reader-relative fidelity (Pearson of predicted vs reader-on-real-next health change; B2 |weighted - real_next|).
Usage: sleep_strat_analysis.py [--seeds 1,2] -> artifacts/eda/spatial_encoder_v1/sleep2_verdict.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402  (paths)

B = 4000
TIERB = {'raw': 'rawpred_s', 'mop': 'spatial_s', 'full': 'full_s'}


def ci(v):
    v = v[~torch.isnan(v)]
    return [round(float(v.quantile(.025)), 5), round(float(v.quantile(.975)), 5)] if len(v) else [None, None]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='1,2'); a = ap.parse_args()
    res = {}
    for s in [int(x) for x in a.seeds.split(',')]:
        U = {arm: torch.load(EV.OUT / f'sleep2_units_{arm}_s{s}.pt', weights_only=False) for arm in TIERB
             if (EV.OUT / f'sleep2_units_{arm}_s{s}.pt').exists()}
        lb = next(iter(U.values()))['labels']['B']
        for u in U.values():
            assert all(torch.equal(u['labels']['B'][k], lb[k]) for k in lb)
        eps = sorted(set(lb['episode'].tolist())); e_of = {e: i for i, e in enumerate(eps)}
        draws = torch.randint(len(eps), (B, len(eps)), generator=torch.Generator().manual_seed(20261100 + s))
        mult = torch.zeros(B, len(eps)).scatter_add_(1, draws, torch.ones(B, len(eps)))      # episode multiplicity per draw
        ep_cell = torch.tensor([e_of[e] for e in lb['episode'].tolist()]).repeat_interleave(63)
        sleep_c, awake_c, zl = lb['sleep'].repeat_interleave(63), (~lb['sleep']).repeat_interleave(63), lb['zl']
        out = {'episodes': len(eps), 'sleeping_frames_B': int(lb['sleep'].sum()),
               'sleeping_episodes_B': len(set(lb['episode'][lb['sleep']].tolist())),
               'zombie_positives_in_sleeping_B': int((zl & sleep_c).sum())}
        stat = {}
        for arm, u in U.items():
            th = u['thresholds']
            for stage in ('deployed', 'backbone_ln'):
                fp = ((u['B'][stage]['z'] > th[stage]['zombie']) & ~zl & sleep_c).double()
                neg = (~zl & sleep_c).double()
                F = torch.zeros(len(eps), dtype=torch.float64).index_add_(0, ep_cell, fp)
                N = torch.zeros(len(eps), dtype=torch.float64).index_add_(0, ep_cell, neg)
                stat[(arm, 'M1', stage)] = ((mult.double() @ F) / (mult.double() @ N).clamp_min(1), float(F.sum() / N.sum()))
            sc = u['B']['deployed']['z'][awake_c]; o = sc.argsort(descending=True)
            l_s, e_s = zl[awake_c][o], ep_cell[awake_c][o]

            def ap_w(w, l_s=l_s, e_s=e_s):
                lab = l_s.repeat_interleave(w[e_s].long()).double()
                return float((lab.cumsum(0) / torch.arange(1, len(lab) + 1) * lab).sum() / lab.sum())
            stat[(arm, 'awakeAP', 'deployed')] = (torch.tensor([ap_w(mult[b]) for b in range(B)], dtype=torch.float64),
                                                  ap_w(torch.ones(len(eps))))
            out[arm] = {'M1_deployed': round(stat[(arm, 'M1', 'deployed')][1], 5), 'M1_backbone_ln': round(stat[(arm, 'M1', 'backbone_ln')][1], 5),
                        'awake_zombie_ap_B_deployed': round(stat[(arm, 'awakeAP', 'deployed')][1], 4),
                        'summary': json.loads((EV.OUT / f'sleep2_{arm}_s{s}.json').read_text())}
        pairs = [(x, y) for x, y in (('full', 'mop'), ('full', 'raw'), ('mop', 'raw')) if x in U and y in U]
        for x, y in pairs:
            out[f'{x}_minus_{y}'] = {f'{m}_{st}': {'diff': round(stat[(x, m, st)][1] - stat[(y, m, st)][1], 5), 'ci95': ci(stat[(x, m, st)][0] - stat[(y, m, st)][0])}
                                     for m, st in (('M1', 'deployed'), ('M1', 'backbone_ln'), ('awakeAP', 'deployed'))}
        # level 3: predictor on certain-empty cells, sleeping-next rows, per-episode sums
        L3 = {}
        for arm, u in U.items():
            t = torch.tensor([x[3:] for x in u['L3']], dtype=torch.float64); e = torch.tensor([e_of[x[0]] for x in u['L3']])
            m = t[:, 0] >= .5
            sums = [torch.zeros(len(eps), dtype=torch.float64).index_add_(0, e[m], t[m, j]) for j in (1, 2, 3, 4)]
            L3[arm] = ([x[:5] for x in u['L3']], [(mult.double() @ v) for v in sums], [float(v.sum()) for v in sums])
            n_b, n = L3[arm][1][0].clamp_min(1), L3[arm][2][0]
            out[arm]['L3_sleeping_next'] = {'empty_cells': int(n), 'fp_pred_expected': round(L3[arm][2][1] / n, 5), 'fp_pred_sample': round(L3[arm][2][2] / n, 5),
                                            'fp_real_next': round(L3[arm][2][3] / n, 5),
                                            'pred_expected_minus_real_next': {'diff': round((L3[arm][2][1] - L3[arm][2][3]) / n, 5),
                                                                              'ci95': ci((L3[arm][1][1] - L3[arm][1][3]) / n_b)}}
        for x, y in pairs:
            assert L3[x][0] == L3[y][0]
            out[f'{x}_minus_{y}']['L3_fp_pred_expected_sleeping_next'] = {
                'diff': round((L3[x][2][1] - L3[y][2][1]) / L3[x][2][0], 5), 'ci95': ci((L3[x][1][1] - L3[y][1][1]) / L3[x][1][0].clamp_min(1))}
        # Tier B reader-relative predictor fidelity (existing units)
        for arm in U:
            f = EV.OUT / f'tierB2_units_{TIERB[arm]}{s}.pt'
            if not f.exists():
                continue
            T = torch.load(f, weights_only=False)
            p, c = torch.tensor([x[3] for x in T['b1']]), torch.tensor([x[5] for x in T['b1']])
            b2 = [r for r in T['b2'] if r['n'] > 0]
            out[arm]['tierB_reader_relative'] = {
                'B1_pearson_pred_vs_reader_real_next': round(float(torch.corrcoef(torch.stack([p, c]))[0, 1]), 4),
                'B2_row_abs_weighted_minus_real_next_per_cell': round(sum(abs(r['weighted'] - r['real_next']) for r in b2) / sum(r['n'] for r in b2), 4)}
        res[f'seed{s}'] = out
    seeds = [k for k in res if 'full_minus_mop' in res[k]]
    if seeds:
        below = lambda k, m: res[k]['full_minus_mop'][m]['ci95'][1] is not None and res[k]['full_minus_mop'][m]['ci95'][1] < 0
        if len(seeds) < 2:
            v = 'incomplete (the reading needs both seeds)'
        elif all(below(k, 'M1_deployed') for k in seeds):
            v = 'traded, not reduced' if any(below(k, 'awakeAP_deployed') for k in seeds) else 'reduces'
        elif not any(below(k, 'M1_deployed') for k in seeds):
            v = 'does not reduce'
        else:
            v = 'mixed'
        res['reading_full_vs_mop_sleep_confound'] = {'seeds': seeds, 'reading': v}
    print(json.dumps(res, indent=1))
    (EV.OUT / 'sleep2_verdict.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
