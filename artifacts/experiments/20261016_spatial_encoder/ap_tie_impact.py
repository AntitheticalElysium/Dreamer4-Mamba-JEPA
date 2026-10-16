"""Impact of the AP tie defect on the REAL-FRAME reader results (Tier A / Tier B reader quality / Amendment 6), recomputed exactly
from sleep_strat_eval.py's saved per-cell scores (sleep2_units_*): original spatial_eval_v2.average_precision vs metrics_tied.
ap_grouped, per encoder, stage, population and frame group; plus the R62 post-hoc sleeping-AP intervals (sleep_ap_ci.py) and the
Amendment 6 awake-AP guard, re-run with the grouped, exactly replicated bootstrap.
Population A = the Tier A test frames with the Tier A readers (so 'A / all' IS the corrected Tier A AP); population B = the Tier B
test roots' real next frames (the frames behind Tier B's reader-quality AP, with Tier A numerics).
Usage: ap_tie_impact.py -> artifacts/eda/spatial_encoder_v1/ap_tie_impact_real_frames.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402
import spatial_eval_v2 as EV2                                                   # noqa: E402
from metrics_tied import ap_grouped, ap_grouped_boot                            # noqa: E402

import torch                                                                    # noqa: E402

B = 4000


def main():
    res = {'definition': 'original = spatial_eval_v2.average_precision (order-dependent inside ties); corrected = metrics_tied.ap_grouped',
           'per_unit': {}, 'max_abs_change': 0.0}
    U = {f'{a}_s{s}': torch.load(EV.OUT / f'sleep2_units_{a}_s{s}.pt', weights_only=False) for a in ('raw', 'mop', 'full') for s in (1, 2)}
    for name, u in U.items():
        for pop in ('A', 'B'):
            lab = u['labels'][pop]
            fr = {'all': torch.ones_like(lab['sleep']), 'awake': ~lab['sleep'], 'sleeping': lab['sleep']}
            for stage in ('backbone_ln', 'deployed'):
                for mob, key, lk in (('zombie', 'z', 'zl'), ('cow', 'c', 'cl')):
                    s, l = u[pop][stage][key].float(), lab[lk]
                    for g, m in fr.items():
                        mm = m.repeat_interleave(63)
                        o, c = EV2.average_precision(s[mm], l[mm]), ap_grouped(s[mm], l[mm])
                        ties = int(len(s[mm]) - len(torch.unique(s[mm])))
                        res['per_unit'][f'{name}/{pop}/{stage}/{mob}/{g}'] = {'original': None if o is None else round(o, 5),
                                                                             'corrected': None if c is None else round(c, 5),
                                                                             'tied_items': ties}
                        if o is not None and c is not None:
                            res['max_abs_change'] = max(res['max_abs_change'], abs(o - c))
    # R62 post-hoc sleeping-AP intervals and the Amendment 6 awake guard, grouped and exactly replicated
    out = {}
    for s in (1, 2):
        for pop in ('A', 'B'):
            lab = U[f'mop_s{s}']['labels'][pop]
            eps = sorted(set(lab['episode'].tolist())); eo = {e: i for i, e in enumerate(eps)}
            epc = torch.tensor([eo[e] for e in lab['episode'].tolist()]).repeat_interleave(63)
            g = torch.Generator().manual_seed(7 + s)
            draws = torch.randint(len(eps), (B, len(eps)), generator=g)
            mult = torch.zeros(B, len(eps), dtype=torch.long).scatter_add_(1, draws, torch.ones(B, len(eps), dtype=torch.long))
            for grp, m in (('sleeping', lab['sleep']), ('awake', ~lab['sleep'])):
                mm = m.repeat_interleave(63)
                for mob, key, lk in (('zombie', 'z', 'zl'), ('cow', 'c', 'cl')):
                    l = lab[lk][mm]
                    for stage in ('deployed', 'backbone_ln'):
                        F = {a: ap_grouped_boot(U[f'{a}_s{s}'][pop][stage][key].float()[mm], l, epc[mm], mult) for a in ('raw', 'mop', 'full')}
                        for x, y in (('full', 'mop'), ('full', 'raw'), ('mop', 'raw')):
                            d = F[x][1] - F[y][1]; d = d[~torch.isnan(d)]
                            out[f's{s}/{pop}/{grp}/{mob}/{stage}/{x}-{y}'] = [round(F[x][0], 3), round(F[y][0], 3),
                                                                              round(float(d.quantile(.025)), 3), round(float(d.quantile(.975)), 3)]
    res['grouped_bootstrap_intervals'] = out
    (EV.OUT / 'ap_tie_impact_real_frames.json').write_text(json.dumps(res, indent=1))
    big = sorted(((abs((v['original'] or 0) - (v['corrected'] or 0)), k, v) for k, v in res['per_unit'].items()), reverse=True)[:12]
    print(json.dumps({'max_abs_change': res['max_abs_change'], 'largest': [(k, v) for _, k, v in big]}, indent=1))


if __name__ == '__main__':
    main()
