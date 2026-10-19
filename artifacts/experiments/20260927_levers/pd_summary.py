"""10-10 PROGRAM, prediction P-D (training dynamics), read on the CPU readouts of the 12 stage-1 snapshot exports (cpu180):
M6 (L1) and DS1 (D loss), the same seed / data / architecture from scratch, at 6k, 12k, 18k, 24k, 30k, 36k.
Predeclared (09:10): each loss improves the classes it allocates more to, with stalls or regressions elsewhere. Refuted if one loss
improves every class at least as fast as the other.
Classes: budget.CATS of the one-step setting (17 actions x 4 env keys from each root), pooled day + night. The learnable excess per
token is mean_L2 - mean_floor (error to the factual key minus the unbiased variance over keys).
  improvement   (excess at 6k - excess at 36k) / excess at 6k, per run
  stalls        steps (12k->18k, ..., 30k->36k) where the excess did not fall
  allocation    the run's mean projection share of that class under its OWN loss over the six snapshots (gradanat stage 1; raw
                backbone gradients -- the --adam reading is pending)
  test          refuted if one run's improvement >= the other's in EVERY class; and the rank correlation (Spearman) across classes
                between the allocation difference (L1 - D) and the improvement difference (M6 - DS1)
Also, per snapshot: currency (map fact-wrong share, HUD value-wrong share), facts at depth 1 and 16 (tile accuracy all / edge /
interior; zombie / cow AUC; health / food / drink / energy R^2; inventory; cow census ratio), errors (gen / V at 1 and 16).
Usage: pd_summary.py  -> artifacts/eda/health_chain_v1/pd_summary.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SBV = ROOT / 'artifacts/eda/scoreboard_v1'
GA = ROOT / 'artifacts/eda/health_chain_v1'
STEPS = (6, 12, 18, 24, 30, 36)
CATS = ('entering', 'entering_seen', 'own_state_changed', 'player_sprite_left', 'neighbour_changed', 'sleep_changed', 'no_state_change',
        'hud_changed', 'hud_same', 'player_token')
RUNS = {'M6': ('l1', 'corrt_raw_teacher_s7_fmamba_u36000'), 'DS1': ('deltairis', 'corrt_raw_teacher_s7_fmamba_di_u36000')}


def spearman(a, b):
    ra, rb = torch.tensor(a).argsort().argsort().float(), torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main():
    res = {'excess': {}, 'allocation': {}, 'facts': {}}
    for run, (loss, gname) in RUNS.items():
        ex, al = {c: [] for c in CATS}, {c: [] for c in CATS}
        for k in STEPS:
            b = json.loads((SBV / f'{run}_{k}k' / 'budget.json').read_text())['one']['categories']
            for c in CATS:
                rows = [v for key, v in b.items() if key.split('/')[0] == c]
                n = sum(v['tokens'] for v in rows)
                ex[c].append(sum(v['tokens'] * (v['mean_L2'] - v['mean_floor']) for v in rows) / n if n else float('nan'))
            g = json.loads((GA / f"gradanat_{gname}{'' if k == 36 else f'_at{k}000'}.json").read_text())[loss]['categories']
            for c in CATS:
                al[c].append(g[c]['projection_share'] if c in g else 0.0)
            sb = json.loads((SBV / f'{run}_{k}k' / 'scoreboard.json').read_text())
            cu = json.loads((SBV / f'{run}_{k}k' / 'currency.json').read_text())
            f1, f16 = sb['facts']['1']['imagined'], sb['facts']['16']['imagined']
            pick = lambda f: {q: f[q] for q in ('tile_all', 'tile_edge', 'tile_interior', 'zombie_auc', 'cow_auc', 'health_r2', 'food_r2',
                                                 'drink_r2', 'energy_r2', 'inventory_r2_mean')} | \
                {'cow_census_ratio': round(f['cow_count'] / f['cow_count_5futures'], 3) if f['cow_count_5futures'] else None}
            res['facts'].setdefault(run, {})[f'{k}k'] = {'depth1': pick(f1), 'depth16': pick(f16),
                                                        'map_fact_wrong_share': cu['fact_wrong_share'], 'hud_value_wrong_share': cu['hud_value_wrong_share'],
                                                        'gen1_over_V': sb['errors']['gen_over_V'][0], 'gen16_over_V': sb['errors']['gen_over_V'][15]}
        res['excess'][run] = {c: [round(x, 3) for x in v] for c, v in ex.items()}
        res['allocation'][run] = {c: round(sum(v) / len(v), 4) for c, v in al.items()}
    test = {}
    for c in CATS:
        e = {r: res['excess'][r][c] for r in RUNS}
        if any(x != x for r in RUNS for x in e[r]):
            continue
        imp = {r: (e[r][0] - e[r][-1]) / e[r][0] if e[r][0] else 0.0 for r in RUNS}
        stalls = {r: sum(1 for a, b in zip(e[r][1:], e[r][2:]) if b >= a) for r in RUNS}
        test[c] = {'improvement_M6': round(imp['M6'], 4), 'improvement_DS1': round(imp['DS1'], 4),
                   'stalls_M6': stalls['M6'], 'stalls_DS1': stalls['DS1'],
                   'allocation_L1_minus_D': round(res['allocation']['M6'][c] - res['allocation']['DS1'][c], 4),
                   'excess_36k_M6': e['M6'][-1], 'excess_36k_DS1': e['DS1'][-1]}
    m6_all = all(v['improvement_M6'] >= v['improvement_DS1'] for v in test.values())
    ds_all = all(v['improvement_DS1'] >= v['improvement_M6'] for v in test.values())
    res['P_D'] = {'classes': test, 'M6_improves_every_class_at_least_as_fast': m6_all, 'DS1_improves_every_class_at_least_as_fast': ds_all,
                  'refuted': m6_all or ds_all,
                  'spearman_allocation_diff_vs_improvement_diff': round(spearman([v['allocation_L1_minus_D'] for v in test.values()],
                                                                                 [v['improvement_M6'] - v['improvement_DS1'] for v in test.values()]), 4)}
    (GA / 'pd_summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['P_D'], indent=1))


if __name__ == '__main__':
    main()
