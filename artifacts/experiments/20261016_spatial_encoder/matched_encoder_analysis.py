"""MATCHED_ENCODER_PREREG.md endpoints, gates, readings and the 12k continuation rule (MoP arm vs Raw arm at one budget).

Inputs (computed by the existing audited scripts with each arm's own encoder spec and readers):
  location_dynamics.py  -> erasure/locdyn2_<label>.pt      consequence_check.py -> erasure/conseq_<label>.pt
One draw matrix (4,000 episode-cluster resamples over the diagnosis episodes, seed 20261109) serves every endpoint and both arms.
AP: tie-grouped, exactly replicated (metrics_tied.ap_grouped_boot). Within-root AUC: consequence_within_root.per_root / summarise
on episode-replicated roots.
  E-C  C1 within-root attack AUC (HUD + near refit on the predicted state)       gate: Raw C1 - action prior (COPY + action) > 0
  E-P  D1 shifted-camera stay AP, mean of zombie and cow (absolute; normalized)   gate: Raw D1 - reader-matched persistence > 0
  E-M  D3 zombie arrival AP (absolute; normalized)                                gate: Raw arrival AP - screen copy > 0
normalized = AP(expected) / AP(real next frame) on the same cell category with the same arm's readers.
Usage: matched_encoder_analysis.py --budget 12k --raw D12k --mop M12k -> artifacts/eda/matched_encoder_v1/matched_<budget>.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402
from metrics_tied import ap_grouped_boot                                        # noqa: E402
from consequence_within_root import per_root, summarise                         # noqa: E402

import torch                                                                    # noqa: E402

B = 4000
D = EV.OUT / 'erasure'
OUTD = EV.OUT.parent / 'matched_encoder_v1'
MOBS = (('zombie', 1), ('cow', 2))
M_AUC, M_REL = 0.05, 0.20


def q(v):
    v = v[~torch.isnan(v)]
    return [round(float(v.quantile(.025)), 4), round(float(v.quantile(.975)), 4)]


def reading(ci, m):
    if ci[0] > 0:
        return 'MoP better'
    if ci[1] < 0:
        return 'MoP worse'
    if -m <= ci[0] and ci[1] <= m:
        return f'equivalent within {round(m, 4)}'
    return 'no difference established'


def location(T, S, ep, mult):
    """per arm: (full, boot) APs for D1 (both mobs) and D3, with baselines and ceilings"""
    moved = T['moved'][:, None].expand(-1, 63).reshape(-1)
    out = {}
    for m, b in MOBS:
        ex1 = ((T['l1'] & b) > 0).reshape(-1)
        at = (((T['l0src'] & b) > 0) & T['srcok']).reshape(-1)
        srcok = T['srcok'].reshape(-1)
        cats = {'stay_shifted': ex1 & at & moved, 'stay_static': ex1 & at & ~moved, 'arrive': ex1 & ~at & srcok}
        for cname, pos in cats.items():
            keep = pos | ~ex1
            for k in ('exp_refit', 'pers_refit', 'copy_refit', 'real_refit'):
                s = S[f'{m}_{k}'].reshape(-1).float()
                out[(m, cname, k)] = ap_grouped_boot(s[keep], pos[keep], ep[keep], mult)
        vm = cats['stay_shifted'] | cats['stay_static'] | (~ex1 & at)
        out[(m, 'counts')] = {c: int(v.sum()) for c, v in cats.items()} | {'vacated': int((~ex1 & at).sum())}
    return out


def consequences(Cd, eo, mult):
    T, P = Cd['units'], Cd['pred']
    keep = T['keys_alive'] >= 2
    rows = per_root(T, P, keep)
    full = summarise(rows)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(eo[r['ep']], []).append(i)
    boot = []
    for bdx in range(len(mult)):
        idx = [i for e, c in enumerate(mult[bdx].tolist()) for _ in range(c) for i in by.get(e, [])]
        boot.append(summarise(rows, idx))
    get = lambda st: (full[st]['within_auc_pairs'], torch.tensor([x[st]['within_auc_pairs'] for x in boot], dtype=torch.float64))
    return {'exp_near': get('exp_near'), 'copy_near': get('copy_near'), 'real_hud': get('real_hud'),
            'comparable_roots': sum(1 for r in rows if r['pairs'] > 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--budget', required=True); ap.add_argument('--raw', required=True); ap.add_argument('--mop', required=True)
    a = ap.parse_args()
    L = {arm: torch.load(D / f'locdyn2_{lb}.pt', weights_only=False) for arm, lb in (('raw', a.raw), ('mop', a.mop))}
    C = {arm: torch.load(D / f'conseq_{lb}.pt', weights_only=False) for arm, lb in (('raw', a.raw), ('mop', a.mop))}
    T0 = L['raw']['units']
    assert torch.equal(L['mop']['units']['seed'], T0['seed']) and torch.equal(L['mop']['units']['future'], T0['future'])
    assert torch.equal(C['mop']['units']['seed'], C['raw']['units']['seed']) and torch.equal(C['mop']['units']['action'], C['raw']['units']['action'])
    eps = sorted(set(T0['seed'].tolist()) | set(C['raw']['units']['seed'].tolist())); eo = {e: i for i, e in enumerate(eps)}
    ep = torch.tensor([eo[int(s)] for s in T0['seed']])[:, None].expand(-1, 63).reshape(-1)
    draws = torch.randint(len(eps), (B, len(eps)), generator=torch.Generator().manual_seed(20261109))
    mult = torch.zeros(B, len(eps), dtype=torch.long).scatter_add_(1, draws, torch.ones(B, len(eps), dtype=torch.long))
    loc = {arm: location(L[arm]['units'], L[arm]['scores'], ep, mult) for arm in L}
    con = {arm: consequences(C[arm], eo, mult) for arm in C}
    res = {'budget': a.budget, 'raw': a.raw, 'mop': a.mop, 'episodes': len(eps), 'draws': B, 'units_location': len(T0['seed'])}

    # descriptive per arm (representation ceilings, copy, baselines)
    for arm in ('raw', 'mop'):
        d = {}
        for m, _ in MOBS:
            for c in ('stay_shifted', 'stay_static', 'arrive'):
                d[f'{m}_{c}'] = {k: round(loc[arm][(m, c, k)][0], 4) for k in ('exp_refit', 'pers_refit', 'copy_refit', 'real_refit')}
            d[f'{m}_counts'] = loc[arm][(m, 'counts')]
        d['C1_within_root_attack_auc'] = {k: round(con[arm][k][0], 4) for k in ('exp_near', 'copy_near', 'real_hud')}
        d['comparable_roots'] = con[arm]['comparable_roots']
        res[f'{arm}_descriptive'] = d

    def mean2(arm, c, k):
        return ((loc[arm][('zombie', c, k)][0] + loc[arm][('cow', c, k)][0]) / 2, (loc[arm][('zombie', c, k)][1] + loc[arm][('cow', c, k)][1]) / 2)

    # E-C
    gC = con['raw']['exp_near'][1] - con['raw']['copy_near'][1]
    dC = con['mop']['exp_near'][1] - con['raw']['exp_near'][1]
    res['E-C'] = {'raw': round(con['raw']['exp_near'][0], 4), 'mop': round(con['mop']['exp_near'][0], 4),
                  'diff': round(con['mop']['exp_near'][0] - con['raw']['exp_near'][0], 4), 'ci95': q(dC),
                  'gate_raw_minus_action_prior_ci95': q(gC), 'gate_pass': q(gC)[0] > 0,
                  'mop_minus_action_prior_ci95': q(con['mop']['exp_near'][1] - con['mop']['copy_near'][1])}
    res['E-C']['reading'] = reading(res['E-C']['ci95'], M_AUC) if res['E-C']['gate_pass'] else 'no reading (Raw gate failed)'

    # E-P (absolute and normalized)
    rE, rP, rR = mean2('raw', 'stay_shifted', 'exp_refit'), mean2('raw', 'stay_shifted', 'pers_refit'), mean2('raw', 'stay_shifted', 'real_refit')
    mE, mR = mean2('mop', 'stay_shifted', 'exp_refit'), mean2('mop', 'stay_shifted', 'real_refit')
    gP = rE[1] - rP[1]
    norm = lambda arm: sum(loc[arm][(m, 'stay_shifted', 'exp_refit')][1] / loc[arm][(m, 'stay_shifted', 'real_refit')][1] for m, _ in MOBS) / 2
    normf = lambda arm: sum(loc[arm][(m, 'stay_shifted', 'exp_refit')][0] / loc[arm][(m, 'stay_shifted', 'real_refit')][0] for m, _ in MOBS) / 2
    res['E-P'] = {'raw': round(rE[0], 4), 'mop': round(mE[0], 4), 'diff': round(mE[0] - rE[0], 4), 'ci95': q(mE[1] - rE[1]),
                  'normalized_raw': round(normf('raw'), 4), 'normalized_mop': round(normf('mop'), 4), 'normalized_ci95': q(norm('mop') - norm('raw')),
                  'ceiling_real_raw': round(rR[0], 4), 'ceiling_real_mop': round(mR[0], 4),
                  'gate_raw_minus_persistence_ci95': q(gP), 'gate_pass': q(gP)[0] > 0}
    res['E-P']['reading'] = reading(res['E-P']['ci95'], M_REL * rE[0]) if res['E-P']['gate_pass'] else 'no reading (Raw gate failed)'
    res['E-P']['normalized_reading'] = reading(res['E-P']['normalized_ci95'], M_REL * normf('raw')) if res['E-P']['gate_pass'] else 'no reading'

    # E-M (zombie arrivals)
    zr, zm = loc['raw'][('zombie', 'arrive', 'exp_refit')], loc['mop'][('zombie', 'arrive', 'exp_refit')]
    zc, zrr, zmr = loc['raw'][('zombie', 'arrive', 'copy_refit')], loc['raw'][('zombie', 'arrive', 'real_refit')], loc['mop'][('zombie', 'arrive', 'real_refit')]
    gM = zr[1] - zc[1]
    res['E-M'] = {'raw': round(zr[0], 4), 'mop': round(zm[0], 4), 'diff': round(zm[0] - zr[0], 4), 'ci95': q(zm[1] - zr[1]),
                  'normalized_raw': round(zr[0] / zrr[0], 4), 'normalized_mop': round(zm[0] / zmr[0], 4),
                  'normalized_ci95': q(zm[1] / zmr[1] - zr[1] / zrr[1]), 'screen_copy_raw': round(zc[0], 4),
                  'gate_raw_minus_screen_copy_ci95': q(gM), 'gate_pass': q(gM)[0] > 0,
                  'primary_budget': '36k (descriptive at 6k / 12k)'}
    res['E-M']['reading'] = reading(res['E-M']['ci95'], M_REL * zr[0]) if res['E-M']['gate_pass'] else 'no reading (Raw gate failed)'

    if a.budget == '12k':                                                         # continuation rule
        worse = lambda e, m: res[e]['gate_pass'] and res[e]['ci95'][1] < -m
        decisive = worse('E-C', M_AUC) and worse('E-P', M_REL * rE[0])
        res['continuation'] = {'decisively_worse_on_both_E-C_and_E-P': decisive, 'decision': 'stop at 12k' if decisive else 'continue to 36k'}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f'matched_{a.budget}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('E-C', 'E-P', 'E-M') if k in res} | ({'continuation': res['continuation']} if 'continuation' in res else {}), indent=1))


if __name__ == '__main__':
    main()
