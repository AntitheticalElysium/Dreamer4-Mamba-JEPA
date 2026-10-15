"""Mechanical verdict for TRANSPORT_IC_PREREG.md, written before any IC result existed (2026-10-15).

Per seed s in {1, 2}, from ic_seed<s>/eval_u3000/analysis.json, under BOTH readers, the arm 'ic' must pass
  G1  all-source valid movement >= 15%, cow valid movement >= 20.5%, distinct valid outcomes >= 2.5
  G2  valid >= 90%, duplicate <= 5%, absent <= 5%
  G4  scored static corruption <= max(free_s, 0.5%)        (free_s: Stage 3 seed-1 free / seed-2 free, analysed in-file)
  G5  scored joint ES <= free_s + 0.002
  N   non-inferiority to the in-process ES operator: valid >= ES - 2pp, valid movement >= ES - 5pp, cow >= ES - 5pp,
      distinct >= ES - 0.5, joint ES <= ES + 0.002
Verdict: "inferred chance established (exploratory)" if IC passes every gate at both seeds; "cost bought with movement" if
G4 passes at both seeds while G1 or N fails somewhere; otherwise "not established" with the failed gates listed.

Mechanism (clarified before results: under canonical labels an EMPTY cell next to a mob legitimately proposes the swap, so
the preregistered P(move | no-mob content) is contaminated by legitimate proposals; the preregistration's own risk note says
to compare through accepted swaps). Two clean measures, from transport_diag.py:
  m1  P(move) at no-mob cells with no mob in any 4-neighbour   (content mode, group 'no_mob_isolated')
  m2  accepted terrain<->terrain swaps per frame               (swaps mode: terrain_same + terrain_different)
"mechanism supported" if IC < ES on both measures at both seeds; "mechanism falsified" if IC >= ES on both measures at both
seeds; otherwise "mixed". The contaminated preregistered measure ('no_mob') is reported alongside.
Secondary (recursion, transport_recursion.py, bright): R1 tile error on seen cells at depth 16 IC < ES replicate;
R2 |cow census at depth 16 - 1| IC < ES replicate; R3 IC phantoms per frame <= 0.01 at every depth.
Usage: transport_ic_verdict.py -> artifacts/eda/oracle_v1/transport/ic_verdict.json
"""
import json
from pathlib import Path

T = Path(__file__).resolve().parents[3] / 'artifacts/eda/oracle_v1/transport'
READERS = ('real', 'bank')


def gates(arms, arm, free, op):
    out = {}
    for r in READERS:
        x, f, o = arms[arm][r], arms[free][r], arms[op][r]
        out[r] = {'G1': x['valid_move'] >= .15 and x['cow_valid_move'] >= .205 and x['distinct_valid'] >= 2.5,
                  'G2': x['valid'] >= .90 and x['duplicate'] <= .05 and x['absent'] <= .05,
                  'G4': arms[arm]['scored_static_corruption'] <= max(arms[free]['scored_static_corruption'], .005),
                  'G5': x['scored_joint_ES'] <= f['scored_joint_ES'] + .002,
                  'N': (x['valid'] >= o['valid'] - .02 and x['valid_move'] >= o['valid_move'] - .05 and
                        x['cow_valid_move'] >= o['cow_valid_move'] - .05 and x['distinct_valid'] >= o['distinct_valid'] - .5 and
                        x['scored_joint_ES'] <= o['scored_joint_ES'] + .002)}
    return out


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    res = {}
    for s in (1, 2):
        d = T / f'ic_seed{s}'
        a = load(d / 'eval_u3000/analysis.json')
        res[f'seed{s}'] = {'ic': gates(a['arms'], 'ic', 'free', 'operator')}
        cont = {arm: load(d / f'diag_{"" if arm == "operator" else arm + "_"}u3000.json') for arm in ('operator', 'ic')}
        sw = {arm: load(d / 'eval_u3000' / f'diag_swaps_{"" if arm == "operator" else arm + "_"}u3000.json') for arm in ('operator', 'ic')}
        mech = {}
        if all(cont.values()):
            for arm in ('operator', 'ic'):
                g = cont[arm]['p_move_by_afterstate_content']
                mech[arm] = {'m1_no_mob_isolated': g['no_mob_isolated']['mean'], 'no_mob_contaminated': g['no_mob']['mean']}
        if all(sw.values()):
            for arm in ('operator', 'ic'):
                c = sw[arm]['swaps_per_frame_by_class']
                mech.setdefault(arm, {})['m2_terrain_swaps_per_frame'] = round(c['terrain_same'] + c['terrain_different'], 4)
                mech[arm]['terrain_different_per_frame'] = c['terrain_different']
        res[f'seed{s}']['mechanism_measures'] = mech
    passes = {s: all(all(v.values()) for v in res[s]['ic'].values()) for s in ('seed1', 'seed2')}
    res['IC_passes'] = passes
    failed = {s: {r: [k for k, v in res[s]['ic'][r].items() if not v] for r in READERS} for s in ('seed1', 'seed2')}
    res['failed_gates'] = failed
    flat = {k for s in failed.values() for r in s.values() for k in r}
    if all(passes.values()):
        verdict = 'inferred chance established (exploratory)'
    elif 'G4' not in flat and ({'G1', 'N'} & flat):
        verdict = 'cost bought with movement'
    else:
        verdict = 'not established'
    res['verdict'] = verdict
    lower = []
    for s in ('seed1', 'seed2'):
        m = res[s]['mechanism_measures']
        if 'ic' in m and 'operator' in m and 'm1_no_mob_isolated' in m['ic'] and 'm2_terrain_swaps_per_frame' in m['ic']:
            lower.append((m['ic']['m1_no_mob_isolated'] < m['operator']['m1_no_mob_isolated'],
                          m['ic']['m2_terrain_swaps_per_frame'] < m['operator']['m2_terrain_swaps_per_frame']))
    if len(lower) == 2:
        res['mechanism'] = ('mechanism supported' if all(all(x) for x in lower) else
                            'mechanism falsified' if not any(any(x) for x in lower) else 'mixed')
    rec = {}
    for s in (1, 2):
        ic, es = load(T / f'recursion_ic_s{s}.json'), load(T / f'recursion_esrep_s{s}.json')
        if ic and es:
            bi, be = ic['bright'], es['bright']
            rec[f'seed{s}'] = {
                'R1_tile_seen_d16': [bi['tile_error_seen_by_depth'][15], be['tile_error_seen_by_depth'][15]],
                'R1': bi['tile_error_seen_by_depth'][15] < be['tile_error_seen_by_depth'][15],
                'R2_cow_census_d16': [bi['census_ratio[zombie,cow,skeleton]_by_depth'][15][1], be['census_ratio[zombie,cow,skeleton]_by_depth'][15][1]],
                'R2': abs(bi['census_ratio[zombie,cow,skeleton]_by_depth'][15][1] - 1) < abs(be['census_ratio[zombie,cow,skeleton]_by_depth'][15][1] - 1),
                'R3': max(bi['phantoms_per_frame_by_depth']) <= .01}
    res['recursion'] = rec
    (T / 'ic_verdict.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
