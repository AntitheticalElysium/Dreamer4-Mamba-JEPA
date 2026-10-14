"""Mechanical verdict for TRANSPORT_COST_PREREG.md (written before any cost or seed-2 result existed). Reads the two seeds'
3,000-update analyses and applies the preregistered gates under BOTH readers:
  R  seed 2: operator - free lower 95% bound > 0 for valid and for all-source valid movement
  C  per seed, operator_cost: G1 movement (all >= 15%, cow >= 20.5%, distinct >= 2.5), G2 coherence (valid >= 90%,
     duplicate <= 5%, absent <= 5%), G4 static (<= max(free_s, 0.5%)), G5 joint ES (<= free_s + 0.002),
     N non-inferiority to the in-process operator (valid >= op - 2pp, valid movement >= op - 5pp, cow >= op - 5pp,
     distinct >= op - 0.5, ES <= op + 0.002)
Verdict: "transition cost established (exploratory)" = C at both seeds and R; otherwise the preregistered failure labels.
Usage: transport_cost_verdict.py -> artifacts/eda/oracle_v1/transport/cost_verdict.json
"""
import json
from pathlib import Path

T = Path(__file__).resolve().parents[3] / 'artifacts/eda/oracle_v1/transport'
READERS = ('real', 'bank')


def gates(arms, arm, free, op=None):
    out = {}
    for r in READERS:
        x, f = arms[arm][r], arms[free][r]
        g = {'G1': x['valid_move'] >= .15 and x['cow_valid_move'] >= .205 and x['distinct_valid'] >= 2.5,
             'G2': x['valid'] >= .90 and x['duplicate'] <= .05 and x['absent'] <= .05,
             'G4': arms[arm]['scored_static_corruption'] <= max(arms[free]['scored_static_corruption'], .005),
             'G5': x['scored_joint_ES'] <= f['scored_joint_ES'] + .002}
        if op is not None:
            o = arms[op][r]
            g['N'] = (x['valid'] >= o['valid'] - .02 and x['valid_move'] >= o['valid_move'] - .05 and
                      x['cow_valid_move'] >= o['cow_valid_move'] - .05 and x['distinct_valid'] >= o['distinct_valid'] - .5 and
                      x['scored_joint_ES'] <= o['scored_joint_ES'] + .002)
        out[r] = g
    return out


def main():
    s1 = json.loads((T / 'seed1_cost/eval_u3000/analysis.json').read_text())
    s2 = json.loads((T / 'seed2/eval_u3000/analysis.json').read_text())
    res = {'seed1': {'operator_cost': gates(s1['arms'], 'operator_cost', 'free', 'operator'),
                     'operator': gates(s1['arms'], 'operator', 'free')},
           'seed2': {'operator_cost': gates(s2['arms'], 'operator_cost', 'free', 'operator'),
                     'operator': gates(s2['arms'], 'operator', 'free')}}
    rep = s2['operator_minus_free']
    res['R_seed2'] = {r: rep[r]['valid']['ci95'][0] > 0 and rep[r]['valid_move']['ci95'][0] > 0 for r in READERS}
    c = {s: all(all(v.values()) for v in res[s]['operator_cost'].values()) for s in ('seed1', 'seed2')}
    res['C'] = c
    R = all(res['R_seed2'].values())
    if c['seed1'] and c['seed2'] and R:
        verdict = 'transition cost established (exploratory)'
    elif c['seed1'] != c['seed2']:
        verdict = 'seed-dependent'
    else:
        failed = {s: {r: [k for k, v in res[s]['operator_cost'][r].items() if not v] for r in READERS} for s in ('seed1', 'seed2')}
        res['failed_gates'] = failed
        flat = {k for s in failed.values() for r in s.values() for k in r}
        verdict = ('cost buys static fidelity with movement' if 'G4' not in flat and ({'G1', 'N'} & flat) else 'not established')
    res['architecture_replication_R'] = R
    res['verdict'] = verdict
    (T / 'cost_verdict.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
