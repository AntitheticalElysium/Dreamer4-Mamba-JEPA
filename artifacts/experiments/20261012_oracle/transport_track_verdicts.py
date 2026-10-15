"""Mechanical verdicts for TRANSPORT_SF_PREREG.md (Track A) and TRANSPORT_PI_PREREG.md (Track B), written 2026-10-15 evening
before either result existed. Readings exactly as preregistered; bright evaluation roots; both readers for one-step gates.

Shared one-step conditions (arm x, analysis.json of its evaluation, BOTH readers): valid >= 90%, duplicate <= 5%, absent <= 5%,
all-source valid movement >= 15%, cow valid movement >= 20.5%; scored static corruption <= max(free, 0.5%).
Recursion readings: persisting-mob score at depth 16 = mob_fading's vanish_rate_by_depth[15].mean_score_of_persisting;
cow census = recursion_<name>.json bright census ratio [15][1]; seen-terrain error = tile_error_seen_by_depth[15];
phantoms = phantoms_per_frame_by_depth (every depth).
SF  (per seed, icsf vs ic at 4,500 updates): P1 score(icsf) - score(ic) >= 0.05; P2 |cow(icsf) - 1| < |cow(ic) - 1|.
    "training failure: self-exposure was a major missing ingredient" if P1, P2 and icsf's one-step conditions hold at both
    seeds; "structural account stands" if neither P1 nor P2 holds at any seed; otherwise "partial" (reported as partial).
PI  (per seed, pi u3000 vs the IC v3 run of that seed): P1 score(pi) - score(ic) >= 0.15; P2 |cow(pi) - 1| < |cow(ic) - 1|;
    P3 seen-terrain(pi) <= seen-terrain(ic) + 0.010; P4 max phantoms(pi) <= 0.05.
    "persistence principle supported" if P1-P4 and pi's one-step conditions hold at both seeds; "not supported" if P1 fails
    at either seed; "persistence helps but decisions are wrong" if P1 holds at both seeds and P3 or P4 fails at either;
    otherwise "partial" with the failed readings listed.
Copy-shortcut guard (NOTEBOOK R19, declared before either result; does not change the verdicts above): a no-op world (root
repeated) passes PI's P1-P4 against IC, so a recursion gain counts as dynamics only if, at both seeds,
  (a) the pooled recursion move rate of cows and of zombies (mob_fading 'move') is >= half the real rate (REAL entry);
  (b) zombie- and cow-adjacency AUC16 (decision_proxies, bright) exceed the no-op's (decision_noop.json);
  (c) camera consistency at depth 16 >= the control's - 0.05.
Evaluation-only PI variants (20261015_whiteboard/PI_CANVAS_PREREG.md), written before either variant's result:
canvas  C1 entering_<name>.json bright.REVISIT.tile_error(pi_canvas) <= 0.5 x pi's; C2 seen terrain d16 <= pi + 0.005, max
        phantoms <= pi + 0.02, camera d16 >= pi - 0.02. "addressed world-aligned memory works in recursion" if C1 and C2 at
        both seeds; "memory recovers terrain but harms the rollout" if C1 holds and C2 fails; "not supported" if C1 fails.
sampled S1 decision health_drop auc_16(pi_sampled) > pi's AND |imagined_rate_within_16 - real| smaller than pi's; S2 seen
        terrain d16 <= pi + 0.005, max phantoms <= pi + 0.02, zombie_adj auc_16 >= pi - 0.02. "rare consequences need sampled
        innovation" if S1 and S2 at both seeds; "sampling draws consequences but harms the rollout" if S1 holds and S2 fails;
        "no gain from sampling" if S1 fails.
Sampled innovation content (20261015_whiteboard/PI_INNOV_PREREG.md), written before its result: per seed, pi_innov vs pi:
  A1 cow arrivals per frame at d16 (census_split_pi_innov_s1_pi_innov_s2.json, cow_absent imagined d16) in [0.25, 1.5] x real;
  A2 cow census d16 closer to 1 than pi's; H persisting-mob score d16 >= pi - 0.05, seen terrain d16 <= pi + 0.005, camera d16
  >= pi - 0.02, phantoms per frame <= real_phantoms per frame at every depth. "sampled innovation restores arrivals" if A1, A2,
  H at both seeds; "arrivals restored but the rollout is harmed" if A1 at both seeds and H fails; "not supported" if A1 fails.
Usage: transport_track_verdicts.py sf|pi|canvas|sampled|innov -> artifacts/eda/oracle_v1/transport/<mode>_verdict.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / 'artifacts/eda/oracle_v1/transport'
W = ROOT / 'artifacts/eda/whiteboard'
READERS = ('real', 'bank')


def onestep(analysis, arm):
    arms = analysis['arms']
    out = {r: bool(arms[arm][r]['valid'] >= .90 and arms[arm][r]['duplicate'] <= .05 and arms[arm][r]['absent'] <= .05 and
                   arms[arm][r]['valid_move'] >= .15 and arms[arm][r]['cow_valid_move'] >= .205) for r in READERS}
    out['static'] = bool(arms[arm]['scored_static_corruption'] <= max(arms['free']['scored_static_corruption'], .005))
    return out


def readings(name, fading_file):
    rec = json.loads((T / f'recursion_{name}.json').read_text())['bright']
    fade = json.loads((W / fading_file).read_text())[name]
    return {'persisting_score_d16': fade['vanish_rate_by_depth'][15]['mean_score_of_persisting'],
            'cow_census_d16': rec['census_ratio[zombie,cow,skeleton]_by_depth'][15][1],
            'seen_terrain_d16': rec['tile_error_seen_by_depth'][15],
            'phantoms_max': max(rec['phantoms_per_frame_by_depth']),
            'zombie_census_d16': rec['census_ratio[zombie,cow,skeleton]_by_depth'][15][0],
            'camera_d16': rec['camera_consistent_by_depth'][15]}


def guard(name, fading_file, control_camera):
    fade = json.loads((W / fading_file).read_text())
    dec = json.loads((W / f'decision_{name}.json').read_text())['bright']
    noop = json.loads((W / 'decision_noop.json').read_text())['bright']
    cam = json.loads((T / f'recursion_{name}.json').read_text())['bright']['camera_consistent_by_depth'][15]
    out = {k: {'move': fade[name][k]['move'], 'real': fade['REAL'][k]['move']} for k in ('cow', 'zombie')}
    out['a'] = all(fade[name][k]['move'] >= .5 * fade['REAL'][k]['move'] for k in ('cow', 'zombie'))
    out['auc16'] = {k: [dec[k]['auc_16'], noop[k]['auc_16']] for k in ('zombie_adj', 'cow_adj')}
    out['b'] = all(dec[k]['auc_16'] > noop[k]['auc_16'] for k in ('zombie_adj', 'cow_adj'))
    out['camera_d16'] = [cam, control_camera]
    out['c'] = cam >= control_camera - .05
    return out


def sf():
    res = {}
    for s in (1, 2):
        f = f'mob_fading_sf_ic_s{s}_sf_icsf_s{s}.json'
        x, c = readings(f'sf_icsf_s{s}', f), readings(f'sf_ic_s{s}', f)
        one = onestep(json.loads((T / f'sf_seed{s}/eval_u4500/analysis.json').read_text()), 'icsf')
        res[f'seed{s}'] = {'icsf': x, 'ic': c, 'one_step_icsf': one,
                           'P1': x['persisting_score_d16'] - c['persisting_score_d16'] >= .05,
                           'P2': abs(x['cow_census_d16'] - 1) < abs(c['cow_census_d16'] - 1),
                           'guard_icsf': guard(f'sf_icsf_s{s}', f, c['camera_d16'])}
    seeds = [res['seed1'], res['seed2']]
    if all(r['P1'] and r['P2'] and all(r['one_step_icsf'].values()) for r in seeds):
        v = 'training failure: self-exposure was a major missing ingredient'
    elif not any(r['P1'] or r['P2'] for r in seeds):
        v = 'structural account stands'
    else:
        v = 'partial'
    res['verdict'] = v
    return res


def pi():
    res = {}
    ic_fading = {1: 'mob_fading_ic_s1_esrep_s1_ic_s1_ref.json', 2: 'mob_fading_ic_s2.json'}
    for s in (1, 2):
        x = readings(f'pi_s{s}', 'mob_fading_pi_s1_pi_s2_pi_s1_ref.json')
        c = readings(f'ic_s{s}', ic_fading[s])
        one = onestep(json.loads((T / f'pi_seed{s}/eval_u3000/analysis.json').read_text()), 'pi')
        res[f'seed{s}'] = {'pi': x, 'ic': c, 'one_step_pi': one,
                           'P1': x['persisting_score_d16'] - c['persisting_score_d16'] >= .15,
                           'P2': abs(x['cow_census_d16'] - 1) < abs(c['cow_census_d16'] - 1),
                           'P3': x['seen_terrain_d16'] <= c['seen_terrain_d16'] + .010,
                           'P4': x['phantoms_max'] <= .05,
                           'guard_pi': guard(f'pi_s{s}', 'mob_fading_pi_s1_pi_s2_pi_s1_ref.json', c['camera_d16'])}
    seeds = [res['seed1'], res['seed2']]
    failed = sorted({k for r in seeds for k in ('P1', 'P2', 'P3', 'P4') if not r[k]} |
                    {'one_step:' + k for r in seeds for k, ok in r['one_step_pi'].items() if not ok})
    if not failed:
        v = 'persistence principle supported'
    elif 'P1' in failed:
        v = 'not supported'
    elif {'P3', 'P4'} & set(failed):
        v = 'persistence helps but decisions are wrong'
    else:
        v = 'partial'
    res['failed'] = failed
    res['verdict'] = v
    return res


def rec(name):
    return json.loads((T / f'recursion_{name}.json').read_text())['bright']


def variant(kind):
    res = {}
    for s in (1, 2):
        base, x = f'pi_s{s}', f'pi_{kind}_s{s}'
        b, v = rec(base), rec(x)
        harm = {'seen_terrain_d16': [v['tile_error_seen_by_depth'][15], b['tile_error_seen_by_depth'][15]],
                'phantoms_max': [max(v['phantoms_per_frame_by_depth']), max(b['phantoms_per_frame_by_depth'])]}
        ok_harm = harm['seen_terrain_d16'][0] <= harm['seen_terrain_d16'][1] + .005 and harm['phantoms_max'][0] <= harm['phantoms_max'][1] + .02
        if kind == 'canvas':
            e = {n: json.loads((W / f'entering_{n}.json').read_text())['bright'] for n in (base, x)}
            harm['camera_d16'] = [v['camera_consistent_by_depth'][15], b['camera_consistent_by_depth'][15]]
            ok_harm = ok_harm and harm['camera_d16'][0] >= harm['camera_d16'][1] - .02
            primary = {'revisit_tile_error': [e[x]['REVISIT']['tile_error'], e[base]['REVISIT']['tile_error']],
                       'new_tile_error': [e[x]['NEW']['tile_error'], e[base]['NEW']['tile_error']]}
            ok_primary = primary['revisit_tile_error'][0] <= .5 * primary['revisit_tile_error'][1]
        else:
            d = {n: json.loads((W / f'decision_{n}.json').read_text())['bright'] for n in (base, x)}
            h = {n: d[n]['health_drop'] for n in (base, x)}
            harm['zombie_adj_auc16'] = [d[x]['zombie_adj']['auc_16'], d[base]['zombie_adj']['auc_16']]
            ok_harm = ok_harm and harm['zombie_adj_auc16'][0] >= harm['zombie_adj_auc16'][1] - .02
            primary = {'health_auc16': [h[x]['auc_16'], h[base]['auc_16']],
                       'health_rate16': [h[x]['imagined_rate_within_16'], h[base]['imagined_rate_within_16'], h[x]['real_rate_within_16']]}
            ok_primary = (h[x]['auc_16'] > h[base]['auc_16'] and
                          abs(h[x]['imagined_rate_within_16'] - h[x]['real_rate_within_16']) < abs(h[base]['imagined_rate_within_16'] - h[base]['real_rate_within_16']))
        res[f'seed{s}'] = {'primary': primary, 'primary_ok': bool(ok_primary), 'no_harm': harm, 'no_harm_ok': bool(ok_harm)}
    seeds = [res['seed1'], res['seed2']]
    labels = {'canvas': ('addressed world-aligned memory works in recursion', 'memory recovers terrain but harms the rollout'),
              'sampled': ('rare consequences need sampled innovation', 'sampling draws consequences but harms the rollout')}[kind]
    if all(r['primary_ok'] and r['no_harm_ok'] for r in seeds):
        res['verdict'] = labels[0]
    elif all(r['primary_ok'] for r in seeds):
        res['verdict'] = labels[1]
    else:
        res['verdict'] = 'not supported' if kind == 'canvas' else 'no gain from sampling'
    return res


def innov():
    res = {}
    split = json.loads((W / 'census_split_pi_innov_s1_pi_innov_s2.json').read_text())
    fade = json.loads((W / 'mob_fading_pi_innov_s1_pi_innov_s2.json').read_text())
    base_fade = json.loads((W / 'mob_fading_pi_s1_pi_s2_pi_s1_ref.json').read_text())
    for s in (1, 2):
        x, b = rec(f'pi_innov_s{s}'), rec(f'pi_s{s}')
        cow = split[f'pi_innov_s{s}']['cow_absent']
        arr, real = cow['imagined_per_frame_d1_d8_d16'][2], cow['real_per_frame_d1_d8_d16'][2]
        census = [x['census_ratio[zombie,cow,skeleton]_by_depth'][15][1], b['census_ratio[zombie,cow,skeleton]_by_depth'][15][1]]
        persist = [fade[f'pi_innov_s{s}']['vanish_rate_by_depth'][15]['mean_score_of_persisting'],
                   base_fade[f'pi_s{s}']['vanish_rate_by_depth'][15]['mean_score_of_persisting']]
        harm = {'persisting_d16': persist, 'seen_terrain_d16': [x['tile_error_seen_by_depth'][15], b['tile_error_seen_by_depth'][15]],
                'camera_d16': [x['camera_consistent_by_depth'][15], b['camera_consistent_by_depth'][15]],
                'phantoms_vs_real_by_depth': [x['phantoms_per_frame_by_depth'], x['real_phantoms_per_frame_by_depth']]}
        h_ok = (persist[0] >= persist[1] - .05 and harm['seen_terrain_d16'][0] <= harm['seen_terrain_d16'][1] + .005 and
                harm['camera_d16'][0] >= harm['camera_d16'][1] - .02 and
                all(p <= r for p, r in zip(x['phantoms_per_frame_by_depth'], x['real_phantoms_per_frame_by_depth'])))
        res[f'seed{s}'] = {'cow_arrivals_d16': [arr, real], 'A1': bool(.25 * real <= arr <= 1.5 * real),
                           'cow_census_d16': census, 'A2': bool(abs(census[0] - 1) < abs(census[1] - 1)), 'no_harm': harm, 'H': bool(h_ok),
                           'zombie_arrivals_d16': [split[f'pi_innov_s{s}']['zombie_absent']['imagined_per_frame_d1_d8_d16'][2],
                                                   split[f'pi_innov_s{s}']['zombie_absent']['real_per_frame_d1_d8_d16'][2]]}
    seeds = [res['seed1'], res['seed2']]
    if all(r['A1'] and r['A2'] and r['H'] for r in seeds):
        res['verdict'] = 'sampled innovation restores arrivals'
    elif all(r['A1'] for r in seeds) and not all(r['H'] for r in seeds):
        res['verdict'] = 'arrivals restored but the rollout is harmed'
    elif not all(r['A1'] for r in seeds):
        res['verdict'] = 'not supported'
    else:
        res['verdict'] = 'partial'
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    res = {'sf': sf, 'pi': pi, 'canvas': lambda: variant('canvas'), 'sampled': lambda: variant('sampled'), 'innov': innov}[which]()
    (T / f'{which}_verdict.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
