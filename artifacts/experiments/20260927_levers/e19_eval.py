"""E19 factual health/position diagnosis with hash-bound, per-root-batch resume.

Reused diagnostic roots, not a new sealed generalization test. Sample0 actual future
inputs, k>=3, ordinary living successors. Health decoding is fixed from true FIT
states. Reports both all diagnostic roots and the existing TEST-seed subset; retains
per-root predictions/counts for paired seed-cluster intervals. No action/risk head is
trained. Run on the parent as well as A/B/C so extra-budget and intervention effects
are distinguished. A successful health treatment still needs imagination/control tests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260926_diagnosis'))
sys.path.insert(0, str(HERE.parent / '20260921_readout_ladder'))
import check_damage_rule as DR
import check_decision_step as CD
import check_position as POS
import h16_resume as RSM
T, SD, H = CD.T, CD.SD, CD.H
OUT = HERE / 'evals'


def metrics(drawn, masks, root_use):
    v = masks['valid'] & masks['k3'] & root_use[:, None]
    hit = masks['drop2']
    same = masks['dh'] == 0
    recovery = masks['dh'] >= 1
    cases = {'fresh': masks['adjacent'] & ~masks['win'] & ~masks['adjwin'],
             'beside_no_hit': masks['adjacent'] & ~masks['win'] & masks['adjwin'],
             'recent_hit': masks['adjacent'] & masks['win'],
             'not_adjacent': ~masks['adjacent']}
    def row(use):
        n = int(use.sum())
        n_hit = int((use & hit).sum())
        n_false = int((use & same).sum())
        return {'n': n, 'n_hit': n_hit, 'hits_drawn': int((drawn & use & hit).sum()),
                'hit_catch': float(drawn[use & hit].float().mean()) if n_hit else None,
                'n_unchanged': n_false, 'false_drops': int((drawn & use & same).sum()),
                'false_drop_rate': float(drawn[use & same].float().mean()) if n_false else None,
                'drawn_rate': float(drawn[use].float().mean()) if n else None,
                'recovery_n': int((use & recovery).sum()),
                'recovery_false_drops': int((drawn & use & recovery).sum())}
    return {'overall': row(v), **{k: row(v & q) for k, q in cases.items()}}


def cluster_contrast(a, b, seeds, selection, draws=2000):
    """B minus A pooled rate; paired episode-seed resampling with original denominators."""
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    num = np.array([float(((b[g].float() - a[g].float()) * selection[g]).sum()) for g in groups])
    den = np.array([float(selection[g].sum()) for g in groups])
    rng = np.random.default_rng(20261005)
    ii = rng.integers(len(groups), size=(draws, len(groups)))
    dd = den[ii].sum(1)
    values = num[ii].sum(1)[dd > 0] / dd[dd > 0]
    return {'B_minus_A': float(num.sum() / den.sum()) if den.sum() else None,
            'interval95': np.quantile(values, [.025, .975]).tolist() if len(values) else None,
            'transitions': int(den.sum()), 'episode_seeds': len(groups)}


def compare(paths):
    states = [torch.load(p, weights_only=False) for p in paths]
    reference = states[0]
    out = {'scope': 'paired episode-seed bootstrap on reused diagnostic roots; not an actor/sealed test', 'contrasts': {}}
    # Parent->A, A->B, B->C when supplied in that order.
    for a, b in zip(states, states[1:]):
        for key in ('seed', 'test_roots'):
            assert torch.equal(a[key], b[key]), 'Different root/split order'
        assert RSM.digest(a['metadata_contract']) == RSM.digest(b['metadata_contract']), 'Different diagnostic inputs'
        masks, seeds = reference['masks'], reference['seed']
        v = masks['valid'] & masks['k3']
        selectors = {'hit': v & masks['drop2'], 'unchanged': v & (masks['dh'] == 0),
                     'fresh_hit': v & masks['drop2'] & masks['adjacent'] & ~masks['win'] & ~masks['adjwin']}
        out['contrasts'][a['name'] + ':' + b['name']] = {
            condition: {subset + '_' + target: cluster_contrast(a['drawn'][condition], b['drawn'][condition], seeds,
                             use & (torch.ones_like(seeds, dtype=torch.bool) if subset == 'all' else a['test_roots'])[:, None])
                        for subset in ('all', 'test') for target, use in selectors.items()}
            for condition in POS.CONDITIONS}
    RSM.atomic_json(OUT / ('e19_contrast_' + states[-1]['name'] + '.json'), out)
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', type=Path, nargs='+')
    parser.add_argument('--compare', action='store_true', help='per-root .pt files in parent,A,B,C order')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.compare:
        compare(args.paths)
        return
    from d4mj.config import config_from_dict
    import spatial as S
    device = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, fit_roots, fit_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache('raw', device)
    probes = T.Probes(cache, meta, fit_roots, fit_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9
    frames = torch.cat([cache['ctx'].cpu(), fut5[:, 0].cpu()], 1)
    actions = torch.cat([cache['ctx_a'].cpu(), cache['fut_a'].cpu()], 1)
    masks = {k: v[:, 0] for k, v in DR.masks(meta).items()}
    count = len(meta['seed'])
    current_hp = torch.stack([health(frames[:, 3 + k]) for k in range(H)], 1)
    true_next_hp = torch.stack([health(frames[:, 4 + k]) for k in range(H)], 1)
    oracle = true_next_hp < current_hp - 1.5
    data_contract = {'frames': RSM.tensor_hash(frames), 'actions': RSM.tensor_hash(actions),
                     'seed': RSM.tensor_hash(meta['seed']),
                     'masks': {k: RSM.tensor_hash(v) for k, v in masks.items()},
                     'hud_probe': RSM.tensor_hash(probes.hud.w) if hasattr(probes.hud, 'w') else str(type(probes.hud))}
    for path in args.paths:
        world, st = T.load_world(path, device)
        store = RSM.frozen_eval_store(path, st['name'] + '__e19_health_positions',
                    {'conditions': POS.CONDITIONS, 'sample': 0, 'k_min': 3, 'drop_threshold': 1.5},
                    {'frames': frames, 'actions': actions, 'current_hp': current_hp,
                     **{'mask_' + k: v for k, v in masks.items()}}, [S.CHECKPOINT])
        with store.lock():
            completed = store.load('result')
            if completed is None:
                saved = world.time.data.clone()
                drawn = {}
                probabilities = {}
                bs = 16 if world.backbone_kind == 'fmamba' else 64
                with torch.no_grad():
                    for condition, (offs, row) in POS.CONDITIONS.items():
                        w = len(offs)
                        world.time.data.copy_(saved)
                        world.time.data[:w].copy_(saved[row:row + w])
                        hp = torch.zeros(count, H)
                        real = torch.tensor([j == w - 1 or offs[j + 1] == offs[j] + 1 for j in range(w)])
                        for i in range(0, count, bs):
                            committed = store.load(condition + '_batch_' + str(i))
                            b = min(bs, count - i)
                            if committed is not None:
                                hp[i:i + b] = committed
                                continue
                            values = torch.zeros(b, H)
                            for k in range(3, H):
                                indices = torch.tensor(offs) + 3 + k
                                wa = torch.where(real, actions[i:i + b, indices], 0)
                                pred = T.step(world, frames[i:i + b, indices].float(), wa, device, config)
                                values[:, k] = health(pred)
                            store.save(condition + '_batch_' + str(i), values, 1)
                            hp[i:i + b] = values
                        drawn[condition] = hp < current_hp - 1.5
                        probabilities[condition] = hp
                        print(json.dumps({'name': st['name'], 'condition': condition,
                                          **metrics(drawn[condition], masks, torch.ones(count, dtype=torch.bool))['overall']}), flush=True)
                world.time.data.copy_(saved)
                completed = {'name': st['name'], 'seed': meta['seed'], 'test_roots': ~fit_roots,
                             'masks': masks, 'drawn': drawn, 'predicted_hp': probabilities,
                             'current_hp': current_hp, 'oracle': oracle, 'metadata_contract': data_contract,
                             'resume_contract': store.contract}
                store.save('result', completed, 1)
            OUT.mkdir(exist_ok=True)
            RSM.atomic_torch(OUT / (st['name'] + '__e19_health_per_root.pt'), completed)
            summaries = {condition: {subset: metrics(values, masks, selection) for subset, selection in
                            (('all_diagnostic_roots', torch.ones(count, dtype=torch.bool)), ('test_seeds', ~fit_roots))}
                         for condition, values in completed['drawn'].items()}
            r = summaries['w5_at0']['all_diagnostic_roots']
            fresh = r['fresh']['hit_catch']
            readings = {'learnable': r['overall']['hit_catch'] >= .3 and fresh is not None and fresh >= .5
                                     and r['overall']['false_drop_rate'] <= .02,
                        'unshortcut': r['overall']['drawn_rate'] > 0 and
                                     summaries['w4_at0']['all_diagnostic_roots']['overall']['drawn_rate']
                                     >= summaries['w5_at0']['all_diagnostic_roots']['overall']['drawn_rate'] / 2,
                        'no_cost': 'Pending matched teval onestep_all/consequences; not inferred from health'}
            report = {'name': st['name'], 'scope': __doc__, 'metrics': summaries, 'readings': readings,
                      'real_successor_oracle': metrics(oracle, masks, torch.ones(count, dtype=torch.bool)),
                      'resume_contract': store.contract}
            RSM.atomic_json(OUT / (st['name'] + '__e19_health.json'), report)
            print(json.dumps({'name': st['name'], 'readings': readings}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
