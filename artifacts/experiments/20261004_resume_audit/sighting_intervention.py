"""Fixed-length historical-token intervention; predeclared in PLAN.md. No training or real-future inputs to predictors."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

LEVER = Path('artifacts/experiments/20260927_levers')
sys.path.insert(0, str(LEVER))
import check_recall as CR
import teval as T
import tworld as TW
from scroll import SHIFTS, estimate

OUT = Path(__file__).parent
SEED = 61004


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def interval(x, clusters):
    """Mean with paired pool-row cluster resampling; strata/arms use identical cases and bootstrap draws."""
    ids, inv = torch.unique(clusters, return_inverse=True)
    sums = torch.zeros(len(ids), dtype=torch.float64).scatter_add_(0, inv, x.double())
    counts = torch.zeros(len(ids), dtype=torch.float64).scatter_add_(0, inv, torch.ones(len(x), dtype=torch.float64))
    draws = torch.randint(len(ids), (2000, len(ids)), generator=torch.Generator().manual_seed(SEED))
    bs = sums[draws].sum(1) / counts[draws].sum(1)
    return {'mean': float(x.double().mean()), 'interval': bs.quantile(torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
            'cases': len(x), 'clusters': len(ids)}


def cases(pool):
    main = torch.where(~pool['terminal'])[0]
    held = main[torch.randperm(len(main), generator=torch.Generator().manual_seed(1))[:2048]]
    items, groups = [], {}
    for row in held.tolist():
        x = pool['tokens'][row].float()
        t, ce, cls, sf, sc, nb = CR.cells(x)
        for v in zip(t.tolist(), ce.tolist(), cls.tolist(), sf.tolist(), sc.tolist(), nb.tolist()):
            tt, c, rec, f, src, nei = v
            if not rec or f >= tt - 1:
                continue
            i = len(items)
            items.append((row, tt, c, f, src, nei))
            groups.setdefault((tt, c, f, src), []).append(i)
    rg = torch.Generator().manual_seed(SEED)
    donors = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        shuffled = torch.tensor(group)[torch.randperm(len(group), generator=rg)].tolist()
        for i, d in zip(shuffled, shuffled[1:] + shuffled[:1]):
            assert items[i][0] != items[d][0]
            donors[i] = d
    chosen = []
    for same in (True, False):
        eligible = [i for i in donors if (items[i][2] == items[i][4]) == same]
        order = torch.randperm(len(eligible), generator=rg).tolist()[:512]
        chosen.extend(eligible[j] for j in order)
    ledger = torch.tensor([items[i] + (items[donors[i]][0],) for i in chosen])
    assert len(ledger) and (ledger[:, 0] != ledger[:, -1]).all()
    print(json.dumps({'stage': 'cases', 'n': len(ledger), 'same_slot': int((ledger[:, 2] == ledger[:, 4]).sum()),
                      'moved_slot': int((ledger[:, 2] != ledger[:, 4]).sum())}), flush=True)
    return ledger


def inputs(pool, ledger):
    arrays, actions, truth, neighbour, sighting, donor = [], [], [], [], [], []
    for row, t, ce, sf, sc, nb, dr in ledger.tolist():
        x = pool['tokens'][row].float()
        ori = x[:t].clone()
        replace, control = ori.clone(), ori.clone()
        d = pool['tokens'][dr, sf, sc].float()
        shifts = torch.tensor(SHIFTS)[estimate(x[:-1], x[1:])]
        off = torch.cat([torch.zeros(1, 2, dtype=torch.long), shifts.cumsum(0)])
        wc = torch.tensor([ce // 9, ce % 9]) + off[t]
        n = 0
        for j in range(t - 1):
            loc = wc - off[j]
            r, c = loc.tolist()
            if not (0 <= r < 7 and 0 <= c < 9):
                continue
            source = r * 9 + c
            unrelated = (source + 17) % 63
            replace[j, source] = d
            control[j, unrelated] = d
            n += 1
        assert n > 0
        assert torch.equal(ori[-1], replace[-1]) and torch.equal(ori[-1], control[-1])
        arrays.append((ori, replace, control))
        actions.append(pool['actions'][row, :t].clone())
        truth.append(x[t, ce]); neighbour.append(x[t, nb]); sighting.append(x[sf, sc]); donor.append(d)
    return arrays, actions, torch.stack(truth), torch.stack(neighbour), torch.stack(sighting), torch.stack(donor)


def main():
    from d4mj.config import config_from_dict
    torch.set_num_threads(2)
    device = torch.device('cuda')
    pool = torch.load(TW.POOLS['raw'] / 'pool.pt', weights_only=False, mmap=True)
    ledger = cases(pool)
    arrays, actions, truth, neighbour, sighting, donor = inputs(pool, ledger)
    config = config_from_dict(torch.load(TW.S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    all_pred, results, sources = {}, {}, {}
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        name = st['name']
        sources[path] = sha(path)
        predictions = torch.empty(len(ledger), 3, 192)
        for t in ledger[:, 1].unique().tolist():
            inds = torch.where(ledger[:, 1] == t)[0]
            for lo in range(0, len(inds), 4):
                batch = inds[lo:lo + 4].tolist()
                acts = torch.stack([actions[i] for i in batch])
                for cond in range(3):
                    frames = torch.stack([arrays[i][cond] for i in batch])
                    pred = T.step(world, frames, acts, device, config)
                    predictions[batch, cond] = pred[torch.arange(len(batch)), ledger[batch, 2]]
            print(json.dumps({'stage': 'predict', 'world': name, 'input_frames': t, 'cases': len(inds)}), flush=True)
        error = (predictions - truth[:, None]).square().sum(-1)
        pull = ((predictions[:, 1:] - predictions[:, :1]) * (donor - sighting)[:, None]).sum(-1) / (donor - sighting).square().sum(-1)[:, None].clamp_min(1e-12)
        by = {}
        for same in (True, False):
            mask = (ledger[:, 2] == ledger[:, 4]) == same
            clusters = ledger[mask, 0]
            base_err = (neighbour[mask] - truth[mask]).square().sum(-1).mean()
            sight_err = (sighting[mask] - truth[mask]).square().sum(-1).mean()
            by['same' if same else 'moved'] = {
                'error': {c: float(error[mask, j].mean()) for j, c in enumerate(('original', 'target_swap', 'unrelated_swap'))},
                'baseline_capture': float((base_err - error[mask, 0].mean()) / (base_err - sight_err)),
                'target_error_increase': interval((error[:, 1] - error[:, 0])[mask], clusters),
                'unrelated_error_increase': interval((error[:, 2] - error[:, 0])[mask], clusters),
                'specific_error_increase': interval((error[:, 1] - error[:, 2])[mask], clusters),
                'target_donor_pull': interval(pull[mask, 0], clusters),
                'unrelated_donor_pull': interval(pull[mask, 1], clusters)}
        results[name] = by
        all_pred[name] = predictions
        print(json.dumps({'stage': 'result', 'world': name, 'result': by}), flush=True)
        del world; torch.cuda.empty_cache()
    torch.save({'ledger_columns': ['pool_row', 'target_t', 'target_cell', 'last_sighting_t', 'sighting_cell', 'inward_neighbour', 'donor_pool_row'],
                'ledger': ledger, 'truth': truth, 'neighbour': neighbour, 'sighting': sighting, 'donor': donor,
                'predictions': all_pred}, OUT / 'sighting_rows.pt')
    results['_provenance'] = {'seed': SEED, 'checkpoints_sha256': sources, 'script_sha256': sha(__file__),
                              'plan_sha256': sha(OUT / 'PLAN.md'), 'pool_sha256': sha(TW.POOLS['raw'] / 'pool.pt')}
    (OUT / 'sighting_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
