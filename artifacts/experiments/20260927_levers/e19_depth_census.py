"""CPU census: actual E19 conditional label frequencies by local predictor row.

Declared before execution. Reconstruct the exact local row for all five targets
in the saved6000x40 TRAIN ledger: A uses row k; B/C use row k before the cut,
otherwise k-cut, with cut=4-offset as in e19.segments. No model/decoder/probe.
Count targets, recorded death, living ordinary damage and selected mask by row.
Uniform death counts are not equal conditional label frequencies when row totals
differ. This is a supervision-design fact, not proof the trained model exploits
the remaining cue or that balancing it repairs fresh damage. All actual targets
must occur exactly once; B/C ledgers/layouts must match. Atomic source/input-bound
resume per completed set; measurements stay in EDA logs, never auto-NOTEBOOK.
"""
import argparse
import json
from pathlib import Path

import torch
import h16_resume as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORLD = ROOT / 'artifacts/eda/levers_tworlds_v1/state'
POOL = ROOT / 'artifacts/eda/spatial_pool_v1/pool.pt'
MASK = ROOT / 'artifacts/eda/hpctx_labels_v1.pt'


def state_path(bb, seed, arm):
    root = WORLD / f'e19_{arm}_s{seed}_{bb}_from36000'
    info = json.loads((root / 'train.json').read_text())
    path = root / info['file']
    assert R.file_hash(path) == info['sha256']
    return root / 'contract.json', root / 'train.json', path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sets', nargs='+', default=['fmamba:7', 'fmamba:8', 'full:7'])
    args = p.parse_args()
    torch.set_num_threads(3)
    pool = torch.load(POOL, map_location='cpu', mmap=True, weights_only=False)
    mask = torch.load(MASK, map_location='cpu', mmap=True, weights_only=False)['beside'].bool()
    for pair in args.sets:
        bb, seed = pair.split(':')
        files = {arm: state_path(bb, seed, arm) for arm in 'ABC'}
        paths = [Path(__file__), HERE / 'e19.py', HERE / 'h16_resume.py', MASK,
                 *[p for ps in files.values() for p in ps]]
        training = {arm: torch.load(ps[-1], map_location='cpu', mmap=True, weights_only=False)
                    for arm, ps in files.items()}
        assert all(s['update'] == 6000 for s in training.values())
        ref = training['C']
        assert all(torch.equal(ref['ledger'], s['ledger']) and torch.equal(ref['layout'], s['layout'])
                   for s in training.values())
        contract = json.loads(files['C'][0].read_text())
        spec = {'scope': __doc__, 'inputs': {str(p): R.file_hash(p) for p in paths},
                'pool': {'path': str(POOL), 'sha256_in_verified_training_contract': contract['inputs'][str(POOL)]},
                'runtime': {'torch': str(torch.__version__), 'precision': 'CPU integer census', 'threads': 3}}
        tag = f'e19_{bb}_s{seed}__depth_census'
        store = R.Store(HERE / 'evals/resume' / tag, spec)
        with store.lock():
            result = store.load('result')
            if result is None:
                idx = ref['ledger'].long()
                k = torch.arange(5).expand(*idx.shape, 5)
                cut = (4 - ref['layout'].long())[..., None]
                local = torch.where(k < cut, k, k - cut)
                alive = pool['alive'][idx, 1:]
                dh = pool['dh'][idx]
                selected = mask[idx]
                classes = {'death': ~alive, 'ordinary_ge2': alive & (dh <= -2),
                           'unchanged': alive & (dh == 0)}
                classes['other'] = ~(classes['death'] | classes['ordinary_ge2'] | classes['unchanged'])
                out = {}
                for label, row in [('A_original', k), ('B_C_random_boundaries', local)]:
                    totals = torch.bincount(row.flatten(), minlength=5)
                    assert int(totals.sum()) == 6000 * 40 * 5
                    rows = []
                    for depth in range(5):
                        use = row == depth
                        n = int(use.sum())
                        counts = {c: int((use & m).sum()) for c, m in classes.items()}
                        assert sum(counts.values()) == n
                        n_selected = int((use & selected).sum())
                        rows.append({'local_input_row': depth, 'targets': n, **counts,
                            'death_rate': counts['death'] / n,
                            'ordinary_damage_rate': counts['ordinary_ge2'] / n,
                            'mask_selected': n_selected,
                            'selected_death_rate': int((use & selected & classes['death']).sum()) / n_selected})
                    out[label] = {'rows': rows, 'max_min_death_rate_ratio':
                        max(r['death_rate'] for r in rows) / min(r['death_rate'] for r in rows)
                        if min(r['death_rate'] for r in rows) else None}
                result = {'scope': __doc__, 'contract': spec, 'sets': out,
                          'B_C_layout_and_A_B_C_windows_equal': True,
                          'targets_reconstructed': int(local.numel())}
                store.save('result', result, 1)
            R.atomic_json(HERE / 'evals' / f'{tag}.json', result)
            print(json.dumps({'name': tag, 'sets': result['sets']}), flush=True)


if __name__ == '__main__':
    main()
