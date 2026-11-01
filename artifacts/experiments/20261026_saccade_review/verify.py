"""Compare separate reproductions and verify a stratified replay sample."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
from saccade.eval import labels
import replay as RP

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'artifacts/eda/s0'


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        return sum((differences(a[k], b[k], f'{path}/{k}') for k in a), [])
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        return sum((differences(x, y, f'{path}/{i}') for i, (x, y) in enumerate(zip(a, b))), [])
    if a == b:
        return []
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return [{'path': path, 'original': a, 'reproduced': b, 'abs_diff': abs(a-b)}]
    raise AssertionError((path, a, b))


def main():
    report = {'comparisons': {}, 'replay_checks': []}
    for stem in ('s0_dev_labels', 's0_train400_labels'):
        with np.load(SOURCE / f'{stem}.npz') as a, np.load(OUT / f'{stem}.npz') as b:
            assert set(a.files) == set(b.files)
            unequal = [k for k in a.files if not np.array_equal(a[k], b[k])]
            report['comparisons'][stem] = {'arrays': len(a.files), 'unequal_arrays': unequal}
    for stem in ('s0_gates', 's0_revisit', 's0_oracle', 's0_probe', 's0_calib'):
        diff = differences(json.loads((SOURCE / f'{stem}.json').read_text()),
                           json.loads((OUT / f'{stem}.json').read_text()))
        report['comparisons'][stem] = {'changed_values': len(diff),
                                      'max_abs_diff': max((d['abs_diff'] for d in diff), default=0),
                                      'differences': diff}
    files = sorted((ROOT / 'saccade').rglob('*.py')) + sorted((ROOT / 'saccade/design').glob('*.md'))
    files += [ROOT / 'artifacts/eda/replay.py', ROOT / 'artifacts/craftax_support_v2/manifest.json']
    report['sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['versions'] = {k: importlib.metadata.version(k) for k in ('torch', 'jax', 'jaxlib', 'numpy', 'lightgbm', 'craftax')}
    report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    path = OUT / 'verification.json'
    path.write_text(json.dumps(report, indent=2))
    L = np.load(OUT / 's0_dev_labels.npz')
    for epsilon in np.unique(L['ep_eps']):
        eligible = np.flatnonzero(L['ep_eps'] == epsilon)
        by_length = eligible[np.argsort(L['ep_T'][eligible])]
        for index in np.unique(by_length[np.array([0, len(by_length)//2, len(by_length)-1])]):
            sh, slot = int(L['ep_shard'][index]), int(L['ep_slot'][index])
            record = {'episode': int(index), 'shard': sh, 'slot': slot,
                      'epsilon': float(epsilon), 'transitions': int(L['ep_T'][index]),
                      'frames_checked': 5, 'max_pixel_difference': RP.verify(sh, slot, checks=5)}
            report['replay_checks'].append(record)
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(record), flush=True)
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'differences'}
                      for k, v in report['comparisons'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
