"""CPU diagnostic: independent REAL futures scored like E22 damage (no alignment filter)."""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import torch

torch.set_num_threads(2)
source = Path(__file__)
data = Path('artifacts/eda/diagnosis_rollouts_v1/meta.pt')
meta = torch.load(data, map_location='cpu', weights_only=False)
vis = meta['future_visible'].float()
root = meta['root_visible'].float()
hp = torch.cat([root[:, None, None, 1512].expand(-1, vis.shape[1], 1), vis[:, :, :, 1512]], 2) * 9
alive = ~meta['future_dead'].cumsum(2).bool().any(1)
drops = (hp[:, :, 1:] < hp[:, :, :-1] - 0.5) & alive[:, None, :]
same = (hp[:, 0, 1:] == hp[:, 0, :-1]) & alive
ref = drops[:, 0]
ensemble = drops[:, 1:].float().mean(1)
seeds = meta['seed'].numpy()
unique = np.unique(seeds)
rng = np.random.default_rng(20261009)
indices = rng.integers(0, len(unique), (2000, len(unique)))
rows = []
for lo, hi in [(0, 16), (0, 1), (1, 4), (4, 8), (8, 16)]:
    same_, ref_, pred_ = same[:, lo:hi], ref[:, lo:hi], ensemble[:, lo:hi]
    sums = np.array([[float((pred_[seeds == s] * ref_[seeds == s]).sum()),
                      int(ref_[seeds == s].sum()),
                      float((pred_[seeds == s] * same_[seeds == s]).sum()),
                      int(same_[seeds == s].sum())] for s in unique])
    boot = sums[indices].sum(1)
    total = sums.sum(0)
    rows.append({'depths': [lo + 1, hi], 'hits': int(total[1]), 'unchanged': int(total[3]),
                 'caught': float(total[0] / total[1]),
                 'caught_ci95': np.quantile(boot[:, 0] / np.maximum(boot[:, 1], 1), [.025, .975]).tolist(),
                 'false': float(total[2] / total[3]),
                 'false_ci95': np.quantile(boot[:, 2] / np.maximum(boot[:, 3], 1), [.025, .975]).tolist()})

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

out = {'control': 'Independent real simulator keys1..4 vs realized key0 under E22 alive-any/drop/same masks. No camera-alignment conditioning; compares caught_all rather than caught_aligned. Empirical real-simulator baseline, not a proven information ceiling.',
       'contract': {'source_sha256': digest(source), 'input': str(data), 'input_sha256': digest(data),
                    'device': 'CPU', 'threads': 2, 'bootstrap_seed': 20261009, 'bootstrap_draws': 2000},
       'episode_clusters': len(unique), 'clustered_intervals': rows}
Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_real_key_event_agreement.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
