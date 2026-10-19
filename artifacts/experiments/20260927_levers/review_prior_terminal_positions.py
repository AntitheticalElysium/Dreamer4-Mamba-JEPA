"""CPU census of death-target positions in dworld.train_b's current sampler.

Measures supervision support; does not test whether the trained head uses time
as a shortcut. E22 imagination does not use this termination head.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[3]
source = Path(__file__)
sampler = source.parent/'dworld.py'
data = root/'artifacts/eda/levers_mamba_long_pools_v1/raw/labels.pt'
labels = torch.load(data, map_location='cpu', mmap=True, weights_only=False)
terminal, alive = labels['terminal'].bool(), labels['alive'].bool()
blocks = 21
assert alive.shape[1] == 64
assert 'BLOCKS, PER, WIDTH = 21' in sampler.read_text()
positive_terminal = (~alive[terminal,64-blocks+1:64]).sum(0)
positive_main = torch.zeros(blocks-1,dtype=torch.int64)
for start in range(64-blocks):
    positive_main += (~alive[~terminal,start+1:start+blocks]).sum(0)
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
report = {'protocol':__doc__, 'terminal_rows':int(terminal.sum()), 'main_rows':int((~terminal).sum()),
          'blocks':blocks, 'main_start_choices':64-blocks,
          'terminal_training_dead_by_target_position_1_to_20':positive_terminal.tolist(),
          'main_all_allowed_windows_dead_by_target_position_1_to_20':positive_main.tolist(),
          'input':str(data.relative_to(root)), 'input_sha256':digest(data),
          'sampler_source_sha256':digest(sampler), 'script_sha256':digest(source), 'device':'cpu', 'threads':2}
out = root/'artifacts/eda/health_chain_v1/d4mj_prior_terminal_position_census_20261009.json'
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
