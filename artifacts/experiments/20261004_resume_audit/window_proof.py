"""Verify the supervised prediction boundary on the actual frozen A16 checkpoint, on CPU. No weights are updated."""
import hashlib
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import teval as T
import tworld as TW

torch.set_num_threads(2)
path = Path('artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_L16b40_from36000.pt')
w, st = T.load_world(path, torch.device('cpu'))
for v in w.parameters():
    v.requires_grad_(False)
w.time.requires_grad_(True)
rg = torch.Generator().manual_seed(61004)
s = torch.randn(1, 16, 81, 192, generator=rg)
a = torch.randint(17, (1, 15), generator=rg)
loss = TW.rollout_losses(w, s, a, 'teacher', False)
loss.backward()
norms = w.time.grad.norm(dim=-1)
assert norms[-1] == 0 and (norms[:-1] > 0).all()
out = {'world': str(path), 'world_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
       'tworld_sha256': hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest(),
       'loss': float(loss.detach()), 'time_row_gradient_norms': norms.tolist(),
       'largest_supervised_input_window': 15, 'unsupervised_input_window': 16,
       'note': 'Synthetic inputs isolate the causal supervision boundary; this is not a semantic performance test.'}
Path(__file__).with_name('window_proof.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
