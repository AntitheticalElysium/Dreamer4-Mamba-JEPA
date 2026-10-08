"""Real canvas training: same saved state, uninterrupted four steps vs two+two. Does not alter research worlds."""
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import tworld as T
import h16_resume as R

HERE = Path(__file__).parent
src = T.OUT / 'state/corrt_raw_teacher_s7_fcanvas_u36000.state.pt'
base = torch.load(src, map_location='cpu', weights_only=False)
start = base['update']
del base
log = lambda **kw: print(json.dumps(kw), flush=True)
paths = [HERE / 'gpu_full.state.pt', HERE / 'gpu_split.state.pt']
for path, stop, initial in [(paths[0], start+4, src), (paths[1], start+2, src), (paths[1], start+4, paths[1])]:
    w, _, _ = T.train('corrt','raw','teacher',7,stop,torch.device('cuda'),log,
                     backbone='fcanvas',state_path=path,resume=initial,state_every=2)
    del w
    gc.collect()
    torch.cuda.empty_cache()
a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
params = max(float((a['world'][k].float()-b['world'][k].float()).abs().max()) for k in a['world'])
moments = max(float((a['optimizer']['state'][k][v].float()-b['optimizer']['state'][k][v].float()).abs().max())
              for k in a['optimizer']['state'] for v in a['optimizer']['state'][k]
              if isinstance(a['optimizer']['state'][k][v], torch.Tensor))
out = {'source':str(src),'source_sha256':R.file_hash(src),'start':start,'stop':start+4,
       'parameter_max_abs':params,'optimizer_tensor_max_abs':moments,
       'order_rng_equal':torch.equal(a['order'],b['order']),
       'cpu_rng_equal':torch.equal(a['rng_cpu'],b['rng_cpu']),
       'cuda_rng_equal':all(torch.equal(x,y) for x,y in zip(a['rng_cuda'],b['rng_cuda'])),
       'scope':'Four actual CUDA canvas updates from historical state; no long-run bitwise guarantee',
       'mechanical_tolerance':1e-6,'sources':{str(Path(T.__file__)):R.file_hash(T.__file__)}}
R.atomic_json(HERE/'gpu_resume.json',out,immutable=True)
print(json.dumps(out),flush=True)
assert params <= 1e-6 and moments <= 1e-6 and out['order_rng_equal'] and out['cpu_rng_equal'] and out['cuda_rng_equal'], out
