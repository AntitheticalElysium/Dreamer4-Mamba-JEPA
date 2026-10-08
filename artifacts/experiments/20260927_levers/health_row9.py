"""Emitted health at the unbiased output row (2026-10-09): for each health_chain world file given (any window; row 9 = window 10),
held hits drawn / fresh / since = 6 hits / unchanged false drops (and those 1..5 transitions after a drop), Probes.hud reader,
1.5 cut; plus the token-63 generator's fidelity on held hits. Usage: health_row9.py <file stem in health_chain_v1> ..."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False); st = torch.load(OUT / 'strata.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    meta, tr, ts = T.split(); P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, tr, ts)
    read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
    N = len(sub['classes']); hit, unch = sub['classes'] == 1, sub['classes'] == 2; held = ~sub['fit']
    cur = torch.cat([read(torch.from_numpy(np.array(mm[i:i + 2048, 14, 63:81]))) for i in range(0, N, 2048)])
    t14 = torch.from_numpy(np.array(mm[:, 14, 63])).float(); t15 = torch.from_numpy(np.array(mm[:, 15, 63])).float()
    cl = (t14 - t15).abs().mean(-1)
    since, fresh = st['since'], sub['fresh']; cool = (since >= 1) & (since <= 5)
    for stem in sys.argv[1:]:
        r = torch.load(OUT / f'{stem}.pt', weights_only=False)
        em = read(r['hud']) < cur - 1.5
        gl = (F.layer_norm(r['gen63'].float(), (192,)) - t15).abs().mean(-1)
        print(f"{stem[-70:]:70s} row {r['window'] - 1:2d}: hits {int((em & hit & held).sum()):4d}/{int((hit & held).sum())} "
              f"fresh {int((em & hit & fresh & held).sum()):3d}/{int((hit & fresh & held).sum())} since6 {int((em & hit & held & (since == 6)).sum()):3d}/"
              f"{int((hit & held & (since == 6)).sum())} | false {int((em & unch & held).sum()):3d}/{int((unch & held).sum())} (cooldown "
              f"{int((em & unch & held & cool).sum())}) | gen beats copy {float((gl[hit & held] < cl[hit & held]).float().mean()):.3f}", flush=True)
