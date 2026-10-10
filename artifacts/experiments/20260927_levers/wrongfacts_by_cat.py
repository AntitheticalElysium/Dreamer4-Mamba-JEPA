# Where do the worlds' WRONG FACTS sit? depth-1 (true 4-frame context) predictions, factual future 0, TEST roots; budget categories
# (entering split seen / never seen in the 4 context frames, as budget.py). Probes = currency.py's (scoreboard.AllProbes).
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
for d in ('artifacts/experiments/20260927_levers', 'artifacts/experiments/20260926_diagnosis', 'artifacts/experiments/20260921_readout_ladder'):
    sys.path.insert(0, str(ROOT / d))
import teval as T, stochdiag as SD, budget as BU, scoreboard as SB
from scroll import estimate
torch.set_num_threads(6)
meta, train_roots, train_seeds = T.split(); cache = T.build_cache('raw', torch.device('cpu')); fut5 = SD.token_cache(torch.device('cpu'))[0]
probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds); thr = torch.tensor(probes.thr)
R = len(meta['seed']); root = cache['ctx'][:, -1]; ctx = cache['ctx']
dctx = torch.stack([torch.cat([estimate(ctx[i:i + 64, t].float(), ctx[i:i + 64, t + 1].float()) for i in range(0, R, 64)]) for t in range(3)], 1)
dvec = torch.tensor(BU.SHIFTS)[dctx]; offs = torch.zeros(R, 4, 2, dtype=torch.long)
for t in (2, 1, 0): offs[:, t] = offs[:, t + 1] - dvec[:, t]
ok = (~meta['future_dead'][:, 0, 0]) & ~train_roots
mapc = torch.tensor(SB.MAPC)
res = {}
for v in ('M16', 'DS', 'DC', 'E22MS_zeroed'):
    g = np.load(ROOT / f'artifacts/eda/scoreboard_v1/{v}/gen.npy', mmap_mode='r')
    cnt = torch.zeros(len(BU.CATS), 2)                       # [cat, (cells, wrong)]
    for i in range(0, R, 64):
        ii = torch.arange(i, min(i + 64, R)); ii = ii[ok[ii]]
        if not len(ii): continue
        y = fut5[ii, 0, 0].float(); p = torch.from_numpy(np.ascontiguousarray(g[ii.numpy(), 0])).float()
        cat = BU.categories(meta['root_visible'][ii], meta['future_visible'][ii, 0, 0], root[ii].float(), y, offs[ii])[0][:, mapc]   # [b,62]
        f_ = SB.AllProbes.facts(meta['future_visible'][ii, 0, 0])
        cells = p[:, mapc].flatten(0, 1)
        wrong = (probes.tile(cells).view(len(ii), 62, 17).argmax(-1) != f_['tile']) | ((probes.mob(cells) > thr).view(len(ii), 62, 4) != f_['mobs']).any(-1)
        for c in range(len(BU.CATS)):
            m = cat == c
            cnt[c, 0] += m.sum(); cnt[c, 1] += (wrong & m).sum()
    tot_wrong = cnt[:, 1].sum()
    res[v] = {c: {'cells': int(cnt[k, 0]), 'wrong_rate': round(float(cnt[k, 1] / cnt[k, 0]), 4), 'share_of_all_wrong': round(float(cnt[k, 1] / tot_wrong), 4)}
              for k, c in enumerate(BU.CATS) if cnt[k, 0] > 0}
    print(v, json.dumps(res[v]), flush=True)
(ROOT / 'artifacts/eda/health_chain_v1/wrongfacts_by_cat.json').write_text(json.dumps(res, indent=1))
