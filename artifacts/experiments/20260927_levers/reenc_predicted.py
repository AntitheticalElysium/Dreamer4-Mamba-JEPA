# Is the encoder's contextual re-encoding PREDICTED by the worlds? depth-1 (true context) predictions, factual future 0.
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
for d in ('artifacts/experiments/20260927_levers', 'artifacts/experiments/20260926_diagnosis', 'artifacts/experiments/20260921_readout_ladder'):
    sys.path.insert(0, str(ROOT / d))
import teval as T, stochdiag as SD, budget as BU, encoder_diag as ED
from scroll import estimate
meta, _, _ = T.split(); cache = T.build_cache('raw', torch.device('cpu')); fut5 = SD.token_cache(torch.device('cpu'))[0]
R = len(meta['seed']); root = cache['ctx'][:, -1]
alive = ~meta['future_dead'][:, 0, 0]
v0, v1 = meta['root_visible'], meta['future_visible'][:, 0, 0]
day = (v0[:, 1521].float() >= 0.5) & (v1[:, 1521].float() >= 0.5)
NC, HS = BU.CATS.index('no_state_change'), BU.CATS.index('hud_same')
out = {}
for v in ('M16', 'DS', 'DC', 'E22MS_zeroed', 'E22MS_sample0'):
    g = np.load(ROOT / f'artifacts/eda/scoreboard_v1/{v}/gen.npy', mmap_mode='r')
    acc = {}
    for i in range(0, R, 64):
        ii = torch.arange(i, min(i + 64, R)); ii = ii[alive[ii] & day[ii]]
        if not len(ii): continue
        x0, y = root[ii].float(), fut5[ii, 0, 0].float()
        p = torch.from_numpy(np.ascontiguousarray(g[ii.numpy(), 0])).float()
        cat = BU.categories(v0[ii], v1[ii], x0, y)[0]
        a0, a1 = ED.frame_state(v0[ii])[0], ED.frame_state(v1[ii])[0]
        frac = []
        for dr, dc in BU.SHIFTS:
            src = ED.neighbour(a0, dr, dc, -1); ov = src >= 0
            frac.append(((src != a1) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
        sh = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != estimate(x0, y)[:, None]).float()).argmin(1)
        # source token of each cell (the content's position at t), for the copy predictor
        srcidx = torch.full((len(ii), 81), -1, dtype=torch.long)
        for s, (dr, dc) in enumerate(BU.SHIFTS):
            m = sh == s
            for c in range(63):
                r_, c_ = c // 9 + dr, c % 9 + dc
                if 0 <= r_ < 7 and 0 <= c_ < 9: srcidx[m, c] = r_ * 9 + c_
            srcidx[m, 63:] = torch.arange(63, 81)
        copy = torch.gather(x0, 1, srcidx.clamp(min=0)[..., None].expand(-1, -1, 192))
        e_w = ((p - y) ** 2).sum(-1); e_c = ((copy - y) ** 2).sum(-1)
        for name, m in (('unchanged_day_scroll', (cat == NC) & (sh != 0)[:, None] & (srcidx >= 0)),
                        ('unchanged_day_still', (cat == NC) & (sh == 0)[:, None] & (srcidx >= 0)),
                        ('hud_same_day', cat == HS)):
            a = acc.setdefault(name, [0.0, 0.0, 0])
            a[0] += float(e_w[m].sum()); a[1] += float(e_c[m].sum()); a[2] += int(m.sum())
    out[v] = {k: {'tokens': n, 'world_err': round(w / n, 4), 'copy_err_(=target change)': round(c / n, 4), 'share_of_change_predicted': round(1 - w / c, 4)} for k, (w, c, n) in acc.items()}
    print(v, json.dumps(out[v]), flush=True)
(ROOT / 'artifacts/eda/health_chain_v1/reenc_predicted.json').write_text(json.dumps(out, indent=1))
