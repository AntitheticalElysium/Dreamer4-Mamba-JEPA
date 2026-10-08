"""Is a hit intrinsically predictable from the visible history? (2026-10-08, health diagnosis 2.) Data only, no model.
Subset of health_chain.py. Visible-history features from true tokens (teval Probes.zombie > 0.3, true scroll 14 -> 15 for the
post-move player cell, dh labels):
  adj_post   a zombie in frame 14 beside the player's POST-move cell (the hit condition's geometry)
  adj_hist   a zombie beside the player in any of frames 11..14
  since      transitions since the last health drop (any dh < 0) among transitions 0..13 (1 = the previous one), 0 = none in 14
Strata: fresh (E20 flag), adj_post & since == 6 (the cooldown period), adj_post & since in 1..5, adj_post & other,
not adj_post. Population weights restore the TRAIN mix at t >= 14 (census 2026-10-07: 36,068 ordinary hits incl. 1,005 fresh,
2,891,130 unchanged): fresh hits x1, other hits x 35,063/5,000, unchanged x 2,891,130/8,000. P(hit | stratum) = weighted
hits / weighted (hits + unchanged). A deterministic L1 world can only be expected to draw strata with P > 0.5.
Usage: health_strata.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def features():
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    cells = [r * 9 + c for r in range(7) for c in range(9)]
    coords = torch.tensor([[c // 9, c % 9] for c in cells])
    N = len(sub['classes'])
    zom = torch.zeros(N, 4, 63, dtype=torch.bool)                     # frames 11..14, all map cells
    for i in range(0, N, 256):
        x = torch.from_numpy(np.array(mm[i:i + 256, 11:15, :63])).float()
        zom[i:i + 256] = (P.zombie(x.flatten(0, 2))[:, 0].view(len(x), 4, 63) > 0.3)
    disp = torch.tensor(SHIFTS)[sub['scroll']]                          # camera shift = player displacement
    post = torch.tensor([3, 4])[None] + disp
    d_post = (coords[None] - post[:, None]).abs().sum(-1)               # [N,63] distance to the post-move cell
    d_now = (coords - torch.tensor([3, 4])).abs().sum(-1)               # [63]
    adj_post = (zom[:, 3] & (d_post == 1)).any(1)
    adj_hist = (zom & (d_now == 1)[None, None]).any(-1).any(-1)
    neg = sub['dh'][:, :14] < 0                                          # transitions 0..13 (13 = the previous one)
    idx = torch.arange(14)
    last = torch.where(neg, idx, torch.tensor(-1)).amax(1)
    since = torch.where(last >= 0, 14 - last, torch.tensor(0))
    return sub, adj_post, adj_hist, since


def main():
    sub, adj_post, adj_hist, since = features()
    cls, fresh = sub['classes'], sub['fresh']
    hit, unch = cls == 1, cls == 2
    w = torch.where(hit & fresh, 1.0, torch.where(hit, 35063 / 5000, torch.where(unch, 2891130 / 8000, 0.0)))
    strata = {'fresh': fresh,
              'adj_post_since6': adj_post & (since == 6) & ~fresh,
              'adj_post_since1to5': adj_post & (since >= 1) & (since <= 5) & ~fresh,
              'adj_post_since7plus': adj_post & (since >= 7) & ~fresh,
              'adj_post_no_recent_drop': adj_post & (since == 0) & ~fresh,
              'not_adj_post': ~adj_post & ~fresh}
    res = {}
    for k, m in strata.items():
        wh, wu = float(w[m & hit].sum()), float(w[m & unch].sum())
        res[k] = {'hits': int((m & hit).sum()), 'unchanged': int((m & unch).sum()),
                  'population_hit_rate': round(wh / max(wh + wu, 1e-9), 4),
                  'share_of_population_hits': round(wh / float(w[hit].sum()), 4)}
    since_tab = {}
    for s in range(15):
        m = adj_post & ~fresh & (since == s)
        wh, wu = float(w[m & hit].sum()), float(w[m & unch].sum())
        if (m & (hit | unch)).sum() >= 5:
            since_tab[s] = {'n': int((m & (hit | unch)).sum()), 'hit_rate': round(wh / max(wh + wu, 1e-9), 4)}
    res['adj_post_hit_rate_by_since'] = since_tab
    res['checks'] = {'fresh_adj_post': float(adj_post[fresh].float().mean()), 'hits_adj_post': float(adj_post[hit].float().mean()),
                     'unchanged_adj_post': float(adj_post[unch].float().mean())}
    torch.save({'adj_post': adj_post, 'adj_hist': adj_hist, 'since': since, 'weights': w,
                'strata': {k: v for k, v in strata.items()}}, OUT / 'strata.pt')
    (OUT / 'strata.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
