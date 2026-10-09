"""10-10 PROGRAM Part II-B (dynamics; prediction P-B), CPU, on rollout_export.py's saved predictions. Where does a world's ONE-STEP
error lie, and how much of it can no world remove?
Two one-step settings with several real outcomes from the SAME history (so their spread is the exact RNG floor):
  one   the 17 actions from the root (4 context frames), truth = the four env keys' successors (stochdiag one4; onestep_visible)
  tf1   the factual first step (tf at depth 1), truth = the five futures' first frames (stochdiag fut5; future_visible)
Every token of the predicted frame is put in ONE category (map cells; entering = no source after the view's scroll, taken from the
TRUE tile maps of the root and key 0's successor, as encoder_diag):
  entering | own_state_changed | player_sprite_left | neighbour_changed | sleep_changed | no_state_change   (map cells, x night/day)
  hud_changed | hud_same  (the 18 HUD tokens; HUD values + inventory changed or not)   |   player_token
"changed" compares the root state with KEY 0's successor (the factual one), as a world trained on one realized outcome sees it.
Per category and world: total L2 error to key 0 (sum over dims of the squared error), total L1 error (sum |error|), the RNG floor
(unbiased variance over the keys = the expected L2 error of the best deterministic prediction given the full history), and the excess
(|p - mean|^2 - floor / K). Shares of each world's total L2 and L1 error per category; floor share of its L2 error.
Usage: budget.py <variant> ...  -> artifacts/eda/scoreboard_v1/<variant>/budget.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CATS = ('entering', 'own_state_changed', 'player_sprite_left', 'neighbour_changed', 'sleep_changed', 'no_state_change',
        'hud_changed', 'hud_same', 'player_token')
NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def categories(v0, v1, x0, x1):
    """root visible v0 [n,1534], successor visible v1 [n,1534] (key 0), tokens x0 / x1 [n,81,192] -> category per token [n,81],
    night [n] (from the successor's light)"""
    import encoder_diag as ED
    from scroll import estimate
    t0, m0, l0, s0 = ED.frame_state(v0); t1, m1, l1, s1 = ED.frame_state(v1)
    frac = []
    for dr, dc in SHIFTS:
        src = ED.neighbour(t0, dr, dc, -1)
        ov = src >= 0
        frac.append(((src != t1) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
    est = estimate(x0, x1)
    sh = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)
    n = len(v0)
    cat = torch.full((n, 81), CATS.index('no_state_change'), dtype=torch.long)
    for si, (dr, dc) in enumerate(SHIFTS):
        sel = sh == si
        if not sel.any():
            continue
        has_src = torch.zeros(7, 9, dtype=torch.bool)
        has_src[max(0, -dr):7 - max(0, dr), max(0, -dc):9 - max(0, dc)] = True
        own = (ED.neighbour(t0[sel], dr, dc, -1) != t1[sel]) | (ED.neighbour(m0[sel], dr, dc, -1) != m1[sel])
        nb = torch.zeros_like(own)
        for ndr, ndc in NB4:
            nb |= ED.neighbour(own & has_src[None], ndr, ndc, 0).bool()
        c = torch.full(own.shape, CATS.index('no_state_change'), dtype=torch.long)
        c = torch.where((s0[sel] != s1[sel])[:, None, None].expand_as(c), torch.full_like(c, CATS.index('sleep_changed')), c)
        c = torch.where(nb, torch.full_like(c, CATS.index('neighbour_changed')), c)
        if 0 <= 3 - dr < 7 and 0 <= 4 - dc < 9:
            c[:, 3 - dr, 4 - dc] = CATS.index('player_sprite_left')
        c = torch.where(own, torch.full_like(c, CATS.index('own_state_changed')), c)
        c = torch.where(~has_src[None].expand_as(c), torch.full_like(c, CATS.index('entering')), c)
        cat[sel, :63] = c.flatten(1)
    hud_same = (v0[:, 1512:1516] == v1[:, 1512:1516]).all(1) & (v0[:, 1522:1534] == v1[:, 1522:1534]).all(1)
    cat[:, 63:] = torch.where(hud_same, CATS.index('hud_same'), CATS.index('hud_changed'))[:, None]
    cat[:, 31] = CATS.index('player_token')
    return cat, l1 < 0.5


def main():
    import teval as T
    import stochdiag as SD
    torch.set_num_threads(6)
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5, one4 = SD.token_cache(torch.device('cpu'))
    R = len(meta['seed'])
    root, rootv = cache['ctx'][:, -1].float(), meta['root_visible']
    alive5 = ~meta['future_dead'][:, :, 0]                                                            # [R,5] alive at the first step
    one_dead = meta['onestep_dead']                                                                   # [R,4,17]
    # categories once (key 0 successors), and the floors
    cat_one = torch.empty(R, 17, 81, dtype=torch.long); night_one = torch.empty(R, 17, dtype=torch.bool)
    cat_tf = torch.empty(R, 81, dtype=torch.long); night_tf = torch.empty(R, dtype=torch.bool)
    for i in range(0, R, 32):
        b = min(32, R - i)
        v0 = rootv[i:i + b].repeat_interleave(17, 0); v1 = meta['onestep_visible'][i:i + b, 0].flatten(0, 1)
        c, nt = categories(v0, v1, root[i:i + b].repeat_interleave(17, 0), one4[i:i + b, 0].float().flatten(0, 1))
        cat_one[i:i + b], night_one[i:i + b] = c.view(b, 17, 81), nt.view(b, 17)
        c, nt = categories(rootv[i:i + b], meta['future_visible'][i:i + b, 0, 0], root[i:i + b], fut5[i:i + b, 0, 0].float())
        cat_tf[i:i + b], night_tf[i:i + b] = c, nt
    valid_one = ~one_dead.any(1)                                                                      # all four keys alive
    valid_tf = alive5.all(1)
    for name in sys.argv[1:]:
        d = OUT / name
        pred_one = torch.from_numpy(np.load(d / 'one.npy', mmap_mode='r'))
        pred_tf = torch.from_numpy(np.load(d / 'tf.npy', mmap_mode='r'))
        res = {'variant': name}
        for setting, K in (('one', 4), ('tf1', 5)):
            acc = torch.zeros(len(CATS), 2, 6, dtype=torch.float64)                                  # cat x night x [L2, L1, floor, excess, tokens, pairs]
            for i in range(0, R, 16):
                b = min(16, R - i)
                if setting == 'one':
                    p = pred_one[i:i + b].float()                                                     # [b,17,81,192]
                    Y = one4[i:i + b].float().permute(0, 2, 1, 3, 4)                                  # [b,17,4,81,192]
                    cat, nt, val = cat_one[i:i + b], night_one[i:i + b], valid_one[i:i + b]
                else:
                    p = pred_tf[i:i + b, 0].float()[:, None]                                          # [b,1,81,192]
                    Y = fut5[i:i + b, :, 0].float()[:, None]                                          # [b,1,5,81,192]
                    cat, nt, val = cat_tf[i:i + b][:, None], night_tf[i:i + b][:, None], valid_tf[i:i + b][:, None]
                mu = Y.mean(2)
                s2 = ((Y - mu[:, :, None]) ** 2).sum((2, -1)) / (K - 1)                               # [b,A,81]
                e = p - Y[:, :, 0]
                e2, e1 = (e ** 2).sum(-1), e.abs().sum(-1)
                ex = ((p - mu) ** 2).sum(-1) - s2 / K
                key = (cat * 2 + nt[..., None].long())[val]                                           # [n,81]
                for col, w in ((0, e2), (1, e1), (2, s2), (3, ex), (4, torch.ones_like(e2))):
                    acc.view(-1, 6)[:, col] += torch.bincount(key.flatten(), weights=w[val].flatten().double(), minlength=len(CATS) * 2)
            acc[..., 5] = acc[..., 4] / 81
            tot2, tot1 = float(acc[..., 0].sum()), float(acc[..., 1].sum())
            rows = {}
            for ci, c in enumerate(CATS):
                for nv, nn in ((0, 'day'), (1, 'night')):
                    a = acc[ci, nv]
                    if a[4] == 0:
                        continue
                    rows[f'{c}/{nn}'] = {'L2_share': round(float(a[0]) / tot2, 4), 'L1_share': round(float(a[1]) / tot1, 4),
                                         'floor_share_of_own_L2': round(float(a[2] / a[0]), 4) if a[0] > 0 else None,
                                         'excess_share_of_own_L2': round(float(a[3] / a[0]), 4) if a[0] > 0 else None,
                                         'tokens': int(a[4]), 'mean_L2': round(float(a[0] / a[4]), 3), 'mean_floor': round(float(a[2] / a[4]), 3)}
            res[setting] = {'total_L2_per_frame': round(tot2 / float(acc[..., 5].sum()), 3), 'total_L1_per_frame': round(tot1 / float(acc[..., 5].sum()), 3),
                            'floor_share_of_L2': round(float(acc[..., 2].sum()) / tot2, 4), 'categories': rows}
        (d / 'budget.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({name: {s: {'floor_share': res[s]['floor_share_of_L2'], 'L2pf': res[s]['total_L2_per_frame']} for s in ('one', 'tf1')}}), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
