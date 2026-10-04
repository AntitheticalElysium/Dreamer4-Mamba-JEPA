"""How much does Craftax reward memory? (2026-10-04, Mamba diagnosis; data only, no model.) S4WM (Deng et al. 2023) built
memory tasks because ordinary environments rarely need long recall; R2I (Samsami et al. 2024) needs the dependency inside the
training sequence. Here the one map content a longer history can supply is terrain re-entering the view after the player walked
back: on a scroll step, an entering cell (scroll.estimate's shift; the new edge row / column) whose world position (cell +
cumulative camera offset, scroll's convention b[r, c] ~ a[r + dr, c + dc]) was in view in one of the previous L frames.
The hit cooldown is not a long-memory demand (check_damage_rule: deterministic ceiling 85.7% at 6 frames, 83.3% at 9).
Long pool (levers_mamba_long_pools_v1/raw, 64-frame TRAIN windows), 1,500 windows (seeded), transitions with both frames alive.
Per lookback L (frames before the target, L = 5 is the 6-frame recipe's maximum, 15 E17's L = 16, 63 the pool's maximum):
  recallable_share   entering cells seen within L frames / all entering cells
  recall_err         squared token error of copying the most recent sighting (a perfect memory)
  memoryless_err     on the same cells, copying the adjacent inward cell (the best a copy/neighbour world without memory does
                     without generating; the corr head's neighbour source)
  incentive_share    sum over recallable cells of max(memoryless_err - recall_err, 0) / the total squared error of the
                     scroll-copy baseline (every cell copied from its source in the previous frame, entering cells from their
                     inward neighbour) over all cells of all used transitions: the share of a copy world's error that perfect
                     memory within L frames removes
Readings, declared before running:
  memory_grows_with_L   recallable_share(15) >= 2 x recallable_share(5)
  memory_incentive_16   incentive_share(15) >= 0.02 (2% of the copy error is removable only with 6-16 frames of memory)
Usage: check_memory.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import tworld as TW  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402
LOOKBACKS = (2, 3, 5, 8, 15, 31, 63)


def source_index():
    """[5, 63] index into [previous frame's 63 map cells, new frame's 63]: each new cell's source under each shift (entering
    cells: their inward neighbour in the new frame); and [5, 63] entering-cell masks"""
    idx, enter = torch.zeros(5, 63, dtype=torch.long), torch.zeros(5, 63, dtype=torch.bool)
    for k, (dr, dc) in enumerate(SHIFTS):
        for r in range(7):
            for c in range(9):
                rs, cs = r + dr, c + dc
                if 0 <= rs < 7 and 0 <= cs < 9:
                    idx[k, r * 9 + c] = rs * 9 + cs
                else:
                    idx[k, r * 9 + c] = 63 + min(max(r - dr, 0), 6) * 9 + min(max(c - dc, 0), 8); enter[k, r * 9 + c] = True
    return idx, enter


def main():
    labels = torch.load(TW.POOLS["rawlong"] / "labels.pt", weights_only=False)
    n = len(labels["terminal"])
    tokens = np.memmap(TW.POOLS["rawlong"] / "tokens.f16", dtype=np.float16, mode="r", shape=(n, 64, 81, 192))
    rows = torch.randperm(n, generator=torch.Generator().manual_seed(20261004))[:1500].sort().values
    IDX, ENTER = source_index()
    cell_r, cell_c = torch.arange(63) // 9, torch.arange(63) % 9
    stats = {L: {"cells": 0, "recallable": 0, "recall_err": 0.0, "memoryless_err": 0.0, "gain": 0.0} for L in LOOKBACKS}
    total_copy_err, transitions, scrolls, enter_cells = 0.0, 0, 0, 0
    frames = torch.arange(64)
    for row in rows.tolist():
        x = torch.from_numpy(np.array(tokens[row])).float()                                       # [64,81,192]
        alive = labels["alive"][row].bool()
        sh = estimate(x[:-1], x[1:])                                                               # [63]: shift into frame t+1
        off = torch.cat([torch.zeros(1, 2, dtype=torch.long), torch.tensor(SHIFTS)[sh].cumsum(0)])   # [64,2] camera offsets
        ok = alive[1:] & alive[:-1]                                                                # transitions t-1 -> t, t = 1..63
        pair = torch.cat([x[:-1, :63], x[1:, :63]], 1)                                             # [63,126,192]
        src = pair.gather(1, IDX[sh][..., None].expand(-1, -1, 192))                               # [63,63,192]
        err = (src - x[1:, :63]).square().sum(-1)                                                  # [63,63]
        hud = (x[1:, 63:] - x[:-1, 63:]).square().sum((-1, -2))
        total_copy_err += float(err[ok].sum() + hud[ok].sum()); transitions += int(ok.sum())
        ent = ENTER[sh] & ok[:, None]                                                              # [63,63]
        scrolls += int(((sh != 0) & ok).sum())
        te, ce = torch.where(ent)                                                                  # transition index, cell
        if len(te) == 0:
            continue
        t = te + 1                                                                                 # target frame
        enter_cells += len(t)
        m_err = err[te, ce]                                                                        # inward-neighbour copy error
        wr, wc = cell_r[ce] + off[t, 0], cell_c[ce] + off[t, 1]
        lr, lc = wr[:, None] - off[None, :, 0], wc[:, None] - off[None, :, 1]                      # [E,64] position in frame s
        inview = (lr >= 0) & (lr < 7) & (lc >= 0) & (lc < 9) & alive[None] & (frames[None] < t[:, None])
        s_last = torch.where(inview, frames[None], -1).max(1).values                               # most recent sighting
        seen = s_last >= 0
        age = t - s_last
        sl = s_last.clamp(min=0)
        rec = x[sl, (lr.gather(1, sl[:, None])[:, 0].clamp(0, 6) * 9 + lc.gather(1, sl[:, None])[:, 0].clamp(0, 8))]
        r_err = (rec - x[t, ce]).square().sum(-1)
        for L in LOOKBACKS:
            st = stats[L]; m = seen & (age <= L)
            st["cells"] += len(t); st["memoryless_err"] += float(m_err.sum())
            st["recallable"] += int(m.sum()); st["recall_err"] += float(r_err[m].sum())
            st["gain"] += float((m_err[m] - r_err[m]).clamp(min=0).sum())
    out = {"windows": len(rows), "transitions": transitions, "scroll_share": scrolls / transitions,
           "entering_cells": enter_cells, "copy_err_per_transition": total_copy_err / transitions}
    for L, st in stats.items():
        k = max(st["recallable"], 1)
        out[f"L{L}"] = {"recallable_share": st["recallable"] / st["cells"], "recall_err": st["recall_err"] / k,
                        "memoryless_err_all_entering": st["memoryless_err"] / st["cells"],
                        "incentive_share": st["gain"] / total_copy_err, "gain_per_transition": st["gain"] / transitions}
        print(json.dumps({f"L{L}": out[f"L{L}"]}), flush=True)
    out["readings"] = {"memory_grows_with_L": out["L15"]["recallable_share"] >= 2 * out["L5"]["recallable_share"],
                       "memory_incentive_16": out["L15"]["incentive_share"] >= 0.02}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
