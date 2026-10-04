"""How much could long-context memory help? The recall ceiling for cells that scroll into view.

Raw long pool (64-frame TRAIN windows, layer-normed patch tokens). Per window: the view scroll between
consecutive frames from the tokens (scroll.estimate, 0.994 accurate against the visible rule), accumulated into
a camera offset, so screen cell (r, c) at frame t sits at world cell (r, c) + O_t. At every scroll, the entering
cells are the map cells whose world cell was not on screen at t-1. For each entering cell:
  seen_within_N   its world cell was on screen at some frame in [t-N, t-2], N = 4, 8, 16, 32, 63 (the last
                  N-1 frames before the previous one): the most that an N-frame memory could recall
  same_token      for cells seen before, whether the entering token matches the token last seen there
                  (squared distance below the median distance between the same world cell on consecutive
                  frames when nothing scrolled) -- content unchanged since it left view
Also: scroll rate, and the share of ALL map-cell predictions that are entering cells.
Usage: memceil.py [windows=2000] -> memceil.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260927_levers"))
from scroll import SHIFTS, estimate  # noqa: E402

POOL = ROOT / "artifacts/eda/levers_mamba_long_pools_v1/raw"
NS = (4, 8, 16, 32, 63)


def main():
    n_windows = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    shape = (19789, 64, 81, 192)
    tokens = np.memmap(POOL / "tokens.f16", mode="r", dtype=np.float16, shape=shape)
    rows = np.random.default_rng(0).choice(shape[0], n_windows, replace=False)
    shifts_t = torch.tensor(SHIFTS)
    counts = {n: 0 for n in NS}; same = {n: 0 for n in NS}
    entering = scrolls = steps = 0
    stable_d, recall_d = [], []
    for i, row in enumerate(rows):
        x = torch.from_numpy(np.asarray(tokens[row])).float()                         # [64,81,192]
        s = estimate(x[:-1], x[1:])                                                   # [63]
        offset = torch.cat([torch.zeros(1, 2, dtype=torch.long), shifts_t[s].cumsum(0)])  # [64,2]
        grid = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(9), indexing="ij"), -1).view(63, 2)
        world = grid[None] + offset[:, None]                                          # [64,63,2]
        key = world[..., 0] * 1000 + world[..., 1]                                    # [64,63]
        for t in range(1, 64):
            steps += 1
            if s[t - 1] == 0:
                a, b = x[t - 1, :63], x[t, :63]                                       # nothing scrolled
                stable_d.append(((a - b) ** 2).sum(-1)[:8])
                continue
            scrolls += 1
            prev = set(key[t - 1].tolist())
            for j in range(63):
                k = int(key[t, j])
                if k in prev:
                    continue
                entering += 1
                last = None
                for u in range(t - 2, -1, -1):
                    hit = (key[u] == k).nonzero()
                    if len(hit):
                        last = (u, int(hit[0]))
                        break
                if last is None:
                    continue
                gap = t - last[0]
                d = float(((x[t, j] - x[last[0], last[1]]) ** 2).sum())
                recall_d.append((gap, d))
                for n in NS:
                    if gap <= n:
                        counts[n] += 1
        if i % 500 == 0:
            print(json.dumps({"windows": i + 1, "entering": entering}), flush=True)
    thr = float(torch.cat(stable_d).median())
    for gap, d in recall_d:
        for n in NS:
            if gap <= n and d <= thr:
                same[n] += 1
    out = {"windows": n_windows, "steps": steps, "scroll_rate": scrolls / steps,
           "entering_cells": entering, "entering_share_of_map_predictions": entering / (steps * 63),
           "stable_token_distance_median": thr,
           "seen_within": {n: counts[n] / entering for n in NS},
           "seen_within_and_unchanged": {n: same[n] / entering for n in NS}}
    (HERE / "memceil.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
