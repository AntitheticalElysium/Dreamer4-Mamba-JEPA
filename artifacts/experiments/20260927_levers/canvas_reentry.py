"""How often the fcanvas convolution-hold defect can affect a rollout input.

On TEST-seed true-frame windows, map each screen cell to the inferred world
canvas coordinates. Count final-frame cells that were observed, left the view,
then reentered before prediction. The Mamba SSM state is held on absent frames
but its causal convolution is not; only cells with a gap and reentry can expose
that inconsistency at the final input frame.
"""
import json
from pathlib import Path

import torch

import teval as T
from scroll import SHIFTS, estimate

HERE = Path(__file__).parent


def sets_for_window(shifts):
    off_r = off_c = 0
    out = []
    for t in range(len(shifts) + 1):
        if t:
            dr, dc = SHIFTS[int(shifts[t - 1])]
            off_r += dr
            off_c += dc
        out.append({(r + off_r, c + off_c) for r in range(7) for c in range(9)})
    return out


def main():
    meta, train_roots, _ = T.split()
    ix = torch.where(~train_roots)[0]
    cache = T.build_cache("raw", torch.device("cuda"))
    true_frames = torch.cat([cache["ctx"][ix], cache["fut"][ix]], 1).float()
    shifts = estimate(true_frames[:, :-1], true_frames[:, 1:])
    alive = ~meta["future_dead"][ix, 0].cumsum(1).bool()
    totals = {label: {"windows": 0, "any_reentry": 0, "affected_final_cells": 0, "final_cells": 0}
              for label in ("1", "2-4", "5-8", "9-16", "all")}
    for i in range(len(ix)):
        for k in range(T.H):
            if not alive[i, k]:
                continue
            length = 4 if k == 0 else 5
            stop = 4 + k
            start = stop - length
            views = sets_for_window(shifts[i, start:stop - 1])
            final = views[-1]
            affected = 0
            for cell in final:
                presence = [cell in view for view in views]
                seen = gap = reentered = False
                for present in presence:
                    if present:
                        if gap:
                            reentered = True
                        seen = True
                    elif seen:
                        gap = True
                affected += reentered
            for label in ("all", "1" if k == 0 else "2-4" if k < 4 else "5-8" if k < 8 else "9-16"):
                row = totals[label]
                row["windows"] += 1
                row["any_reentry"] += affected > 0
                row["affected_final_cells"] += affected
                row["final_cells"] += 63
    for row in totals.values():
        row["fraction_windows_with_reentry"] = row["any_reentry"] / row["windows"]
        row["fraction_final_cells_affected"] = row["affected_final_cells"] / row["final_cells"]
    out = {"roots": len(ix), "test_walk_seeds": int(meta["seed"][ix].unique().numel()),
           "estimate_reference": "true encoded frames; one-step simulator accuracy 0.9935",
           "by_depth": totals}
    (HERE / "canvas_reentry.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
