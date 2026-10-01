"""E11g. Is the fast drift beside the player the tile the player was standing on? (E11d: known tiles beside the player drift
5.5-9x faster at the first imagined step; E11e refuted contextual tokens; E11f found no general faced-tile effect on steps with
no scroll, so the drift is on scroll steps.) The rendered frame draws the PLAYER on its own tile, so that tile's terrain is hidden.
After a move, the tile behind the player is the one it left: unseen in the root frame, and unseen in every context frame if the
player had not moved during the context.

Depth 1 only (one imagined step from the 4 true context frames, the factual action), diagnosis futures. Steps kept: the factual
action is a move and all five samples scrolled the same way (a move that succeeded, no randomness in position). Cells: the four
neighbours of the player's new position, and ring 3+, with no mob in any sample. Per cell: stochdiag's excess (|g - mu|^2 -
s^2/5, / V); here the five samples share the terrain, so the floor is ~0 for terrain cells.
  behind_hidden   the tile the player left, which no context frame drew (no scroll in the context: the player stood on it)
  behind_seen     the same tile, drawn in at least one context frame (the player moved onto it during the context)
  ahead           the next tile in the move direction (two tiles ahead at the root, drawn)
  sides           the two tiles beside the new position (diagonal neighbours at the root, drawn)
  ring3plus       distant map cells
Reading, declared before running:
  occlusion_driven  behind_hidden per-cell excess >= 3x ahead AND >= 3x behind_seen
Usage: occlusion.py <world.pt> ... -> evals/occlusion_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import stochdiag as SD  # noqa: E402
T = SD.T
S = SD.S
SHIFT_OF_MOVE = {1: 3, 2: 4, 3: 1, 4: 2}          # action (left, right, up, down) -> scroll.SHIFTS index (scroll.MOVE_SHIFT)
STEP = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from dataclasses import replace
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    if device.type == "cpu":
        config = replace(config, runtime=replace(config.runtime, device="cpu"))
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", torch.device("cpu"))
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    shift = torch.stack([estimate(root.float(), fut5[:, s, 0].float()) for s in range(S)], 1)            # [R,5]
    ctx_shift = torch.stack([estimate(ctx[:, j].float(), ctx[:, j + 1].float()) for j in range(3)], 1)   # [R,3]
    a0 = fa[:, 0]
    move = (a0 >= 1) & (a0 <= 4)
    want = torch.tensor([SHIFT_OF_MOVE.get(int(a), -1) for a in a0])
    kept = move & (shift == want[:, None]).all(1) & ~meta["future_dead"][:, :, 0].any(1)
    hidden = (ctx_shift == 0).all(1)                                                                     # player stood still
    mobs = (meta["future_visible"][:, :, 0, 1071:1512].float().reshape(R, S, 63, 7).sum(-1) > 0).any(1)   # [R,63]
    ring = torch.maximum((torch.arange(63) // 9 - 3).abs(), (torch.arange(63) % 9 - 4).abs())
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        g = torch.cat([T.step(world, ctx[i:i + 16].float(), torch.cat([ca[i:i + 16], fa[i:i + 16, :1]], 1), device, config)
                       for i in range(0, R, 16)])
        x = fut5[:, :, 0].float(); mu = x.mean(1)
        s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
        ex = (((g - mu) ** 2).sum(-1) - s2 / S)[:, :63] / V                                               # [R,63]
        acc = {k: [0.0, 0] for k in ("behind_hidden", "behind_seen", "ahead", "sides", "ring3plus")}
        for r in torch.where(kept)[0].tolist():
            dr, dc = STEP[int(a0[r])]
            cell = lambda rr, cc: rr * 9 + cc
            behind, ahead = cell(3 - dr, 4 - dc), cell(3 + dr, 4 + dc)
            sides = [cell(3 + dc, 4 + dr), cell(3 - dc, 4 - dr)]                                         # perpendicular
            groups = {("behind_hidden" if hidden[r] else "behind_seen"): [behind], "ahead": [ahead], "sides": sides,
                      "ring3plus": torch.where(ring >= 3)[0].tolist()}
            for gname, cells in groups.items():
                for c in cells:
                    if mobs[r, c]:
                        continue
                    acc[gname][0] += float(ex[r, c]); acc[gname][1] += 1
        per_cell = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
        res = {"world": name, "moves_kept": int(kept.sum()), "context_still": int((kept & hidden).sum()),
               "per_cell_excess": per_cell, "cells": {k: v[1] for k, v in acc.items()}}
        res["readings"] = {"behind_hidden_over_ahead": per_cell["behind_hidden"] / max(per_cell["ahead"], 1e-12),
                           "behind_hidden_over_behind_seen": per_cell["behind_hidden"] / max(per_cell["behind_seen"], 1e-12),
                           "occlusion_driven": per_cell["behind_hidden"] >= 3 * per_cell["ahead"] and
                           per_cell["behind_hidden"] >= 3 * per_cell["behind_seen"]}
        (out_dir / f"occlusion_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "moves_kept": res["moves_kept"], "context_still": res["context_still"],
                          **{k: round(v, 6) for k, v in per_cell.items()}, **res["readings"]}), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
