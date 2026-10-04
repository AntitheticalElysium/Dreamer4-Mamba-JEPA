"""E11d. Does known content drift faster next to the player? (E11c: the move decision flips when the imagined current frame's TARGET
tile -- always one of the player's four neighbours -- has drifted, and 69-87% of those tiles were observable at the root. Tokens are
contextual ViT tokens: a neighbour's token changes when the player sprite turns even if the tile did not, so the copy head must
regenerate rather than copy there.)

Same rollouts as driftanat.py. Cells counted: map cells whose world cell was inside the root view (observable), with no mob in
any of the five samples at that depth, at depths where the imagined offset is still right and all samples are alive and agree.
Grouped by Chebyshev ring around the player (row 3, col 4): ring 1 = the 8 surrounding cells (the four move targets among them),
ring 2, ring 3+; plus the four move targets alone. Per group and depth: stochdiag's excess per cell (|g - mu|^2 - s^2/5, / V).
Reading, declared before running:
  adjacent_drift  per-cell excess of the four move targets >= 2x that of ring 3+ at depth 8
Usage: adjdrift.py <world.pt> ... -> evals/adjdrift_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import driftanat as DA  # noqa: E402
SD, T = DA.SD, DA.T
H, S = SD.H, SD.S
TARGETS = (22, 30, 32, 40)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    true_shift = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        true_shift.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_off = DA.offsets(torch.stack(true_shift, 1))
    agree = (true_off == true_off[:, :1]).all(-1).all(1)
    vis = meta["future_visible"].float()
    mobs = (vis[..., 1071:1071 + 441].reshape(R, S, H, 7, 9, 7).sum(-1) > 0).any(1).flatten(2)    # [R,16,63]
    rr = torch.arange(63) // 9; cc = torch.arange(63) % 9
    ring = torch.maximum((rr - 3).abs(), (cc - 4).abs())                                         # [63]
    groups = {"targets": torch.isin(torch.arange(63), torch.tensor(TARGETS)), "ring1": ring == 1, "ring2": ring == 2,
              "ring3plus": ring >= 3}
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        batch = 16 if world.backbone_kind == "full" else 4
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        for i in range(0, R, batch):
            b = min(batch, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
            for k in range(H):
                w = 4 if k == 0 else 5
                g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1), device, config)
                gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fk[:, k])
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_off = DA.offsets(torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)]))
        aligned = (img_off == true_off[:, 0]).all(-1) & alive & agree                              # [R,16]
        acc = {g: {"excess": torch.zeros(H), "cells": torch.zeros(H)} for g in groups}
        for i in range(0, R, 16):
            b = min(16, R - i)
            x = fut5[i:i + b].float(); mu = x.mean(1)
            s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
            ex = (((gen[i:i + b].float() - mu) ** 2).sum(-1) - s2 / S)[..., :63]                  # [b,16,63]
            off = true_off[i:i + b, 0]
            wr, wc = rr + off[..., 0:1], cc + off[..., 1:2]
            observable = (wr >= 0) & (wr < 7) & (wc >= 0) & (wc < 9)
            base = observable & ~mobs[i:i + b] & aligned[i:i + b, :, None]
            for gname, gm in groups.items():
                m = base & gm
                acc[gname]["excess"] += (ex * m).sum((0, 2)); acc[gname]["cells"] += m.sum((0, 2)).float()
        per_cell = {g: (acc[g]["excess"] / acc[g]["cells"].clamp(min=1) / V).tolist() for g in groups}
        res = {"world": name, "per_cell_excess": per_cell, "cells": {g: acc[g]["cells"].tolist() for g in groups},
               "readings": {"targets_over_ring3plus_8": per_cell["targets"][7] / max(per_cell["ring3plus"][7], 1e-12),
                            "adjacent_drift": per_cell["targets"][7] >= 2 * per_cell["ring3plus"][7]}}
        (out_dir / f"adjdrift_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, **res["readings"],
                          "d1": {g: round(v[0], 6) for g, v in per_cell.items()},
                          "d8": {g: round(v[7], 6) for g, v in per_cell.items()}}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
