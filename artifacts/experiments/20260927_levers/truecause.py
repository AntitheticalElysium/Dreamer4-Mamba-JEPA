"""E11m (exploratory, reported only). What moves the true tokens of content-unchanged cells? (E11l: ~0.00026 / V per cell per step,
about the worlds' drift rate; not specific to the player's neighbours.) True tokens only, no world.

Diagnosis futures, no-scroll steps k >= 1 where all five samples are alive and none scrolled. Per (root, step): the mean over
content-unchanged cells (as truechange.py: observable at the root, no mob at k-1 or k, same tile class) of the noise-corrected
change of the sample-mean token, / V. Tabulated against features of that step (sample 0's visible state):
  light change |dL|; night (L < 0.5); HUD change (health / food / drink / energy digits); facing change; number of OTHER map
  cells whose tile class changed; number of map cells with a mob at k-1 or k (any sample); sleeping.
Usage: truecause.py -> evals/truecause.json
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


@torch.no_grad()
def main():
    from scroll import estimate
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(torch.device("cpu"))
    cache = T.build_cache("raw", torch.device("cpu"))
    R = len(meta["seed"])
    root = cache["ctx"][:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    shifts = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        shifts.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    still = (torch.stack(shifts, 1) == 0).all(1)                                                            # [R,16]
    vis = meta["future_visible"].float()
    mobs_any = (vis[..., 1071:1512].reshape(R, S, H, 63, 7).sum(-1) > 0).any(1)                              # [R,16,63]
    v0 = vis[:, 0]
    tiles = v0[..., :1071].reshape(R, H, 63, 17).argmax(-1)
    hud, facing, sleep, light = v0[..., 1512:1516], v0[..., 1516:1520].argmax(-1), v0[..., 1520], v0[..., 1521]
    rows = []
    for i in range(0, R, 16):
        b = min(16, R - i)
        x = fut5[i:i + b].float(); mu = x.mean(1)
        s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
        for j in range(b):
            r = i + j
            for k in range(1, H):
                if not (alive[r, k] and still[r, k]):
                    continue
                unchanged = (tiles[r, k] == tiles[r, k - 1]) & ~mobs_any[r, k] & ~mobs_any[r, k - 1]
                if not unchanged.any():
                    continue
                det = (((mu[j, k, :63] - mu[j, k - 1, :63]) ** 2).sum(-1) - (s2[j, k, :63] + s2[j, k - 1, :63]) / S) / V
                rows.append({"det": float(det[unchanged].mean()), "dlight": abs(float(light[r, k] - light[r, k - 1])),
                             "night": float(light[r, k]) < 0.5, "hud": bool((hud[r, k] != hud[r, k - 1]).any()),
                             "turn": bool(facing[r, k] != facing[r, k - 1]),
                             "other_changed": int((tiles[r, k] != tiles[r, k - 1]).sum()),
                             "mob_cells": int((mobs_any[r, k] | mobs_any[r, k - 1]).sum()), "sleep": bool(sleep[r, k] > 0.5)})
    det = torch.tensor([r["det"] for r in rows])
    def split(name, cond):
        m = torch.tensor([cond(r) for r in rows])
        return {"true": {"mean": float(det[m].mean()) if m.any() else None, "n": int(m.sum())},
                "false": {"mean": float(det[~m].mean()) if (~m).any() else None, "n": int((~m).sum())}}
    res = {"steps": len(rows), "overall_mean": float(det.mean()),
           "splits": {"night": split("night", lambda r: r["night"]), "hud_change": split("hud", lambda r: r["hud"]),
                      "turn": split("turn", lambda r: r["turn"]), "sleep": split("sleep", lambda r: r["sleep"]),
                      "dlight_ge_0.005": split("dl", lambda r: r["dlight"] >= 0.005),
                      "other_tile_changed": split("oc", lambda r: r["other_changed"] > 0),
                      "mob_cells_ge_2": split("mc", lambda r: r["mob_cells"] >= 2)}}
    (HERE / "evals" / "truecause.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
