"""E11l. Do the TRUE tokens of content-unchanged cells move from step to step? (E11j: exact (hard) copying makes known-cell drift
WORSE, 0.00039 vs 0.00024 per cell per step, so the target itself moves; E11k: light changes do not explain it; E11e: unchanged
distant cells change ~4.5 in squared token distance between consecutive true frames.) No world: true tokens only.

Diagnosis futures, five samples, steps k >= 1 (frame k-1 -> k) where all five samples are alive and scroll the same way. Cells as
scrolldrift.py: each map cell at k matched to the same world cell at k-1, observable at the root, no mob in any sample at k-1 or
k, same drawn content (tile class, sample 0) at k-1 and k. Per cell:
  det_change   |mu_k - mu_{k-1}|^2 - (s2_k + s2_{k-1}) / 5, / V: the deterministic (sample-mean) change of the true token,
               unbiased for the change of the conditional mean (the noise of each mean removed)
Reported: pooled per cell, scroll vs no-scroll steps, day vs night (light 0.5), and by Chebyshev ring around the player.
Reading, declared before running:
  true_target_moves  pooled det_change on no-scroll steps >= 0.5x the soft worlds' mean pooled no-scroll increment (E11h:
                     teacher s7/s8, suffix s7/s8: 0.000243, 0.000303, 0.000280, 0.000275; mean 0.000275)
Scroll-step split (added after the first run, reported only): the four move targets after a move are the tile the player LEFT
(behind: its drawn pixels change from the player sprite to terrain, so "content unchanged" in the visible state is not
"pixels unchanged"), the next tile AHEAD, and the two SIDES; only ahead and sides are pure context changes.
Usage: truechange.py -> evals/truechange.json
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
SOFT_STILL = 0.000275


@torch.no_grad()
def main():
    from scroll import SHIFTS, estimate
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(torch.device("cpu"))
    cache = T.build_cache("raw", torch.device("cpu"))
    R = len(meta["seed"])
    root = cache["ctx"][:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    true_shift = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        true_shift.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(true_shift, 1)
    same_shift = (true_shift == true_shift[:, :1]).all(1)                                                   # [R,16]
    true_off = DA.offsets(true_shift)
    vis = meta["future_visible"].float()
    mobs = (vis[..., 1071:1512].reshape(R, S, H, 63, 7).sum(-1) > 0).any(1)
    tiles0 = vis[:, 0, :, :1071].reshape(R, H, 63, 17).argmax(-1)
    light = vis[:, 0, :, 1521]
    rr = torch.arange(63) // 9; cc = torch.arange(63) % 9
    ring = torch.maximum((rr - 3).abs(), (cc - 4).abs())
    acc = {}
    def add(key, val, m):
        t = acc.setdefault(key, [0.0, 0]); t[0] += float(val[m].sum()); t[1] += int(m.sum())
    for i in range(0, R, 16):
        b = min(16, R - i)
        x = fut5[i:i + b].float(); mu = x.mean(1)                                                           # [b,16,81,192]
        s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)                                               # [b,16,81]
        for j in range(b):
            r = i + j
            for k in range(1, H):
                if not (alive[r, k] and same_shift[r, k]):
                    continue
                sidx = int(true_shift[r, 0, k]); dr, dc = SHIFTS[sidx]
                pr, pc = rr + dr, cc + dc
                inside = (pr >= 0) & (pr < 7) & (pc >= 0) & (pc < 9)
                prev_idx = pr.clamp(0, 6) * 9 + pc.clamp(0, 8)
                wr, wc = rr + int(true_off[r, 0, k, 0]), cc + int(true_off[r, 0, k, 1])
                observable = (wr >= 0) & (wr < 7) & (wc >= 0) & (wc < 9)
                keep = inside & observable & ~mobs[r, k] & ~mobs[r, k - 1][prev_idx] & (tiles0[r, k] == tiles0[r, k - 1][prev_idx])
                det = (((mu[j, k, :63] - mu[j, k - 1][prev_idx]) ** 2).sum(-1) - (s2[j, k, :63] + s2[j, k - 1][prev_idx]) / S) / V
                kind = "scroll" if sidx != 0 else "still"
                tod = "night" if float(light[r, k]) < 0.5 else "day"
                add((kind, "all"), det, keep); add((kind, tod), det, keep)
                for g, gm in (("targets", torch.isin(torch.arange(63), torch.tensor((22, 30, 32, 40)))), ("ring2", ring == 2),
                              ("ring3plus", ring >= 3)):
                    add((kind, g), det, keep & gm)
                if sidx != 0:
                    cell = lambda a, b_: torch.arange(63) == a * 9 + b_
                    for g, gm in (("behind", cell(3 - dr, 4 - dc)), ("ahead", cell(3 + dr, 4 + dc)),
                                  ("sides", cell(3 + dc, 4 + dr) | cell(3 - dc, 4 - dr))):
                        add((kind, g), det, keep & gm)
    pooled = {f"{k}_{g}": {"per_cell": v[0] / max(v[1], 1), "cells": v[1]} for (k, g), v in acc.items()}
    res = {"V": V, "det_change": pooled, "soft_world_still_increment_mean": SOFT_STILL}
    res["readings"] = {"still_det_over_soft_increment": pooled["still_all"]["per_cell"] / SOFT_STILL,
                       "true_target_moves": pooled["still_all"]["per_cell"] >= 0.5 * SOFT_STILL}
    (HERE / "evals" / "truechange.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: (round(v["per_cell"], 7), v["cells"]) for k, v in pooled.items()} | res["readings"], indent=1), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
