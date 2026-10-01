"""E11e. Are our patch tokens contextual next to the player? (E11d: known tiles beside the player drift 5.5-9x faster at the first
imagined step; conjecture: a neighbour's token changes when the player sprite changes even if the tile did not, so a copy head
must regenerate it there.) A property of the frozen encoder only: TRUE tokens, no world.

Diagnosis futures, all five samples, consecutive true frames (root -> step 1 -> ... -> step 16) with no view scroll
(scroll.estimate) and the sample still alive. For each map cell whose drawn content is unchanged between the two frames
(same tile class and same mob channels in the visible state), the token change |x_t - x_{t-1}|^2 (layer-normed tokens).
Split by cell group (the four move targets around the player; Chebyshev ring 3+) and by whether the player's facing changed
(visible facing one-hot) between the two frames. Also the player tile itself.
Reading, declared before running:
  contextual_neighbour  for content-unchanged move targets, the mean token change when the facing changed is >= 3x the same
                        cells' change when it did not, AND >= 3x ring 3+'s change when the facing changed
Usage: ctxtok.py -> evals/ctxtok.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import stochdiag as SD  # noqa: E402
T = SD.T
H, S = SD.H, SD.S
TARGETS = (22, 30, 32, 40)


@torch.no_grad()
def main():
    from scroll import estimate
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(torch.device("cpu"))
    cache = T.build_cache("raw", torch.device("cpu"))
    R = len(meta["seed"])
    root = cache["ctx"][:, -1]
    vis = torch.cat([meta["root_visible"][:, None, None].expand(R, S, 1, 1534), meta["future_visible"]], 2).float()  # [R,5,17,..]
    dead = meta["future_dead"].cumsum(2).bool()                                                           # [R,5,16]
    tiles = vis[..., :1071].reshape(R, S, H + 1, 63, 17).argmax(-1)                                       # [R,5,17,63]
    mobs = vis[..., 1071:1071 + 441].reshape(R, S, H + 1, 63, 7)
    facing = vis[..., 1516:1520].argmax(-1)                                                               # [R,5,17]
    ring = torch.maximum((torch.arange(63) // 9 - 3).abs(), (torch.arange(63) % 9 - 4).abs())
    groups = {"targets": torch.isin(torch.arange(63), torch.tensor(TARGETS)), "ring3plus": ring >= 3}
    acc = {(g, f): [0.0, 0] for g in (*groups, "player") for f in ("turned", "same")}
    for s in range(S):
        for i in range(0, R, 64):
            b = min(64, R - i)
            prev = torch.cat([root[i:i + b, None], fut5[i:i + b, s, :-1]], 1).float()                   # [b,16,81,192]
            cur = fut5[i:i + b, s].float()
            shift = estimate(prev, cur)                                                                   # [b,16]
            d = ((cur - prev) ** 2).sum(-1)                                                               # [b,16,81]
            same_content = (tiles[i:i + b, s, 1:] == tiles[i:i + b, s, :-1]) & \
                           (mobs[i:i + b, s, 1:] == mobs[i:i + b, s, :-1]).all(-1)                       # [b,16,63]
            turned = facing[i:i + b, s, 1:] != facing[i:i + b, s, :-1]                                   # [b,16]
            ok = (shift == 0) & ~dead[i:i + b, s]
            for f, fm in (("turned", turned), ("same", ~turned)):
                step = ok & fm
                for g, gm in groups.items():
                    m = step[..., None] & same_content & gm
                    acc[(g, f)][0] += float(d[..., :63][m].sum()); acc[(g, f)][1] += int(m.sum())
                acc[("player", f)][0] += float(d[..., 31][step].sum()); acc[("player", f)][1] += int(step.sum())
    mean = {f"{g}_{f}": acc[(g, f)][0] / max(acc[(g, f)][1], 1) for g, f in acc}
    counts = {f"{g}_{f}": acc[(g, f)][1] for g, f in acc}
    res = {"mean_token_change": mean, "counts": counts,
           "ratios": {"targets_turned_over_targets_same": mean["targets_turned"] / max(mean["targets_same"], 1e-9),
                      "targets_turned_over_ring3plus_turned": mean["targets_turned"] / max(mean["ring3plus_turned"], 1e-9)}}
    res["readings"] = {"contextual_neighbour": res["ratios"]["targets_turned_over_targets_same"] >= 3 and
                       res["ratios"]["targets_turned_over_ring3plus_turned"] >= 3}
    (HERE / "evals" / "ctxtok.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
