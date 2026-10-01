"""E11f. Why do known tiles beside the player drift fastest? (E11d: 5.5-9x faster per cell than distant tiles at the first imagined
step. E11e refuted the contextual-token explanation: content-unchanged neighbour tokens change LESS when the player turns.)
Conjecture tested here: the FACED tile is where actions take effect (DO collects / attacks it, PLACE_* puts a block there), so the
world regenerates it instead of copying it, even when nothing changes there.

Depth 1 only (one imagined step from the 4 true context frames, the factual action), diagnosis futures, world on CPU or GPU.
Cells kept: steps with no view scroll in any of the five samples; map cells whose drawn content (tile class and mob channels)
is unchanged from the root to step 1 in ALL five samples. Per cell: stochdiag's excess (|g - mu|^2 - s^2/5, / V).
Groups: the faced tile (the neighbour in the root facing direction); the other three move targets; the four diagonals; ring 2;
ring 3+. Action classes (factual action at step 1): move (1-4), DO (5), place (7-10), noop (0), sleep (6), other (11-16).
Reading, declared before running:
  consequence_site  on the faced tile, per-cell excess under DO + place >= 3x under move + noop, AND >= 3x the other three
                    targets' excess under DO + place
Usage: targetdrift.py <world.pt> ... -> evals/targetdrift_<name>.json
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
TARGET = {0: 30, 1: 32, 2: 22, 3: 40}                     # facing one-hot index (direction - 1: left, right, up, down) -> tile
ACTIONS = {"move": (1, 2, 3, 4), "DO": (5,), "place": (7, 8, 9, 10), "noop": (0,), "sleep": (6,), "other": tuple(range(11, 17))}


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
    rv = meta["root_visible"].float(); v1 = meta["future_visible"][:, :, 0].float()                      # [R,1534], [R,5,1534]
    tiles0 = rv[:, :1071].reshape(R, 63, 17).argmax(-1); tiles1 = v1[..., :1071].reshape(R, S, 63, 17).argmax(-1)
    mobs0 = rv[:, 1071:1512].reshape(R, 63, 7); mobs1 = v1[..., 1071:1512].reshape(R, S, 63, 7)
    unchanged = ((tiles1 == tiles0[:, None]) & (mobs1 == mobs0[:, None]).all(-1)).all(1)               # [R,63]
    shift = torch.stack([estimate(root.float(), fut5[:, s, 0].float()) for s in range(S)], 1)          # [R,5]
    no_scroll = (shift == 0).all(1) & ~meta["future_dead"][:, :, 0].any(1)
    facing = rv[:, 1516:1520].argmax(-1)
    faced = torch.tensor([TARGET[int(f)] for f in facing])                                             # [R]
    cell = torch.arange(63)
    ring = torch.maximum((cell // 9 - 3).abs(), (cell % 9 - 4).abs())
    targets = torch.isin(cell, torch.tensor(tuple(TARGET.values())))
    group_of = lambda r: {"faced": cell == faced[r], "other_targets": targets & (cell != faced[r]),
                          "diagonals": (ring == 1) & ~targets, "ring2": ring == 2, "ring3plus": ring >= 3}
    act = fa[:, 0]
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        g = torch.cat([T.step(world, ctx[i:i + 16].float(), torch.cat([ca[i:i + 16], fa[i:i + 16, :1]], 1), device, config)
                       for i in range(0, R, 16)])                                                       # [R,81,192]
        x = fut5[:, :, 0].float()
        mu = x.mean(1)
        s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
        ex = (((g - mu) ** 2).sum(-1) - s2 / S)[:, :63] / V                                             # [R,63]
        acc = {}
        for r in torch.where(no_scroll)[0].tolist():
            a = int(act[r])
            aclass = next(k for k, v in ACTIONS.items() if a in v)
            for gname, gm in group_of(r).items():
                m = gm & unchanged[r]
                key = (gname, aclass)
                t = acc.setdefault(key, [0.0, 0])
                t[0] += float(ex[r][m].sum()); t[1] += int(m.sum())
        table = {f"{gname}|{ac}": {"per_cell": v[0] / max(v[1], 1), "cells": v[1]} for (gname, ac), v in acc.items()}
        pc = lambda gname, classes: (sum(acc.get((gname, c), [0, 0])[0] for c in classes) /
                                     max(sum(acc.get((gname, c), [0, 0])[1] for c in classes), 1))
        act_on, passive = ("DO", "place"), ("move", "noop")
        res = {"world": name, "no_scroll_roots": int(no_scroll.sum()), "table": table,
               "summary": {"faced_act": pc("faced", act_on), "faced_passive": pc("faced", passive),
                           "other_targets_act": pc("other_targets", act_on), "other_targets_passive": pc("other_targets", passive),
                           "ring3plus_all": pc("ring3plus", tuple(ACTIONS))}}
        s_ = res["summary"]
        res["readings"] = {"consequence_site": s_["faced_act"] >= 3 * s_["faced_passive"] and s_["faced_act"] >= 3 * s_["other_targets_act"]}
        (out_dir / f"targetdrift_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "no_scroll_roots": res["no_scroll_roots"], **{k: round(v, 6) for k, v in s_.items()},
                          **res["readings"]}), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
