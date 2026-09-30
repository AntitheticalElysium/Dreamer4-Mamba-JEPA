"""Stage 3 prerequisite. The aleatoric ceiling of teval's imagined-fact metrics, from the simulator's own spread.

The futures roll the SAME factual 16 actions from the FULL root state under the walk's key (sample 0, the label every
world is scored on) and 4 other keys (samples 1-4). For each teval fact, on teval's own rows (test-seed roots alive at
depth k, labels from sample 0's visible state) and definitions (teval.facts_of, NEAR, INTERIOR, EDGE):
  ceiling     the prediction from samples 1-4 alone: mean zombie presence per cell (AUC), per-cell modal tile
              class (accuracy), mean health / food (R^2), modal facing (accuracy). A 4-sample estimate of the
              conditional law given the full state and the actions, so it slightly UNDER-states the true ceiling
              for a predictor that knew that law; a world that sees only the observation cannot exceed it by
              more than that.
  one_key     sample 1 alone as the prediction (what one exact simulator rollout scores)
  copy_root   the root's own facts (teval's copy_root)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
import teval as T  # noqa: E402
from compound import auc  # noqa: E402


def score(pred, f):
    """pred: dict tile [n,63] long, zombie [n,63] float, hud [n,4], facing [n] -> teval.Probes.read's metrics."""
    idx = {c: T.MAP.index(c) for c in T.MAP}
    correct = (pred["tile"] == f["tile"]).float()
    region = lambda cells: float(correct[:, [idx[c] for c in cells]].mean())
    near = [idx[c] for c in T.NEAR]
    r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9))
    return {"tile_near": region(T.NEAR), "tile_interior": region(T.INTERIOR), "tile_edge": region(T.EDGE),
            "zombie_auc": auc(pred["zombie"], f["zombie"]),
            "zombie_near_auc": auc(pred["zombie"][:, near], f["zombie"][:, near]),
            "health_r2": r2(pred["hud"][:, 0], f["hud"][:, 0]), "food_r2": r2(pred["hud"][:, 1], f["hud"][:, 1]),
            "facing_acc": float((pred["facing"] == f["facing"]).float().mean())}


def main():
    meta, train_roots, _ = T.split()
    test = ~train_roots
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    fv = meta["future_visible"]                                                        # [R,5,16,1534]
    root = T.facts_of(meta["root_visible"])
    out = {}
    for k in (1, 2, 4, 8, 16):
        m = alive[:, k - 1] & test
        label = T.facts_of(fv[m, 0, k - 1])
        others = T.facts_of(fv[m, 1:5, k - 1])                                         # [n,4,...]
        ceiling = {"tile": others["tile"].mode(1).values, "zombie": others["zombie"].mean(1),
                   "hud": others["hud"].mean(1), "facing": others["facing"].mode(1).values}
        one = {key: v[:, 0] for key, v in others.items()}
        copy = {key: v[m] for key, v in root.items()}
        out[k] = {"n": int(m.sum()), "ceiling": score(ceiling, label), "one_key": score(one, label),
                  "copy_root": score(copy, label)}
        print(k, json.dumps({n: {a: round(b, 3) for a, b in v.items()} for n, v in out[k].items() if n != "n"}), flush=True)
    (HERE / "ceiling.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
