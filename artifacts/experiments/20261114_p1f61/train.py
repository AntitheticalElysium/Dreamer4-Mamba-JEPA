"""P1-F6.1: F5's recipe without any visual forecast (F6's token forecaster removed, TSP's forecast mode not restored)
and with Dreamer 4's multi-distance reward forecast, distances 1-8 (PLAN.md). From scratch, F5's schedule; resumable;
--updates stops early (2,500 = checkpoint 1)."""
import argparse
import importlib
import os
from pathlib import Path

from saccade import train as TR

HERE = Path(__file__).resolve().parent
F5 = importlib.import_module("artifacts.experiments.20261111_p1f5.train_p1f5")
WORLD_SEED, C = F5.WORLD_SEED, F5.C
AHEAD = {"tsp": False, "reward_distances": 8, "reward_init_seed": 615003,   # Dreamer 4 section 3.3: L = 8
         "reward_ahead_reduction": "mean"}   # L1 + mean(L2..L8) (Codex review; LOSSES principle 1), not Eq. 9's raw sum


def recipe(_canonical_cfg=None):
    cfg = F5.check_recipe(F5.recipe())
    cfg["heads"]["forecast"].update(AHEAD)
    return cfg


def check_recipe(cfg):
    assert cfg == recipe(), "F6.1 recipe differs from F5 plus the declared forecast changes"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    TR.p1(check_recipe(recipe()), WORLD_SEED, args.out, updates=args.updates, checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
