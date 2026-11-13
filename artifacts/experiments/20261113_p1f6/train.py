"""P1-F6: F5's recipe with the change-form token forecast in place of TSP's forecast mode (PLAN.md). From scratch,
F5's schedule; resumable; --updates stops early (2,500 = checkpoint 1, 5,000 = checkpoint 2)."""
import argparse
import importlib
import os
from pathlib import Path

from saccade import train as TR

HERE = Path(__file__).resolve().parent
F5 = importlib.import_module("artifacts.experiments.20261111_p1f5.train_p1f5")
WORLD_SEED, C = F5.WORLD_SEED, F5.C
TOKENS = {"init_seed": 615002, "position_dim": 32, "belief_dim": 128, "hidden": 256}   # the gate's TokenHead (gate.py)


def recipe(_canonical_cfg=None):
    cfg = F5.check_recipe(F5.recipe())
    cfg["heads"]["forecast"]["tokens"] = dict(TOKENS)
    return cfg


def check_recipe(cfg):
    assert cfg == recipe(), "F6 recipe differs from F5 plus the token forecast"
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
