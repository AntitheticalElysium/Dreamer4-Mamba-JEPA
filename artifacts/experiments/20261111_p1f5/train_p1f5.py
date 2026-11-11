"""F5, from scratch with the P1-HUD/F4 schedule, corpus and learned gaze."""
import argparse
import copy
import importlib
import os
from pathlib import Path

import torch

from saccade import train as TR

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "20261109_p1_hud_change/run/ckpt_10.pt"
CONTROL_SHA = "967d1a15979049fa332c3832cf05f3021b50b8f7067cf1413648775ffcc68cfa"
WORLD_SEED = 1
C = importlib.import_module("artifacts.experiments.20261103_p1f2.chooser")
FORECAST_SEED = 615001


def recipe(_canonical_cfg=None):
    # The saved recipe is the source of truth, including the data and optimizer.
    ck = torch.load(CONTROL, map_location="cpu", weights_only=False, mmap=True)
    cfg = copy.deepcopy(ck["cfg"])
    cfg["fixation"] = dict(chooser=copy.deepcopy(ck["fixation_controller"]["config"]))
    cfg["heads"].update(change_map=None, forecast={"seed": FORECAST_SEED})
    cfg["losses"]["groups"] = ["G1", "G2", "G3", "G4"]
    cfg["p1"].pop("change_targets_dir", None)
    return cfg


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, "HUD control changed"
    ck = torch.load(CONTROL, map_location="cpu", weights_only=False, mmap=True)
    assert ck["world_seed"] == WORLD_SEED
    assert cfg == recipe(), "F5 recipe differs from the declared changes to HUD"
    assert cfg["fixation"]["chooser"] == C.config(), "learned chooser differs from the control"
    assert TR.run_provenance(cfg)["manifest_sha256"] == ck["provenance"]["manifest_sha256"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    TR.p1(check_recipe(recipe()), WORLD_SEED, args.out, updates=args.updates,
          checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
