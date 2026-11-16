"""P1-F7 (user 11-16; the run the user called P1.5): F6.1's recipe plus the spatial history, route by role, the
harmoniser at its fixed point, the gaze forecast and Mamba-3 (PLAN.md). From scratch, F5's schedule; resumable;
--updates stops early, --steps fixes the chunk length (the throughput checks)."""
import argparse
import importlib
import os
from pathlib import Path

from saccade import train as TR

HERE = Path(__file__).resolve().parent
F61 = importlib.import_module("artifacts.experiments.20261114_p1f61.train")
WORLD_SEED = F61.WORLD_SEED
READ = None                                                              # "linear" or "mlp": test 3's verdict (PLAN.md)
RECORD = {"basis": "s4d_lin", "modes": 32, "dt": 1 / 32, "init_seed": 711601}   # tests 2-3 (20261116_temporal_memory)


def recipe(route_by_role=True):
    cfg = F61.check_recipe(F61.recipe())
    cfg["record"] = dict(RECORD, read=READ)
    cfg["losses"].update(route_by_role=route_by_role, harmoniser_fixed_point_rate=0.01,
                         groups=["G1", "G2", "G3", "G4", "G5"])
    cfg["heads"]["gaze_forecast"] = {"init_seed": 711602}
    cfg["belief"].update(mixer="mamba3", mamba3={"chunk_size": 64})     # mamba_ssm's recommended SISO chunk
    return cfg


def check_recipe(cfg):
    assert READ in ("linear", "mlp"), "the record's read is set from test 3's verdict"
    assert cfg in (recipe(), recipe(False)), "P1-F7 recipe differs from F6.1 plus the declared changes"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--no-route", action="store_true")                   # the throughput comparison only
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    TR.p1(check_recipe(recipe(not args.no_route)), WORLD_SEED, args.out, updates=args.updates, steps=args.steps,
          checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
