"""Train P1.5 to the full budget, then queue the checkpoint-selected evaluation in the same service (F2.1's one-shot
worker, as F5 / F6 / F6.1). The evaluation module is written during training in this folder (the trainer and this
queue are sealed by SOURCE_PINS.json; saccade/ must not change); it opens with its own smoke of the record adapter
and raises before any reading if the record is missing or misplaced (PLAN)."""
import importlib
from types import SimpleNamespace

from . import train as launch

WORKER = importlib.import_module("artifacts.experiments.20261104_p1f21.run")
L = SimpleNamespace(recipe=lambda _cfg=None: launch.recipe(), check_recipe=launch.check_recipe,   # the worker's
                    WORLD_SEED=launch.WORLD_SEED, C=launch.F61.C)                                 # interface


if __name__ == "__main__":
    WORKER.main(L=L, here=launch.HERE, module=f"{__package__}.train", gates=("checks.json",),
                evaluation_module=f"{__package__}.evaluation")
