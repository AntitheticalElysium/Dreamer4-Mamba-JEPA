"""Train F6.1 to the full budget, then queue the checkpoint-selected readings (one service, F5's and F6's worker)."""
import importlib

from . import train as launch

WORKER = importlib.import_module("artifacts.experiments.20261104_p1f21.run")


if __name__ == "__main__":
    WORKER.main(L=launch, here=launch.HERE, module=f"{__package__}.train",
                gates=("checks.json", "eval_checks.json"), evaluation_module=f"{__package__}.evaluation")
