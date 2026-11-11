"""Train and queue the checkpoint-selected F5 readings in the same service."""
import importlib

from . import train_p1f5 as launch

WORKER = importlib.import_module("artifacts.experiments.20261104_p1f21.run")


if __name__ == "__main__":
    WORKER.main(L=launch, here=launch.HERE, module=f"{__package__}.train_p1f5",
                gates=("checks.json", "eval_checks.json"), evaluation_module=f"{__package__}.readout")
