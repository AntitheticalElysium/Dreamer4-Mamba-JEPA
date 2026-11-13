"""Checkpoint-bound, resumable F6 evaluation queue: F5's queue (../20261111_p1f5/readout.py, same tools and order) with
the forecast scores read from the token forecast, without the F4 / HUD-Change control pairs (F5's readings hold
them), followed by the clock readings and F5's dissection follow-ups (stages.py).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_p1f6.evaluation CHECKPOINT [--out DIR --smoke]
"""
import importlib
from pathlib import Path

R = importlib.import_module("artifacts.experiments.20261111_p1f5.readout")
HERE = Path(__file__).resolve().parent
F5_STEPS = R.steps
EXTRA = ("clocks", "trajectory", "movement", "mining", "sleep", "entities", "compare")


def expected(name, checkpoint, readings):
    k = int(checkpoint.stem[-2:])
    return {"clocks": readings / f"trace_{k:02d}.json", "trajectory": readings / "trajectory/ckpt_10.json",
            "entities": readings / "dissection/entities.json", "compare": readings / "comparison.json"}.get(
                name, readings / f"followups/{name}/result.json")


def steps(checkpoint, readings, smoke):
    rows = []
    for name, command, out in F5_STEPS(checkpoint, readings, smoke):
        if name.startswith("native_pairs_"):
            continue
        if name == "forecast_scores":
            command = ["-m", f"{__package__}.forecast_scores", *command[2:]]
        if name == "interface":
            command = [c.replace("f5_4steps=", "f6_4steps=") for c in command]
        rows.append((name, command, out))
    if not smoke:
        rows += [(name, ["-m", f"{__package__}.stages", name, str(checkpoint.resolve()), str(readings)],
                  expected(name, checkpoint, readings)) for name in EXTRA]
    return rows


if __name__ == "__main__":
    R.steps, R.HERE = steps, HERE
    R.main()
