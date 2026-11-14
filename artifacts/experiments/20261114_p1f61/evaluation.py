"""Checkpoint-bound, resumable F6.1 evaluation queue: F5's queue (../20261111_p1f5/readout.py, same tools and order)
without the forecast scores (F6.1 has no visual forecast) and without the F4 / HUD-Change control pairs (F5's readings
hold them), followed by the clock readings, F5's dissection follow-ups and the sleep split (stages.py).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.evaluation CHECKPOINT [--out DIR --smoke]
"""
import importlib
from pathlib import Path

R = importlib.import_module("artifacts.experiments.20261111_p1f5.readout")
HERE = Path(__file__).resolve().parent
F5_STEPS = R.steps
EXTRA = ("clocks", "trajectory", "movement", "mining", "sleep", "sleep_split", "entities", "compare")


def expected(name, checkpoint, readings):
    k = int(checkpoint.stem[-2:])
    return {"clocks": readings / f"trace_{k:02d}.json", "trajectory": readings / "trajectory/ckpt_10.json",
            "sleep_split": readings / "followups/sleep_split.json",
            "entities": readings / "dissection/entities.json", "compare": readings / "comparison.json"}.get(
                name, readings / f"followups/{name}/result.json")


def steps(checkpoint, readings, smoke):
    rows = []
    for name, command, out in F5_STEPS(checkpoint, readings, smoke):
        if name.startswith("native_pairs_") or name == "forecast_scores":
            continue
        if name == "interface":
            command = [c.replace("f5_4steps=", "f61_4steps=") for c in command]
        rows.append((name, command, out))
    if not smoke:
        rows += [(name, ["-m", f"{__package__}.stages", name, str(checkpoint.resolve()), str(readings)],
                  expected(name, checkpoint, readings)) for name in EXTRA]
    return rows


if __name__ == "__main__":
    R.steps, R.HERE = steps, HERE
    R.main()
