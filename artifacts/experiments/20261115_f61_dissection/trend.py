"""F6.1 dissection, check 9 (PLAN.md): is the denoiser's use still improving at checkpoint 10? The quiet bank's own
generation (`../20261107_danger_adversarial/sleep.py`, unchanged) at F6.1 checkpoints 5 and 8, beside 3 and 10 (saved),
read by check 3's hazard reader.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.trend      (GPU)
"""
import importlib
import json
from pathlib import Path

SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
HZ = importlib.import_module("artifacts.experiments.20261115_f61_dissection.hazard")
HERE = Path(__file__).resolve().parent
F61 = HZ.EXP / "20261114_p1f61"


def main():
    bank = SL.HERE / "bank.pt"
    runs = {3: F61 / "readings/early_03/followups/sleep/run.pt", 10: F61 / "readings/followups/sleep/run.pt"}
    for k in (5, 8):
        d = HERE / f"run/trend/{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "run.pt").exists():
            (d / "bank.pt").write_bytes(bank.read_bytes())
            SL.HERE, SL.CKPT = d, f"artifacts/experiments/20261114_p1f61/run/ckpt_{k:02d}.pt"
            SL.run()
        runs[k] = d / "run.pt"
    res = {k: HZ.read(runs[k]) for k in sorted(runs)}
    for k, r in res.items():
        print(k, json.dumps(r["hazard_by_step_relative_to_due"]), r["spearman_true_vs_generated_median"], flush=True)
    (HERE / "trend.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
