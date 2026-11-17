"""F6's numbers for F6.1's checkpoint-3 review, read with F6.1's own instruments (PLAN.md, readings): the gradient
budget and the reward forecast's sleep readout at F6's checkpoints 1-3, and the quiet sleep bank with its split at F6's
checkpoint 3 (F6 ran it only at checkpoint 10). Writes readings/reference_f6/.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.reference
"""
import importlib
import os

from . import readout as RD, stages as ST

F6 = ST.F6
OUT = RD.OUT / "reference_f6"


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        ck = (F6 / f"run/ckpt_{k:02d}.pt").resolve()
        for name, fn in (("budget", RD.budget), ("ahead", RD.ahead)):
            if not (OUT / f"{name}_{k:02d}.json").exists():
                fn(ck, OUT / f"{name}_{k:02d}.json")
    ck = (F6 / "run/ckpt_03.pt").resolve()
    FU = importlib.import_module(f"{ST.F5}.followups")
    FU.OUT = OUT / "ckpt_03/followups"
    if not (FU.OUT / "sleep/result.json").exists():
        FU.run_stage("sleep", ck)
    ST.sleep_split(ck, OUT / "ckpt_03")
