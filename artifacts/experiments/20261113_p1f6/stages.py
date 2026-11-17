"""F6's readings beyond F5's standard queue, each F5's own instrument pointed at this run (one stage per process).

  clocks      readout.py at checkpoint 2 and the selected checkpoint: the token forecast on the exact-history pairs, belief probes,
              reset registration, trace.
  trajectory  ../20261111_p1f5/trajectory.py at all ten checkpoints (F5 RESULTS 11.2's table).
  movement, mining, sleep   ../20261111_p1f5/followups.py's stages (their "f5" key is the evaluated checkpoint).
  entities    ../20261111_p1f5/entities.py with this checkpoint in F5's place, beside F4 and HUD-Change.
  compare     ../20261111_p1f5/compare.py with an "f6" column.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_p1f6.stages STAGE CHECKPOINT READINGS
"""
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F5 = "artifacts.experiments.20261111_p1f5"


def main(name, checkpoint, readings):
    k = int(checkpoint.stem[-2:])
    assert checkpoint == (HERE / f"run/ckpt_{k:02d}.pt").resolve()
    if name == "clocks":
        RD = importlib.import_module(f"{__package__}.readout")
        RD.OUT = readings
        for j in sorted({2, k}):                                         # checkpoint 2: F5's controls exist at 1, 2 and 10
            RD.main(j, RD.STAGES)
    elif name == "trajectory":
        TJ = importlib.import_module(f"{F5}.trajectory")
        TJ.HERE, TJ.OUT = HERE, readings / "trajectory"
        TJ.OUT.mkdir(parents=True, exist_ok=True)
        for j in range(1, 11):
            TJ.checkpoint(j)
    elif name in ("movement", "mining", "sleep"):
        FU = importlib.import_module(f"{F5}.followups")
        FU.OUT = readings / "followups"
        FU.run_stage(name, checkpoint)
    elif name == "entities":
        EN = importlib.import_module(f"{F5}.entities")
        EN.OUT = readings / "dissection"
        EN.OUT.mkdir(parents=True, exist_ok=True)
        EN.DS.CKPTS = dict(EN.DS.CKPTS, f5=checkpoint)
        EN.main()
        path = EN.OUT / "entities.json"
        res = json.loads(path.read_text())
        res["f6"] = res.pop("f5")
        path.write_text(json.dumps(res, indent=1) + "\n")
    elif name == "compare":
        CP = importlib.import_module(f"{F5}.compare")
        CP.HERE = HERE
        CP.CURVES["f6"] = readings / "consequences/curves/raw.npz"
        CP.INTERFACE["f6"] = readings / "interface.json"
        CP.RECURSION["f6"] = readings / "recursion_4/raw.npz"
        CP.main()
    else:
        raise SystemExit(f"unknown stage {name}")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main(sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
