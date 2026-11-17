"""F5 against F4 and HUD-Change (and the MultiStep arm, run invalid as a test of its idea, shown for reference) on the
same saved instruments: wake per stratum and woken beside a zombie, each (item, count) transition, the actor interface
per class, own-output recursion counts. Reuses ../20261109_p1_hud_change/compare.py's readers. CPU, saved outputs only.

    python -m artifacts.experiments.20261111_p1f5.compare
"""
import importlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
H = importlib.import_module("artifacts.experiments.20261109_p1_hud_change.compare")
MS = EXP / "20261111_p1_multistep_mamba/readings"
CURVES = {"f4": H.CURVES["f4"], "hud": H.CURVES["hud"], "ms": MS / "consequences/curves/raw.npz",
          "f5": HERE / "readings/consequences/curves/raw.npz"}
INTERFACE = {"f4": H.INTERFACE["f4"], "hud": H.INTERFACE["hud"], "ms": MS / "interface.json",
             "f5": HERE / "readings/interface.json"}
RECURSION = {"f4": H.RECURSION["f4_4"], "hud": H.RECURSION["hud_4"], "ms": MS / "recursion_steps4/raw.npz",
             "f5": HERE / "readings/recursion_4/raw.npz"}
r3 = H.r3


def interface():
    out = {}
    for arm, p in INTERFACE.items():
        cks = json.loads(p.read_text())["checkpoints"]
        (key, v), = [(k, v) for k, v in cks.items() if k.endswith("4steps") or len(cks) == 1] or [next(iter(cks.items()))]
        for cl, c in v["classes"].items():
            row = out.setdefault(cl, {})
            row.setdefault("recorded", {q: r3(c["recorded"]["achievement_unlock" if q == "unlock_p" else q]["est"])
                                        for q in ("unlock_p", "neg_p", "term_p")})
            row[arm] = {q: r3(c["gen"][q]["est"]) for q in ("unlock_p", "neg_p", "term_p")}
    return out


def main():
    out = {"curves": {k: H.curves(np.load(p, allow_pickle=True)) for k, p in CURVES.items() if p.exists()},
           "interface": interface(),
           "recursion_steps_1_4_8_16": {k: H.recursion(np.load(p)) for k, p in RECURSION.items() if p.exists()}}
    (HERE / "readings/comparison.json").write_text(json.dumps(out, indent=1) + "\n")
    print("saved", HERE / "readings/comparison.json")


if __name__ == "__main__":
    main()
