"""Learned-fixation premise test D (PLAN.md): is there a substrate for value of information on P1-F? The fovea probes
F1 / F2 (10-29) unchanged on P1-F checkpoint 10 (S1's seeded random fixations, first N dev episodes, seeds 2029), then
analyse.py and Codex's corrected F1 one-step contrast and F2 sets (check_saved_outputs.py) on this run. World 1's run
of the same scripts (20261029_fovea_probes/run, 250 episodes) is the control. CPU.
Usage (project venv, CPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.substrate OUT_DIR [N_EPISODES]
"""
import importlib, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
FP = importlib.import_module("artifacts.experiments.20261029_fovea_probes.measure")
AN = importlib.import_module("artifacts.experiments.20261029_fovea_probes.analyse")
CK = importlib.import_module("artifacts.experiments.20261029_completed_diagnosis_review.check_saved_outputs")
P1F = Path(__file__).resolve().parents[1] / "20261030_p1_guided/run/ckpt_10.pt"


def main():
    out, n_ep = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "250"
    FP.CKPT = P1F
    sys.argv = ["measure", str(out), n_ep]
    FP.main()
    sys.argv = ["analyse", str(out)]
    AN.main()
    CK.RUN, CK.OUT = out, out                                           # writes OUT / checks.json
    CK.main()


if __name__ == "__main__":
    main()
