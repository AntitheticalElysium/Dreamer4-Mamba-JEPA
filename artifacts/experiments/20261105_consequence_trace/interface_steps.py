"""F2's actor interface (artifacts/experiments/20261030_actor_interface/interface.py, unchanged) at 3 and 16 sampler steps
on the same histories, classes and seeds (user 11-05; Codex: read false rewards / damage / termination too, keep the
3-step result, compare at equal budgets). The checkpoint path is spelled two ways; the loader sets edm.sampler.steps per
spelling. R0: the 3-step label must reproduce F2's numbers in F2.1's interface_random.json exactly.

Usage: python -m artifacts.experiments.20261105_consequence_trace.interface_steps OUT_JSON [CKPT [LABEL]]  (default F2;
F2.1 added 11-05 to test whether F2.1-F2 class swings survive at equal 16-step budgets)"""
import copy, importlib, sys

I = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
CK = sys.argv[2] if len(sys.argv) > 2 else "artifacts/experiments/20261103_p1f2/run/ckpt_10.pt"
NAME = sys.argv[3] if len(sys.argv) > 3 else "f2"
A, B = CK, CK.replace("/ckpt_", "/./ckpt_")
STEPS = {A: 3, B: 16}
_load = I.load


def load(path):
    model, cfg = _load(path)
    cfg = copy.deepcopy(cfg)
    cfg["edm"]["sampler"]["steps"] = STEPS[path]
    return model, cfg


if __name__ == "__main__":
    I.load = load
    sys.argv = ["interface", sys.argv[1], "--policy=random", f"{NAME}_3steps={A}", f"{NAME}_16steps={B}"]
    I.main()
