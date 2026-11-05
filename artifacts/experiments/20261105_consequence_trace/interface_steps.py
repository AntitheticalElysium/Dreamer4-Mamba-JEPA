"""F2's actor interface (artifacts/experiments/20261030_actor_interface/interface.py, unchanged) at 3 and 16 sampler steps
on the same histories, classes and seeds (user 11-05; Codex: read false rewards / damage / termination too, keep the
3-step result, compare at equal budgets). The checkpoint path is spelled two ways; the loader sets edm.sampler.steps per
spelling. R0: the 3-step label must reproduce F2's numbers in F2.1's interface_random.json exactly.

Usage: python -m artifacts.experiments.20261105_consequence_trace.interface_steps OUT_JSON"""
import copy, importlib, sys

I = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
A, B = "artifacts/experiments/20261103_p1f2/run/ckpt_10.pt", "artifacts/experiments/20261103_p1f2/run/./ckpt_10.pt"
STEPS = {A: 3, B: 16}
_load = I.load


def load(path):
    model, cfg = _load(path)
    cfg = copy.deepcopy(cfg)
    cfg["edm"]["sampler"]["steps"] = STEPS[path]
    return model, cfg


if __name__ == "__main__":
    I.load = load
    sys.argv = ["interface", sys.argv[1], "--policy=random", f"f2_3steps={A}", f"f2_16steps={B}"]
    I.main()
