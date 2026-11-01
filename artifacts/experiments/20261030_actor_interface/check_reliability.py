"""Check of interface.reliability on simulated worlds (CPU; Codex's perfect-predictor counterexample, 10-31): histories
with true P(event) 0.1 or 0.6, one recorded draw each, K = 4 generated draws. A perfect world must read calibrated
(held-out predicted = recorded in every bin); a world carrying 30% of the event must read under-predicted. The first
version (binning and reading the same 4-draw mean) read a perfect world 0.0 vs 0.12 and 1.0 vs 0.60.
Usage: python -m artifacts.experiments.20261030_actor_interface.check_reliability
"""
import importlib, json
from pathlib import Path

import numpy as np

I = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")


def main():
    rng = np.random.default_rng(0)
    H, K = 20000, 4
    p = np.where(rng.random(H) < 0.5, 0.1, 0.6)
    tup, rec = np.repeat(np.arange(H), K), np.repeat(rng.random(H) < p, K)
    res = {}
    for name, rate in (("perfect", 1.0), ("carries_30pct", 0.3)):
        gen = (rng.random(H * K) < np.repeat(p * rate, K)).astype(float)
        res[name] = I.reliability({"tuple": tup, "rec_event": rec, "gen_unlock_p": gen, "rec_head_unlock_p": rec.astype(float)},
                                  "unlock_p", "rec_event")["gen"]
    ok = all(abs(b["mean_predicted_heldout"] - b["recorded"]) < 3 * b["recorded_se"] + 0.01 for b in res["perfect"]) and \
        all(b["mean_predicted_heldout"] < b["recorded"] - 3 * b["recorded_se"] for b in res["carries_30pct"])
    res["pass"] = bool(ok)
    (Path(__file__).resolve().parent / "check_reliability.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert ok


if __name__ == "__main__":
    main()
