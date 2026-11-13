"""D1 (PLAN.md round 2): where each reset is lost. The section-6 registration protocol of ../20261111_p1f5/dissect.py
(logistic on |change|, 5 episode folds, AP with an episode bootstrap) at three stages, from gate2.py's saved features:
the periphery tokens (81 x 16), Mamba's input (percept_embed) and Mamba's corrected state b. CPU only.

    python -m artifacts.experiments.20261113_token_forecast_gate.d1 [checkpoints...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("GATE_OUT", "readings/round2")
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
SECTION6 = HERE.parent / "20261111_p1f5/readings/dissection/registration.json"
DIGITS = ("health", "food", "drink", "energy")


def events(T, rows):
    body, true, awake = T["body"][rows], T["true_body"][rows], ~T["asleep"][rows]
    return {"health_drop": true[:, 0] < body[:, 0], "health_rise": true[:, 0] > body[:, 0],
            "food_drop_awake": (true[:, 1] < body[:, 1]) & awake, "food_rise": true[:, 1] > body[:, 1],
            "drink_drop_awake": (true[:, 2] < body[:, 2]) & awake, "drink_rise": true[:, 2] > body[:, 2],
            "energy_drop_awake": (true[:, 3] < body[:, 3]) & awake}


def probe(X, y, fold, ep):
    s = np.zeros(len(y))
    for f in range(5):
        m = fold == f
        s[m] = RT.logistic(X[~m], y[~m], X[m])
    return {"prevalence": float(y.mean()), "events": int(y.sum()), "ap": RT.ap_boot(y, s, ep)}


def checkpoint(k):
    path = OUT / f"d1_ckpt_{k:02d}.json"
    if path.exists():
        return
    z = np.load(OUT / f"d1_ckpt_{k:02d}.npz")
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    rows = np.sort(np.random.default_rng(2120).choice(np.flatnonzero(T["t"] >= CV.P), 40000, replace=False))
    assert np.array_equal(rows, z["rows"])
    ep = T["ep"][rows]
    fold = RT.folds_of(ep)
    res = {"checkpoint": k, "rows": len(rows)}
    X = {s: torch.as_tensor(z[s]).double() for s in ("p", "x", "b")}
    if k == 10:                                                        # section 6's any-change rows, input stage
        ref = json.loads(SECTION6.read_text())["f5"]
        res["validation"] = {}
        for i, name in enumerate(DIGITS):
            y = (T["true_body"][rows][:, i] != T["body"][rows][:, i]).astype(float)
            r = probe(X["x"], y, fold, ep)
            res["validation"][name] = {"now": r["ap"]["ap"], "section6": ref[name]["ap"]["ap"]}
            print(k, "validation", name, res["validation"][name], flush=True)
    for name, y in events(T, rows).items():
        res[name] = {}
        for stage in ("p", "x", "b"):
            res[name][stage] = probe(X[stage], y.astype(float), fold, ep)
            print(k, name, stage, res[name][stage]["events"], round(res[name][stage]["ap"]["ap"], 4),
                  [round(c, 4) for c in res[name][stage]["ap"]["ci95"]], flush=True)
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(8)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 10)):
        checkpoint(k)
