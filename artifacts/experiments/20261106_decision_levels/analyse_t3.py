"""T3's declared readings (PLAN.md), from readings/eval_{f3,C,H,F}.pt (same 208 pairs, same order and seeds in every
arm): per-arm event rates and PAIRED arm differences (episode bootstrap, 2,000, seed 1106).
Usage: python -m artifacts.experiments.20261106_decision_levels.analyse_t3   -> readings/t3.json
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch

R = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
HERE = Path(__file__).resolve().parent
ARMS = ("f3", "C", "H", "F")


def values(rows, kind, recipe, m):
    P_ = [r for r in rows if r["kind"] == kind]
    return np.array([r["members"][m][recipe] for r in P_]), np.array([r["e"] for r in P_])


def verdict_energy(c):
    """PLAN: >= 33 pp contrast = learned, < 10 pp = not."""
    lo, hi = c["ci95"]
    return "LEARNED" if c["est"] >= 0.33 and lo > 0.10 else "NOT LEARNED" if hi < 0.10 else "PARTIAL / UNRESOLVED"


def main():
    E = {a: torch.load(HERE / f"readings/eval_{a}.pt", weights_only=False) for a in ARMS if (HERE / f"readings/eval_{a}.pt").exists()}
    keys = [[(r["kind"], r["e"], r["stratum"]) for r in rows] for rows in E.values()]
    assert all(k == keys[0] for k in keys), "arms evaluated on different pairs"
    out = {"arms": list(E)}
    for a, rows in E.items():
        x9, ep = values(rows, "energy", "prefix20", 1)
        x8, _ = values(rows, "energy", "prefix20", 0)
        con = R.boot(x9 - x8, ep)
        out[a] = {"energy_contrast_prefix20": {**con, "verdict": verdict_energy(con)},
                  "energy_wake_at_9_prefix20": R.boot(x9, ep), "energy_wake_at_8_prefix20": R.boot(x8, ep),
                  "energy_context_only_at_20 (f, member 1 - 0)": R.boot(values(rows, "energy", "pure20_f", 1)[0] - values(rows, "energy", "pure20_f", 0)[0], ep),
                  "sleep_false_wake_base5": R.boot(*values(rows, "sleep", "base5", 0)),
                  "sleep_false_wake_prefix20": R.boot(*values(rows, "sleep", "prefix20", 0)),
                  "sleep_awake_control_prefix20": R.boot(*values(rows, "sleep", "prefix20", 1)),
                  "coal_increment_prefix20": R.boot(*values(rows, "coal", "prefix20", 0)),
                  "wood_increment_prefix20": R.boot(*values(rows, "wood", "prefix20", 0)),
                  "false_increments_noop_prefix20": R.boot(np.r_[values(rows, "coal", "prefix20", 1)[0], values(rows, "wood", "prefix20", 1)[0]],
                                                           np.r_[values(rows, "coal", "prefix20", 1)[1], values(rows, "wood", "prefix20", 1)[1]])}
    for a in ("H", "F", "C"):
        if a not in E or "C" not in E or a == "C":
            continue
        d = {}
        for name, (kind, recipe, m) in {"coal_increment_prefix20": ("coal", "prefix20", 0), "sleep_false_wake_base5": ("sleep", "base5", 0),
                                        "energy_wake_at_9_prefix20": ("energy", "prefix20", 1), "wood_increment_prefix20": ("wood", "prefix20", 0),
                                        "sleep_false_wake_prefix20": ("sleep", "prefix20", 0)}.items():
            xa, ep = values(E[a], kind, recipe, m)
            xc, _ = values(E["C"], kind, recipe, m)
            d[name] = R.boot(xa - xc, ep)
        lo, hi = d["coal_increment_prefix20"]["ci95"]
        d["coal_reading"] = "REFUTES the decision-level split (coal improved > 5 pp)" if lo > 0.05 else \
            "consistent (|difference| <= 5 pp)" if lo > -0.05 and hi < 0.05 else "unresolved at this n"
        lo, hi = d["sleep_false_wake_base5"]["ci95"]
        d["base5_reading"] = "base-5 failure also high-noise under-training (lower by > 15 pp)" if hi < -0.15 else \
            "consistent with an inference mismatch (within 15 pp)" if lo > -0.15 and hi < 0.15 else \
            "H WORSE at the original start (> 15 pp)" if lo > 0.15 else "unresolved"   # Codex 11-06: bound both ends
        out[f"{a}_minus_C"] = d
    if all(a in out for a in ("C", "H", "F")):
        v = {a: out[a]["energy_contrast_prefix20"]["verdict"] for a in ("C", "H", "F")}
        out["primary"] = ("H-learn SUPPORTED" if v["H"] == "LEARNED" and v["C"] == "NOT LEARNED" else
                          "learnable when exposed; the natural mix at this budget is too thin" if v["H"] == v["C"] == "NOT LEARNED" and v["F"] == "LEARNED" else
                          "H-learn REFUTED at this budget (routing / capacity)" if v["H"] == v["C"] == v["F"] == "NOT LEARNED" else
                          "more training of any kind" if v["C"] == "LEARNED" else f"mixed: {v}")
    (HERE / "readings/t3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
