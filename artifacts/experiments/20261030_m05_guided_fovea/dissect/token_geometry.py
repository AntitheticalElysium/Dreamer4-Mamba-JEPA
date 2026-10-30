"""Dissection of M0.5 arm G: how far apart HUD values lie in the glimpse token space the denoiser is trained in (CPU,
evaluate.py's saved readers). Per counter, window centre and checkpoint: the step between consecutive values' real
mean tokens |mu_(v+1) - mu_v|, beside the values' own spread (99th-percentile radius) and the step's share of the
token's squared norm; appearance (0 -> 1) apart from increments (v -> v + 1, v >= 1); the periphery token of the same
counter (16-d, the cell's own token) beside the glimpse token (64-d) for comparison.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.dissect.token_geometry
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
LABELS = ("base", "guided_0300", "guided_0750", "guided_1500")


def main():
    res = {}
    for label in LABELS:
        rd = torch.load(EVAL / f"{label}_readers.pt", weights_only=False)["readers"]
        g = {**rd["g"], **{f"{n}@periphery": r for n, r in rd["p"].items()}}
        res[label] = {}
        for key, r in g.items():
            if not r["qualified"] or key.startswith(("sapling", "health")):
                continue
            sup = {v: (mu.float(), rad) for (_, v), (mu, rad) in r["support"].items()}
            rows = []
            for v in sorted(sup):
                if v + 1 in sup:
                    step = float((sup[v + 1][0] - sup[v][0]).norm())
                    rows.append({"from": v, "step": step, "step_over_radius": step / ((sup[v][1] + sup[v + 1][1]) / 2),
                                 "step_sq_over_norm_sq": step ** 2 / float(sup[v + 1][0].square().sum())})
            if rows:
                app = [x for x in rows if x["from"] == 0]
                inc = [x for x in rows if x["from"] >= 1]
                res[label][key] = {"appearance": app[0] if app else None,
                                   "increment_mean": {k: float(np.mean([x[k] for x in inc])) for k in rows[0] if k != "from"} if inc else None,
                                   "transitions": rows}
    (HERE / "token_geometry.json").write_text(json.dumps(res, indent=1))
    for label, r in res.items():
        for key, x in r.items():
            a, i = x["appearance"], x["increment_mean"]
            print(f"{label:12s} {key:10s} appearance step {a and round(a['step'], 2)} (/radius {a and round(a['step_over_radius'], 2)})"
                  f"  increments step {i and round(i['step'], 2)} (/radius {i and round(i['step_over_radius'], 2)}, "
                  f"share of |token|^2 {i and round(i['step_sq_over_norm_sq'], 3)})")


if __name__ == "__main__":
    main()
