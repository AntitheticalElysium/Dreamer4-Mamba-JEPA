"""PLAN.md decision rules applied to CHECKS.json -> SUMMARY.json. Written before the checks finished. CPU, no models."""
import json, sys
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent


def main(name="CHECKS.json", boot=2000, seed=0):
    r = json.loads((P / name).read_text())
    noise = r["noise"]
    nf, ng = np.array([x["change"] for x in noise]), np.array([x["g1"] for x in noise])
    dark = np.array([x["night"] for x in noise])
    floors = {k: {"change": float(nf[m].mean()), "g1": float(ng[m].mean()), "states": int(m.sum())}
              for k, m in (("all", np.ones_like(dark)), ("day", ~dark), ("dark", dark))}
    rng = np.random.default_rng(seed)
    out = {"floors": floors, "checks": {}}
    foil_ok = True
    for c in ("C1", "C2"):
        rows = [x for x in r["rows"][c] if "excluded" not in x]
        sf, sg = np.array([x["change"] for x in rows]), np.array([x["g1"] for x in rows])
        ratio = lambda a, b, fa, fb: (a / fa) / (b / fb)
        R = ratio(sf, sg, nf.mean(), ng.mean())
        bs = []
        for _ in range(boot):
            i, j = rng.integers(len(rows), size=len(rows)), rng.integers(len(noise), size=len(noise))
            bs.append(np.median(ratio(sf[i], sg[i], nf[j].mean(), ng[j].mean())))
        foils = {a: np.array([x["foil"][a]["change"] for x in rows]) for a in ("1", "3", "6")}
        foils_g1 = {a: np.array([x["foil"][a]["g1"] for x in rows]) for a in ("1", "3", "6")}
        controls = max((max(x["controls"]) for x in rows), default=0.0)
        f_ok = all(bool((foils[a] < sf).all()) for a in foils)
        foil_ok &= f_ok
        out["checks"][c] = {
            "pairs": len(rows), "excluded": [x for x in r["rows"][c] if "excluded" in x],
            "change_cost": {"min": float(sf.min()), "median": float(np.median(sf)), "max": float(sf.max())},
            "g1_cost": {"min": float(sg.min()), "median": float(np.median(sg)), "max": float(sg.max())},
            "R_median": float(np.median(R)), "R_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "R_day_floor": float(np.median(ratio(sf, sg, floors["day"]["change"], floors["day"]["g1"]))),
            "R_dark_floor": float(np.median(ratio(sf, sg, floors["dark"]["change"], floors["dark"]["g1"]))),
            "consequence_signal_to_floor": {"change": float(np.median(sf) / nf.mean()), "g1": float(np.median(sg) / ng.mean())},
            "control_max": float(controls),
            "foil_over_consequence_max": {a: float((foils[a] / sf).max()) for a in foils},
            "foil_over_consequence_max_g1": {a: float((foils_g1[a] / sg).max()) for a in foils_g1},
            "pass_positive": bool((sf > 0).all()), "pass_controls": controls <= 1e-12,
            "pass_R": float(np.median(R)) >= 3, "pass_foil": f_ok,
            "gap_cells": sorted({tuple(g[:3]) for x in rows for g in x["gap_cells"]}),
        }
    for c in ("C1", "C2"):
        k = out["checks"][c]
        k["pass"] = k["pass_positive"] and k["pass_controls"] and k["pass_R"]
    out["C3_pass"] = foil_ok
    out["all_pass"] = out["C3_pass"] and out["checks"]["C1"]["pass"] and out["checks"]["C2"]["pass"]
    (P / name.replace("CHECKS", "SUMMARY")).write_text(json.dumps(out, indent=1, default=lambda x: list(x)))
    print(json.dumps(out, indent=1, default=lambda x: list(x)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
