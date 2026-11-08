"""F3's ACTUAL G1 allocation on a consequence-trace B4 directory (Codex review 11-05, finding 1): per draw, physical =
SCALE x (periphery + lighting) raw error x c_out^2, glimpse = its raw (EDM) error, consequence cells = SCALE x cells x
c_out^2. The trace's as_trained (all-parts EDM) and x_space_weighting (all-parts x-space, no relative scale) columns
are counterfactuals for F3. Teaching share as in weighting_allocation.py (cells at sigma_p >= sigma* over the whole
term, x event rate), with its multiplier over the all-EDM term on the same errors.

Usage: python f3_allocation.py TRACE_DIR [OUT_JSON]"""
import json, math, sys
from pathlib import Path
import numpy as np

SCALE = 3.425                                                            # train_p1f3.WEIGHTING['scale']
BANDS = [(0, 0.3), (0.3, 1), (1, 3), (3, 7), (7, np.inf)]
KINDS = ("coal", "iron", "wood", "drink", "stone", "risky", "adjacent")


def load(r, kind):
    z = np.load(r / f"{kind}.npz")
    return {k: z[f"obj_{k}"].astype(float).ravel() for k in ("sig_p", "sig_g", "cells", "peri", "glim", "light", "co_p")}


def terms(q, hybrid):
    phys = SCALE * (q["peri"] + q["light"]) * q["co_p"] if hybrid else q["peri"] + q["light"]
    cells = SCALE * q["cells"] * q["co_p"] if hybrid else q["cells"]
    return phys, q["glim"], cells


def main():
    r = Path(sys.argv[1]); res_t = json.loads((r / "results.json").read_text())
    rnd = load(r, "random"); out = {"scale": SCALE}
    for name, hybrid in (("f3_hybrid", True), ("all_edm", False)):
        phys, glim, _ = terms(rnd, hybrid)
        tot = phys.sum() + glim.sum()
        out[name] = {"glimpse_share": float(glim.sum() / tot), "by_band": {
            f"{a}-{b}": float((phys[(rnd["sig_p"] >= a) & (rnd["sig_p"] < b)].sum() + glim[(rnd["sig_g"] >= a) & (rnd["sig_g"] < b)].sum()) / tot)
            for a, b in BANDS}, "teaching_share": {}}
        for kind in KINDS:
            if not (r / f"{kind}.npz").exists() or kind not in res_t["classes"]:
                continue
            q, c = load(r, kind), res_t["classes"][kind]
            p0 = res_t["opportunities"][kind] / res_t["target_frames"]
            s_star = c["geometry"]["hud"]["separation_median"] / math.sqrt(2 * abs(math.log(p0 / (1 - p0))))
            _, _, cells = terms(q, hybrid)
            out[name]["teaching_share"][kind] = {"sigma_star": s_star, "share": float(p0 * (cells * (q["sig_p"] >= s_star)).mean() / (tot / len(phys)))}
    for kind, t in out["f3_hybrid"]["teaching_share"].items():
        t["x_over_all_edm"] = t["share"] / out["all_edm"]["teaching_share"][kind]["share"]
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    for name in ("f3_hybrid", "all_edm"):
        o = out[name]
        print(f"{name:10s} glimpse {100 * o['glimpse_share']:.2f}% | bands " + " ".join(f"{k}: {100 * v:.2f}%" for k, v in o["by_band"].items()))
    print("teaching share x over all-EDM:", {k: round(t["x_over_all_edm"], 2) for k, t in out["f3_hybrid"]["teaching_share"].items()})


if __name__ == "__main__":
    main()
