"""POST-HOC (11-05, after F2's declared readings; not a declared reading): on the saved per-row arrays, (1) the joint
map-cell / HUD outcome of the same row's first estimate and sample, (2) HUD by start count (appearance 0 -> 1 vs
increment), (3) the HUD field beside a conditioning-blind exact two-outcome denoiser on real successor + noise (prior
= the class's opportunity rate over target frames, i.e. what a denoiser that ignores the event would assume).

Usage: python posthoc.py READINGS_DIR (e.g. readings/f2)"""
import json, sys
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

SHOW = (0.377, 0.7, 1.2, 2.0, 3.5, 6.67, 20.0)
ZG, WG = hermgauss(40)
ZG, WG = ZG * np.sqrt(2), WG / np.sqrt(np.pi)


def blind(sep, s, p0):
    L = np.log(p0 / (1 - p0)) + sep[:, None] ** 2 / (2 * s * s) + (sep[:, None] / s) * ZG[None]
    return float((expit(L) @ WG).mean())


def main():
    r = sys.argv[1].rstrip("/") + "/"
    d = json.load(open(r + "results.json"))
    levels, out = d["levels"], {}
    sel = [i for i, s in enumerate(levels) if s in SHOW]
    for kind in [k for k in ("coal", "iron", "wood", "drink", "stone") if k in d["classes"]]:
        z = np.load(r + kind + ".npz")
        v = z["valid"]
        start = z["ctx"][z["row_ctx"]][v, 3]
        o = out[kind] = {"rows": int(v.sum()), "start0": int((start == 0).sum())}
        for stage in ("first", "prod"):
            h = z[f"base_{stage}_hud"][v]
            if f"base_{stage}_cell" in z.files:
                c = z[f"base_{stage}_cell"][v]
                o[f"{stage}_joint"] = {"cell+hud+": float(np.mean((c > .5) & (h > .5))), "cell+hud-": float(np.mean((c > .5) & (h <= .5))),
                                       "cell-hud+": float(np.mean((c <= .5) & (h > .5))), "cell-hud-": float(np.mean((c <= .5) & (h <= .5)))}
            for nm, m in (("start0", start == 0), ("start_gt0", start > 0)):
                if m.sum():
                    o[f"{stage}_hud_{nm}"] = {"n": int(m.sum()), "mean_f": float(h[m].mean()), "frac_gt_half": float(np.mean(h[m] > .5))}
        F = z["base_field_hud"][v]
        for nm, m in (("start0", start == 0), ("start_gt0", start > 0)):
            if m.sum():
                o[f"field_hud_{nm}"] = {levels[i]: float(F[m, i].mean()) for i in sel}
        p0 = d["opportunities"][kind] / d["target_frames"]
        o["blind_prior"] = p0
        o["blind_exact"] = {levels[i]: blind(z["sep"][v, 0], levels[i], p0) for i in sel}
        if "cell" in d["classes"][kind]["parts"]:
            o["field_cell"] = {levels[i]: float(z["base_field_cell"][v][:, i].mean()) for i in sel}
    (open(r + "posthoc.json", "w")).write(json.dumps(out, indent=1))
    fmt = lambda c: " ".join(f"{float(k):.3g}:{x:.2f}" for k, x in c.items())
    for kind, o in out.items():
        print(f"== {kind}: rows {o['rows']} (start 0: {o['start0']})")
        for stage in ("first", "prod"):
            if f"{stage}_joint" in o:
                print(f"  {stage} joint", {k: round(x, 3) for k, x in o[f"{stage}_joint"].items()})
            print(f"  {stage} hud", {k[len(stage) + 5:]: (x["n"], round(x["mean_f"], 3)) for k, x in o.items() if k.startswith(f"{stage}_hud_")})
        for k in ("field_hud_start0", "field_hud_start_gt0", "blind_exact", "field_cell"):
            if k in o:
                print(f"  {k:20s} {fmt(o[k])}")


if __name__ == "__main__":
    main()
