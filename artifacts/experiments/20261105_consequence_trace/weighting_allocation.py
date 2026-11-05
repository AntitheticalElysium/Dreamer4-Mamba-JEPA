"""POST-HOC (11-05, N1 research): candidate G1 weightings applied to F2's MEASURED training-form term (B4: per row,
the drawn physical / glimpse sigma, the raw-output squared errors and c_out^2, so x-space error = raw x c_out^2). A
weighting w(sigma) on the x-space error reweights the same errors; EDM's (ours, sigma_data 1) is w = 1 + 1 / sigma^2
(= 1 / c_out^2, Kingma & Gao 2023: x-loss + eps-loss). First-order proxy: F2's current errors, not a trained world.

Per weighting: the whole term's allocation by sigma band (random frames), the share below sigma 0.3 (clean-up /
fidelity), and per event class the TEACHING SHARE = event rate x E_event[w x consequence-cell x-error, sigma_p >=
sigma*] / E_random[w x whole x-error], sigma* = HUD separation / sqrt(2 |logit p0|) (where a conditioning-blind field
starts to miss, RESULTS.md 2), with its multiplier over EDM. Usage: python weighting_allocation.py READINGS_DIR"""
import json, math, sys
from pathlib import Path
import numpy as np

BANDS = [(0, 0.3), (0.3, 1), (1, 3), (3, 7), (7, np.inf)]
D4_RAMP = lambda s: 0.9 / (1 + s) + 0.1                                 # Dreamer 4 w(tau) = 0.9 tau + 0.1, tau = 1 / (1 + sigma)
W = {
    "edm_sigma_data_1 (ours)": lambda s: 1 + 1 / s ** 2,
    "edm_sigma_data_0.5 (DIAMOND)": lambda s: 1 + 0.25 / s ** 2,
    "x_loss (Dreamer 4 main)": lambda s: np.ones_like(s),
    "x_loss_ramp (Dreamer 4 full)": D4_RAMP,
    "min_snr_5 (Hang 2023)": lambda s: np.minimum(1 / s ** 2, 5.0),
    "p2_k1_g1 (Choi 2022, on eps)": lambda s: 1 / (1 + s ** 2),
    "sigmoid_k2 (Kingma & Gao 2023)": lambda s: math.e ** 2 / (math.e ** 2 * s ** 2 + 1),
}


def load(r, kind):
    z = np.load(r / f"{kind}.npz")
    q = {k: z[f"obj_{k}"].astype(float).ravel() for k in ("sig_p", "sig_g", "cells", "peri", "glim", "light", "co_p", "co_g")}
    return q


def main():
    r = Path(sys.argv[1])
    res_t = json.loads((r / "results.json").read_text())
    rnd = load(r, "random")
    phys, glim = (rnd["peri"] + rnd["light"]) * rnd["co_p"], rnd["glim"] * rnd["co_g"]
    # EDM2-style equalisation (Karras 2024, uncertainty per level): w ~ 1 / E[x-error | sigma], binned in log sigma
    edges = np.quantile(np.log(rnd["sig_p"]), np.linspace(0, 1, 41))
    def equal(s):
        b = np.clip(np.searchsorted(edges, np.log(s)) - 1, 0, 39)
        m = np.array([phys[(np.clip(np.searchsorted(edges, np.log(rnd["sig_p"])) - 1, 0, 39) == i)].mean() for i in range(40)])
        return 1 / m[b]
    W["equalised_per_level (EDM2-style)"] = equal
    out = {}
    for name, w in W.items():
        tot_rows = w(rnd["sig_p"]) * phys + w(rnd["sig_g"]) * glim
        total = tot_rows.mean()
        bands = {f"{a}-{b}": float(((w(rnd["sig_p"]) * phys)[(rnd["sig_p"] >= a) & (rnd["sig_p"] < b)].sum()
                                    + (w(rnd["sig_g"]) * glim)[(rnd["sig_g"] >= a) & (rnd["sig_g"] < b)].sum()) / tot_rows.sum())
                 for a, b in BANDS}
        o = out[name] = {"by_band": bands, "below_0.3": bands["0-0.3"], "teaching_share": {}}
        for kind in ("coal", "iron", "wood", "drink", "risky", "adjacent"):
            if not (r / f"{kind}.npz").exists():
                continue
            q, c = load(r, kind), res_t["classes"][kind]
            p0 = res_t["opportunities"][kind] / res_t["target_frames"]
            s_star = c["geometry"]["hud"]["separation_median"] / math.sqrt(2 * abs(math.log(p0 / (1 - p0))))
            cells = w(q["sig_p"]) * q["cells"] * q["co_p"] * (q["sig_p"] >= s_star)
            o["teaching_share"][kind] = {"sigma_star": s_star, "share": float(p0 * cells.mean() / total)}
    base = out["edm_sigma_data_1 (ours)"]["teaching_share"]
    for name, o in out.items():
        for kind, t in o["teaching_share"].items():
            t["x_over_ours"] = t["share"] / base[kind]["share"]
    (r / "weighting_allocation.json").write_text(json.dumps(out, indent=1))
    print(f"{'weighting':34s} " + " ".join(f"{a}-{b}".rjust(8) for a, b in BANDS) + " | teaching share x ours: "
          + " ".join(k.rjust(8) for k in out[next(iter(out))]["teaching_share"]))
    for name, o in out.items():
        print(f"{name:34s} " + " ".join(f"{100 * v:7.1f}%" for v in o["by_band"].values()) + " | "
              + " ".join(f"{t['x_over_ours']:8.2f}" for t in o["teaching_share"].values()))
    print("sigma*:", {k: round(t["sigma_star"], 2) for k, t in out[next(iter(out))]["teaching_share"].items()},
          "| ours' teaching share:", {k: f"{t['share']:.2e}" for k, t in base.items()})


if __name__ == "__main__":
    main()
