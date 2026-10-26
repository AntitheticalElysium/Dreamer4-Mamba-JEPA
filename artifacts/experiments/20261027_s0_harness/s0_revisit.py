"""S0 step 3 (2026-10-27): revisit references and their power (TEST_SPEC v1.1, S2.2).

A revisit = a cell entering the view that was seen before; its gap = steps since it was last in view. For each gap
stratum (1-16, 17-64, 65-256, > 256): how often revisits happen (per transition) and how often the terrain agrees with
the last sighting (entity-covered cells excluded both times), with episode-bootstrap SE; on all of dev and on the
eps = 1.0 episodes the 16-step S2 windows come from. Power: dev half A (even episodes) as reference; half B must pass;
half B with terrain FORGETTING (revisited terrain redrawn from the terrain marginal with p = 0.05) must be rejected
(|B - A| > 3 joint SE). The minimal detectable drop (3 joint SE) is reported per stratum.
Usage (any venv with numpy): s0_revisit.py LABELS_NPZ OUT_JSON
"""
import json, sys
import numpy as np
import s0_gates as G


def refs(R, n_trans):
    out = G.revisit_estimates(R)
    for (lo, hi), k in zip(((1, 16), (17, 64), (65, 256), (257, 10 ** 9)), out):
        out[k]["per_transition"] = float(((R[:, 1] >= lo) & (R[:, 1] <= hi)).sum() / n_trans)
    return out


def main():
    L = dict(np.load(sys.argv[1])); eps = G.episode_grids(L); rng = np.random.default_rng(20261029)
    marg = np.bincount(L["s_view"].ravel(), minlength=17) / L["s_view"].size
    out = {}
    for name, sel in (("all_dev", np.ones(len(eps), bool)), ("eps1", L["ep_eps"] == 1.0)):
        ids = np.flatnonzero(sel); R = G.revisit_events([eps[i] for i in ids])
        out[name] = {"n_episodes": len(ids), "reference": refs(R, int(L["ep_T"][ids].sum()))}
        A = R[R[:, 0] % 2 == 0]; B = R[R[:, 0] % 2 == 1]
        RF = G.revisit_events([eps[i] for i in ids], drift=0.05, rng=rng, marg=marg); F = RF[RF[:, 0] % 2 == 1]
        ra, rb, rf = G.revisit_estimates(A), G.revisit_estimates(B), G.revisit_estimates(F)
        power = {}
        for k in ra:
            if ra[k] is None or rb[k] is None:
                continue
            se_b, se_f = np.hypot(ra[k]["se"], rb[k]["se"]), np.hypot(ra[k]["se"], rf[k]["se"])
            power[k] = {"real_B_diff": rb[k]["est"] - ra[k]["est"], "real_B_rejected": bool(abs(rb[k]["est"] - ra[k]["est"]) > 3 * se_b),
                        "forget_diff": rf[k]["est"] - ra[k]["est"], "forget_rejected": bool(abs(rf[k]["est"] - ra[k]["est"]) > 3 * se_f),
                        "min_detectable_drop": float(3 * se_b), "n_A": ra[k]["n_cond"], "n_B": rb[k]["n_cond"]}
        out[name]["power"] = power
        print(name, json.dumps(out[name], indent=1), flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
