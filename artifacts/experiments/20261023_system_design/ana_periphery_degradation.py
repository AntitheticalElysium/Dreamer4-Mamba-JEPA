"""P1 analysis (protocol in eda_periphery_degradation.py). Scratch venv with lightgbm + numpy.
For each entity type and resolution r in {1, 2, 3, 7}: LightGBM on (a) the pooled cell alone, (b) the pooled cell + its 8
neighbours, (c) (b) + a motion cue (the cell pooled to 1x1 one step earlier, and the difference); episode-level split.
Average precision on test cells, by stratum: all / day (light >= 0.5) / night (light < 0.5); the motion variant also on
camera-static steps. Prevalence is the AP of a constant predictor.
Usage: ana_periphery_degradation.py DATA_NPZ OUT_JSON [RESOLUTIONS e.g. 7]
"""
import json, sys
import numpy as np
import lightgbm as lgb

MOBS = ("zombies", "cows", "skeletons", "arrows")


def ap(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]; s = s[o]
    tp = np.cumsum(y); n = y.sum()
    if n == 0:
        return None
    # tie-aware: evaluate precision at the last index of each tied score group
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    prec = tp[last] / (last + 1); rec = tp[last] / n
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def neighbours(x, cells):
    """x [N, F] in frame-major order (63 cells per frame, row-major 7x9) -> [N, 9F] with the 3x3 neighbourhood (NaN pad)."""
    F = x.shape[1]; fr = x.reshape(-1, 7, 9, F)
    pad = np.full((fr.shape[0], 9, 11, F), np.nan, np.float32); pad[:, 1:8, 1:10] = fr
    parts = [pad[:, 1 + dr:8 + dr, 1 + dc:10 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    return np.concatenate(parts, -1).reshape(-1, 9 * F)


def main():
    D = np.load(sys.argv[1])
    out = json.load(open(sys.argv[2])) if len(sys.argv) > 3 else {}
    meta = D["meta"]; ep = meta[:, 0].astype(int); light = meta[:, 1]; static = meta[:, 3] > 0.5
    u = np.unique(ep); rng = np.random.default_rng(0); perm = rng.permutation(u)
    tr = np.isin(ep, perm[: int(.6 * len(u))]); va = np.isin(ep, perm[int(.6 * len(u)): int(.8 * len(u))]); te = ~(tr | va)
    prev = D["prev_r1"].astype(np.float32)
    res_list = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 7]
    for r in res_list:
        x = D[f"r{r}"].astype(np.float32)
        variants = {"cell": x}
        if r != 7:                                  # neighbourhood variants at full resolution exceed memory; the
            cur1 = D["r1"].astype(np.float32)       # ceiling control only needs the cell alone
            variants["cell+nbrs"] = neighbours(x, 63)
            variants["cell+nbrs+motion"] = np.concatenate([variants["cell+nbrs"], prev, cur1 - prev], 1)
        for vname, X in variants.items():
            for m in MOBS:
                y = D[f"y_{m}"].astype(int)
                p = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1,
                     "num_threads": 12, "deterministic": True, "seed": 0}
                dtr = lgb.Dataset(X[tr], y[tr]); dva = lgb.Dataset(X[va], y[va], reference=dtr)
                mdl = lgb.train(p, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
                s = mdl.predict(X[te], num_iteration=mdl.best_iteration)
                yt = y[te]; lt = light[te]; st = static[te]
                res = {"prevalence": float(yt.mean()), "ap_all": ap(yt, s), "ap_day": ap(yt[lt >= 0.5], s[lt >= 0.5]),
                       "ap_night": ap(yt[lt < 0.5], s[lt < 0.5]), "n_pos_test": int(yt.sum()),
                       "n_pos_night": int(yt[lt < 0.5].sum())}
                if vname == "cell+nbrs+motion":
                    res["ap_static_cam"] = ap(yt[st], s[st])
                out[f"r{r}|{vname}|{m}"] = res
                print(f"r={r} {vname:18s} {m:9s} AP all {res['ap_all']:.3f} day {res['ap_day']:.3f} "
                      f"night {res['ap_night'] if res['ap_night'] is None else round(res['ap_night'], 3)} prev {res['prevalence']:.4f}",
                      flush=True)
                json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
