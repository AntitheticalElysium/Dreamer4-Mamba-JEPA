"""A1 robustness: fit variance. The episode bootstrap in ana_signal_incentives.py treats the fitted models as fixed; with
rare targets, which features the full model happens to use can move every incentive together. Here each binary target
is refit on R different training subsamples (same episode split, same test set); the incentive per group is reported
as the mean over refits with the across-refit range, next to a refit-noise reference: CE(full, refit r) - CE(full,
refit 0), the spread of the full model alone.
Usage: ana_signal_incentives_refit.py DATA_NPZ OUT_JSON MAX_TRAIN_ROWS R TARGET[,TARGET...]
"""
import json, sys, zlib
import numpy as np
import ana_signal_incentives as A


def main():
    data, out_json, max_rows, R = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    wanted = sys.argv[5].split(",")
    D = dict(np.load(data))
    ep = D["episode"]; starts, ends = A.episodes(ep)
    G, CAT = A.features(D)
    for k in ("zadj", "znear", "zfar", "cow", "skel"):
        CAT[k] = []
    r = D["r"].astype(np.float64); Ar = np.round(r + 0.25); H = np.round(10 * (r - Ar))
    u = np.unique(ep); rng = np.random.default_rng(7); perm = rng.permutation(len(u))
    lab = np.empty(len(u), object); lab[perm[: int(.6 * len(u))]] = "tr"
    lab[perm[int(.6 * len(u)): int(.8 * len(u))]] = "va"; lab[perm[int(.8 * len(u)):]] = "te"
    split = lab[np.searchsorted(u, ep)]
    a = D["a"].astype(int); dead = D["dead"] & D["done"]
    out = {}
    for name in wanted:
        tname, k, var = name.split("_"); k = int(k)
        y = {"dmg": A.window_any(H < 0, k, starts, ends), "ach": A.window_any(Ar > 0, k, starts, ends),
             "term": A.window_any(dead, k, starts, ends)}[tname].astype(int)
        extra, cat = {}, {}
        if var == "act":
            extra = {"fut": np.stack([A.shift(a, j, starts, ends, 17) for j in range(k)], 1).astype(np.float32)}
            cat = {"fut": list(range(k))}
        CATx = dict(CAT); CATx.update(cat)
        rows = np.arange(len(ep))
        te = rows[split == "te"]; te = np.sort(np.random.default_rng(zlib.crc32(name.encode())).choice(te, min(len(te), max_rows // 2), replace=False))
        va = rows[split == "va"]; va = np.sort(np.random.default_rng(1).choice(va, min(len(va), max_rows // 2), replace=False))
        res = {"full_refit_noise": [], "groups": {g: [] for g in A.GROUPS}}
        full0 = None
        for rep in range(R):
            tr = rows[split == "tr"]
            tr = np.sort(np.random.default_rng(100 + rep).choice(tr, min(len(tr), max_rows), replace=False))
            parts = {"tr": tr, "va": va, "te": te}

            def ce(drop=()):
                X = {s: A.assemble(G, CATx, extra, drop, (), None, parts[s]) for s in parts}
                l, _ = A.fit_eval(X["tr"][0], y[tr], X["va"][0], y[va], X["te"][0], y[te], X["tr"][1], 2)
                return l
            full = ce()
            if full0 is None:
                full0 = full
            res["full_refit_noise"].append(float(full.mean() - full0.mean()))
            for g in A.GROUPS:
                res["groups"][g].append(float((ce((g,)) - full).mean()))
            print(name, rep, {g: round(v[-1], 4) for g, v in res["groups"].items()}, flush=True)
        res["summary"] = {g: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for g, v in res["groups"].items()}
        out[name] = res
        json.dump(out, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
