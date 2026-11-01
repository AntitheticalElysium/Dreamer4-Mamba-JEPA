"""A1 analysis (protocol in eda_signal_incentives.py). Run in an environment with lightgbm + numpy (scratch venv).

Part 1, incentives: for each target Y and conditioning variant, fit LightGBM on all content groups (full) and with one
  group dropped; incentive = test CE(drop) - CE(full) in bits per step, episode-clustered bootstrap. Floor: the same group
  replaced by a within-split permutation of itself (X independent of Y given nothing; the model may still overfit it).
  Targets (all derivable from the logged data without simulator labels):
    act_logged  logged action a_t (awake steps, eps 0 / 0.1 / 0.25)        variants: none | hist (a_t-1..a_t-4)
    act_expert  a fresh sample from the expert's logits (awake, eps < 1)    variants: none | hist
    dmg_k       some step in [t, t+k) has a negative health part of the reward (r = A + 0.1 H, A = round(r + 0.25))
    ach_k       some step in [t, t+k) has an achievement part A > 0
    term_k      the episode ends in death within k steps
    food/drink/energy_k   sign of the HUD value change over k steps (windows inside the episode only)
    idm         a_t from (s_t, s_t+1): state features at t and t+1, with zombie/cow groups dropped at both times
  World-model targets have variants: act (a_t .. a_t+k-1 in C, action-conditioned) | marg (no actions, behaviour-marginal).
Part 2, self-predictability of entity encodings (Littwin et al. 2024: JEPA suppresses features with low regression
  coefficients onto their own future): one-step slope and R^2 of zombie / cow features at several granularities.
Usage: ana_signal_incentives.py DATA_NPZ OUT_JSON [MAX_TRAIN_ROWS]
"""
import json, sys, time, zlib
import numpy as np
import lightgbm as lgb

RES = {"water": 3, "stone": 4, "tree": 5, "coal": 8, "iron": 9, "diamond": 10, "table": 11, "furnace": 12, "lava": 14,
       "plant": 15, "ripe": 16}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
LN2 = np.log(2)


def episodes(ep):
    starts = np.r_[0, np.nonzero(np.diff(ep))[0] + 1]
    return starts, np.r_[starts[1:], len(ep)]


def shift(x, k, starts, ends, fill):
    """x[t + k] within the same episode (k may be negative), else fill."""
    out = np.full_like(x, fill)
    for s, e in zip(starts, ends):
        if e - s > abs(k):
            if k >= 0:
                out[s:e - k] = x[s + k:e]
            else:
                out[s - k:e] = x[s:e + k]
    return out


def window_any(flag, k, starts, ends):
    """any(flag[t .. t+k-1]) within the episode (truncated at its end)."""
    out = np.zeros(len(flag), bool)
    for s, e in zip(starts, ends):
        c = np.r_[0, np.cumsum(flag[s:e])]
        idx = np.arange(e - s)
        out[s:e] = (c[np.minimum(idx + k, e - s)] - c[idx]) > 0
    return out


def mob_band(rel, mask, lo, hi):
    """nearest visible mob with Manhattan distance in [lo, hi]: present, dr, dc, dist; and count."""
    vis = mask & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
    d = np.abs(rel).sum(-1).astype(float)
    ok = vis & (d >= lo) & (d <= hi)
    dd = np.where(ok, d, 99)
    j = dd.argmin(1); n = np.arange(len(j))
    pres = ok.any(1)
    f = np.stack([pres, np.where(pres, rel[n, j, 0], 0), np.where(pres, rel[n, j, 1], 0), np.where(pres, dd[n, j], 0),
                  ok.sum(1)], 1).astype(np.float32)
    return f


def features(D):
    v = D["view"].astype(np.int16)
    dirc = D["dir"].astype(int)
    dr = np.array([0, 0, 0, -1, 1])[dirc]; dc = np.array([0, -1, 1, 0, 0])[dirc]
    faced = v[np.arange(len(v)), 3 + dr, 4 + dc]
    neigh = np.stack([v[:, 3 + dr, 4 + dc] for dr, dc in DIRS.values()], 1)
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    terr = []
    for name, b in RES.items():
        m = v == b
        dist = np.where(m, np.abs(rr) + np.abs(cc), 99).reshape(len(v), -1)
        j = dist.argmin(1); n = np.arange(len(j)); pres = m.reshape(len(v), -1).any(1)
        cols = [np.where(pres, dist[n, j], 99)]
        if name in ("water", "tree", "stone", "table", "lava", "coal", "iron"):
            cols += [np.where(pres, rr.reshape(-1)[j], 0), np.where(pres, cc.reshape(-1)[j], 0)]
        terr += cols
    body = D["body"].astype(np.float32)
    G = {
        "zadj": mob_band(D["zombies_rel"], D["zombies_mask"], 1, 1),
        "znear": mob_band(D["zombies_rel"], D["zombies_mask"], 2, 4),
        "zfar": mob_band(D["zombies_rel"], D["zombies_mask"], 5, 99),
        "cow": mob_band(D["cows_rel"], D["cows_mask"], 1, 99),
        "skel": np.concatenate([mob_band(D["skeletons_rel"], D["skeletons_mask"], 1, 99),
                                mob_band(D["arrows_rel"], D["arrows_mask"], 1, 99)], 1),
        "health": body[:, :1], "food": body[:, 1:2], "drink": body[:, 2:3], "energy": body[:, 3:4],
        "sleep": D["sleep"].astype(np.float32)[:, None],
        "faced": faced.astype(np.float32)[:, None], "neigh": neigh.astype(np.float32),
        "terrain": np.stack(terr, 1).astype(np.float32), "inv": D["inv"].astype(np.float32),
        "light": D["light"].astype(np.float32)[:, None],
        "base": np.stack([dirc, D["ach"].astype(int)], 1).astype(np.float32),
    }
    CAT = {"faced": [0], "neigh": [0, 1, 2, 3], "base": [0]}
    return G, CAT


GROUPS = ["zadj", "znear", "zfar", "zall", "cow", "skel", "health", "food", "drink", "energy", "sleep", "faced", "neigh",
          "terrain", "inv", "light"]


def assemble(G, CAT, extra, drop=(), permute=(), rng=None, idx=None):
    cols, cats, names = [], [], []
    for g, x in list(G.items()) + list(extra.items()):
        if g in drop or (g in ("zadj", "znear", "zfar") and "zall" in drop):
            continue
        x = x[idx] if idx is not None else x
        if g in permute or (g in ("zadj", "znear", "zfar") and "zall" in permute):
            x = x[rng.permutation(len(x))]
        for j in range(x.shape[1]):
            if j in CAT.get(g, []):
                cats.append(len(cols))
            cols.append(x[:, j]); names.append(f"{g}{j}")
    return np.stack(cols, 1), cats


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, cats, ncls):
    p = {"objective": "binary" if ncls == 2 else "multiclass", "learning_rate": 0.1, "num_leaves": 63,
         "min_data_in_leaf": 100, "feature_fraction": 1.0, "deterministic": True, "seed": 0, "verbose": -1,
         "num_threads": 12, "max_cat_to_onehot": 32}
    if ncls > 2:
        p["num_class"] = ncls
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, categorical_feature=cats, reference=dtr)
    m = lgb.train(p, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    pr = m.predict(Xte, num_iteration=m.best_iteration)
    if ncls == 2:
        pr = np.stack([1 - pr, pr], 1)
    return -np.log(np.clip(pr[np.arange(len(yte)), yte], 1e-7, 1)) / LN2, m.best_iteration


def boot(delta, ep, n=1000, seed=0):
    u, inv = np.unique(ep, return_inverse=True)
    s = np.bincount(inv, delta); c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(u), (n, len(u)))
    m = s[b].sum(1) / c[b].sum(1)
    return [float(delta.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run_target(name, y, rows, G, CAT, extra, D, split, max_rows, present):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    rows = rows[(y[rows] >= 0)]
    parts = {}
    for s in ("tr", "va", "te"):
        r = rows[split[rows] == s]
        cap = max_rows if s == "tr" else max_rows // 2
        parts[s] = np.sort(rng.choice(r, min(cap, len(r)), replace=False))
    classes = np.unique(y[parts["tr"]])
    lut = -np.ones(int(y.max()) + 1, int); lut[classes] = np.arange(len(classes))
    yy = {s: lut[y[parts[s]]] for s in parts}
    keep = {s: yy[s] >= 0 for s in parts}
    parts = {s: parts[s][keep[s]] for s in parts}; yy = {s: yy[s][keep[s]] for s in parts}
    ncls = len(classes)
    ep_te = D["episode"][parts["te"]]
    out = {"n_train": int(len(parts["tr"])), "n_test": int(len(parts["te"])), "classes": int(ncls),
           "entropy_bits": None, "groups": {}}
    pri = np.bincount(yy["tr"], minlength=ncls) / len(yy["tr"])
    out["entropy_bits"] = float(-(pri * np.log2(np.clip(pri, 1e-12, 1))).sum())

    def ce(drop=(), permute=()):
        X = {s: assemble(G, CAT, extra, drop, permute, rng, parts[s]) for s in parts}
        l, it = fit_eval(X["tr"][0], yy["tr"], X["va"][0], yy["va"], X["te"][0], yy["te"], X["tr"][1], ncls)
        return l, it

    full, it = ce()
    out["ce_full_bits"] = float(full.mean()); out["iters_full"] = it
    for g in GROUPS:
        if g not in G and g != "zall":
            continue
        l, _ = ce(drop=(g,))
        d = l - full
        res = {"incentive_bits": boot(d, ep_te)}
        if g in present:
            m = present[g][parts["te"]]
            res["prevalence"] = float(m.mean())
            res["incentive_when_present"] = boot(d[m], ep_te[m]) if m.sum() > 50 else None
        if g in ("zall", "znear"):
            lp, _ = ce(permute=(g,))
            res["permutation_floor_bits"] = boot(l - lp, ep_te)     # drop vs permuted: ~0 if the model ignores noise
            res["permuted_minus_full_bits"] = boot(lp - full, ep_te)
        out["groups"][g] = res
        print(f"  {name:22s} {g:8s} {res['incentive_bits'][0]:+.4f} [{res['incentive_bits'][1]:+.4f}, "
              f"{res['incentive_bits'][2]:+.4f}]", flush=True)
    return out


def self_predictability(D, starts, ends):
    """Part 2: one-step slope / R^2 of entity encodings onto their own next value (same episode)."""
    nxt = lambda x, fill: shift(x, 1, starts, ends, fill)
    valid = nxt(np.ones(len(D["t"]), bool), False)
    awake = ~D["sleep"].astype(bool)
    pos = D["pos"].astype(int)
    out = {}
    for m in ("zombies", "cows"):
        rel, mask = D[m + "_rel"].astype(int), D[m + "_mask"]
        vis = mask & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        cam = np.zeros((len(rel), 7, 9), bool)
        for j in range(rel.shape[1]):
            i = np.nonzero(vis[:, j])[0]
            cam[i, rel[i, j, 0] + 3, rel[i, j, 1] + 4] = True
        res = {}
        # (a) camera-frame cell occupancy: same screen cell at t and t+1
        x = cam.reshape(len(cam), -1); y = nxt(x, False)
        sel = valid & awake
        res["cell_camera"] = slope_r2(x[sel].ravel(), y[sel].ravel())
        # (b) world-frame cell occupancy: the world cell occupied at t, still in view at t+1
        dpos = nxt(pos, 0) - pos
        rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
        r1 = rr[None] - dpos[:, 0, None, None]; c1 = cc[None] - dpos[:, 1, None, None]
        inside = (np.abs(r1) <= 3) & (np.abs(c1) <= 4)
        y_cam = nxt(cam, False)
        yw = np.zeros_like(cam)
        ii, aa, bb = np.nonzero(inside)
        yw[ii, aa, bb] = y_cam[ii, r1[ii, aa, bb] + 3, c1[ii, aa, bb] + 4]
        selw = (valid & awake)[:, None, None] & inside
        res["cell_world"] = slope_r2(cam[selw], yw[selw])
        # (c) entity-level: presence within 4, count in view, nearest offset (when present at t and t+1)
        d = np.where(vis, np.abs(rel).sum(-1), 99)
        near4 = (d <= 4).any(1); cnt = vis.sum(1).astype(float)
        res["present_within4"] = slope_r2(near4[sel].astype(float), nxt(near4, False)[sel].astype(float))
        res["count_in_view"] = slope_r2(cnt[sel], nxt(cnt, 0.0)[sel])
        j = d.argmin(1); n = np.arange(len(j)); pres = vis.any(1)
        off = rel[n, j].astype(float); off_n = nxt(off, 0.0); pres_n = nxt(pres, False)
        both = sel & pres & pres_n
        res["nearest_offset_row"] = slope_r2(off[both, 0], off_n[both, 0])
        res["nearest_offset_col"] = slope_r2(off[both, 1], off_n[both, 1])
        # (d) the same entity tracked by identity (slot), visible at t and t+1: offset from the player, and whether it
        #     moved in the world
        wpos = rel + pos[:, None, :]
        wn = nxt(wpos, 0); vn = nxt(vis, False); reln = nxt(rel, 0)
        bs = (sel[:, None] & vis & vn)
        res["same_entity_offset_row"] = slope_r2(rel[..., 0][bs].astype(float), reln[..., 0][bs].astype(float))
        res["same_entity_offset_col"] = slope_r2(rel[..., 1][bs].astype(float), reln[..., 1][bs].astype(float))
        res["same_entity_moved_frac"] = float((np.abs(wn - wpos).sum(-1)[bs] > 0).mean())
        out[m] = res
    return out


def slope_r2(x, y):
    x = x.astype(float); y = y.astype(float)
    vx = x.var()
    if vx == 0:
        return None
    b = ((x - x.mean()) * (y - y.mean())).mean() / vx
    r2 = 1 - ((y - y.mean() - b * (x - x.mean())) ** 2).mean() / max(y.var(), 1e-12)
    return {"slope": float(b), "r2": float(r2), "n": int(len(x)), "mean": float(x.mean())}


def main():
    data, out_json = sys.argv[1], sys.argv[2]
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    t0 = time.time()
    D = dict(np.load(data))
    ep = D["episode"]; starts, ends = episodes(ep)
    assert all((np.diff(D["t"][s:e]) == 1).all() for s, e in zip(starts[:50], ends[:50])), "episodes not contiguous"
    G, CAT = features(D)
    r = D["r"].astype(np.float64)
    A = np.round(r + 0.25); H = np.round(10 * (r - A))
    eps = D["eps"].astype(float); awake = ~D["sleep"].astype(bool)
    # split by episode: 60 / 20 / 20
    u = np.unique(ep); rng = np.random.default_rng(7); perm = rng.permutation(len(u))
    lab = np.empty(len(u), object); lab[perm[: int(.6 * len(u))]] = "tr"
    lab[perm[int(.6 * len(u)): int(.8 * len(u))]] = "va"; lab[perm[int(.8 * len(u)):]] = "te"
    split = lab[np.searchsorted(u, ep)]
    a = D["a"].astype(int)
    hist = np.stack([shift(a, -k, starts, ends, 17) for k in (1, 2, 3, 4)], 1).astype(np.float32)
    present = {"zadj": G["zadj"][:, 0] > 0, "znear": G["znear"][:, 0] > 0, "zfar": G["zfar"][:, 0] > 0,
               "zall": (G["zadj"][:, 0] + G["znear"][:, 0] + G["zfar"][:, 0]) > 0, "cow": G["cow"][:, 0] > 0,
               "skel": (G["skel"][:, 0] + G["skel"][:, 5]) > 0}
    for k in ("zadj", "znear", "zfar", "cow", "skel"):
        CAT[k] = []
    RES_ = {"rows": int(len(ep)), "episodes": int(len(u)),
            "steps_by_eps": {str(e): int((eps == e).sum()) for e in np.unique(eps)},
            "sleep_frac": float(D["sleep"].mean()), "targets": {}}
    all_rows = np.arange(len(ep))
    dead = D["dead"] & D["done"]
    jobs = []
    for k in (1, 4, 16):
        fut = {"fut": np.stack([shift(a, j, starts, ends, 17) for j in range(k)], 1).astype(np.float32)}
        CATf = {"fut": list(range(k))}
        for tname, y in (("dmg", window_any(H < 0, k, starts, ends)), ("ach", window_any(A > 0, k, starts, ends)),
                         ("term", window_any(dead, k, starts, ends))):
            for var, extra, cat in (("act", fut, CATf), ("marg", {}, {})):
                jobs.append((f"{tname}_{k}_{var}", y.astype(int), all_rows, extra, cat))
    for k in (16,):
        fut = {"fut": np.stack([shift(a, j, starts, ends, 17) for j in range(k)], 1).astype(np.float32)}
        for hi, hname in ((1, "food"), (2, "drink"), (3, "energy")):
            b0 = D["body"][:, hi].astype(int); bk = shift(b0, k, starts, ends, -99)
            y = np.where(bk == -99, -1, np.sign(bk - b0) + 1)
            for var, extra, cat in (("act", fut, {"fut": list(range(k))}), ("marg", {}, {})):
                jobs.append((f"{hname}_{k}_{var}", y.astype(int), all_rows, extra, cat))
    act_rows = np.nonzero(awake & (eps <= 0.25))[0]
    exp_rows = np.nonzero(awake & (eps < 1))[0]
    lg = D["logits"].astype(np.float64); pz = np.exp(lg - lg.max(1, keepdims=True)); pz /= pz.sum(1, keepdims=True)
    a_exp = (np.random.default_rng(11).random(len(pz))[:, None] > np.cumsum(pz, 1)).sum(1)
    for var, extra, cat in (("none", {}, {}), ("hist", {"hist": hist}, {"hist": [0, 1, 2, 3]})):
        jobs.append((f"act_logged_{var}", a, act_rows, extra, cat))
        jobs.append((f"act_expert_{var}", a_exp, exp_rows, extra, cat))
    # inverse dynamics (one-step, and multi-step as in ACRO / AC-State): features of every group at t+k as extra,
    # dropped together with their group
    for kk, nm in ((1, "idm"), (4, "idm_4")):
        nxtG = {g: shift(x, kk, starts, ends, -1) for g, x in G.items() if g != "base"}
        idm_rows = np.nonzero(shift(np.ones(len(ep), bool), kk, starts, ends, False))[0]
        jobs.append((nm, a, idm_rows, {"next_" + g: x for g, x in nxtG.items()}, {"next_faced": [0], "next_neigh": [0, 1, 2, 3]}))
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    for name, y, rows, extra, cat in jobs:
        if only and name not in only:
            continue
        t1 = time.time()
        CATx = dict(CAT); CATx.update(cat)
        if name.startswith("idm"):
            res = run_target_idm(name, y, rows, G, CATx, extra, D, split, max_rows, present)
        else:
            res = run_target(name, y, rows, G, CATx, extra, D, split, max_rows, present)
        res["seconds"] = time.time() - t1
        RES_["targets"][name] = res
        json.dump(RES_, open(out_json, "w"), indent=1)
        print(f"{name}: CE full {res['ce_full_bits']:.4f} bits, H {res['entropy_bits']:.4f}, {res['seconds']:.0f}s", flush=True)
    sub = np.isin(ep, np.random.default_rng(3).choice(u, min(len(u), 1500), replace=False))
    Ds = {k: v[sub] for k, v in D.items()}; s2, e2 = episodes(Ds["episode"])
    RES_["self_predictability"] = self_predictability(Ds, s2, e2)
    RES_["seconds"] = time.time() - t0
    json.dump(RES_, open(out_json, "w"), indent=1)
    print(json.dumps(RES_["self_predictability"], indent=1))


def run_target_idm(name, y, rows, G, CAT, extra, D, split, max_rows, present):
    """Inverse dynamics: dropping a group drops it at t AND t+1 (extra holds next_<group>)."""
    G2 = dict(G)
    for g, x in extra.items():
        G2[g] = x
    global GROUPS
    saved = GROUPS
    try:
        # map each group to (group, next_group) by wrapping assemble's drop through a combined group dict
        comb = {}
        for g in G:
            if g == "base":
                comb[g] = G[g]
            else:
                comb[g] = np.concatenate([G[g], extra["next_" + g]], 1)
        CAT2 = {}
        for g, idx in CAT.items():
            if g in comb:
                CAT2[g] = list(idx) + [i + G[g].shape[1] for i in idx] if g in ("faced", "neigh") else list(idx)
        return run_target(name, y, rows, comb, CAT2, {}, D, split, max_rows, present)
    finally:
        GROUPS = saved


if __name__ == "__main__":
    main()
