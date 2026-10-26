"""A9 (2026-10-25): can a label-free change target be made reliable at night?
A5: the aligned pixel residual marks independent changes by day (AUC 0.978) but not at night (0.87). P1: entity presence
is decodable at night from a full-resolution cell (AP 0.97 zombies), so the evidence exists; the fixed statistic is
the weak link. Candidate label-free recipe: label day pairs with the clean day residual (threshold by Otsu on log
residuals, no simulator labels), train a small detector on those day pairs after GENERIC night-like corruption
(darkening, desaturation, uniform grey noise blended at a random strength, independently per frame), apply it to real
night pairs. Simulator labels are used for evaluation only (and for the ceiling control).

CLAIM: the detector trained on corrupted day pairs with label-free targets beats the raw residual on real night pairs.
COMPETING EXPLANATIONS: (a) the corruption is unlike real night static, so nothing transfers; (b) the residual is
  already near the pixel ceiling at night.
CONTROLS: raw residual (A5 statistic); detector trained on clean day pairs only; ceiling = detector trained on real
  night pairs with simulator labels (episode split); real day pairs as a sanity check.
OUTCOMES: corrupted-day detector close to the ceiling -> a label-free night target is feasible; close to the raw
  residual -> the night change target stays weak and night entities must be carried by tracking / memory instead.
Usage: (project venv) eda_change_night.py build OUT_NPZ N_EPISODES STRIDE ; (scratch venv) eda_change_night.py fit NPZ OUT_JSON
"""
import json, os, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")
SHIFTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def cells(frame):
    return frame[:49].astype(np.float32).reshape(7, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4)


def est_shift(c0, c1):
    cost = []
    for dr, dc in SHIFTS:
        r0, r1 = max(0, -dr), min(7, 7 - dr); q0, q1 = max(0, -dc), min(9, 9 - dc)
        cost.append(np.abs(c0[r0:r1, q0:q1] - c1[r0 + dr:r1 + dr, q0 + dc:q1 + dc]).mean())
    return SHIFTS[int(np.argmin(cost))]


def pool3(c):
    e = [0, 2, 5, 7]
    return np.stack([np.stack([c[..., e[i]:e[i + 1], e[j]:e[j + 1], :].mean((-3, -2)) for j in range(3)], -2)
                     for i in range(3)], -3)                                  # [..., 3, 3, 3]


def feats(a, b):
    """a: target cell at t+1, b: aligned source cell at t; [N, 7, 7, 3] each -> [N, F]."""
    pa, pb = pool3(a).reshape(len(a), -1), pool3(b).reshape(len(b), -1)
    d = np.abs(a - b).reshape(len(a), -1)
    hf = lambda x: (x - x.mean((1, 2), keepdims=True)).std((1, 2, 3))
    return np.concatenate([pa, pb, np.abs(pa - pb), d.mean(1, keepdims=True), np.median(d, 1, keepdims=True),
                           (d > 40).mean(1, keepdims=True), (d > 80).mean(1, keepdims=True),
                           hf(a)[:, None], hf(b)[:, None]], 1).astype(np.float32)


def corrupt(frame, rng):
    """Generic night-like corruption of a [63, 63, 3] uint8 frame (map rows only)."""
    x = frame.astype(np.float32)
    b = rng.uniform(0.35, 1.0); s = rng.uniform(0.0, 1.0); m = rng.uniform(0.0, 0.8)
    lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = (s * lum + (1 - s) * x) * b
    noise = rng.uniform(32, 127, x.shape[:2])[..., None]
    x[:49] = (1 - m) * x[:49] + m * noise[:49]
    return np.clip(x, 0, 255)


def build(out, n_eps, stride):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp
    sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
    import replay as RP
    rng = np.random.default_rng(20261027)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps_ = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps_) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "light": s.light_level, "map": s.map.astype(jnp.int8),
                     **{f"{m}_p": getattr(s, m).position for m in KINDS}, **{f"{m}_m": getattr(s, m).mask for m in KINDS}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    run = jax.jit(run)
    F = {"real": [], "corrupt": []}; Y = {"real": [], "corrupt": []}; G = {"real": [], "corrupt": []}
    NIGHT = {"real": [], "corrupt": []}; LF = {"real": [], "corrupt": []}
    for e, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        frames = f["observations"].numpy()
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"].astype(int); M = R["map"]; H = M.shape[1]
        for t in range(0, T, stride):
            night = float(R["light"][t + 1]) < 0.5
            true_shift = tuple(-(pos[t + 1] - pos[t]))
            occ0, occ1 = {}, {}
            for m in KINDS:
                P_, M_ = R[f"{m}_p"].astype(int), R[f"{m}_m"]
                for j in range(P_.shape[1]):
                    if M_[t, j]:
                        occ0[tuple(P_[t, j])] = (m, j)
                    if M_[t + 1, j]:
                        occ1[tuple(P_[t + 1, j])] = (m, j)
            variants = [("real", frames[t], frames[t + 1])]
            if not night:
                variants.append(("corrupt", corrupt(frames[t], rng), corrupt(frames[t + 1], rng)))
            clean_c0, clean_c1 = cells(frames[t]), cells(frames[t + 1])
            clean_shift = est_shift(clean_c0, clean_c1)
            for name, f0, f1 in variants:
                c0, c1 = cells(f0), cells(f1)
                sh_ = est_shift(c0, c1)
                A, B, ys, lf = [], [], [], []
                for rr in range(7):
                    for qq in range(9):
                        if (rr, qq) == (3, 4):
                            continue
                        sr, sq = rr - true_shift[0], qq - true_shift[1]
                        if not (0 <= sr < 7 and 0 <= sq < 9) or (sr, sq) == (3, 4):
                            continue
                        w1 = (pos[t + 1][0] + rr - 3, pos[t + 1][1] + qq - 4)
                        if not (0 <= w1[0] < H and 0 <= w1[1] < H):
                            continue
                        ar, aq = rr - sh_[0], qq - sh_[1]
                        cr, cq = rr - clean_shift[0], qq - clean_shift[1]
                        if not (0 <= ar < 7 and 0 <= aq < 9 and 0 <= cr < 7 and 0 <= cq < 9):
                            continue
                        e0, e1 = occ0.get(w1), occ1.get(w1)
                        ys.append(int(M[t][w1] != M[t + 1][w1] or e0 != e1))
                        A.append(c1[rr, qq]); B.append(c0[ar, aq])
                        lf.append(float(np.abs(clean_c1[rr, qq] - clean_c0[cr, cq]).mean()))   # clean day residual
                if A:
                    F[name].append(feats(np.stack(A), np.stack(B))); Y[name].extend(ys); LF[name].extend(lf)
                    G[name].extend([e] * len(ys)); NIGHT[name].extend([night] * len(ys))
        print(f"episode {e + 1}/{len(picks)} T={T}", flush=True)
    np.savez_compressed(out, **{f"{k}_X": np.concatenate(F[k]) for k in F}, **{f"{k}_y": np.array(Y[k], np.int8) for k in Y},
                        **{f"{k}_g": np.array(G[k], np.int32) for k in G}, **{f"{k}_night": np.array(NIGHT[k], bool) for k in NIGHT},
                        **{f"{k}_lf": np.array(LF[k], np.float32) for k in LF})
    print("saved", out)


def otsu(v):
    v = np.log1p(v); h, edges = np.histogram(v, 256); c = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(h); w2 = w[-1] - w; m1 = np.cumsum(h * c) / np.maximum(w, 1); m2 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w2, 1)
    return float(np.expm1(c[np.argmax(w * w2 * (m1 - m2) ** 2)]))


def fit(npz, out_json):
    import lightgbm as lgb
    D = np.load(npz)
    def auc(pos, neg):
        s = np.concatenate([pos, neg]); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
        r = (np.bincount(inv, r) / cnt)[inv]
        return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    Xr, yr, gr, nr, lfr = D["real_X"], D["real_y"], D["real_g"], D["real_night"], D["real_lf"]
    Xc, gc, lfc = D["corrupt_X"], D["corrupt_g"], D["corrupt_lf"]
    eps_ = np.unique(gr); rng = np.random.default_rng(0); perm = rng.permutation(eps_)
    tr_e, te_e = perm[: int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    tau = otsu(lfr[(~nr) & np.isin(gr, tr_e)])                                # label-free day threshold
    p = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1, "num_threads": 12, "seed": 0}

    def train(X, y):
        n = len(y); idx = np.random.default_rng(1).permutation(n); v = idx[: n // 10]; t = idx[n // 10:]
        dt = lgb.Dataset(X[t], y[t]); dv = lgb.Dataset(X[v], y[v], reference=dt)
        return lgb.train(p, dt, 400, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])

    def score(s, y):
        thr = np.percentile(s[y == 0], 95)
        return {"auc": auc(s[y == 1], s[y == 0]), "recall_at_5pct_fp": float((s[y == 1] > thr).mean()), "n_pos": int(y.sum()), "n": int(len(y))}

    res = {"label_free_day_threshold": tau}
    mc = np.isin(gc, tr_e)
    corrupt_model = train(Xc[mc], (lfc[mc] > tau).astype(int))
    md = (~nr) & np.isin(gr, tr_e)
    clean_model = train(Xr[md], (lfr[md] > tau).astype(int))
    mn = nr & np.isin(gr, tr_e)
    ceiling = train(Xr[mn], yr[mn])
    for split_name, night in (("day", False), ("night", True)):
        te = (nr == night) & np.isin(gr, te_e)
        y = yr[te]
        res[split_name] = {"raw_residual": score(Xr[te][:, 81], y),
                           "detector_clean_day_labelfree": score(clean_model.predict(Xr[te]), y),
                           "detector_corrupted_day_labelfree": score(corrupt_model.predict(Xr[te]), y)}
        if night:
            res[split_name]["ceiling_night_sim_labels"] = score(ceiling.predict(Xr[te]), y)
        print(split_name, res[split_name], flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        fit(sys.argv[2], sys.argv[3])
