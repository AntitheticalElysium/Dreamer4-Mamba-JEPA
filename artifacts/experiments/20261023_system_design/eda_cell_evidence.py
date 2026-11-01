"""A10 (2026-10-25): what does a single full-resolution cell reveal, by distance from the player and light level?
Perception pass: the periphery encoder is cell-local (Dedieu et al. 2025 tokenise 7x7 tiles independently). Night
static blends uniform grey noise with strength 2 (0.5 - light) x (1 - exp(-2 r^2)) (renderer), i.e. weakest at the
player and strongest at the view edge. This measures, per ring of distance, how well terrain type and entity presence
can be read from one cell. Simulator state labels cells for evaluation only; nothing in the world model is trained.

CLAIM: by day every ring is fully readable; at night the outer rings lose terrain and entity evidence, so the
  periphery is partly blind there regardless of fovea placement (a glimpse sees the same pixels).
COMPETING EXPLANATION: static leaves enough structure (sprites are high-contrast) that a cell classifier still reads
  outer rings at night. CONTROLS: Chebyshev ring 1..4 around the player; light bins day >= 0.5, dusk 0.25-0.5,
  night < 0.25; episode-level split; terrain accuracy vs the majority-class floor; entity AP vs prevalence.
OUTCOMES: low outer-ring night readability -> store traces need reliability, first-seen terrain at night edges comes
  from the prior, and the belief must carry entities through the dark band by tracking; high readability -> the
  encoder alone can carry night perception.
Usage: (project venv) eda_cell_evidence.py build OUT_NPZ N_EPISODES STRIDE ; (scratch venv) eda_cell_evidence.py fit NPZ OUT_JSON
"""
import json, os, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")


def build(out, n_eps, stride):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp
    sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
    import replay as RP
    rng = np.random.default_rng(20261028)
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
    X, BT, ENT, RING, LIGHT, EP = [], [], [], [], [], []
    rr, qq = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
    ring = np.maximum(np.abs(rr - 3), np.abs(qq - 4)).reshape(-1)
    for e, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        frames = f["observations"].numpy()
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"].astype(int); H = R["map"].shape[1]
        for t in range(0, T + 1, stride):
            cells = frames[t][:49].reshape(7, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4).reshape(63, -1)
            wr = pos[t][0] + rr.reshape(-1) - 3; wc = pos[t][1] + qq.reshape(-1) - 4
            inb = (wr >= 0) & (wr < H) & (wc >= 0) & (wc < H)
            bt = np.full(63, -1); bt[inb] = R["map"][t][wr[inb], wc[inb]]
            occ = np.zeros(63, bool)
            for m in KINDS:
                P_, M_ = R[f"{m}_p"].astype(int), R[f"{m}_m"]
                for j in range(P_.shape[1]):
                    if M_[t, j]:
                        d = P_[t, j] - pos[t]
                        if abs(d[0]) <= 3 and abs(d[1]) <= 4:
                            occ[(d[0] + 3) * 9 + d[1] + 4] = True
            keep = inb & (ring > 0)
            X.append(cells[keep]); BT.append(bt[keep]); ENT.append(occ[keep]); RING.append(ring[keep])
            LIGHT.append(np.full(keep.sum(), float(R["light"][t]), np.float32)); EP.append(np.full(keep.sum(), e))
        print(f"episode {e + 1}/{len(picks)} T={T}", flush=True)
    np.savez_compressed(out, X=np.concatenate(X).astype(np.uint8), bt=np.concatenate(BT).astype(np.int16),
                        ent=np.concatenate(ENT), ring=np.concatenate(RING).astype(np.int8),
                        light=np.concatenate(LIGHT), ep=np.concatenate(EP).astype(np.int32))
    print("saved", out)


def fit(npz, out_json):
    import lightgbm as lgb
    D = np.load(npz)
    X, bt, ent, ring, light, ep = D["X"].astype(np.float32), D["bt"], D["ent"], D["ring"], D["light"], D["ep"]
    u = np.unique(ep); perm = np.random.default_rng(0).permutation(u)
    tr = np.isin(ep, perm[: int(0.7 * len(u))]); te = ~tr
    lbin = np.where(light >= 0.5, 0, np.where(light >= 0.25, 1, 2)); names = ("day", "dusk", "night")
    p = {"learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1, "num_threads": 12, "seed": 0}
    # terrain type of cells without an entity (entities cover the block)
    classes, y = np.unique(bt, return_inverse=True)
    t_mask = ~ent
    sub = np.nonzero(tr & t_mask)[0]; sub = np.random.default_rng(1).choice(sub, min(len(sub), 400000), replace=False)
    # regularised: unregularised multiclass boosting blew up on rare block types (first fit, 10-25; max |raw| checked)
    mt = lgb.train({**p, "objective": "multiclass", "num_class": len(classes), "lambda_l2": 1.0,
                    "min_sum_hessian_in_leaf": 1.0, "learning_rate": 0.05}, lgb.Dataset(X[sub], y[sub]), 150)
    pt = mt.predict(X[te & t_mask]).argmax(1); yt = y[te & t_mask]
    sub2 = np.nonzero(tr)[0]; sub2 = np.random.default_rng(2).choice(sub2, min(len(sub2), 400000), replace=False)
    me = lgb.train({**p, "objective": "binary"}, lgb.Dataset(X[sub2], ent[sub2].astype(int)), 200)
    pe = me.predict(X[te]); ye = ent[te]

    def ap(y, s):
        o = np.argsort(-s, kind="stable"); y = y[o]; tp = np.cumsum(y)
        return float(np.sum(tp[y == 1] / (np.nonzero(y == 1)[0] + 1)) / max(y.sum(), 1)) if y.sum() else None
    res = {"terrain_classes": int(len(classes))}
    rt, lt = ring[te & t_mask], lbin[te & t_mask]; re_, le = ring[te], lbin[te]
    for b in range(3):
        for r in range(1, 5):
            s = (rt == r) & (lt == b)
            floor = float(np.bincount(yt[s]).max() / s.sum()) if s.any() else None
            s2 = (re_ == r) & (le == b)
            res[f"{names[b]}|ring{r}"] = {"terrain_acc": float((pt[s] == yt[s]).mean()) if s.any() else None, "terrain_floor": floor,
                                          "n_terrain": int(s.sum()), "entity_ap": ap(ye[s2], pe[s2]),
                                          "entity_prevalence": float(ye[s2].mean()) if s2.any() else None, "n": int(s2.sum())}
            print(names[b], r, res[f"{names[b]}|ring{r}"], flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        fit(sys.argv[2], sys.argv[3])
