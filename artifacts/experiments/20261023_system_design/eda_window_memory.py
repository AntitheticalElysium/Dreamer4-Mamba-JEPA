"""M2 (2026-10-23): how much memory does a COHERENT 16-step rollout need? Simulator replay of TRAIN-split support-v2
episodes; evaluation-only state; nothing trained.

For every 16-step window starting at frame s (stride 8), the content that enters view during the window is classified by
when it was last seen BEFORE the window started (never / age <= 16 / <= 64 / <= 256 / > 256 steps):
  terrain: map cells entering view (world positions not in view at the previous frame);
  entities: mobs (zombie, cow, skeleton) becoming visible during the window, by entity identity.
A world model that only remembers the last L frames before the rollout must regenerate everything older than L; the
shares below are the fraction of re-entering content it would get from memory vs have to invent. Also reported: the
distance (Manhattan) of the re-entering terrain from the player's position at window start.
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

BINS = (16, 64, 256)


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, **{f"{m}_p": getattr(s, m).position for m in ("zombies", "cows", "skeletons")},
                     **{f"{m}_m": getattr(s, m).mask for m in ("zombies", "cows", "skeletons")}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


def bucket(age):
    if age is None:
        return "never"
    for b in BINS:
        if age <= b:
            return f"le{b}"
    return f"gt{BINS[-1]}"


def main():
    n_eps, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(20261030)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 300])[:3].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    ter = {}; ent = {}; dist = []
    for sh, slot in picks:
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"][:T + 1].astype(int)
        # last-seen time for every world cell, per frame (terrain) and per entity (identity while alive)
        last_cell = {}; cell_hist = []                                   # snapshot dicts are too big; use arrays
        seen_t = np.full((64, 64), -1, np.int64); seen_at = np.zeros((T + 1, 64, 64), np.int32) - 1
        for t in range(T + 1):
            r0, c0 = pos[t, 0] - 3, pos[t, 1] - 4
            rs, re, cs, ce = max(r0, 0), min(r0 + 7, 64), max(c0, 0), min(c0 + 9, 64)
            seen_at[t] = seen_t                                          # state BEFORE frame t's view
            seen_t[rs:re, cs:ce] = t
        V = {}
        for m in ("zombies", "cows", "skeletons"):
            P = R[f"{m}_p"][:T + 1].astype(int); M = R[f"{m}_m"][:T + 1]
            d = P - pos[:, None, :]
            V[m] = (M & (np.abs(d[..., 0]) <= 3) & (np.abs(d[..., 1]) <= 4), M)
        for s in range(16, T - 16, 8):
            for t in range(s + 1, s + 17):
                prev = {(int(pos[t - 1, 0] + r - 3), int(pos[t - 1, 1] + c - 4)) for r in range(7) for c in range(9)}
                for r in range(7):
                    for c in range(9):
                        w = (int(pos[t, 0] + r - 3), int(pos[t, 1] + c - 4))
                        if w in prev or not (0 <= w[0] < 64 and 0 <= w[1] < 64):
                            continue
                        last = int(seen_at[s][w])                         # last seen before the window started
                        k = bucket(None if last < 0 else s - last)
                        ter[k] = ter.get(k, 0) + 1
                        if last >= 0:
                            dist.append(abs(w[0] - pos[s, 0]) + abs(w[1] - pos[s, 1]))
            for m, (vis, alive) in V.items():
                for j in range(vis.shape[1]):
                    win = vis[s + 1:s + 17, j]
                    if not win.any() or vis[s, j]:
                        continue                                          # visible at window start: not a re-entry
                    t1 = s + 1 + int(np.argmax(win))
                    born = np.nonzero(~alive[:t1 + 1, j])[0]
                    born = born[-1] + 1 if len(born) else 0
                    before = np.nonzero(vis[born:s + 1, j])[0]
                    k = bucket(None if len(before) == 0 else s - (born + before[-1]))
                    ent.setdefault(m, {}); ent[m][k] = ent[m].get(k, 0) + 1
    norm = lambda d: {k: round(v / sum(d.values()), 4) for k, v in sorted(d.items())}
    res = {"episodes": len(picks), "terrain_entering_by_age": norm(ter), "terrain_n": sum(ter.values()),
           "entities_appearing_by_age": {m: norm(v) for m, v in ent.items()},
           "entities_n": {m: sum(v.values()) for m, v in ent.items()},
           "reentering_terrain_distance_from_window_start": {"p50": float(np.median(dist)), "p90": float(np.percentile(dist, 90)),
                                                             "max": int(np.max(dist))}}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
