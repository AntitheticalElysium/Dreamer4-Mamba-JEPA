"""P1 (2026-10-25): how much entity information survives a degraded (blurred) periphery in Craftax-Classic frames?
User question: if the periphery saw a low-resolution frame and only the fovea saw full resolution, could attention still
know where to look (e.g. find a zombie)? A fact about the renderer and our data; nothing is trained except the
estimators. Simulator state labels cells for evaluation only.

CLAIM: entity presence (zombie, cow, skeleton, arrow) in a 7x7-pixel map cell stays decodable after area-pooling the cell
  to r x r pixels (r = 1, 2, 3), so a degraded periphery could still direct attention.
COMPETING EXPLANATIONS: (a) decodability comes only from neighbourhood context (e.g. zombies on grass at night), not
  from the cell; (b) at night, renderer static masks pooled cells; (c) the estimator overfits.
CONTROLS: r = 7 (full resolution) as the ceiling; cell-only vs cell + 8 neighbours; day vs night strata; a motion cue (the
  same screen cell, pooled, one step earlier) evaluated on camera-static steps only (a scroll changes every cell);
  episode-level train / valid / test split; prevalence as the floor.
OUTCOMES: a resolution at which zombie presence is not reliably decodable (day or night), with or without the motion
  cue, rules that resolution out as a periphery for Craftax: attention could not be drawn to zombies by appearance.
  High decodability leaves degradation feasible; it does not show a learned system would use it.
Usage (project venv): eda_periphery_degradation.py OUT_NPZ N_EPISODES STRIDE
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

MOBS = ("zombies", "cows", "skeletons", "arrows")
SPLITS = {1: [7], 2: [4, 3], 3: [2, 3, 2], 7: [1] * 7}


def pool(cells, r):
    """cells [..., 7, 7, 3] uint8 -> area-pooled [..., r, r, 3] float32 over a fixed partition of the 7 pixels."""
    edges = np.cumsum([0] + SPLITS[r])
    out = np.zeros(cells.shape[:-3] + (r, r, 3), np.float32)
    c = cells.astype(np.float32)
    for i in range(r):
        for j in range(r):
            out[..., i, j, :] = c[..., edges[i]:edges[i + 1], edges[j]:edges[j + 1], :].mean((-3, -2))
    return out


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "light": s.light_level, "sleep": s.is_sleeping, "map": s.map.astype(jnp.int8),
                     **{f"{m}_p": getattr(s, m).position for m in MOBS}, **{f"{m}_m": getattr(s, m).mask for m in MOBS}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


def main():
    out, n_eps, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261025)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    feats = {r: [] for r in SPLITS}; prev1 = []; lab = {m: [] for m in MOBS}; meta = []
    for e, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        frames = f["observations"].numpy()                                   # [T+1, 63, 63, 3]
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        cells = frames[:, :49, :, :].reshape(T + 1, 7, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5)   # [T+1, 7, 9, 7, 7, 3]
        pos = R["pos"][:T + 1].astype(int)
        for t in range(1, T + 1, stride):
            occ = {m: np.zeros((7, 9), bool) for m in MOBS}
            for m in MOBS:
                rel = R[f"{m}_p"][t].astype(int) - pos[t]; msk = R[f"{m}_m"][t]
                for j in range(len(msk)):
                    if msk[j] and abs(rel[j, 0]) <= 3 and abs(rel[j, 1]) <= 4:
                        occ[m][rel[j, 0] + 3, rel[j, 1] + 4] = True
            static_cam = bool((pos[t] == pos[t - 1]).all())
            for r in SPLITS:
                feats[r].append(pool(cells[t], r).reshape(63, -1))
            prev1.append(pool(cells[t - 1], 1).reshape(63, -1))
            for m in MOBS:
                lab[m].append(occ[m].reshape(63))
            meta.append(np.stack([np.full(63, e), np.full(63, float(R["light"][t])), np.full(63, float(R["sleep"][t])),
                                  np.full(63, float(static_cam)), np.arange(63)], 1))
        print(f"episode {e + 1}/{len(picks)} T={T}", flush=True)
    np.savez_compressed(out, **{f"r{r}": np.concatenate(v).astype(np.float16) for r, v in feats.items()},
                        prev_r1=np.concatenate(prev1).astype(np.float16),
                        **{f"y_{m}": np.concatenate(v) for m, v in lab.items()}, meta=np.concatenate(meta).astype(np.float32))
    print("saved", out)


if __name__ == "__main__":
    main()
