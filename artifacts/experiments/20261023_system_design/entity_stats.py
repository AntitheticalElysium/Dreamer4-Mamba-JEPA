"""Entity statistics in Craftax-Classic: how many entities, how often they enter / leave view, and whether the ones entering
are entities already seen (identity memory) or new. (2026-10-23, design measurement; evaluation-only simulator state.)

TRAIN-split support-v2 episodes replayed exactly. Mob arrays keep per-index identity while the mask stays on; an index whose
mask goes off and on again is treated as a new entity. Visible = inside the 7 x 9 map view around the player.
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
TYPES = ("zombies", "cows", "skeletons", "arrows")


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, **{f"{m}_p": getattr(s, m).position for m in TYPES},
                     **{f"{m}_m": getattr(s, m).mask for m in TYPES}}

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
    n_eps, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(20261025)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        cand = [i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 64]
        for i in rng.permutation(cand)[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    vis_hist = {m: np.zeros(12, np.int64) for m in TYPES}; tot_hist = np.zeros(16, np.int64)
    entries = {m: {"first_sight": 0, "re_entry": 0, "spawn_in_view": 0} for m in TYPES}
    exits = {m: {"left_view_alive": 0, "despawned_or_died_in_view": 0} for m in TYPES}
    gaps = {m: [] for m in TYPES}; n_steps = 0; near_offscreen = []
    for sh, slot in picks:
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        for m in TYPES:
            P, M, pp = R[f"{m}_p"][:T + 1], R[f"{m}_m"][:T + 1], R["pos"][:T + 1]
            d = P - pp[:, None, :]
            V = M & (np.abs(d[..., 0]) <= 3) & (np.abs(d[..., 1]) <= 4)
            for t in range(T + 1):
                vis_hist[m][min(int(V[t].sum()), 11)] += 1
            for j in range(M.shape[1]):
                seen = False; last_exit = None
                for t in range(1, T + 1):
                    born = M[t, j] and not M[t - 1, j]
                    if born:
                        seen = False; last_exit = None
                        if V[t, j]:
                            entries[m]["spawn_in_view"] += 1; seen = True
                        continue
                    if V[t, j] and not V[t - 1, j] and M[t - 1, j]:
                        if seen:
                            entries[m]["re_entry"] += 1
                            if last_exit is not None: gaps[m].append(t - last_exit)
                        else:
                            entries[m]["first_sight"] += 1
                        seen = True
                    if V[t - 1, j] and not V[t, j]:
                        if M[t, j]:
                            exits[m]["left_view_alive"] += 1; last_exit = t
                        else:
                            exits[m]["despawned_or_died_in_view"] += 1
                    if t == 1 and V[0, j]:
                        seen = True
        Vz = None
        allv = sum((R[f"{m}_m"][:T + 1] & (np.abs(R[f"{m}_p"][:T + 1] - R["pos"][:T + 1, None, :])[..., 0] <= 3)
                    & (np.abs(R[f"{m}_p"][:T + 1] - R["pos"][:T + 1, None, :])[..., 1] <= 4)).sum(1) for m in TYPES[:3])
        for t in range(T + 1): tot_hist[min(int(allv[t]), 15)] += 1
        for m in ("zombies", "cows"):
            P, M, pp = R[f"{m}_p"][:T + 1], R[f"{m}_m"][:T + 1], R["pos"][:T + 1]
            d = np.abs(P - pp[:, None, :])
            off = M & ~((d[..., 0] <= 3) & (d[..., 1] <= 4)) & (d.sum(-1) <= 8)
            near_offscreen.append(off.sum(1))
        n_steps += T + 1
    no = np.concatenate(near_offscreen)
    res = {"episodes": len(picks), "frames": n_steps,
           "visible_per_frame": {m: {"mean": float((np.arange(12) * vis_hist[m]).sum() / vis_hist[m].sum()),
                                     "share_frames_ge1": float(vis_hist[m][1:].sum() / vis_hist[m].sum()),
                                     "hist": vis_hist[m].tolist()} for m in TYPES},
           "visible_mobs_total_hist(zombie+cow+skeleton)": tot_hist.tolist(),
           "entries": entries, "exits": exits,
           "re_entry_gap_steps": {m: ({"n": len(g), "p50": float(np.median(g)), "p90": float(np.percentile(g, 90))} if g else None) for m, g in gaps.items()},
           "alive_offscreen_within_manhattan8_per_frame_zombie_plus_cow_sum_over_types": float(no.mean())}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
