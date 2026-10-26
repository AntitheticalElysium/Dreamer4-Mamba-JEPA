"""Entities vs places (2026-10-23): how tied are Craftax entities to places, and what would object records need to carry?
Simulator replay of TRAIN-split support-v2 episodes; evaluation-only state; nothing trained.

E-a Tracked-set size: per frame, entities (zombie, cow, skeleton) that are alive, within Manhattan 12 of the player, and
    were visible within the last 64 steps (what an object-file memory would have to hold), visible + recently seen.
E-b Identity ambiguity at re-entry: when an entity re-enters view (seen before, alive throughout), the number of OTHER
    same-type entities that are out of view, were last seen within 64 steps, and whose last-seen position is within the
    re-entering entity's displacement + 2 cells of the re-entry position (plausible confusions).
E-c Habitat: P(type | terrain under the entity) for visible entities (are entities predictable from place content?).
E-d Hidden per-entity state at re-entry: fraction of re-entering zombies / skeletons with attack cooldown > 0, and of
    re-entering entities with health below the type's maximum observed (state a place-keyed memory cannot carry when the
    entity moves).
"""
import json, os, sys
from collections import Counter, defaultdict
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
TYPES = ("zombies", "cows", "skeletons")


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "map": s.map.astype(jnp.int8),
                     **{f"{m}_{f}": getattr(getattr(s, m), f) for m in TYPES for f in ("position", "mask", "health", "attack_cooldown")}}

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
    rng = np.random.default_rng(20261102)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:3].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    tracked = Counter(); visible_n = Counter(); amb = Counter(); habitat = defaultdict(Counter)
    hidden = defaultdict(lambda: Counter()); maxhp = defaultdict(int)
    for sh, slot in picks:
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"][:T + 1].astype(int)
        D = {}
        for m in TYPES:
            P = R[f"{m}_position"][:T + 1].astype(int); M = R[f"{m}_mask"][:T + 1]
            V = M & (np.abs(P[..., 0] - pos[:, None, 0]) <= 3) & (np.abs(P[..., 1] - pos[:, None, 1]) <= 4)
            hp = R[f"{m}_health"][:T + 1]; cd = R[f"{m}_attack_cooldown"][:T + 1]
            maxhp[m] = max(maxhp[m], int(hp[M].max()) if M.any() else 0)
            last = np.full(M.shape, -1)                                   # last visible time per (t, j), alive-continuous
            for j in range(M.shape[1]):
                lv = -1
                for t in range(T + 1):
                    if not M[t, j]:
                        lv = -1
                    elif V[t, j]:
                        lv = t
                    last[t, j] = lv
            D[m] = (P, M, V, hp, cd, last)
        for t in range(T + 1):
            n_tr = 0; n_vis = 0
            for m in TYPES:
                P, M, V, hp, cd, last = D[m]
                near = M[t] & (np.abs(P[t] - pos[t]).sum(-1) <= 12)
                n_vis += int(V[t].sum())
                n_tr += int((near & (last[t] >= 0) & (t - last[t] <= 64)).sum())
                for j in np.nonzero(V[t])[0]:
                    habitat[int(R["map"][t][tuple(P[t, j])])][m] += 1
            tracked[min(n_tr, 12)] += 1; visible_n[min(n_vis, 12)] += 1
        for m in TYPES:
            P, M, V, hp, cd, last = D[m]
            for j in range(M.shape[1]):
                for t in range(1, T + 1):
                    if V[t, j] and not V[t - 1, j] and M[t - 1, j] and last[t - 1, j] >= 0:
                        t_last = last[t - 1, j]; dist = int(np.abs(P[t, j] - P[t_last, j]).sum())
                        others = 0
                        for k in range(M.shape[1]):
                            if k == j or not M[t, k] or V[t, k]:
                                continue
                            lk = last[t, k]
                            if lk >= 0 and t - lk <= 64 and np.abs(P[lk, k] - P[t, j]).sum() <= dist + 2:
                                others += 1
                        amb[(m, min(others, 3))] += 1
                        hidden[m]["n"] += 1
                        if m in ("zombies", "skeletons"):
                            hidden[m]["cooldown_gt0"] += int(cd[t, j] > 0)
                        hidden[m]["damaged"] += int(hp[t, j] < maxhp[m])
    tot = sum(tracked.values())
    res = {"episodes": len(picks),
           "E-a_tracked_set_size_dist": {k: round(v / tot, 4) for k, v in sorted(tracked.items())},
           "E-a_visible_count_dist": {k: round(v / tot, 4) for k, v in sorted(visible_n.items())},
           "E-b_reentry_ambiguity": {m: {str(k): round(amb[(m, k)] / max(1, sum(amb[(m, x)] for x in range(4))), 4) for k in range(4)}
                                     for m in TYPES},
           "E-b_reentries_n": {m: sum(amb[(m, x)] for x in range(4)) for m in TYPES},
           "E-c_habitat_P_type_given_block": {str(bk): {m: round(c[m] / sum(c.values()), 3) for m in TYPES} | {"n": sum(c.values())}
                                              for bk, c in habitat.items() if sum(c.values()) >= 100},
           "E-c_P_block_given_type": {m: {str(bk): round(habitat[bk][m] / max(1, sum(habitat[x][m] for x in habitat)), 3)
                                          for bk in habitat if habitat[bk][m] > 0} for m in TYPES},
           "E-d_hidden_state_at_reentry": {m: {k: (v / hidden[m]["n"] if k != "n" else v) for k, v in hidden[m].items()} for m in TYPES},
           "max_health_observed": dict(maxhp)}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
