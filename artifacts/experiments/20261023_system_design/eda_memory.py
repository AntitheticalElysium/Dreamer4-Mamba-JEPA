"""E1-E3 (2026-10-23): facts that decide memory / anchoring requirements. Simulator replay, TRAIN-split support-v2
episodes, evaluation-only state. Nothing is trained.

E1  Returning entities: for every mob that re-enters view (same entity, alive throughout, seen earlier in the episode),
    the gap since it was last visible; split into all returns and returns that matter: a zombie / skeleton returning
    within Manhattan 3 of the player, or followed by damage to the player within 16 steps while it is adjacent; a cow
    eaten within 16 steps.
E2  Revisited terrain: for every map cell entering view, whether its world position was seen before in the episode and
    the gap; whether it is consequential (water, tree, stone, ores, table, furnace, plants, lava) and whether the agent
    interacts with it (DO / place facing it, or table / furnace used by crafting nearby) within 16 steps.
E3  Visibility of causes: for zombie damage, the attacker's visibility 1 / 4 / 16 steps earlier; for each achievement
    unlocked by interacting with a faced cell, whether that cell was in view 4 / 16 steps earlier.
Reported separately for epsilon 0.1 (near-expert) and all episodes.
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

CONSEQ = {3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16}          # water stone tree coal iron diamond table furnace lava plant ripe
DO, PLACES = 5, (7, 8, 9, 10)
CRAFT = (11, 12, 13, 14, 15, 16)


def scan_fn():
    from craftax.craftax_classic.constants import DIRECTIONS
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "dir": s.player_direction, "map": s.map.astype(jnp.int8),
                     "health": s.player_health, "ach": s.achievements,
                     **{f"{m}_p": getattr(s, m).position for m in ("zombies", "cows", "skeletons")},
                     **{f"{m}_m": getattr(s, m).mask for m in ("zombies", "cows", "skeletons")}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)

    return jax.jit(run), np.asarray(DIRECTIONS)


def in_view(p, pos):
    d = p - pos
    return (np.abs(d[..., 0]) <= 3) & (np.abs(d[..., 1]) <= 4)


def main():
    n_eps, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(20261029)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 32])[:3].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run, DIRS = scan_fn()
    R1 = {"all": [], "matter": []}; R1e = {"all": [], "matter": []}
    E2 = {"entering": 0, "revisit": 0, "gaps": [], "conseq_revisit_used": [], "conseq_new_used": 0}
    E2e = {"entering": 0, "revisit": 0, "gaps": [], "conseq_revisit_used": [], "conseq_new_used": 0}
    E3 = {"hits": 0, "vis1": 0, "vis4": 0, "vis16": 0, "ach": 0, "ach_vis4": 0, "ach_vis16": 0}
    E3e = dict.fromkeys(E3, 0)
    t0 = time.time()
    for e_i, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        expert = f["epsilon"] == 0.1
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"][:T + 1].astype(int); H = R["health"][:T + 1]
        hit = np.zeros(T + 1, bool); hit[1:] = H[1:] < H[:-1]
        # E1 + E3 (zombie hits)
        for m in ("zombies", "cows", "skeletons"):
            P = R[f"{m}_p"][:T + 1].astype(int); M = R[f"{m}_m"][:T + 1]
            V = M & in_view(P, pos[:, None, :])
            for j in range(M.shape[1]):
                last_vis = 0 if (M[0, j] and V[0, j]) else -1; born = 0
                for t in range(1, T + 1):
                    if M[t, j] and not M[t - 1, j]:
                        last_vis = t if V[t, j] else -1; born = t; continue
                    if not M[t, j]:
                        last_vis = -1; continue
                    if V[t, j] and not V[t - 1, j] and last_vis >= 0:
                        gap = t - last_vis
                        dist = np.abs(P[t, j] - pos[t]).sum()
                        fut = slice(t, min(T + 1, t + 17))
                        adj = np.abs(P[fut, j] - pos[fut]).sum(-1) == 1
                        matter = False
                        if m in ("zombies", "skeletons"):
                            matter = dist <= 3 or bool((hit[fut] & adj).any())
                        else:
                            matter = bool(((~M[fut, j]) & adj).any())
                        for D in ((R1, R1e) if expert else (R1,)):
                            D["all"].append(gap)
                            if matter: D["matter"].append(gap)
                    if V[t, j]:
                        last_vis = t
                if m == "zombies":
                    for t in np.nonzero(hit)[0]:
                        if t == 0 or not M[t - 1, j] or np.abs(P[t - 1, j] - pos[t - 1]).sum() > 1:
                            continue
                        for D in ((E3, E3e) if expert else (E3,)):
                            D["hits"] += 1
                            for k in (1, 4, 16):
                                tk = t - 1 - k
                                if tk >= 0 and M[tk, j] and V[tk, j]:
                                    D[f"vis{k}"] += 1
        # E2 terrain revisits
        seen_last = {}
        mp = R["map"]
        for t in range(T + 1):
            cells = [(int(pos[t, 0] + r - 3), int(pos[t, 1] + c - 4)) for r in range(7) for c in range(9)]
            if t >= 1:
                prev = {(int(pos[t - 1, 0] + r - 3), int(pos[t - 1, 1] + c - 4)) for r in range(7) for c in range(9)}
                for w in cells:
                    if w in prev or not (0 <= w[0] < 64 and 0 <= w[1] < 64):
                        continue
                    blk = int(mp[t][w])
                    used = False
                    if blk in CONSEQ:
                        for u in range(t, min(T, t + 16)):
                            face = (int(pos[u, 0] + DIRS[R["dir"][u]][0]), int(pos[u, 1] + DIRS[R["dir"][u]][1]))
                            if face == w and (acts[u] == DO or acts[u] in PLACES):
                                used = True; break
                            if blk in (11, 12) and acts[u] in CRAFT and max(abs(pos[u, 0] - w[0]), abs(pos[u, 1] - w[1])) <= 1:
                                used = True; break
                    for D in ((E2, E2e) if expert else (E2,)):
                        D["entering"] += 1
                        if w in seen_last:
                            D["revisit"] += 1; g = t - seen_last[w]; D["gaps"].append(g)
                            if blk in CONSEQ and used: D["conseq_revisit_used"].append(g)
                        elif blk in CONSEQ and used:
                            D["conseq_new_used"] += 1
            for w in cells:
                seen_last[w] = t
        # E3 achievements tied to the faced cell
        ach = R["ach"][:T + 1]
        for t in range(1, T + 1):
            if (ach[t] & ~ach[t - 1]).any() and acts[t - 1] in (DO,) + PLACES:
                u = t - 1
                w = (int(pos[u, 0] + DIRS[R["dir"][u]][0]), int(pos[u, 1] + DIRS[R["dir"][u]][1]))
                for D in ((E3, E3e) if expert else (E3,)):
                    D["ach"] += 1
                    for k in (4, 16):
                        tk = u - k
                        if tk >= 0 and abs(w[0] - pos[tk, 0]) <= 3 and abs(w[1] - pos[tk, 1]) <= 4:
                            D[f"ach_vis{k}"] += 1
        if (e_i + 1) % 100 == 0:
            print(e_i + 1, round(time.time() - t0), flush=True)

    def gapstats(g):
        g = np.array(g)
        if len(g) == 0:
            return None
        return {"n": int(len(g)), "le4": float((g <= 4).mean()), "le16": float((g <= 16).mean()), "le64": float((g <= 64).mean()),
                "p50": float(np.median(g)), "p90": float(np.percentile(g, 90))}

    def pack(R1_, E2_, E3_):
        return {"E1_returns_all": gapstats(R1_["all"]), "E1_returns_that_matter": gapstats(R1_["matter"]),
                "E2_entering_cells": E2_["entering"], "E2_revisit_share": E2_["revisit"] / max(E2_["entering"], 1),
                "E2_revisit_gaps": gapstats(E2_["gaps"]), "E2_used_consequential_revisits": gapstats(E2_["conseq_revisit_used"]),
                "E2_used_consequential_first_sightings": E2_["conseq_new_used"],
                "E3_zombie_hits": E3_["hits"], **{f"E3_attacker_visible_{k}_before": E3_[f"vis{k}"] / max(E3_["hits"], 1) for k in (1, 4, 16)},
                "E3_faced_achievements": E3_["ach"], **{f"E3_target_visible_{k}_before": E3_[f"ach_vis{k}"] / max(E3_["ach"], 1) for k in (4, 16)}}

    res = {"episodes": len(picks), "all": pack(R1, E2, E3), "epsilon_0.1": pack(R1e, E2e, E3e)}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
