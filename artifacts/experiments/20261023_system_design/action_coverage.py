"""How much counterfactual action coverage does our offline data contain? (2026-10-23, design measurement)

The fifth kind of unpredictability: what the model cannot know because the data never tried it. Evaluation-only use of
simulator state; nothing is trained. TRAIN-split support-v2 episodes (all epsilon levels) are replayed exactly and every
transition is assigned a coarse context from the state BEFORE the action:
  faced block group, zombie adjacent (and whether one is ready to attack: hidden cooldown <= 0), zombie at distance 2-3,
  sleeping / can sleep (awake, energy < 9), health bucket, night, food or drink at 0.
Reports how many of the 17 actions each context has seen (frequency-weighted), and the action counts plus observed
16-step death rates in decision-critical contexts, split by behaviour noise (epsilon 0.1 vs 1.0).
"""
import json, os, sys, time
from collections import defaultdict
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

GROUP = {2: "walk", 7: "walk", 13: "walk", 5: "tree", 4: "stone", 8: "ore", 9: "ore", 10: "ore", 3: "water",
         11: "table", 12: "furnace", 14: "lava", 15: "plant", 16: "ripe"}
ACTIONS = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "P_STONE", "P_TABLE", "P_FURN", "P_PLANT",
           "M_WPICK", "M_SPICK", "M_IPICK", "M_WSWORD", "M_SSWORD", "M_ISWORD"]


def scan_fn():
    from craftax.craftax_classic.constants import DIRECTIONS
    env, params, _, _, _ = RP.env_and_render()

    def feats(s):
        face = s.player_position + DIRECTIONS[s.player_direction]
        face = jnp.clip(face, 0, 63)
        zd = jnp.abs(s.zombies.position - s.player_position).sum(-1)
        adj = s.zombies.mask & (zd == 1)
        return {"face": s.map[face[0], face[1]], "zadj": adj.any(), "zready": (adj & (s.zombies.attack_cooldown <= 0)).any(),
                "znear": (s.zombies.mask & (zd >= 2) & (zd <= 3)).any(), "sleep": s.is_sleeping,
                "cansleep": (~s.is_sleeping) & (s.player_energy < 9), "health": s.player_health,
                "night": s.light_level < 0.5, "starve": (s.player_food == 0) | (s.player_drink == 0)}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, feats(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), feats(s0), seq)

    return jax.jit(run)


def main():
    n_eps, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(20261024)
    man = RP.manifest()
    picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        cand = [i for i, e in enumerate(eps) if e["split"] == "train"]
        for i in rng.permutation(cand)[:3].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    counts = defaultdict(lambda: np.zeros(17, np.int64))
    by_eps = defaultdict(lambda: defaultdict(lambda: np.zeros(17, np.int64)))
    death16 = defaultdict(lambda: np.zeros((17, 2), np.int64))   # [action, (n, died within 16)]
    t0 = time.time(); n_tr = 0
    for e_i, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot)
        acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        died = bool(f["terminated"][-1])
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T)
        pad = np.zeros(b, np.int32); pad[:T] = acts
        F = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        for t in range(T):
            h = int(F["health"][t]); hb = "low" if h <= 3 else ("mid" if h <= 6 else "high")
            ctx = (GROUP.get(int(F["face"][t]), "other"), bool(F["zadj"][t]), bool(F["zready"][t]), bool(F["znear"][t]),
                   bool(F["sleep"][t]), bool(F["cansleep"][t]), hb, bool(F["night"][t]), bool(F["starve"][t]))
            a = int(acts[t]); counts[ctx][a] += 1; by_eps[f["epsilon"]][ctx][a] += 1
            death16[ctx][a, 0] += 1; death16[ctx][a, 1] += int(died and (T - 1 - t) < 16)
            n_tr += 1
        if (e_i + 1) % 200 == 0:
            print(f"{e_i + 1} episodes, {n_tr} transitions, {time.time() - t0:.0f}s", flush=True)
    tot = sum(c.sum() for c in counts.values())
    res = {"episodes": len(picks), "transitions": int(tot), "contexts": len(counts)}
    for k in (1, 5, 20):
        res[f"transition_share_context_has_all17_with_ge{k}"] = float(sum(c.sum() for c in counts.values() if (c >= k).all()) / tot)
        res[f"mean_actions_with_ge{k}_weighted"] = float(sum(c.sum() * (c >= k).sum() for c in counts.values()) / tot)
    # decision-critical contexts (aggregated over the remaining features)
    def agg(pred):
        c = np.zeros(17, np.int64); d = np.zeros((17, 2), np.int64); ce = {}
        for ctx, v in counts.items():
            if pred(ctx):
                c += v; d += death16[ctx]
        for e, dd in by_eps.items():
            ce[str(e)] = int(sum(v.sum() for ctx, v in dd.items() if pred(ctx)))
        rate = {ACTIONS[i]: (round(float(d[i, 1] / d[i, 0]), 3) if d[i, 0] else None) for i in range(17)}
        return {"n": int(c.sum()), "by_epsilon": ce, "action_counts": {ACTIONS[i]: int(c[i]) for i in range(17)},
                "death_within_16_by_action": rate}
    res["critical"] = {
        "zombie_ready_adjacent_awake": agg(lambda x: x[2] and not x[4]),
        "zombie_ready_adjacent_low_health": agg(lambda x: x[2] and x[6] == "low"),
        "can_sleep_zombie_near_or_adjacent": agg(lambda x: x[5] and (x[1] or x[3])),
        "can_sleep_night_no_zombie": agg(lambda x: x[5] and x[7] and not (x[1] or x[3])),
        "facing_lava": agg(lambda x: x[0] == "lava"),
        "facing_water_starving": agg(lambda x: x[0] == "water" and x[8]),
        "facing_table": agg(lambda x: x[0] == "table"),
    }
    # how much of the counterfactual coverage comes from noisy behaviour
    share = {}
    for e, dd in by_eps.items():
        share[str(e)] = int(sum(v.sum() for v in dd.values()))
    res["transitions_by_epsilon"] = share
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
