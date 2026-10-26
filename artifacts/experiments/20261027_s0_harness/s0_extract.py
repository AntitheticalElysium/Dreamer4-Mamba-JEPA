"""S0 step 1 (2026-10-27): simulator labels for a corpus split, for EVALUATION only (TEST_SPEC v1.1).

Replays every dev episode of support-v2 exactly (artifacts/eda/replay.py) and records, per state s_0..s_T: player
position / direction / light / sleep / health / food / drink / energy / inventory / achievements, and per entity kind
(zombies, cows, skeletons, arrows) position, mask, cooldown and health; per transition: action, reward, termination and
the collector's exogenous flag (uniform-random branch, regenerated from the seed tree as in A8); per state also the
7x9 block types of the view (the terrain a reader would report). Frames are not copied: they are read from the corpus
by (shard, slot) when needed.
Usage (project venv): s0_extract.py OUT_NPZ [SPLIT [MAX_EPISODES]]   (default: dev, all)
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
from craftax.craftax_classic.constants import BlockType

KINDS = ("zombies", "cows", "skeletons", "arrows")
N_ACTIONS = 17


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    inv = lambda s: jnp.stack([getattr(s.inventory, f) for f in s.inventory.__dataclass_fields__])
    view = lambda s: jax.lax.dynamic_slice(jnp.pad(s.map, 4, constant_values=BlockType.OUT_OF_BOUNDS.value),
                                           (s.player_position[0] + 1, s.player_position[1]), (7, 9)).astype(jnp.int8)
    rec = lambda s: {"pos": s.player_position, "dir": s.player_direction, "light": s.light_level, "sleep": s.is_sleeping,
                     "body": jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]),
                     "inv": inv(s), "ach": s.achievements, "view": view(s),
                     **{f"{m}_p": getattr(s, m).position for m in KINDS}, **{f"{m}_m": getattr(s, m).mask for m in KINDS},
                     **{f"{m}_cd": getattr(s, m).attack_cooldown for m in KINDS}, **{f"{m}_hp": getattr(s, m).health for m in KINDS}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


@jax.jit
def take_flags(step_keys, eps):
    def one(k):
        ka, kn, ku, sk = jax.random.split(k, 4)
        return jax.random.uniform(ku, (RP.NUM_ENVS,)) < eps
    return jax.vmap(one)(step_keys)


def main():
    out = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "dev"
    max_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    man = RP.manifest(); run = scan_fn()
    S, Tr, meta = {}, {"action": [], "reward": [], "terminated": [], "exogenous": []}, []
    for sh in range(len(man["shards"])):
        eps_list = RP._shard(sh)["episodes"]
        _, step_keys = RP.round_keys(sh)
        flag_cache = {}
        for slot, e in enumerate(eps_list):
            if e["split"] != split or len(meta) >= max_eps:
                continue
            acts = e["actions_taken"].numpy().astype(np.int32); T = len(acts)
            reset_key, env_keys = RP._slot_keys(sh, slot)
            b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
            R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
            for k, v in R.items():
                S.setdefault(k, []).append(v[: T + 1])
            eps = float(e["epsilon"])
            if eps not in flag_cache:
                flag_cache[eps] = np.asarray(take_flags(step_keys, eps))
            Tr["action"].append(acts.astype(np.int8)); Tr["reward"].append(e["rewards"].numpy().astype(np.float32))
            Tr["terminated"].append(e["terminated"].numpy().astype(bool))
            Tr["exogenous"].append(flag_cache[eps][:T, slot])
            meta.append((sh, slot, eps, T))
        print(f"shard {sh + 1}/{len(man['shards'])} episodes {len(meta)}", flush=True)
    meta = np.array(meta, np.float64)
    np.savez_compressed(out, **{f"s_{k}": np.concatenate(v) for k, v in S.items()},
                        **{f"t_{k}": np.concatenate(v) for k, v in Tr.items()},
                        ep_shard=meta[:, 0].astype(np.int32), ep_slot=meta[:, 1].astype(np.int32),
                        ep_eps=meta[:, 2].astype(np.float32), ep_T=meta[:, 3].astype(np.int32))
    print("saved", out, "episodes", len(meta), "transitions", int(meta[:, 3].sum()))


if __name__ == "__main__":
    main()
