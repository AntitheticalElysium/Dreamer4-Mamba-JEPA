"""T1 data: symbolic per-cell observations by exact replay (T1_PROTOCOL.md). Same episode selection as t2_data.py.

Usage: t1_data.py OUT_DIR N_TRAIN N_DEV  ->  OUT_DIR/{train,dev}_sym.npz with, per frame:
  view [7, 9] block id; mob [7, 9] (0 none, 1 zombie, 2 cow, 3 skeleton, 4 arrow); vit [4]; inv [12]; sleep; light;
  pos [2] (world position: used only by E* and for evaluation events); action (the action taken FROM this frame, -1 at the
  episode's last frame); and for DEV the mob world positions / masks per type and index (evaluation events only).
"""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
from t2_data import pick

MOBS = ("zombies", "cows", "skeletons", "arrows")
INV = ("wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
       "wood_sword", "stone_sword", "iron_sword")
PRIORITY = (1, 3, 2, 4)                     # mob code written last wins: arrow < cow < skeleton < zombie


def scan():
    env, params, _, _, _ = RP.env_and_render()

    def lab(s):
        padded = jnp.pad(s.map, 4, constant_values=1)
        view = jax.lax.dynamic_slice(padded, (s.player_position[0] - 3 + 4, s.player_position[1] - 4 + 4), (7, 9))
        grid = jnp.zeros((7, 9), jnp.int32)
        for code, m in sorted(zip((1, 2, 3, 4), MOBS), key=lambda x: {4: 0, 2: 1, 3: 2, 1: 3}[x[0]]):
            mm = getattr(s, m)
            r = mm.position[:, 0] - s.player_position[0] + 3
            c = mm.position[:, 1] - s.player_position[1] + 4
            ok = mm.mask & (r >= 0) & (r < 7) & (c >= 0) & (c < 9)
            cur = grid[jnp.clip(r, 0, 6), jnp.clip(c, 0, 8)]
            grid = grid.at[jnp.clip(r, 0, 6), jnp.clip(c, 0, 8)].set(jnp.where(ok, code, cur))
        inv = jnp.stack([getattr(s.inventory, n) for n in INV]).astype(jnp.int32)
        vit = jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]).astype(jnp.int32)
        out = {"view": view.astype(jnp.int8), "mob": grid.astype(jnp.int8), "vit": vit, "inv": inv,
               "sleep": s.is_sleeping, "light": s.light_level, "pos": s.player_position}
        for m in MOBS:
            out[f"{m}_p"] = getattr(s, m).position; out[f"{m}_m"] = getattr(s, m).mask
        return out

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, lab(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), lab(s0), seq)

    return jax.jit(run)


def build(picks, mobs_too):
    run = scan()
    cols = {k: [] for k in ("view", "mob", "vit", "inv", "sleep", "light", "pos", "action")}
    if mobs_too:
        for m in MOBS:
            cols[f"{m}_p"] = []; cols[f"{m}_m"] = []
    starts, lens = [], []; p = 0
    for sh, i in picks:
        f = RP.episode_fields(sh, i); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, i)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        L = T + 1
        for k in ("view", "mob", "vit", "inv", "sleep", "light", "pos"):
            cols[k].append(R[k][:L].astype({"view": np.int8, "mob": np.int8, "vit": np.int8, "inv": np.int8,
                                            "sleep": bool, "light": np.float32, "pos": np.int16}[k]))
        a = np.full(L, -1, np.int8); a[:T] = acts; cols["action"].append(a)
        if mobs_too:
            for m in MOBS:
                cols[f"{m}_p"].append(R[f"{m}_p"][:L].astype(np.int16)); cols[f"{m}_m"].append(R[f"{m}_m"][:L])
        starts.append(p); lens.append(L); p += L
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["starts"] = np.array(starts); out["lengths"] = np.array(lens)
    return out


def main():
    out_dir, n_tr, n_dv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True); t0 = time.time()
    tr = pick("train", n_tr, 20261026); dv = pick("dev", n_dv, 20261027)          # identical to t2_data.py
    np.savez_compressed(os.path.join(out_dir, "train_sym.npz"), **build(tr, False))
    np.savez_compressed(os.path.join(out_dir, "dev_sym.npz"), **build(dv, True))
    print(json.dumps({"train": len(tr), "dev": len(dv), "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
