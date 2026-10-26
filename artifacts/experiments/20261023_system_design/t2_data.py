"""T2 data: a shared memmap of TRAIN-split support-v2 frames + actions / rewards / terminations, and a labelled DEV set.

Usage: t2_data.py OUT_DIR N_TRAIN_EPISODES N_DEV_EPISODES
  OUT_DIR/train_frames.u8  (N, 63, 63, 3) uint8 memmap; train_meta.npz: actions, rewards, terms (transition i -> i+1;
  the last frame of an episode has action -1), episode starts / lengths, episode ids.
  OUT_DIR/dev_frames.u8 + dev_meta.npz: the same for DEV episodes, plus simulator labels per frame (evaluation only):
  block id per map cell (7 x 9), mob presence per map cell for zombie / cow / skeleton / arrow, vitals, light.
Frame i of an episode is the observation after i actions (verified by artifacts/eda/replay.py: render == stored frame).
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

MOBS = ("zombies", "cows", "skeletons", "arrows")


def label_scan():
    env, params, _, _, _ = RP.env_and_render()

    def lab(s):
        padded = jnp.pad(s.map, 4, constant_values=1)                       # 1 = OUT_OF_BOUNDS
        view = jax.lax.dynamic_slice(padded, (s.player_position[0] - 3 + 4, s.player_position[1] - 4 + 4), (7, 9))
        mob = []
        for m in MOBS:
            mm = getattr(s, m)
            r = mm.position[:, 0] - s.player_position[0] + 3
            c = mm.position[:, 1] - s.player_position[1] + 4
            ok = mm.mask & (r >= 0) & (r < 7) & (c >= 0) & (c < 9)
            grid = jnp.zeros((7, 9), jnp.int32).at[jnp.clip(r, 0, 6), jnp.clip(c, 0, 8)].max(ok.astype(jnp.int32))
            mob.append(grid > 0)
        vit = jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]).astype(jnp.int32)
        return {"view": view.astype(jnp.int8), "mob": jnp.stack(mob), "vit": vit, "light": s.light_level}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, lab(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), lab(s0), seq)

    return jax.jit(run)


def pick(split, n, seed):
    rng = np.random.default_rng(seed)
    man = RP.manifest()
    out = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        taken = 0
        for i in rng.permutation(len(eps)).tolist():
            if eps[i]["split"] == split and len(eps[i]["actions_taken"]) >= 32:
                out.append((sh, i)); taken += 1
                if taken == 3 or len(out) >= n:
                    break
        if len(out) >= n:
            break
    return out


def build(out_dir, split, picks, labels):
    lens = [len(RP.episode_fields(sh, i)["actions_taken"]) + 1 for sh, i in picks]
    N = int(sum(lens))
    frames = np.lib.format.open_memmap(os.path.join(out_dir, f"{split}_frames.npy"), mode="w+", dtype=np.uint8,
                                       shape=(N, 63, 63, 3))
    actions = np.full(N, -1, np.int8); rewards = np.zeros(N, np.float32); terms = np.zeros(N, bool)
    starts = np.zeros(len(picks), np.int64); ids = []
    lab = {"view": np.zeros((N, 7, 9), np.int8), "mob": np.zeros((N, 4, 7, 9), bool), "vit": np.zeros((N, 4), np.int8),
           "light": np.zeros(N, np.float32)} if labels else None
    run = label_scan() if labels else None
    p = 0; worst = 0
    for e_i, ((sh, i), L) in enumerate(zip(picks, lens)):
        f = RP.episode_fields(sh, i)
        frames[p:p + L] = f["observations"].numpy()
        actions[p:p + L - 1] = f["actions_taken"].numpy(); rewards[p:p + L - 1] = f["rewards"].numpy()
        terms[p:p + L - 1] = f["terminated"].numpy()
        starts[e_i] = p; ids.append(f["episode_id"])
        if labels:
            reset_key, env_keys = RP._slot_keys(sh, i)
            T = L - 1; b = next(x for x in RP.BUCKETS if x >= T)
            pad = np.zeros(b, np.int32); pad[:T] = f["actions_taken"].numpy()
            R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
            for k in lab:
                lab[k][p:p + L] = R[k][:L]
            if e_i < 5:                                                    # alignment check: render == stored frame
                _, _, _, _, frame = RP.env_and_render()
                for t in (0, L // 2, L - 1):
                    st = RP.advance_to(sh, i, t)
                    worst = max(worst, int(np.abs(np.asarray(frame(st)).astype(int) - frames[p + t].astype(int)).max()))
        p += L
        if (e_i + 1) % 200 == 0:
            print(split, e_i + 1, p, flush=True)
    frames.flush()
    meta = {"actions": actions, "rewards": rewards, "terms": terms, "starts": starts, "lengths": np.array(lens),
            "ids": np.array(ids)}
    if labels:
        meta.update({f"lab_{k}": v for k, v in lab.items()})
    np.savez(os.path.join(out_dir, f"{split}_meta.npz"), **meta)
    return N, worst


def main():
    out_dir, n_train, n_dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tr = pick("train", n_train, 20261026); dv = pick("dev", n_dev, 20261027)
    n1, _ = build(out_dir, "train", tr, labels=False)
    n2, worst = build(out_dir, "dev", dv, labels=True)
    info = {"train_episodes": len(tr), "train_frames": n1, "dev_episodes": len(dv), "dev_frames": n2,
            "render_check_maxdiff": worst, "seconds": time.time() - t0}
    json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
