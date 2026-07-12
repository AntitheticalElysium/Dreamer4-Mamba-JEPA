"""Collect REAL Crafter trajectories and cache frozen V-JEPA latents + actions +
rewards + terminals. Crafter: dense reward (achievements +1, health +/-0.1), 64x64,
Discrete(17) — dense enough for imagination training AND rich/surprising for the
honesty study. Real data only (true env dynamics)."""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import crafter
from vjepa_encoder import VJEPAEncoder


def collect(n_steps, seed=0):
    env = crafter.Env(seed=seed)
    rng = np.random.default_rng(seed)
    frames, actions, rewards, terms = [], [], [], []
    ob = env.reset()
    frames.append(ob)
    for _ in range(n_steps):
        a = int(rng.integers(env.action_space.n))
        ob, r, done, info = env.step(a)
        actions.append(a); rewards.append(float(r)); terms.append(bool(done))
        frames.append(ob)
        if done:
            ob = env.reset(); frames[-1] = ob
    return (np.asarray(frames, dtype=np.uint8), np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32), np.asarray(terms, dtype=bool),
            env.action_space.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pool", type=str, default="grid", choices=["mean", "grid", "none"])
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--motion", type=int, default=1)  # 1: encode real 2-frame clips (motion)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data", "crafter_vjepa.npz"))
    args = ap.parse_args()

    print("collecting real Crafter trajectory...")
    frames, actions, rewards, terms, adim = collect(args.steps, seed=args.seed)
    print(f"frames={frames.shape} actions={actions.shape} action_dim={adim}")
    print(f"REWARD: sum={rewards.sum():.1f} nonzero={int((rewards != 0).sum())}/{len(rewards)} "
          f"({100 * (rewards != 0).mean():.2f}%)  terminals={int(terms.sum())}")

    enc = VJEPAEncoder(res=args.res, pool=args.pool, grid=args.grid, device="cuda")
    print(f"encoding {len(frames)} frames (pool={args.pool} motion={args.motion} dim={enc.dim})...")
    lat = enc.encode(frames, motion=bool(args.motion)).numpy().astype(np.float32)
    print(f"latents={lat.shape}  mean|z|={np.abs(lat).mean():.3f}  std={lat.std():.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, latents=lat, actions=actions, rewards=rewards, terminals=terms, action_dim=adim)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
