"""Collect REAL Memory-Maze trajectories and cache their frozen V-JEPA latents +
actions to disk (encode once — the encoder is frozen). Output feeds the Mamba
dynamics training. Real data only: frames/actions come from the true environment.
"""
import os
import sys
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
sys.path.insert(0, DRAMA)
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")

from envs.my_memory_maze import MemoryMaze
from vjepa_encoder import VJEPAEncoder


def collect(env, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    frames, actions, rewards, terms = [], [], [], []
    ob, _ = env.reset()
    frames.append(ob)
    for _ in range(n_steps):
        a = int(rng.integers(env.action_space.n))
        ob, r, is_last, info = env.step(a)
        actions.append(a)
        rewards.append(float(r))
        terms.append(bool(info.get("is_terminal", False)))
        frames.append(ob)
        if is_last or info.get("is_terminal", False):
            ob, _ = env.reset()
            frames[-1] = ob
    return (np.asarray(frames, dtype=np.uint8), np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32), np.asarray(terms, dtype=bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data", "mm_vjepa_latents.npz"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = MemoryMaze("memory_maze:MemoryMaze-9x9-v0", size=(64, 64))
    print("collecting real maze trajectory...")
    frames, actions, rewards, terms = collect(env, args.steps, seed=args.seed)
    print(f"frames={frames.shape} actions={actions.shape}")
    print(f"REWARD sparsity: sum={rewards.sum():.1f}  nonzero={int((rewards != 0).sum())}/{len(rewards)} "
          f"({100 * (rewards != 0).mean():.2f}%)  terminals={int(terms.sum())}")

    enc = VJEPAEncoder(res=args.res, pool=True, device="cuda")
    print(f"encoding {len(frames)} frames with frozen V-JEPA (dim={enc.dim})...")
    lat = enc.encode(frames).numpy().astype(np.float32)   # [T+1, 768]
    print(f"latents={lat.shape}  mean|z|={np.abs(lat).mean():.3f}  std={lat.std():.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, latents=lat, actions=actions, rewards=rewards, terminals=terms,
             action_dim=env.action_space.n)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
