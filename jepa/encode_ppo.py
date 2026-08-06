"""Parse yifeizhou/crafter-ppo trajectory files (competent PPO play, ~14/22 achievements)
into segmented V-JEPA latents for BC-pretraining. Each file = 1000 fixed 4-step segments
of (obs RGB (4,3,64,64), actions (4,1), rewards (4,1)). We motion-encode WITHIN each
segment (prev frame respects the segment boundary) and keep the (S,4) segment structure.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vjepa_encoder import VJEPAEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=str, default=os.path.join(REPO, "data", "crafter_ppo", "ex", "*.pt"))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data", "crafter_ppo_seg.npz"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.files))
    print(f"loading {len(files)} PPO files...")
    frames, prevs, acts, rews = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for t in d:
            ob = t["obs"].numpy()                              # (T,3,64,64) [0,1]
            T = ob.shape[0]
            hwc = np.clip(ob.transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)  # (T,64,64,3)
            prev = np.concatenate([hwc[:1], hwc[:-1]], 0)      # f_{t-1} within segment
            frames.append(hwc); prevs.append(prev)
            acts.append(t["actions"].reshape(-1).numpy().astype(np.int64))
            rews.append(t["rewards"].reshape(-1).numpy().astype(np.float32))
    S = len(acts); T = acts[0].shape[0]
    frames = np.concatenate(frames, 0); prevs = np.concatenate(prevs, 0)
    acts = np.stack(acts); rews = np.stack(rews)
    print(f"segments={S} seglen={T} total_frames={len(frames)}  "
          f"reward/seg mean {rews.sum(1).mean():.2f}")

    enc = VJEPAEncoder(res=args.res, pool="mean", device="cuda")
    print(f"encoding {len(frames)} frames (motion, dim={enc.dim})...")
    lat = enc.encode(frames, motion=True, prev_frames=prevs).numpy().astype(np.float32)
    lat = lat.reshape(S, T, -1)                                # (S,T,dim)
    print(f"seg_latents={lat.shape}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, seg_latents=lat, seg_actions=acts, seg_rewards=rews, action_dim=17)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
