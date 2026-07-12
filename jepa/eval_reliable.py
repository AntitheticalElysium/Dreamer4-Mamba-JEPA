"""Reliable, PAIRED multi-checkpoint Crafter eval.

The eval-noise problem: single-checkpoint achievement variance is huge (std ~1.5 on a
~3-ach mean), so 8-24 episodes can't resolve the ~0.3-0.5 ach effects our interventions
produce. Fix: evaluate every checkpoint on the SAME set of env seeds, then compare PAIRED
per-seed differences. Per-seed difficulty (some seeds are just easier) cancels in the
difference, so the paired SE is far smaller than the raw SE -> we can actually measure
small effects with a tractable episode budget.

Usage:
  python jepa/eval_reliable.py --episodes 60 \
      --spec bc:ckpt/jepa_wm.pt:ckpt/jepa_agent_bc.pt \
      --spec bc_gated:ckpt/jepa_wm_bc_gated.pt:ckpt/jepa_agent_bc_gated.pt \
      --spec kl_long:ckpt/jepa_wm_kl_long.pt:ckpt/jepa_agent_kl_long.pt
Always includes a random baseline on the same seeds.
"""
import os
import sys
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)

import crafter
from agents import ActorCriticAgent
from dynamics import JEPADynamics
from vjepa_encoder import VJEPAEncoder
from train_agent import shim_config


@torch.inference_mode()
def run_policy(model, agent, enc, mean, std, dev, seeds, max_steps=1000):
    """Run the filtering policy on each seed; return (rewards, achs) arrays aligned to seeds."""
    rewards, achs = [], []
    for sd in seeds:
        env = crafter.Env(seed=int(sd)); obs = env.reset(); prev = None
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        tot, ach = 0.0, set()
        for _ in range(max_steps):
            pf = obs if prev is None else prev
            z = (enc.encode(np.stack([pf, obs]), motion=True)[1:2].to(dev) - mean) / std
            action, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
            _, h, states = model.step(z.unsqueeze(1), action, states)
            prev = obs
            obs, r, done, info = env.step(int(action.item())); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: ach.add(k)
            if done: break
        rewards.append(tot); achs.append(len(ach))
    return np.array(rewards), np.array(achs)


def run_random(seeds, max_steps=1000):
    rng = np.random.default_rng(0)
    rewards, achs = [], []
    for sd in seeds:
        env = crafter.Env(seed=int(sd)); obs = env.reset()
        tot, ach = 0.0, set()
        for _ in range(max_steps):
            obs, r, done, info = env.step(int(rng.integers(17))); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: ach.add(k)
            if done: break
        rewards.append(tot); achs.append(len(ach))
    return np.array(rewards), np.array(achs)


def paired(a, b):
    """Paired mean difference a-b with its SE and an approx z (mean/se)."""
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--base_seed", type=int, default=5000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--spec", action="append", default=[], help="tag:wm_ckpt:agent_ckpt")
    args = ap.parse_args()
    dev = "cuda"
    seeds = np.arange(args.base_seed, args.base_seed + args.episodes)

    enc = VJEPAEncoder(res=args.res, pool="mean", device=dev)  # shared frozen encoder
    results = {}  # tag -> (rewards, achs)

    for spec in args.spec:
        tag, wm_path, ag_path = spec.split(":")
        ck = torch.load(wm_path, map_location=dev, weights_only=False)
        model = JEPADynamics(latent_dim=ck["latent_dim"], action_dim=ck["action_dim"], n_heads=5).to(dev).eval()
        model.load_state_dict(ck["model"])
        mean = torch.tensor(ck["mean"], device=dev); std = torch.tensor(ck["std"], device=dev)
        agent = ActorCriticAgent(shim_config(ck["latent_dim"], model.d_model), ck["action_dim"], dev).to(dev).eval()
        agent.load_state_dict(torch.load(ag_path, map_location=dev, weights_only=False))
        print(f"[{tag}] running {args.episodes} paired episodes...", flush=True)
        results[tag] = run_policy(model, agent, enc, mean, std, dev, seeds)
        del model, agent; torch.cuda.empty_cache()

    print(f"[random] running {args.episodes} paired episodes...", flush=True)
    results["random"] = run_random(seeds)

    rew_rand, ach_rand = results["random"]
    print("\n=== ACHIEVEMENTS (mean ± SE over {} paired seeds) ===".format(args.episodes))
    for tag, (rew, ach) in results.items():
        se = ach.std(ddof=1) / np.sqrt(len(ach))
        line = f"  {tag:14s} {ach.mean():.3f} ± {se:.3f}   (reward {rew.mean():+.2f})"
        if tag != "random":
            dm, dse, z = paired(ach, ach_rand)
            line += f"   | vs random Δ {dm:+.3f} ± {dse:.3f}  (z={z:+.2f})"
        print(line)

    # paired comparisons between every non-random pair
    tags = [t for t in results if t != "random"]
    if len(tags) > 1:
        print("\n=== PAIRED ACHIEVEMENT DIFFERENCES (row - col) ===")
        for i in range(len(tags)):
            for j in range(len(tags)):
                if i >= j: continue
                dm, dse, z = paired(results[tags[i]][1], results[tags[j]][1])
                star = " *" if abs(z) >= 2 else ""
                print(f"  {tags[i]:12s} - {tags[j]:12s}: Δ {dm:+.3f} ± {dse:.3f}  (z={z:+.2f}){star}")


if __name__ == "__main__":
    main()
