"""Reliable eval for the TRAINED-encoder Mamba-JEPA agent, in real Crafter. Paired seeds vs a
random baseline; reports achievements mean +/- SE. Mirrors eval_reliable.py's paired design but
for the {encoder, model, agent} checkpoint format.
"""
import os, sys, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)

import crafter
from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from agents import ActorCriticAgent
from train_agent import shim_config
from train_online_jepa import enc_one


@torch.no_grad()
def run_policy(enc, model, agent, dev, seeds, max_steps=1000):
    rew, ach = [], []
    for sd in seeds:
        env = crafter.Env(seed=int(sd)); obs = env.reset()
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        tot, acc = 0.0, set()
        for _ in range(max_steps):
            z, _ = enc_one(enc, obs, dev)
            action, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
            _, h, states = model.step(z.unsqueeze(1), action, states)
            obs, r, done, info = env.step(int(action.item())); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: acc.add(k)
            if done: break
        rew.append(tot); ach.append(len(acc))
    return np.array(rew), np.array(ach)


def run_random(seeds, max_steps=1000):
    rng = np.random.default_rng(0); rew, ach = [], []
    for sd in seeds:
        env = crafter.Env(seed=int(sd)); obs = env.reset(); tot, acc = 0.0, set()
        for _ in range(max_steps):
            obs, r, done, info = env.step(int(rng.integers(17))); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: acc.add(k)
            if done: break
        rew.append(tot); ach.append(len(acc))
    return np.array(rew), np.array(ach)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=str, required=True)
    ap.add_argument("--agent", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--base_seed", type=int, default=5000)
    args = ap.parse_args()
    dev = "cuda"
    seeds = np.arange(args.base_seed, args.base_seed + args.episodes)

    ck = torch.load(args.wm, map_location=dev, weights_only=False); D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    agent = ActorCriticAgent(shim_config(D, model.d_model), 17, dev).to(dev).eval()
    agent.load_state_dict(torch.load(args.agent, map_location=dev, weights_only=False))

    print(f"eval {args.episodes} paired episodes...", flush=True)
    pr, pa = run_policy(enc, model, agent, dev, seeds)
    rr, ra = run_random(seeds)
    dse = (pa - ra).std(ddof=1) / np.sqrt(len(pa))
    print(f"POLICY  ach {pa.mean():.3f} +/- {pa.std(ddof=1)/np.sqrt(len(pa)):.3f}   reward {pr.mean():+.2f}")
    print(f"RANDOM  ach {ra.mean():.3f} +/- {ra.std(ddof=1)/np.sqrt(len(ra)):.3f}   reward {rr.mean():+.2f}")
    print(f"PAIRED  delta ach {(pa-ra).mean():+.3f} +/- {dse:.3f}  (z={(pa-ra).mean()/dse:+.2f})")


if __name__ == "__main__":
    main()
