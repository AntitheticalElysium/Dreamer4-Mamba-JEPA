"""Real-Crafter evaluation: run the imagination-trained policy in the ACTUAL env and
compare to a random baseline. The true test of the artifact — does training a policy
inside the JEPA dream produce a policy that beats random in reality?

Filtering/acting loop (mirrors Dreamer): encode the real observation (motion clip) ->
update the WM recurrent state -> policy acts from [z_real, h].
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
from box import Box
from agents import ActorCriticAgent
from dynamics import JEPADynamics
from vjepa_encoder import VJEPAEncoder
from train_agent import shim_config


@torch.inference_mode()
def run_episodes(policy_fn, n_eps, seed, max_steps=1000):
    """policy_fn(reset) -> callable(obs)->action, plus per-episode reset hook.
    Returns list of (episode_reward, n_achievements)."""
    results = []
    for ep in range(n_eps):
        env = crafter.Env(seed=seed + ep)
        obs = env.reset()
        policy_fn("reset")
        total, ach = 0.0, set()
        for _ in range(max_steps):
            a = policy_fn(obs)
            obs, r, done, info = env.step(a)
            total += r
            for k, v in info.get("achievements", {}).items():
                if v > 0:
                    ach.add(k)
            if done:
                break
        results.append((total, len(ach)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm.pt"))
    ap.add_argument("--agent", type=str, default=os.path.join(REPO, "ckpt", "jepa_agent.pt"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    model = JEPADynamics(latent_dim=ck["latent_dim"], action_dim=ck["action_dim"], n_heads=5).to(dev).eval()
    model.load_state_dict(ck["model"])
    mean = torch.tensor(ck["mean"], device=dev); std = torch.tensor(ck["std"], device=dev)
    agent = ActorCriticAgent(shim_config(ck["latent_dim"], model.d_model), ck["action_dim"], dev).to(dev)
    agent.load_state_dict(torch.load(args.agent, map_location=dev, weights_only=False))
    agent.eval()
    enc = VJEPAEncoder(res=args.res, pool="mean", device=dev)

    class Actor:
        def __init__(self, greedy): self.greedy = greedy
        def __call__(self, obs):
            if isinstance(obs, str):  # reset hook
                self.states = model.init_state(1, 4096, dev)
                self.h = torch.zeros(1, 1, model.d_model, device=dev)
                self.prev = None
                return None
            prev = obs if self.prev is None else self.prev
            z = enc.encode(np.stack([prev, obs]), motion=True)[1:2].to(dev)   # (1,D) motion latent of obs
            z = ((z - mean) / std).unsqueeze(1)                              # (1,1,D)
            agent_in = torch.cat([z.squeeze(1), self.h.squeeze(1)], dim=-1)  # (1, D+Dh)
            action, _ = agent.sample(agent_in, greedy=self.greedy)
            a = int(action.item())
            _, self.h, self.states = model.step(z, action, self.states)     # update state with real z + a
            self.prev = obs
            return a

    print(f"evaluating {args.episodes} episodes each...")
    pol = run_episodes(Actor(greedy=False), args.episodes, seed=100)
    rng = np.random.default_rng(0)
    rand = run_episodes(lambda o: None if isinstance(o, str) else int(rng.integers(17)),
                        args.episodes, seed=100)
    pr = np.array(pol); rr = np.array(rand)
    print(f"POLICY : reward {pr[:,0].mean():+.2f}±{pr[:,0].std():.2f}   achievements {pr[:,1].mean():.2f}")
    print(f"RANDOM : reward {rr[:,0].mean():+.2f}±{rr[:,0].std():.2f}   achievements {rr[:,1].mean():.2f}")


if __name__ == "__main__":
    main()
