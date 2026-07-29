"""Is the actor-vs-BC gate ceiling-limited, or is the actor simply not better?

Runs direct greedy execution of the imagination actor and its paired BC prior on
the SAME seeds, at several CartPole time limits. The execution path, context,
policies and checkpoints are identical to the sealed evaluator; the only thing
that varies is the environment's episode cap.

If the actor-minus-BC delta grows once the cap is lifted, the sealed gate was
measuring a saturated benchmark. If the delta stays flat near zero, the actor is
genuinely not better and no protocol change would rescue it.

This is a DIAGNOSTIC on development seeds. It never touches the sealed tiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d4_mamba_jepa.cartpole_baseline import (
    ACTION_REPEAT,
    CartPolePixels,
    _bc_policy_action,
    load_bc_policy,
    paired_bootstrap_interval,
)
from d4_mamba_jepa.checkpoint import file_sha256, load_checkpoint
from d4_mamba_jepa.imagination_actor_critic import load_imagination_actor_critic

CONTEXT = 8


class CappedPixels(CartPolePixels):
    """CartPolePixels with an explicit episode cap (default v1 value is 500)."""

    def __init__(self, *, image_size: int = 64, max_episode_steps: int = 500):
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import gymnasium as gym

        from d4_mamba_jepa.cartpole_baseline import ENVIRONMENT_ID
        from d4_mamba_jepa.source import verify_installed_cartpole

        verify_installed_cartpole()
        self.env = gym.make(
            ENVIRONMENT_ID,
            render_mode="rgb_array",
            max_episode_steps=int(max_episode_steps),
        )
        if int(self.env.action_space.n) != 2:
            raise RuntimeError("CartPole action contract drift")
        self.image_size = int(image_size)
        self.state = None
        self.previous_rgb = None


@torch.inference_mode()
def run_episode(world, policy, *, seed: int, device, cap: int) -> float:
    env = CappedPixels(image_size=world.cfg.image_size, max_episode_steps=cap)
    try:
        observation = env.reset(seed=seed)
        observations, led = [observation], [-1]
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = _bc_policy_action(
                world, policy,
                observations=observations, led_to_actions=led,
                context=min(CONTEXT, len(observations)), device=device,
            )
            observation, reward, _, terminated, truncated = env.step(action)
            observations.append(observation)
            led.append(action)
            total += reward
        return total
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--bc", type=Path, required=True)
    ap.add_argument("--actor", type=Path, required=True)
    ap.add_argument("--seeds", default="970000:970030")
    ap.add_argument("--caps", default="500,1000,2000")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsha = file_sha256(args.world)
    world, _, _ = load_checkpoint(
        args.world, device=device, expected_sha256=wsha,
        strict_implementation=False,
    )
    world.eval()
    bc, _ = load_bc_policy(
        args.bc, expected_sha256=file_sha256(args.bc),
        expected_world_sha256=wsha, device=device,
    )
    actor, prior, _, _ = load_imagination_actor_critic(
        args.actor, expected_sha256=file_sha256(args.actor),
        expected_world_sha256=wsha, expected_bc_sha256=file_sha256(args.bc),
        device=device,
    )
    start, stop = args.seeds.split(":")
    seeds = list(range(int(start), int(stop)))
    caps = [int(c) for c in args.caps.split(",")]

    print(f"arm={world.cfg.arm_id}  seeds={len(seeds)}  caps={caps}")
    report = {"arm_id": world.cfg.arm_id, "seeds": seeds, "caps": {}}
    print(f"\n{'cap':>6} {'actor':>9} {'BC':>9} {'delta':>9} {'95% CI':>22} "
          f"{'BC@cap':>8} {'both@cap':>9}")
    for cap in caps:
        a = [run_episode(world, actor, seed=s, device=device, cap=cap) for s in seeds]
        b = [run_episode(world, bc, seed=s, device=device, cap=cap) for s in seeds]
        d = [x - y for x, y in zip(a, b)]
        ci = paired_bootstrap_interval(d, seed=cap)
        at_cap = sum(1 for y in b if y >= cap)
        both = sum(1 for x, y in zip(a, b) if x >= cap and y >= cap)
        report["caps"][cap] = {
            "actor_mean": float(np.mean(a)), "bc_mean": float(np.mean(b)),
            "delta_mean": float(np.mean(d)), "ci": ci,
            "ci_above_zero": ci[0] > 0,
            "bc_at_cap": at_cap, "both_at_cap": both, "n": len(seeds),
            "actor_returns": a, "bc_returns": b,
        }
        print(f"{cap:>6} {np.mean(a):>9.2f} {np.mean(b):>9.2f} {np.mean(d):>+9.2f} "
              f"[{ci[0]:>9.2f},{ci[1]:>9.2f}] {at_cap:>4}/{len(seeds):<3} "
              f"{both:>5}/{len(seeds):<3}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
