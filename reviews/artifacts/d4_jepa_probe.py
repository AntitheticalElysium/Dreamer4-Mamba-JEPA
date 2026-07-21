"""Gate-2 probe: is the imagined return policy-dependent?

Rolls BC, anti-BC, and uniform policies in imagination on a given world and
reports the imagined discounted return of each. For the generative T-BASE world
these are nearly identical (the world imagines ~flat returns); the non-generative
JEPA world should make good and bad control diverge, which is the whole point of
going non-generative.

Usage:
  python d4_jepa_probe.py --world <world.pt> --world-sha <sha> --bc <bc.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch import nn

from d4_mamba_jepa.cartpole_baseline import load_cartpole_replay, load_bc_policy
from d4_mamba_jepa.checkpoint import file_sha256, load_checkpoint
from d4_mamba_jepa.imagination_actor_critic import (
    ReplayContextSampler,
    imagine_trajectory,
)

MAIN = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"
REPLAY = f"{MAIN}/outputs/d4_mamba_jepa/cartpole_baseline_v3/expert_dev_replay.pt"
GAMMA, HORIZON, DENOISE, CONTEXT, BATCH = 0.997, 32, 4, 8, 64


class Scaled(nn.Module):
    def __init__(self, base: nn.Module, scale: float):
        super().__init__()
        self.base, self.scale = base, scale

    def forward(self, x):
        return self.base(x) * self.scale


def disc_return(traj) -> torch.Tensor:
    B, H = traj.rewards.shape
    survival = torch.ones(B, device=traj.rewards.device)
    total = torch.zeros(B, device=traj.rewards.device)
    for t in range(H):
        total = total + (GAMMA ** t) * survival * traj.rewards[:, t]
        survival = survival * traj.continues[:, t]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--world-sha", required=True)
    ap.add_argument("--bc", required=True)
    ap.add_argument("--bc-world-sha", default=None,
                    help="world sha the BC was paired to (defaults to --world-sha)")
    args = ap.parse_args()
    bc_world_sha = args.bc_world_sha or args.world_sha
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, _, _ = load_checkpoint(
        args.world, device=dev, expected_sha256=args.world_sha,
        strict_implementation=False,
    )
    world = world.to(dev).eval()
    bc, _ = load_bc_policy(
        Path(args.bc), expected_sha256=file_sha256(Path(args.bc)),
        expected_world_sha256=bc_world_sha, device=dev,
    )
    replay, _ = load_cartpole_replay(Path(REPLAY))
    sampler = ReplayContextSampler(replay, context=CONTEXT, device=dev, seed=7)
    batch = sampler.sample(BATCH)

    print(f"arm={world.cfg.arm_id} batch={BATCH} horizon={HORIZON}")
    results = {}
    for name, pol in (
        ("BC (good)", bc),
        ("anti-BC (bad)", Scaled(bc, -4.0).to(dev).eval()),
        ("uniform", Scaled(bc, 0.0).to(dev).eval()),
    ):
        gen = torch.Generator(device=dev).manual_seed(123)
        traj = imagine_trajectory(
            world, pol, batch, horizon=HORIZON, denoise_steps=DENOISE,
            context=CONTEXT, generator=gen, device=dev,
        )
        R = disc_return(traj)
        results[name] = float(R.mean())
        print(
            f"{name:14s} | imagined disc-return mean={R.mean():.3f} std={R.std():.3f}"
            f" | reward/step={traj.rewards.mean():.3f}"
            f" | continue mean={traj.continues.mean():.5f} min={traj.continues.min():.5f}"
        )
    gap = results["BC (good)"] - results["anti-BC (bad)"]
    print(f"GOOD_MINUS_BAD_IMAGINED_RETURN_GAP = {gap:.3f}")


if __name__ == "__main__":
    main()
