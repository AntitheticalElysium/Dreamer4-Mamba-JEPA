"""Measure steady-state PyTorch CUDA peaks for each pre-training phase.

This is a synthetic-shape memory probe, not a learning experiment. It warms each
optimizer/kernel once, then reports allocated and reserved peaks for one update.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from agent import ActorCritic  # noqa: E402
from model import M3HJWM, ModelConfig  # noqa: E402
from train import TrainConfig, actor_critic_update, world_update  # noqa: E402


def synchronize():
    torch.cuda.synchronize()


def random_batch(cfg: ModelConfig, batch: int, observations: int, device):
    return {
        "obs": torch.randint(
            0,
            256,
            (batch, observations, 3, cfg.image_size, cfg.image_size),
            dtype=torch.uint8,
            device=device,
        ),
        "actions": torch.randint(
            0, cfg.action_dim, (batch, observations - 1), device=device
        ),
        "rewards": torch.randn(batch, observations - 1, device=device),
        "continues": torch.ones(batch, observations - 1, device=device),
    }


def phase_measure(fn):
    torch.cuda.empty_cache()
    synchronize()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fn()
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "elapsed_ms": elapsed_ms,
        "before_allocated_mib": before_allocated / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "incremental_peak_allocated_mib": (peak_allocated - before_allocated) / 2**20,
        "before_reserved_mib": before_reserved / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gru", "mamba2"], required=True)
    parser.add_argument("--predictor", choices=["deterministic", "mixture"], required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--imagination-batch", type=int, default=16)
    parser.add_argument("--imagination-horizon", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    cfg = ModelConfig(
        patch_size=8,
        token_dim=64,
        registers=2,
        spatial_depth=1,
        spatial_heads=4,
        temporal_backend=args.backend,
        temporal_depth=1,
        mamba_d_state=32,
        mamba_headdim=16,
        predictor=args.predictor,
        predictor_depth=2,
        modes=2,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch,
        sequence_length=args.sequence,
        imagination_batch=args.imagination_batch,
        imagination_horizon=args.imagination_horizon,
        amp=True,
    )
    world = M3HJWM(cfg).to(device)
    agent = ActorCritic(cfg.token_dim, cfg.action_dim, critics=3).to(device)
    batch = random_batch(cfg, args.batch, args.sequence, device)
    world_optimizer = torch.optim.AdamW(world.parameters(), lr=train_cfg.world_lr)
    actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=train_cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(agent.critics.parameters(), lr=train_cfg.critic_lr)

    # Warm the world kernels and allocate optimizer moments.
    world_update(world, batch, world_optimizer, train_cfg)
    synchronize()
    world_result = phase_measure(
        lambda: world_update(world, batch, world_optimizer, train_cfg)
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    start_state = world.initial_state(
        args.imagination_batch, device, dtype=amp_dtype
    )
    actor_critic_update(
        world,
        agent,
        start_state,
        actor_optimizer,
        critic_optimizer,
        train_cfg,
    )
    synchronize()
    start_state = world.initial_state(
        args.imagination_batch, device, dtype=amp_dtype
    )
    actor_result = phase_measure(
        lambda: actor_critic_update(
            world,
            agent,
            start_state,
            actor_optimizer,
            critic_optimizer,
            train_cfg,
        )
    )

    free, total = torch.cuda.mem_get_info()
    result = {
        "configuration": {
            "backend": args.backend,
            "predictor": args.predictor,
            "batch": args.batch,
            "sequence": args.sequence,
            "streams": world.streams,
            "dim": cfg.token_dim,
            "imagination_batch": args.imagination_batch,
            "imagination_horizon": args.imagination_horizon,
            "amp_dtype": str(amp_dtype),
            "world_parameters": sum(parameter.numel() for parameter in world.parameters()),
            "agent_parameters": sum(parameter.numel() for parameter in agent.parameters()),
        },
        "world_update": world_result,
        "actor_critic_update": actor_result,
        "device_total_mib": total / 2**20,
        "device_free_after_mib": free / 2**20,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
