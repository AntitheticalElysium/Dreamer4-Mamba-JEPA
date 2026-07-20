"""Bounded overfit ladder before any Crafter-scale training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import torch

from .config import D4LiteConfig
from .diagnostics import moving_square_batch
from .model import D4LiteWorld, build_tokenizer
from .objectives import optimizer_groups
from .smoke import ARMS
from .training import (
    WorldLossNormalizer,
    tokenizer_full_reconstruction_mse,
    tokenizer_reconstruction_loss,
    world_loss,
)


def run_overfit(
    *,
    arm: str,
    device: torch.device,
    tokenizer_steps: int,
    world_steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    temporal, objective = ARMS[arm]
    cfg = replace(
        D4LiteConfig(),
        temporal_backend=temporal,
        representation_objective=objective,
    )
    torch.manual_seed(20260720)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260720)
        torch.cuda.reset_peak_memory_stats(device)
    batch = moving_square_batch(
        cfg, batch_size=batch_size, device=device, seed=1701
    )

    tokenizer = build_tokenizer(cfg, training_mask=True).to(device).train()
    tokenizer_optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=learning_rate, weight_decay=1e-2
    )
    initial_reconstruction = float(
        tokenizer_full_reconstruction_mse(
            tokenizer, batch.observations, patch_size=cfg.patch_size
        ).item()
    )
    tokenizer_history = []
    start = time.perf_counter()
    for step in range(tokenizer_steps):
        tokenizer_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _ = tokenizer_reconstruction_loss(
                tokenizer, batch.observations, patch_size=cfg.patch_size
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        tokenizer_optimizer.step()
        tokenizer_history.append(float(loss.detach().item()))
    tokenizer_seconds = time.perf_counter() - start
    final_reconstruction = float(
        tokenizer_full_reconstruction_mse(
            tokenizer, batch.observations, patch_size=cfg.patch_size
        ).item()
    )

    world = D4LiteWorld(cfg).to(device)
    world.encoder.load_state_dict(tokenizer.encoder.state_dict(), strict=True)
    world.decoder.load_state_dict(tokenizer.decoder.state_dict(), strict=True)
    if objective == "base":
        world.freeze_tokenizer()
    groups = optimizer_groups(world, learning_rate)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    normalizer = WorldLossNormalizer().to(device)

    world_history: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(world_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, metrics = world_loss(
                world,
                batch,
                normalizer=normalizer,
                global_step=step,
                bootstrap_rows=0,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in groups for parameter in group["params"]],
            1.0,
        )
        optimizer.step()
        world_history.append(
            {
                key: float(value.item())
                for key, value in metrics.items()
                if key.startswith("loss/")
            }
        )
    world_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    window = max(1, min(10, len(world_history)))
    first = world_history[:window]
    last = world_history[-window:]

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    report = {
        "arm": arm,
        "tokenizer": {
            "steps": tokenizer_steps,
            "initial_full_mse": initial_reconstruction,
            "final_full_mse": final_reconstruction,
            "first_masked_loss": tokenizer_history[0],
            "last_masked_loss": tokenizer_history[-1],
            "seconds": tokenizer_seconds,
        },
        "world": {
            "steps": world_steps,
            "first_window": {
                key: mean(first, key)
                for key in first[0]
            },
            "last_window": {
                key: mean(last, key)
                for key in last[0]
            },
            "seconds": world_seconds,
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), default="T-BASE")
    parser.add_argument("--tokenizer-steps", type=int, default=100)
    parser.add_argument("--world-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    report = run_overfit(
        arm=args.arm,
        device=torch.device(args.device),
        tokenizer_steps=args.tokenizer_steps,
        world_steps=args.world_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
