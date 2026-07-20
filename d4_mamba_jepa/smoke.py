"""One-update source, shape, gradient, and VRAM smoke for any factorial arm."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import torch

from .config import D4LiteConfig
from .diagnostics import moving_square_batch
from .model import D4LiteWorld
from .objectives import optimizer_groups
from .source import source_report
from .training import WorldLossNormalizer, world_loss


ARMS = {
    "T-BASE": ("transformer", "base"),
    "M-BASE": ("mamba2", "base"),
    "T-CDP": ("transformer", "cdp"),
    "M-CDP": ("mamba2", "cdp"),
}


def run_smoke(arm: str, device: torch.device) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
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

    world = D4LiteWorld(cfg).to(device)
    normalizer = WorldLossNormalizer().to(device)
    groups = optimizer_groups(world, 1e-4)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    batch = moving_square_batch(cfg, batch_size=2, device=device, seed=7)

    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    amp = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp,
    ):
        loss, metrics = world_loss(world, batch, normalizer=normalizer)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in world.parameters()
    )
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "arm": arm,
        "config": cfg.to_dict(),
        "sources": source_report(),
        "parameters": {
            "total": sum(parameter.numel() for parameter in world.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in world.parameters()
                if parameter.requires_grad
            ),
        },
        "losses": {
            key: float(value.item()) for key, value in metrics.items()
        },
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": finite_gradients,
        "seconds": elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), default="T-BASE")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    report = run_smoke(args.arm, torch.device(args.device))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["finite_loss"] or not report["finite_gradients"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
