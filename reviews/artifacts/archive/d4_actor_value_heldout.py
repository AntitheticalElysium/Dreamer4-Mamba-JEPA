"""Held-out imagined-value diagnostic for the selected D4-lite actor.

This analysis performs no update. It loads the strict world/BC/actor pairing,
starts rollouts from a separate held-out replay, and measures the selected
value head against freshly generated TD-lambda targets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from d4_mamba_jepa.cartpole_baseline import (
    _atomic_json,
    load_cartpole_replay,
)
from d4_mamba_jepa.checkpoint import file_sha256
from d4_mamba_jepa.imagination_actor_critic import (
    ReplayContextSampler,
    decode_symlog_distribution,
    freeze_module,
    imagine_trajectory,
    load_imagination_actor_critic,
    module_state_sha256,
    td_lambda_returns,
    twohot_symlog_targets,
)
from d4_mamba_jepa.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--world-checkpoint-sha256", required=True)
    parser.add_argument("--bc-checkpoint-sha256", required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint-sha256", required=True)
    parser.add_argument("--heldout-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260799)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.batches < 1:
        raise ValueError("batches must be positive")
    device = torch.device(args.device)
    world, _, _ = load_checkpoint(
        args.world_checkpoint,
        device=device,
        expected_sha256=args.world_checkpoint_sha256,
        strict_implementation=False,
    )
    freeze_module(world)
    actor, prior, value, actor_payload = load_imagination_actor_critic(
        args.actor_checkpoint,
        expected_sha256=args.actor_checkpoint_sha256,
        expected_world_sha256=args.world_checkpoint_sha256,
        expected_bc_sha256=args.bc_checkpoint_sha256,
        device=device,
    )
    replay, records = load_cartpole_replay(args.heldout_replay)
    sampler = ReplayContextSampler(
        replay,
        context=args.context,
        device=device,
        seed=args.seed,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    hashes_before = {
        "world": module_state_sha256(world),
        "actor": module_state_sha256(actor),
        "prior": module_state_sha256(prior),
        "value": module_state_sha256(value),
    }

    rows: list[dict[str, float | int]] = []
    with torch.inference_mode():
        for _ in range(args.batches):
            trajectory = imagine_trajectory(
                world,
                actor,
                sampler.sample(args.batch_size),
                horizon=args.horizon,
                denoise_steps=args.denoise_steps,
                context=args.context,
                generator=generator,
                device=device,
            )
            logits, centers = value(trajectory.states.float())
            values = decode_symlog_distribution(logits, centers)
            returns = td_lambda_returns(
                trajectory.rewards.float(),
                trajectory.continues.float(),
                values,
                gamma=args.gamma,
                lambda_=args.lambda_,
            )
            targets = twohot_symlog_targets(returns, centers)
            loss = -(
                targets
                * logits[:, :-1].float().log_softmax(dim=-1)
            ).sum(dim=-1)
            advantages = returns - values[:, :-1]
            tensors = (
                trajectory.states,
                trajectory.rewards,
                trajectory.continues,
                logits,
                values,
                returns,
                targets,
                loss,
            )
            if not all(bool(torch.isfinite(item).all()) for item in tensors):
                raise RuntimeError("non-finite held-out value diagnostic")
            rows.append(
                {
                    "value_cross_entropy": float(loss.mean().item()),
                    "value_mae": float(
                        (values[:, :-1] - returns).abs().mean().item()
                    ),
                    "mean_value": float(values[:, :-1].mean().item()),
                    "mean_return": float(returns.mean().item()),
                    "return_std": float(
                        returns.std(unbiased=False).item()
                    ),
                    "mean_reward": float(
                        trajectory.rewards.mean().item()
                    ),
                    "mean_continue": float(
                        trajectory.continues.mean().item()
                    ),
                    "positive_advantages": int(
                        (advantages >= 0).sum().item()
                    ),
                    "negative_advantages": int(
                        (advantages < 0).sum().item()
                    ),
                }
            )

    hashes_after = {
        "world": module_state_sha256(world),
        "actor": module_state_sha256(actor),
        "prior": module_state_sha256(prior),
        "value": module_state_sha256(value),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("diagnostic changed a frozen tensor")
    means = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "value_cross_entropy",
            "value_mae",
            "mean_value",
            "mean_return",
            "return_std",
            "mean_reward",
            "mean_continue",
        )
    }
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("positive_advantages", "negative_advantages")
    }
    report = {
        "format": "d4_lite_actor_value_heldout_v1",
        "status": "PASS",
        "claim_boundary": (
            "finite value predictions and TD-lambda targets on fresh "
            "imagined trajectories from a separate replay; no update"
        ),
        "config": {
            "batches": args.batches,
            "batch_size": args.batch_size,
            "context": args.context,
            "horizon": args.horizon,
            "denoise_steps": args.denoise_steps,
            "gamma": args.gamma,
            "lambda": args.lambda_,
            "seed": args.seed,
        },
        "metrics": {**means, **totals},
        "rows": rows,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "provenance": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": file_sha256(Path(__file__)),
            "world_checkpoint": str(args.world_checkpoint),
            "world_checkpoint_sha256": args.world_checkpoint_sha256,
            "bc_checkpoint_sha256": args.bc_checkpoint_sha256,
            "actor_checkpoint": str(args.actor_checkpoint),
            "actor_checkpoint_sha256": args.actor_checkpoint_sha256,
            "actor_training_implementation_sha256": actor_payload[
                "provenance"
            ]["current_implementation_sha256"],
            "heldout_replay": str(args.heldout_replay),
            "heldout_replay_sha256": file_sha256(args.heldout_replay),
            "heldout_replay_episodes": len(records),
        },
    }
    _atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
