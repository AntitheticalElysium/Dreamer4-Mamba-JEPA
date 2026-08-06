"""Full Craftax expert-data run driver: world -> BC -> imagination, per arm.

This is the launcher for the first Craftax baseline. It changes NOTHING about
the architecture or the optimization recipe: every budget below is INHERITED
verbatim from the screened CartPole `T-JEPA`/`M-JEPA` protocol (D030-D037), so
the only thing that moves between the CartPole result and this one is the
environment and its data. Inherited-and-unexamined quantities are listed in the
report under ``inherited`` so they are visible as ablation candidates rather
than as choices:

  * ``jepa_jumps=5``          selected on CartPole under the D043 confound (D041)
  * ``jepa_terminal_fraction=0.5`` inherited from D035
  * ``sequence_length=16``    the ``D4LiteConfig`` default (CartPole used 12)
  * world 20k x batch 8, BC 3k x batch 16, actor 500 x batch 64 / horizon 32

Episodes are split 80/10/10 into train/dev/sealed by WHOLE episode before any
training; sealed indices are recorded and never loaded into a training replay.

Torch only -- it consumes the hash-pinned replay produced offline and never
imports JAX. Executed achievement evaluation is a separate job
(``craftax_achievement``), because that one does need the live environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .common import _clean_agent_tokens
from .checkpoint import file_sha256, implementation_sha256
from .craftax_runners import (
    craftax_jepa_config,
    train_craftax_bc,
    train_craftax_imagination,
    train_craftax_jepa_world,
)
from .data import (
    load_episode_replay,
    replay_sample_to_sequence,
    subset_replay,
    whole_episode_splits,
)
from .objectives import jepa_self_prediction_loss
from .source import craftax_source_report, source_report

FORMAT = "d4_mamba_jepa_craftax_run_v1"
SPLIT_SEED = 20260727

# Every value below is inherited from the screened CartPole JEPA protocol.
WORLD_STEPS = 20_000
WORLD_BATCH = 8
BC_STEPS = 3_000
BC_BATCH = 16
ACTOR_STEPS = 500
ACTOR_BATCH = 64
ACTOR_CONTEXT = 8
ACTOR_HORIZON = 32
LEARNING_RATE = 1e-4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fixed_dev_batches(dev_replay, *, cfg, count, batch_size, seed):
    """Materialize a fixed dev batch list once so every arm sees identical data."""
    rng = np.random.default_rng(seed)
    return [
        replay_sample_to_sequence(
            dev_replay.sample(
                batch_size, cfg.sequence_length, torch.device("cpu"), rng=rng
            )
        )
        for _ in range(count)
    ]


@torch.no_grad()
def _dev_cosine(world, batches, device) -> float:
    """Held-out self-prediction cosine (the world-quality read-out)."""
    was_training = world.training
    # SIGReg resamples projections by incrementing an internal global-step
    # buffer. Evaluation must not advance that training state.
    sigreg_step = None
    if world.sigreg_test is not None:
        sigreg_step = world.sigreg_test.global_step.detach().clone()
    world.eval()
    try:
        values = []
        for batch in batches:
            observations = batch.observations.to(device)
            actions = batch.led_to_actions.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                clean = world.encode_frames(observations, frozen=True).packed
                _, metric = jepa_self_prediction_loss(
                    world, frames=observations, clean=clean, led_to_actions=actions,
                )
            values.append(float(metric["jepa_cosine"].item()))
    finally:
        if sigreg_step is not None:
            world.sigreg_test.global_step.copy_(sigreg_step)
        world.train(was_training)
    return float(np.mean(values))


@torch.no_grad()
def _dev_bc_accuracy(world, policy, batches, device) -> float:
    """Held-out demonstration-action accuracy of the frozen BC head."""
    world.eval()
    policy.eval()
    correct = total = 0
    for batch in batches:
        observations = batch.observations.to(device)
        actions = batch.led_to_actions.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            agent = _clean_agent_tokens(
                world,
                type(batch)(
                    observations=observations,
                    led_to_actions=actions,
                    led_to_rewards=batch.led_to_rewards.to(device),
                    led_to_continues=batch.led_to_continues.to(device),
                    outcome_valid=batch.outcome_valid.to(device),
                ),
            )
            logits = policy(agent.float())
        predicted = logits[:, :-1].float().argmax(dim=-1)
        target = actions[:, 1:]
        correct += int((predicted == target).sum().item())
        total += int(target.numel())
    policy.train()
    return correct / max(1, total)


def run_arm(
    *,
    temporal_backend: str,
    train_replay,
    dev_replay,
    output_dir: Path,
    device: torch.device,
    seed: int,
    world_steps: int,
    bc_steps: int,
    actor_steps: int,
    encoder_learning_rate: float | None = None,
) -> dict:
    """world -> BC -> imagination for one temporal backend."""
    cfg = craftax_jepa_config(temporal_backend)
    arm_dir = output_dir / cfg.arm_id.lower().replace("-", "_")
    arm_dir.mkdir(parents=True, exist_ok=True)
    dev_batches = _fixed_dev_batches(
        dev_replay, cfg=cfg, count=16, batch_size=8, seed=SPLIT_SEED + 1
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[{cfg.arm_id}] world: {world_steps} updates", flush=True)
    started = time.perf_counter()
    world, _, world_history = train_craftax_jepa_world(
        replay=train_replay, cfg=cfg, world_steps=world_steps,
        batch_size=WORLD_BATCH, learning_rate=LEARNING_RATE,
        encoder_learning_rate=encoder_learning_rate, seed=seed,
        device=device, ema_schedule_steps=world_steps, output_dir=arm_dir,
    )
    world_seconds = time.perf_counter() - started
    world_sha = json.loads((arm_dir / "world_report.json").read_text())[
        "world_checkpoint_sha256"
    ]
    dev_cosine = _dev_cosine(world, dev_batches, device)
    print(f"[{cfg.arm_id}] world done in {world_seconds / 60:.1f} min, "
          f"dev cosine {dev_cosine:.4f}", flush=True)

    print(f"[{cfg.arm_id}] BC: {bc_steps} updates", flush=True)
    started = time.perf_counter()
    bc, bc_losses = train_craftax_bc(
        world=world, replay=train_replay, steps=bc_steps, batch_size=BC_BATCH,
        learning_rate=LEARNING_RATE, seed=seed + 1, device=device,
        output_dir=arm_dir, world_checkpoint_sha256=world_sha,
    )
    bc_seconds = time.perf_counter() - started
    bc_accuracy = _dev_bc_accuracy(world, bc, dev_batches, device)
    print(f"[{cfg.arm_id}] BC done in {bc_seconds / 60:.1f} min, "
          f"held-out action accuracy {bc_accuracy:.4f}", flush=True)

    print(f"[{cfg.arm_id}] imagination: {actor_steps} updates", flush=True)
    started = time.perf_counter()
    _, _, actor_history = train_craftax_imagination(
        world=world, bc=bc, replay=train_replay, steps=actor_steps,
        batch_size=ACTOR_BATCH, context=ACTOR_CONTEXT, horizon=ACTOR_HORIZON,
        learning_rate=LEARNING_RATE, seed=seed + 2, device=device,
        output_dir=arm_dir, world_checkpoint_sha256=world_sha,
    )
    actor_seconds = time.perf_counter() - started
    print(f"[{cfg.arm_id}] imagination done in {actor_seconds / 60:.1f} min",
          flush=True)

    tail = world_history[-100:]
    report = {
        "arm_id": cfg.arm_id,
        "config": asdict(cfg),
        "world": {
            "updates": world_steps,
            "batch_size": WORLD_BATCH,
            "learning_rate": LEARNING_RATE,
            "encoder_learning_rate": (
                LEARNING_RATE
                if encoder_learning_rate is None
                else encoder_learning_rate
            ),
            "separate_encoder_group": encoder_learning_rate is not None,
            "seconds": world_seconds,
            "checkpoint_sha256": world_sha,
            "final_jepa_loss": world_history[-1]["jepa"],
            "train_cosine_last_100": float(np.mean([r["cosine"] for r in tail])),
            "online_std_last_100": float(np.mean([r["online_std"] for r in tail])),
            "dev_cosine": dev_cosine,
        },
        "bc": {
            "updates": bc_steps,
            "batch_size": BC_BATCH,
            "seconds": bc_seconds,
            "first_loss": bc_losses[0],
            "last_loss": bc_losses[-1],
            "dev_action_accuracy": bc_accuracy,
        },
        "imagination": {
            "updates": actor_steps,
            "batch_size": ACTOR_BATCH,
            "context": ACTOR_CONTEXT,
            "horizon": ACTOR_HORIZON,
            "seconds": actor_seconds,
            "final": actor_history[-1] if actor_history else None,
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
    }
    _write_json(arm_dir / "arm_report.json", report)
    return report


def run(
    *,
    replay_path: Path,
    replay_sha256: str,
    output_dir: Path,
    device: torch.device,
    seed: int,
    world_steps: int,
    bc_steps: int,
    actor_steps: int,
    backends: list[str],
    encoder_learning_rate: float | None = None,
) -> dict:
    replay = load_episode_replay(replay_path, expected_sha256=replay_sha256)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])
    # `sealed` is deliberately never materialized into a replay in this process.
    print(
        f"replay {replay_path.name}: {len(replay.episodes)} episodes / "
        f"{replay.steps} transitions -> train {len(splits['train'])} "
        f"({train_replay.steps}), dev {len(splits['dev'])} ({dev_replay.steps}), "
        f"sealed {len(splits['sealed'])} (held back)",
        flush=True,
    )
    terminal_train = sum(
        1 for ep in train_replay.episodes if float(ep.continues[-1]) == 0.0
    )
    print(f"train terminal episodes: {terminal_train}/{len(train_replay.episodes)}",
          flush=True)

    arms = {}
    for backend in backends:
        result = run_arm(
            temporal_backend=backend, train_replay=train_replay,
            dev_replay=dev_replay, output_dir=output_dir, device=device,
            seed=seed, world_steps=world_steps, bc_steps=bc_steps,
            actor_steps=actor_steps,
            encoder_learning_rate=encoder_learning_rate,
        )
        arms[result["arm_id"]] = result
        _write_json(output_dir / "report.json", {"partial": True, "arms": arms})

    payload = {
        "format": FORMAT,
        "status": "completed",
        "claim_boundary": (
            (
                "Craftax expert-data encoder-timescale ablation of the screened "
                "JEPA recipe; the world encoder LR is the only moved training "
                "axis, BC and imagination remain unchanged, no executed "
                "evaluation, and no selection performed. Sealed episodes untouched."
            )
            if encoder_learning_rate is not None
            else (
                "first Craftax expert-data run of the screened CartPole JEPA "
                "recipe; no recipe or architecture change, no executed evaluation, "
                "and no selection performed. Sealed episodes untouched."
            )
        ),
        "data": {
            "replay_path": str(replay_path),
            "replay_sha256": replay_sha256,
            "episodes": len(replay.episodes),
            "transitions": int(replay.steps),
            "split_seed": SPLIT_SEED,
            "splits": splits,
            "train_terminal_episodes": terminal_train,
        },
        "inherited": {
            "note": "carried from the CartPole protocol unexamined; ablation candidates",
            "jepa_jumps": arms[next(iter(arms))]["config"]["jepa_jumps"],
            "jepa_terminal_fraction": arms[next(iter(arms))]["config"][
                "jepa_terminal_fraction"
            ],
            "sequence_length": arms[next(iter(arms))]["config"]["sequence_length"],
            "world_updates": world_steps,
            "bc_updates": bc_steps,
            "actor_updates": actor_steps,
            "actor_horizon": ACTOR_HORIZON,
            "world_learning_rate": LEARNING_RATE,
            "encoder_learning_rate": (
                LEARNING_RATE
                if encoder_learning_rate is None
                else encoder_learning_rate
            ),
        },
        "intervention": {
            "axis": (
                "encoder_learning_rate"
                if encoder_learning_rate is not None
                else "none"
            ),
            "encoder_learning_rate": encoder_learning_rate,
            "bc_and_imagination_world_frozen": True,
        },
        "provenance": {
            "implementation_sha256": implementation_sha256(),
            # `source_report()` (no config) keeps the historical three-source
            # block so this report's schema is unchanged. The Craftax digests
            # are recorded alongside because the run spans both backends and
            # the environment is not in that legacy block.
            "sources": source_report(),
            "craftax": craftax_source_report(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "arms": arms,
    }
    _write_json(output_dir / "report.json", payload)
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Craftax expert-data run.")
    parser.add_argument(
        "--replay", type=Path,
        default=repo_root / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt",
    )
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=repo_root / "outputs/d4_mamba_jepa/craftax_expert_v1",
    )
    parser.add_argument("--world-steps", type=int, default=WORLD_STEPS)
    parser.add_argument("--bc-steps", type=int, default=BC_STEPS)
    parser.add_argument("--actor-steps", type=int, default=ACTOR_STEPS)
    parser.add_argument(
        "--encoder-lr", type=float, default=None,
        help=(
            "optional world-encoder AdamW group LR; all other world parameters "
            "stay at 1e-4, while BC/imagination continue to freeze the world"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--backends", default="transformer,mamba2",
        help="comma-separated temporal backends to run in order",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    payload = run(
        replay_path=args.replay,
        replay_sha256=args.replay_sha256,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        seed=args.seed,
        world_steps=args.world_steps,
        bc_steps=args.bc_steps,
        actor_steps=args.actor_steps,
        backends=[b.strip() for b in args.backends.split(",") if b.strip()],
        encoder_learning_rate=args.encoder_lr,
    )
    print(json.dumps({
        arm: {
            "world": result["world"],
            "bc": result["bc"],
            "imagination": result["imagination"],
        }
        for arm, result in payload["arms"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["FORMAT", "run", "run_arm"]
