"""Bounded real-Crafter preflight for the unchanged Transformer baseline.

This runner is deliberately not a four-arm trainer. It answers a narrower
question first: can the source-pinned, reduced-scale D4-style baseline learn
the existing Crafter replay at all? Mamba and CDP remain disabled.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch
from torch import Tensor

from .checkpoint import (
    implementation_sha256,
    save_checkpoint,
    save_tokenizer_checkpoint,
)
from .config import D4LiteConfig
from .data import (
    SequenceBatch,
    load_episode_replay,
    replay_sample_to_sequence,
)
from .model import D4LiteWorld, build_tokenizer
from .objectives import optimizer_groups, shortcut_flow_loss
from .rollout import sample_next_packed, shortcut_schedule
from .source import load_mmbench2_model, source_report
from .training import (
    WorldLossNormalizer,
    tokenizer_full_reconstruction_mse,
    tokenizer_reconstruction_loss,
    world_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_REPLAY = REPO_ROOT / "data/replay_40k_v1.pt"
TRAIN_REPLAY_SHA256 = (
    "c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad"
)
DEV_REPLAY = REPO_ROOT / "data/heldout_20ep_v1.pt"
DEV_REPLAY_SHA256 = (
    "709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad"
)
RUNNER_FORMAT = "d4_mamba_jepa_crafter_preflight_v1"


def _to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        observations=batch.observations.to(device),
        led_to_actions=batch.led_to_actions.to(device),
        led_to_rewards=batch.led_to_rewards.to(device),
        led_to_continues=batch.led_to_continues.to(device),
        outcome_valid=batch.outcome_valid.to(device),
    )


def _sample_sequence(
    replay,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    rng: np.random.Generator,
) -> SequenceBatch:
    sampled = replay.sample(
        batch=batch_size,
        observations=sequence_length,
        device=device,
        rng=rng,
    )
    return replay_sample_to_sequence(sampled)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _pearson(prediction: Tensor, target: Tensor) -> float | None:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    if prediction.numel() < 2:
        return None
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return None
    return float((prediction * target).sum().div(denominator).item())


def _binary_auroc(score: Tensor, label: Tensor) -> float | None:
    """Exact Mann-Whitney AUROC for the small preflight readout."""
    score = score.float().reshape(-1)
    label = label.bool().reshape(-1)
    positive = score[label]
    negative = score[~label]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparison = positive[:, None] - negative[None, :]
    auc = (
        (comparison > 0).float() + 0.5 * (comparison == 0).float()
    ).mean()
    return float(auc.item())


def _decode_reward(logits: Tensor, centers: Tensor) -> Tensor:
    upstream = load_mmbench2_model()
    probabilities = logits.float().softmax(dim=-1)
    expected_symlog = (
        probabilities * centers.float().view(*([1] * (logits.ndim - 1)), -1)
    ).sum(dim=-1)
    return upstream.symexp(expected_symlog)


def _reward_metrics(prediction: Tensor, target: Tensor) -> dict:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    event = target != 0
    zero = ~event
    return {
        "rows": int(target.numel()),
        "event_rows": int(event.sum().item()),
        "mae": float((prediction - target).abs().mean().item()),
        "pearson": _pearson(prediction, target),
        "event_auroc_abs_prediction": _binary_auroc(
            prediction.abs(), event
        ),
        "zero_target_mean_prediction": (
            float(prediction[zero].mean().item()) if bool(zero.any()) else None
        ),
        "zero_target_mean_abs_prediction": (
            float(prediction[zero].abs().mean().item())
            if bool(zero.any())
            else None
        ),
    }


def _continuation_metrics(probability: Tensor, target: Tensor) -> dict:
    probability = probability.float().reshape(-1)
    target = target.float().reshape(-1)
    terminal = target == 0
    continuing = ~terminal
    return {
        "rows": int(target.numel()),
        "terminal_rows": int(terminal.sum().item()),
        "brier": float((probability - target).square().mean().item()),
        "terminal_auroc": _binary_auroc(1.0 - probability, terminal),
        "mean_p_continue_terminal": (
            float(probability[terminal].mean().item())
            if bool(terminal.any())
            else None
        ),
        "mean_p_continue_nonterminal": (
            float(probability[continuing].mean().item())
            if bool(continuing.any())
            else None
        ),
    }


def _fixed_eval_batches(
    replay,
    *,
    cfg: D4LiteConfig,
    count: int,
    batch_size: int,
    seed: int,
) -> list[SequenceBatch]:
    rng = np.random.default_rng(seed)
    return [
        _sample_sequence(
            replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            device=torch.device("cpu"),
            rng=rng,
        )
        for _ in range(count)
    ]


def _aligned_eval_batches(
    replay,
    *,
    target: str,
    context: int,
    batch_size: int,
) -> list[SequenceBatch]:
    """Put a reward event or terminal transition at one fixed generated depth."""
    if target not in {"reward_event", "terminal"}:
        raise ValueError(f"unsupported aligned target {target!r}")
    rows = []
    for episode in replay.episodes:
        transition_count = len(episode.actions)
        for transition in range(transition_count):
            selected = (
                episode.rewards[transition] != 0
                if target == "reward_event"
                else episode.continues[transition] == 0
            )
            if not selected:
                continue
            target_observation = transition + 1
            start = target_observation - context
            if start < 0:
                continue
            stop = target_observation + 1
            previous = np.full(context + 1, -1, dtype=np.int64)
            if start > 0:
                previous[0] = episode.actions[start - 1]
            previous[1:] = episode.actions[start:transition + 1]
            rows.append(
                {
                    "obs": episode.obs[start:stop],
                    "actions": episode.actions[start:transition + 1],
                    "rewards": episode.rewards[start:transition + 1],
                    "continues": episode.continues[start:transition + 1],
                    "previous_actions": previous,
                }
            )

    batches = []
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        sample = {
            name: torch.from_numpy(np.stack([row[name] for row in chunk]))
            for name in (
                "obs",
                "actions",
                "rewards",
                "continues",
                "previous_actions",
            )
        }
        batches.append(replay_sample_to_sequence(sample))
    return batches


@torch.inference_mode()
def evaluate_tokenizer(
    tokenizer,
    batches: list[SequenceBatch],
    *,
    cfg: D4LiteConfig,
    device: torch.device,
) -> dict:
    tokenizer.eval()
    values = []
    for cpu_batch in batches:
        frames = cpu_batch.observations.to(device)
        values.append(
            float(
                tokenizer_full_reconstruction_mse(
                    tokenizer, frames, patch_size=cfg.patch_size
                ).item()
            )
        )
    return {"full_reconstruction_mse": _mean(values), "batches": len(values)}


@torch.inference_mode()
def evaluate_world(
    world: D4LiteWorld,
    batches: list[SequenceBatch],
    *,
    cfg: D4LiteConfig,
    device: torch.device,
    seed: int,
) -> dict:
    """Evaluate real-state heads and deployed one-step generation separately."""
    world.eval()
    raw_losses: dict[str, list[float]] = {}
    real_rewards: list[Tensor] = []
    real_reward_targets: list[Tensor] = []
    real_continues: list[Tensor] = []
    real_continue_targets: list[Tensor] = []
    generated_rewards: list[Tensor] = []
    generated_reward_targets: list[Tensor] = []
    generated_continues: list[Tensor] = []
    generated_continue_targets: list[Tensor] = []
    actual_errors: list[Tensor] = []
    wrong_errors: list[Tensor] = []
    action_separations: list[Tensor] = []
    true_action_flow: list[float] = []
    shuffled_action_flow: list[float] = []

    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        for index, cpu_batch in enumerate(batches):
            batch = _to_device(cpu_batch, device)
            # Reset the stochastic flow evaluation per batch so before/after
            # comparisons see identical noise.
            torch.manual_seed(seed + index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + index)
            evaluator = WorldLossNormalizer().to(device)
            _, loss_metrics = world_loss(
                world,
                batch,
                normalizer=evaluator,
                global_step=0,
                bootstrap_rows=max(
                    0,
                    min(
                        batch.observations.shape[0] - 1,
                        round(0.25 * batch.observations.shape[0]),
                    ),
                ),
                bootstrap_start=10_000,
            )
            for name in (
                "loss/flow",
                "loss/reward",
                "loss/continuation",
                "flow/flow_mse",
            ):
                raw_losses.setdefault(name, []).append(
                    float(loss_metrics[name].item())
                )

            encoded = world.encode_frames(batch.observations, frozen=True)
            B, T = encoded.packed.shape[:2]
            paired_seed = seed + 5_000 + index
            torch.manual_seed(paired_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(paired_seed)
            true_flow, _ = shortcut_flow_loss(
                world.dynamics,
                clean=encoded.packed,
                led_to_actions=batch.led_to_actions,
                k_max=cfg.k_max,
                bootstrap_rows=0,
            )
            torch.manual_seed(paired_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(paired_seed)
            shuffled_flow, _ = shortcut_flow_loss(
                world.dynamics,
                clean=encoded.packed,
                led_to_actions=batch.led_to_actions.roll(1, dims=0),
                k_max=cfg.k_max,
                bootstrap_rows=0,
            )
            true_action_flow.append(float(true_flow.item()))
            shuffled_action_flow.append(float(shuffled_flow.item()))

            steps = torch.full(
                (B, T),
                cfg.max_step_index,
                device=device,
                dtype=torch.long,
            )
            signals = torch.full(
                (B, T), cfg.k_max, device=device, dtype=torch.long
            )
            _, agent = world.forward_dynamics(
                encoded.packed, batch.led_to_actions, steps, signals
            )
            heads = world.forward_task_heads(agent)
            reward = _decode_reward(
                heads["reward_logits"][:, :, 0],
                heads["reward_centers"],
            )
            continuation = heads["continue_logits"][:, :, 0].float().sigmoid()
            valid = batch.outcome_valid
            real_rewards.append(reward[valid].cpu())
            real_reward_targets.append(batch.led_to_rewards[valid].float().cpu())
            real_continues.append(continuation[valid].cpu())
            real_continue_targets.append(
                batch.led_to_continues[valid].float().cpu()
            )

            context = min(8, T - 1)
            past = encoded.packed[:, :context]
            context_actions = batch.led_to_actions[:, :context]
            actual_action = batch.led_to_actions[:, context]
            wrong_action = (actual_action + 1) % cfg.n_actions
            actual_led_to = torch.cat(
                [context_actions, actual_action[:, None]], dim=1
            )
            wrong_led_to = torch.cat(
                [context_actions, wrong_action[:, None]], dim=1
            )
            schedule = shortcut_schedule(cfg.k_max, cfg.k_max)
            actual_generator = torch.Generator(device=device).manual_seed(
                seed + 10_000 + index
            )
            wrong_generator = torch.Generator(device=device).manual_seed(
                seed + 10_000 + index
            )
            actual_latent, actual_agent = sample_next_packed(
                world,
                past_packed=past,
                led_to_actions=actual_led_to,
                schedule=schedule,
                use_cache=True,
                generator=actual_generator,
            )
            wrong_latent, _ = sample_next_packed(
                world,
                past_packed=past,
                led_to_actions=wrong_led_to,
                schedule=schedule,
                use_cache=True,
                generator=wrong_generator,
            )
            target = encoded.packed[:, context]
            actual_errors.append(
                (actual_latent.float() - target.float())
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
            wrong_errors.append(
                (wrong_latent.float() - target.float())
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
            action_separations.append(
                (actual_latent.float() - wrong_latent.float())
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
            generated_heads = world.forward_task_heads(actual_agent)
            generated_rewards.append(
                _decode_reward(
                    generated_heads["reward_logits"][:, 0, 0],
                    generated_heads["reward_centers"],
                ).cpu()
            )
            generated_reward_targets.append(
                batch.led_to_rewards[:, context].float().cpu()
            )
            generated_continues.append(
                generated_heads["continue_logits"][:, 0, 0]
                .float()
                .sigmoid()
                .cpu()
            )
            generated_continue_targets.append(
                batch.led_to_continues[:, context].float().cpu()
            )
    finally:
        torch.set_rng_state(saved_cpu_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)

    actual = torch.cat(actual_errors)
    wrong = torch.cat(wrong_errors)
    advantage = wrong - actual
    true_flow_mean = _mean(true_action_flow)
    shuffled_flow_mean = _mean(shuffled_action_flow)
    return {
        "raw_losses": {
            name: _mean(values) for name, values in raw_losses.items()
        },
        "real_state_reward": _reward_metrics(
            torch.cat(real_rewards), torch.cat(real_reward_targets)
        ),
        "real_state_continuation": _continuation_metrics(
            torch.cat(real_continues), torch.cat(real_continue_targets)
        ),
        "generated_k4_reward": _reward_metrics(
            torch.cat(generated_rewards), torch.cat(generated_reward_targets)
        ),
        "generated_k4_continuation": _continuation_metrics(
            torch.cat(generated_continues),
            torch.cat(generated_continue_targets),
        ),
        "one_step_action_conditioning": {
            "rows": int(actual.numel()),
            "actual_action_latent_mse": float(actual.mean().item()),
            "wrong_action_latent_mse": float(wrong.mean().item()),
            "wrong_minus_actual_mse": float(advantage.mean().item()),
            "actual_better_fraction": float((advantage > 0).float().mean().item()),
            "actual_wrong_generated_separation_mse": float(
                torch.cat(action_separations).mean().item()
            ),
        },
        "paired_action_shuffle": {
            "batches": len(true_action_flow),
            "true_action_flow_loss": true_flow_mean,
            "shuffled_action_flow_loss": shuffled_flow_mean,
            "shuffled_minus_true": (
                shuffled_flow_mean - true_flow_mean
                if true_flow_mean is not None and shuffled_flow_mean is not None
                else None
            ),
            "shuffled_over_true": (
                shuffled_flow_mean / max(true_flow_mean, 1e-12)
                if true_flow_mean is not None and shuffled_flow_mean is not None
                else None
            ),
        },
    }


def run_preflight(
    *,
    output_dir: Path,
    device: torch.device,
    tokenizer_steps: int,
    world_steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    if tokenizer_steps < 1 or world_steps < 1:
        raise ValueError("preflight requires positive tokenizer and world steps")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    cfg = D4LiteConfig(
        temporal_backend="transformer",
        representation_objective="base",
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    train_replay = load_episode_replay(
        TRAIN_REPLAY, expected_sha256=TRAIN_REPLAY_SHA256
    )
    dev_replay = load_episode_replay(
        DEV_REPLAY, expected_sha256=DEV_REPLAY_SHA256
    )
    load_seconds = time.perf_counter() - start
    fixed_eval = _fixed_eval_batches(
        dev_replay,
        cfg=cfg,
        count=eval_batches,
        batch_size=eval_batch_size,
        seed=seed + 2,
    )
    event_eval = _aligned_eval_batches(
        dev_replay,
        target="reward_event",
        context=8,
        batch_size=eval_batch_size,
    )
    terminal_eval = _aligned_eval_batches(
        dev_replay,
        target="terminal",
        context=8,
        batch_size=eval_batch_size,
    )

    tokenizer = build_tokenizer(cfg, training_mask=True).to(device).train()
    tokenizer_optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    tokenizer_before = evaluate_tokenizer(
        tokenizer, fixed_eval[: min(4, len(fixed_eval))], cfg=cfg, device=device
    )
    tokenizer_rng = np.random.default_rng(seed + 3)
    tokenizer_losses = []
    tokenizer_start = time.perf_counter()
    for step in range(tokenizer_steps):
        batch = _sample_sequence(
            train_replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            device=device,
            rng=tokenizer_rng,
        )
        tokenizer_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _ = tokenizer_reconstruction_loss(
                tokenizer, batch.observations, patch_size=cfg.patch_size
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite tokenizer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        tokenizer_optimizer.step()
        tokenizer_losses.append(float(loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tokenizer_seconds = time.perf_counter() - tokenizer_start
    tokenizer_after = evaluate_tokenizer(
        tokenizer, fixed_eval[: min(4, len(fixed_eval))], cfg=cfg, device=device
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer_t_base.pt"
    tokenizer_sha = save_tokenizer_checkpoint(
        tokenizer_path,
        tokenizer=tokenizer,
        config=cfg,
        step=tokenizer_steps,
        extra={
            "runner_format": RUNNER_FORMAT,
            "train_replay_sha256": TRAIN_REPLAY_SHA256,
            "dev_replay_sha256": DEV_REPLAY_SHA256,
            "seed": seed,
        },
    )

    world = D4LiteWorld(cfg).to(device)
    world.encoder.load_state_dict(tokenizer.encoder.state_dict(), strict=True)
    world.decoder.load_state_dict(tokenizer.decoder.state_dict(), strict=True)
    world.freeze_tokenizer()
    normalizer = WorldLossNormalizer().to(device)
    groups = optimizer_groups(world, learning_rate)
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    world_before = {
        "uniform": evaluate_world(
            world,
            fixed_eval,
            cfg=cfg,
            device=device,
            seed=seed + 4,
        ),
        "reward_event_aligned": evaluate_world(
            world,
            event_eval,
            cfg=cfg,
            device=device,
            seed=seed + 400,
        ),
        "terminal_aligned": evaluate_world(
            world,
            terminal_eval,
            cfg=cfg,
            device=device,
            seed=seed + 800,
        ),
    }
    world.train()
    world_rng = np.random.default_rng(seed + 5)
    world_history: list[dict[str, float]] = []
    bootstrap_rows = max(0, min(batch_size - 1, round(0.25 * batch_size)))
    world_start = time.perf_counter()
    for step in range(world_steps):
        batch = _sample_sequence(
            train_replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            device=device,
            rng=world_rng,
        )
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
                bootstrap_rows=bootstrap_rows,
                bootstrap_start=10_000,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite world loss at step {step}")
        loss.backward()
        trainable = [
            parameter
            for group in groups
            for parameter in group["params"]
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        if step < 1_000:
            warmup = float(step + 1) / 1_000.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * warmup
        optimizer.step()
        world_history.append(
            {
                "loss/flow": float(metrics["loss/flow"].item()),
                "loss/reward": float(metrics["loss/reward"].item()),
                "loss/continuation": float(
                    metrics["loss/continuation"].item()
                ),
                "loss/total": float(metrics["loss/total"].item()),
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    world_seconds = time.perf_counter() - world_start
    world_after = {
        "uniform": evaluate_world(
            world,
            fixed_eval,
            cfg=cfg,
            device=device,
            seed=seed + 4,
        ),
        "reward_event_aligned": evaluate_world(
            world,
            event_eval,
            cfg=cfg,
            device=device,
            seed=seed + 400,
        ),
        "terminal_aligned": evaluate_world(
            world,
            terminal_eval,
            cfg=cfg,
            device=device,
            seed=seed + 800,
        ),
    }

    world_path = output_dir / "world_t_base.pt"
    world_sha = save_checkpoint(
        world_path,
        world=world,
        normalizer=normalizer,
        optimizer=optimizer,
        numpy_rng=world_rng,
        step=world_steps,
        extra={
            "runner_format": RUNNER_FORMAT,
            "train_replay_sha256": TRAIN_REPLAY_SHA256,
            "dev_replay_sha256": DEV_REPLAY_SHA256,
            "tokenizer_checkpoint_sha256": tokenizer_sha,
            "seed": seed,
            "bootstrap_rows": bootstrap_rows,
            "bootstrap_start": 10_000,
            "warmup_steps": 1_000,
        },
    )

    window = min(20, len(world_history))

    def history_mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(sum(row[key] for row in rows) / len(rows))
            for key in rows[0]
        }

    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    report = {
        "format": RUNNER_FORMAT,
        "status": "completed",
        "arm": cfg.arm_id,
        "claim_boundary": (
            "bounded preflight on previously used replay; not a sealed final "
            "evaluation and not evidence of control"
        ),
        "config": asdict(cfg),
        "provenance": {
            "implementation_sha256": implementation_sha256(),
            "sources": source_report(),
            "train_replay": {
                "path": str(TRAIN_REPLAY),
                "sha256": TRAIN_REPLAY_SHA256,
                "episodes": len(train_replay.episodes),
                "transitions": train_replay.steps,
            },
            "dev_replay": {
                "path": str(DEV_REPLAY),
                "sha256": DEV_REPLAY_SHA256,
                "episodes": len(dev_replay.episodes),
                "transitions": dev_replay.steps,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "load_seconds": load_seconds,
            "tokenizer_seconds": tokenizer_seconds,
            "world_seconds": world_seconds,
            "peak_vram_bytes": peak,
        },
        "optimization": {
            "seed": seed,
            "batch_size": batch_size,
            "sequence_length": cfg.sequence_length,
            "learning_rate": learning_rate,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.999],
            "gradient_clip": 1.0,
            "tokenizer_steps": tokenizer_steps,
            "world_steps": world_steps,
            "world_warmup_steps": 1_000,
            "bootstrap_rows": bootstrap_rows,
            "bootstrap_start": 10_000,
            "eval_batches": eval_batches,
            "eval_batch_size": eval_batch_size,
            "reward_event_aligned_rows": sum(
                batch.observations.shape[0] for batch in event_eval
            ),
            "terminal_aligned_rows": sum(
                batch.observations.shape[0] for batch in terminal_eval
            ),
        },
        "tokenizer": {
            "before": tokenizer_before,
            "after": tokenizer_after,
            "first_20_training_loss": _mean(
                tokenizer_losses[: min(20, len(tokenizer_losses))]
            ),
            "last_20_training_loss": _mean(
                tokenizer_losses[-min(20, len(tokenizer_losses)) :]
            ),
            "checkpoint": str(tokenizer_path),
            "checkpoint_sha256": tokenizer_sha,
        },
        "world": {
            "before": world_before,
            "after": world_after,
            "first_training_window": history_mean(world_history[:window]),
            "last_training_window": history_mean(world_history[-window:]),
            "checkpoint": str(world_path),
            "checkpoint_sha256": world_sha,
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/d4_mamba_jepa/preflight_t_base",
    )
    parser.add_argument("--tokenizer-steps", type=int, default=500)
    parser.add_argument("--world-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    report = run_preflight(
        output_dir=args.output_dir,
        device=torch.device(args.device),
        tokenizer_steps=args.tokenizer_steps,
        world_steps=args.world_steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    summary = {
        "status": report["status"],
        "arm": report["arm"],
        "tokenizer": report["tokenizer"],
        "world": report["world"],
        "runtime": report["runtime"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
