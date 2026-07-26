"""Frozen-world / tokenizer evaluation helpers.

Formerly ``crafter_preflight.py``. The legacy danijar/m3 preflight CLI
(``run_preflight``/``main`` on pinned legacy replay paths) was removed in the
Craftax migration -- the Craftax-native end-to-end gate is
``craftax_runners.craftax_preflight``. What remains here is the source-agnostic
evaluation machinery (``evaluate_tokenizer``, ``evaluate_world`` and their
metric helpers), still used by ``cartpole_baseline`` and ``backend_pair``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .config import D4LiteConfig
from .data import SequenceBatch, replay_sample_to_sequence
from .model import D4LiteWorld
from .objectives import shortcut_flow_loss
from .rollout import sample_next_packed, shortcut_schedule
from .source import load_mmbench2_model
from .training import (
    WorldLossNormalizer,
    tokenizer_full_reconstruction_mse,
    world_loss,
)


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
