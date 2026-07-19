"""Shared training-only machinery for the Stage-2G relevance factorial.

Protocol:
    reviews/2026-07-19-stage2g-shared-reward-relevance-protocol.md

Transition convention:
    (obs_t, action_t) -> (obs_{t+1}, reward_t, continue_t)

The auxiliary consumes only generated K1/K2 planner-state inputs. It never
calls the planner reward or continuation heads.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model import assert_encoder_frozen, frozen_dynamics_recipe
from stage1b_equal_update_control import state_digest
from stage2_ab import BATCH, PREFIX, make_batch
from stage2_objectives import (
    GeneratedLossWeights,
    generated_step_components,
    weighted_generated_loss,
)


PROTOCOL = (
    "reviews/2026-07-19-stage2g-shared-reward-relevance-protocol.md"
)
WINDOW = PREFIX + 2
STEPS = 2
FULL_UPDATES = 16_000
SMOKE_UPDATES = 256
AUXILIARY_SEED = 70_505
PROBE_SEED = 71_505
SCHEDULE_SEED = 72_505
GRADIENT_BATCHES = 16
PROBE_COUNTS = {
    "zero": 32,
    "positive": 16,
    "negative": 16,
}
ARM_REWARD_WEIGHTS = {
    "G-LA": 0.0,
    "G-LRA": 0.10,
}
BASE_REFERENCE = {
    "G-LA": "C-L",
    "G-LRA": "C-LR",
}


@dataclass(frozen=True)
class RelevancePools:
    zero: tuple[tuple[int, int], ...]
    positive: tuple[tuple[int, int], ...]
    negative: tuple[tuple[int, int], ...]
    mixed: tuple[tuple[int, int], ...]
    terminal: tuple[tuple[int, int], ...]


class RewardRelevanceHeads(nn.Module):
    """Training-only linear probes whose input is the planner reward state."""

    def __init__(self, dim: int):
        super().__init__()
        self.event = nn.Linear(dim, 1)
        self.sign = nn.Linear(dim, 1)

    def forward(
        self,
        planner_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.event(planner_state).squeeze(-1),
            self.sign(planner_state).squeeze(-1),
        )


def build_relevance_heads(
    dim: int,
    device: torch.device,
) -> RewardRelevanceHeads:
    """Initialize without consuming the base world's global RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(AUXILIARY_SEED)
        heads = RewardRelevanceHeads(dim)
    return heads.to(device)


def module_state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1)
            .view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _generated_labels(
    episode: dict,
    start: int,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(
        episode["rewards"][
            start + PREFIX - 1:start + PREFIX - 1 + STEPS
        ],
        dtype=np.float32,
    )
    continues = np.asarray(
        episode["continues"][
            start + PREFIX - 1:start + PREFIX - 1 + STEPS
        ],
        dtype=np.float32,
    )
    if rewards.shape != (STEPS,) or continues.shape != (STEPS,):
        raise RuntimeError("relevance window does not reach K1/K2")
    return rewards, continues


def relevance_pools(train: list[dict]) -> RelevancePools:
    pools = {
        name: []
        for name in ("zero", "positive", "negative", "mixed", "terminal")
    }
    for episode_index, episode in enumerate(train):
        for start in range(len(episode["obs"]) - WINDOW + 1):
            rewards, continues = _generated_labels(episode, start)
            pair = (episode_index, start)
            if bool(np.any(continues < 0.5)):
                pools["terminal"].append(pair)
                continue
            positive = bool(np.any(rewards > 1e-6))
            negative = bool(np.any(rewards < -1e-6))
            if positive and negative:
                pools["mixed"].append(pair)
            elif positive:
                pools["positive"].append(pair)
            elif negative:
                pools["negative"].append(pair)
            else:
                pools["zero"].append(pair)
    return RelevancePools(**{
        name: tuple(values) for name, values in pools.items()
    })


def _shuffled(
    values: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    order = rng.permutation(len(values))
    return [values[int(index)] for index in order]


def build_auxiliary_contract(
    pools: RelevancePools,
    *,
    updates: int = FULL_UPDATES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    """Build a disjoint probe and balanced with-replacement train schedule."""
    probe_rng = np.random.default_rng(PROBE_SEED)
    shuffled = {
        name: _shuffled(getattr(pools, name), probe_rng)
        for name in PROBE_COUNTS
    }
    for name, required in PROBE_COUNTS.items():
        if len(shuffled[name]) <= required:
            raise RuntimeError(
                f"not enough {name} windows for probe and training"
            )
    probe = []
    remaining = {}
    for name, required in PROBE_COUNTS.items():
        probe.extend(shuffled[name][:required])
        remaining[name] = shuffled[name][required:]

    schedule_rng = np.random.default_rng(SCHEDULE_SEED)
    schedule = []
    for _ in range(updates):
        row = [
            remaining["zero"][int(schedule_rng.integers(
                len(remaining["zero"])
            ))],
            remaining["zero"][int(schedule_rng.integers(
                len(remaining["zero"])
            ))],
            remaining["positive"][int(schedule_rng.integers(
                len(remaining["positive"])
            ))],
            remaining["negative"][int(schedule_rng.integers(
                len(remaining["negative"])
            ))],
        ]
        order = schedule_rng.permutation(BATCH)
        schedule.extend(row[int(index)] for index in order)

    probe_set = set(probe)
    if probe_set.intersection(schedule):
        raise RuntimeError("auxiliary probe overlaps auxiliary schedule")
    schedule_array = np.asarray(schedule, dtype=np.int64)
    probe_array = np.asarray(probe, dtype=np.int64)
    info = {
        "pool_counts": {
            name: len(getattr(pools, name))
            for name in (
                "zero", "positive", "negative", "mixed", "terminal"
            )
        },
        "probe_counts": dict(PROBE_COUNTS),
        "probe_sha256": hashlib.sha256(
            probe_array.tobytes()
        ).hexdigest(),
        "schedule_updates": updates,
        "schedule_sha256": hashlib.sha256(
            schedule_array.tobytes()
        ).hexdigest(),
        "probe_schedule_overlap": 0,
    }
    return schedule, probe, info


def schedule_label_audit(
    train: list[dict],
    picks: list[tuple[int, int]],
) -> dict:
    rewards = []
    continues = []
    event_windows = 0
    positive_windows = 0
    negative_windows = 0
    for episode, start in picks:
        row_reward, row_continue = _generated_labels(
            train[episode], start
        )
        rewards.append(row_reward)
        continues.append(row_continue)
        event_windows += int(bool(np.any(np.abs(row_reward) > 1e-6)))
        positive_windows += int(bool(np.any(row_reward > 1e-6)))
        negative_windows += int(bool(np.any(row_reward < -1e-6)))
    reward = np.stack(rewards)
    continuation = np.stack(continues)
    events = np.abs(reward) > 1e-6
    event_rewards = reward[events]
    return {
        "windows": len(picks),
        "rows": int(reward.size),
        "event_window_fraction": event_windows / len(picks),
        "positive_window_fraction": positive_windows / len(picks),
        "negative_window_fraction": negative_windows / len(picks),
        "event_row_fraction": float(events.mean()),
        "positive_event_rows": int(np.sum(event_rewards > 0)),
        "negative_event_rows": int(np.sum(event_rewards < 0)),
        "terminal_row_fraction": float(np.mean(continuation < 0.5)),
    }


def generated_planner_states(
    world,
    batch: dict[str, torch.Tensor],
    *,
    detach_world: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return generated K1/K2 planner states and aligned rewards."""
    state = world.initial_state(batch["obs"].shape[0], batch["obs"].device)
    for time_index in range(PREFIX):
        state = world.observe_step(
            batch["obs"][:, time_index],
            batch["previous_actions"][:, time_index],
            state,
        )
    planner_states = []
    labels = []
    for generated_index in range(STEPS):
        transition = PREFIX - 1 + generated_index
        state, _, _, _ = world.imagine_step(
            state,
            batch["actions"][:, transition],
            deterministic_mode=True,
        )
        value = world.pool(state.tokens)
        planner_states.append(value.detach() if detach_world else value)
        labels.append(batch["rewards"][:, transition])
    return torch.stack(planner_states, 1), torch.stack(labels, 1)


def relevance_loss(
    world,
    heads: RewardRelevanceHeads,
    batch: dict[str, torch.Tensor],
    *,
    detach_world: bool = False,
) -> dict[str, torch.Tensor]:
    planner_state, reward = generated_planner_states(
        world, batch, detach_world=detach_world
    )
    event_logits, sign_logits = heads(planner_state)
    event = reward.abs() > 1e-6
    event_loss = F.binary_cross_entropy_with_logits(
        event_logits, event.to(event_logits.dtype)
    )
    if not bool(event.any()):
        raise RuntimeError("relevance batch contains no reward event")
    sign_loss = F.binary_cross_entropy_with_logits(
        sign_logits[event],
        (reward[event] > 0).to(sign_logits.dtype),
    )
    return {
        "loss": event_loss + sign_loss,
        "event": event_loss,
        "sign": sign_loss,
        "event_logits": event_logits,
        "sign_logits": sign_logits,
        "reward": reward,
        "planner_state": planner_state,
    }


def _module_gradient_l2(modules: tuple[nn.Module, ...]) -> float:
    device = next(modules[0].parameters()).device
    squared = torch.zeros((), device=device)
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.float().pow(2).sum()
    return float(squared.sqrt())


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def shared_modules(world) -> tuple[nn.Module, ...]:
    return world.action_input, world.future, world.temporal


def component_gradient_norms(world, heads) -> dict:
    return {
        "shared": _module_gradient_l2(shared_modules(world)),
        "reward_head": _module_gradient_l2((world.reward,)),
        "continuation_head": _module_gradient_l2((world.continuation,)),
        "online_encoder": _module_gradient_l2((world.online_encoder,)),
        "target_encoder": _module_gradient_l2((world.target_encoder,)),
        "auxiliary_heads": _module_gradient_l2((heads,)),
    }


@torch.no_grad()
def probe_relevance(
    world,
    heads: RewardRelevanceHeads,
    train: list[dict],
    picks: list[tuple[int, int]],
) -> dict:
    device = next(world.parameters()).device
    event_scores = []
    sign_scores = []
    rewards = []
    planner_rewards = []
    event_losses = []
    sign_losses = []
    for start in range(0, len(picks), BATCH):
        batch_picks = picks[start:start + BATCH]
        if len(batch_picks) != BATCH:
            raise RuntimeError("probe size must be divisible by batch")
        batch = make_batch(
            train, batch_picks, device, window=WINDOW
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            planner_state, reward = generated_planner_states(world, batch)
            event_logits, sign_logits = heads(planner_state)
            event = reward.abs() > 1e-6
            event_loss = F.binary_cross_entropy_with_logits(
                event_logits, event.to(event_logits.dtype)
            )
            sign_loss = F.binary_cross_entropy_with_logits(
                sign_logits[event],
                (reward[event] > 0).to(sign_logits.dtype),
            )
            reward_logits = world.reward(planner_state)
            decoded = world.reward.decode(reward_logits)
        event_scores.append(event_logits.float().cpu().reshape(-1))
        sign_scores.append(sign_logits.float().cpu()[event])
        rewards.append(reward.float().cpu().reshape(-1))
        planner_rewards.append(decoded.float().cpu().reshape(-1))
        event_losses.append(float(event_loss))
        sign_losses.append(float(sign_loss))
    event_scores_np = torch.cat(event_scores).numpy()
    sign_scores_np = torch.cat(sign_scores).numpy()
    rewards_np = torch.cat(rewards).numpy()
    decoded_np = torch.cat(planner_rewards).numpy()
    event_labels = np.abs(rewards_np) > 1e-6
    sign_labels = rewards_np[event_labels] > 0
    return {
        "loss": float(np.mean(event_losses) + np.mean(sign_losses)),
        "event_loss": float(np.mean(event_losses)),
        "sign_loss": float(np.mean(sign_losses)),
        "event_auroc": binary_auroc(event_scores_np, event_labels),
        "sign_auroc": binary_auroc(sign_scores_np, sign_labels),
        "decoded_absolute_maximum": float(np.abs(decoded_np).max()),
        "decoded_absolute_mean": float(np.abs(decoded_np).mean()),
        "event_rows": int(event_labels.sum()),
        "zero_rows": int((~event_labels).sum()),
    }


def train_relevance_world(
    world,
    heads: RewardRelevanceHeads,
    train: list[dict],
    base_schedule: list[tuple[int, int]],
    auxiliary_schedule: list[tuple[int, int]],
    probe: list[tuple[int, int]],
    *,
    arm: str,
    lambda_aux: float,
    updates: int,
    probe_updates: tuple[int, ...] = (),
) -> tuple[object, RewardRelevanceHeads, dict]:
    if arm not in ARM_REWARD_WEIGHTS:
        raise ValueError(f"unknown Stage-2G arm {arm!r}")
    if len(base_schedule) < updates * BATCH:
        raise ValueError("base schedule is shorter than requested training")
    if len(auxiliary_schedule) < updates * BATCH:
        raise ValueError("auxiliary schedule is shorter than requested")
    if not 0.01 <= lambda_aux <= 10.0:
        raise ValueError("registered auxiliary coefficient out of range")

    device = next(world.parameters()).device
    base_weights = frozen_dynamics_recipe()
    generated_weights = GeneratedLossWeights(
        latent=1.0,
        reward=ARM_REWARD_WEIGHTS[arm],
        continuation=0.0,
    )
    world_parameters = [
        parameter for parameter in world.parameters()
        if parameter.requires_grad
    ]
    auxiliary_parameters = list(heads.parameters())
    world_optimizer = torch.optim.AdamW(world_parameters, lr=1e-4)
    auxiliary_optimizer = torch.optim.AdamW(
        auxiliary_parameters, lr=1e-4
    )
    histories = {
        name: []
        for name in (
            "total",
            "base_jepa",
            "base_reward",
            "base_continuation",
            "base_rollout",
            "generated_latent",
            "generated_reward",
            "generated_weighted",
            "auxiliary",
            "auxiliary_event",
            "auxiliary_sign",
            "world_gradient_norm",
            "auxiliary_gradient_norm",
        )
    }
    probes = {}
    if 0 in probe_updates:
        probes["u0"] = probe_relevance(
            world, heads, train, probe
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update in range(updates):
        base_picks = base_schedule[
            update * BATCH:(update + 1) * BATCH
        ]
        auxiliary_picks = auxiliary_schedule[
            update * BATCH:(update + 1) * BATCH
        ]
        base_batch = make_batch(train, base_picks, device)
        auxiliary_batch = make_batch(
            train, auxiliary_picks, device, window=WINDOW
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = world(base_batch, base_weights)
            components = generated_step_components(
                world,
                base_batch,
                prefix=PREFIX,
                steps=STEPS,
            )
            generated = weighted_generated_loss(
                components, generated_weights
            )
            auxiliary = relevance_loss(
                world, heads, auxiliary_batch
            )
            loss = (
                base.loss
                + generated
                + lambda_aux * auxiliary["loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"{arm} non-finite loss at update {update + 1}"
            )
        world_optimizer.zero_grad(set_to_none=True)
        auxiliary_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        world_gradient = torch.nn.utils.clip_grad_norm_(
            world_parameters, 100.0
        )
        auxiliary_gradient = torch.nn.utils.clip_grad_norm_(
            auxiliary_parameters, 100.0
        )
        if not bool(torch.isfinite(world_gradient)):
            raise RuntimeError(
                f"{arm} non-finite world gradient at update {update + 1}"
            )
        if not bool(torch.isfinite(auxiliary_gradient)):
            raise RuntimeError(
                f"{arm} non-finite auxiliary gradient at "
                f"update {update + 1}"
            )
        world_optimizer.step()
        auxiliary_optimizer.step()
        world.mark_parameters_updated()

        row = {
            "total": float(loss.detach()),
            "base_jepa": float(base.metrics["jepa"]),
            "base_reward": float(base.metrics["reward"]),
            "base_continuation": float(base.metrics["continuation"]),
            "base_rollout": float(base.metrics["rollout"]),
            "generated_latent": float(components["latent"].detach()),
            "generated_reward": float(components["reward"].detach()),
            "generated_weighted": float(generated.detach()),
            "auxiliary": float(auxiliary["loss"].detach()),
            "auxiliary_event": float(auxiliary["event"].detach()),
            "auxiliary_sign": float(auxiliary["sign"].detach()),
            "world_gradient_norm": float(world_gradient),
            "auxiliary_gradient_norm": float(auxiliary_gradient),
        }
        for name, value in row.items():
            histories[name].append(value)

        completed = update + 1
        if completed in probe_updates:
            probes[f"u{completed}"] = probe_relevance(
                world, heads, train, probe
            )
        if updates <= SMOKE_UPDATES or completed % 1000 == 0:
            for name, parameter in (
                list(world.named_parameters())
                + [
                    (f"auxiliary.{name}", parameter)
                    for name, parameter in heads.named_parameters()
                ]
            ):
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(
                        f"{arm} non-finite parameter {name} at "
                        f"update {completed}"
                    )
        if completed % 4000 == 0:
            print(
                f"[{arm}] {completed}: "
                f"jepa={np.mean(histories['base_jepa'][-500:]):.5f} "
                f"gen-reward="
                f"{np.mean(histories['generated_reward'][-500:]):.5f} "
                f"aux={np.mean(histories['auxiliary'][-500:]):.5f}",
                flush=True,
            )

    assert_encoder_frozen(world, world_optimizer)
    info = {
        "updates": updates,
        "train_seconds": time.perf_counter() - started,
        "world_final_digest": state_digest(
            world, exclude_heads=False
        ),
        "auxiliary_final_digest": module_state_digest(heads),
        "world_trainable_names": [
            name for name, parameter in world.named_parameters()
            if parameter.requires_grad
        ],
        "auxiliary_trainable_names": [
            name for name, _ in heads.named_parameters()
        ],
        "histories": histories,
        "probes": probes,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20
        ),
        "last500": {
            name: float(np.mean(values[-500:]))
            for name, values in histories.items()
        },
    }
    return world, heads, info


def train_reference_world(
    world,
    train: list[dict],
    base_schedule: list[tuple[int, int]],
    *,
    reward_weight: float,
    updates: int,
) -> tuple[object, dict]:
    """Exact no-aux Stage-2C update for a bounded preflight reference."""
    if reward_weight not in (0.0, 0.10):
        raise ValueError("reference reward weight must be 0 or .10")
    if len(base_schedule) < updates * BATCH:
        raise ValueError("base schedule is shorter than requested training")
    device = next(world.parameters()).device
    base_weights = frozen_dynamics_recipe()
    generated_weights = GeneratedLossWeights(
        latent=1.0,
        reward=reward_weight,
        continuation=0.0,
    )
    trainable = [
        parameter for parameter in world.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    history = []
    for update in range(updates):
        picks = base_schedule[update * BATCH:(update + 1) * BATCH]
        batch = make_batch(train, picks, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = world(batch, base_weights)
            components = generated_step_components(
                world, batch, prefix=PREFIX, steps=STEPS
            )
            generated = weighted_generated_loss(
                components, generated_weights
            )
            loss = base.loss + generated
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(
                f"non-finite reference gradient at update {update + 1}"
            )
        optimizer.step()
        world.mark_parameters_updated()
        history.append(float(loss.detach()))
    assert_encoder_frozen(world, optimizer)
    return world, {
        "updates": updates,
        "reward_weight": reward_weight,
        "final_digest": state_digest(world, exclude_heads=False),
        "total_history_sha256": hashlib.sha256(
            np.asarray(history, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
