"""Shared generated-state objectives for Stage-2 diagnostics.

This module exists to make loss routing executable and testable.  The original
Stage-2 runner summed latent, reward, and continuation losses inside one scalar,
which made it impossible to test or scale the components independently.

Transition convention:
    (obs_t, action_t) -> (obs_{t+1}, reward_t, continue_t)

After observing ``prefix`` real observations, generated step ``k=0`` consumes
``action[prefix - 1]`` and targets ``obs[prefix]``, ``reward[prefix - 1]``,
and ``continue[prefix - 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import cosine_distance


@dataclass(frozen=True)
class GeneratedLossWeights:
    latent: float = 1.0
    reward: float = 0.0
    continuation: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("latent", self.latent),
            ("reward", self.reward),
            ("continuation", self.continuation),
        ):
            if value < 0:
                raise ValueError(f"{name} generated-loss weight must be nonnegative")


def _validate_batch(batch: dict[str, torch.Tensor], prefix: int,
                    steps: int) -> None:
    if prefix < 1:
        raise ValueError("prefix must contain at least one observation")
    if steps < 1:
        raise ValueError("generated steps must be positive")
    observations = batch["obs"]
    if observations.ndim < 3:
        raise ValueError("obs must be [B,T,...]")
    needed_observations = prefix + steps
    if observations.shape[1] < needed_observations:
        raise ValueError(
            f"need at least {needed_observations} observations, "
            f"got {observations.shape[1]}"
        )
    needed_transitions = prefix + steps - 1
    for name in ("actions", "rewards", "continues"):
        value = batch[name]
        if value.shape[:2] != (observations.shape[0],
                               observations.shape[1] - 1):
            raise ValueError(
                f"{name} must be [B,T-1] and align with obs; "
                f"got {tuple(value.shape)}"
            )
        if value.shape[1] < needed_transitions:
            raise ValueError(f"{name} does not reach generated step {steps}")
    previous = batch["previous_actions"]
    if previous.shape[:2] != observations.shape[:2]:
        raise ValueError("previous_actions must be [B,T]")


def generated_step_components(
    world,
    batch: dict[str, torch.Tensor],
    *,
    prefix: int = 8,
    steps: int = 2,
) -> dict[str, torch.Tensor]:
    """Return independently routable generated latent/reward/continue losses.

    Each component is a fixed-denominator mean over ``batch * steps``. Rows
    after a terminal contribute zero; they are not renormalized away. This
    exactly matches the masking and reduction in the committed Stage-2 A/B
    runner, while exposing each term separately.
    """
    _validate_batch(batch, prefix, steps)
    device = batch["obs"].device
    batch_size = batch["obs"].shape[0]
    state = world.initial_state(batch_size, device)
    for time in range(prefix):
        state = world.observe_step(
            batch["obs"][:, time],
            batch["previous_actions"][:, time],
            state,
        )

    alive = torch.ones(batch_size, device=device)
    totals = {
        name: torch.zeros((), device=device)
        for name in ("latent", "reward", "continuation")
    }
    for generated_index in range(steps):
        transition = prefix - 1 + generated_index
        state, reward_logits, continue_logits, prediction = world.imagine_step(
            state,
            batch["actions"][:, transition],
            deterministic_mode=True,
        )
        with torch.no_grad():
            target = world.target_encoder(
                batch["obs"][:, prefix + generated_index]
            ).float()
        per_example = {
            "latent": cosine_distance(
                prediction.selected.float(), target
            ).mean(-1),
            "reward": world.reward.loss(
                reward_logits, batch["rewards"][:, transition]
            ),
            "continuation": F.binary_cross_entropy_with_logits(
                continue_logits,
                batch["continues"][:, transition],
                reduction="none",
            ),
        }
        for name, value in per_example.items():
            totals[name] = totals[name] + (alive * value).mean()
        alive = alive * (
            batch["continues"][:, transition] > 0.5
        ).to(alive.dtype)

    return {name: value / steps for name, value in totals.items()}


def weighted_generated_loss(
    components: dict[str, torch.Tensor],
    weights: GeneratedLossWeights,
) -> torch.Tensor:
    """Combine explicitly named components under a fixed registered recipe."""
    weights.validate()
    missing = {"latent", "reward", "continuation"} - set(components)
    if missing:
        raise ValueError(f"missing generated-loss components: {sorted(missing)}")
    return (
        weights.latent * components["latent"]
        + weights.reward * components["reward"]
        + weights.continuation * components["continuation"]
    )
