"""Crafter timing adapters over the already-audited episode replay."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from .source import COMPACT_DATA, CRAFTER_CANONICAL, verify_source

verify_source(COMPACT_DATA)
verify_source(CRAFTER_CANONICAL)
from m3_hjwm_compact.data import CrafterAdapter, Episode, EpisodeReplay


@dataclass(frozen=True)
class SequenceBatch:
    """Episode-bounded sequence in the block-causal led-to convention."""

    observations: torch.Tensor  # uint8 [B,T,C,H,W]
    led_to_actions: torch.Tensor  # int64 [B,T], -1 is the start action
    led_to_rewards: torch.Tensor  # float [B,T]
    led_to_continues: torch.Tensor  # float [B,T]
    outcome_valid: torch.Tensor  # bool [B,T]


def replay_sample_to_sequence(sample: dict[str, torch.Tensor]) -> SequenceBatch:
    observations = sample["obs"]
    actions = sample["actions"]
    rewards = sample["rewards"]
    continues = sample["continues"]
    previous = sample["previous_actions"]
    if observations.ndim != 5:
        raise ValueError("obs must have shape [B,T,C,H,W]")
    B, T = observations.shape[:2]
    expected = (B, T - 1)
    for name, tensor in (
        ("actions", actions),
        ("rewards", rewards),
        ("continues", continues),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected}")
    if previous.shape != (B, T):
        raise ValueError(
            f"previous_actions shape {tuple(previous.shape)} != {(B, T)}"
        )

    led_rewards = torch.zeros((B, T), device=rewards.device, dtype=rewards.dtype)
    led_continues = torch.zeros(
        (B, T), device=continues.device, dtype=continues.dtype
    )
    valid = torch.zeros((B, T), device=observations.device, dtype=torch.bool)
    led_rewards[:, 1:] = rewards
    led_continues[:, 1:] = continues
    valid[:, 1:] = True
    return SequenceBatch(
        observations=observations,
        led_to_actions=previous.to(torch.long),
        led_to_rewards=led_rewards,
        led_to_continues=led_continues,
        outcome_valid=valid,
    )


def transitions_to_led_to(actions: torch.Tensor, start_action: int = -1) -> torch.Tensor:
    """Map actions ``a_t`` to the state slot they produce.

    Input shape is ``[B,T-1]`` for a sequence containing ``T`` observations.
    Output slot zero is the start action and output slot ``t+1`` is ``a_t``.
    """
    if actions.ndim != 2:
        raise ValueError("actions must have shape [B,T-1]")
    out = torch.full(
        (actions.shape[0], actions.shape[1] + 1),
        int(start_action),
        device=actions.device,
        dtype=torch.long,
    )
    out[:, 1:] = actions.to(torch.long)
    return out


def load_episode_replay(
    path: str | Path,
    *,
    expected_sha256: str,
    capacity_steps: int = 500_000,
) -> EpisodeReplay:
    """Load and fully validate a hash-pinned list of episode dictionaries."""
    source = Path(path)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"replay digest drift for {source}: {actual} != {expected_sha256}"
        )
    records = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise RuntimeError("replay payload must be a list of episodes")
    replay = EpisodeReplay(capacity_steps=capacity_steps)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"episode {index} is not a dictionary")
        missing = {"obs", "actions", "rewards", "continues"} - set(record)
        if missing:
            raise RuntimeError(f"episode {index} is missing {sorted(missing)}")
        replay.add(
            Episode(
                obs=record["obs"],
                actions=record["actions"],
                rewards=record["rewards"],
                continues=record["continues"],
            )
        )
    return replay


__all__ = [
    "CrafterAdapter",
    "Episode",
    "EpisodeReplay",
    "SequenceBatch",
    "replay_sample_to_sequence",
    "transitions_to_led_to",
    "load_episode_replay",
]
