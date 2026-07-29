"""Deterministic small data used only for implementation discrimination."""
from __future__ import annotations

import torch

from d4_mamba_jepa.config import D4LiteConfig
from d4_mamba_jepa.data import SequenceBatch


def moving_square_batch(
    cfg: D4LiteConfig,
    *,
    batch_size: int,
    sequence_length: int | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> SequenceBatch:
    """Action-conditioned RGB sequence with known reward/terminal timing."""
    T = sequence_length or cfg.sequence_length
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions = torch.randint(
        0, 5, (batch_size, T - 1), generator=generator, dtype=torch.long
    )
    led_to = torch.full((batch_size, T), -1, dtype=torch.long)
    led_to[:, 1:] = actions

    side = cfg.image_size
    square = max(2, side // 8)
    limit = side - square
    positions = torch.randint(
        0, max(1, limit + 1), (batch_size, 2), generator=generator
    )
    observations = torch.zeros(
        batch_size, T, 3, side, side, dtype=torch.uint8
    )
    rewards = torch.zeros(batch_size, T)
    continues = torch.ones(batch_size, T)
    valid = torch.zeros(batch_size, T, dtype=torch.bool)
    valid[:, 1:] = True

    moves = torch.tensor(
        [
            [0, 0],   # no-op
            [0, 1],   # right
            [1, 0],   # down
            [0, -1],  # left
            [-1, 0],  # up
        ],
        dtype=torch.long,
    )
    for time in range(T):
        for row in range(batch_size):
            y, x = positions[row].tolist()
            observations[row, time, 0, y : y + square, x : x + square] = 255
            observations[row, time, 1] = int(255 * time / max(1, T - 1))
        if time == T - 1:
            continue
        positions = (positions + moves[actions[:, time]]).clamp(0, limit)
        # Dense, exactly action-conditioned event for the first overfit gate.
        # This avoids interpreting a no-event batch as reward-head success.
        rewards[:, time + 1] = (actions[:, time] == 1).float()
        if time == T - 2:
            continues[:, time + 1] = 0.0

    return SequenceBatch(
        observations=observations.to(device),
        led_to_actions=led_to.to(device),
        led_to_rewards=rewards.to(device),
        led_to_continues=continues.to(device),
        outcome_valid=valid.to(device),
    )
