from dataclasses import replace

import pytest
import torch

from d4mj.config import Config
from d4mj.data import Batch, Episode

CPU = replace(Config(), device="cpu")


@pytest.fixture
def config() -> Config:
    """Every test runs on CPU. The gates cover the deployment device; these lock
    semantics, which must not depend on where they run."""
    return CPU


def episode(index: int, config: Config, length: int | None = None) -> Episode:
    """Every array identifies its own index, so a one-step shift cannot be mistaken
    for a plausible value. One task event sits mid-episode."""
    steps = length or config.burn_in + config.sequence_long + 8 + index
    generator = torch.Generator().manual_seed(index)
    shape = (steps + 1, config.resolution, config.resolution, config.channels)
    return Episode(
        observations=torch.randint(0, 255, shape, generator=generator, dtype=torch.uint8),
        actions_taken=torch.arange(steps) % config.n_actions,
        rewards=torch.arange(steps).float(),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
        events=torch.arange(steps) == steps // 2,
    )


@pytest.fixture
def episodes(config: Config) -> list[Episode]:
    return [episode(index, config) for index in range(4)]


def latent_batch(config: Config, rows: int, blocks: int, relevant=None, seed: int = 0,
                 support=None) -> Batch:
    return Batch(
        led_to_action=torch.zeros(rows, blocks, dtype=torch.long),
        reward=torch.zeros(rows, blocks),
        terminated=torch.zeros(rows, blocks, dtype=torch.bool),
        truncated=torch.zeros(rows, blocks, dtype=torch.bool),
        valid=torch.ones(rows, blocks, dtype=torch.bool),
        scored=torch.ones(rows, blocks, dtype=torch.bool),
        relevant=None if relevant is None else torch.tensor(relevant),
        support=None if support is None else torch.tensor(support),
        burn_in=0,
        latents=torch.randn(
            rows,
            blocks,
            config.n_spatial,
            config.d_spatial,
            generator=torch.Generator().manual_seed(seed),
        ).tanh(),
    )


def window_start(batch: Batch, row: int) -> int:
    """The window's episode offset, read back out of the reward it carries."""
    return int(batch.reward[row][0]) + 1 if bool(batch.valid[row][0]) else 0
