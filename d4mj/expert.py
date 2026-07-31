from typing import Callable

import torch

from .config import Config
from .data import Episode
from .env import reset, step


def train_expert(config: Config):
    raise NotImplementedError("regeneration settings and acceptance threshold are open")


def collect(policy: Callable[[torch.Tensor, int], int], count: int, config: Config, limit: int = 2500):
    """Roll a policy and store episodes unshifted, with terminated and truncated
    kept apart.

    The archived replay stays available for smoke tests, but nothing reported comes
    from it: its expert has no byte-level provenance, and its terminal windows come
    from a 58-episode support that half of every batch would resample.
    """
    episodes = []
    for index in range(count):
        observation, env_state = reset(config.seed + index)
        frames, actions, rewards, terminals, timeouts = [observation], [], [], [], []

        for offset in range(limit):
            action = policy(observation, config.seed + index * limit + offset)
            observation, env_state, reward, terminated, truncated = step(
                env_state, action, config.seed + index * limit + offset
            )
            frames.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated)
            timeouts.append(truncated)
            if terminated or truncated:
                break

        episodes.append(
            Episode(
                observations=torch.stack(frames),
                actions_taken=torch.tensor(actions),
                rewards=torch.tensor(rewards, dtype=torch.float32),
                terminated=torch.tensor(terminals),
                truncated=torch.tensor(timeouts),
            )
        )
    return episodes
