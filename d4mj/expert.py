from typing import Callable

import torch

from .config import Config
from .data import Episode
from .env import reset, step


def train_expert(config: Config):
    raise NotImplementedError("regeneration settings and acceptance threshold are open")


def collect(
    policy: Callable[[torch.Tensor, int], int],
    count: int,
    config: Config,
    limit: int = 2500,
):
    """Roll a policy and store episodes unshifted, with terminated and truncated
    kept apart.

    `events[t]` marks the steps at which the achievement count rises, which is what
    "accomplish one of the tasks" means on Craftax. It is recorded per step, so the
    relevant sampler draws windows around a task event rather than anywhere inside a
    successful episode.

    Hitting the cap marks the last transition truncated: storing it as neither would
    tell the continuation head that the trajectory continues past data that does not
    exist. Each episode owns a disjoint seed range.
    """
    episodes = []
    for index in range(count):
        base = config.seed + index * (limit + 1)
        observation, env_state = reset(base)
        frames, actions, rewards, terminals, timeouts, events = [observation], [], [], [], [], []
        achieved = int(env_state.achievements.sum())

        for offset in range(limit):
            action = policy(observation, base + offset + 1)
            observation, env_state, reward, terminated, truncated = step(
                env_state, action, base + offset + 1
            )
            unlocked = int(env_state.achievements.sum())
            frames.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated)
            timeouts.append(truncated)
            events.append(unlocked > achieved)
            achieved = unlocked
            if terminated or truncated:
                break
            if offset + 1 == limit:
                timeouts[-1] = True

        episodes.append(
            Episode(
                observations=torch.stack(frames),
                actions_taken=torch.tensor(actions),
                rewards=torch.tensor(rewards, dtype=torch.float32),
                terminated=torch.tensor(terminals),
                truncated=torch.tensor(timeouts),
                events=torch.tensor(events),
            )
        )
    return episodes
