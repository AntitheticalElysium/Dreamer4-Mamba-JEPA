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
    relevant: bool = False,
):
    """Roll a policy and store episodes unshifted, with terminated and truncated
    kept apart.

    `relevant` is declared by the caller, not inferred here: Dreamer 4's mixture is
    50% uniform sequences and 50% that "accomplish one of the tasks", and which pool
    a rollout belongs to is a property of how it was selected, not of anything
    visible in the trajectory. It defaults to uniform because unfiltered rollouts
    are what this function produces; the relevant pool is the filtered one, and a
    caller that does not filter must not be able to claim it by omission.

    The archived replay stays available for smoke tests, but nothing reported comes
    from it: its expert has no byte-level provenance, its terminal windows come from
    a 68-episode support that half of every batch would resample, and it is 100%
    relevant, which leaves the dynamics loss with no uniform half to score.

    Hitting the collector's own cap marks the last transition truncated. Storing it
    as neither terminated nor truncated would tell the continuation head that a
    trajectory continues past data that does not exist.

    Each episode owns a disjoint seed range, so one episode's reset key is never
    another's step key -- harmless in practice, but the pattern that produced
    correlated streams before.
    """
    episodes = []
    for index in range(count):
        base = config.seed + index * (limit + 1)
        observation, env_state = reset(base)
        frames, actions, rewards, terminals, timeouts = [observation], [], [], [], []

        for offset in range(limit):
            action = policy(observation, base + offset + 1)
            observation, env_state, reward, terminated, truncated = step(
                env_state, action, base + offset + 1
            )
            frames.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated)
            timeouts.append(truncated)
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
                relevant=relevant,
            )
        )
    return episodes
