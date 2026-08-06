from pathlib import Path
from typing import Callable

import torch

from .config import Config
from .data import Episode
from .env import reset, step

ARCHIVE = "d4_mamba_jepa_craftax_expert_replay_v1"


def load_archive(path: Path, config: Config, limit: int | None = None) -> list[Episode]:
    """The archived Craftax replay as `Episode`, converted exactly and lazily: the
    crop from 64x64 back to Craftax's native 63x63 and the permute are both views,
    so the 8.6 GB file is never resident (S49).

    `terminated` is `continues == 0`, which is death only -- the archive capped at
    2500 and Craftax's horizon is 10000, so no episode reached it -- and `truncated`
    marks the last transition of every episode that did not die. `events` comes from
    the per-frame cumulative achievements the archive already stores.
    """
    payload = torch.load(path, weights_only=False, mmap=True)
    episodes = []
    for record in payload[:limit]:
        steps = len(record["actions"])
        terminated = record["continues"] == 0
        truncated = torch.zeros(steps, dtype=torch.bool)
        if not bool(terminated.any()):
            truncated[-1] = True
        unlocked = record["achievements"].sum(-1)
        episodes.append(
            Episode(
                observations=record["obs"][:, :, : config.resolution, : config.resolution].permute(
                    0, 2, 3, 1
                ),
                actions_taken=record["actions"],
                rewards=record["rewards"],
                terminated=terminated,
                truncated=truncated,
                events=unlocked[1:] > unlocked[:-1],
            )
        )
    return episodes


def collect(
    policy: Callable[[torch.Tensor, int], int],
    count: int,
    config: Config,
    limit: int = 2500,
):
    """Roll a policy and store episodes unshifted, terminated and truncated apart.

    `events[t]` marks the steps at which the achievement count rises. Hitting the cap
    marks the last transition truncated, not terminal. Each episode owns a disjoint
    seed range. `bc_eligible` is the caller's declaration: degraded or exploratory
    rollouts belong in the world-model pool but not in behaviour cloning.
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
