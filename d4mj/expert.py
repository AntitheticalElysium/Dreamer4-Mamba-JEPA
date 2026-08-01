from pathlib import Path
from typing import Callable

import torch

from .config import Config
from .data import Episode
from .env import reset, step

ARCHIVE = "d4_mamba_jepa_craftax_expert_replay_v1"


def load_archive(path: Path, config: Config, limit: int | None = None) -> list[Episode]:
    """The archived Craftax replay, converted to `Episode` without re-encoding.

    It replaces `train_expert`: the archive already holds an expert whose weights
    are hashed in its manifest, so there is nothing to retrain. What is missing from
    it is the *uniform* half, which `collect` produces from any unfiltered policy
    and needs no expert at all.

    Conversion is exact and lazy. Frames are stored CHW at 64x64, zero-padded from
    Craftax's native 63x63; the crop and the permute are both views, so an episode
    costs nothing until a window indexes it and the 8.6 GB file is never resident.

    Two fields are reconstructed rather than read, because the archive does not
    separate them. `terminated` is `continues == 0`, which is death only: Craftax's
    native horizon is 10000 and the archive capped at 2500, so no episode there ever
    reached it. `truncated` therefore marks the last transition of every episode
    that did not die -- 252 of 320 -- and that cap is ours, not the environment's.

    `events` comes from the per-frame cumulative achievements the archive already
    stores, which is what makes its windows usable by the relevant sampler.
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
