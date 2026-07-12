from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray


class EpisodeReplay:
    """Simple episode replay for the initial Crafter implementation.

    Stores raw uint8 frames so encoder targets never become stale.
    """
    def __init__(self, capacity_steps: int = 200_000):
        self.capacity_steps = capacity_steps
        self.episodes: list[Episode] = []
        self.steps = 0

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.steps += len(episode.actions)
        while self.steps > self.capacity_steps and self.episodes:
            old = self.episodes.pop(0)
            self.steps -= len(old.actions)

    def sample(self, batch_size: int, length: int, device: torch.device) -> dict[str, Tensor]:
        valid = [e for e in self.episodes if len(e.obs) >= length]
        if not valid:
            raise RuntimeError("replay does not contain an episode long enough")
        obs, actions, rewards, continues = [], [], [], []
        for _ in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            start = np.random.randint(0, len(ep.obs) - length + 1)
            obs.append(ep.obs[start:start + length])
            actions.append(ep.actions[start:start + length - 1])
            rewards.append(ep.rewards[start:start + length - 1])
            continues.append(ep.continues[start:start + length - 1])
        return {
            "obs": torch.from_numpy(np.stack(obs)).to(device),
            "actions": torch.from_numpy(np.stack(actions)).to(device),
            "rewards": torch.from_numpy(np.stack(rewards)).to(device),
            "continues": torch.from_numpy(np.stack(continues)).to(device),
        }
