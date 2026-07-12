"""Minimal raw-frame replay and Crafter adapter."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Episode:
    # obs length = actions length + 1
    obs: np.ndarray          # uint8 [T+1,C,H,W]
    actions: np.ndarray      # int64 [T]
    rewards: np.ndarray      # float32 [T]
    continues: np.ndarray    # float32 [T]


class EpisodeReplay:
    def __init__(self, capacity_steps: int = 200_000):
        self.capacity_steps = capacity_steps
        self.episodes: list[Episode] = []
        self.steps = 0

    def add(self, episode: Episode):
        if len(episode.obs) != len(episode.actions) + 1:
            raise ValueError("episode obs must have one more element than actions")
        self.episodes.append(episode)
        self.steps += len(episode.actions)
        while self.steps > self.capacity_steps:
            old = self.episodes.pop(0)
            self.steps -= len(old.actions)

    def sample(self, batch: int, observations: int, device):
        valid = [ep for ep in self.episodes if len(ep.obs) >= observations]
        if not valid:
            raise RuntimeError("no sufficiently long episode in replay")
        obs, actions, rewards, continues = [], [], [], []
        for _ in range(batch):
            ep = valid[np.random.randint(len(valid))]
            start = np.random.randint(0, len(ep.obs) - observations + 1)
            obs.append(ep.obs[start:start + observations])
            actions.append(ep.actions[start:start + observations - 1])
            rewards.append(ep.rewards[start:start + observations - 1])
            continues.append(ep.continues[start:start + observations - 1])
        return {
            "obs": torch.from_numpy(np.stack(obs)).to(device),
            "actions": torch.from_numpy(np.stack(actions)).to(device),
            "rewards": torch.from_numpy(np.stack(rewards)).to(device),
            "continues": torch.from_numpy(np.stack(continues)).to(device),
        }


class CrafterAdapter:
    """Thin adapter for danijar/crafter; returns CHW uint8 observations."""
    def __init__(self, seed: int = 0):
        try:
            import crafter
        except ImportError as exc:
            raise RuntimeError("install the official danijar/crafter package") from exc
        self.env = crafter.Env(seed=seed)

    @property
    def action_dim(self):
        return int(self.env.action_space.n)

    @staticmethod
    def chw(obs):
        return np.ascontiguousarray(obs.transpose(2, 0, 1))

    def reset(self):
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self.chw(obs)

    def step(self, action: int):
        result = self.env.step(int(action))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        return self.chw(obs), float(reward), float(not done), info
