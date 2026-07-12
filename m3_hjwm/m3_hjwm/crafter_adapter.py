from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CrafterStep:
    obs: np.ndarray
    reward: float
    terminated: bool
    info: dict


class CrafterAdapter:
    """Thin compatibility wrapper around danijar/crafter.

    Crafter observations are uint8 HWC images and the action space is Discrete(17).
    The wrapper returns CHW observations and a continuation flag suitable for the
    world-model transition convention.
    """
    def __init__(self, seed: int = 0):
        try:
            import crafter
        except ImportError as exc:
            raise RuntimeError("Install the official Crafter package/repository to use this adapter") from exc
        self.env = crafter.Env(seed=seed)

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.n)

    @staticmethod
    def _chw(obs: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(obs.transpose(2, 0, 1))

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._chw(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, float, dict]:
        result = self.env.step(int(action))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        return self._chw(obs), float(reward), float(not done), info
