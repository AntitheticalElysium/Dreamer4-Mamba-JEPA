"""Native Craftax-Classic pixel environment adapter.

Replaces the old danijar/crafter-via-``m3_hjwm_compact`` adapter. This module
imports JAX + Craftax and is therefore imported ONLY by data-generation and
executed-evaluation scripts, never by the training modules, so ``import
d4_mamba_jepa.data`` (and the torch training stack) never pulls in JAX.

Verified facts this adapter is built on (Craftax-Classic, commit-pinned via
``source.verify_installed_craftax``):

- ``Craftax-Classic-Pixels-v1`` native observation is ``(63, 63, 3)`` float32 in
  ``[0, 1]`` (a 9x9 tile grid at ``BLOCK_PIXEL_SIZE_AGENT = 7``). 63x63 is the
  agent's true information ceiling; the dataset/human renders at 16px (144x144)
  carry no additional game information, so we render at 7px and pad to 64x64
  rather than downsample 144 -> 64 (which would add interpolation blur).
- 17 actions in Crafter order; 22 achievements; ``max_timesteps = 10000``.
- reward = +1 per newly-unlocked achievement + 0.1 * health_delta.
- termination = ``timestep >= max_timesteps`` (timeout) OR lava OR death.

``continues`` follows Crafter's ``discount = 1 - dead`` convention: a timed-out
episode is a truncation with ``continues[-1] == 1`` (bootstrap), while death /
lava give ``continues[-1] == 0`` (absorbing).
"""
from __future__ import annotations

import os

# Keep the GPU for torch; Craftax runs on CPU. A caller wanting GPU JAX must set
# JAX_PLATFORMS before importing this module.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import Episode

NATIVE_OBS_HW = 63
DEFAULT_TARGET_SIZE = 64
N_ACTIONS = 17
N_ACHIEVEMENTS = 22
CRAFTAX_ENV_NAME = "Craftax-Classic-Pixels-v1"

# Privileged simulator-state labels for the representation oracle. These are
# read ONLY by the oracle's probe-data collector and are never available to any
# training loop (probe payloads are marked and rejected by the replay loader).
VITAL_FIELDS = ("player_health", "player_food", "player_drink", "player_energy")
INVENTORY_FIELDS = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)


def privileged_labels(state) -> dict:
    """Extract ground-truth simulator labels from a Craftax ``EnvState``.

    Returns numpy arrays: ``vitals`` [4], ``inventory`` [12] (counts),
    ``achievements`` [22] bool, and scalar ``timestep``. Field names verified
    against the installed Craftax-Classic ``EnvState``/``Inventory``.
    """
    inventory = state.inventory
    vitals = np.array(
        [float(getattr(state, f)) for f in VITAL_FIELDS], dtype=np.float32
    )
    counts = np.array(
        [float(getattr(inventory, f)) for f in INVENTORY_FIELDS], dtype=np.float32
    )
    return {
        "vitals": vitals,
        "inventory": counts,
        "achievements": np.asarray(state.achievements, dtype=bool),
        "timestep": int(state.timestep),
    }


def achievement_names() -> list[str]:
    """The 22 achievement names in Craftax ``Achievement`` enum order."""
    from craftax.craftax_classic.constants import Achievement

    return [a.name.lower() for a in Achievement]


def obs_to_chw_uint8(
    obs: np.ndarray, *, target_size: int = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """Convert a Craftax HWC float ``[0,1]`` frame to CHW uint8 ``[0,255]``.

    The native 63x63 frame is padded with zeros at the bottom and right to
    ``target_size`` (default 64). No interpolation is used, so no game
    information is altered; padding only appends blank border rows/columns.
    """
    arr = np.asarray(obs)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC RGB frame, got shape {arr.shape}")
    h, w, _ = arr.shape
    if h != NATIVE_OBS_HW or w != NATIVE_OBS_HW:
        raise ValueError(
            f"expected {NATIVE_OBS_HW}x{NATIVE_OBS_HW} native frame, got {h}x{w}"
        )
    if target_size < NATIVE_OBS_HW:
        raise ValueError("target_size must be >= native 63")
    scaled = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    pad_h = target_size - h
    pad_w = target_size - w
    if pad_h or pad_w:
        scaled = np.pad(
            scaled, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))  # CHW


@dataclass
class StepResult:
    obs: np.ndarray          # CHW uint8 [C, target, target]
    reward: float
    continuation: float      # 1 - dead (timeout -> 1)
    done: bool               # env termination (timeout OR death OR lava)
    terminal: bool           # true absorbing failure (death/lava, not timeout)
    achievements: np.ndarray  # bool [22], cumulative
    score: float             # per-episode Craftax "score" (= Crafter formula)


class CraftaxPixelEnv:
    """Stateful wrapper over the functional Craftax-Classic pixels env.

    Holds the JAX PRNG key and env state so callers get a simple
    ``reset() -> obs`` / ``step(action) -> StepResult`` loop. Observations are
    CHW uint8 at ``target_size``.
    """

    def __init__(self, *, seed: int = 0, target_size: int = DEFAULT_TARGET_SIZE):
        import jax
        from craftax.craftax_env import make_craftax_env_from_name

        self._jax = jax
        self.target_size = int(target_size)
        self.env = make_craftax_env_from_name(CRAFTAX_ENV_NAME, auto_reset=False)
        self.params = self.env.default_params
        self.max_timesteps = int(self.params.max_timesteps)
        self._key = jax.random.PRNGKey(int(seed))
        self._state = None
        self._render_hi = None  # 16px (144x144) renderer, built on first use

    def high_res(self) -> np.ndarray:
        """Render the CURRENT state at 16px/tile -> CHW uint8 [3,144,144].

        This is the dataset/human render resolution (BLOCK_PIXEL_SIZE_IMG), used
        ONLY by the Stage-0 resolution-parity oracle to compare 144 vs the 63->64
        agent render on identical states. Not used for training or deployment.
        """
        if self._state is None:
            raise RuntimeError("call reset() before high_res()")
        if self._render_hi is None:
            from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

            # JIT so the 144x144 render compiles once instead of running eager
            # (eager CPU rendering is ~0.5s/frame; jitted is milliseconds).
            self._render_hi = self._jax.jit(make_craftax_pixel_renderer(16))
        frame = np.asarray(self._render_hi(self._state))  # HWC [0,255]
        scaled = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(scaled.transpose(2, 0, 1))

    def _split(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def reset(self) -> np.ndarray:
        obs, self._state = self.env.reset(self._split(), self.params)
        return obs_to_chw_uint8(obs, target_size=self.target_size)

    def privileged(self) -> dict:
        """Ground-truth simulator labels for the CURRENT state (oracle only)."""
        if self._state is None:
            raise RuntimeError("call reset() before privileged()")
        return privileged_labels(self._state)

    def step(self, action: int) -> StepResult:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        obs, self._state, reward, done, info = self.env.step(
            self._split(), self._state, int(action), self.params
        )
        done = bool(done)
        timeout = int(self._state.timestep) >= self.max_timesteps
        terminal = done and not timeout
        continuation = 0.0 if terminal else 1.0
        achievements = np.asarray(self._state.achievements, dtype=bool)
        return StepResult(
            obs=obs_to_chw_uint8(obs, target_size=self.target_size),
            reward=float(reward),
            continuation=continuation,
            done=done,
            terminal=terminal,
            achievements=achievements,
            score=float(info["score"]),
        )


@dataclass
class CollectedEpisode:
    episode: Episode
    achievements: np.ndarray  # bool [22] final cumulative
    score: float              # official-formula score for this single episode
    length: int               # number of transitions
    timed_out: bool


def collect_episode(
    *,
    seed: int,
    action_fn: Callable[[np.ndarray, int], int],
    max_steps: int,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> CollectedEpisode:
    """Roll one full episode-bounded trajectory in Craftax.

    ``action_fn(obs_chw_uint8, t)`` returns the action index for step ``t``. The
    episode ends on environment termination or after ``max_steps`` transitions
    (whichever first). Returns an ``Episode`` in the ``[T+1, T, T, T]`` contract.
    """
    env = CraftaxPixelEnv(seed=seed, target_size=target_size)
    obs0 = env.reset()
    observations = [obs0]
    actions: list[int] = []
    rewards: list[float] = []
    continues: list[float] = []
    timed_out = False
    final_achievements = np.zeros(N_ACHIEVEMENTS, dtype=bool)
    final_score = 0.0
    for t in range(int(max_steps)):
        action = int(action_fn(observations[-1], t))
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} out of range [0,{N_ACTIONS})")
        result = env.step(action)
        observations.append(result.obs)
        actions.append(action)
        rewards.append(result.reward)
        continues.append(result.continuation)
        final_achievements = result.achievements
        final_score = result.score
        if result.done:
            timed_out = not result.terminal
            break
    episode = Episode(
        obs=np.stack(observations).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        continues=np.asarray(continues, dtype=np.float32),
    )
    return CollectedEpisode(
        episode=episode,
        achievements=final_achievements,
        score=final_score,
        length=len(actions),
        timed_out=timed_out,
    )


__all__ = [
    "CRAFTAX_ENV_NAME",
    "N_ACTIONS",
    "N_ACHIEVEMENTS",
    "NATIVE_OBS_HW",
    "achievement_names",
    "obs_to_chw_uint8",
    "CraftaxPixelEnv",
    "StepResult",
    "CollectedEpisode",
    "collect_episode",
]
