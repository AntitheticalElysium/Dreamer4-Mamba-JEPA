"""Self-contained episode replay and sequence adapters.

This module has NO dependency on any other project track (``m3_hjwm_compact``)
or on the danijar/crafter package. ``Episode`` and ``EpisodeReplay`` are pure
numpy/torch containers, reimplemented locally so the ``d4_mamba_jepa`` package
is a closed experimental boundary (supersedes the old cross-track import; see
the Craftax migration). The live environment adapter lives in
``craftax_env.py`` and is imported only by data-generation / executed-eval
scripts, never by the training modules, so importing this file never pulls in
JAX.

Contract (unchanged from the validated baseline so all downstream torch code is
byte-compatible):

- ``Episode.obs`` is uint8 ``[T+1, C, H, W]``; ``actions``/``rewards``/
  ``continues`` are ``[T]``. Observations are one longer than transitions.
- ``EpisodeReplay.sample`` returns episode-bounded windows in the block-causal
  led-to convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch


@dataclass
class Episode:
    """One episode-bounded trajectory.

    ``obs`` length equals ``actions`` length + 1: the terminal observation has
    no outgoing action.
    """

    obs: np.ndarray          # uint8 [T+1, C, H, W]
    actions: np.ndarray      # int64 [T]
    rewards: np.ndarray      # float32 [T]
    continues: np.ndarray    # float32 [T]


class EpisodeReplay:
    """FIFO episode-bounded replay with reproducible window sampling.

    Reimplemented locally from the validated baseline contract. Verification
    code always passes an explicitly seeded ``numpy.random.Generator`` so that
    architecture arms observe identical window indices.
    """

    def __init__(self, capacity_steps: int = 200_000):
        self.capacity_steps = capacity_steps
        self.episodes: list[Episode] = []
        self.steps = 0

    def add(self, episode: Episode) -> None:
        transitions = len(episode.actions)
        if (
            len(episode.obs) != transitions + 1
            or len(episode.rewards) != transitions
            or len(episode.continues) != transitions
        ):
            raise ValueError(
                "episode actions/rewards/continues must have equal length and "
                "obs must have one additional element"
            )
        self.episodes.append(episode)
        self.steps += transitions
        while self.steps > self.capacity_steps and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.steps -= len(old.actions)

    def sample(
        self,
        batch: int,
        observations: int,
        device,
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``batch`` episode-bounded windows of ``observations`` frames.

        No window crosses an episode boundary. ``previous_actions[t]`` is the
        action that led to observation ``t`` (``-1`` at a true episode start).
        """
        valid = [ep for ep in self.episodes if len(ep.obs) >= observations]
        if not valid:
            raise RuntimeError("no sufficiently long episode in replay")
        randint = np.random.randint if rng is None else rng.integers
        obs, actions, rewards, continues, previous_actions = [], [], [], [], []
        for _ in range(batch):
            ep = valid[int(randint(len(valid)))]
            start = int(randint(0, len(ep.obs) - observations + 1))
            obs.append(ep.obs[start:start + observations])
            actions.append(ep.actions[start:start + observations - 1])
            rewards.append(ep.rewards[start:start + observations - 1])
            continues.append(ep.continues[start:start + observations - 1])
            previous = np.full(observations, -1, dtype=np.int64)
            if start > 0:
                previous[0] = ep.actions[start - 1]
            previous[1:] = ep.actions[start:start + observations - 1]
            previous_actions.append(previous)
        return {
            "obs": torch.from_numpy(np.stack(obs)).to(device),
            "actions": torch.from_numpy(np.stack(actions)).to(device),
            "rewards": torch.from_numpy(np.stack(rewards)).to(device),
            "continues": torch.from_numpy(np.stack(continues)).to(device),
            "previous_actions": torch.from_numpy(
                np.stack(previous_actions)
            ).to(device),
        }


@dataclass(frozen=True)
class SequenceBatch:
    """Episode-bounded sequence in the block-causal led-to convention."""

    observations: torch.Tensor  # uint8 [B,T,C,H,W]
    led_to_actions: torch.Tensor  # int64 [B,T], -1 is the start action
    led_to_rewards: torch.Tensor  # float [B,T]
    led_to_continues: torch.Tensor  # float [B,T]
    outcome_valid: torch.Tensor  # bool [B,T]


def replay_sample_to_sequence(sample: dict[str, torch.Tensor]) -> SequenceBatch:
    observations = sample["obs"]
    actions = sample["actions"]
    rewards = sample["rewards"]
    continues = sample["continues"]
    previous = sample["previous_actions"]
    if observations.ndim != 5:
        raise ValueError("obs must have shape [B,T,C,H,W]")
    B, T = observations.shape[:2]
    expected = (B, T - 1)
    for name, tensor in (
        ("actions", actions),
        ("rewards", rewards),
        ("continues", continues),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected}")
    if previous.shape != (B, T):
        raise ValueError(
            f"previous_actions shape {tuple(previous.shape)} != {(B, T)}"
        )

    led_rewards = torch.zeros((B, T), device=rewards.device, dtype=rewards.dtype)
    led_continues = torch.zeros(
        (B, T), device=continues.device, dtype=continues.dtype
    )
    valid = torch.zeros((B, T), device=observations.device, dtype=torch.bool)
    led_rewards[:, 1:] = rewards
    led_continues[:, 1:] = continues
    valid[:, 1:] = True
    return SequenceBatch(
        observations=observations,
        led_to_actions=previous.to(torch.long),
        led_to_rewards=led_rewards,
        led_to_continues=led_continues,
        outcome_valid=valid,
    )


def transitions_to_led_to(
    actions: torch.Tensor, start_action: int = -1
) -> torch.Tensor:
    """Map actions ``a_t`` to the state slot they produce.

    Input shape is ``[B,T-1]`` for a sequence containing ``T`` observations.
    Output slot zero is the start action and output slot ``t+1`` is ``a_t``.
    """
    if actions.ndim != 2:
        raise ValueError("actions must have shape [B,T-1]")
    out = torch.full(
        (actions.shape[0], actions.shape[1] + 1),
        int(start_action),
        device=actions.device,
        dtype=torch.long,
    )
    out[:, 1:] = actions.to(torch.long)
    return out


def load_episode_replay(
    path: str | Path,
    *,
    expected_sha256: str,
    capacity_steps: int = 500_000,
) -> EpisodeReplay:
    """Load and fully validate a hash-pinned list of episode dictionaries."""
    source = Path(path)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"replay digest drift for {source}: {actual} != {expected_sha256}"
        )
    records = torch.load(source, map_location="cpu", weights_only=False)
    # Bidirectional probe isolation: privileged-label probe payloads carry a
    # marker and must never be trainable, even if their hash were supplied.
    if isinstance(records, dict) and str(records.get("marker", "")).startswith(
        "d4_mamba_jepa_craftax_probe_only"
    ):
        raise RuntimeError(
            "refusing to load a probe-only payload as training replay"
        )
    if not isinstance(records, list):
        raise RuntimeError("replay payload must be a list of episodes")
    replay = EpisodeReplay(capacity_steps=capacity_steps)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"episode {index} is not a dictionary")
        missing = {"obs", "actions", "rewards", "continues"} - set(record)
        if missing:
            raise RuntimeError(f"episode {index} is missing {sorted(missing)}")
        replay.add(
            Episode(
                obs=np.asarray(record["obs"]),
                actions=np.asarray(record["actions"]),
                rewards=np.asarray(record["rewards"]),
                continues=np.asarray(record["continues"]),
            )
        )
    return replay


__all__ = [
    "Episode",
    "EpisodeReplay",
    "SequenceBatch",
    "replay_sample_to_sequence",
    "transitions_to_led_to",
    "load_episode_replay",
]
