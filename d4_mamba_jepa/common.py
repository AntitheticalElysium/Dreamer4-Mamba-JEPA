"""Env-agnostic helpers shared by the training, evaluation and probe paths.

Extracted verbatim from the retired CartPole baseline module. Nothing here is
task-specific.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import Tensor, nn

from .checkpoint import file_sha256
from .config import D4LiteConfig
from .data import (
    Episode,
    EpisodeReplay,
    SequenceBatch,
    replay_sample_to_sequence,
)
from .model import D4LiteWorld

POLICY_FORMAT = "d4_lite_cartpole_bc_policy_v1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        with open(temporary_name, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return file_sha256(path)


class BCPolicy(nn.Module):
    """Categorical port of MMBench2's gradient-isolated BC policy head."""

    def __init__(self, *, d_model: int, n_actions: int):
        super().__init__()
        from .source import load_mmbench2_model

        upstream = load_mmbench2_model()
        self.d_model = int(d_model)
        self.n_actions = int(n_actions)
        self.pool_query = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.pool_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.projector = upstream.MLP(
            d_model=self.d_model,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        self.out = nn.Linear(self.d_model, self.n_actions)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        if agent_tokens.ndim != 4:
            raise ValueError("agent tokens must have shape [B,T,N,D]")
        _, _, _, width = agent_tokens.shape
        key, value = self.pool_kv(agent_tokens).chunk(2, dim=-1)
        query = self.pool_query.to(dtype=key.dtype)
        scores = (key * query).sum(dim=-1) / np.sqrt(width)
        pooled = (scores.softmax(dim=-1)[..., None] * value).sum(dim=2)
        return self.out(self.projector(pooled))


def _episode_window(
    episode: Episode,
    *,
    start: int,
    observations: int,
) -> dict[str, np.ndarray]:
    stop = start + observations
    previous = np.full(observations, -1, dtype=np.int64)
    if start > 0:
        previous[0] = episode.actions[start - 1]
    previous[1:] = episode.actions[start : stop - 1]
    return {
        "obs": episode.obs[start:stop],
        "actions": episode.actions[start : stop - 1],
        "rewards": episode.rewards[start : stop - 1],
        "continues": episode.continues[start : stop - 1],
        "previous_actions": previous,
    }


def sample_sequences(
    replay: EpisodeReplay,
    *,
    batch_size: int,
    sequence_length: int,
    terminal_fraction: float,
    device: torch.device,
    rng: np.random.Generator,
) -> SequenceBatch:
    """Sample ordinary and episode-end windows without crossing boundaries."""
    if not 0.0 <= terminal_fraction <= 1.0:
        raise ValueError("terminal_fraction must lie in [0,1]")
    terminal_count = min(
        batch_size, max(0, int(round(batch_size * terminal_fraction)))
    )
    rows: list[dict[str, np.ndarray]] = []
    if terminal_count:
        terminal_episodes = [
            episode
            for episode in replay.episodes
            if len(episode.obs) >= sequence_length
            and len(episode.continues)
            and float(episode.continues[-1]) == 0.0
        ]
        if not terminal_episodes:
            raise RuntimeError("terminal sampling requested but no end windows exist")
        for _ in range(terminal_count):
            episode = terminal_episodes[int(rng.integers(len(terminal_episodes)))]
            rows.append(
                _episode_window(
                    episode,
                    start=len(episode.obs) - sequence_length,
                    observations=sequence_length,
                )
            )

    remaining = batch_size - terminal_count
    if remaining:
        uniform = replay.sample(
            batch=remaining,
            observations=sequence_length,
            device=torch.device("cpu"),
            rng=rng,
        )
        for index in range(remaining):
            rows.append(
                {
                    name: uniform[name][index].numpy()
                    for name in (
                        "obs",
                        "actions",
                        "rewards",
                        "continues",
                        "previous_actions",
                    )
                }
            )
    order = rng.permutation(len(rows))
    sample = {
        name: torch.from_numpy(
            np.stack([rows[int(index)][name] for index in order])
        ).to(device)
        for name in (
            "obs",
            "actions",
            "rewards",
            "continues",
            "previous_actions",
        )
    }
    return replay_sample_to_sequence(sample)


def _clean_agent_tokens(
    world: D4LiteWorld,
    batch: SequenceBatch,
) -> torch.Tensor:
    encoded = world.encode_frames(batch.observations, frozen=True)
    batch_size, time = encoded.packed.shape[:2]
    steps = torch.full(
        (batch_size, time),
        world.cfg.max_step_index,
        device=encoded.packed.device,
        dtype=torch.long,
    )
    signals = torch.full(
        (batch_size, time),
        world.cfg.k_max,
        device=encoded.packed.device,
        dtype=torch.long,
    )
    _, agent = world.forward_dynamics(
        encoded.packed,
        batch.led_to_actions,
        steps,
        signals,
    )
    return agent


def load_bc_policy(
    path: Path,
    *,
    expected_sha256: str,
    expected_world_sha256: str,
    device: torch.device,
) -> tuple[BCPolicy, dict]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"policy checkpoint digest drift: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != POLICY_FORMAT:
        raise RuntimeError("unsupported CartPole policy checkpoint")
    if payload.get("world_checkpoint_sha256") != expected_world_sha256:
        raise RuntimeError("policy/world checkpoint pairing drift")
    policy = BCPolicy(**payload["config"]).to(device)
    policy.load_state_dict(payload["policy"], strict=True)
    policy.eval()
    return policy, payload


def paired_bootstrap_interval(
    differences: list[float],
    *,
    seed: int,
    draws: int = 20_000,
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2:
        raise ValueError("paired interval requires at least two rows")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]
