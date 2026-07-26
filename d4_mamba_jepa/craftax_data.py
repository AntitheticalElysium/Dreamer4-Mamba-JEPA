"""Generate hash-pinned Craftax-Classic replay datasets.

Rolls episodes in the native Craftax-Classic pixels env (via ``craftax_env``)
and serializes them in the ``load_episode_replay`` schema (a list of episode
dicts with ``obs``/``actions``/``rewards``/``continues``), plus the cumulative
per-transition achievement array needed for the oracle's privileged labels.

Imports JAX/Craftax (through ``craftax_env``); run as a standalone generation
job, never from the training process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .craftax_env import (
    N_ACHIEVEMENTS,
    N_ACTIONS,
    achievement_names,
    collect_episode,
)


FORMAT = "d4_mamba_jepa_craftax_replay_v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def random_action_fn(seed: int):
    """A per-episode-seeded uniform random policy over the 17 actions."""
    rng = np.random.default_rng(seed)

    def action_fn(_obs: np.ndarray, _t: int) -> int:
        return int(rng.integers(N_ACTIONS))

    return action_fn


# Action indices (Crafter order): 1-4 movement, 5 DO, 7 PLACE_STONE,
# 8 PLACE_TABLE, 10 PLACE_PLANT, 11 MAKE_WOOD_PICKAXE, 14 MAKE_WOOD_SWORD.
def forage_action_fn(seed: int):
    """Do/movement-biased exploratory policy that actually collects resources.

    Uniform-random play almost never accumulates inventory, so the resolution-
    sensitive labels (rare item counts, achievements) have no variance to probe.
    This scripted mixture wanders and interacts, unlocking early achievements
    (collect wood/stone/sapling, place table) far more often. It is a diagnostic
    data source for the oracle, NOT a learned or evaluated policy.
    """
    rng = np.random.default_rng(seed)
    actions = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 0])
    weights = np.array([0.14, 0.14, 0.14, 0.14, 0.30, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01])
    weights = weights / weights.sum()

    def action_fn(_obs: np.ndarray, _t: int) -> int:
        return int(rng.choice(actions, p=weights))

    return action_fn


@dataclass
class ReplayManifest:
    format: str
    n_episodes: int
    n_transitions: int
    env_seed_base: int
    max_steps: int
    target_size: int
    policy: str
    replay_sha256: str
    achievement_names: list[str]
    created: str


def generate_random_replay(
    *,
    out_path: str | Path,
    n_episodes: int,
    max_steps: int,
    env_seed_base: int = 0,
    target_size: int = 64,
) -> ReplayManifest:
    """Generate ``n_episodes`` uniform-random episodes and save the replay.

    Each saved episode dict additionally carries ``achievements`` as a bool
    array ``[T+1, 22]`` (cumulative per frame), reconstructed from the final
    achievement set so the oracle can read per-frame privileged labels without
    re-simulating. (Random rollouts rarely unlock anything, so the cumulative
    trace is dominated by zeros; the real signal comes from expert data later.)
    """
    out = Path(out_path)
    records: list[dict] = []
    total_transitions = 0
    for i in range(int(n_episodes)):
        seed = int(env_seed_base) + i
        collected = collect_episode(
            seed=seed,
            action_fn=random_action_fn(seed),
            max_steps=max_steps,
            target_size=target_size,
        )
        ep = collected.episode
        # Per-frame cumulative achievements: unknown per-step timing from a
        # single rollout, so store the final cumulative vector broadcast to the
        # terminal frame and zeros before (a conservative, non-leaking default;
        # the probe collector in the oracle reads exact per-step state instead).
        achievements = np.zeros((ep.obs.shape[0], N_ACHIEVEMENTS), dtype=bool)
        achievements[-1] = collected.achievements
        records.append(
            {
                "obs": torch.from_numpy(ep.obs),
                "actions": torch.from_numpy(ep.actions),
                "rewards": torch.from_numpy(ep.rewards),
                "continues": torch.from_numpy(ep.continues),
                "achievements": torch.from_numpy(achievements),
                "env_seed": seed,
                "timed_out": bool(collected.timed_out),
            }
        )
        total_transitions += collected.length

    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = _serialize(records)
    replay_sha = _sha256_bytes(buffer)
    out.write_bytes(buffer)

    manifest = ReplayManifest(
        format=FORMAT,
        n_episodes=len(records),
        n_transitions=total_transitions,
        env_seed_base=int(env_seed_base),
        max_steps=int(max_steps),
        target_size=int(target_size),
        policy="uniform_random",
        replay_sha256=replay_sha,
        achievement_names=achievement_names(),
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest


def _serialize(records: list[dict]) -> bytes:
    """Deterministic torch serialization to a bytes buffer (for hashing)."""
    import io

    buffer = io.BytesIO()
    torch.save(records, buffer)
    return buffer.getvalue()


def whole_episode_splits(
    n_episodes: int, *, seed: int, fractions=(0.8, 0.1, 0.1)
) -> dict[str, list[int]]:
    """Deterministic disjoint whole-episode train/dev/sealed index split."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(n_episodes))
    n_train = int(round(fractions[0] * n_episodes))
    n_dev = int(round(fractions[1] * n_episodes))
    train = sorted(int(i) for i in order[:n_train])
    dev = sorted(int(i) for i in order[n_train:n_train + n_dev])
    sealed = sorted(int(i) for i in order[n_train + n_dev:])
    return {"train": train, "dev": dev, "sealed": sealed}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a Craftax replay.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=64)
    args = parser.parse_args()
    manifest = generate_random_replay(
        out_path=args.out,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        env_seed_base=args.seed_base,
        target_size=args.target_size,
    )
    print(json.dumps(asdict(manifest), indent=2))
    print(f"host: {platform.node()}")


if __name__ == "__main__":
    _cli()


__all__ = [
    "FORMAT",
    "ReplayManifest",
    "generate_random_replay",
    "whole_episode_splits",
    "random_action_fn",
    "forage_action_fn",
]
