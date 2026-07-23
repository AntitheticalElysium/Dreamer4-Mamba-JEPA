"""Source-pinned, small executed-control baseline for the D4-lite stack.

This runner deliberately keeps both research switches off. It trains the
unchanged MMBench2 Transformer tokenizer/dynamics at reduced scale on official
Gymnasium CartPole RGB observations, then evaluates a frozen world with
receding-horizon categorical random shooting. The collection controller is
used only to create an offline dynamics replay; executed planner evaluation
has access to pixels and past actions only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn

from .checkpoint import (
    file_sha256,
    implementation_sha256,
    load_checkpoint,
    save_checkpoint,
    save_tokenizer_checkpoint,
)
from .config import D4LiteConfig
from .crafter_preflight import evaluate_tokenizer, evaluate_world
from .data import Episode, EpisodeReplay, SequenceBatch, replay_sample_to_sequence
from .model import D4LiteWorld, build_tokenizer
from .objectives import jepa_self_prediction_loss, optimizer_groups
from .rollout import categorical_random_shooting, shortcut_schedule
from .source import GYMNASIUM_CARTPOLE, source_report, verify_installed_cartpole
from .training import (
    WorldLossNormalizer,
    tokenizer_reconstruction_loss,
    world_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT = "d4_lite_cartpole_control_v1"
DATA_FORMAT = "d4_lite_cartpole_pixels_v1"
POLICY_FORMAT = "d4_lite_cartpole_bc_policy_v1"
ENVIRONMENT_ID = "CartPole-v1"
ACTION_REPEAT = 2


def cartpole_config() -> D4LiteConfig:
    """The named control-baseline configuration; no Mamba or CDP."""
    return D4LiteConfig(
        image_size=64,
        channels=3,
        patch_size=8,
        sequence_length=12,
        n_actions=2,
        tokenizer_d_model=64,
        tokenizer_heads=4,
        tokenizer_depth=4,
        tokenizer_time_every=2,
        tokenizer_mlp_ratio=4.0,
        n_latents=16,
        d_bottleneck=16,
        mae_p_min=0.0,
        mae_p_max=0.9,
        dynamics_d_model=64,
        dynamics_heads=4,
        dynamics_depth=4,
        dynamics_time_every=2,
        dynamics_mlp_ratio=4.0,
        packing_factor=4,
        n_register=2,
        n_agent=2,
        k_max=4,
        reward_horizon=8,
        reward_bins=255,
        reward_log_low=-10.0,
        reward_log_high=10.0,
        continuation_horizon=8,
        temporal_backend="transformer",
        representation_objective="base",
    )


def cartpole_jepa_config(temporal_backend: str = "transformer") -> D4LiteConfig:
    """Non-generative JEPA arm: identical to the T-BASE control except the
    representation objective. Tokenizer scale, pixel adapter, heads, and every
    non-temporal axis are held fixed.

    ``temporal_backend="mamba2"`` selects the combined `M-JEPA` arm. It carries
    the D022 state expansion (`d_state=64`, `headdim=64`) explicitly, because the
    dataclass defaults are still the parameter-matched `d_state=16, headdim=32`
    of the rejected D021 configuration.
    """
    cfg = replace(cartpole_config(), representation_objective="jepa")
    if temporal_backend == "transformer":
        return cfg
    if temporal_backend != "mamba2":
        raise ValueError(f"unsupported temporal_backend={temporal_backend!r}")
    return replace(
        cfg,
        temporal_backend="mamba2",
        mamba_d_state=64,
        mamba_headdim=64,
        mamba_expand=1,
        mamba_d_conv=4,
    )


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


def _cart_center(frame: np.ndarray) -> tuple[int, int]:
    cart_pixels = np.all(
        frame == np.asarray([129, 132, 203], dtype=np.uint8), axis=-1
    )
    coordinates = np.argwhere(cart_pixels)
    if len(coordinates):
        center_y, center_x = np.median(coordinates, axis=0).round().astype(int)
        return int(center_y), int(center_x)
    # A position-terminal cart can be just outside the official viewport.
    return 300, frame.shape[1] // 2


def _foreground_view(
    frame: np.ndarray,
    *,
    image_size: int,
    crop: tuple[int, int, int, int] | None = None,
    thicken: bool,
) -> np.ndarray:
    """Generic high-contrast foreground extracted only from RGB pixels."""
    source = Image.fromarray(frame)
    if crop is not None:
        source = source.crop(crop)
    resized = np.asarray(
        source.resize(
            (image_size, image_size),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    background_distance = np.max(
        255 - resized.astype(np.int16),
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(background_distance)
    if thicken:
        image = image.filter(ImageFilter.MaxFilter(3))
    return np.asarray(image, dtype=np.uint8)


def preprocess_rgb(
    frame: np.ndarray,
    image_size: int = 64,
    *,
    previous_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Convert the official render to three generic, pixel-only control views.

    Channel 0 is the full-scene foreground, channel 1 is a cart-localized zoom,
    and channel 2 is full-scene frame difference. Foreground is simply distance
    from the white render background followed by a 3-pixel max filter. This is
    a documented small-resolution adapter, not simulator-state rendering.
    """
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError("CartPole render must be HWC uint8 RGB")
    if previous_frame is None:
        previous_frame = frame
    if previous_frame.shape != frame.shape or previous_frame.dtype != np.uint8:
        raise ValueError("previous CartPole frame contract differs")
    center_y, center_x = _cart_center(frame)
    crop_size = 256
    left = int(
        np.clip(center_x - crop_size // 2, 0, frame.shape[1] - crop_size)
    )
    top = int(
        np.clip(
            center_y - 3 * crop_size // 4,
            0,
            frame.shape[0] - crop_size,
        )
    )
    crop = (left, top, left + crop_size, top + crop_size)
    full = _foreground_view(
        frame, image_size=image_size, thicken=True
    )
    zoom = _foreground_view(
        frame, image_size=image_size, crop=crop, thicken=True
    )
    previous_full = _foreground_view(
        previous_frame, image_size=image_size, thicken=False
    )
    current_full = _foreground_view(
        frame, image_size=image_size, thicken=False
    )
    motion = np.abs(
        current_full.astype(np.int16) - previous_full.astype(np.int16)
    ).clip(0, 255).astype(np.uint8)
    motion = np.asarray(
        Image.fromarray(motion).filter(ImageFilter.MaxFilter(3)),
        dtype=np.uint8,
    )
    return np.ascontiguousarray(np.stack([full, zoom, motion], axis=0))


class CartPolePixels:
    """Thin adapter around the exact installed and source-pinned environment."""

    def __init__(self, *, image_size: int = 64, max_episode_steps: int | None = None):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        verify_installed_cartpole()
        import gymnasium as gym

        # max_episode_steps=None keeps the pinned CartPole-v1 registration exactly
        # (500). An explicit value overrides only the TimeLimit truncation wrapper;
        # dynamics, reward, termination and the renderer are untouched.
        extra = (
            {} if max_episode_steps is None
            else {"max_episode_steps": int(max_episode_steps)}
        )
        self.env = gym.make(ENVIRONMENT_ID, render_mode="rgb_array", **extra)
        if int(self.env.action_space.n) != 2:
            raise RuntimeError("CartPole action contract drift")
        self.image_size = int(image_size)
        self.state: np.ndarray | None = None
        self.previous_rgb: np.ndarray | None = None

    def reset(self, *, seed: int) -> np.ndarray:
        state, _ = self.env.reset(seed=int(seed))
        self.state = np.asarray(state, dtype=np.float32)
        frame = self.env.render()
        self.previous_rgb = frame.copy()
        return preprocess_rgb(
            frame,
            self.image_size,
            previous_frame=self.previous_rgb,
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, float, bool, bool]:
        total_reward = 0.0
        terminated = truncated = False
        state = None
        for _ in range(ACTION_REPEAT):
            state, reward, terminated, truncated, _ = self.env.step(int(action))
            total_reward += float(reward)
            if terminated or truncated:
                break
        assert state is not None
        self.state = np.asarray(state, dtype=np.float32)
        frame = self.env.render()
        previous = self.previous_rgb
        if previous is None:
            raise RuntimeError("CartPole pixel adapter was not reset")
        processed = preprocess_rgb(
            frame,
            self.image_size,
            previous_frame=previous,
        )
        self.previous_rgb = frame.copy()
        done = bool(terminated or truncated)
        return (
            processed,
            total_reward,
            float(not done),
            bool(terminated),
            bool(truncated),
        )

    def close(self) -> None:
        self.env.close()


class CartPoleBCPolicy(nn.Module):
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


def _collection_action(
    policy: str,
    state: np.ndarray,
    *,
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    if policy == "random":
        return int(rng.integers(2))
    if policy != "noisy_balance":
        raise ValueError(f"unsupported collection policy {policy!r}")
    # Transparent data-coverage controller, never called by planner evaluation.
    action = int(float(state[2]) + 0.5 * float(state[3]) > 0.0)
    if float(rng.random()) < epsilon:
        action = int(rng.integers(2))
    return action


def collect_dataset(
    *,
    path: Path,
    random_episodes: int,
    noisy_balance_episodes: int,
    seed_base: int,
    epsilon: float,
    image_size: int = 64,
) -> dict:
    """Collect a deterministic, episode-bounded offline pixel replay."""
    if random_episodes < 1 or noisy_balance_episodes < 1:
        raise ValueError("both replay collection policies require episodes")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")

    records: list[dict[str, np.ndarray]] = []
    policy_stats: dict[str, list[int]] = {"random": [], "noisy_balance": []}
    start = time.perf_counter()
    for policy_index, (policy, count) in enumerate(
        (
            ("random", random_episodes),
            ("noisy_balance", noisy_balance_episodes),
        )
    ):
        for episode_index in range(count):
            environment_seed = (
                int(seed_base) + 100_000 * policy_index + episode_index
            )
            policy_rng = np.random.default_rng(environment_seed + 50_000)
            environment = CartPolePixels(image_size=image_size)
            observations = [environment.reset(seed=environment_seed)]
            states = [environment.state.copy()]
            actions: list[int] = []
            rewards: list[float] = []
            continues: list[float] = []
            terminated = truncated = False
            try:
                while not (terminated or truncated):
                    assert environment.state is not None
                    action = _collection_action(
                        policy,
                        environment.state,
                        rng=policy_rng,
                        epsilon=epsilon,
                    )
                    frame, reward, continuation, terminated, truncated = (
                        environment.step(action)
                    )
                    observations.append(frame)
                    states.append(environment.state.copy())
                    actions.append(action)
                    rewards.append(reward)
                    continues.append(continuation)
            finally:
                environment.close()
            record = {
                "obs": np.stack(observations).astype(np.uint8, copy=False),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "continues": np.asarray(continues, dtype=np.float32),
                "states": np.stack(states).astype(np.float32, copy=False),
                "environment_seed": np.asarray(environment_seed, dtype=np.int64),
                "collection_policy": np.asarray(policy),
            }
            records.append(record)
            policy_stats[policy].append(len(actions))
            if (episode_index + 1) % 10 == 0:
                print(
                    f"collect {policy}: {episode_index + 1}/{count}",
                    flush=True,
                )

    replay_sha256 = _atomic_torch_save(path, records)
    summary = {
        "format": DATA_FORMAT,
        "path": str(path),
        "sha256": replay_sha256,
        "source": {
            "environment_id": ENVIRONMENT_ID,
            "cartpole_commit": GYMNASIUM_CARTPOLE.commit,
            "cartpole_sha256": verify_installed_cartpole(),
        },
        "preprocessing": {
            "source_resolution_hwc": [400, 600, 3],
            "output_resolution_chw": [3, image_size, image_size],
            "view": (
                "channels = full foreground, 256px pixel-localized zoom, "
                "full-scene frame difference"
            ),
            "cart_localization": "exact RGB [129,132,203], no simulator state",
            "foreground": (
                "distance from white RGB background plus 3px max filter"
            ),
            "resize": "Pillow bilinear",
            "action_repeat": ACTION_REPEAT,
        },
        "collection": {
            "seed_base": seed_base,
            "random_episodes": random_episodes,
            "noisy_balance_episodes": noisy_balance_episodes,
            "noisy_balance_rule": "action = int(theta + 0.5 * theta_dot > 0)",
            "epsilon": epsilon,
            "privileged_state_used_only_for_replay_collection": True,
        },
        "episodes": len(records),
        "transitions": int(sum(len(record["actions"]) for record in records)),
        "policy_lengths": {
            policy: {
                "episodes": len(lengths),
                "mean": float(np.mean(lengths)),
                "minimum": int(np.min(lengths)),
                "maximum": int(np.max(lengths)),
            }
            for policy, lengths in policy_stats.items()
        },
        "collection_seconds": time.perf_counter() - start,
    }
    _atomic_json(path.with_suffix(".json"), summary)
    return summary


def load_cartpole_replay(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[EpisodeReplay, list[dict]]:
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(
            f"CartPole replay digest drift: {actual} != {expected_sha256}"
        )
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise RuntimeError("CartPole replay must be a non-empty episode list")
    replay = EpisodeReplay(capacity_steps=2_000_000)
    for index, record in enumerate(records):
        missing = {"obs", "actions", "rewards", "continues"} - set(record)
        if missing:
            raise RuntimeError(f"episode {index} missing {sorted(missing)}")
        replay.add(
            Episode(
                obs=record["obs"],
                actions=record["actions"],
                rewards=record["rewards"],
                continues=record["continues"],
            )
        )
    return replay, records


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


def sample_cartpole_sequences(
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


def _fixed_batches(
    replay: EpisodeReplay,
    *,
    cfg: D4LiteConfig,
    count: int,
    batch_size: int,
    terminal_fraction: float,
    seed: int,
) -> list[SequenceBatch]:
    rng = np.random.default_rng(seed)
    return [
        sample_cartpole_sequences(
            replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=terminal_fraction,
            device=torch.device("cpu"),
            rng=rng,
        )
        for _ in range(count)
    ]


def _mean_window(history: list[dict[str, float]], count: int = 100) -> dict:
    rows = history[:count] if count > 0 else history
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def train_jepa_world(
    *,
    train_replay_path: Path,
    dev_replay_path: Path,
    output_dir: Path,
    device: torch.device,
    world_steps: int,
    batch_size: int,
    learning_rate: float,
    terminal_fraction: float,
    seed: int,
    anticollapse: str = "ema",
    sigreg_lambda: float | None = None,
    temporal_backend: str = "transformer",
    jepa_jumps: int | None = None,
) -> dict:
    """Train the non-generative JEPA world in one joint phase.

    Online encoder + dynamics + action-conditioned predictor + task heads are
    trained together by SPR/BYOL EMA-target self-prediction. There is no
    tokenizer MAE phase, no decoder, no flow. The EMA target encoder/projection
    are updated after every optimizer step with the I-JEPA/V-JEPA momentum ramp.
    """
    overrides = {"jepa_anticollapse": anticollapse}
    if jepa_jumps is not None:
        overrides["jepa_jumps"] = int(jepa_jumps)
    if sigreg_lambda is not None:
        overrides["jepa_sigreg_lambda"] = sigreg_lambda
    cfg = replace(cartpole_jepa_config(temporal_backend), **overrides)
    if cfg.representation_objective != "jepa":
        raise RuntimeError("JEPA arm requires the jepa representation objective")
    if cfg.temporal_backend not in {"transformer", "mamba2"}:
        raise RuntimeError(f"unsupported temporal backend {cfg.temporal_backend!r}")
    if cfg.temporal_backend == "mamba2" and (
        cfg.mamba_d_state != 64 or cfg.mamba_headdim != 64
    ):
        raise RuntimeError(
            "M-JEPA must use the D022 state expansion (d_state=64, headdim=64); "
            "the parameter-matched d_state=16/headdim=32 is the rejected D021"
        )
    # Oversample terminal windows so the continuation head sees enough failures.
    terminal_fraction = max(terminal_fraction, cfg.jepa_terminal_fraction)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_replay, train_records = load_cartpole_replay(train_replay_path)
    dev_replay, dev_records = load_cartpole_replay(dev_replay_path)
    train_sha = file_sha256(train_replay_path)
    dev_sha = file_sha256(dev_replay_path)
    fixed_eval = _fixed_batches(
        dev_replay, cfg=cfg, count=8, batch_size=8,
        terminal_fraction=0.5, seed=seed + 1,
    )

    world = D4LiteWorld(cfg).to(device).train()
    normalizer = WorldLossNormalizer().to(device)
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999),
    )
    world_rng = np.random.default_rng(seed + 3)

    def dev_cosine() -> float:
        world.eval()
        values: list[float] = []
        with torch.no_grad():
            for batch in fixed_eval:
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    observations = batch.observations.to(device)
                    actions = batch.led_to_actions.to(device)
                    clean = world.encode_frames(
                        observations, frozen=True
                    ).packed
                    _, metric = jepa_self_prediction_loss(
                        world, frames=observations, clean=clean,
                        led_to_actions=actions,
                    )
                values.append(float(metric["jepa_cosine"].item()))
        world.train()
        return float(np.mean(values))

    cosine_before = dev_cosine()
    history: list[dict[str, float]] = []
    world_start = time.perf_counter()
    for step in range(world_steps):
        batch = sample_cartpole_sequences(
            train_replay, batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=terminal_fraction, device=device, rng=world_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, metrics = world_loss(world, batch, normalizer=normalizer)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite JEPA world loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite gradient at step {step}")
        if step < 1_000:
            scale = float(step + 1) / 1_000.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
        optimizer.step()
        if cfg.jepa_anticollapse == "ema":
            frac = step / max(1, world_steps - 1)
            tau = cfg.jepa_ema_tau + (cfg.jepa_ema_tau_final - cfg.jepa_ema_tau) * frac
            world.update_jepa_target(tau)
        history.append({
            "jepa": float(metrics["loss/jepa"].item()),
            "reward": float(metrics["loss/reward"].item()),
            "continuation": float(metrics["loss/continuation"].item()),
            "cosine": float(metrics["jepa/jepa_cosine"].item()),
            "online_std": float(metrics["jepa/jepa_online_std"].item()),
        })
        if (step + 1) % 500 == 0:
            recent = history[-100:]
            print(
                f"jepa-world {step + 1}/{world_steps}: "
                f"jepa={np.mean([r['jepa'] for r in recent]):.5f} "
                f"cos={np.mean([r['cosine'] for r in recent]):.4f} "
                f"onstd={np.mean([r['online_std'] for r in recent]):.4f} "
                f"reward={np.mean([r['reward'] for r in recent]):.4f} "
                f"cont={np.mean([r['continuation'] for r in recent]):.4f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    world_seconds = time.perf_counter() - world_start
    cosine_after = dev_cosine()

    output_dir.mkdir(parents=True, exist_ok=True)
    world_path = output_dir / "world.pt"
    world_sha = save_checkpoint(
        world_path, world=world, normalizer=normalizer, optimizer=optimizer,
        numpy_rng=world_rng, step=world_steps,
        extra={
            "format": FORMAT,
            "train_replay_sha256": train_sha,
            "dev_replay_sha256": dev_sha,
            "seed": seed,
            "terminal_fraction": terminal_fraction,
            "representation_objective": "jepa",
            "jepa_ema_tau": cfg.jepa_ema_tau,
            "jepa_ema_tau_final": cfg.jepa_ema_tau_final,
        },
    )
    report = {
        "format": FORMAT,
        "status": "trained",
        "arm_id": cfg.arm_id,
        "config": asdict(cfg),
        "world_checkpoint": {"path": str(world_path), "sha256": world_sha},
        "provenance": {
            "implementation_sha256": implementation_sha256(),
            "sources": source_report(),
            "train_replay": {"path": str(train_replay_path), "sha256": train_sha,
                             "episodes": len(train_records)},
            "dev_replay": {"path": str(dev_replay_path), "sha256": dev_sha,
                           "episodes": len(dev_records)},
        },
        "metrics": {
            "dev_cosine_before": cosine_before,
            "dev_cosine_after": cosine_after,
            "final_jepa": history[-1]["jepa"] if history else None,
            "final_online_std": history[-1]["online_std"] if history else None,
        },
        "optimization": {
            "seed": seed, "world_steps": world_steps, "batch_size": batch_size,
            "sequence_length": cfg.sequence_length, "learning_rate": learning_rate,
            "terminal_fraction": terminal_fraction,
            "warmup": {"world": 1_000},
        },
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device),
            "world_seconds": world_seconds,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
        },
    }
    _atomic_json(output_dir / "jepa_world_report.json", report)
    return report


def train_baseline(
    *,
    train_replay_path: Path,
    dev_replay_path: Path,
    output_dir: Path,
    device: torch.device,
    tokenizer_steps: int,
    world_steps: int,
    batch_size: int,
    learning_rate: float,
    terminal_fraction: float,
    seed: int,
) -> dict:
    """Train the source-pinned Transformer baseline from a fresh initialization."""
    cfg = cartpole_config()
    if cfg.temporal_backend != "transformer" or cfg.representation_objective != "base":
        raise RuntimeError("control baseline research switches must remain off")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_replay, train_records = load_cartpole_replay(train_replay_path)
    dev_replay, dev_records = load_cartpole_replay(dev_replay_path)
    train_sha = file_sha256(train_replay_path)
    dev_sha = file_sha256(dev_replay_path)
    fixed_eval = _fixed_batches(
        dev_replay,
        cfg=cfg,
        count=8,
        batch_size=8,
        terminal_fraction=0.5,
        seed=seed + 1,
    )

    tokenizer = build_tokenizer(cfg, training_mask=True).to(device).train()
    tokenizer_optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    tokenizer_before = evaluate_tokenizer(
        tokenizer, fixed_eval[:4], cfg=cfg, device=device
    )
    tokenizer_rng = np.random.default_rng(seed + 2)
    tokenizer_history: list[float] = []
    tokenizer_start = time.perf_counter()
    for step in range(tokenizer_steps):
        batch = sample_cartpole_sequences(
            train_replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=terminal_fraction,
            device=device,
            rng=tokenizer_rng,
        )
        tokenizer_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _ = tokenizer_reconstruction_loss(
                tokenizer, batch.observations, patch_size=cfg.patch_size
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite tokenizer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        if step < 250:
            scale = float(step + 1) / 250.0
            for group in tokenizer_optimizer.param_groups:
                group["lr"] = learning_rate * scale
        tokenizer_optimizer.step()
        tokenizer_history.append(float(loss.detach().item()))
        if (step + 1) % 500 == 0:
            print(
                f"tokenizer {step + 1}/{tokenizer_steps}: "
                f"{np.mean(tokenizer_history[-100:]):.6f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tokenizer_seconds = time.perf_counter() - tokenizer_start
    tokenizer_after = evaluate_tokenizer(
        tokenizer, fixed_eval[:4], cfg=cfg, device=device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.pt"
    tokenizer_sha = save_tokenizer_checkpoint(
        tokenizer_path,
        tokenizer=tokenizer,
        config=cfg,
        step=tokenizer_steps,
        extra={
            "format": FORMAT,
            "train_replay_sha256": train_sha,
            "dev_replay_sha256": dev_sha,
            "seed": seed,
        },
    )

    world = D4LiteWorld(cfg).to(device)
    world.encoder.load_state_dict(tokenizer.encoder.state_dict(), strict=True)
    world.decoder.load_state_dict(tokenizer.decoder.state_dict(), strict=True)
    world.freeze_tokenizer()
    normalizer = WorldLossNormalizer().to(device)
    groups = optimizer_groups(world, learning_rate)
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    world_rng = np.random.default_rng(seed + 3)
    world_history: list[dict[str, float]] = []
    bootstrap_rows = max(0, min(batch_size - 1, round(0.25 * batch_size)))
    trainable = [
        parameter for group in groups for parameter in group["params"]
    ]
    world_start = time.perf_counter()
    world.train()
    for step in range(world_steps):
        batch = sample_cartpole_sequences(
            train_replay,
            batch_size=batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=terminal_fraction,
            device=device,
            rng=world_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, metrics = world_loss(
                world,
                batch,
                normalizer=normalizer,
                global_step=step,
                bootstrap_rows=bootstrap_rows,
                bootstrap_start=10_000,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite world loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        if step < 1_000:
            scale = float(step + 1) / 1_000.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
        optimizer.step()
        world_history.append(
            {
                "flow": float(metrics["loss/flow"].item()),
                "reward": float(metrics["loss/reward"].item()),
                "continuation": float(metrics["loss/continuation"].item()),
                "total": float(metrics["loss/total"].item()),
            }
        )
        if (step + 1) % 500 == 0:
            recent = world_history[-100:]
            print(
                f"world {step + 1}/{world_steps}: "
                f"flow={np.mean([row['flow'] for row in recent]):.5f} "
                f"reward={np.mean([row['reward'] for row in recent]):.5f} "
                f"continue={np.mean([row['continuation'] for row in recent]):.5f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    world_seconds = time.perf_counter() - world_start
    offline = evaluate_world(
        world,
        fixed_eval,
        cfg=cfg,
        device=device,
        seed=seed + 4,
    )

    world_path = output_dir / "world.pt"
    world_sha = save_checkpoint(
        world_path,
        world=world,
        normalizer=normalizer,
        optimizer=optimizer,
        numpy_rng=world_rng,
        step=world_steps,
        extra={
            "format": FORMAT,
            "train_replay_sha256": train_sha,
            "dev_replay_sha256": dev_sha,
            "tokenizer_checkpoint_sha256": tokenizer_sha,
            "seed": seed,
            "terminal_fraction": terminal_fraction,
            "task_head_route": "same noised shortcut-flow forward as MMBench2",
            "bootstrap_rows": bootstrap_rows,
            "bootstrap_start": 10_000,
        },
    )
    report = {
        "format": FORMAT,
        "status": "trained",
        "claim_boundary": (
            "small official CartPole pixel control baseline; no Mamba, no CDP, "
            "and no claim of official Dreamer 4 reproduction"
        ),
        "config": asdict(cfg),
        "provenance": {
            "implementation_sha256": implementation_sha256(),
            "sources": source_report(),
            "train_replay": {
                "path": str(train_replay_path),
                "sha256": train_sha,
                "episodes": len(train_records),
                "transitions": train_replay.steps,
            },
            "dev_replay": {
                "path": str(dev_replay_path),
                "sha256": dev_sha,
                "episodes": len(dev_records),
                "transitions": dev_replay.steps,
            },
        },
        "optimization": {
            "seed": seed,
            "tokenizer_steps": tokenizer_steps,
            "world_steps": world_steps,
            "batch_size": batch_size,
            "sequence_length": cfg.sequence_length,
            "learning_rate": learning_rate,
            "terminal_fraction": terminal_fraction,
            "warmup": {"tokenizer": 250, "world": 1_000},
            "shortcut_bootstrap": {
                "rows": bootstrap_rows,
                "start": 10_000,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "tokenizer_seconds": tokenizer_seconds,
            "world_seconds": world_seconds,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "tokenizer": {
            "before": tokenizer_before,
            "after": tokenizer_after,
            "first_100_loss": float(np.mean(tokenizer_history[:100])),
            "last_100_loss": float(np.mean(tokenizer_history[-100:])),
            "checkpoint": str(tokenizer_path),
            "checkpoint_sha256": tokenizer_sha,
        },
        "world": {
            "first_100_loss": _mean_window(world_history[:100]),
            "last_100_loss": _mean_window(world_history[-100:]),
            "offline_dev": offline,
            "checkpoint": str(world_path),
            "checkpoint_sha256": world_sha,
        },
    }
    _atomic_json(output_dir / "training_report.json", report)
    return report


def _collection_policy_name(record: dict) -> str:
    value = record.get("collection_policy")
    if value is None:
        raise RuntimeError("CartPole replay lacks collection-policy provenance")
    return str(np.asarray(value).item())


def _policy_replay(records: list[dict], policy_name: str) -> EpisodeReplay:
    replay = EpisodeReplay(capacity_steps=2_000_000)
    for record in records:
        if _collection_policy_name(record) != policy_name:
            continue
        replay.add(
            Episode(
                obs=record["obs"],
                actions=record["actions"],
                rewards=record["rewards"],
                continues=record["continues"],
            )
        )
    if not replay.episodes:
        raise RuntimeError(f"no {policy_name!r} episodes in replay")
    return replay


@torch.inference_mode()
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


@torch.inference_mode()
def _evaluate_bc_accuracy(
    world: D4LiteWorld,
    policy: CartPoleBCPolicy,
    replay: EpisodeReplay,
    *,
    device: torch.device,
    seed: int,
    batches: int = 32,
) -> dict:
    rng = np.random.default_rng(seed)
    correct = total = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    policy.eval()
    for _ in range(batches):
        batch = sample_cartpole_sequences(
            replay,
            batch_size=8,
            sequence_length=world.cfg.sequence_length,
            terminal_fraction=0.0,
            device=device,
            rng=rng,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = policy(_clean_agent_tokens(world, batch))
        prediction = logits[:, :-1].argmax(dim=-1)
        target = batch.led_to_actions[:, 1:]
        correct += int((prediction == target).sum().item())
        total += int(target.numel())
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
    prediction = torch.cat([item.reshape(-1) for item in predictions])
    target = torch.cat([item.reshape(-1) for item in targets])
    return {
        "rows": total,
        "accuracy": correct / total,
        "target_action_one_fraction": float(target.float().mean().item()),
        "predicted_action_one_fraction": float(
            prediction.float().mean().item()
        ),
        "accuracy_action_0": float(
            (prediction[target == 0] == 0).float().mean().item()
        ),
        "accuracy_action_1": float(
            (prediction[target == 1] == 1).float().mean().item()
        ),
    }


def train_bc_policy(
    *,
    world_checkpoint: Path,
    world_checkpoint_sha256: str,
    train_replay_path: Path,
    dev_replay_path: Path,
    output: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    """Train only the source-shaped policy head on demonstration actions."""
    world, _, world_payload = load_checkpoint(
        world_checkpoint,
        device=device,
        expected_sha256=world_checkpoint_sha256,
        strict_implementation=False,
    )
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    _, train_records = load_cartpole_replay(train_replay_path)
    _, dev_records = load_cartpole_replay(dev_replay_path)
    train_replay = _policy_replay(train_records, "noisy_balance")
    dev_replay = _policy_replay(dev_records, "noisy_balance")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    policy = CartPoleBCPolicy(
        d_model=world.cfg.dynamics_d_model,
        n_actions=world.cfg.n_actions,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    rng = np.random.default_rng(seed + 1)
    losses: list[float] = []
    started = time.perf_counter()
    policy.train()
    for step in range(steps):
        batch = sample_cartpole_sequences(
            train_replay,
            batch_size=batch_size,
            sequence_length=world.cfg.sequence_length,
            terminal_fraction=0.0,
            device=device,
            rng=rng,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            agent = _clean_agent_tokens(world, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = policy(agent.detach())
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, world.cfg.n_actions),
                batch.led_to_actions[:, 1:].reshape(-1),
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite BC loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if step < 250:
            scale = float(step + 1) / 250.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (step + 1) % 500 == 0:
            print(
                f"policy {step + 1}/{steps}: "
                f"{np.mean(losses[-100:]):.6f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    accuracy = {
        "train_demonstrations": _evaluate_bc_accuracy(
            world,
            policy,
            train_replay,
            device=device,
            seed=seed + 2,
        ),
        "heldout_demonstrations": _evaluate_bc_accuracy(
            world,
            policy,
            dev_replay,
            device=device,
            seed=seed + 3,
        ),
    }
    payload = {
        "format": POLICY_FORMAT,
        "world_checkpoint_sha256": world_checkpoint_sha256,
        "world_checkpoint_stored_implementation_sha256": world_payload[
            "provenance"
        ]["implementation_sha256"],
        "policy": {
            name: tensor.detach().cpu()
            for name, tensor in policy.state_dict().items()
        },
        "config": {
            "d_model": world.cfg.dynamics_d_model,
            "n_actions": world.cfg.n_actions,
        },
        "optimization": {
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 1e-2,
            "gradient_clip": 1.0,
            "demonstration_policy": "noisy_balance",
            "world_and_tokenizer_frozen": True,
            "bc_gradients_enter_policy_head_only": True,
            "train_replay_sha256": file_sha256(train_replay_path),
            "dev_replay_sha256": file_sha256(dev_replay_path),
        },
        "metrics": {
            "first_100_loss": float(np.mean(losses[:100])),
            "last_100_loss": float(np.mean(losses[-100:])),
            "accuracy": accuracy,
            "train_seconds": train_seconds,
        },
        "provenance": {
            "evaluation_implementation_sha256": implementation_sha256(),
            "sources": source_report(),
        },
    }
    policy_sha256 = _atomic_torch_save(output, payload)
    report = {
        **payload,
        "policy": {
            "path": str(output),
            "sha256": policy_sha256,
        },
    }
    _atomic_json(output.with_suffix(".json"), report)
    return report


def load_bc_policy(
    path: Path,
    *,
    expected_sha256: str,
    expected_world_sha256: str,
    device: torch.device,
) -> tuple[CartPoleBCPolicy, dict]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"policy checkpoint digest drift: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != POLICY_FORMAT:
        raise RuntimeError("unsupported CartPole policy checkpoint")
    if payload.get("world_checkpoint_sha256") != expected_world_sha256:
        raise RuntimeError("policy/world checkpoint pairing drift")
    policy = CartPoleBCPolicy(**payload["config"]).to(device)
    policy.load_state_dict(payload["policy"], strict=True)
    policy.eval()
    return policy, payload


@torch.inference_mode()
def _planner_action(
    world: D4LiteWorld,
    *,
    observations: list[np.ndarray],
    led_to_actions: list[int],
    context: int,
    horizon: int,
    candidates: int,
    schedule: dict,
    discount: float,
    generator: torch.Generator,
    device: torch.device,
    common_random_numbers: bool,
    selection: str,
    enumerate_all: bool,
) -> tuple[int, dict]:
    frames = torch.from_numpy(np.stack(observations[-context:]))[None].to(device)
    actions = torch.tensor(
        led_to_actions[-context:],
        device=device,
        dtype=torch.long,
    )[None]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        packed = world.encode_frames(frames, frozen=True).packed
        result = categorical_random_shooting(
            world,
            context_packed=packed,
            context_led_to_actions=actions,
            horizon=horizon,
            candidates=candidates,
            schedule=schedule,
            discount=discount,
            use_cache=True,
            generator=generator,
            common_random_numbers=common_random_numbers,
            selection=selection,
            enumerate_all=enumerate_all,
        )
    scores = result.scores.float().cpu()
    first_actions = result.plans[:, 0].cpu()
    action_means = {
        str(action): float(scores[first_actions == action].mean().item())
        for action in range(world.cfg.n_actions)
    }
    return result.action, {
        "selected_score": result.score,
        "score_std": float(scores.std(unbiased=False).item()),
        "score_range": float((scores.max() - scores.min()).item()),
        "first_action_score_mean": action_means,
    }


@torch.inference_mode()
def _bc_policy_action(
    world: D4LiteWorld,
    policy: CartPoleBCPolicy,
    *,
    observations: list[np.ndarray],
    led_to_actions: list[int],
    context: int,
    device: torch.device,
) -> tuple[int, list[float]]:
    frames = torch.from_numpy(np.stack(observations[-context:]))[None].to(device)
    actions = torch.tensor(
        led_to_actions[-context:],
        device=device,
        dtype=torch.long,
    )[None]
    batch = SequenceBatch(
        observations=frames,
        led_to_actions=actions,
        led_to_rewards=torch.zeros_like(actions, dtype=torch.float32),
        led_to_continues=torch.zeros_like(actions, dtype=torch.float32),
        outcome_valid=torch.zeros_like(actions, dtype=torch.bool),
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = policy(_clean_agent_tokens(world, batch))[:, -1].float()
    probabilities = logits.softmax(dim=-1)[0]
    return int(probabilities.argmax().item()), probabilities.cpu().tolist()


def _run_control_episode(
    *,
    world: D4LiteWorld,
    policy: str,
    environment_seed: int,
    policy_seed: int,
    device: torch.device,
    context: int,
    horizon: int,
    candidates: int,
    denoise_steps: int,
    discount: float,
    common_random_numbers: bool,
    selection: str,
    enumerate_all: bool,
    bc_policy: CartPoleBCPolicy | None = None,
    max_episode_steps: int | None = None,
) -> dict:
    environment = CartPolePixels(
        image_size=world.cfg.image_size, max_episode_steps=max_episode_steps
    )
    observation = environment.reset(seed=environment_seed)
    observations = [observation]
    led_to_actions = [-1]
    policy_rng = np.random.default_rng(policy_seed)
    planner_rng = torch.Generator(device=device).manual_seed(policy_seed)
    schedule = shortcut_schedule(world.cfg.k_max, denoise_steps)
    actions: list[int] = []
    rewards: list[float] = []
    score_ranges: list[float] = []
    action_score_margins: list[float] = []
    policy_confidences: list[float] = []
    terminated = truncated = False
    started = time.perf_counter()
    try:
        while not (terminated or truncated):
            if policy == "random":
                action = int(policy_rng.integers(2))
            elif policy == "planner":
                action, planner = _planner_action(
                    world,
                    observations=observations,
                    led_to_actions=led_to_actions,
                    context=min(context, len(observations)),
                    horizon=horizon,
                    candidates=candidates,
                    schedule=schedule,
                    discount=discount,
                    generator=planner_rng,
                    device=device,
                    common_random_numbers=common_random_numbers,
                    selection=selection,
                    enumerate_all=enumerate_all,
                )
                score_ranges.append(planner["score_range"])
                means = planner["first_action_score_mean"]
                action_score_margins.append(abs(means["1"] - means["0"]))
            elif policy == "bc_policy":
                if bc_policy is None:
                    raise RuntimeError("bc_policy module is required")
                action, probabilities = _bc_policy_action(
                    world,
                    bc_policy,
                    observations=observations,
                    led_to_actions=led_to_actions,
                    context=min(context, len(observations)),
                    device=device,
                )
                policy_confidences.append(max(probabilities))
            elif policy == "oracle_reference":
                assert environment.state is not None
                action = int(
                    float(environment.state[2])
                    + 0.5 * float(environment.state[3])
                    > 0.0
                )
            else:
                raise ValueError(f"unsupported policy {policy!r}")
            observation, reward, _, terminated, truncated = environment.step(action)
            observations.append(observation)
            led_to_actions.append(action)
            actions.append(action)
            rewards.append(reward)
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    return {
        "policy": policy,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "return": float(sum(rewards)),
        "decision_length": len(actions),
        "physical_steps": int(round(sum(rewards))),
        "terminated": terminated,
        "truncated": truncated,
        "action_one_fraction": float(np.mean(actions)) if actions else None,
        "mean_candidate_score_range": (
            float(np.mean(score_ranges)) if score_ranges else None
        ),
        "mean_first_action_score_margin": (
            float(np.mean(action_score_margins))
            if action_score_margins
            else None
        ),
        "mean_policy_confidence": (
            float(np.mean(policy_confidences))
            if policy_confidences
            else None
        ),
        "wall_seconds": elapsed,
    }


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


def evaluate_executed_control(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    device: torch.device,
    seeds: list[int],
    context: int,
    horizon: int,
    candidates: int,
    denoise_steps: int,
    discount: float,
    policy_seed_base: int,
    common_random_numbers: bool,
    selection: str,
    enumerate_all: bool,
) -> dict:
    """Run matched random/planner episodes and decide the working-baseline gate."""
    world, _, checkpoint_payload = load_checkpoint(
        checkpoint,
        device=device,
        expected_sha256=checkpoint_sha256,
        strict_implementation=False,
    )
    world.eval()
    if world.cfg.arm_id not in {"T-BASE", "T-JEPA", "M-JEPA"} or world.cfg.n_actions != 2:
        raise RuntimeError("checkpoint is not a registered CartPole control arm")
    rows = []
    for policy_index, policy in enumerate(
        ("random", "planner", "oracle_reference")
    ):
        for index, environment_seed in enumerate(seeds):
            row = _run_control_episode(
                world=world,
                policy=policy,
                environment_seed=environment_seed,
                policy_seed=(
                    policy_seed_base
                    + 1_000_000 * policy_index
                    + environment_seed
                ),
                device=device,
                context=context,
                horizon=horizon,
                candidates=candidates,
                denoise_steps=denoise_steps,
                discount=discount,
                common_random_numbers=common_random_numbers,
                selection=selection,
                enumerate_all=enumerate_all,
            )
            rows.append(row)
            print(
                f"{policy} {index + 1}/{len(seeds)} "
                f"seed={environment_seed} return={row['return']:.0f}",
                flush=True,
            )
            _atomic_json(
                output.with_name(f".{output.name}.progress"),
                {"status": "running", "rows": rows},
            )

    by_policy = {
        policy: [
            row for row in rows if row["policy"] == policy
        ]
        for policy in ("random", "planner", "oracle_reference")
    }
    random_by_seed = {
        row["environment_seed"]: row for row in by_policy["random"]
    }
    planner_by_seed = {
        row["environment_seed"]: row for row in by_policy["planner"]
    }
    differences = [
        planner_by_seed[seed]["return"] - random_by_seed[seed]["return"]
        for seed in seeds
    ]
    ci = paired_bootstrap_interval(
        differences, seed=policy_seed_base + 9_000_000
    )
    summaries = {
        policy: {
            "episodes": len(policy_rows),
            "mean_return": float(
                np.mean([row["return"] for row in policy_rows])
            ),
            "median_return": float(
                np.median([row["return"] for row in policy_rows])
            ),
            "minimum_return": float(
                np.min([row["return"] for row in policy_rows])
            ),
            "maximum_return": float(
                np.max([row["return"] for row in policy_rows])
            ),
            "total_wall_seconds": float(
                sum(row["wall_seconds"] for row in policy_rows)
            ),
        }
        for policy, policy_rows in by_policy.items()
    }
    planner_mean = summaries["planner"]["mean_return"]
    random_mean = summaries["random"]["mean_return"]
    gate = {
        "paired_mean_delta": float(np.mean(differences)),
        "paired_bootstrap_95_ci": ci,
        "paired_wins": int(sum(value > 0 for value in differences)),
        "paired_ties": int(sum(value == 0 for value in differences)),
        "paired_losses": int(sum(value < 0 for value in differences)),
        "planner_mean_at_least_50": planner_mean >= 50.0,
        "planner_at_least_1_5x_random": planner_mean >= 1.5 * random_mean,
        "paired_ci_excludes_zero": ci[0] > 0.0,
        "paired_win_rate_at_least_60_percent": (
            sum(value > 0 for value in differences) / len(differences) >= 0.6
        ),
    }
    gate["working_baseline"] = all(
        gate[name]
        for name in (
            "planner_mean_at_least_50",
            "planner_at_least_1_5x_random",
            "paired_ci_excludes_zero",
            "paired_win_rate_at_least_60_percent",
        )
    )
    payload = {
        "format": FORMAT,
        "status": "completed",
        "result": (
            "WORKING_BASELINE"
            if gate["working_baseline"]
            else "BASELINE_NOT_YET_WORKING"
        ),
        "claim_boundary": (
            "frozen world-model planning on fresh official CartPole-v1 seeds; "
            "the noisy-balance collector and oracle reference use state, but "
            "the evaluated planner sees only rendered pixels and past actions"
        ),
        "protocol": {
            "seeds": seeds,
            "fresh_from_replay": True,
            "policies": [
                "uniform_random",
                "pixel_world_model_random_shooting",
                "privileged_state_oracle_reference",
            ],
            "context": context,
            "horizon": horizon,
            "candidates": candidates,
            "denoise_steps": denoise_steps,
            "discount": discount,
            "execute_first_action_only": True,
            "environment_action_repeat": ACTION_REPEAT,
            "planner_common_random_numbers": common_random_numbers,
            "planner_selection": selection,
            "planner_enumerates_all_action_sequences": enumerate_all,
            "learning_during_evaluation": False,
            "paired_bootstrap_draws": 20_000,
            "predeclared_working_gate": {
                "planner_mean_return": ">= 50",
                "planner_over_random": ">= 1.5x",
                "paired_delta_95_ci": "lower bound > 0",
                "paired_win_rate": ">= 60%",
            },
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_step": checkpoint_payload["step"],
            "checkpoint_implementation_sha256": checkpoint_payload[
                "provenance"
            ]["implementation_sha256"],
            "evaluation_implementation_sha256": implementation_sha256(),
            "sources": source_report(),
        },
        "rows": rows,
        "summary": summaries,
        "paired_differences": differences,
        "gate": gate,
    }
    _atomic_json(output, payload)
    output.with_name(f".{output.name}.progress").unlink(missing_ok=True)
    return payload


def evaluate_bc_control(
    *,
    world_checkpoint: Path,
    world_checkpoint_sha256: str,
    policy_checkpoint: Path,
    policy_checkpoint_sha256: str,
    output: Path,
    device: torch.device,
    seeds: list[int],
    context: int,
    policy_seed_base: int,
) -> dict:
    """Evaluate the frozen source-shaped BC head against matched random control."""
    world, _, world_payload = load_checkpoint(
        world_checkpoint,
        device=device,
        expected_sha256=world_checkpoint_sha256,
        strict_implementation=False,
    )
    world.eval()
    policy, policy_payload = load_bc_policy(
        policy_checkpoint,
        expected_sha256=policy_checkpoint_sha256,
        expected_world_sha256=world_checkpoint_sha256,
        device=device,
    )
    rows = []
    for policy_index, policy_name in enumerate(
        ("random", "bc_policy", "oracle_reference")
    ):
        for index, environment_seed in enumerate(seeds):
            row = _run_control_episode(
                world=world,
                policy=policy_name,
                environment_seed=environment_seed,
                policy_seed=(
                    policy_seed_base
                    + 1_000_000 * policy_index
                    + environment_seed
                ),
                device=device,
                context=context,
                horizon=1,
                candidates=2,
                denoise_steps=1,
                discount=1.0,
                common_random_numbers=False,
                selection="best_plan",
                enumerate_all=False,
                bc_policy=policy,
            )
            rows.append(row)
            print(
                f"{policy_name} {index + 1}/{len(seeds)} "
                f"seed={environment_seed} return={row['return']:.0f}",
                flush=True,
            )
            _atomic_json(
                output.with_name(f".{output.name}.progress"),
                {"status": "running", "rows": rows},
            )
    by_policy = {
        name: [row for row in rows if row["policy"] == name]
        for name in ("random", "bc_policy", "oracle_reference")
    }
    random_by_seed = {
        row["environment_seed"]: row for row in by_policy["random"]
    }
    bc_by_seed = {
        row["environment_seed"]: row for row in by_policy["bc_policy"]
    }
    differences = [
        bc_by_seed[seed]["return"] - random_by_seed[seed]["return"]
        for seed in seeds
    ]
    ci = paired_bootstrap_interval(
        differences, seed=policy_seed_base + 8_000_000
    )
    summaries = {
        name: {
            "episodes": len(policy_rows),
            "mean_return": float(
                np.mean([row["return"] for row in policy_rows])
            ),
            "median_return": float(
                np.median([row["return"] for row in policy_rows])
            ),
            "minimum_return": float(
                np.min([row["return"] for row in policy_rows])
            ),
            "maximum_return": float(
                np.max([row["return"] for row in policy_rows])
            ),
            "total_wall_seconds": float(
                sum(row["wall_seconds"] for row in policy_rows)
            ),
        }
        for name, policy_rows in by_policy.items()
    }
    bc_mean = summaries["bc_policy"]["mean_return"]
    random_mean = summaries["random"]["mean_return"]
    wins = sum(value > 0 for value in differences)
    gate = {
        "paired_mean_delta": float(np.mean(differences)),
        "paired_bootstrap_95_ci": ci,
        "paired_wins": wins,
        "paired_ties": int(sum(value == 0 for value in differences)),
        "paired_losses": int(sum(value < 0 for value in differences)),
        "bc_mean_at_least_100": bc_mean >= 100.0,
        "bc_at_least_2x_random": bc_mean >= 2.0 * random_mean,
        "paired_ci_excludes_zero": ci[0] > 0.0,
        "paired_win_rate_at_least_80_percent": wins / len(seeds) >= 0.8,
    }
    gate["working_baseline"] = all(
        gate[name]
        for name in (
            "bc_mean_at_least_100",
            "bc_at_least_2x_random",
            "paired_ci_excludes_zero",
            "paired_win_rate_at_least_80_percent",
        )
    )
    payload = {
        "format": FORMAT,
        "status": "completed",
        "result": (
            "WORKING_BASELINE"
            if gate["working_baseline"]
            else "BASELINE_NOT_YET_WORKING"
        ),
        "claim_boundary": (
            "frozen pixel policy head trained on offline demonstration actions; "
            "no simulator state, learning, Mamba, or CDP at evaluation"
        ),
        "protocol": {
            "seeds": seeds,
            "fresh_from_replay": True,
            "policies": [
                "uniform_random",
                "frozen_pixel_bc_policy",
                "privileged_state_oracle_reference",
            ],
            "context": context,
            "environment_action_repeat": ACTION_REPEAT,
            "learning_during_evaluation": False,
            "predeclared_working_gate": {
                "bc_mean_return": ">= 100",
                "bc_over_random": ">= 2x",
                "paired_delta_95_ci": "lower bound > 0",
                "paired_win_rate": ">= 80%",
            },
        },
        "provenance": {
            "world_checkpoint": str(world_checkpoint),
            "world_checkpoint_sha256": file_sha256(world_checkpoint),
            "world_checkpoint_step": world_payload["step"],
            "policy_checkpoint": str(policy_checkpoint),
            "policy_checkpoint_sha256": file_sha256(policy_checkpoint),
            "policy_world_checkpoint_sha256": policy_payload[
                "world_checkpoint_sha256"
            ],
            "evaluation_implementation_sha256": implementation_sha256(),
            "sources": source_report(),
        },
        "rows": rows,
        "summary": summaries,
        "paired_differences": differences,
        "gate": gate,
    }
    _atomic_json(output, payload)
    output.with_name(f".{output.name}.progress").unlink(missing_ok=True)
    return payload


def _parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start_text, stop_text = text.split(":", 1)
        return list(range(int(start_text), int(stop_text)))
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--path", type=Path, required=True)
    collect.add_argument("--random-episodes", type=int, default=60)
    collect.add_argument("--noisy-balance-episodes", type=int, default=60)
    collect.add_argument("--seed-base", type=int, required=True)
    collect.add_argument("--epsilon", type=float, default=0.30)

    train = subparsers.add_parser("train")
    train.add_argument("--train-replay", type=Path, required=True)
    train.add_argument("--dev-replay", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--tokenizer-steps", type=int, default=2_000)
    train.add_argument("--world-steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--terminal-fraction", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=20260720)
    train.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    train_jepa = subparsers.add_parser("train-jepa")
    train_jepa.add_argument("--train-replay", type=Path, required=True)
    train_jepa.add_argument("--dev-replay", type=Path, required=True)
    train_jepa.add_argument("--output-dir", type=Path, required=True)
    train_jepa.add_argument("--world-steps", type=int, default=20_000)
    train_jepa.add_argument("--batch-size", type=int, default=8)
    train_jepa.add_argument("--learning-rate", type=float, default=3e-4)
    train_jepa.add_argument("--terminal-fraction", type=float, default=0.25)
    train_jepa.add_argument("--seed", type=int, default=20260720)
    train_jepa.add_argument(
        "--anticollapse", choices=("ema", "sigreg"), default="ema"
    )
    train_jepa.add_argument("--sigreg-lambda", type=float, default=None)
    train_jepa.add_argument(
        "--temporal-backend",
        choices=("transformer", "mamba2"),
        default="transformer",
        help="transformer = T-JEPA; mamba2 = M-JEPA (D022 d_state=64, headdim=64)",
    )
    train_jepa.add_argument(
        "--jepa-jumps", type=int, default=None,
        help="SPR multi-step rollout length (D034/D041); omit to keep the "
             "configured default of 5",
    )
    train_jepa.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    train_policy = subparsers.add_parser("train-policy")
    train_policy.add_argument("--world-checkpoint", type=Path, required=True)
    train_policy.add_argument("--world-checkpoint-sha256", required=True)
    train_policy.add_argument("--train-replay", type=Path, required=True)
    train_policy.add_argument("--dev-replay", type=Path, required=True)
    train_policy.add_argument("--output", type=Path, required=True)
    train_policy.add_argument("--steps", type=int, default=3_000)
    train_policy.add_argument("--batch-size", type=int, default=16)
    train_policy.add_argument("--learning-rate", type=float, default=3e-4)
    train_policy.add_argument("--seed", type=int, default=20260723)
    train_policy.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seeds", default="900000:900030")
    evaluate.add_argument("--context", type=int, default=4)
    evaluate.add_argument("--horizon", type=int, default=8)
    evaluate.add_argument("--candidates", type=int, default=64)
    evaluate.add_argument("--denoise-steps", type=int, default=4)
    evaluate.add_argument("--discount", type=float, default=0.99)
    evaluate.add_argument(
        "--selection",
        choices=("best_plan", "first_action_mean"),
        default="best_plan",
    )
    evaluate.add_argument(
        "--common-random-numbers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    evaluate.add_argument(
        "--enumerate-all",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    evaluate.add_argument("--policy-seed-base", type=int, default=20260720)
    evaluate.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluate_policy = subparsers.add_parser("evaluate-policy")
    evaluate_policy.add_argument(
        "--world-checkpoint", type=Path, required=True
    )
    evaluate_policy.add_argument("--world-checkpoint-sha256", required=True)
    evaluate_policy.add_argument(
        "--policy-checkpoint", type=Path, required=True
    )
    evaluate_policy.add_argument("--policy-checkpoint-sha256", required=True)
    evaluate_policy.add_argument("--output", type=Path, required=True)
    evaluate_policy.add_argument("--seeds", default="950000:950030")
    evaluate_policy.add_argument("--context", type=int, default=8)
    evaluate_policy.add_argument(
        "--policy-seed-base", type=int, default=20260723
    )
    evaluate_policy.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    if args.command == "collect":
        report = collect_dataset(
            path=args.path,
            random_episodes=args.random_episodes,
            noisy_balance_episodes=args.noisy_balance_episodes,
            seed_base=args.seed_base,
            epsilon=args.epsilon,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "train":
        report = train_baseline(
            train_replay_path=args.train_replay,
            dev_replay_path=args.dev_replay,
            output_dir=args.output_dir,
            device=torch.device(args.device),
            tokenizer_steps=args.tokenizer_steps,
            world_steps=args.world_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            terminal_fraction=args.terminal_fraction,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "train-jepa":
        report = train_jepa_world(
            train_replay_path=args.train_replay,
            dev_replay_path=args.dev_replay,
            output_dir=args.output_dir,
            device=torch.device(args.device),
            world_steps=args.world_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            terminal_fraction=args.terminal_fraction,
            seed=args.seed,
            anticollapse=args.anticollapse,
            sigreg_lambda=args.sigreg_lambda,
            temporal_backend=args.temporal_backend,
            jepa_jumps=args.jepa_jumps,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "train-policy":
        report = train_bc_policy(
            world_checkpoint=args.world_checkpoint,
            world_checkpoint_sha256=args.world_checkpoint_sha256,
            train_replay_path=args.train_replay,
            dev_replay_path=args.dev_replay,
            output=args.output,
            device=torch.device(args.device),
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    elif args.command == "evaluate":
        report = evaluate_executed_control(
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            output=args.output,
            device=torch.device(args.device),
            seeds=_parse_seeds(args.seeds),
            context=args.context,
            horizon=args.horizon,
            candidates=args.candidates,
            denoise_steps=args.denoise_steps,
            discount=args.discount,
            policy_seed_base=args.policy_seed_base,
            common_random_numbers=args.common_random_numbers,
            selection=args.selection,
            enumerate_all=args.enumerate_all,
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(json.dumps(report["gate"], indent=2, sort_keys=True))
    elif args.command == "evaluate-policy":
        report = evaluate_bc_control(
            world_checkpoint=args.world_checkpoint,
            world_checkpoint_sha256=args.world_checkpoint_sha256,
            policy_checkpoint=args.policy_checkpoint,
            policy_checkpoint_sha256=args.policy_checkpoint_sha256,
            output=args.output,
            device=torch.device(args.device),
            seeds=_parse_seeds(args.seeds),
            context=args.context,
            policy_seed_base=args.policy_seed_base,
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(json.dumps(report["gate"], indent=2, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
