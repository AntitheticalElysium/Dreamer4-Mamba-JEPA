"""Craftax-wired world / BC / imagination runners + an end-to-end preflight.

The numerical core is unchanged and env-agnostic (``world_loss``,
``imagine_trajectory``, ``actor_critic_update``, ``ReplayContextSampler``,
``CartPoleValueHead``, ``CartPoleBCPolicy``). What was CartPole-specific was only
the config factory (``cartpole_jepa_config``, n_actions=2), the replay loaders
(``load_cartpole_replay``/``sample_cartpole_sequences``) and a hard
``n_actions == 2`` assertion in ``train_imagination_actor_critic``. This module
provides the 17-action Craftax equivalents so the FULL architecture -- not just
the model/data primitives -- runs on Craftax.

``craftax_preflight`` chains world -> BC -> imagination at a tiny budget on a
real Craftax replay and asserts every phase runs (finite losses, gradients flow,
no CartPole gate). It is the executable "the architecture runs on Craftax" gate,
the Craftax-native replacement for the legacy danijar/m3 preflight CLI that was
removed from ``world_eval`` (formerly ``crafter_preflight``).

Imports torch (no JAX): consumes a hash-pinned Craftax replay produced offline by
``craftax_data``; it does not touch the live JAX environment.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .cartpole_baseline import (
    CartPoleBCPolicy,
    POLICY_FORMAT,
    _atomic_torch_save,
    _clean_agent_tokens,
    load_bc_policy,
    sample_cartpole_sequences,
)
from .checkpoint import file_sha256, load_checkpoint, save_checkpoint
from .config import D4LiteConfig
from .data import EpisodeReplay, load_episode_replay, replay_sample_to_sequence
from .imagination_actor_critic import (
    CartPoleValueHead,
    ReplayContextSampler,
    actor_critic_update,
    freeze_module,
    unfreeze_module,
)
from .model import D4LiteWorld
from .training import WorldLossNormalizer, world_loss


def craftax_jepa_config(temporal_backend: str = "transformer") -> D4LiteConfig:
    """17-action, 64x64 Craftax JEPA world config.

    For the Mamba arm the D022 state expansion (d_state=64, headdim=64) is set
    explicitly; the ``D4LiteConfig`` defaults (16/32) are the rejected D021.
    """
    overrides = dict(
        representation_objective="jepa",
        n_actions=17,
        image_size=64,
        temporal_backend=temporal_backend,
    )
    if temporal_backend == "mamba2":
        overrides.update(mamba_d_state=64, mamba_headdim=64)
    return D4LiteConfig(**overrides)


def _sample_sequences(replay, batch_size, sequence_length, device, rng):
    sample = replay.sample(batch_size, sequence_length, device, rng=rng)
    return replay_sample_to_sequence(sample)


def _write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_policy_checkpoint(
    path: str | Path, policy: CartPoleBCPolicy, *, world_checkpoint_sha256: str
) -> str:
    """Serialize a policy head in ``POLICY_FORMAT`` (loadable by ``load_bc_policy``)."""
    payload = {
        "format": POLICY_FORMAT,
        "world_checkpoint_sha256": world_checkpoint_sha256,
        "config": {
            "d_model": policy.d_model,
            "n_actions": policy.n_actions,
        },
        "policy": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
    }
    return _atomic_torch_save(Path(path), payload)


def train_craftax_jepa_world(
    *,
    replay: EpisodeReplay,
    cfg: D4LiteConfig,
    world_steps: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    seed: int = 0,
    device: torch.device,
    warmup: int = 1_000,
    terminal_fraction: float | None = None,
    output_dir: str | Path | None = None,
) -> tuple[D4LiteWorld, WorldLossNormalizer, list[dict]]:
    """Joint-phase JEPA world training on a Craftax replay (reuses world_loss).

    Terminal windows are oversampled (``terminal_fraction``, default
    ``cfg.jepa_terminal_fraction``) so the continuation head sees enough episode
    ends (D035); the replay must therefore contain terminal (death/lava, not
    timeout) episodes.
    """
    if cfg.representation_objective != "jepa":
        raise RuntimeError("Craftax world runner requires the jepa objective")
    if cfg.n_actions != 17:
        raise RuntimeError("Craftax world config must have n_actions=17")
    fraction = cfg.jepa_terminal_fraction if terminal_fraction is None else terminal_fraction
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    world = D4LiteWorld(cfg).to(device).train()
    normalizer = WorldLossNormalizer().to(device)
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-2)
    rng = np.random.default_rng(seed + 3)
    history: list[dict] = []
    for step in range(world_steps):
        batch = sample_cartpole_sequences(
            replay, batch_size=batch_size, sequence_length=cfg.sequence_length,
            terminal_fraction=fraction, device=device, rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, metrics = world_loss(world, batch, normalizer=normalizer)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite JEPA world loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite gradient at step {step}")
        if step < warmup:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * float(step + 1) / warmup
        optimizer.step()
        if cfg.jepa_anticollapse == "ema":
            frac = step / max(1, world_steps - 1)
            tau = cfg.jepa_ema_tau + (cfg.jepa_ema_tau_final - cfg.jepa_ema_tau) * frac
            world.update_jepa_target(tau)
        history.append({
            "jepa": float(metrics["loss/jepa"].item()),
            "cosine": float(metrics["jepa/jepa_cosine"].item()),
            "online_std": float(metrics["jepa/jepa_online_std"].item()),
        })
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        world_sha = save_checkpoint(
            out / "world.pt", world=world, normalizer=normalizer,
            optimizer=optimizer, numpy_rng=rng, step=world_steps,
            extra={"format": "craftax_jepa_world_v1", "seed": seed,
                   "terminal_fraction": fraction},
        )
        _write_report(out / "world_report.json", {
            "world_checkpoint": "world.pt", "world_checkpoint_sha256": world_sha,
            "arm_id": cfg.arm_id, "config": cfg.to_dict(),
            "final": history[-1] if history else None,
        })
    return world, normalizer, history


def train_craftax_bc(
    *,
    world: D4LiteWorld,
    replay: EpisodeReplay,
    steps: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    seed: int = 0,
    device: torch.device,
    warmup: int = 250,
    output_dir: str | Path | None = None,
    world_checkpoint_sha256: str = "in_memory",
) -> tuple[CartPoleBCPolicy, list[float]]:
    """Train the gradient-isolated BC policy head on Craftax demonstration actions.

    When ``output_dir`` is given, ``bc.pt`` is saved paired to
    ``world_checkpoint_sha256`` (pass the sha from the saved world so
    ``load_bc_policy`` accepts the pairing).
    """
    freeze_module(world)
    world.eval()
    policy = CartPoleBCPolicy(
        d_model=world.cfg.dynamics_d_model, n_actions=world.cfg.n_actions
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for step in range(steps):
        batch = _sample_sequences(replay, batch_size, world.cfg.sequence_length, device, rng)
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda"):
            agent = _clean_agent_tokens(world, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = policy(agent.detach())
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, world.cfg.n_actions),
                batch.led_to_actions[:, 1:].reshape(-1),
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite BC loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if step < warmup:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * float(step + 1) / warmup
        optimizer.step()
        losses.append(float(loss.detach().item()))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bc_sha = save_policy_checkpoint(
            out / "bc.pt", policy, world_checkpoint_sha256=world_checkpoint_sha256
        )
        _write_report(out / "bc_report.json", {
            "bc_checkpoint": "bc.pt", "bc_checkpoint_sha256": bc_sha,
            "world_checkpoint_sha256": world_checkpoint_sha256,
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
        })
    return policy, losses


def train_craftax_imagination(
    *,
    world: D4LiteWorld,
    bc: CartPoleBCPolicy,
    replay: EpisodeReplay,
    steps: int,
    batch_size: int,
    context: int,
    horizon: int,
    learning_rate: float = 1e-4,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    alpha: float = 0.5,
    beta: float = 0.3,
    gradient_clip: float = 1.0,
    seed: int = 0,
    device: torch.device,
    output_dir: str | Path | None = None,
    world_checkpoint_sha256: str = "in_memory",
) -> tuple[CartPoleBCPolicy, CartPoleValueHead, list[dict]]:
    """Dreamer-4 actor/value imagination on a frozen Craftax world (reuses the
    exact PMPO/TD-lambda core; only the CartPole n_actions==2 gate is dropped)."""
    freeze_module(world)
    actor = unfreeze_module(copy.deepcopy(bc).to(device))
    prior = freeze_module(copy.deepcopy(bc).to(device))
    value = CartPoleValueHead(
        d_model=world.cfg.dynamics_d_model, num_bins=world.cfg.reward_bins,
        log_low=world.cfg.reward_log_low, log_high=world.cfg.reward_log_high,
    ).to(device)
    actor.train()
    value.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()), lr=learning_rate
    )
    sampler = ReplayContextSampler(replay, context=context, device=device, seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    history: list[dict] = []
    for _ in range(steps):
        context_batch = sampler.sample(batch_size)
        metrics = actor_critic_update(
            world=world, actor=actor, prior=prior, value=value,
            optimizer=optimizer, context_batch=context_batch,
            horizon=horizon, denoise_steps=world.cfg.k_max, context=context,
            gamma=gamma, lambda_=lambda_, alpha=alpha, beta=beta,
            gradient_clip=gradient_clip, generator=generator, device=device,
        )
        history.append(metrics)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        actor_sha = save_policy_checkpoint(
            out / "actor.pt", actor, world_checkpoint_sha256=world_checkpoint_sha256
        )
        torch.save(
            {"value": {k: v.detach().cpu() for k, v in value.state_dict().items()}},
            out / "value.pt",
        )
        _write_report(out / "imagination_report.json", {
            "actor_checkpoint": "actor.pt", "actor_checkpoint_sha256": actor_sha,
            "value_checkpoint": "value.pt",
            "world_checkpoint_sha256": world_checkpoint_sha256,
            "final": history[-1] if history else None,
        })
    return actor, value, history


def craftax_preflight(
    *,
    replay_path: str | Path,
    replay_sha256: str,
    device: torch.device,
    temporal_backend: str = "transformer",
    world_steps: int = 5,
    bc_steps: int = 5,
    imagination_steps: int = 2,
    batch_size: int = 4,
    context: int = 8,
    horizon: int = 4,
    seed: int = 0,
) -> dict:
    """Run world -> BC -> imagination end-to-end on Craftax at a tiny budget.

    Proves the full architecture executes on 17-action, 64x64 Craftax data:
    every phase produces finite losses and the imagination update runs the exact
    PMPO/TD-lambda core with a 17-action world. Not a training run -- a gate.
    """
    replay = load_episode_replay(
        replay_path, expected_sha256=replay_sha256, capacity_steps=10 ** 8
    )
    cfg = craftax_jepa_config(temporal_backend)
    world, _, world_history = train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=world_steps, batch_size=batch_size,
        seed=seed, device=device, warmup=max(1, world_steps),
    )
    bc, bc_losses = train_craftax_bc(
        world=world, replay=replay, steps=bc_steps, batch_size=batch_size,
        seed=seed + 1, device=device, warmup=max(1, bc_steps),
    )
    actor, value, imag_history = train_craftax_imagination(
        world=world, bc=bc, replay=replay, steps=imagination_steps,
        batch_size=batch_size, context=context, horizon=horizon,
        seed=seed + 2, device=device,
    )
    return {
        "status": "ran",
        "arm_id": cfg.arm_id,
        "n_actions": cfg.n_actions,
        "image_size": cfg.image_size,
        "world_final_jepa": world_history[-1]["jepa"],
        "world_final_cosine": world_history[-1]["cosine"],
        "world_online_std": world_history[-1]["online_std"],
        "bc_first_loss": bc_losses[0],
        "bc_last_loss": bc_losses[-1],
        "imagination_total_loss": imag_history[-1]["total_loss"],
        "imagination_mean_return": imag_history[-1]["mean_return"],
        "imagination_entropy": imag_history[-1]["entropy"],
        "all_phases_finite": bool(
            np.isfinite(world_history[-1]["jepa"])
            and np.isfinite(bc_losses[-1])
            and np.isfinite(imag_history[-1]["total_loss"])
        ),
    }


__all__ = [
    "craftax_jepa_config",
    "save_policy_checkpoint",
    "train_craftax_jepa_world",
    "train_craftax_bc",
    "train_craftax_imagination",
    "craftax_preflight",
]
