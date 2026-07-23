"""Dreamer-4-style actor/value learning inside the frozen D4-lite world.

Primary algorithm references:

* Dreamer 4, arXiv:2509.24527v1, Section 3.3 and equations 10-11.
* ``edwhu/dreamer4-jax`` commit 8144b940, ``scripts/train_policy.py``.
* DreamerV3, arXiv:2301.04104v2, equations 4-7.
* ``danijar/dreamerv3`` commit e3f02248, ``dreamerv3/agent.py``.

The implementation deliberately does not train or call a shooting planner.
The tokenizer, world, reward/continuation heads, and behavioral prior are
frozen. Only the actor copied from BC and a zero-output categorical value head
are optimized.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .cartpole_baseline import (
    ACTION_REPEAT,
    CartPoleBCPolicy,
    _atomic_json,
    _atomic_torch_save,
    _episode_window,
    _run_control_episode,
    load_bc_policy,
    load_cartpole_replay,
    paired_bootstrap_interval,
)
from .checkpoint import file_sha256, implementation_sha256, load_checkpoint
from .data import EpisodeReplay, SequenceBatch, replay_sample_to_sequence
from .model import D4LiteWorld
from .rollout import sample_next_packed, shortcut_schedule
from .source import (
    REPO_ROOT,
    SourceIdentity,
    load_mmbench2_model,
    source_report,
    verify_source,
)


FORMAT = "d4_lite_cartpole_imagination_actor_critic_v1"
EVALUATION_FORMAT = "d4_lite_cartpole_actor_parity_v1"

DREAMER4_PAPER = SourceIdentity(
    name="Dreamer 4 paper:arXiv:2509.24527v1",
    path=REPO_ROOT / "third_party/papers/2509.24527v1.pdf",
    commit="arXiv:2509.24527v1",
    sha256="8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb",
    license="paper",
)
EDWHU_POLICY = SourceIdentity(
    name="edwhu/dreamer4-jax:scripts/train_policy.py",
    path=REPO_ROOT
    / "third_party/sources/edwhu__dreamer4-jax/scripts/train_policy.py",
    commit="8144b940d801971f12ec5633553b95001e555949",
    sha256="d16d9e6ba220664afbb73e7f4f80056371dd6fffb3c592d2d09a7ef2b840d7d1",
    license="no license file in inspected checkout; read-only reference",
)
EDWHU_IMAGINATION = SourceIdentity(
    name="edwhu/dreamer4-jax:dreamer/imagination.py",
    path=REPO_ROOT
    / "third_party/sources/edwhu__dreamer4-jax/dreamer/imagination.py",
    commit="8144b940d801971f12ec5633553b95001e555949",
    sha256="562bab8c4bd5d465c8661022cefdeca37cce419b52e16c5d63db8ddca0b4d4ac",
    license="no license file in inspected checkout; read-only reference",
)
DREAMERV3_AGENT = SourceIdentity(
    name="danijar/dreamerv3:dreamerv3/agent.py",
    path=REPO_ROOT
    / "third_party/sources/danijar__dreamerv3/dreamerv3/agent.py",
    commit="e3f02248693a79dc8b0ebd62c93683888ddaccfe",
    sha256="adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269",
    license="MIT",
)
DREAMERV3_CONFIG = SourceIdentity(
    name="danijar/dreamerv3:dreamerv3/configs.yaml",
    path=REPO_ROOT
    / "third_party/sources/danijar__dreamerv3/dreamerv3/configs.yaml",
    commit="e3f02248693a79dc8b0ebd62c93683888ddaccfe",
    sha256="9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc",
    license="MIT",
)


def actor_source_report() -> dict[str, dict[str, str]]:
    """Verify the actor/value references without changing old world provenance."""
    identities = (
        DREAMER4_PAPER,
        EDWHU_POLICY,
        EDWHU_IMAGINATION,
        DREAMERV3_AGENT,
        DREAMERV3_CONFIG,
    )
    return {
        identity.name: {
            "path": str(identity.path),
            "commit": identity.commit,
            "sha256": verify_source(identity),
            "license": identity.license,
        }
        for identity in identities
    }


def module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, types, shapes, and exact bytes deterministically."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        # reshape(-1) makes 0-dim tensors (e.g. BatchNorm num_batches_tracked)
        # viewable as bytes; it is a no-op on the byte content otherwise.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def state_dict_l2_distance(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> float:
    if set(first) != set(second):
        raise ValueError("state dictionaries have different keys")
    squared = 0.0
    for name in first:
        left = first[name].detach().float().cpu()
        right = second[name].detach().float().cpu()
        squared += float((left - right).square().sum().item())
    return math.sqrt(squared)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
    }


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    module.train()
    for parameter in module.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    return module


class CartPoleValueHead(nn.Module):
    """Dreamer categorical value distribution over current agent tokens.

    The attention pooling and MLP shape match the source-shaped local policy
    and MMBench2 task heads. DreamerV3 and Dreamer 4 prescribe a categorical
    symexp-twohot critic with a zero-initialized output layer.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_bins: int = 255,
        log_low: float = -10.0,
        log_high: float = 10.0,
    ):
        super().__init__()
        upstream = load_mmbench2_model()
        self.d_model = int(d_model)
        self.num_bins = int(num_bins)
        self.log_low = float(log_low)
        self.log_high = float(log_high)
        self.pool_query = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.pool_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.projector = upstream.MLP(
            d_model=self.d_model,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        self.out = nn.Linear(self.d_model, self.num_bins)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer(
            "centers_log",
            torch.linspace(
                self.log_low,
                self.log_high,
                self.num_bins,
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def forward(
        self, agent_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if agent_tokens.ndim != 4:
            raise ValueError("agent tokens must have shape [B,T,N,D]")
        _, _, _, width = agent_tokens.shape
        key, value = self.pool_kv(agent_tokens).chunk(2, dim=-1)
        query = self.pool_query.to(dtype=key.dtype)
        scores = (key * query).sum(dim=-1) / math.sqrt(width)
        pooled = (scores.softmax(dim=-1)[..., None] * value).sum(dim=2)
        logits = self.out(self.projector(pooled))
        return logits, self.centers_log


def decode_symlog_distribution(
    logits: torch.Tensor,
    centers_log: torch.Tensor,
) -> torch.Tensor:
    upstream = load_mmbench2_model()
    expected_symlog = (
        logits.float().softmax(dim=-1)
        * centers_log.float().to(logits.device)
    ).sum(dim=-1)
    return upstream.symexp(expected_symlog)


def twohot_symlog_targets(
    values: torch.Tensor,
    centers_log: torch.Tensor,
) -> torch.Tensor:
    upstream = load_mmbench2_model()
    with torch.no_grad():
        transformed = upstream.symlog(values.float())
        return upstream.twohot_from_symlog(
            transformed,
            centers_log.float().to(values.device),
        )


def td_lambda_returns(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute returns for ``s_t,a_t,r_{t+1},c_{t+1},s_{t+1}``."""
    if rewards.ndim != 2 or continues.shape != rewards.shape:
        raise ValueError("rewards and continues must have shape [B,H]")
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("values must have shape [B,H+1]")
    next_return = values[:, -1].detach()
    outputs: list[torch.Tensor] = []
    for index in reversed(range(rewards.shape[1])):
        next_value = values[:, index + 1].detach()
        current = rewards[:, index] + (
            float(gamma)
            * continues[:, index]
            * (
                (1.0 - float(lambda_)) * next_value
                + float(lambda_) * next_return
            )
        )
        outputs.append(current)
        next_return = current
    return torch.stack(list(reversed(outputs)), dim=1)


def pmpo_loss(
    actor_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dreamer-4 equation 11 with balanced advantage-sign partitions."""
    if actor_logits.shape != prior_logits.shape:
        raise ValueError("actor and prior logits must have identical shapes")
    if actor_logits.shape[:-1] != actions.shape:
        raise ValueError("actions must match actor batch/time axes")
    if advantages.shape != actions.shape:
        raise ValueError("advantages must match actions")
    log_actor = actor_logits.float().log_softmax(dim=-1)
    log_prior = prior_logits.float().log_softmax(dim=-1)
    probabilities = log_actor.exp()
    selected = log_actor.gather(-1, actions.long()[..., None]).squeeze(-1)
    positive = advantages >= 0
    negative = ~positive

    zero = selected.sum() * 0.0
    negative_loss = (
        (1.0 - float(alpha)) * selected[negative].mean()
        if bool(negative.any())
        else zero
    )
    positive_loss = (
        -float(alpha) * selected[positive].mean()
        if bool(positive.any())
        else zero
    )
    kl_per_state = (
        probabilities * (log_actor - log_prior)
    ).sum(dim=-1)
    kl_loss = float(beta) * kl_per_state.mean()
    total = negative_loss + positive_loss + kl_loss
    entropy = -(probabilities * log_actor).sum(dim=-1).mean()
    return total, {
        "negative_loss": negative_loss,
        "positive_loss": positive_loss,
        "kl_loss": kl_loss,
        "kl_mean": kl_per_state.mean(),
        "entropy": entropy,
        "positive_count": positive.sum(),
        "negative_count": negative.sum(),
    }


@dataclass(frozen=True)
class ImaginationTrajectory:
    states: torch.Tensor  # [B,H+1,N,D]
    actions: torch.Tensor  # [B,H]
    rewards: torch.Tensor  # [B,H], reward arriving at next state
    continues: torch.Tensor  # [B,H], continuation arriving at next state


class ReplayContextSampler:
    """Episode-bounded, non-terminal context sampler without batch duplicates."""

    def __init__(
        self,
        replay: EpisodeReplay,
        *,
        context: int,
        device: torch.device,
        seed: int,
    ):
        if context < 1:
            raise ValueError("context must be positive")
        self.replay = replay
        self.context = int(context)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.windows: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(replay.episodes):
            # A valid start leaves the final context observation non-terminal.
            for start in range(max(0, len(episode.obs) - self.context)):
                self.windows.append((episode_index, start))
        if not self.windows:
            raise RuntimeError("replay contains no non-terminal context windows")

    def sample(self, batch_size: int) -> SequenceBatch:
        if batch_size < 1 or batch_size > len(self.windows):
            raise ValueError("invalid context batch size")
        selected = self.rng.choice(
            len(self.windows),
            size=batch_size,
            replace=False,
        )
        rows = []
        for window_index in selected:
            episode_index, start = self.windows[int(window_index)]
            rows.append(
                _episode_window(
                    self.replay.episodes[episode_index],
                    start=start,
                    observations=self.context,
                )
            )
        sample = {
            name: torch.from_numpy(
                np.stack([row[name] for row in rows])
            ).to(self.device)
            for name in (
                "obs",
                "actions",
                "rewards",
                "continues",
                "previous_actions",
            )
        }
        batch = replay_sample_to_sequence(sample)
        if bool(
            batch.outcome_valid[:, -1].any()
            and (batch.led_to_continues[:, -1] <= 0).any()
        ):
            raise RuntimeError("terminal state leaked into imagination starts")
        return batch


@torch.no_grad()
def imagine_trajectory(
    world: D4LiteWorld,
    actor: CartPoleBCPolicy,
    context_batch: SequenceBatch,
    *,
    horizon: int,
    denoise_steps: int,
    context: int,
    generator: torch.Generator,
    device: torch.device,
) -> ImaginationTrajectory:
    """Generate one policy rollout per replay context with a frozen world."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if context_batch.observations.shape[1] != context:
        raise ValueError("context batch length differs from registered context")
    schedule = shortcut_schedule(world.cfg.k_max, denoise_steps)
    world.eval()
    actor.eval()

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        encoded = world.encode_frames(context_batch.observations, frozen=True)
        past = encoded.packed
        led_to = context_batch.led_to_actions
        batch_size, time = past.shape[:2]
        clean_steps = torch.full(
            (batch_size, time),
            world.cfg.max_step_index,
            device=device,
            dtype=torch.long,
        )
        clean_signals = torch.full(
            (batch_size, time),
            world.cfg.k_max,
            device=device,
            dtype=torch.long,
        )
        _, context_agents = world.forward_dynamics(
            past,
            led_to,
            clean_steps,
            clean_signals,
        )

    states = [context_agents[:, -1].detach()]
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continues: list[torch.Tensor] = []
    upstream = load_mmbench2_model()

    for _ in range(horizon):
        logits = actor(states[-1][:, None].float())[:, 0].float()
        probabilities = logits.softmax(dim=-1)
        action = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        led_to_with_action = torch.cat(
            [led_to, action[:, None]],
            dim=1,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            next_latent, next_agent = sample_next_packed(
                world,
                past_packed=past,
                led_to_actions=led_to_with_action,
                schedule=schedule,
                use_cache=True,
                generator=generator,
            )
            heads = world.forward_task_heads(next_agent)
        reward_logits = heads["reward_logits"][:, 0, 0].float()
        expected_symlog = (
            reward_logits.softmax(dim=-1)
            * heads["reward_centers"].float().to(device)
        ).sum(dim=-1)
        reward = upstream.symexp(expected_symlog)
        continuation = heads["continue_logits"][:, 0, 0].float().sigmoid()

        actions.append(action.detach())
        rewards.append(reward.detach())
        continues.append(continuation.detach())
        states.append(next_agent[:, 0].detach())

        past = torch.cat([past, next_latent[:, None]], dim=1)[:, -context:]
        led_to = led_to_with_action[:, -context:]

    return ImaginationTrajectory(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
        rewards=torch.stack(rewards, dim=1),
        continues=torch.stack(continues, dim=1),
    )


def actor_critic_update(
    *,
    world: D4LiteWorld,
    actor: CartPoleBCPolicy,
    prior: CartPoleBCPolicy,
    value: CartPoleValueHead,
    optimizer: torch.optim.Optimizer,
    context_batch: SequenceBatch,
    horizon: int,
    denoise_steps: int,
    context: int,
    gamma: float,
    lambda_: float,
    alpha: float,
    beta: float,
    gradient_clip: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    trajectory = imagine_trajectory(
        world,
        actor,
        context_batch,
        horizon=horizon,
        denoise_steps=denoise_steps,
        context=context,
        generator=generator,
        device=device,
    )
    actor.train()
    value.train()
    optimizer.zero_grad(set_to_none=True)

    states = trajectory.states.float()
    value_logits, centers_log = value(states)
    values = decode_symlog_distribution(value_logits, centers_log)
    returns = td_lambda_returns(
        trajectory.rewards.float(),
        trajectory.continues.float(),
        values,
        gamma=gamma,
        lambda_=lambda_,
    ).detach()
    targets = twohot_symlog_targets(returns, centers_log)
    value_log_probabilities = value_logits[:, :-1].float().log_softmax(dim=-1)
    value_loss = -(
        targets * value_log_probabilities
    ).sum(dim=-1).mean()

    actor_logits = actor(states[:, :-1])
    with torch.no_grad():
        prior_logits = prior(states[:, :-1])
    advantages = (returns - values[:, :-1].detach()).detach()
    actor_loss, actor_metrics = pmpo_loss(
        actor_logits,
        prior_logits,
        trajectory.actions,
        advantages,
        alpha=alpha,
        beta=beta,
    )
    total_loss = actor_loss + value_loss
    if not bool(torch.isfinite(total_loss)):
        raise RuntimeError("non-finite actor/value loss")
    total_loss.backward()
    parameters = [
        parameter
        for module in (actor, value)
        for parameter in module.parameters()
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        gradient_clip,
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("non-finite actor/value gradient norm")
    optimizer.step()

    with torch.no_grad():
        value_mae = (
            values[:, :-1] - returns
        ).abs().mean()
    return {
        "total_loss": float(total_loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "pmpo_negative_loss": float(
            actor_metrics["negative_loss"].detach().item()
        ),
        "pmpo_positive_loss": float(
            actor_metrics["positive_loss"].detach().item()
        ),
        "prior_kl_loss": float(
            actor_metrics["kl_loss"].detach().item()
        ),
        "prior_kl": float(actor_metrics["kl_mean"].detach().item()),
        "entropy": float(actor_metrics["entropy"].detach().item()),
        "positive_count": int(actor_metrics["positive_count"].item()),
        "negative_count": int(actor_metrics["negative_count"].item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_return": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "mean_reward": float(trajectory.rewards.mean().item()),
        "mean_continue": float(trajectory.continues.mean().item()),
        "value_mae": float(value_mae.item()),
        "gradient_norm": float(gradient_norm.detach().item()),
    }


def _mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize empty metrics")
    return {
        key: float(np.mean([float(row[key]) for row in materialized]))
        for key in materialized[0]
    }


def train_imagination_actor_critic(
    *,
    world_checkpoint: Path,
    world_checkpoint_sha256: str,
    bc_checkpoint: Path,
    bc_checkpoint_sha256: str,
    replay_path: Path,
    output: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    context: int,
    horizon: int,
    denoise_steps: int,
    learning_rate: float,
    gamma: float,
    lambda_: float,
    alpha: float,
    beta: float,
    gradient_clip: float,
    seed: int,
) -> dict:
    """Train only actor and value heads on trajectories generated in imagination."""
    if steps < 1:
        raise ValueError("steps must be positive")
    actor_sources = actor_source_report()
    world, _, world_payload = load_checkpoint(
        world_checkpoint,
        device=device,
        expected_sha256=world_checkpoint_sha256,
        strict_implementation=False,
    )
    if world.cfg.arm_id not in {"T-BASE", "T-JEPA", "M-JEPA"} or world.cfg.n_actions != 2:
        raise RuntimeError("world is not a registered CartPole control arm")
    freeze_module(world)
    loaded_bc, bc_payload = load_bc_policy(
        bc_checkpoint,
        expected_sha256=bc_checkpoint_sha256,
        expected_world_sha256=world_checkpoint_sha256,
        device=device,
    )
    freeze_module(loaded_bc)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    actor = unfreeze_module(copy.deepcopy(loaded_bc).to(device))
    prior = freeze_module(copy.deepcopy(loaded_bc).to(device))
    value = CartPoleValueHead(
        d_model=world.cfg.dynamics_d_model,
        num_bins=world.cfg.reward_bins,
        log_low=world.cfg.reward_log_low,
        log_high=world.cfg.reward_log_high,
    ).to(device)
    actor.train()
    value.train()

    initial_actor_state = _cpu_state_dict(actor)
    initial_value_state = _cpu_state_dict(value)
    initial_actor_hash = module_state_sha256(actor)
    prior_hash = module_state_sha256(prior)
    if initial_actor_hash != prior_hash:
        raise RuntimeError("actor and BC prior did not initialize identically")
    with torch.no_grad():
        probe = torch.randn(
            2,
            3,
            world.cfg.n_agent,
            world.cfg.dynamics_d_model,
            device=device,
        )
        initial_values = decode_symlog_distribution(
            *value(probe)
        )
    if float(initial_values.abs().max().item()) > 1e-6:
        raise RuntimeError("value expectation is not numerically zero at initialization")

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    replay, replay_records = load_cartpole_replay(replay_path)
    sampler = ReplayContextSampler(
        replay,
        context=context,
        device=device,
        seed=seed + 1,
    )
    rollout_generator = torch.Generator(device=device).manual_seed(seed + 2)
    world_hash_before = module_state_sha256(world)
    prior_hash_before = module_state_sha256(prior)
    history: list[dict[str, float]] = []
    positive_total = 0
    negative_total = 0
    started = time.perf_counter()
    for step in range(steps):
        metrics = actor_critic_update(
            world=world,
            actor=actor,
            prior=prior,
            value=value,
            optimizer=optimizer,
            context_batch=sampler.sample(batch_size),
            horizon=horizon,
            denoise_steps=denoise_steps,
            context=context,
            gamma=gamma,
            lambda_=lambda_,
            alpha=alpha,
            beta=beta,
            gradient_clip=gradient_clip,
            generator=rollout_generator,
            device=device,
        )
        history.append(metrics)
        positive_total += metrics["positive_count"]
        negative_total += metrics["negative_count"]
        if (step + 1) % 25 == 0 or step == 0:
            recent = _mean_metrics(history[-25:])
            print(
                f"imagination {step + 1}/{steps}: "
                f"actor={recent['actor_loss']:.5f} "
                f"value={recent['value_loss']:.5f} "
                f"return={recent['mean_return']:.3f} "
                f"kl={recent['prior_kl']:.5f} "
                f"pos={positive_total} neg={negative_total}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    world_hash_after = module_state_sha256(world)
    prior_hash_after = module_state_sha256(prior)
    if world_hash_after != world_hash_before:
        raise RuntimeError("frozen world changed during actor/value training")
    if prior_hash_after != prior_hash_before:
        raise RuntimeError("frozen BC prior changed during actor/value training")
    actor_delta = state_dict_l2_distance(
        initial_actor_state,
        _cpu_state_dict(actor),
    )
    value_delta = state_dict_l2_distance(
        initial_value_state,
        _cpu_state_dict(value),
    )
    if actor_delta <= 0.0 or value_delta <= 0.0:
        raise RuntimeError("actor or value head did not update")
    if positive_total <= 0 or negative_total <= 0:
        raise RuntimeError("PMPO did not observe both advantage signs")

    payload = {
        "format": FORMAT,
        "world_checkpoint_sha256": world_checkpoint_sha256,
        "bc_checkpoint_sha256": bc_checkpoint_sha256,
        "actor": _cpu_state_dict(actor),
        "prior": _cpu_state_dict(prior),
        "value": _cpu_state_dict(value),
        "optimizer": optimizer.state_dict(),
        "config": {
            "d_model": world.cfg.dynamics_d_model,
            "n_actions": world.cfg.n_actions,
            "value_bins": world.cfg.reward_bins,
            "value_log_low": world.cfg.reward_log_low,
            "value_log_high": world.cfg.reward_log_high,
        },
        "algorithm": {
            "steps": steps,
            "batch_size": batch_size,
            "context": context,
            "horizon": horizon,
            "denoise_steps": denoise_steps,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "gamma": gamma,
            "lambda": lambda_,
            "pmpo_alpha": alpha,
            "reverse_prior_kl_beta": beta,
            "gradient_clip": gradient_clip,
            "one_rollout_per_context": True,
            "actor_initialized_from_bc": True,
            "world_frozen": True,
            "prior_frozen": True,
            "reward_and_continuation_frozen": True,
            "actor_value_losses_use_imagined_data_only": True,
            "shooting_planner_used": False,
            "seed": seed,
        },
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "rollout_generator": rollout_generator.get_state(),
            "numpy_context_generator": copy.deepcopy(
                sampler.rng.bit_generator.state
            ),
        },
        "metrics": {
            "first_25": _mean_metrics(history[:25]),
            "last_25": _mean_metrics(history[-25:]),
            "positive_count_total": positive_total,
            "negative_count_total": negative_total,
            "actor_l2_delta_from_bc": actor_delta,
            "value_l2_delta_from_initial": value_delta,
            "train_seconds": train_seconds,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "history": history,
        "frozen_invariants": {
            "world_tensor_sha256_before": world_hash_before,
            "world_tensor_sha256_after": world_hash_after,
            "prior_tensor_sha256_before": prior_hash_before,
            "prior_tensor_sha256_after": prior_hash_after,
            "initial_actor_tensor_sha256": initial_actor_hash,
        },
        "provenance": {
            "actor_sources": actor_sources,
            "world_sources": source_report(),
            "current_implementation_sha256": implementation_sha256(),
            "world_stored_implementation_sha256": world_payload[
                "provenance"
            ]["implementation_sha256"],
            "bc_stored_implementation_sha256": bc_payload["provenance"][
                "evaluation_implementation_sha256"
            ],
            "world_checkpoint": str(world_checkpoint),
            "bc_checkpoint": str(bc_checkpoint),
            "replay": {
                "path": str(replay_path),
                "sha256": file_sha256(replay_path),
                "episodes": len(replay_records),
                "context_windows": len(sampler.windows),
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    checkpoint_sha256 = _atomic_torch_save(output, payload)
    report = {
        key: value_
        for key, value_ in payload.items()
        if key not in {"actor", "prior", "value", "optimizer", "rng", "history"}
    }
    report["checkpoint"] = {
        "path": str(output),
        "sha256": checkpoint_sha256,
    }
    _atomic_json(output.with_suffix(".json"), report)
    return report


def load_imagination_actor_critic(
    path: Path,
    *,
    expected_sha256: str,
    expected_world_sha256: str,
    expected_bc_sha256: str,
    device: torch.device,
) -> tuple[CartPoleBCPolicy, CartPoleBCPolicy, CartPoleValueHead, dict]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"actor/critic checkpoint digest drift: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise RuntimeError("unsupported actor/critic checkpoint format")
    if payload.get("world_checkpoint_sha256") != expected_world_sha256:
        raise RuntimeError("actor/critic world pairing drift")
    if payload.get("bc_checkpoint_sha256") != expected_bc_sha256:
        raise RuntimeError("actor/critic BC pairing drift")
    if payload["provenance"]["actor_sources"] != actor_source_report():
        raise RuntimeError("actor/critic primary-source provenance drift")
    cfg = payload["config"]
    actor = CartPoleBCPolicy(
        d_model=cfg["d_model"],
        n_actions=cfg["n_actions"],
    ).to(device)
    prior = CartPoleBCPolicy(
        d_model=cfg["d_model"],
        n_actions=cfg["n_actions"],
    ).to(device)
    value = CartPoleValueHead(
        d_model=cfg["d_model"],
        num_bins=cfg["value_bins"],
        log_low=cfg["value_log_low"],
        log_high=cfg["value_log_high"],
    ).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    prior.load_state_dict(payload["prior"], strict=True)
    value.load_state_dict(payload["value"], strict=True)
    freeze_module(actor)
    freeze_module(prior)
    freeze_module(value)
    if module_state_sha256(prior) != payload["frozen_invariants"][
        "prior_tensor_sha256_after"
    ]:
        raise RuntimeError("stored prior tensor identity drift")
    return actor, prior, value, payload


def _summary(rows: list[dict], policy: str) -> dict[str, float | int]:
    selected = [row for row in rows if row["policy"] == policy]
    returns = [row["return"] for row in selected]
    return {
        "episodes": len(selected),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "minimum_return": float(np.min(returns)),
        "maximum_return": float(np.max(returns)),
        "total_wall_seconds": float(
            sum(row["wall_seconds"] for row in selected)
        ),
    }


def _direct_execution_policy(policy_name: str) -> str:
    """Map report labels onto direct environment controllers.

    Both learned heads use the existing direct pixel-policy execution path.
    Keeping this allowlist separate makes it impossible for actor evaluation
    to silently route through the shooting planner.
    """
    if policy_name in {"bc_policy", "imagination_actor"}:
        return "bc_policy"
    if policy_name in {"random", "oracle_reference"}:
        return policy_name
    raise ValueError(f"unsupported direct policy {policy_name!r}")


def evaluate_actor_parity(
    *,
    world_checkpoint: Path,
    world_checkpoint_sha256: str,
    bc_checkpoint: Path,
    bc_checkpoint_sha256: str,
    actor_checkpoint: Path,
    actor_checkpoint_sha256: str,
    output: Path,
    seeds: list[int],
    context: int,
    policy_seed_base: int,
    historical_bc_mean: float,
    noninferiority_margin: float,
    device: torch.device,
) -> dict:
    """Evaluate direct actor execution against its frozen paired BC prior."""
    world, _, world_payload = load_checkpoint(
        world_checkpoint,
        device=device,
        expected_sha256=world_checkpoint_sha256,
        strict_implementation=False,
    )
    freeze_module(world)
    bc, _ = load_bc_policy(
        bc_checkpoint,
        expected_sha256=bc_checkpoint_sha256,
        expected_world_sha256=world_checkpoint_sha256,
        device=device,
    )
    freeze_module(bc)
    actor, prior, _, actor_payload = load_imagination_actor_critic(
        actor_checkpoint,
        expected_sha256=actor_checkpoint_sha256,
        expected_world_sha256=world_checkpoint_sha256,
        expected_bc_sha256=bc_checkpoint_sha256,
        device=device,
    )
    if module_state_sha256(bc) != module_state_sha256(prior):
        raise RuntimeError("actor checkpoint prior is not the paired BC policy")
    if module_state_sha256(world) != actor_payload["frozen_invariants"][
        "world_tensor_sha256_after"
    ]:
        raise RuntimeError("evaluation world differs from training world")

    policies = (
        ("random", None),
        ("bc_policy", bc),
        ("imagination_actor", actor),
        ("oracle_reference", None),
    )
    rows: list[dict] = []
    for policy_index, (policy_name, policy_module) in enumerate(policies):
        for index, environment_seed in enumerate(seeds):
            execution_name = _direct_execution_policy(policy_name)
            row = _run_control_episode(
                world=world,
                policy=execution_name,
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
                bc_policy=policy_module,
            )
            row["policy"] = policy_name
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

    names = tuple(name for name, _ in policies)
    summaries = {name: _summary(rows, name) for name in names}
    by_policy_seed = {
        name: {
            row["environment_seed"]: row
            for row in rows
            if row["policy"] == name
        }
        for name in names
    }
    actor_minus_bc = [
        by_policy_seed["imagination_actor"][seed]["return"]
        - by_policy_seed["bc_policy"][seed]["return"]
        for seed in seeds
    ]
    actor_minus_random = [
        by_policy_seed["imagination_actor"][seed]["return"]
        - by_policy_seed["random"][seed]["return"]
        for seed in seeds
    ]
    parity_ci = paired_bootstrap_interval(
        actor_minus_bc,
        seed=policy_seed_base + 7_000_000,
    )
    random_ci = paired_bootstrap_interval(
        actor_minus_random,
        seed=policy_seed_base + 8_000_000,
    )
    actor_mean = float(summaries["imagination_actor"]["mean_return"])
    bc_mean = float(summaries["bc_policy"]["mean_return"])
    evidence = actor_payload["metrics"]
    invariants = actor_payload["frozen_invariants"]
    gate = {
        "actor_mean_at_least_paired_bc": actor_mean >= bc_mean,
        "actor_mean_at_least_historical_reference": (
            actor_mean >= historical_bc_mean
        ),
        "actor_minus_bc_mean": float(np.mean(actor_minus_bc)),
        "actor_minus_bc_bootstrap_95_ci": parity_ci,
        "actor_noninferiority_ci_lower_at_least_margin": (
            parity_ci[0] >= -noninferiority_margin
        ),
        "actor_changed_from_bc": (
            evidence["actor_l2_delta_from_bc"] > 0.0
        ),
        "value_head_trained": (
            evidence["value_l2_delta_from_initial"] > 0.0
        ),
        "both_pmpo_sign_sets_observed": (
            evidence["positive_count_total"] > 0
            and evidence["negative_count_total"] > 0
        ),
        "world_frozen": (
            invariants["world_tensor_sha256_before"]
            == invariants["world_tensor_sha256_after"]
        ),
        "prior_frozen": (
            invariants["prior_tensor_sha256_before"]
            == invariants["prior_tensor_sha256_after"]
        ),
        "actor_minus_random_mean": float(np.mean(actor_minus_random)),
        "actor_minus_random_bootstrap_95_ci": random_ci,
        "actor_beats_random_with_ci": random_ci[0] > 0.0,
    }
    required = (
        "actor_mean_at_least_paired_bc",
        "actor_noninferiority_ci_lower_at_least_margin",
        "actor_beats_random_with_ci",
        "actor_changed_from_bc",
        "value_head_trained",
        "both_pmpo_sign_sets_observed",
        "world_frozen",
        "prior_frozen",
    )
    gate["parity_achieved"] = all(bool(gate[name]) for name in required)
    payload = {
        "format": EVALUATION_FORMAT,
        "status": "completed",
        "result": (
            "DREAMER4_ACTOR_CRITIC_PARITY"
            if gate["parity_achieved"]
            else "PARITY_NOT_ACHIEVED"
        ),
        "claim_boundary": (
            "direct greedy execution of an actor trained only on frozen-world "
            "imagination; no shooting, online learning, simulator state, "
            "Mamba, or CDP/JEPA"
        ),
        "protocol": {
            "seeds": seeds,
            "fresh_from_replay": True,
            "context": context,
            "environment_action_repeat": ACTION_REPEAT,
            "learning_during_evaluation": False,
            "shooting_planner_used": False,
            "historical_bc_mean": historical_bc_mean,
            "historical_bc_mean_is_descriptive_only": True,
            "noninferiority_margin": noninferiority_margin,
            "paired_bootstrap_draws": 20_000,
        },
        "provenance": {
            "world_checkpoint": str(world_checkpoint),
            "world_checkpoint_sha256": file_sha256(world_checkpoint),
            "world_checkpoint_step": world_payload["step"],
            "bc_checkpoint": str(bc_checkpoint),
            "bc_checkpoint_sha256": file_sha256(bc_checkpoint),
            "actor_checkpoint": str(actor_checkpoint),
            "actor_checkpoint_sha256": file_sha256(actor_checkpoint),
            "evaluation_implementation_sha256": implementation_sha256(),
            "actor_sources": actor_source_report(),
            "world_sources": source_report(),
        },
        "rows": rows,
        "summary": summaries,
        "actor_minus_bc": actor_minus_bc,
        "actor_minus_random": actor_minus_random,
        "gate": gate,
    }
    _atomic_json(output, payload)
    output.with_name(f".{output.name}.progress").unlink(missing_ok=True)
    return payload


def _parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--world-checkpoint", type=Path, required=True)
    train.add_argument("--world-checkpoint-sha256", required=True)
    train.add_argument("--bc-checkpoint", type=Path, required=True)
    train.add_argument("--bc-checkpoint-sha256", required=True)
    train.add_argument("--replay", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=1_000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--context", type=int, default=8)
    train.add_argument("--horizon", type=int, default=32)
    train.add_argument("--denoise-steps", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--gamma", type=float, default=0.997)
    train.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    train.add_argument("--alpha", type=float, default=0.5)
    train.add_argument("--beta", type=float, default=0.3)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=20260725)
    train.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--world-checkpoint", type=Path, required=True)
    evaluate.add_argument("--world-checkpoint-sha256", required=True)
    evaluate.add_argument("--bc-checkpoint", type=Path, required=True)
    evaluate.add_argument("--bc-checkpoint-sha256", required=True)
    evaluate.add_argument("--actor-checkpoint", type=Path, required=True)
    evaluate.add_argument("--actor-checkpoint-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seeds", default="970000:970030")
    evaluate.add_argument("--context", type=int, default=8)
    evaluate.add_argument("--policy-seed-base", type=int, default=20260725)
    evaluate.add_argument("--historical-bc-mean", type=float, default=288.7)
    evaluate.add_argument("--noninferiority-margin", type=float, default=25.0)
    evaluate.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    if args.command == "train":
        report = train_imagination_actor_critic(
            world_checkpoint=args.world_checkpoint,
            world_checkpoint_sha256=args.world_checkpoint_sha256,
            bc_checkpoint=args.bc_checkpoint,
            bc_checkpoint_sha256=args.bc_checkpoint_sha256,
            replay_path=args.replay,
            output=args.output,
            device=torch.device(args.device),
            steps=args.steps,
            batch_size=args.batch_size,
            context=args.context,
            horizon=args.horizon,
            denoise_steps=args.denoise_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            lambda_=args.lambda_,
            alpha=args.alpha,
            beta=args.beta,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
        )
        print(json.dumps(report["metrics"], indent=2, sort_keys=True))
        print(json.dumps(report["checkpoint"], indent=2, sort_keys=True))
    elif args.command == "evaluate":
        report = evaluate_actor_parity(
            world_checkpoint=args.world_checkpoint,
            world_checkpoint_sha256=args.world_checkpoint_sha256,
            bc_checkpoint=args.bc_checkpoint,
            bc_checkpoint_sha256=args.bc_checkpoint_sha256,
            actor_checkpoint=args.actor_checkpoint,
            actor_checkpoint_sha256=args.actor_checkpoint_sha256,
            output=args.output,
            seeds=_parse_seeds(args.seeds),
            context=args.context,
            policy_seed_base=args.policy_seed_base,
            historical_bc_mean=args.historical_bc_mean,
            noninferiority_margin=args.noninferiority_margin,
            device=torch.device(args.device),
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(json.dumps(report["gate"], indent=2, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
