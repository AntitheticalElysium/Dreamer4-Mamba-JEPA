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

from .common import (
    BCPolicy,
    _atomic_torch_save,
    _episode_window,
    load_bc_policy,
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
# TD-lambda here follows Dreamer 4 equation 10, whose lambda-return carries the
# continuation factor c_t (see ``td_lambda_returns``, which multiplies by
# ``continues``). The inspected ``edwhu/dreamer4-jax`` runner computes TD-lambda
# from V1..VT and r1..rT with a bootstrap VT and no continuation term, so it is
# a read-only reference for the actor/value loop and optimizer shape, NOT the
# source of the return equation.
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


class ValueHead(nn.Module):
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
    """Compute returns for ``s_t,a_t,r_{t+1},c_{t+1},s_{t+1}``.

    Follows Dreamer 4 equation 10: the lambda-return is discounted by
    ``gamma * c_t``. The inspected ``edwhu/dreamer4-jax`` runner omits the
    continuation factor and is therefore not the source of this equation.
    """
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
    actor: BCPolicy,
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
    actor: BCPolicy,
    prior: BCPolicy,
    value: ValueHead,
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

def load_imagination_actor_critic(
    path: Path,
    *,
    expected_sha256: str,
    expected_world_sha256: str,
    expected_bc_sha256: str,
    device: torch.device,
) -> tuple[BCPolicy, BCPolicy, ValueHead, dict]:
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
    actor = BCPolicy(
        d_model=cfg["d_model"],
        n_actions=cfg["n_actions"],
    ).to(device)
    prior = BCPolicy(
        d_model=cfg["d_model"],
        n_actions=cfg["n_actions"],
    ).to(device)
    value = ValueHead(
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
