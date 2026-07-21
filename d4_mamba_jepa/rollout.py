"""D4-style shortcut rollout and minimal categorical planning.

The denoising equations and context-cache structure follow
``nicklashansen/mmbench2`` commit
``3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9``,
``src/interactive.py:sample_one_timestep_packed``. Local changes are limited
to categorical actions, explicit random generators, and returning tensor
diagnostics rather than UI values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .source import load_mmbench2_model


def shortcut_schedule(k_max: int, denoise_steps: int) -> dict:
    if denoise_steps < 1 or denoise_steps & (denoise_steps - 1):
        raise ValueError("denoise_steps must be a positive power of two")
    if denoise_steps > k_max or k_max % denoise_steps:
        raise ValueError("denoise_steps must divide k_max")
    exponent = int(round(math.log2(denoise_steps)))
    scale = k_max // denoise_steps
    tau = [index / denoise_steps for index in range(denoise_steps)] + [1.0]
    tau_index = [
        index * scale for index in range(denoise_steps)
    ] + [k_max]
    return {
        "K": denoise_steps,
        "e": exponent,
        "tau": tau,
        "tau_index": tau_index,
        "dt": 1.0 / denoise_steps,
    }


def _sample_next_jepa(
    world, past_packed: Tensor, led_to_actions: Tensor
) -> tuple[Tensor, Tensor]:
    """Deterministic non-generative rollout step: one dynamics pass over the
    clean context, the action-conditioned predictor for the next embedding, and
    a second clean pass for the post-transition agent tokens. No denoising."""
    B, time = past_packed.shape[:2]
    device = past_packed.device
    max_step = world.cfg.max_step_index
    k_max = world.cfg.k_max
    steps = torch.full((B, time), max_step, device=device, dtype=torch.long)
    signals = torch.full((B, time), k_max, device=device, dtype=torch.long)
    _, agent_ctx = world.forward_dynamics(
        past_packed, led_to_actions[:, :time], steps, signals
    )
    next_action_tokens = world.dynamics.action_encoder(
        led_to_actions[:, time : time + 1],
        batch_time_shape=(B, 1),
        act_mask=None,
    )[:, :, 0]
    next_latent = world.jepa_predictor(agent_ctx[:, -1:], next_action_tokens)[:, 0]
    new_sequence = torch.cat([past_packed, next_latent[:, None]], dim=1)
    steps2 = torch.full((B, time + 1), max_step, device=device, dtype=torch.long)
    signals2 = torch.full((B, time + 1), k_max, device=device, dtype=torch.long)
    _, agent_new = world.forward_dynamics(
        new_sequence, led_to_actions, steps2, signals2
    )
    return next_latent, agent_new[:, -1:]


@torch.inference_mode()
def sample_next_packed(
    world,
    *,
    past_packed: Tensor,
    led_to_actions: Tensor,
    schedule: dict,
    use_cache: bool,
    generator: torch.Generator | None = None,
    initial_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if getattr(world.cfg, "representation_objective", "base") == "jepa":
        if past_packed.ndim != 4:
            raise ValueError("past_packed must have shape [B,t,S,D]")
        if led_to_actions.shape != (past_packed.shape[0], past_packed.shape[1] + 1):
            raise ValueError("led_to_actions must have shape [B,t+1]")
        return _sample_next_jepa(world, past_packed, led_to_actions)
    """Generate one next latent and its post-transition agent tokens.

    ``past_packed`` has ``t`` clean states. ``led_to_actions`` has ``t+1``
    slots: actions for the existing states followed by the candidate action
    that will lead to the generated state.
    """
    if past_packed.ndim != 4:
        raise ValueError("past_packed must have shape [B,t,S,D]")
    B, time, n_spatial, d_spatial = past_packed.shape
    if led_to_actions.shape != (B, time + 1):
        raise ValueError(
            f"led_to_actions shape {tuple(led_to_actions.shape)} "
            f"!= {(B, time + 1)}"
        )
    device, dtype = past_packed.device, past_packed.dtype
    K = int(schedule["K"])
    exponent = int(schedule["e"])
    tau = schedule["tau"]
    tau_index = schedule["tau_index"]
    dt = float(schedule["dt"])

    expected_noise_shape = (B, 1, n_spatial, d_spatial)
    if initial_noise is None:
        z = torch.randn(
            expected_noise_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if initial_noise.shape != expected_noise_shape:
            raise ValueError(
                f"initial_noise shape {tuple(initial_noise.shape)} "
                f"!= {expected_noise_shape}"
            )
        z = initial_noise.to(device=device, dtype=dtype)
    max_step = world.cfg.max_step_index

    cache = None
    if use_cache and time > 0:
        context_steps = torch.full(
            (B, time), max_step, device=device, dtype=torch.long
        )
        context_signals = torch.full(
            (B, time), world.cfg.k_max, device=device, dtype=torch.long
        )
        _, _, cache = world.dynamics(
            led_to_actions[:, :time],
            context_steps,
            context_signals,
            past_packed,
            act_mask=None,
            agent_tokens=None,
            lang_emb=None,
            return_kv_cache=True,
        )

    final_agent = None
    full_steps = torch.full(
        (B, time + 1), max_step, device=device, dtype=torch.long
    )
    full_steps[:, -1] = exponent
    full_signals = torch.full(
        (B, time + 1), world.cfg.k_max, device=device, dtype=torch.long
    )

    for index in range(K):
        tau_i = float(tau[index])
        signal_i = int(tau_index[index])
        if cache is not None:
            new_steps = torch.full(
                (B, 1), exponent, device=device, dtype=torch.long
            )
            new_signals = torch.full(
                (B, 1), signal_i, device=device, dtype=torch.long
            )
            prediction, agent = world.dynamics(
                led_to_actions[:, -1:],
                new_steps,
                new_signals,
                z,
                act_mask=None,
                agent_tokens=None,
                lang_emb=None,
                kv_cache=cache,
            )
        else:
            full_signals[:, -1] = signal_i
            sequence = torch.cat([past_packed, z], dim=1)
            prediction_full, agent_full = world.dynamics(
                led_to_actions,
                full_steps,
                full_signals,
                sequence,
                act_mask=None,
                agent_tokens=None,
                lang_emb=None,
            )
            prediction = prediction_full[:, -1:]
            agent = agent_full[:, -1:]

        final_agent = agent
        velocity = (
            prediction.float() - z.float()
        ) / max(1e-4, 1.0 - tau_i)
        z = (z.float() + velocity * dt).to(dtype)

    if final_agent is None:
        raise RuntimeError("shortcut schedule executed no steps")
    return z[:, 0], final_agent


@torch.inference_mode()
def score_action_plans(
    world,
    *,
    context_packed: Tensor,
    context_led_to_actions: Tensor,
    action_plans: Tensor,
    schedule: dict,
    discount: float = 0.99,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    common_random_numbers: bool = False,
) -> dict[str, Tensor]:
    """Score categorical plans by predicted reward and continuation."""
    if action_plans.ndim != 2:
        raise ValueError("action_plans must have shape [N,H]")
    candidates, horizon = action_plans.shape
    if context_packed.shape[0] not in (1, candidates):
        raise ValueError("context batch must be one or match candidate count")
    if context_led_to_actions.shape[0] not in (1, candidates):
        raise ValueError("context action batch must be one or match candidates")
    past = context_packed.expand(candidates, -1, -1, -1).contiguous()
    led_to = context_led_to_actions.expand(candidates, -1).contiguous()

    upstream = load_mmbench2_model()
    score = torch.zeros(candidates, device=past.device, dtype=torch.float32)
    survival = torch.ones_like(score)
    predicted_rewards = []
    predicted_continues = []
    generated = []

    for step in range(horizon):
        led_to = torch.cat([led_to, action_plans[:, step : step + 1]], dim=1)
        initial_noise = None
        if common_random_numbers:
            shared = torch.randn(
                (
                    1,
                    1,
                    past.shape[2],
                    past.shape[3],
                ),
                device=past.device,
                dtype=past.dtype,
                generator=generator,
            )
            initial_noise = shared.expand(candidates, -1, -1, -1).clone()
        next_latent, agent = sample_next_packed(
            world,
            past_packed=past,
            led_to_actions=led_to,
            schedule=schedule,
            use_cache=use_cache,
            generator=generator,
            initial_noise=initial_noise,
        )
        heads = world.forward_task_heads(agent)
        reward_logits = heads["reward_logits"][:, 0, 0]
        probabilities = reward_logits.float().softmax(dim=-1)
        expected_symlog = (
            probabilities * heads["reward_centers"].float()
        ).sum(dim=-1)
        reward = upstream.symexp(expected_symlog)
        continuation = heads["continue_logits"][:, 0, 0].float().sigmoid()

        score = score + (discount ** step) * survival * reward
        survival = survival * continuation
        predicted_rewards.append(reward)
        predicted_continues.append(continuation)
        generated.append(next_latent)
        past = torch.cat([past, next_latent[:, None]], dim=1)

    return {
        "score": score,
        "rewards": torch.stack(predicted_rewards, dim=1),
        "continues": torch.stack(predicted_continues, dim=1),
        "generated": torch.stack(generated, dim=1),
    }


@dataclass(frozen=True)
class ShootingResult:
    action: int
    plan: Tensor
    score: float
    plans: Tensor
    scores: Tensor


@torch.inference_mode()
def categorical_random_shooting(
    world,
    *,
    context_packed: Tensor,
    context_led_to_actions: Tensor,
    horizon: int,
    candidates: int,
    schedule: dict,
    discount: float = 0.99,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    common_random_numbers: bool = False,
    selection: str = "best_plan",
    enumerate_all: bool = False,
) -> ShootingResult:
    if candidates < world.cfg.n_actions:
        raise ValueError("candidates must cover every first action at least once")
    if enumerate_all:
        expected = world.cfg.n_actions ** horizon
        if candidates != expected:
            raise ValueError(
                f"full enumeration requires {expected} candidates, got {candidates}"
            )
        numbers = torch.arange(candidates, device=context_packed.device)
        digits = []
        for power in reversed(range(horizon)):
            divisor = world.cfg.n_actions ** power
            digits.append((numbers // divisor) % world.cfg.n_actions)
        plans = torch.stack(digits, dim=1)
    else:
        plans = torch.randint(
            0,
            world.cfg.n_actions,
            (candidates, horizon),
            device=context_packed.device,
            generator=generator,
        )
        # Stratify the first decision. Later actions remain random.
        plans[:, 0] = torch.arange(
            candidates, device=plans.device
        ) % world.cfg.n_actions
    outputs = score_action_plans(
        world,
        context_packed=context_packed,
        context_led_to_actions=context_led_to_actions,
        action_plans=plans,
        schedule=schedule,
        discount=discount,
        use_cache=use_cache,
        generator=generator,
        common_random_numbers=common_random_numbers,
    )
    if selection == "best_plan":
        best = int(outputs["score"].argmax().item())
    elif selection == "first_action_mean":
        action_values = torch.stack(
            [
                outputs["score"][plans[:, 0] == action].mean()
                for action in range(world.cfg.n_actions)
            ]
        )
        chosen_action = int(action_values.argmax().item())
        eligible = torch.where(plans[:, 0] == chosen_action)[0]
        best = int(
            eligible[outputs["score"][eligible].argmax()].item()
        )
    else:
        raise ValueError(f"unsupported shooting selection {selection!r}")
    selected = plans[best].clone()
    return ShootingResult(
        action=int(selected[0].item()),
        plan=selected,
        score=float(outputs["score"][best].item()),
        plans=plans,
        scores=outputs["score"],
    )
