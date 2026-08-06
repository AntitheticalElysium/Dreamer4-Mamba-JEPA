"""6 GB-safe optimisation steps and a deliberately small experiment skeleton."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
import torch

from model import M3HJWM, LossConfig, WorldState, online_hybrid_recipe
from agent import ActorCritic, imagine, actor_critic_losses, frozen


@dataclass(frozen=True)
class TrainConfig:
    # 6 GB-safe starting point; measure rather than assume.
    batch_size: int = 4
    sequence_length: int = 16
    imagination_batch: int = 32
    imagination_horizon: int = 8
    grad_accumulation: int = 4

    world_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    grad_clip: float = 100.0
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_coef: float = 3e-4

    # Must remain false until held-out reliability calibration succeeds.
    use_reliability_weights: bool = False
    amp: bool = True


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    # Prefer bfloat16 if the GPU supports it; otherwise float16.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def world_update(
    model: M3HJWM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    loss_weights: LossConfig | None = None,
):
    # This generic path backpropagates into the online encoder and EMA-updates
    # the target, so it defaults to the ONLINE recipe (anti-collapse on,
    # rollout off). Frozen-dynamics runs pass frozen_dynamics_recipe()
    # explicitly (2026-07-15 phase-recipe split).
    if loss_weights is None:
        loss_weights = online_hybrid_recipe()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(next(model.parameters()).device, cfg.amp):
        output = model(batch, loss_weights)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    model.mark_parameters_updated()
    model.update_target()
    return {name: float(value) for name, value in output.metrics.items()}


def actor_critic_update(
    world: M3HJWM,
    agent: ActorCritic,
    start: WorldState,
    actor_optimizer,
    critic_optimizer,
    cfg: TrainConfig,
):
    # Categorical policy and mode selection use score-function gradients. The
    # imagination helper clones/detaches the start cache and evaluates world-model
    # transitions under no_grad, leaving actor and critic graphs disjoint.
    with autocast_context(next(world.parameters()).device, cfg.amp):
        trajectory = imagine(world, agent, start, cfg.imagination_horizon)
        actor_loss, critic_loss, metrics = actor_critic_losses(
            trajectory,
            gamma=cfg.gamma,
            lambda_=cfg.lambda_,
            entropy_coef=cfg.entropy_coef,
            use_reliability=cfg.use_reliability_weights,
        )

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.grad_clip)
    actor_optimizer.step()

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.critics.parameters(), cfg.grad_clip)
    critic_optimizer.step()
    return {name: float(value) for name, value in metrics.items()}


def estimated_parameter_megabytes(module: torch.nn.Module, bytes_per_parameter: int = 4):
    return sum(p.numel() for p in module.parameters()) * bytes_per_parameter / 2**20


# The repository agent should build the actual collect/train/evaluate runner only
# after validating:
#   1. official Mamba step/sequence equivalence on the target GPU;
#   2. peak memory for world update and imagination update;
#   3. transition indexing and reset handling;
#   4. representation rank and mode verification controls.
