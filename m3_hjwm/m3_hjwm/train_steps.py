from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import Tensor
from .world_model import M3HJWM, WorldModelState
from .actor_critic import ActorCritic
from .config import TrainConfig
from .imagination import imagine, actor_critic_losses


@contextmanager
def freeze_parameters(module: torch.nn.Module):
    req = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), req):
            p.requires_grad_(r)


def world_model_step(
    model: M3HJWM,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    out = model(batch, cfg)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    model.update_target()
    return {k: float(v) for k, v in out.metrics.items()}


def actor_critic_step(
    model: M3HJWM,
    actor_critic: ActorCritic,
    start: WorldModelState,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    # Freeze WM parameters while retaining differentiability wrt actions/states if
    # the selected estimator needs it. Categorical actions use score-function gradients.
    with freeze_parameters(model):
        traj = imagine(
            model, actor_critic, start, cfg.imagination_horizon,
            reliability_temperature=cfg.reliability_temperature,
        )
    actor_loss, critic_loss, metrics = actor_critic_losses(
        traj, cfg.gamma, cfg.lambda_, cfg.entropy_coef,
        apply_confidence=not cfg.reliability_shadow_only,
    )

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), cfg.grad_clip)
    actor_opt.step()

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.critics.parameters(), cfg.grad_clip)
    critic_opt.step()
    return {k: float(v) for k, v in metrics.items()}
