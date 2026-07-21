"""Small-scale training composition around pinned upstream model components."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import SequenceBatch
from .objectives import (
    cdp_cosine_loss,
    continuation_mtp_loss,
    jepa_self_prediction_loss,
    reconstruction_anchor_loss,
    shortcut_flow_loss,
)
from .source import load_mmbench2_model


@dataclass(frozen=True)
class LossWeights:
    flow: float = 1.0
    reward: float = 1.0
    continuation: float = 1.0
    jepa: float = 1.0


class WorldLossNormalizer(nn.Module):
    """The upstream running-RMS normalization contract for every active term."""

    def __init__(self):
        super().__init__()
        upstream = load_mmbench2_model()
        self.terms = nn.ModuleDict(
            {
                name: upstream.EmaRms()
                for name in ("flow", "reward", "continuation", "cdp", "reconstruction")
            }
        )

    def apply(self, name: str, loss: Tensor, *, active: bool) -> Tensor:
        if not active:
            return loss
        normalizer = self.terms[name]
        normalizer.update(loss)
        return normalizer.normalize(loss)


def tokenizer_reconstruction_loss(
    tokenizer: nn.Module, frames: Tensor, *, patch_size: int
) -> tuple[Tensor, dict[str, Tensor]]:
    """Unchanged upstream MAE reconstruction operator."""
    upstream = load_mmbench2_model()
    pixels = frames.float()
    if frames.dtype == torch.uint8:
        pixels = pixels / 255.0
    patches = upstream.temporal_patchify(pixels, patch_size)
    prediction, mask, keep_probability = tokenizer(patches)
    loss = upstream.recon_loss_from_mae(prediction, patches, mask)
    return loss, {
        "masked_fraction": mask.float().mean().detach(),
        "keep_probability": keep_probability.float().mean().detach(),
    }


@torch.inference_mode()
def tokenizer_full_reconstruction_mse(
    tokenizer: nn.Module, frames: Tensor, *, patch_size: int
) -> Tensor:
    """Evaluate all-patch reconstruction without changing stored weights."""
    upstream = load_mmbench2_model()
    pixels = frames.float()
    if frames.dtype == torch.uint8:
        pixels = pixels / 255.0
    patches = upstream.temporal_patchify(pixels, patch_size)
    mae = tokenizer.encoder.mae
    saved = (mae.p_min, mae.p_max)
    mae.p_min = mae.p_max = 0.0
    try:
        prediction, _, _ = tokenizer(patches)
    finally:
        mae.p_min, mae.p_max = saved
    return (prediction.float() - patches.float()).pow(2).mean()


def _mtp_scalar_targets(
    values: Tensor, valid: Tensor, horizon: int
) -> tuple[Tensor, Tensor]:
    B, T = values.shape
    targets = torch.zeros((B, T, horizon), device=values.device, dtype=values.dtype)
    mask = torch.zeros((B, T, horizon), device=values.device, dtype=torch.bool)
    for lead in range(horizon):
        usable = T - lead
        if usable <= 0:
            break
        targets[:, :usable, lead] = values[:, lead:]
        mask[:, :usable, lead] = valid[:, lead:]
    return targets, mask


def reward_mtp_loss(
    logits: Tensor,
    centers: Tensor,
    led_to_rewards: Tensor,
    valid: Tensor,
) -> Tensor:
    """Upstream symlog two-hot operator with explicit led-to MTP alignment."""
    upstream = load_mmbench2_model()
    if logits.ndim != 4:
        raise ValueError("reward logits must have shape [B,T,L,K]")
    B, T, L, K = logits.shape
    if led_to_rewards.shape != (B, T) or valid.shape != (B, T):
        raise ValueError("reward targets and mask must have shape [B,T]")
    target, mask = _mtp_scalar_targets(
        upstream.symlog(led_to_rewards.float()), valid, L
    )
    return upstream.dist_cross_entropy_from_symlog(
        logits=logits.float().reshape(-1, K),
        target_symlog=target.reshape(-1),
        centers_log=centers,
        mask=mask.reshape(-1),
    )


def _jepa_world_loss(
    world: nn.Module,
    batch: SequenceBatch,
    *,
    normalizer: WorldLossNormalizer,
    weights: LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Non-generative JEPA world loss: SPR self-prediction (which trains the
    online encoder + dynamics + predictor) plus the reward/continuation heads,
    read from a deterministic clean dynamics pass matching deployment. There is
    no flow, cdp, or reconstruction term. The EMA target is updated by the
    caller after ``optimizer.step()``."""
    encoded = world.encode_frames(batch.observations, frozen=False)
    clean = encoded.packed
    jepa, jepa_metrics = jepa_self_prediction_loss(
        world,
        frames=batch.observations,
        clean=clean,
        led_to_actions=batch.led_to_actions,
    )
    B, T = clean.shape[:2]
    steps = torch.full(
        (B, T), world.cfg.max_step_index, device=clean.device, dtype=torch.long
    )
    signals = torch.full(
        (B, T), world.cfg.k_max, device=clean.device, dtype=torch.long
    )
    _, agent_tokens = world.forward_dynamics(
        clean, batch.led_to_actions, steps, signals
    )
    heads = world.forward_task_heads(agent_tokens)
    reward = reward_mtp_loss(
        heads["reward_logits"],
        heads["reward_centers"],
        batch.led_to_rewards,
        batch.outcome_valid,
    )
    continuation = continuation_mtp_loss(
        heads["continue_logits"],
        batch.led_to_continues,
        batch.outcome_valid,
        terminal_weight=getattr(world.cfg, "jepa_terminal_weight", 1.0),
    )
    reward_n = normalizer.apply("reward", reward, active=weights.reward > 0)
    continuation_n = normalizer.apply(
        "continuation", continuation, active=weights.continuation > 0
    )
    total = (
        weights.jepa * jepa
        + weights.reward * reward_n
        + weights.continuation * continuation_n
    )
    terms = {"jepa": jepa, "reward": reward, "continuation": continuation}
    metrics = {
        **{f"loss/{name}": value.detach() for name, value in terms.items()},
        **{f"jepa/{name}": value for name, value in jepa_metrics.items()},
        "loss/total": total.detach(),
    }
    return total, metrics


def world_loss(
    world: nn.Module,
    batch: SequenceBatch,
    *,
    normalizer: WorldLossNormalizer,
    weights: LossWeights = LossWeights(),
    global_step: int = 0,
    bootstrap_rows: int = 0,
    bootstrap_start: int = 10_000,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute independently reportable D4 flow, task, and CDP terms."""
    if world.cfg.representation_objective == "jepa":
        return _jepa_world_loss(
            world, batch, normalizer=normalizer, weights=weights
        )
    train_encoder = world.cfg.representation_objective == "cdp"
    encoded = world.encode_frames(
        batch.observations, frozen=not train_encoder
    )

    # The pinned D4 flow and task objectives operate on tokenizer latents
    # without changing the representation. This stays true in the CDP arm:
    # encoder gradients enter only through the registered CDP and
    # reconstruction routes below.
    detached = encoded.packed.detach()
    flow, flow_metrics, agent_tokens = shortcut_flow_loss(
        world.dynamics,
        clean=detached,
        led_to_actions=batch.led_to_actions,
        k_max=world.cfg.k_max,
        bootstrap_rows=bootstrap_rows,
        global_step=global_step,
        bootstrap_start=bootstrap_start,
        return_agent_tokens=True,
    )

    # MMBench2 trains the reward head from the agent tokens returned by the
    # same noised shortcut-flow forward pass. Reusing those tokens is essential:
    # at deployment the head reads a partially denoised generated slot, not a
    # second clean-latent pass. The local continuation head follows that same
    # routing.
    heads = world.forward_task_heads(agent_tokens)
    reward = reward_mtp_loss(
        heads["reward_logits"],
        heads["reward_centers"],
        batch.led_to_rewards,
        batch.outcome_valid,
    )
    continuation = continuation_mtp_loss(
        heads["continue_logits"],
        batch.led_to_continues,
        batch.outcome_valid,
    )

    cdp = torch.zeros((), device=detached.device)
    reconstruction = torch.zeros((), device=detached.device)
    if train_encoder:
        cdp, _ = cdp_cosine_loss(
            world,
            clean=encoded.packed,
            led_to_actions=batch.led_to_actions,
        )
        reconstruction = reconstruction_anchor_loss(
            world,
            frames=batch.observations,
            bottleneck=encoded.bottleneck,
        )

    terms = {
        "flow": flow,
        "reward": reward,
        "continuation": continuation,
        "cdp": cdp,
        "reconstruction": reconstruction,
    }
    normalized = {
        "flow": normalizer.apply("flow", flow, active=weights.flow > 0),
        "reward": normalizer.apply("reward", reward, active=weights.reward > 0),
        "continuation": normalizer.apply(
            "continuation", continuation, active=weights.continuation > 0
        ),
        "cdp": normalizer.apply(
            "cdp", cdp, active=train_encoder and world.cfg.cdp_weight > 0
        ),
        "reconstruction": normalizer.apply(
            "reconstruction",
            reconstruction,
            active=(
                train_encoder and world.cfg.reconstruction_anchor_weight > 0
            ),
        ),
    }
    total = (
        weights.flow * normalized["flow"]
        + weights.reward * normalized["reward"]
        + weights.continuation * normalized["continuation"]
        + world.cfg.cdp_weight * normalized["cdp"]
        + world.cfg.reconstruction_anchor_weight * normalized["reconstruction"]
    )
    metrics = {
        **{f"loss/{name}": value.detach() for name, value in terms.items()},
        **{f"flow/{name}": value for name, value in flow_metrics.items()},
        "loss/total": total.detach(),
    }
    return total, metrics
