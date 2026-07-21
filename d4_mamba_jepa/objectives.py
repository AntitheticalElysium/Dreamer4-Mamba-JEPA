"""Training objectives with line-level provenance to primary implementations."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .source import load_mmbench2_model


def _max_step_index(k_max: int) -> int:
    return int(round(torch.log2(torch.tensor(float(k_max))).item()))


def _sample_step_excluding_finest(
    device: torch.device, batch: int, time: int, k_max: int
) -> tuple[Tensor, Tensor]:
    """Port of MMBench2 train_dynamics.py:_sample_step_excluding_dmin."""
    emax = _max_step_index(k_max)
    step_index = torch.randint(
        low=0,
        high=max(1, emax),
        size=(batch, time),
        device=device,
        dtype=torch.long,
    )
    step = 1.0 / (1 << step_index).to(torch.float32)
    return step, step_index


def _sample_tau(
    device: torch.device, batch: int, time: int, k_max: int, step_index: Tensor
) -> tuple[Tensor, Tensor]:
    """Port of MMBench2 train_dynamics.py:_sample_tau_for_step."""
    K = (1 << step_index).to(torch.long)
    unit = torch.rand((batch, time), device=device, dtype=torch.float32)
    j_index = torch.floor(unit * K.to(torch.float32)).to(torch.long)
    tau = j_index.to(torch.float32) / K.to(torch.float32)
    scale = torch.div(
        torch.tensor(k_max, device=device), K, rounding_mode="floor"
    )
    return tau, j_index * scale


def shortcut_flow_loss(
    dynamics: nn.Module,
    *,
    clean: Tensor,
    led_to_actions: Tensor,
    k_max: int,
    bootstrap_rows: int = 0,
    global_step: int = 0,
    bootstrap_start: int = 10_000,
    return_agent_tokens: bool = False,
) -> (
    tuple[Tensor, dict[str, Tensor]]
    | tuple[Tensor, dict[str, Tensor], Tensor]
):
    """MMBench2 shortcut-flow objective, with task heads deliberately absent.

    This is a direct PyTorch port of the flow portion of
    ``nicklashansen/mmbench2`` commit
    ``3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9``,
    ``src/train_dynamics.py:dynamics_pretrain_loss``. The only interface
    deviation is categorical ``led_to_actions`` and omission of the separately
    switchable reward/BC losses.
    """
    device = clean.device
    B, T = clean.shape[:2]
    if not 0 <= bootstrap_rows < B:
        raise ValueError("bootstrap_rows must satisfy 0 <= rows < batch")
    empirical_rows = B - bootstrap_rows
    max_step = _max_step_index(k_max)

    empirical_step_index = torch.full(
        (empirical_rows, T), max_step, device=device, dtype=torch.long
    )
    if bootstrap_rows:
        step, self_step_index = _sample_step_excluding_finest(
            device, bootstrap_rows, T, k_max
        )
        full_step_index = torch.cat(
            [empirical_step_index, self_step_index], dim=0
        )
    else:
        step = torch.zeros((0, T), device=device)
        self_step_index = torch.zeros((0, T), device=device, dtype=torch.long)
        full_step_index = empirical_step_index

    tau, signal_index = _sample_tau(device, B, T, k_max, full_step_index)
    tau_empirical = tau[:empirical_rows]
    tau_self = tau[empirical_rows:]
    signal_self = signal_index[empirical_rows:]

    noise = torch.randn_like(clean)
    noised = (1.0 - tau)[..., None, None] * noise + tau[
        ..., None, None
    ] * clean
    noised_self = noised[empirical_rows:]

    empirical_weight = 0.9 * tau_empirical + 0.1
    self_weight = 0.9 * tau_self + 0.1

    predicted, agent_tokens = dynamics(
        led_to_actions,
        full_step_index,
        signal_index,
        noised,
        act_mask=None,
        agent_tokens=None,
        lang_emb=None,
    )
    predicted_empirical = predicted[:empirical_rows]
    predicted_self = predicted[empirical_rows:]

    flow_per = (
        predicted_empirical.float() - clean[:empirical_rows].float()
    ).pow(2).mean(dim=(2, 3))
    empirical_loss = (flow_per * empirical_weight).mean()

    bootstrap_mse = torch.zeros((), device=device)
    self_loss = torch.zeros((), device=device)
    if bootstrap_rows and global_step >= bootstrap_start:
        half_step = step / 2.0
        half_step_index = self_step_index + 1
        tau_plus = tau_self + half_step
        signal_plus = (
            signal_self
            + (torch.tensor(k_max, device=device) * half_step).to(torch.long)
        ).clamp(0, k_max)
        actions_self = led_to_actions[empirical_rows:]

        with torch.no_grad():
            half_prediction_1, _ = dynamics(
                actions_self,
                half_step_index,
                signal_self,
                noised_self,
                act_mask=None,
                agent_tokens=None,
                lang_emb=None,
            )
            velocity_1 = (
                half_prediction_1.float() - noised_self.float()
            ) / (1.0 - tau_self).clamp_min(1e-6)[..., None, None]
            intermediate = noised_self.float() + velocity_1 * half_step[
                ..., None, None
            ]
            half_prediction_2, _ = dynamics(
                actions_self,
                half_step_index,
                signal_plus,
                intermediate.to(noised_self.dtype),
                act_mask=None,
                agent_tokens=None,
                lang_emb=None,
            )
            velocity_2 = (
                half_prediction_2.float() - intermediate.float()
            ) / (1.0 - tau_plus).clamp_min(1e-6)[..., None, None]

        predicted_velocity = (
            predicted_self.float() - noised_self.float()
        ) / (1.0 - tau_self).clamp_min(1e-6)[..., None, None]
        target_velocity = (velocity_1 + velocity_2) / 2.0
        bootstrap_per = (1.0 - tau_self).pow(2) * (
            predicted_velocity - target_velocity
        ).pow(2).mean(dim=(2, 3))
        self_loss = (bootstrap_per * self_weight).mean()
        bootstrap_mse = bootstrap_per.mean()

    loss = (
        empirical_loss * empirical_rows + self_loss * bootstrap_rows
    ) / B
    metrics = {
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": bootstrap_mse.detach(),
        "empirical_loss": empirical_loss.detach(),
        "self_loss": self_loss.detach(),
        "tau_mean": tau.mean().detach(),
    }
    if return_agent_tokens:
        if agent_tokens is None:
            raise RuntimeError(
                "task-head routing requested but dynamics returned no agent tokens"
            )
        return loss, metrics, agent_tokens
    return loss, metrics


def continuation_mtp_loss(
    logits: Tensor,
    led_to_continues: Tensor,
    valid: Tensor,
    *,
    terminal_weight: float = 1.0,
) -> Tensor:
    """Align MTP head ``l`` with continuation at state slot ``t+l``.

    ``terminal_weight`` (default 1.0, unchanged for base/cdp) up-weights the
    rare terminal (continue==0) targets so the head does not collapse to the
    majority "continue" label on a sparse-terminal task.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,T,L]")
    B, T, L = logits.shape
    if led_to_continues.shape != (B, T) or valid.shape != (B, T):
        raise ValueError("continuation targets and valid mask must have shape [B,T]")
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for lead in range(L):
        usable = T - lead
        if usable <= 0:
            break
        targets[:, :usable, lead] = led_to_continues[:, lead:]
        mask[:, :usable, lead] = valid[:, lead:]
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weights = mask.to(per.dtype)
    if terminal_weight != 1.0:
        weights = weights * torch.where(
            targets == 0.0,
            torch.as_tensor(float(terminal_weight), device=logits.device, dtype=per.dtype),
            torch.ones((), device=logits.device, dtype=per.dtype),
        )
    return (per * weights).sum() / weights.sum().clamp_min(1.0)


def cdp_cosine_loss(
    world: nn.Module,
    *,
    clean: Tensor,
    led_to_actions: Tensor,
) -> tuple[Tensor, Tensor]:
    """CDP-shaped next-representation loss with a stop-gradient target.

    The predictor consumes causal state ``t`` plus ``action_t`` and predicts
    the packed representation of observation ``t+1``. The future target is
    detached exactly at this boundary. Unlike the upstream flow loss, no noisy
    portion of the future target is an input to this predictor.
    """
    prediction = world.predict_cdp(clean, led_to_actions)
    target = clean[:, 1:].detach()
    loss = 1.0 - torch.nn.functional.cosine_similarity(
        prediction.float(), target.float(), dim=-1, eps=1e-8
    )
    return loss.mean(), prediction


def jepa_self_prediction_loss(
    world: nn.Module,
    *,
    frames: Tensor,
    clean: Tensor,
    led_to_actions: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Non-generative SPR/BYOL action-conditioned self-prediction.

    Faithful to ``mila-iqia/spr`` (commit ``0b9dd4e7``) ``do_spr_loss`` and
    ``spr_loss``: the deterministic action-conditioned predictor produces the
    next embedding (the rollout primitive), which is projected+predicted online;
    the target is the stop-gradient EMA target encoder applied to the *actual*
    next observation, projected by the EMA target projection. The loss is the
    L2-normalized MSE (proportional in gradient to negative cosine). There is no
    denoising, no decoder, and no reconstruction term.

    ``clean`` are the online-encoder packed latents (gradients flow to the
    online encoder); ``frames`` are the observations whose future is the target.
    """
    B, T, N, D = clean.shape
    K = int(getattr(world.cfg, "jepa_jumps", 1))
    context = T - K
    if context < 1:
        raise ValueError("jepa_jumps must leave at least one context step")
    device = clean.device
    max_step = world.cfg.max_step_index
    k_max = world.cfg.k_max
    with torch.no_grad():
        target_latent = world.encode_frames_target(frames)  # [B,T,N,D]

    # Autoregressive multi-step rollout: feed the predictor's own output back,
    # matching each step to the EMA target of the actual future frame. This is
    # the SPR `jumps` mechanism and makes training match the imagined rollout.
    past = clean[:, :context]  # online latents; gradients flow to encoder
    online_steps, target_steps = [], []
    for jump in range(K):
        pos = context + jump  # position being predicted
        time = past.shape[1]
        steps = torch.full((B, time), max_step, device=device, dtype=torch.long)
        signals = torch.full((B, time), k_max, device=device, dtype=torch.long)
        _, agent = world.forward_dynamics(
            past, led_to_actions[:, :time], steps, signals
        )
        next_action_tokens = world.dynamics.action_encoder(
            led_to_actions[:, pos : pos + 1], batch_time_shape=(B, 1), act_mask=None
        )[:, :, 0]
        pred_z = world.jepa_predictor(agent[:, -1:], next_action_tokens)  # [B,1,N,D]
        online_steps.append(world.jepa_online_project(pred_z))  # [B,1,P]
        target_steps.append(world.jepa_target_project(target_latent[:, pos : pos + 1]))
        past = torch.cat([past, pred_z], dim=1)

    online = torch.cat(online_steps, dim=1)  # [B,K,P]
    target = torch.cat(target_steps, dim=1)  # [B,K,P], no grad

    f_online = torch.nn.functional.normalize(
        online.float(), p=2.0, dim=-1, eps=1e-3
    )
    f_target = torch.nn.functional.normalize(
        target.float().detach(), p=2.0, dim=-1, eps=1e-3
    )
    per = torch.nn.functional.mse_loss(f_online, f_target, reduction="none").sum(-1)
    loss = per.mean()
    cosine = (f_online * f_target).sum(-1).mean()
    # Collapse diagnostics: dispersion of online predictions and targets.
    online_std = f_online.reshape(-1, f_online.shape[-1]).std(dim=0).mean()
    target_std = f_target.reshape(-1, f_target.shape[-1]).std(dim=0).mean()
    metrics = {
        "jepa_cosine": cosine.detach(),
        "jepa_online_std": online_std.detach(),
        "jepa_target_std": target_std.detach(),
    }
    return loss, metrics


def reconstruction_anchor_loss(
    world: nn.Module,
    *,
    frames: Tensor,
    bottleneck: Tensor,
) -> Tensor:
    """Full-frame frozen-decoder anchor for the slowly updated CDP encoder.

    MMBench2 trains its tokenizer with masked reconstruction and then freezes
    it. The CDP arm reopens only the encoder, so this full reconstruction loss
    keeps its representations compatible with the fixed upstream decoder.
    This is a registered stabilizer, not claimed to be Dreamer-CDP behavior.
    """
    upstream = load_mmbench2_model()
    pixels = frames.float()
    if frames.dtype == torch.uint8:
        pixels = pixels / 255.0
    target = upstream.temporal_patchify(pixels, world.cfg.patch_size)
    prediction = world.decoder(bottleneck)
    return (prediction.float() - target.float()).pow(2).mean()


def optimizer_groups(world: nn.Module, base_lr: float) -> list[dict]:
    """Disjoint optimizer groups with explicit CDP encoder LR separation."""
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    encoder = [
        parameter for parameter in world.encoder.parameters()
        if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in encoder}
    main = [
        parameter
        for parameter in world.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups = []
    if encoder:
        groups.append(
            {
                "name": "encoder",
                "params": encoder,
                "lr": base_lr * world.cfg.encoder_lr_ratio,
            }
        )
    groups.append({"name": "main", "params": main, "lr": base_lr})
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    decoder_ids = {id(parameter) for parameter in world.decoder.parameters()}
    if decoder_ids.intersection(all_ids):
        raise RuntimeError("frozen reconstruction decoder entered optimizer")
    return groups
