"""Faithful same-frame I-JEPA pretraining, isolated from action prediction.

Port of the pinned source (facebookresearch/ijepa @ 52c1ae9):
- masks: src/masks/multiblock.py (4 pred blocks 0.15-0.2, aspect 0.75-1.5;
  context 0.85-1.0 minus pred overlap; per-batch min-trim to uniform counts);
- predictor: src/models/vision_transformer.py::VisionTransformerPredictor
  (fixed 2D-sincos positions, learned mask token + target-position queries,
  per-target-block prediction with the context repeated per block);
- objective: src/train.py (targets = EMA encoder on the full frame, LayerNorm
  over features, gathered at target positions; smooth-L1 at those positions).

Labelled deviations and the pre-registered gates live in
reviews/2026-07-13-step1-protocol.md. This module trains ONLY the encoder,
its EMA copy, and a disposable predictor.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import (
    ConvBlock,
    EMARepresentationEncoder,
    ModelConfig,
    RepresentationEncoder,
    SpatialBlock,
)


def _masked_group_norm(norm: nn.GroupNorm, x: Tensor, active: Tensor) -> Tensor:
    """GroupNorm(1, C) with statistics over visible positions only.

    SparK computes norm statistics on non-masked positions (sparse_encoder.py::
    sp_bn_forward in the pinned CNN-JEPA); with dense statistics the masked
    zeros would bias mean/var and leak the mask pattern itself.
    """
    if norm.num_groups != 1:
        raise NotImplementedError("masked path supports GroupNorm(1, C) only")
    weight = active.expand_as(x)
    count = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    mean = (x * weight).sum(dim=(1, 2, 3), keepdim=True) / count
    var = ((x - mean).pow(2) * weight).sum(dim=(1, 2, 3), keepdim=True) / count
    out = (x - mean) / (var + norm.eps).sqrt()
    out = out * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)
    return out * active


def _sparse_stage(module: nn.Module, x: Tensor, active: Tensor) -> Tensor:
    """Execute a stem module sparsely: re-zero masked positions after every conv
    and use visible-only norm statistics (SparK sp_conv_forward recipe)."""
    if isinstance(module, nn.Conv2d):
        x = module(x)
        if module.stride[0] > 1:
            active = F.max_pool2d(active, kernel_size=module.stride, stride=module.stride)
        return x * active, active
    if isinstance(module, nn.GroupNorm):
        return _masked_group_norm(module, x, active), active
    if isinstance(module, ConvBlock):
        residual = x
        y = x
        for child in module.net:
            y, active = _sparse_stage(child, y, active)
        return residual + y, active
    x = module(x)  # pointwise activations keep zeros at zero (SiLU(0) == 0)
    return x * active, active


def sparse_stem_tokens(
    encoder: RepresentationEncoder, obs: Tensor, patch_active: Tensor
) -> Tensor:
    """Leak-free local tokens: masked-patch content cannot influence visible
    tokens. `patch_active` is [B, grid, grid] True at VISIBLE patches."""
    x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
    scale = x.shape[-1] // patch_active.shape[-1]
    active = patch_active[:, None].float()
    active = active.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    x = x * active
    for module in encoder.stem:
        x, active = _sparse_stage(module, x, active)
    x = encoder.project(x) * active
    return x.flatten(2).transpose(1, 2)


def sincos_2d_pos_embed(dim: int, grid: int) -> Tensor:
    """Fixed 2D sin-cos embeddings, as in the pinned I-JEPA (pos_embs.py)."""
    if dim % 4:
        raise ValueError("sincos embedding needs dim divisible by 4")
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    coords = torch.arange(grid, dtype=torch.float64)
    out = torch.einsum("p,f->pf", coords, omega)
    per_axis = torch.cat([out.sin(), out.cos()], dim=1)      # [grid, dim/2]
    row = per_axis[:, None].expand(grid, grid, dim // 2)
    col = per_axis[None, :].expand(grid, grid, dim // 2)
    return torch.cat([row, col], dim=-1).reshape(grid * grid, dim).float()


def _sample_block(grid: int, scale: tuple, aspect: tuple, generator) -> Tensor:
    """One rectangular block of grid cells (official _sample_block_size/_mask)."""
    rand = torch.rand(3, generator=generator)
    target_scale = scale[0] + float(rand[0]) * (scale[1] - scale[0])
    max_keep = max(1, int(grid * grid * target_scale))
    log_low, log_high = math.log(aspect[0]), math.log(aspect[1])
    ratio = math.exp(log_low + float(rand[1]) * (log_high - log_low))
    h = max(1, min(grid, int(round(math.sqrt(max_keep * ratio)))))
    w = max(1, min(grid, int(round(math.sqrt(max_keep / ratio)))))
    top = int(torch.randint(0, grid - h + 1, (1,), generator=generator))
    left = int(torch.randint(0, grid - w + 1, (1,), generator=generator))
    block = torch.zeros(grid, grid, dtype=torch.bool)
    block[top:top + h, left:left + w] = True
    return block.flatten()


def sample_ijepa_masks(
    batch: int,
    grid: int,
    generator,
    pred_scale: tuple = (0.15, 0.2),
    enc_scale: tuple = (0.85, 1.0),
    aspect: tuple = (0.75, 1.5),
    n_pred: int = 4,
    min_keep: int = 4,
):
    """Returns (context_index [B,Kc], pred_indices list of n_pred [B,Kp]).

    Counts are trimmed to the per-batch minimum, as in the official collator, so
    every sample in the batch has uniform index shapes.
    """
    contexts, preds = [], [[] for _ in range(n_pred)]
    min_ctx, min_pred = grid * grid, [grid * grid] * n_pred
    for _ in range(batch):
        pred_blocks = [
            _sample_block(grid, pred_scale, aspect, generator) for _ in range(n_pred)
        ]
        union = torch.stack(pred_blocks).any(0)
        context = _sample_block(grid, enc_scale, (1.0, 1.0), generator) & ~union
        if int(context.sum()) < min_keep:
            candidates = (~union).nonzero().flatten()
            if len(candidates) < min_keep:  # pathological union; keep any cells
                candidates = torch.arange(grid * grid)
            context = torch.zeros(grid * grid, dtype=torch.bool)
            context[candidates[:min_keep]] = True
        contexts.append(context.nonzero().flatten())
        min_ctx = min(min_ctx, len(contexts[-1]))
        for i, block in enumerate(pred_blocks):
            idx = block.nonzero().flatten()
            preds[i].append(idx)
            min_pred[i] = min(min_pred[i], len(idx))
    context_index = torch.stack([c[:min_ctx] for c in contexts])
    pred_indices = [
        torch.stack([p[:min_pred[i]] for p in preds[i]]) for i in range(n_pred)
    ]
    return context_index, pred_indices


class IJEPAPredictor(nn.Module):
    """Narrow ViT-style predictor over [context tokens + mask-token queries]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        grid = cfg.image_size // cfg.patch_size
        self.registers = cfg.registers
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.token_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed", sincos_2d_pos_embed(cfg.token_dim, grid), persistent=False
        )
        self.blocks = nn.ModuleList(
            [SpatialBlock(cfg.token_dim, cfg.spatial_heads) for _ in range(cfg.predictor_depth)]
        )
        self.norm = nn.LayerNorm(cfg.token_dim)
        self.proj = nn.Linear(cfg.token_dim, cfg.token_dim)

    def forward(self, context_tokens: Tensor, context_index: Tensor, pred_index: Tensor):
        """context_tokens: [B, registers+Kc, D] (registers first, as encoded);
        indices address the local grid. Returns predictions [B, Kp, D]."""
        b, _, d = context_tokens.shape
        pos = self.pos_embed.to(context_tokens.dtype)
        ctx = context_tokens.clone()
        ctx_pos = pos[context_index]                             # [B, Kc, D]
        ctx = torch.cat(
            [ctx[:, : self.registers], ctx[:, self.registers:] + ctx_pos], dim=1
        )
        queries = self.mask_token.expand(b, pred_index.shape[1], d) + pos[pred_index]
        x = torch.cat([ctx, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x[:, -pred_index.shape[1]:]))


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization, ported from the pinned
    official LeJEPA (rbalestr-lab/lejepa @ c293d29, MINIMAL.md): empirical
    characteristic function of 256 random 1-D projections matched to the
    standard Gaussian on a 17-knot quadrature grid over [0, 3], scaled by the
    sample count.

    Adaptations (reviews/2026-07-13-step1-protocol.md, amendment 1c): the
    sample axis is observations-at-a-fixed-stream (corrected axis semantics),
    statistic averaged over streams; applied to encoder tokens directly, no
    projector (a projector was measured to hide encoder collapse).
    """

    def __init__(self, knots: int = 17, projections: int = 256):
        super().__init__()
        self.projections = projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [B, S, D]; samples = B observations at each fixed stream."""
        proj = tokens.float().transpose(0, 1)            # [S, B, D]
        directions = torch.randn(
            proj.size(-1), self.projections, device=proj.device
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
        x_t = (proj @ directions).unsqueeze(-1) * self.t  # [S, B, P, K]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class IJEPAPretrainer(nn.Module):
    def __init__(self, cfg: ModelConfig, leak_free: bool = True):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.leak_free = leak_free
        self.grid = cfg.image_size // cfg.patch_size
        self.online_encoder = RepresentationEncoder(cfg)
        self.target_encoder = EMARepresentationEncoder(self.online_encoder, cfg.ema_decay)
        self.predictor = IJEPAPredictor(cfg)
        # LeJEPA composition (amendment 1c): loss = (1-l)*prediction + l*SIGReg.
        # sigreg_weight = 0 recovers the pure I-JEPA arm.
        self.sigreg = SIGReg()
        self.sigreg_weight = 0.0

    def _encode_context(self, obs: Tensor, context_index: Tensor) -> Tensor:
        if not self.leak_free:
            return self.online_encoder(obs, visible_index=context_index)
        # SparK-style sparse execution (pinned CNN-JEPA): masked-patch content
        # cannot reach visible tokens through the conv receptive field.
        active = torch.zeros(
            obs.shape[0], self.grid * self.grid, dtype=torch.bool, device=obs.device
        )
        active.scatter_(1, context_index, True)
        local = sparse_stem_tokens(
            self.online_encoder, obs, active.reshape(-1, self.grid, self.grid)
        )
        local = local.gather(
            1, context_index[..., None].expand(-1, -1, local.shape[-1])
        )
        return self.online_encoder.mix(local)

    def loss(self, obs: Tensor, generator) -> Tensor:
        context_index, pred_indices = sample_ijepa_masks(
            obs.shape[0], self.grid, generator
        )
        device = obs.device
        context_index = context_index.to(device)
        context = self._encode_context(obs, context_index)
        with torch.no_grad():
            full = self.target_encoder(obs)
            targets = F.layer_norm(full, (full.shape[-1],))[:, self.cfg.registers:]
        total = obs.new_zeros(())
        for pred_index in pred_indices:
            pred_index = pred_index.to(device)
            prediction = self.predictor(context, context_index, pred_index)
            block_target = targets.gather(
                1, pred_index[..., None].expand(-1, -1, targets.shape[-1])
            )
            total = total + F.smooth_l1_loss(prediction, block_target)
        prediction_loss = total / len(pred_indices)
        if self.sigreg_weight == 0.0:
            return prediction_loss
        # Dense online pass so mask randomness cannot satisfy the regularizer
        # (re-audit correction) and gradients shape the whole token map.
        dense = self.online_encoder(obs)
        sigreg_loss = self.sigreg(dense)
        return (
            (1.0 - self.sigreg_weight) * prediction_loss
            + self.sigreg_weight * sigreg_loss
        )

    @torch.no_grad()
    def update_target(self):
        self.target_encoder.update(self.online_encoder)
