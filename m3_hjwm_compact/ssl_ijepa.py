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
    EMARepresentationEncoder,
    ModelConfig,
    RepresentationEncoder,
    SpatialBlock,
)


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


class IJEPAPretrainer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.grid = cfg.image_size // cfg.patch_size
        self.online_encoder = RepresentationEncoder(cfg)
        self.target_encoder = EMARepresentationEncoder(self.online_encoder, cfg.ema_decay)
        self.predictor = IJEPAPredictor(cfg)

    def loss(self, obs: Tensor, generator) -> Tensor:
        context_index, pred_indices = sample_ijepa_masks(
            obs.shape[0], self.grid, generator
        )
        device = obs.device
        context_index = context_index.to(device)
        context = self.online_encoder(obs, visible_index=context_index)
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
        return total / len(pred_indices)

    @torch.no_grad()
    def update_target(self):
        self.target_encoder.update(self.online_encoder)
