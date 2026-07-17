"""2026-07-17 exploratory screen: topology (flattened no-bypass state) and
predictor action-conditioning strength (AdaLN-zero).

Protocol: reviews/2026-07-17-exploratory-topology-protocol.md. EXPLORATORY:
run during user absence under the standing exploration authorization; no
defaults change; results license at most a registered confirmation.

Two labelled hypotheses from the 4b post-mortem and the literature round:

H-T (topology): the pooled bottleneck + dense residual bypass — not the
  backend — limits action-discriminative dynamics. Test: full-grid
  bottleneck / no-bypass cores (DRAMA-inspired input stem) where the
  recurrent state CARRIES the whole context, GRU vs Mamba-2 inside the
  identical adapter. (2026-07-17 companion audit: this arm moves capacity,
  mixing, projections, and bypass TOGETHER — see the mechanism screen for
  factor isolation; "H-T positive" licenses only "promising architecture
  family", not a causal pooling/bypass claim.)
H-C (conditioning): BYOL-AC says action-conditioned prediction selects
  action-distinguishing features in proportion to conditioning strength.
  Test: LeWM-faithful AdaLN-zero modulation added on top of the existing
  token conditioning (ConditionalBlock, lucas-maes__le-wm/module.py:88).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import (  # noqa: E402
    FuturePredictor, ModelConfig, Prediction, TemporalState)
from consolidation import build_world  # noqa: E402

FLAT_HIDDEN = 256
FLAT_DEPTH = 2
FLAT_DSTATE = 64
FLAT_HEADDIM = 64


# --------------------------------------------------------------------------
# H-T: flattened-latent, no-bypass temporal cores
# --------------------------------------------------------------------------

class FlattenedGRUTemporal(nn.Module):
    """Full-grid bottleneck / no-bypass JEPA ablation with a DRAMA-INSPIRED
    input stem (2026-07-17 companion relabel: NOT "RSSM-shaped" — CDP's RSSM
    updates a prior from the previous stochastic state before folding in
    observations, semantics this deterministic core does not reproduce; the
    only shared invariant with DRAMA is flattened-latent-through-stem,
    mixer_seq_simple.py:188). The FULL S*D token grid is flattened through a
    stem into the recurrent state and the per-token context is emitted PURELY
    from that state — no dense residual bypass. NOTE (companion): this arm
    changes capacity, mixing, projections, AND bypass together; the mechanism
    screen isolates those factors."""

    def __init__(self, dim: int, streams: int, hidden: int, depth: int = FLAT_DEPTH):
        super().__init__()
        self.dim, self.streams, self.hidden, self.depth = dim, streams, hidden, depth
        self.stem = nn.Linear(dim * streams, hidden)
        self.cells = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim * streams)

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        assert streams == self.streams, "flattened core is stream-count-specific"
        cache = [torch.zeros(batch, self.hidden, device=device, dtype=dtype)
                 for _ in self.cells]
        output = torch.zeros(batch, streams, self.dim, device=device, dtype=dtype)
        return TemporalState(cache, output)

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        b, s, d = x.shape
        y = self.stem(x.reshape(b, s * d))
        old = list(state.cache)
        if reset is not None:
            keep = (~reset.bool())[:, None].to(y.dtype)
            old = [h * keep for h in old]
        new = []
        for cell, norm, h in zip(self.cells, self.norms, old):
            h = cell(y, h)
            y = norm(h)
            new.append(h)
        out = self.out_proj(self.final_norm(y)).reshape(b, s, d)   # NO + x
        return out, TemporalState(new, out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        state = self.init_state(x.shape[0], x.shape[2], x.device, x.dtype)
        outputs = []
        for index in range(x.shape[1]):
            output, state = self.step(
                x[:, index], state, None if resets is None else resets[:, index])
            outputs.append(output)
        return torch.stack(outputs, 1), state


class FlattenedMambaTemporal(nn.Module):
    """Flattened no-bypass core with official Mamba-2 blocks (same external
    contract as FlattenedGRUTemporal; the H-T backend contrast changes only
    the recurrent operator)."""

    def __init__(self, dim: int, streams: int, hidden: int = FLAT_HIDDEN,
                 depth: int = FLAT_DEPTH, d_state: int = FLAT_DSTATE,
                 headdim: int = FLAT_HEADDIM):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2
        self.dim, self.streams, self.hidden, self.depth = dim, streams, hidden, depth
        self.stem = nn.Linear(dim * streams, hidden)
        self.layers = nn.ModuleList([
            Mamba2(d_model=hidden, d_state=d_state, headdim=headdim,
                   use_mem_eff_path=False)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.layers])
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim * streams)

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        assert streams == self.streams, "flattened core is stream-count-specific"
        caches = [
            tuple(layer.allocate_inference_cache(batch, max_seqlen=1,
                                                 device=device, dtype=dtype))
            for layer in self.layers
        ]
        output = torch.zeros(batch, streams, self.dim, device=device, dtype=dtype)
        return TemporalState(caches, output)

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        if state.cache is None:
            raise RuntimeError("flattened Mamba step requires official caches")
        b, s, d = x.shape
        if reset is not None:
            rows = reset.bool()
            for cache in state.cache:
                for tensor in cache:
                    tensor[rows] = 0
        y = self.stem(x.reshape(b, s * d))
        next_caches = []
        for layer, norm, cache in zip(self.layers, self.norms, state.cache):
            update, *next_cache = layer.step(norm(y)[:, None], *cache)
            y = y + update[:, 0]
            next_caches.append(tuple(next_cache))
        out = self.out_proj(self.final_norm(y)).reshape(b, s, d)   # NO + x
        return out, TemporalState(next_caches, out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        if resets is not None and bool(resets[:, 1:].any()):
            raise NotImplementedError("segment sequences at episode boundaries")
        b, t, s, d = x.shape
        y = self.stem(x.reshape(b, t, s * d))
        for layer, norm in zip(self.layers, self.norms):
            y = y + layer(norm(y))
        out = self.out_proj(self.final_norm(y)).reshape(b, t, s, d)
        return out, TemporalState(None, out[:, -1])


def flattened_gru_parameter_count(dim: int, streams: int, hidden: int,
                                  depth: int = FLAT_DEPTH) -> int:
    """Exact algebraic count for FlattenedGRUTemporal (no instantiation)."""
    flat = dim * streams
    stem = flat * hidden + hidden
    out = hidden * flat + flat
    cells = depth * (6 * hidden * hidden + 6 * hidden)
    norms = 2 * hidden * (depth + 1)
    return stem + out + cells + norms


def matched_flat_gru_hidden(target: int, dim: int, streams: int,
                            low: int = 64, high: int = 512) -> int:
    """Mechanically select FL-G width to match FL-M parameters (pre-outcome)."""
    return min(range(low, high + 1),
               key=lambda h: (abs(flattened_gru_parameter_count(dim, streams, h)
                                  - target), h))


# --------------------------------------------------------------------------
# H-C: AdaLN-zero conditioned predictor (LeWM ConditionalBlock, module.py:88)
# --------------------------------------------------------------------------

class ConditionalSpatialBlock(nn.Module):
    """LeWM/DiT AdaLN-zero block adapted to the compact spatial block. Norms
    lose affine parameters (modulation supplies shift/scale, source-faithful)
    and gates are zero-initialized, so residual branches start OFF."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(condition)[:, None].chunk(6, dim=-1))
        y = self.n1(x) * (1 + scale_msa) + shift_msa
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + gate_msa * y
        y = self.mlp(self.n2(x) * (1 + scale_mlp) + shift_mlp)
        return x + gate_mlp * y


class AdaLNFuturePredictor(FuturePredictor):
    """FuturePredictor with action/horizon AdaLN-zero modulation added on top
    of the existing conditioning tokens (BYOL-AC-motivated, LeWM-INSPIRED —
    an adapted block, not LeWM-faithful; 2026-07-17 companion correction:
    zero-init gates initially SUPPRESS the conditioned branches, so this does
    not 'strictly increase' conditioning strength at initialization)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList([
            ConditionalSpatialBlock(cfg.token_dim, cfg.spatial_heads)
            for _ in range(cfg.predictor_depth)
        ])

    def all_predictions(self, context: Tensor, action: Tensor, horizon: Tensor):
        b, s, d = context.shape
        x = context
        if s == self.streams:
            x = x + self.pos_embed
        x = x[:, None].expand(b, self.modes, s, d) + self.mode_embed[None, :, None]
        x = x.reshape(b * self.modes, s, d)
        action_embed = self.action(action)
        horizon_embed = self.horizon(horizon)
        conditioning = torch.stack([action_embed, horizon_embed], dim=1)
        conditioning = conditioning[:, None].expand(
            b, self.modes, 2, d).reshape(b * self.modes, 2, d)
        x = torch.cat([conditioning, x], dim=1)
        condition = (action_embed + horizon_embed)[:, None].expand(
            b, self.modes, d).reshape(b * self.modes, d)
        for block in self.blocks:
            x = block(x, condition)
        modes = self.out(self.norm(x[:, 2:])).reshape(b, self.modes, s, d)
        route_context = torch.cat(
            [context.mean(1), action_embed, horizon_embed], dim=-1)
        logits = self.router(route_context)
        return modes, logits


# --------------------------------------------------------------------------
# arm construction
# --------------------------------------------------------------------------

def build_exploratory_world(arm: str, seed: int, device):
    """Arms: X-FLG / X-FLM (topology), X-ADA (conditioning), each on the
    validated frozen encoder with the standard T=16 contract."""
    torch.manual_seed(seed)
    world = build_world("global_gru", 64, device)   # paired RNG base
    streams = world.streams
    dim = world.cfg.token_dim
    if arm == "X-FLM":
        world.temporal.impl = FlattenedMambaTemporal(dim, streams).to(device)
        world.temporal.name = "flattened_mamba2"
    elif arm == "X-FLG":
        mamba_params = sum(p.numel() for p in
                           FlattenedMambaTemporal(dim, streams).parameters())
        hidden = matched_flat_gru_hidden(mamba_params, dim, streams)
        world.temporal.impl = FlattenedGRUTemporal(dim, streams, hidden).to(device)
        world.temporal.name = "flattened_gru"
    elif arm == "X-ADA":
        world.future = AdaLNFuturePredictor(world.cfg).to(device)
        world.temporal.name = "global_gru64_adaln"
    elif arm == "X-BASE":
        world.temporal.name = "global_gru64"
    else:
        raise ValueError(f"unknown exploratory arm {arm}")
    expected = {"X-FLM": FlattenedMambaTemporal, "X-FLG": FlattenedGRUTemporal}
    if arm in expected and type(world.temporal.impl) is not expected[arm]:
        raise RuntimeError(f"silent temporal fallback for {arm}")
    if arm == "X-ADA" and type(world.future) is not AdaLNFuturePredictor:
        raise RuntimeError("silent predictor fallback for X-ADA")
    return world
