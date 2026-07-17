"""2026-07-17 mechanism screen: which factor drives the full-grid effect?

Protocol: reviews/2026-07-17-mechanism-screen-protocol.md. EXPLORATORY,
companion-directed factor isolation (its HOLD condition on the fresh-seed
confirmation). The exploratory H-T arm moved capacity, flattening,
projections, mixing, and bypass together; these controls move ONE lever each
relative to either the pooled-64 baseline or the full-grid arm:

  MS-PC  capacity: pooled + bypass topology (step-4 shape) scaled to the
         full-grid arm's ~3.03M temporal parameters (ProjectedGlobalGRU,
         hidden matched mechanically pre-outcome).
  MS-FB  bypass: the exact FlattenedGRUTemporal with the dense residual
         bypass ADDED BACK (out = x + proj(state)); nothing else changes.
  MS-FF  recurrence: full-grid feedforward control — same stem/out
         projections, matched parameters, NO temporal state at all. If this
         matches the full-grid recurrent arm, the gain is the large learned
         projections/mixing, not temporal memory.

GRU-only: the backend contrast was at parity inside both topologies (step 4,
exploratory screen), so backend is not a material interaction for these
factor questions (recorded limitation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import TemporalState  # noqa: E402
from consolidation import build_world  # noqa: E402
from exploratory_topology import (  # noqa: E402
    FlattenedGRUTemporal, FlattenedMambaTemporal, flattened_gru_parameter_count,
    matched_flat_gru_hidden)
from long_context_scale import (  # noqa: E402
    ProjectedGlobalGRUTemporal, matched_gru_hidden)


class BypassFlattenedGRUTemporal(FlattenedGRUTemporal):
    """MS-FB: identical to FlattenedGRUTemporal except the dense residual
    bypass is restored (out = x + proj(state)) — the single-factor bypass
    control."""

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
        out = x + self.out_proj(self.final_norm(y)).reshape(b, s, d)   # + x
        return out, TemporalState(new, out)


class FullGridFeedforward(nn.Module):
    """MS-FF: same full-grid stem/out projections, parameter-matched, but NO
    recurrence — the per-step context is a pure function of the current
    tokens. Distinguishes 'temporal state matters' from 'the big learned
    projections/mixing matter'."""

    def __init__(self, dim: int, streams: int, hidden: int, depth: int = 2):
        super().__init__()
        self.dim, self.streams, self.hidden, self.depth = dim, streams, hidden, depth
        self.stem = nn.Linear(dim * streams, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.act = nn.GELU()
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim * streams)

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        assert streams == self.streams
        return TemporalState(
            [], torch.zeros(batch, streams, self.dim, device=device, dtype=dtype))

    def _apply_grid(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        y = self.stem(x.reshape(b, s * d))
        for layer, norm in zip(self.layers, self.norms):
            y = y + self.act(layer(norm(y)))
        return self.out_proj(self.final_norm(y)).reshape(b, s, d)

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        out = self._apply_grid(x)
        return out, TemporalState([], out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        b, t, s, d = x.shape
        out = self._apply_grid(x.reshape(b * t, s, d)).reshape(b, t, s, d)
        return out, TemporalState([], out[:, -1])


def feedforward_parameter_count(dim: int, streams: int, hidden: int,
                                depth: int = 2) -> int:
    flat = dim * streams
    stem = flat * hidden + hidden
    out = hidden * flat + flat
    layers = depth * (hidden * hidden + hidden)
    norms = 2 * hidden * (depth + 1)
    return stem + out + layers + norms


def matched_ff_hidden(target: int, dim: int, streams: int,
                      low: int = 64, high: int = 640) -> int:
    return min(range(low, high + 1),
               key=lambda h: (abs(feedforward_parameter_count(dim, streams, h)
                                  - target), h))


def build_mechanism_world(arm: str, seed: int, device):
    torch.manual_seed(seed)
    world = build_world("global_gru", 64, device)   # paired RNG base
    streams, dim = world.streams, world.cfg.token_dim
    flg_target = flattened_gru_parameter_count(
        dim, streams, matched_flat_gru_hidden(
            sum(p.numel() for p in FlattenedMambaTemporal(dim, streams).parameters()),
            dim, streams))
    if arm == "MS-PC":
        hidden = matched_gru_hidden(flg_target, dim=dim, depth=2, high=1024)
        world.temporal.impl = ProjectedGlobalGRUTemporal(
            dim=dim, hidden=hidden, depth=2).to(device)
        world.temporal.name = "mech_pooled_capacity"
        expected = ProjectedGlobalGRUTemporal
    elif arm == "MS-FB":
        flm_params = sum(p.numel() for p in
                         FlattenedMambaTemporal(dim, streams).parameters())
        hidden = matched_flat_gru_hidden(flm_params, dim, streams)
        world.temporal.impl = BypassFlattenedGRUTemporal(
            dim, streams, hidden).to(device)
        world.temporal.name = "mech_flattened_bypass"
        expected = BypassFlattenedGRUTemporal
    elif arm == "MS-FF":
        hidden = matched_ff_hidden(flg_target, dim, streams)
        world.temporal.impl = FullGridFeedforward(dim, streams, hidden).to(device)
        world.temporal.name = "mech_fullgrid_feedforward"
        expected = FullGridFeedforward
    else:
        raise ValueError(f"unknown mechanism arm {arm}")
    if type(world.temporal.impl) is not expected:
        raise RuntimeError(f"silent temporal fallback for {arm}")
    return world
