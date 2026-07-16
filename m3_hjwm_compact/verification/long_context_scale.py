"""Step-4b long-context and temporal-scale controls.

Protocol: reviews/2026-07-16-long-context-scale-protocol.md.

The large adapters intentionally preserve the selected pooled-global topology.
They test temporal scale without silently changing spatial tokenization.  They
are verification arms, not new defaults and not DRAMA reproductions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import (  # noqa: E402
    GlobalGRUTemporal,
    GlobalMambaTemporal,
    TemporalState,
    cosine_distance,
)
from consolidation import build_world  # noqa: E402
from fork_oracle_v2 import encode  # noqa: E402


LONG_CONTEXT = 128
LARGE_MAMBA_WIDTH = 512
LARGE_MAMBA_DEPTH = 2
LARGE_MAMBA_DSTATE = 64
LARGE_MAMBA_HEADDIM = 64
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")


class ProjectedGlobalGRUTemporal(nn.Module):
    """Large pooled-global GRU control with a projected hidden channel."""

    def __init__(self, dim: int = 64, hidden: int = 524, depth: int = 2):
        super().__init__()
        self.dim, self.hidden, self.depth = dim, hidden, depth
        self.in_proj = nn.Linear(dim, hidden)
        self.cells = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        cache = [torch.zeros(batch, self.hidden, device=device, dtype=dtype)
                 for _ in self.cells]
        output = torch.zeros(batch, streams, self.dim, device=device, dtype=dtype)
        return TemporalState(cache, output)

    def step(self, x: Tensor, state: TemporalState,
             reset: Tensor | None = None):
        y = self.in_proj(x.mean(1))
        old = list(state.cache)
        if reset is not None:
            keep = (~reset.bool())[:, None].to(y.dtype)
            old = [h * keep for h in old]
        new = []
        for cell, norm, h in zip(self.cells, self.norms, old):
            h = cell(y, h)
            y = norm(h)
            new.append(h)
        y = self.final_norm(y)
        out = x + self.out_proj(y)[:, None]
        return out, TemporalState(new, out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        state = self.init_state(x.shape[0], x.shape[2], x.device, x.dtype)
        outputs = []
        for index in range(x.shape[1]):
            output, state = self.step(
                x[:, index], state,
                None if resets is None else resets[:, index])
            outputs.append(output)
        return torch.stack(outputs, 1), state


class ProjectedGlobalMambaTemporal(nn.Module):
    """Width-512/depth-2 official Mamba-2 pooled-global scale arm."""

    def __init__(self, dim: int = 64, hidden: int = LARGE_MAMBA_WIDTH,
                 depth: int = LARGE_MAMBA_DEPTH,
                 d_state: int = LARGE_MAMBA_DSTATE,
                 headdim: int = LARGE_MAMBA_HEADDIM):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.dim, self.hidden, self.depth = dim, hidden, depth
        self.in_proj = nn.Linear(dim, hidden)
        self.layers = nn.ModuleList([
            Mamba2(d_model=hidden, d_state=d_state, headdim=headdim,
                   use_mem_eff_path=False)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.layers])
        # Official complete Mamba stacks apply a final norm after residual blocks.
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        caches = [
            tuple(layer.allocate_inference_cache(
                batch, max_seqlen=1, device=device, dtype=dtype))
            for layer in self.layers
        ]
        output = torch.zeros(batch, streams, self.dim, device=device, dtype=dtype)
        return TemporalState(caches, output)

    def step(self, x: Tensor, state: TemporalState,
             reset: Tensor | None = None):
        if state.cache is None:
            raise RuntimeError("large Mamba recurrent step requires official caches")
        if reset is not None:
            rows = reset.bool()
            for cache in state.cache:
                for tensor in cache:
                    tensor[rows] = 0
        y = self.in_proj(x.mean(1))
        next_caches = []
        for layer, norm, cache in zip(self.layers, self.norms, state.cache):
            update, *next_cache = layer.step(norm(y)[:, None], *cache)
            y = y + update[:, 0]
            next_caches.append(tuple(next_cache))
        y = self.final_norm(y)
        out = x + self.out_proj(y)[:, None]
        return out, TemporalState(next_caches, out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        if resets is not None and bool(resets[:, 1:].any()):
            raise NotImplementedError("segment sequences at episode boundaries")
        y = self.in_proj(x.mean(2))
        for layer, norm in zip(self.layers, self.norms):
            y = y + layer(norm(y))
        y = self.final_norm(y)
        out = x + self.out_proj(y)[:, :, None]
        return out, TemporalState(None, out[:, -1])


def temporal_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def projected_gru_parameter_count(dim: int, hidden: int, depth: int) -> int:
    """Exact parameter count for :class:`ProjectedGlobalGRUTemporal`.

    Keeping this algebraic avoids constructing hundreds of multi-million
    parameter candidates each time a paired world is built.
    """
    projections = (dim * hidden + hidden) + (hidden * dim + dim)
    cells = depth * (6 * hidden * hidden + 6 * hidden)
    norms = 2 * hidden * (depth + 1)
    return projections + cells + norms


def matched_gru_hidden(target_parameters: int, dim: int = 64,
                       depth: int = 2, low: int = 64,
                       high: int = 768) -> int:
    """Mechanically select the closest integer GRU width before training."""
    return min(
        range(low, high + 1),
        key=lambda hidden: (
            abs(projected_gru_parameter_count(dim, hidden, depth)
                - target_parameters),
            hidden,
        ),
    )


def large_temporal_pair() -> tuple[ProjectedGlobalGRUTemporal,
                                   ProjectedGlobalMambaTemporal]:
    mamba = ProjectedGlobalMambaTemporal()
    hidden = matched_gru_hidden(temporal_parameter_count(mamba))
    gru = ProjectedGlobalGRUTemporal(hidden=hidden)
    return gru, mamba


def build_long_world(arm: str, seed: int, reference_shared: dict[str, Tensor],
                     device) -> nn.Module:
    """Build one separately named arm with identical non-temporal state."""
    torch.manual_seed(seed)
    # Always construct the same base first so large-arm RNG position is paired.
    world = build_world("global_gru", 64, device)
    if arm == "LS-G64":
        world.temporal.name = "long_small_gru64"
    elif arm == "LS-M64":
        world.temporal.impl = GlobalMambaTemporal(world.cfg).to(device)
        world.temporal.name = "long_small_mamba2"
    elif arm == "LL-G":
        gru, _ = large_temporal_pair()
        world.temporal.impl = gru.to(device)
        world.temporal.name = "long_large_gru"
    elif arm == "LL-M":
        world.temporal.impl = ProjectedGlobalMambaTemporal().to(device)
        world.temporal.name = "long_large_mamba2"
    else:
        raise ValueError(f"unknown long-context arm {arm}")

    expected = {
        "LS-G64": GlobalGRUTemporal,
        "LS-M64": GlobalMambaTemporal,
        "LL-G": ProjectedGlobalGRUTemporal,
        "LL-M": ProjectedGlobalMambaTemporal,
    }[arm]
    if type(world.temporal.impl) is not expected:
        raise RuntimeError(
            f"silent temporal fallback for {arm}: "
            f"{type(world.temporal.impl).__name__} != {expected.__name__}")

    state = world.state_dict()
    for name, tensor in reference_shared.items():
        if name.startswith("temporal."):
            continue
        if name not in state or state[name].shape != tensor.shape:
            raise RuntimeError(f"shared state mismatch for {arm}: {name}")
        state[name] = tensor.to(device=device, dtype=state[name].dtype)
    world.load_state_dict(state, strict=True)
    return world


@torch.no_grad()
def long_openloop_anchor(world, anchor: dict, suffix_actions, device) -> Tensor:
    """Observe the complete stored prefix, then imagine the 8-step suffix."""
    obs = torch.from_numpy(anchor["obs_hist"][None]).to(device)
    actions = anchor["act_hist"]
    if len(actions) != len(anchor["obs_hist"]):
        raise ValueError("long-prefix obs/action history lengths must match")
    state = world.initial_state(1, device)
    for index in range(len(actions)):
        previous = torch.tensor([int(actions[index])], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = world.observe_step(obs[:, index], previous, state)
    predictions = []
    for action_value in suffix_actions:
        action = torch.tensor([int(action_value)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state, _, _, prediction = world.imagine_step(
                state, action, deterministic_mode=True)
        predictions.append(prediction.selected[0].float().cpu())
    return torch.stack(predictions)


def _retrievals(distance: np.ndarray) -> tuple[float, float]:
    legacy, fractional = [], []
    for row in range(distance.shape[0]):
        values = distance[row]
        legacy.append(float(np.argmin(values) == row))
        winners = np.flatnonzero(np.isclose(
            values, values.min(), atol=1e-12, rtol=0.0))
        fractional.append(1.0 / len(winners) if row in winners else 0.0)
    return float(np.mean(legacy)), float(np.mean(fractional))


@torch.no_grad()
def evaluate_long_bundle(world, encoder, anchors: list[dict], device) -> list[dict]:
    """Symmetric suffix matrices and per-horizon long-prefix metrics."""
    rows = []
    registers = world.cfg.registers
    for anchor_index, anchor in enumerate(anchors):
        targets = []
        for name in SUFFIX_NAMES:
            tokens = encode(encoder, anchor["branches"][name]["frames"], device)
            targets.append(F.normalize(tokens.float(), dim=-1).mean(0))
        targets = torch.stack(targets)  # [4,K,S,D]
        predictions = torch.stack([
            long_openloop_anchor(world, anchor, anchor["suffixes"][name], device)
            for name in SUFFIX_NAMES
        ])  # [4,K,S,D]

        separation_all, separation_patch = [], []
        retrieval_legacy, retrieval_tie = [], []
        for horizon in range(predictions.shape[1]):
            all_distance = np.zeros((4, 4), dtype=np.float64)
            patch_distance = np.zeros((4, 4), dtype=np.float64)
            for predicted_suffix in range(4):
                for target_suffix in range(4):
                    distance = cosine_distance(
                        predictions[predicted_suffix, horizon],
                        targets[target_suffix, horizon])
                    all_distance[predicted_suffix, target_suffix] = float(distance.mean())
                    patch_distance[predicted_suffix, target_suffix] = \
                        float(distance[registers:].mean())
            off = ~np.eye(4, dtype=bool)
            separation_all.append(float(
                all_distance[off].mean() - np.diag(all_distance).mean()))
            separation_patch.append(float(
                patch_distance[off].mean() - np.diag(patch_distance).mean()))
            legacy, fractional = _retrievals(all_distance)
            retrieval_legacy.append(legacy)
            retrieval_tie.append(fractional)

        true_frames = anchor["branches"]["true"]["frames"][:, -1]
        true_outcomes = anchor["branches"]["true"]["outcomes"]
        pixel_effective = any(
            not np.array_equal(branch["frames"][:, -1], true_frames)
            for name, branch in anchor["branches"].items() if name != "true")
        task_effective = any(
            branch["outcomes"] != true_outcomes
            for name, branch in anchor["branches"].items() if name != "true")
        rows.append({
            "anchor": anchor_index,
            "env_seed": int(anchor["env_seed"]),
            "night": bool(anchor["night"]),
            "pixel_effective": bool(pixel_effective),
            "task_effective": bool(task_effective),
            "separation_all_k": separation_all,
            "separation_patch_k": separation_patch,
            "retrieval_legacy_k": retrieval_legacy,
            "retrieval_tie_k": retrieval_tie,
        })
    return rows


def summarize_long_rows(rows: list[dict]) -> dict:
    def mean_at(key: str, index: int = -1, subset=None):
        selected = rows if subset is None else [row for row in rows if subset(row)]
        return float(np.mean([row[key][index] for row in selected])) if selected else None

    per_env = {}
    for env_seed in sorted({row["env_seed"] for row in rows}):
        subset = [row for row in rows if row["env_seed"] == env_seed]
        per_env[str(env_seed)] = {
            "separation_all": float(np.mean([r["separation_all_k"][-1] for r in subset])),
            "separation_patch": float(np.mean([r["separation_patch_k"][-1] for r in subset])),
            "retrieval_tie": float(np.mean([r["retrieval_tie_k"][-1] for r in subset])),
            "retrieval_legacy": float(np.mean([r["retrieval_legacy_k"][-1] for r in subset])),
        }
    return {
        "n_anchors": len(rows),
        "separation_all": mean_at("separation_all_k"),
        "separation_patch": mean_at("separation_patch_k"),
        "retrieval_tie": mean_at("retrieval_tie_k"),
        "retrieval_legacy": mean_at("retrieval_legacy_k"),
        "separation_all_curve": [mean_at("separation_all_k", k) for k in range(8)],
        "retrieval_tie_curve": [mean_at("retrieval_tie_k", k) for k in range(8)],
        "day_separation": mean_at("separation_all_k", subset=lambda r: not r["night"]),
        "night_separation": mean_at("separation_all_k", subset=lambda r: r["night"]),
        "pixel_effective_separation": mean_at(
            "separation_all_k", subset=lambda r: r["pixel_effective"]),
        "task_effective_separation": mean_at(
            "separation_all_k", subset=lambda r: r["task_effective"]),
        "per_env_seed": per_env,
    }
