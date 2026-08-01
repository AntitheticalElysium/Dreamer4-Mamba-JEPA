import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import Config
from .state import Memory
from .time_mixer import time_mixer

LATENT, IMAGE, ACTION, CONDITION, SPATIAL, REGISTER, AGENT = range(7)


@dataclass(frozen=True)
class Layout:
    """Token segments within one timestep, in fixed order. Agent slots are present
    from Phase 1B and sit last, so S, the mask shape, the stream count and every
    state shape are fixed once for every phase.
    """

    segments: tuple[tuple[int, int], ...]

    @staticmethod
    def encoder(config: Config) -> "Layout":
        return Layout(((LATENT, config.n_latents), (IMAGE, config.n_patches)))

    @staticmethod
    def dynamics(config: Config) -> "Layout":
        return Layout(
            (
                (ACTION, 1),
                (CONDITION, 1),
                (SPATIAL, config.n_spatial),
                (REGISTER, config.n_register),
                (AGENT, config.n_agent),
            )
        )

    @property
    def size(self) -> int:
        return sum(count for _, count in self.segments)

    @property
    def kinds(self) -> Tensor:
        return torch.cat([torch.full((count,), kind) for kind, count in self.segments])

    def span(self, kind: int) -> slice:
        start = 0
        for other, count in self.segments:
            if other == kind:
                return slice(start, start + count)
            start += count
        raise KeyError(kind)


def space_mask(layout: Layout, mode: str, agent_active: bool = True) -> Tensor:
    """Within-timestep attention. `dynamics` is the one-way agent firewall: agent
    queries read everything, and nothing reads agent keys, so world features are
    agent-free by induction over depth and need no partition of the memory.
    """
    kinds = layout.kinds
    query, key = kinds[:, None], kinds[None, :]

    if mode == "encoder":
        return torch.where(query == LATENT, torch.ones_like(query, dtype=torch.bool), query == key)
    if mode == "decoder":
        return torch.where(query == LATENT, key == LATENT, (query == key) | (key == LATENT))
    if mode == "dynamics":
        agent_query, agent_key = query == AGENT, key == AGENT
        if not agent_active:
            return ~agent_query & ~agent_key
        return torch.where(agent_query, torch.ones_like(agent_key), ~agent_key)
    raise ValueError(mode)


def rope(x: Tensor, offset: int = 0, base: float = 10000.0) -> Tensor:
    """Rotary positions over the axis before the head dimension. `offset` is the
    cached prefix length, without which a decoded step re-uses position zero."""
    length, dim = x.shape[-2], x.shape[-1]
    freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    steps = torch.arange(offset, offset + length, device=x.device, dtype=torch.float32)
    angle = torch.outer(steps, freq)
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """QKNorm as cosine attention with a learned per-head temperature.

    The temperature is clamped, as the pinned source clamps it. S14 drops the
    paper's attention-logit soft capping as a 2B-parameter stability trick, and
    that decision was taken alone: without the clamp too, nothing at all would
    bound the logits, and a runaway temperature saturates attention to one-hot with
    no diagnostic. GQA is also dropped -- Table 2 adopts it for KV bandwidth at a
    cost of one FVD point.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.full((n_heads,), math.log(math.sqrt(self.head_dim))))

    def forward(self, x, mask=None, causal=False, cache=None, offset=0, limit=None):
        """`offset` is the absolute position of the first new token, supplied by the
        caller rather than inferred from cache length -- once the cache is bounded
        those two differ, and RoPE reading the wrong one silently re-dates history.
        """
        n, length, _ = x.shape
        q, k, v = self.qkv(x).view(n, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rope(q, offset), F.normalize(rope(k, offset), dim=-1)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
            mask, causal = _decode_mask(length, k.shape[2], limit, x.device), False
        elif limit is not None and length > limit:
            mask, causal = _decode_mask(length, length, limit, x.device), False
        scale = self.log_scale.exp().clamp(max=100.0).view(1, -1, 1, 1)
        q = F.normalize(q, dim=-1) * scale * math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        if limit is not None:
            k, v = k[:, :, -limit:], v[:, :, -limit:]
        return self.out(y.transpose(1, 2).reshape(n, length, -1)), (k, v)


def _decode_mask(new: int, total: int, limit: int | None, device) -> Tensor:
    """Causal, and window-bounded when a limit is set.

    A cache truncated to `limit` while a single scan attends over everything makes
    the batched and stepped paths compute different latents for the same frame --
    the bound has to hold in both or it is not part of Z*.
    """
    query = torch.arange(total - new, total, device=device)[:, None]
    key = torch.arange(total, device=device)[None, :]
    allowed = query >= key
    return allowed if limit is None else allowed & (query - key < limit)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ratio: float):
        super().__init__()
        hidden = int(d_model * ratio)
        self.gate = nn.Linear(d_model, 2 * hidden)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.gate(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


class Block(nn.Module):
    """Pre-norm space, optional time, then MLP -- the source layer order. Time
    mixing appears every `time_every` layers and is the only place the two arms
    differ.
    """

    def __init__(self, config: Config, mask: Tensor, index: int, d_model: int, n_heads: int, context: int | None):
        super().__init__()
        self.mixes_time = (index + 1) % config.time_every == 0
        self.register_buffer("mask", mask, persistent=False)
        self.norm_space = nn.RMSNorm(d_model)
        self.space = Attention(d_model, n_heads)
        self.norm_mlp = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, config.mlp_ratio)
        if self.mixes_time:
            self.norm_time = nn.RMSNorm(d_model)
            self.time = time_mixer(config, d_model, context)

    def forward(self, x: Tensor, memory: tuple[Tensor, Tensor] | None, offset: int = 0):
        b, t, s, d = x.shape
        spaced, _ = self.space(self.norm_space(x).reshape(b * t, s, d), mask=self.mask)
        x = x + spaced.view(b, t, s, d)
        if self.mixes_time:
            streams = self.norm_time(x).transpose(1, 2).reshape(b * s, t, d)
            mixed, memory = self.time(streams, memory, offset)
            x = x + mixed.view(b, s, t, d).transpose(1, 2)
        return x + self.mlp(self.norm_mlp(x)), memory


class Backbone(nn.Module):
    def __init__(self, config: Config, layout: Layout, mode: str, d_model: int, n_heads: int, depth: int, context: int | None = None):
        super().__init__()
        mask = space_mask(layout, mode, agent_active=True)
        self.checkpointing = config.gradient_checkpointing
        self.blocks = nn.ModuleList(
            Block(config, mask, index, d_model, n_heads, context) for index in range(depth)
        )

    def forward(self, x: Tensor, memory: Memory | None = None, offset: int = 0) -> tuple[Tensor, Memory]:
        """`memory` holds one entry per *time-mixing* block, not per block, so it is
        consumed by an iterator rather than zipped against the full stack.

        With Mamba memory present exactly one block may be supplied: its recurrent
        step accepts one token. Attention may decode several, because `_decode_mask`
        keeps them causal against each other -- which is what lets the frozen
        encoder cache an episode in chunks rather than one dense call.

        Blocks are recomputed in the backward pass rather than stored, which is what
        makes the declared architecture fit 6 GB at all: measured, Phase 1A goes from
        fitting only batch 1 at the short length to fitting batch 8 short and batch 4
        long. It is exact -- the same values, computed twice -- so it changes cost,
        not results. It is skipped whenever there is nothing to recompute for: with
        gradients off, and on the cached/recurrent path, where `memory` is carried
        and re-entering a block would discard the state it returned.
        """
        recurrent = any(type(b.time).__name__ == "TimeMamba" for b in self.blocks if b.mixes_time)
        assert memory is None or x.shape[1] == 1 or not recurrent, (
            "Mamba decodes one block at a time; attention may decode several"
        )
        expected = sum(block.mixes_time for block in self.blocks)
        assert memory is None or len(memory) == expected, "memory does not match the time layers"
        carried = iter(memory if memory is not None else ())
        updated: list[tuple[Tensor, Tensor]] = []
        recompute = self.checkpointing and memory is None and torch.is_grad_enabled()
        for block in self.blocks:
            state = next(carried, None) if block.mixes_time else None
            if recompute:
                x, state = checkpoint(block, x, state, offset, use_reentrant=False)
            else:
                x, state = block(x, state, offset)
            if block.mixes_time:
                updated.append(state)
        return x, tuple(updated)
