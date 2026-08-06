from __future__ import annotations
import torch
from torch import Tensor, nn
from .utils import RMSNorm


class SpatialBlock(nn.Module):
    """Global attention is affordable at 68 tokens on Crafter.

    The interface deliberately permits a later window-attention replacement without
    altering the world-model state or training loop.
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class SpatialMixer(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, registers: int):
        super().__init__()
        self.registers = nn.Parameter(torch.randn(1, registers, dim) * 0.02)
        self.blocks = nn.ModuleList([SpatialBlock(dim, heads) for _ in range(depth)])
        self.register_count = registers

    def forward(self, local_tokens: Tensor) -> Tensor:
        regs = self.registers.expand(local_tokens.shape[0], -1, -1)
        x = torch.cat([regs, local_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return x
