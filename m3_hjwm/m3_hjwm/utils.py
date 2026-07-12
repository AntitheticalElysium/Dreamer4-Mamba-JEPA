from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(target: Tensor, bins: int, low: float, high: float) -> Tensor:
    """Two-hot encode symlog targets on a uniformly spaced support."""
    target = symlog(target).clamp(low, high)
    pos = (target - low) / (high - low) * (bins - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    out = torch.zeros(*target.shape, bins, device=target.device, dtype=target.dtype)
    out.scatter_add_(-1, lo.unsqueeze(-1), (hi.float() - pos + (hi == lo)).unsqueeze(-1))
    out.scatter_add_(-1, hi.unsqueeze(-1), (pos - lo.float()).unsqueeze(-1))
    return out


def decode_two_hot(logits: Tensor, low: float, high: float) -> Tensor:
    probs = logits.softmax(-1)
    support = torch.linspace(low, high, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return symexp((probs * support).sum(-1))


def cosine_distance(pred: Tensor, target: Tensor) -> Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return 1.0 - (pred * target).sum(-1)


def effective_rank(tokens: Tensor, eps: float = 1e-8) -> Tensor:
    """Effective rank of flattened features. Diagnostic only."""
    x = tokens.reshape(-1, tokens.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / (s.sum() + eps)
    return torch.exp(-(p * (p + eps).log()).sum())


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    src = dict(source.named_parameters())
    for name, p_tgt in target.named_parameters():
        p_tgt.mul_(decay).add_(src[name], alpha=1.0 - decay)
    src_b = dict(source.named_buffers())
    for name, b_tgt in target.named_buffers():
        if name in src_b:
            b_tgt.copy_(src_b[name])


def reset_where(x: Tensor, reset: Tensor) -> Tensor:
    """Zero state elements where reset is true. reset shape: [B]."""
    shape = [reset.shape[0]] + [1] * (x.ndim - 1)
    return x * (~reset.bool()).view(*shape).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def assert_finite(named: Iterable[tuple[str, Tensor]]) -> None:
    for name, x in named:
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite tensor: {name}")
