from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .utils import RMSNorm, two_hot, decode_two_hot


class RewardHead(nn.Module):
    def __init__(self, dim: int, bins: int, low: float, high: float):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, bins))
        self.bins, self.low, self.high = bins, low, high

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)

    def loss(self, logits: Tensor, reward: Tensor) -> Tensor:
        target = two_hot(reward, self.bins, self.low, self.high)
        return -(target * logits.log_softmax(-1)).sum(-1)

    def decode(self, logits: Tensor) -> Tensor:
        return decode_two_hot(logits, self.low, self.high)


class ContinueHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, 1))

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).squeeze(-1)

    def loss(self, logits: Tensor, continuation: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, continuation.float(), reduction="none")
