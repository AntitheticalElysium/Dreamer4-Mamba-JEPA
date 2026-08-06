from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .utils import RMSNorm


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int, depth: int = 3):
        super().__init__()
        layers = []
        d = inp
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), RMSNorm(hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, value_ensemble: int = 3):
        super().__init__()
        self.actor = MLP(state_dim, 256, action_dim)
        self.critics = nn.ModuleList([MLP(state_dim, 512, 1) for _ in range(value_ensemble)])
        self.action_dim = action_dim

    def policy(self, state: Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(state))

    def value_members(self, state: Tensor) -> Tensor:
        return torch.stack([c(state).squeeze(-1) for c in self.critics], dim=0)

    def value(self, state: Tensor) -> Tensor:
        return self.value_members(state).mean(0)

    def sample(self, state: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.policy(state)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()
