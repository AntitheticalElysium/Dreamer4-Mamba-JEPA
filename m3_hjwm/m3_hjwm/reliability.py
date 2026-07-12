from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .utils import RMSNorm


@dataclass
class ReliabilitySignals:
    mode_dispersion: Tensor        # [B]
    energy: Tensor                 # [B]
    manifold_residual: Tensor      # [B]
    value_disagreement: Tensor     # [B]

    def stack(self) -> Tensor:
        return torch.stack(
            [self.mode_dispersion, self.energy, self.manifold_residual, self.value_disagreement],
            dim=-1,
        )


class TargetManifoldProjector(nn.Module):
    """Denoising projector trained only on real EMA target features.

    Its residual on imagined features is a JEPA-native off-manifold signal. The
    projector should be delayed/frozen when used for gating so it cannot co-adapt.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_loss(self, real_target: Tensor, noise_std: float = 0.05) -> Tensor:
        noisy = real_target + noise_std * torch.randn_like(real_target)
        return F.mse_loss(self(noisy), real_target.detach())


class CompatibilityEnergy(nn.Module):
    """Scores context/action/future compatibility.

    Train positives from real transitions and negatives from shuffled targets.
    """
    def __init__(self, dim: int, action_dim: int):
        super().__init__()
        self.action = nn.Embedding(action_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1)
        )

    def forward(self, context: Tensor, action: Tensor, future: Tensor) -> Tensor:
        c = context.mean(1)
        f = future.mean(1)
        a = self.action(action)
        return self.net(torch.cat([c, a, f], -1)).squeeze(-1)

    def contrastive_loss(self, context: Tensor, action: Tensor, real_future: Tensor) -> Tensor:
        pos = self(context, action, real_future)
        neg = self(context, action, real_future.roll(1, 0))
        # Low energy for positives, high energy for negatives.
        return F.softplus(pos).mean() + F.softplus(-neg).mean()


class ReliabilityPredictor(nn.Module):
    """Predicts actual held-out rollout error from shadow signals.

    It is not permitted to change actor/critic losses until calibration has been
    measured on held-out replay data.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, signals: ReliabilitySignals) -> Tensor:
        return F.softplus(self.net(signals.stack()).squeeze(-1))

    @staticmethod
    def confidence(predicted_error: Tensor, temperature: float = 1.0) -> Tensor:
        return torch.exp(-predicted_error / max(temperature, 1e-6))


def mode_dispersion(all_modes: Tensor | None) -> Tensor:
    if all_modes is None:
        raise ValueError("mode dispersion requires a mixture predictor")
    mean = all_modes.mean(1, keepdim=True)
    return ((all_modes - mean) ** 2).mean((1, 2, 3))
