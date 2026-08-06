from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .utils import RMSNorm, cosine_distance


@dataclass
class PredictorOutput:
    prediction: Tensor             # [B,S,D], selected training prediction
    all_modes: Tensor | None       # [B,K,S,D]
    assignment: Tensor | None      # [B], hard target assignment
    mode_logits: Tensor | None     # [B,K]
    per_sample_loss: Tensor
    commitment_loss: Tensor
    balance_loss: Tensor


class PredictorBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class DeterministicPredictor(nn.Module):
    def __init__(self, dim: int, action_dim: int, depth: int, horizon_bins: int):
        super().__init__()
        self.action = nn.Embedding(action_dim, dim)
        self.horizon = nn.Embedding(horizon_bins, dim)
        self.blocks = nn.ModuleList([PredictorBlock(dim) for _ in range(depth)])
        self.out = nn.Linear(dim, dim)

    def _features(self, context: Tensor, action: Tensor, horizon: Tensor) -> Tensor:
        x = context + self.action(action)[:, None] + self.horizon(horizon)[:, None]
        for block in self.blocks:
            x = block(x)
        return self.out(x)

    def forward(self, context: Tensor, action: Tensor, target: Tensor | None, horizon: Tensor) -> PredictorOutput:
        pred = self._features(context, action, horizon)
        loss = torch.zeros(context.shape[0], device=context.device)
        if target is not None:
            loss = cosine_distance(pred, target).mean(-1)
        zero = loss.new_zeros(())
        return PredictorOutput(pred, None, None, None, loss, zero, zero)

    def sample(self, context: Tensor, action: Tensor, horizon: Tensor) -> Tensor:
        return self._features(context, action, horizon)


class HardModeMixturePredictor(nn.Module):
    """Single-pass multimodal JEPA predictor.

    Training assigns each observed target to the closest mode (hard EM / vector
    quantisation style). Unlike the old RPF heads:
      - one mode is selected consistently for the full next state;
      - heads are explicitly trained to partition successor modes;
      - mode probabilities are learned from context;
      - disagreement is not automatically treated as epistemic uncertainty.
    """
    def __init__(self, dim: int, action_dim: int, depth: int, horizon_bins: int, modes: int):
        super().__init__()
        self.modes = modes
        self.action = nn.Embedding(action_dim, dim)
        self.horizon = nn.Embedding(horizon_bins, dim)
        self.mode_embed = nn.Parameter(torch.randn(modes, dim) * 0.02)
        self.blocks = nn.ModuleList([PredictorBlock(dim) for _ in range(depth)])
        self.out = nn.Linear(dim, dim)
        self.router = nn.Sequential(RMSNorm(dim), nn.Linear(dim, modes))

    def _all_modes(self, context: Tensor, action: Tensor, horizon: Tensor) -> tuple[Tensor, Tensor]:
        b, s, d = context.shape
        base = context + self.action(action)[:, None] + self.horizon(horizon)[:, None]
        x = base[:, None] + self.mode_embed[None, :, None, :]
        x = x.reshape(b * self.modes, s, d)
        for block in self.blocks:
            x = block(x)
        modes = self.out(x).reshape(b, self.modes, s, d)
        logits = self.router(context.mean(1))
        return modes, logits

    def forward(self, context: Tensor, action: Tensor, target: Tensor | None, horizon: Tensor) -> PredictorOutput:
        modes, logits = self._all_modes(context, action, horizon)
        b = context.shape[0]
        if target is None:
            idx = torch.distributions.Categorical(logits=logits).sample()
            pred = modes[torch.arange(b, device=context.device), idx]
            zero = context.new_zeros(())
            return PredictorOutput(pred, modes, idx, logits, context.new_zeros(b), zero, zero)

        distances = cosine_distance(modes, target[:, None]).mean(-1)  # [B,K], average token distances
        assignment = distances.argmin(-1)
        pred = modes[torch.arange(b, device=context.device), assignment]
        pred_loss = distances.gather(1, assignment[:, None]).squeeze(1)

        # Router learns to predict the hard mode without allowing router gradients to
        # alter the target assignment.
        commitment = F.cross_entropy(logits, assignment.detach())

        # Avoid dead modes. This is a weak usage prior, not a claim that modes are
        # uniformly probable in every state.
        usage = F.one_hot(assignment, self.modes).float().mean(0)
        uniform = torch.full_like(usage, 1.0 / self.modes)
        balance = ((usage - uniform) ** 2).mean()
        return PredictorOutput(pred, modes, assignment, logits, pred_loss, commitment, balance)

    def sample(self, context: Tensor, action: Tensor, horizon: Tensor, deterministic: bool = False) -> Tensor:
        modes, logits = self._all_modes(context, action, horizon)
        idx = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        return modes[torch.arange(context.shape[0], device=context.device), idx]
