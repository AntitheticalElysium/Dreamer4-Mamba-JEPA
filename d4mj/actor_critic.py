import torch
import torch.nn.functional as F
from torch import Tensor

from .agent import twohot
from .config import Config
from .imagination import Trajectory


def lambda_returns(trajectory: Trajectory, config: Config) -> Tensor:
    """G_t = r_{t+1} + gamma c_{t+1} [(1-lam) v_{t+1} + lam G_{t+1}], G_T = v_T.
    Eq. 10's printed same-index form shifts every critic target by one step (S6)."""
    horizon = trajectory.reward.shape[1]
    returns = [trajectory.value[:, -1]]
    for step in reversed(range(horizon)):
        bootstrap = (1 - config.lam) * trajectory.value[:, step + 1] + config.lam * returns[-1]
        discounted = config.gamma * trajectory.continuation[:, step] * bootstrap
        returns.append(trajectory.reward[:, step] + discounted)
    return torch.stack(returns[:0:-1], dim=1)


def actor_loss(trajectory: Trajectory, returns: Tensor, prior_logits: Tensor, config: Config) -> Tensor:
    """PMPO on the sign of the advantage, plus a reverse KL to the frozen prior.
    Magnitude is discarded by construction. The prior enters as log-probabilities:
    `log` of a softmax underflows to -inf and NaNs the whole loss."""
    advantage = (returns - trajectory.value[:, :-1]).detach()
    log_prob = F.log_softmax(trajectory.logits, dim=-1).gather(
        -1, trajectory.action[..., None]
    ).squeeze(-1)

    positive, negative = advantage >= 0, advantage < 0
    gain = config.pmpo_alpha * _masked_mean(log_prob, positive)
    loss = (1 - config.pmpo_alpha) * _masked_mean(log_prob, negative)

    log_prior = F.log_softmax(prior_logits, dim=-1)
    kl = F.softmax(trajectory.logits, -1) * (F.log_softmax(trajectory.logits, -1) - log_prior)
    return loss - gain + config.prior_beta * kl.sum(-1).mean()


def critic_loss(logits: Tensor, returns: Tensor, centers: Tensor) -> Tensor:
    target = twohot(returns.sign() * torch.log1p(returns.abs()), centers).detach()
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Zero when the set is empty, rather than a division by zero that surfaces as
    a silent NaN in the actor gradient."""
    return (values * mask).sum() / mask.sum().clamp(min=1)
