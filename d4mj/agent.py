import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import SwiGLU
from .config import Config
from .data import Batch


class Heads(nn.Module):
    """Policy, reward, continuation and value, all reading the same pooled agent
    tokens through one output layer per multi-token distance.

    Every head uses the same pooling. Two heads over the same tokens with
    different poolers is a difference nothing declares and nothing measures.

    Policy and value share one body; reward and continuation share another. A
    single trunk would let Phase 3 move the reward model it is being scored
    against -- measured: one policy/value step changed both reward and
    continuation logits. Dreamer 4 freezes the world and reward model there and
    describes the heads as small MLPs, not a mandatory common trunk.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("centers", _centers(config), persistent=True)
        width, leads = config.d_model, config.mtp_leads
        self.actor_body = SwiGLU(width, 2.0)
        self.model_body = SwiGLU(width, 2.0)
        self.policy = nn.Linear(width, leads * config.n_actions)
        self.reward = nn.Linear(width, leads * config.bins)
        self.continuation = nn.Linear(width, leads)
        self.value = nn.Linear(width, config.bins)

    def actor_parameters(self):
        """What Phase 3 may move: policy, value and their shared body only."""
        return [*self.actor_body.parameters(), *self.policy.parameters(), *self.value.parameters()]

    def forward(self, agent: Tensor) -> dict[str, Tensor]:
        b, t = agent.shape[:2]
        pooled = agent.mean(dim=2)
        actor, model = self.actor_body(pooled), self.model_body(pooled)
        leads, config = self.config.mtp_leads, self.config
        return {
            "policy": self.policy(actor).view(b, t, leads, config.n_actions),
            "reward": self.reward(model).view(b, t, leads, config.bins),
            "continuation": self.continuation(model).view(b, t, leads),
            "value": self.value(actor),
        }


def twohot(values: Tensor, centers: Tensor) -> Tensor:
    """Linear interpolation between the two neighbouring bin centres, so a value
    between them is represented exactly and bin count sets grid density rather
    than a quantisation floor."""
    clamped = values.clamp(centers[0], centers[-1]).unsqueeze(-1)
    upper = torch.searchsorted(centers, clamped.contiguous()).clamp(1, len(centers) - 1)
    lower = upper - 1
    low, high = centers[lower], centers[upper]
    weight = ((clamped - low) / (high - low).clamp(min=1e-8)).squeeze(-1)
    target = torch.zeros(*values.shape, len(centers), device=values.device)
    return target.scatter_(-1, upper, weight.unsqueeze(-1)).scatter_(
        -1, lower, (1 - weight).unsqueeze(-1)
    )


def head_targets(batch: Batch, config: Config) -> dict[str, Tensor]:
    """Multi-token targets under the led-to convention.

    Padding past the window uses class 0, not the BOS index: BOS is an input
    embedding row, not a policy class. `action_valid` is what makes those entries
    inert, so the filler only has to be in range.

    Reward lead 0 at block t is the reward that *arrived* at t, so the reward
    caused by the action chosen at t is lead 0 at t+1 -- never lead 0 at t. Policy
    lead 0 is the outgoing action, which in led-to storage lives one block later.
    """
    leads = config.mtp_leads
    outgoing = _shift(batch.led_to_action.float(), fill=0.0)
    return {
        "action": _leads(outgoing, leads, fill=0.0),
        "reward": _leads(batch.reward, leads, fill=0.0),
        "continuation": _leads((~batch.terminated).float(), leads, fill=1.0),
        "valid": _leads(batch.valid.float(), leads, fill=0.0),
        "action_valid": _leads(_shift(batch.valid.float(), fill=0.0), leads, fill=0.0),
    }


def head_loss(
    predictions: dict[str, Tensor], targets: dict[str, Tensor], config: Config
) -> dict[str, Tensor]:
    """Returned per head, not summed: Dreamer 4 normalises every concurrent loss by
    its own running RMS, and merging them first lets whichever head has the largest
    natural scale set the others' effective weight."""
    centers = predictions["centers"]
    policy = F.cross_entropy(
        predictions["policy"].flatten(0, 2), targets["action"].flatten().long(), reduction="none"
    ).view_as(targets["action"])
    reward = _distribution_loss(predictions["reward"], targets["reward"], centers)
    continuation = F.binary_cross_entropy_with_logits(
        predictions["continuation"], targets["continuation"], reduction="none"
    )
    valid, actions = targets["valid"], targets["action_valid"]
    return {
        "policy": (policy * actions).sum() / actions.sum().clamp(min=1.0),
        "reward": (reward * valid).sum() / valid.sum().clamp(min=1.0),
        "continuation": (continuation * valid).sum() / valid.sum().clamp(min=1.0),
    }


def _distribution_loss(logits: Tensor, values: Tensor, centers: Tensor) -> Tensor:
    target = twohot(_symlog(values), centers)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


def _centers(config: Config) -> Tensor:
    half = torch.linspace(-config.symlog_limit, 0.0, config.bins // 2 + 1)
    return torch.cat([half, -half[:-1].flip(0)])


def _symlog(x: Tensor) -> Tensor:
    return x.sign() * torch.log1p(x.abs())


def _shift(values: Tensor, fill: float) -> Tensor:
    """(B, T) -> (B, T) where entry [b, t] is values[b, t + 1]. The outgoing action
    at block t lives one block later in led-to storage; keeping the length means the
    policy head has a target at every block the reward head does."""
    return F.pad(values[:, 1:], (0, 1), value=fill)


def _leads(values: Tensor, leads: int, fill: float) -> Tensor:
    """(B, T) -> (B, T, leads) where entry [b, t, n] is values[b, t + n], padded
    past the window end with `fill`."""
    padded = F.pad(values, (0, leads - 1), value=fill)
    return padded.unfold(1, leads, 1)[:, : values.shape[1]]
