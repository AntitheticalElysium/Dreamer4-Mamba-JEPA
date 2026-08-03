import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import SwiGLU
from .config import Config
from .data import Batch


class Heads(nn.Module):
    """Policy, reward, continuation and value over the same pooled agent tokens.

    Three separate bodies. Reward and continuation are split from the actor because
    a single trunk would let Phase 3 move the reward model it is scored against --
    measured, one policy step changed both reward and continuation logits. The
    *critic* is split from the policy for the same reason one step further in: with
    a shared trunk a value-only backward put gradient 17654 into the body the policy
    reads, so the critic reshaped policy features outside PMPO and outside the
    prior KL that is supposed to bound how far the actor may move. D4 calls it "an
    additional value head" and does not ask for a shared trunk.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("centers", _centers(config), persistent=True)
        width, leads = config.d_model, config.mtp_leads
        self.actor_body = SwiGLU(width, 2.0)
        self.critic_body = SwiGLU(width, 2.0)
        self.model_body = SwiGLU(width, 2.0)
        self.policy = nn.Linear(width, leads * config.n_actions)
        self.reward = nn.Linear(width, leads * config.bins)
        self.continuation = nn.Linear(width, leads)
        self.value = nn.Linear(width, config.bins)
        # Output scales from the pinned DreamerV3 config: rewhead and value 0.0,
        # policy 0.01, conhead 1.0. A value head that starts at random emits random
        # advantages on Phase 3's first steps, and PMPO reads only their sign.
        for head, scale in ((self.reward, 0.0), (self.value, 0.0), (self.policy, 0.01)):
            head.weight.data.mul_(scale)
            head.bias.data.zero_()

    def actor_parameters(self):
        """What Phase 3 may move: the policy and the critic, each with its own body."""
        return [
            *self.actor_body.parameters(),
            *self.policy.parameters(),
            *self.critic_body.parameters(),
            *self.value.parameters(),
        ]

    def forward(self, agent: Tensor) -> dict[str, Tensor]:
        b, t = agent.shape[:2]
        pooled = agent.mean(dim=2)
        actor, critic, model = self.actor_body(pooled), self.critic_body(pooled), self.model_body(pooled)
        leads, config = self.config.mtp_leads, self.config
        return {
            "policy": self.policy(actor).view(b, t, leads, config.n_actions),
            "reward": self.reward(model).view(b, t, leads, config.bins),
            "continuation": self.continuation(model).view(b, t, leads),
            "value": self.value(critic),
        }


def twohot(values: Tensor, centers: Tensor) -> Tensor:
    """Linear interpolation between neighbouring centres, so bin count sets grid
    density rather than a quantisation floor."""
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

    Reward lead 0 at block t is the reward that *arrived* at t, so the reward caused
    by the action chosen at t is lead 0 at t+1 (S22). Policy lead 0 is the outgoing
    action, one block later in led-to storage. Padding past the window uses class 0,
    not BOS, which is an input embedding row and not a policy class.
    """
    assert batch.relevant is not None, "behaviour cloning needs the §4.1 mixture"
    leads = config.mtp_leads
    device = batch.led_to_action.device
    outgoing = _shift(batch.led_to_action.float(), fill=0.0)
    return {
        "action": _leads(outgoing, leads, fill=0.0),
        "reward": _leads(batch.reward, leads, fill=0.0),
        "continuation": _leads((~batch.terminated).float(), leads, fill=1.0),
        "valid": _leads(batch.valid.float(), leads, fill=0.0),
        "action_valid": _leads(_shift(batch.valid.float(), fill=0.0), leads, fill=0.0),
        "policy_rows": batch.rows("policy").to(device).float()[:, None, None],
        "reward_rows": batch.rows("reward").to(device).float()[:, None, None],
    }


def head_loss(
    predictions: dict[str, Tensor], targets: dict[str, Tensor], config: Config
) -> dict[str, Tensor]:
    """Returned per head, not summed: Dreamer 4 normalises every concurrent loss by
    its own running RMS, and merging them first lets whichever head has the largest
    natural scale set the others' effective weight.

    Behaviour cloning reads the relevant half only (§4.1). The main batch supplies
    reward and continuation; terminal tails use `terminal_loss` separately.
    """
    centers = predictions["centers"]
    policy = F.cross_entropy(
        predictions["policy"].flatten(0, 2), targets["action"].flatten().long(), reduction="none"
    ).view_as(targets["action"])
    reward = _distribution_loss(predictions["reward"], targets["reward"], centers)
    continuation = F.binary_cross_entropy_with_logits(
        predictions["continuation"], targets["continuation"], reduction="none"
    )
    valid = targets["valid"]
    actions = targets["action_valid"] * targets["policy_rows"]
    rewarded = valid * targets["reward_rows"]
    return {
        "policy": (policy * actions).sum() / actions.sum().clamp(min=1.0),
        "reward": (reward * rewarded).sum() / rewarded.sum().clamp(min=1.0),
        "continuation": (continuation * valid).sum() / valid.sum().clamp(min=1.0),
    }


def terminal_loss(predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    """Ordinary continuation BCE on a tail known to contain a terminal."""
    loss = F.binary_cross_entropy_with_logits(
        predictions["continuation"], targets["continuation"], reduction="none"
    )
    valid = targets["valid"]
    terminal = valid * (1.0 - targets["continuation"])
    assert terminal.sum() > 0, "terminal batch contains no terminal target"
    return (loss * valid).sum() / valid.sum()


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
    """(B, T) -> (B, T, leads), entry [b, t, n] = values[b, t + n]."""
    padded = F.pad(values, (0, leads - 1), value=fill)
    return padded.unfold(1, leads, 1)[:, : values.shape[1]]
