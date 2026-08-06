"""Actor, critic, and correctly indexed imagination loop."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import M3HJWM, TemporalState, WorldState


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int, depth: int = 3):
        super().__init__()
        layers, d = [], inp
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, critics: int = 3):
        super().__init__()
        self.actor = MLP(state_dim, 128, action_dim)
        self.critics = nn.ModuleList([MLP(state_dim, 256, 1) for _ in range(critics)])

    def distribution(self, state: Tensor):
        return torch.distributions.Categorical(logits=self.actor(state))

    def sample(self, state: Tensor, deterministic: bool = False):
        dist = self.distribution(state)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def value_members(self, state: Tensor):
        return torch.stack([net(state).squeeze(-1) for net in self.critics])

    def value(self, state: Tensor):
        return self.value_members(state).mean(0)


@dataclass
class ImaginedTrajectory:
    states: Tensor          # [B,H+1,D]
    actions: Tensor         # [B,H]
    log_probs: Tensor       # [B,H]
    entropies: Tensor       # [B,H]
    rewards: Tensor         # [B,H], reward caused by actions[:,t]
    continues: Tensor       # [B,H]
    values: Tensor          # [B,H+1]
    predicted_error: Tensor # [B,H]
    confidence: Tensor      # [B,H]


def imagine(
    world: M3HJWM,
    agent: ActorCritic,
    start: WorldState,
    horizon: int,
    reliability_temperature: float = 1.0,
    deterministic_policy: bool = False,
    deterministic_modes: bool = False,
):
    def clone_detached(value):
        if isinstance(value, Tensor):
            return value.detach().clone()
        if isinstance(value, list):
            return [clone_detached(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_detached(item) for item in value)
        return value

    # Official recurrent kernels update cache tensors in place. Clone the start
    # state so repeated imagination calls cannot mutate a replay-prefix state, and
    # detach it so critic losses cannot flow into the preceding world-model graph.
    state = WorldState(
        TemporalState(
            clone_detached(start.temporal.cache),
            start.temporal.output.detach().clone(),
        ),
        start.tokens.detach().clone(),
        start.revision,
    )
    controls = [world.pool(state.tokens)]
    actions, logps, ents, rewards, conts, errors, confidence = [], [], [], [], [], [], []

    for _ in range(horizon):
        control = controls[-1]
        action, logp, entropy = agent.sample(control, deterministic_policy)
        # Categorical actions use a score-function estimator; no pathwise gradient
        # through the frozen world is required. Keeping this block graph-free also
        # prevents simultaneous world/actor/critic graphs on a 6 GB device.
        with torch.no_grad():
            next_state, reward_logits, continue_logits, pred = world.imagine_step(
                state, action, deterministic_modes
            )
            reward = world.reward.decode(reward_logits)
            continuation = torch.sigmoid(continue_logits)
            next_control = world.pool(next_state.tokens)

            value_disagreement = agent.value_members(next_control).var(0, unbiased=False)
            signals = world.reliability.signals(
                state.tokens, action, pred.selected, pred.all_modes, value_disagreement
            )
            predicted_error = world.reliability.predicted_error(signals)
            weight = torch.exp(-predicted_error / max(reliability_temperature, 1e-6))

        actions.append(action); logps.append(logp); ents.append(entropy)
        rewards.append(reward); conts.append(continuation)
        errors.append(predicted_error); confidence.append(weight)
        controls.append(next_control)
        state = next_state

    states = torch.stack(controls, 1)
    return ImaginedTrajectory(
        states=states,
        actions=torch.stack(actions, 1),
        log_probs=torch.stack(logps, 1),
        entropies=torch.stack(ents, 1),
        rewards=torch.stack(rewards, 1),
        continues=torch.stack(conts, 1),
        values=agent.value(states),
        predicted_error=torch.stack(errors, 1),
        confidence=torch.stack(confidence, 1),
    )


def lambda_returns(rewards: Tensor, continues: Tensor, values: Tensor, gamma: float, lambda_: float):
    """rewards[:,t] is caused by actions[:,t]; values has H+1 states."""
    out = torch.empty_like(rewards)
    carry = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        bootstrap = (1 - lambda_) * values[:, t + 1] + lambda_ * carry
        carry = rewards[:, t] + gamma * continues[:, t] * bootstrap
        out[:, t] = carry
    return out


def actor_critic_losses(
    trajectory: ImaginedTrajectory,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    entropy_coef: float = 3e-4,
    use_reliability: bool = False,
):
    returns = lambda_returns(
        trajectory.rewards, trajectory.continues, trajectory.values, gamma, lambda_
    )
    advantage = (returns - trajectory.values[:, :-1]).detach()
    weight = trajectory.confidence.detach() if use_reliability else torch.ones_like(advantage)
    actor = -(weight * trajectory.log_probs * advantage).mean() - entropy_coef * trajectory.entropies.mean()
    critic = (weight * (trajectory.values[:, :-1] - returns.detach()) ** 2).mean()
    return actor, critic, {
        "actor_loss": actor.detach(),
        "critic_loss": critic.detach(),
        "imagined_return": trajectory.rewards.sum(1).mean().detach(),
        "entropy": trajectory.entropies.mean().detach(),
        "predicted_error": trajectory.predicted_error.mean().detach(),
        "confidence": trajectory.confidence.mean().detach(),
    }


@contextmanager
def frozen(module: nn.Module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)
