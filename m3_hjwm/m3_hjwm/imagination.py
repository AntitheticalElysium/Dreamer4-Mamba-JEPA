from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .world_model import M3HJWM, WorldModelState
from .actor_critic import ActorCritic
from .reliability import ReliabilitySignals, mode_dispersion


@dataclass
class ImaginedTrajectory:
    states: Tensor                 # [B,H+1,D]
    actions: Tensor                # [B,H]
    log_probs: Tensor              # [B,H]
    entropies: Tensor              # [B,H]
    rewards: Tensor                # [B,H] = reward_{t+1}
    continues: Tensor              # [B,H]
    values: Tensor                 # [B,H+1]
    confidence: Tensor             # [B,H]
    predicted_error: Tensor        # [B,H]


def pool_control(model: M3HJWM, tokens: Tensor) -> Tensor:
    return model._pool(tokens)


def imagine(
    model: M3HJWM,
    actor_critic: ActorCritic,
    start: WorldModelState,
    horizon: int,
    reliability_temperature: float = 1.0,
    deterministic_policy: bool = False,
    deterministic_modes: bool = False,
) -> ImaginedTrajectory:
    """Generate a Dreamer-style trajectory with unambiguous indexing.

    For every i:
        states[:, i], actions[:, i]
            -> rewards[:, i], continues[:, i], states[:, i+1]

    World-model parameters should usually be frozen for actor/critic updates. Whether
    actor gradients flow through generated states is a separate training choice.
    """
    state = start
    state_list = [pool_control(model, state.tokens)]
    actions, logps, ents, rewards, conts, confs, errs = [], [], [], [], [], [], []
    for _ in range(horizon):
        control = state_list[-1]
        action, logp, entropy = actor_critic.sample(control, deterministic_policy)
        next_state, reward_logits, continue_logits, pred = model.imagine_step(
            state, action, deterministic_modes
        )
        reward = model.reward.decode(reward_logits)
        continuation = torch.sigmoid(continue_logits)

        if pred.all_modes is not None:
            dispersion = mode_dispersion(pred.all_modes)
        else:
            dispersion = reward.new_zeros(reward.shape)
        energy = model.energy(state.tokens, action, pred.prediction)
        projected = model.manifold(pred.prediction)
        manifold = ((pred.prediction - projected) ** 2).mean((1, 2))
        next_control = pool_control(model, next_state.tokens)
        value_members = actor_critic.value_members(next_control)
        value_disagreement = value_members.var(0, unbiased=False)
        signals = ReliabilitySignals(dispersion, energy, manifold, value_disagreement)
        predicted_error = model.reliability(signals)
        confidence = model.reliability.confidence(predicted_error, reliability_temperature)

        actions.append(action)
        logps.append(logp)
        ents.append(entropy)
        rewards.append(reward)
        conts.append(continuation)
        confs.append(confidence)
        errs.append(predicted_error)
        state = next_state
        state_list.append(next_control)

    states = torch.stack(state_list, 1)
    values = actor_critic.value(states)
    return ImaginedTrajectory(
        states=states,
        actions=torch.stack(actions, 1),
        log_probs=torch.stack(logps, 1),
        entropies=torch.stack(ents, 1),
        rewards=torch.stack(rewards, 1),
        continues=torch.stack(conts, 1),
        values=values,
        confidence=torch.stack(confs, 1),
        predicted_error=torch.stack(errs, 1),
    )


def lambda_returns(
    rewards: Tensor,
    continues: Tensor,
    values: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    """Compute returns aligned with actions.

    rewards[:, t] and continues[:, t] are consequences of action_t.
    values has H+1 entries.
    """
    h = rewards.shape[1]
    out = torch.empty_like(rewards)
    carry = values[:, -1]
    for t in reversed(range(h)):
        bootstrap = (1.0 - lambda_) * values[:, t + 1] + lambda_ * carry
        carry = rewards[:, t] + gamma * continues[:, t] * bootstrap
        out[:, t] = carry
    return out


def actor_critic_losses(
    traj: ImaginedTrajectory,
    gamma: float,
    lambda_: float,
    entropy_coef: float,
    apply_confidence: bool,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    returns = lambda_returns(traj.rewards, traj.continues, traj.values, gamma, lambda_)
    advantage = (returns - traj.values[:, :-1]).detach()
    weights = traj.confidence.detach() if apply_confidence else torch.ones_like(advantage)

    actor = -(weights * traj.log_probs * advantage).mean() - entropy_coef * traj.entropies.mean()
    critic = (weights * (traj.values[:, :-1] - returns.detach()) ** 2).mean()
    metrics = {
        "actor_loss": actor.detach(),
        "critic_loss": critic.detach(),
        "imagined_return": traj.rewards.sum(1).mean().detach(),
        "policy_entropy": traj.entropies.mean().detach(),
        "mean_confidence": traj.confidence.mean().detach(),
        "mean_predicted_error": traj.predicted_error.mean().detach(),
    }
    return actor, critic, metrics
