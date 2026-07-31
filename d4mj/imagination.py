from dataclasses import dataclass

import torch
from torch import Tensor

from .agent import Heads
from .config import Config
from .state import WorldState
from .transition import World, advance


@dataclass(frozen=True)
class Trajectory:
    """`continuation` is the head's probability, not a boolean: imagination has no
    ground-truth termination, and a hard flag would need termination sampled.

    `agent` is the readout at every state including the start, so the frozen prior
    and the critic can be evaluated where the actions were actually chosen rather
    than at the rollout's first state alone.

    There is no boundary mask. The horizon end is the return recursion's initial
    condition, not a per-step signal, so a boundary field would be a column of
    zeros that nothing reads.
    """

    action: Tensor
    logits: Tensor
    reward: Tensor
    continuation: Tensor
    value: Tensor
    agent: Tensor


def imagine(
    world: World,
    heads: Heads,
    state: WorldState,
    agent: Tensor,
    rng: torch.Generator,
    config: Config,
) -> Trajectory:
    """One rollout per starting state, per Dreamer 4.

    The caller builds `state` and its readout `agent` from dataset context through
    `observe`; imagination encodes nothing and owns no encoder. It takes the
    readout rather than recomputing it because the memory already covers that
    block, and re-evaluating would ingest the same latent twice.

    The action chosen at h_t is committed into the next block, so the reward it
    causes is read at lead 0 of the *next* readout. Reading lead 0 of the current
    one returns the previous action's reward and shifts every return by a step.

    Policy sampling goes through the supplied generator. `Categorical.sample()`
    reads the global stream, which makes a rollout irreproducible from its own seed
    and breaks the paired comparison the whole lattice rests on.
    """
    readout = heads(agent)
    actions, step_logits, rewards, continuations = [], [], [], []
    values = [_expect(readout["value"][:, -1], heads.centers)]
    readouts = [agent[:, -1]]

    for _ in range(config.horizon):
        logits = readout["policy"][:, -1, 0]
        action = torch.multinomial(logits.softmax(-1), 1, generator=rng).squeeze(-1)
        state, agent = advance(world, state, action[:, None], rng, config)
        readout = heads(agent)

        actions.append(action)
        step_logits.append(logits)
        rewards.append(_expect(readout["reward"][:, -1, 0], heads.centers))
        continuations.append(readout["continuation"][:, -1, 0].sigmoid())
        values.append(_expect(readout["value"][:, -1], heads.centers))
        readouts.append(agent[:, -1])

    return Trajectory(
        action=torch.stack(actions, dim=1),
        logits=torch.stack(step_logits, dim=1),
        reward=torch.stack(rewards, dim=1),
        continuation=torch.stack(continuations, dim=1),
        value=torch.stack(values, dim=1),
        agent=torch.stack(readouts, dim=1),
    )


def _expect(logits: Tensor, centers: Tensor) -> Tensor:
    """Mean of the two-hot distribution, mapped out of symlog space."""
    mean = (logits.softmax(-1) * centers).sum(-1)
    return mean.sign() * torch.expm1(mean.abs())
