import copy

import torch
from torch import nn

from .actor_critic import actor_loss, critic_loss, lambda_returns
from .agent import Heads, head_loss, head_targets
from .config import Config
from .data import Batch, Episode, sample_batch
from .imagination import imagine
from .representation import Decoder, Encoder, pack, reconstruction_loss
from .state import WorldState
from .transition import World, transition_loss


def optimizer(modules: list[nn.Module], config: Config) -> torch.optim.AdamW:
    """The only place parameter groups are built, and the only place Mamba-2's
    no-weight-decay contract is honoured.

    Upstream marks A_log, D and dt_bias exempt. Decaying them is an asymmetry
    applied to exactly one arm of the single comparison the project exists to
    make, which is why it cannot live at a call site.
    """
    decayed, exempt = [], []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                (exempt if getattr(parameter, "_no_weight_decay", False) else decayed).append(parameter)
    groups = [{"params": decayed, "weight_decay": config.weight_decay}]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def train_representation(episodes: list[Episode], steps: int, config: Config) -> tuple[Encoder, Decoder]:
    """Phase 1A. Windows carry a burn-in that updates encoder memory and scores
    nothing, so every scored latent has the same history it will have at
    deployment. Afterwards the encoder is frozen and each episode is scanned once
    into the cache under that same window rule."""
    encoder, decoder = Encoder(config), Decoder(config)
    optimiser = optimizer([encoder, decoder], config)
    rng = torch.Generator().manual_seed(config.seed)

    for step in range(steps):
        batch = sample_batch(episodes, rng, config)
        z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        scored = slice(batch.burn_in, None)
        loss = reconstruction_loss(
            predicted[:, scored], batch.patches[:, scored], masked[:, scored]
        )
        _update(optimiser, loss, [encoder, decoder], config, step)
    return encoder.eval(), decoder.eval()


def train_dynamics(episodes: list[Episode], steps: int, config: Config) -> World:
    """Phase 1B, on the frozen cache. No agent losses yet, but the agent slots are
    already present and masked, so no state shape changes at the Phase 2 boundary."""
    world = World(config)
    optimiser = optimizer([world], config)
    rng = torch.Generator().manual_seed(config.seed + 1)

    for step in range(steps):
        batch = sample_batch(episodes, rng, config)
        _update(optimiser, transition_loss(world, batch, rng, config), [world], config, step)
    return world


def train_agent(episodes: list[Episode], world: World, steps: int, config: Config) -> Heads:
    """Phase 2. The dynamics objective continues alongside the head losses, which
    is what keeps the world model from drifting while the heads fit it."""
    heads = Heads(config)
    optimiser = optimizer([world, heads], config)
    rng = torch.Generator().manual_seed(config.seed + 2)

    for step in range(steps):
        batch = sample_batch(episodes, rng, config)
        _, agent, _ = _teacher_forced(world, batch, rng, config)
        readout = heads(agent) | {"centers": heads.centers}
        loss = transition_loss(world, batch, rng, config) + head_loss(
            readout, head_targets(batch, config), config
        )
        _update(optimiser, loss, [world, heads], config, step)
    return heads


def train_actor(episodes: list[Episode], world: World, heads: Heads, steps: int, config: Config) -> Heads:
    """Phase 3. The world is frozen and the behaviour-cloned policy is copied and
    frozen as the prior, so the actor cannot improve by reshaping the model it is
    being scored inside."""
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    prior = copy.deepcopy(heads).eval()
    optimiser = optimizer([heads], config)
    rng = torch.Generator().manual_seed(config.seed + 3)

    for step in range(steps):
        batch = sample_batch(episodes, rng, config)
        with torch.no_grad():
            _, agent, memory = _teacher_forced(world, batch, rng, config)
        state = WorldState(batch.latents[:, -1:], memory, batch.latents.shape[1])
        trajectory = imagine(world, heads, state, agent[:, -1:], rng, config)

        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        critic = heads(trajectory.agent[:, :-1])["value"]
        loss = actor_loss(trajectory, returns, reference, config) + critic_loss(
            critic, returns, heads.centers
        )
        _update(optimiser, loss, [heads], config, step)
    return heads


def _teacher_forced(world: World, batch: Batch, rng, config: Config):
    from .transition import _commit_inputs

    committed, conditioning = _commit_inputs(batch.latents, rng, config)
    return world(None, batch.led_to_action, committed, conditioning)


def _update(optimiser, loss, modules, config: Config, step: int) -> None:
    for group in optimiser.param_groups:
        group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
    optimiser.zero_grad()
    loss.backward()
    for module in modules:
        torch.nn.utils.clip_grad_norm_(module.parameters(), config.grad_clip)
    optimiser.step()
