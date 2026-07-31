from dataclasses import dataclass

import torch

from .agent import Heads
from .config import Config
from .data import patchify
from .env import reset, step
from .representation import Encoder
from .transition import World, observe


@dataclass(frozen=True)
class Result:
    steps: int
    reward: float
    terminated: bool
    truncated: bool


def run_episode(
    world: World, encoder: Encoder, heads: Heads, seed: int, config: Config, limit: int = 2500
) -> Result:
    """The deployed loop, and the only place the whole system runs together.

    The flow arm corrupts its committed real latents here too, because its training
    never presents an uncorrupted one -- so executed control carries a third
    randomness source beyond environment seed and policy sampling, and any metric
    comparing arms has to control it. Both draws go through one seeded generator
    on the configured device, so a run is reproducible from its seed alone.
    """
    device = config.device
    world, encoder, heads = world.to(device).eval(), encoder.to(device).eval(), heads.to(device).eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    observation, env_state = reset(seed)
    state, total = None, 0.0
    action = torch.full((1, 1), config.n_actions, dtype=torch.long, device=device)

    with torch.no_grad():
        for index in range(limit):
            patches = patchify(observation[None, None], config.patch).to(device)
            state, agent = observe(world, encoder, state, action, patches, rng, config)
            logits = heads(agent)["policy"][:, -1, 0]
            choice = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))

            observation, env_state, reward, terminated, truncated = step(env_state, choice, seed + index)
            total += reward
            action = torch.full((1, 1), choice, dtype=torch.long, device=device)
            if terminated or truncated:
                return Result(index + 1, total, terminated, truncated)
    return Result(limit, total, False, True)
