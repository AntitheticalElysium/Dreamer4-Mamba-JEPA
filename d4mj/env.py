import functools

import jax
import numpy as np
import torch
from torch import Tensor


@functools.cache
def _env():
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    return env, env.default_params


def reset(seed: int) -> tuple[Tensor, object]:
    env, params = _env()
    observation, state = env.reset(jax.random.PRNGKey(seed), params)
    return _frame(observation), state


def step(state, action: int, seed: int) -> tuple[Tensor, object, float, bool, bool]:
    """Terminated and truncated as separate facts. Craftax folds death, lava and the
    step cap into `is_terminal`, so death is read from the state directly: deriving
    it as `is_terminal and not timed_out` drops a death that lands on the cap. Both
    flags may be true."""
    from craftax.craftax_classic.constants import BlockType

    env, params = _env()
    observation, state, reward, done, _ = env.step(jax.random.PRNGKey(seed), state, action, params)
    lava = state.map[state.player_position[0], state.player_position[1]] == BlockType.LAVA.value
    terminated = bool(lava) or bool(state.player_health <= 0)
    return _frame(observation), state, float(reward), terminated, _timed_out(state, params)


def _timed_out(state, params) -> bool:
    return bool(state.timestep >= params.max_timesteps)


def _frame(observation) -> Tensor:
    """Craftax renders float32 in [0, 1]; stored uint8, which `patchify` undoes."""
    return torch.from_numpy(np.asarray(observation) * 255.0).round().to(torch.uint8)
