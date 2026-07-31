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
    """Returns terminated and truncated as separate raw facts.

    Death and lava are absorbing; the step cap is not. Folding them into one flag
    is what makes a time limit look like a terminal state to the critic, so the
    split is preserved all the way to the continuation target.
    """
    env, params = _env()
    key = jax.random.PRNGKey(seed)
    observation, state, reward, done, _ = env.step(key, state, action, params)
    terminated = bool(_env()[0].is_terminal(state, params)) and not _timed_out(state, params)
    return _frame(observation), state, float(reward), terminated, _timed_out(state, params)


def _timed_out(state, params) -> bool:
    return bool(state.timestep >= params.max_timesteps)


def _frame(observation) -> Tensor:
    """Craftax renders float32 in [0, 1]. Stored as uint8 because an episode is
    thousands of frames and 4x is the difference between fitting in memory and not;
    `patchify` divides by 255 to undo it."""
    return torch.from_numpy(np.asarray(observation) * 255.0).round().to(torch.uint8)
