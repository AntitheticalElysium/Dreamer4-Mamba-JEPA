"""N3: `continues = 1 - dead` must hold even when death lands on the timeout.

Craftax `game_logic.is_game_over` is `done_steps | in_lava | is_dead`. The
adapter used to recover the absorbing half by inference -- `done and not
timeout` -- which is only correct while the disjuncts are mutually exclusive.
On the exact transition where the native step cap is reached, a simultaneous
death or lava entry was recorded as a bootstrappable truncation.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _fresh_state():
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(0), params)
    return env, params, state


def test_is_dead_matches_the_absorbing_half_of_is_game_over():
    from craftax.craftax_classic.constants import BlockType
    from craftax.craftax_classic.game_logic import is_game_over

    from d4_mamba_jepa.craftax_env import is_dead

    _, params, state = _fresh_state()
    assert not is_dead(state)
    assert not bool(is_game_over(state, params))

    dead = state.replace(player_health=0)
    assert is_dead(dead) and bool(is_game_over(dead, params))

    row, col = int(state.player_position[0]), int(state.player_position[1])
    lava = state.replace(map=state.map.at[row, col].set(BlockType.LAVA.value))
    assert is_dead(lava) and bool(is_game_over(lava, params))


def test_death_on_the_native_horizon_is_absorbing_not_a_truncation():
    """The exact case the old `done and not timeout` inference got wrong."""
    from craftax.craftax_classic.game_logic import is_game_over

    from d4_mamba_jepa.craftax_env import is_dead

    _, params, state = _fresh_state()
    horizon = int(params.max_timesteps)

    collision = state.replace(player_health=0, timestep=horizon)
    assert bool(is_game_over(collision, params))
    # The retired inference: `terminal = done and not (timestep >= horizon)`.
    assert (int(collision.timestep) >= horizon) is True, "timeout disjunct fires"
    stale_terminal = not (int(collision.timestep) >= horizon)
    assert stale_terminal is False, "the old rule called this a truncation"
    # The state-read rule keeps `continues = 1 - dead` exact.
    assert is_dead(collision) is True

    # A pure timeout with the player alive stays a bootstrappable truncation.
    timeout_only = state.replace(timestep=horizon)
    assert bool(is_game_over(timeout_only, params))
    assert is_dead(timeout_only) is False


def test_step_reports_continuation_from_the_state():
    from d4_mamba_jepa.craftax_env import CraftaxPixelEnv

    env = CraftaxPixelEnv(seed=0)
    env.reset()
    result = env.step(0)
    # A fresh episode cannot be over after one noop.
    assert result.continuation == 1.0 and not result.terminal and not result.done
