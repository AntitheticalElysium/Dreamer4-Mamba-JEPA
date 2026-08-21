"""Exact single-slot replay of a support-v2 episode, recovering simulator state.

`collect_support.rollout_batch` is a deterministic `jax.lax.scan` over
`jax.random.PRNGKey(seed + round)`, and each episode stores its executed actions,
so one env slot can be replayed without the policy: only the reset key and the
per-step environment key are needed, and both come from the same split tree.
`verify` renders the replayed states and compares them with the stored frames.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
STORE = ROOT / "artifacts/craftax_support_v2"
SUPPORT_SEED = 20260731 + 10_000
NUM_ENVS = 24
MAX_STEPS = 2500
BUCKETS = (64, 128, 256, 512, 1024, 2500)


@functools.cache
def env_and_render():
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    render = make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))
    step = jax.jit(functools.partial(env.step, params=env.default_params))
    reset = jax.jit(functools.partial(env.reset, params=env.default_params))
    frame = jax.jit(lambda s: jnp.clip(jnp.round(render(s)), 0, 255).astype(jnp.uint8))
    return env, env.default_params, reset, step, frame


@functools.cache
def manifest() -> dict:
    return json.loads((STORE / "manifest.json").read_text())


@functools.cache
def _shard(shard_index: int):
    return torch.load(STORE / manifest()["shards"][shard_index]["file"],
                      weights_only=False, mmap=True)


def episode_fields(shard_index: int, slot: int) -> dict:
    return _shard(shard_index)["episodes"][slot]


def round_keys(round_index: int, limit: int = MAX_STEPS):
    key = jax.random.PRNGKey(SUPPORT_SEED + round_index)
    key, reset_key = jax.random.split(key)
    return jax.random.split(reset_key, NUM_ENVS), jax.random.split(key, limit)


@functools.cache
def _scan_to():
    env, params, _, _, _ = env_and_render()

    def run(reset_key, env_keys, actions, stop):
        _, state = env.reset(reset_key, params)

        def body(carry, xs):
            state, index = carry
            key, action = xs
            _, nxt, _, _, _ = env.step(key, state, action, params)
            live = index < stop
            state = jax.tree.map(lambda a, b: jnp.where(live, a, b), nxt, state)
            return (state, index + 1), None

        (state, _), _ = jax.lax.scan(body, (state, 0), (env_keys, actions))
        return state

    return jax.jit(run)


@functools.cache
def _slot_keys(shard_index: int, slot: int):
    reset_keys, step_keys = round_keys(shard_index)
    selected = jax.vmap(
        lambda k: jax.random.split(jax.random.split(k, 4)[3], NUM_ENVS)[slot]
    )(step_keys)
    return reset_keys[slot], selected


def advance_to(shard_index: int, slot: int, t: int):
    """The simulator state after `t` stored actions, via the compiled scan."""
    reset_key, env_keys = _slot_keys(shard_index, slot)
    actions = episode_fields(shard_index, slot)["actions_taken"].numpy()
    bucket = next(size for size in BUCKETS if size >= max(t, 1))
    padded = jnp.zeros(bucket, dtype=jnp.int32).at[: min(len(actions), bucket)].set(
        actions[:bucket]
    )
    return _scan_to()(reset_key, env_keys[:bucket], padded, t)


def verify(shard_index: int, slot: int, checks: int = 8) -> float:
    """Max absolute pixel difference between replayed renders and stored frames."""
    _, _, _, _, frame = env_and_render()
    fields = episode_fields(shard_index, slot)
    stored = fields["observations"].numpy()
    n = len(fields["actions_taken"])
    worst = 0.0
    for t in np.linspace(0, n, min(checks, n + 1), dtype=int).tolist():
        rendered = np.asarray(frame(advance_to(shard_index, slot, int(t))))
        worst = max(worst, float(np.abs(rendered.astype(np.int32) - stored[t].astype(np.int32)).max()))
    return worst


def scalars(state) -> dict:
    inv = state.inventory
    return {
        "health": float(state.player_health), "food": float(state.player_food),
        "drink": float(state.player_drink), "energy": float(state.player_energy),
        "sleeping": bool(state.is_sleeping), "light": float(state.light_level),
        "achievements": int(state.achievements.sum()),
        "wood": int(inv.wood), "stone": int(inv.stone), "coal": int(inv.coal),
        "iron": int(inv.iron), "diamond": int(inv.diamond), "sapling": int(inv.sapling),
        "n_zombies": int(state.zombies.mask.sum()),
        "n_skeletons": int(state.skeletons.mask.sum()),
        "n_arrows": int(state.arrows.mask.sum()),
    }


def is_dead(state) -> bool:
    from craftax.craftax_classic.constants import BlockType

    lava = state.map[state.player_position[0], state.player_position[1]] == BlockType.LAVA.value
    return bool(lava) or bool(state.player_health <= 0)
