"""Tests for the PPO expert's vendored math + params I/O (JAX on CPU, no craftax).

JAX is forced to CPU here so these never contend with the torch GPU tests.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from d4_mamba_jepa.expert.ppo_expert import (
    ActorCriticRNN,
    ScannedRNN,
    cat_entropy,
    cat_log_prob,
    load_expert,
)


def test_categorical_helpers_match_definition():
    # log_prob of the chosen action equals log of its probability.
    logits = jnp.log(jnp.array([[0.1, 0.2, 0.7]]))
    lp = cat_log_prob(logits, jnp.array([2]))
    assert abs(float(lp[0]) - float(jnp.log(0.7))) < 1e-5
    # entropy of a uniform categorical over n classes is log(n).
    uniform = jnp.zeros((4, 5))
    assert np.allclose(np.asarray(cat_entropy(uniform)), np.log(5.0), atol=1e-5)


def test_expert_params_save_load_roundtrip(tmp_path):
    net = ActorCriticRNN(17, 64)
    hidden = ScannedRNN.initialize_carry(1, 64)
    x = (jnp.zeros((1, 1, 50)), jnp.zeros((1, 1)))
    params = net.init(jax.random.PRNGKey(0), hidden, x)

    path = tmp_path / "expert.msgpack"
    path.write_bytes(serialization.to_bytes(params))
    _, loaded, _ = load_expert(path, obs_dim=50, action_dim=17, layer_size=64)

    a = jax.tree.leaves(params)
    b = jax.tree.leaves(loaded)
    assert len(a) == len(b) and len(a) > 0
    assert all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
