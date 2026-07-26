"""Guards the expert replay generation pipeline (JAX on CPU; one tiny episode)."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def test_generate_expert_replay_produces_valid_replay(tmp_path):
    import jax
    import jax.numpy as jnp
    from flax import serialization

    from d4_mamba_jepa.expert.ppo_expert import ActorCriticRNN, ScannedRNN
    from d4_mamba_jepa.expert.generate import generate_expert_replay
    from d4_mamba_jepa.data import load_episode_replay

    # random-init expert (mechanics only; no training)
    net = ActorCriticRNN(17, 32)
    hidden = ScannedRNN.initialize_carry(1, 32)
    params = net.init(
        jax.random.PRNGKey(0), hidden, (jnp.zeros((1, 1, 1345)), jnp.zeros((1, 1)))
    )
    pp = tmp_path / "expert.msgpack"
    pp.write_bytes(serialization.to_bytes(params))

    out = tmp_path / "expert_replay.pt"
    # 3 episodes from 2 envs => exercises the multi-batch loop + per-slot slicing.
    manifest = generate_expert_replay(
        params_path=pp, out_path=out, n_episodes=3, max_steps=8,
        layer_size=32, seed=100, num_envs=2,
    )
    assert manifest.n_episodes == 3
    assert 0.0 <= manifest.noop_fraction <= 1.0
    assert manifest.replay_sha256

    replay = load_episode_replay(
        out, expected_sha256=manifest.replay_sha256, capacity_steps=10 ** 7
    )
    assert len(replay.episodes) == 3
    for ep in replay.episodes:
        assert ep.obs.shape[1:] == (3, 64, 64) and ep.obs.dtype.name == "uint8"
        assert ep.obs.shape[0] == ep.actions.shape[0] + 1  # obs == actions + 1
