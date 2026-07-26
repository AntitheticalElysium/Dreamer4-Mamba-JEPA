"""The full architecture runs on 17-action Craftax (no JAX; synthetic replay)."""
from __future__ import annotations

import numpy as np
import torch

from d4_mamba_jepa.craftax_runners import (
    craftax_jepa_config,
    train_craftax_bc,
    train_craftax_imagination,
    train_craftax_jepa_world,
)
from d4_mamba_jepa.data import Episode, EpisodeReplay
from d4_mamba_jepa.imagination_actor_critic import module_state_sha256

DEVICE = torch.device("cpu")


def _synthetic_replay(n_ep=4, length=20, seed=0):
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(capacity_steps=10 ** 6)
    for _ in range(n_ep):
        replay.add(Episode(
            obs=rng.integers(0, 256, (length + 1, 3, 64, 64), dtype=np.uint8),
            actions=rng.integers(0, 17, length).astype(np.int64),
            rewards=rng.random(length).astype(np.float32),
            continues=np.ones(length, dtype=np.float32),
        ))
    return replay


def test_craftax_jepa_config_is_17_action_64px():
    cfg = craftax_jepa_config("transformer")
    assert cfg.n_actions == 17 and cfg.image_size == 64
    assert cfg.representation_objective == "jepa" and cfg.arm_id == "T-JEPA"
    m = craftax_jepa_config("mamba2")
    assert m.arm_id == "M-JEPA" and m.mamba_d_state == 64 and m.mamba_headdim == 64


def test_full_architecture_runs_on_craftax_replay():
    replay = _synthetic_replay()
    cfg = craftax_jepa_config("transformer")

    world, _, history = train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=3, batch_size=4, seed=0,
        device=DEVICE, warmup=1,
    )
    assert all(np.isfinite(h["jepa"]) for h in history)
    assert history[-1]["online_std"] > 0.0  # not collapsed

    bc, losses = train_craftax_bc(
        world=world, replay=replay, steps=3, batch_size=4, seed=1,
        device=DEVICE, warmup=1,
    )
    assert all(np.isfinite(x) for x in losses)
    # BC head really has 17 outputs.
    assert bc.n_actions == 17

    prior_hash = module_state_sha256(bc)
    actor, value, imag = train_craftax_imagination(
        world=world, bc=bc, replay=replay, steps=2, batch_size=4,
        context=8, horizon=4, seed=2, device=DEVICE,
    )
    assert np.isfinite(imag[-1]["total_loss"])
    # The imagination update actually moved the actor off its BC prior.
    assert module_state_sha256(actor) != prior_hash
