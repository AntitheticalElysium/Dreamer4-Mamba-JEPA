"""The full architecture runs on 17-action Craftax (no JAX; synthetic replay)."""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import replace

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
    for i in range(n_ep):
        continues = np.ones(length, dtype=np.float32)
        if i % 2 == 0:
            continues[-1] = 0.0  # terminal episodes for the continuation head
        replay.add(Episode(
            obs=rng.integers(0, 256, (length + 1, 3, 64, 64), dtype=np.uint8),
            actions=rng.integers(0, 17, length).astype(np.int64),
            rewards=rng.random(length).astype(np.float32),
            continues=continues,
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


def test_craftax_world_terminal_oversampling_requires_terminal_episodes():
    # A replay whose episodes never terminate (continues all 1) must be rejected
    # by the terminal-oversampling world runner, proving oversampling is active.
    import pytest
    rng = np.random.default_rng(0)
    replay = EpisodeReplay(capacity_steps=10 ** 6)
    for _ in range(3):
        replay.add(Episode(
            obs=rng.integers(0, 256, (20, 3, 64, 64), dtype=np.uint8),
            actions=rng.integers(0, 17, 19).astype(np.int64),
            rewards=rng.random(19).astype(np.float32),
            continues=np.ones(19, dtype=np.float32),  # never terminal
        ))
    with pytest.raises(RuntimeError, match="terminal"):
        train_craftax_jepa_world(
            replay=replay, cfg=craftax_jepa_config("transformer"),
            world_steps=1, batch_size=2, seed=0, device=DEVICE, warmup=1,
        )


def test_craftax_world_can_freeze_only_encoder_with_separate_lr_group():
    from d4_mamba_jepa.model import D4LiteWorld

    replay = _synthetic_replay()
    cfg = craftax_jepa_config("transformer")
    torch.manual_seed(13)
    initial = D4LiteWorld(cfg)
    initial_encoder = {
        name: value.detach().clone()
        for name, value in initial.encoder.state_dict().items()
    }
    initial_predictor = {
        name: value.detach().clone()
        for name, value in initial.jepa_predictor.state_dict().items()
    }
    world, _, _ = train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=2, batch_size=2,
        learning_rate=1e-4, encoder_learning_rate=0.0,
        terminal_fraction=0.0, seed=13, device=DEVICE, warmup=1,
    )
    assert all(
        torch.equal(value, initial_encoder[name])
        for name, value in world.encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_predictor[name])
        for name, value in world.jepa_predictor.state_dict().items()
    )


def test_equal_encoder_lr_group_is_a_noop_control():
    replay = _synthetic_replay()
    cfg = craftax_jepa_config("transformer")
    default, _, default_history = train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=2, batch_size=2,
        learning_rate=1e-4, encoder_learning_rate=None,
        terminal_fraction=0.0, seed=19, device=DEVICE, warmup=1,
    )
    grouped, _, grouped_history = train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=2, batch_size=2,
        learning_rate=1e-4, encoder_learning_rate=1e-4,
        terminal_fraction=0.0, seed=19, device=DEVICE, warmup=1,
    )
    assert default_history == grouped_history
    assert all(
        torch.equal(value, grouped.state_dict()[name])
        for name, value in default.state_dict().items()
    )


def test_sigreg_dev_cosine_does_not_advance_training_slice_state():
    from d4_mamba_jepa.craftax_run import _dev_cosine, _fixed_dev_batches
    from d4_mamba_jepa.model import D4LiteWorld

    replay = _synthetic_replay()
    cfg = replace(
        craftax_jepa_config("transformer"),
        jepa_anticollapse="sigreg",
        jepa_sigreg_slices=16,
    )
    world = D4LiteWorld(cfg)
    batches = _fixed_dev_batches(
        replay, cfg=cfg, count=1, batch_size=2, seed=7
    )
    before = world.sigreg_test.global_step.detach().clone()
    cosine = _dev_cosine(world, batches, DEVICE)
    assert np.isfinite(cosine)
    assert torch.equal(world.sigreg_test.global_step, before)


def test_craftax_checkpoint_roundtrip(tmp_path):
    from d4_mamba_jepa.checkpoint import file_sha256, load_checkpoint
    from d4_mamba_jepa.cartpole_baseline import load_bc_policy
    # a replay with a terminal episode so the world runner's oversampling works
    rng = np.random.default_rng(1)
    replay = EpisodeReplay(capacity_steps=10 ** 6)
    for i in range(4):
        cont = np.ones(19, dtype=np.float32)
        if i == 0:
            cont[-1] = 0.0  # one terminal episode
        replay.add(Episode(
            obs=rng.integers(0, 256, (20, 3, 64, 64), dtype=np.uint8),
            actions=rng.integers(0, 17, 19).astype(np.int64),
            rewards=rng.random(19).astype(np.float32), continues=cont))
    cfg = craftax_jepa_config("transformer")
    train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=2, batch_size=2, seed=0,
        device=DEVICE, warmup=1, output_dir=tmp_path)
    import json
    wr = json.loads((tmp_path / "world_report.json").read_text())
    world_sha = wr["world_checkpoint_sha256"]
    assert file_sha256(tmp_path / "world.pt") == world_sha
    # world loads back
    world, _, _ = load_checkpoint(tmp_path / "world.pt", device=DEVICE,
                                  expected_sha256=world_sha, strict_implementation=False)
    assert world.cfg.n_actions == 17
    # BC saves + loads paired to the world sha
    train_craftax_bc(world=world, replay=replay, steps=2, batch_size=2, seed=1,
                     device=DEVICE, warmup=1, output_dir=tmp_path,
                     world_checkpoint_sha256=world_sha)
    br = json.loads((tmp_path / "bc_report.json").read_text())
    bc, _ = load_bc_policy(tmp_path / "bc.pt", expected_sha256=br["bc_checkpoint_sha256"],
                           expected_world_sha256=world_sha, device=DEVICE)
    assert bc.n_actions == 17
