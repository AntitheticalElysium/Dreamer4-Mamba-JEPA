from dataclasses import replace

import numpy as np
import pytest
import torch

from d4_mamba_jepa.config import D4LiteConfig
from d4_mamba_jepa.data import (
    Episode,
    EpisodeReplay,
    replay_sample_to_sequence,
    transitions_to_led_to,
    load_episode_replay,
)
from d4_mamba_jepa.model import D4LiteWorld, DiscreteActionEncoder, build_tokenizer
from d4_mamba_jepa.objectives import continuation_mtp_loss, shortcut_flow_loss
from d4_mamba_jepa.source import (
    MMBENCH2_MODEL,
    SourceDriftError,
    load_mmbench2_model,
    verify_source,
)


def tiny_config(**overrides) -> D4LiteConfig:
    cfg = D4LiteConfig(
        image_size=16,
        patch_size=4,
        sequence_length=4,
        tokenizer_d_model=16,
        tokenizer_heads=2,
        tokenizer_depth=2,
        tokenizer_time_every=1,
        n_latents=4,
        d_bottleneck=4,
        dynamics_d_model=16,
        dynamics_heads=2,
        dynamics_depth=2,
        dynamics_time_every=1,
        dynamics_mlp_ratio=2.0,
        packing_factor=2,
        n_register=1,
        n_agent=1,
        k_max=2,
        reward_horizon=2,
        reward_bins=17,
        continuation_horizon=2,
        mamba_headdim=8,
        mamba_d_state=8,
    )
    return replace(cfg, **overrides)


def test_registered_upstream_source_loads_unchanged():
    assert verify_source(MMBENCH2_MODEL) == MMBENCH2_MODEL.sha256
    upstream = load_mmbench2_model()
    assert upstream.Dynamics.__module__ == "d4_mamba_jepa._pinned_mmbench2_model"
    assert upstream.TimeSelfAttention.__module__ == upstream.Dynamics.__module__


def test_source_drift_is_a_hard_failure(tmp_path):
    identity = replace(MMBENCH2_MODEL, path=tmp_path / "model.py")
    identity.path.write_text("not upstream")
    with pytest.raises(SourceDriftError, match="digest drift"):
        verify_source(identity)


def test_arm_names_and_invalid_axes():
    assert tiny_config().arm_id == "T-BASE"
    assert tiny_config(temporal_backend="mamba2").arm_id == "M-BASE"
    assert tiny_config(representation_objective="cdp").arm_id == "T-CDP"
    with pytest.raises(ValueError, match="temporal_backend"):
        tiny_config(temporal_backend="silent-gru")


def test_tokenizer_is_upstream_and_reconstructs_expected_shape():
    cfg = tiny_config()
    upstream = load_mmbench2_model()
    tokenizer = build_tokenizer(cfg)
    assert isinstance(tokenizer, upstream.Tokenizer)
    video = torch.rand(2, cfg.sequence_length, 3, 16, 16)
    patches = upstream.temporal_patchify(video, cfg.patch_size)
    prediction, mask, keep = tokenizer(patches)
    assert prediction.shape == patches.shape
    assert mask.shape == (*patches.shape[:-1], 1)
    assert keep.shape == (2, cfg.sequence_length, 1)


def test_discrete_action_encoder_has_explicit_start_and_separates_actions():
    torch.manual_seed(0)
    encoder = DiscreteActionEncoder(d_model=16, n_actions=17)
    actions = torch.tensor([[-1, 0, 1, 16]])
    tokens = encoder(actions)
    assert tokens.shape == (1, 4, 1, 16)
    assert not torch.equal(tokens[:, 0], tokens[:, 1])
    assert not torch.equal(tokens[:, 1], tokens[:, 2])
    with pytest.raises(ValueError, match="must lie"):
        encoder(torch.tensor([[17]]))


def test_replay_adapter_preserves_led_to_transition_timing():
    observations = np.zeros((5, 3, 16, 16), np.uint8)
    episode = Episode(
        obs=observations,
        actions=np.array([2, 3, 4, 5], np.int64),
        rewards=np.array([0.0, 1.0, -1.0, 2.0], np.float32),
        continues=np.array([1.0, 1.0, 1.0, 0.0], np.float32),
    )
    replay = EpisodeReplay()
    replay.add(episode)
    sample = replay.sample(
        batch=1,
        observations=5,
        device=torch.device("cpu"),
        rng=np.random.default_rng(7),
    )
    batch = replay_sample_to_sequence(sample)
    assert batch.led_to_actions.tolist() == [[-1, 2, 3, 4, 5]]
    assert batch.led_to_rewards.tolist() == [[0.0, 0.0, 1.0, -1.0, 2.0]]
    assert batch.led_to_continues.tolist() == [[0.0, 1.0, 1.0, 1.0, 0.0]]
    assert batch.outcome_valid.tolist() == [[False, True, True, True, True]]
    transitions = torch.tensor([[2, 3, 4, 5]])
    assert torch.equal(batch.led_to_actions, transitions_to_led_to(transitions))


def test_replay_loader_rejects_digest_drift(tmp_path):
    path = tmp_path / "episodes.pt"
    torch.save([], path)
    with pytest.raises(RuntimeError, match="digest drift"):
        load_episode_replay(path, expected_sha256="0" * 64)


def test_transformer_baseline_forward_and_flow_gradients():
    torch.manual_seed(3)
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    upstream = load_mmbench2_model()
    assert all(
        isinstance(layer.time, upstream.TimeSelfAttention)
        for layer in world.dynamics.transformer.layers
        if layer.do_time
    )

    frames = torch.randint(
        0, 256, (2, cfg.sequence_length, 3, 16, 16), dtype=torch.uint8
    )
    encoded = world.encode_frames(frames)
    actions = torch.tensor([[-1, 0, 1, 2], [-1, 3, 4, 5]])
    loss, metrics = shortcut_flow_loss(
        world.dynamics,
        clean=encoded.packed,
        led_to_actions=actions,
        k_max=cfg.k_max,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["flow_mse"].item() >= 0.0
    assert world.dynamics.flow_x_head.weight.grad is not None
    assert torch.isfinite(world.dynamics.flow_x_head.weight.grad).all()


def test_block_causal_dynamics_has_no_future_leakage():
    torch.manual_seed(5)
    cfg = tiny_config()
    world = D4LiteWorld(cfg).eval()
    B, T = 1, cfg.sequence_length
    clean = torch.randn(B, T, cfg.n_spatial, cfg.d_spatial)
    altered = clean.clone()
    altered[:, -1] = torch.randn_like(altered[:, -1]) * 100
    actions = torch.tensor([[-1, 0, 1, 2]])
    steps = torch.full((B, T), cfg.max_step_index, dtype=torch.long)
    signals = torch.full((B, T), cfg.k_max, dtype=torch.long)
    with torch.no_grad():
        first, _ = world.forward_dynamics(clean, actions, steps, signals)
        second, _ = world.forward_dynamics(altered, actions, steps, signals)
    torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0, atol=0)


def test_continuation_mtp_alignment_masks_undefined_start():
    logits = torch.zeros(1, 4, 2, requires_grad=True)
    continues = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    valid = torch.tensor([[False, True, True, True]])
    loss = continuation_mtp_loss(logits, continues, valid)
    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-6)
    loss.backward()
    assert logits.grad is not None
    # Position zero, lead zero addresses the undefined incoming transition.
    assert logits.grad[0, 0, 0].item() == 0.0


def test_baseline_has_no_silent_research_modules():
    world = D4LiteWorld(tiny_config())
    assert world.cdp_predictor is None
