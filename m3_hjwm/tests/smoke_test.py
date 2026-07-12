from __future__ import annotations
import torch
from m3_hjwm.config import ModelConfig, TrainConfig
from m3_hjwm.world_model import M3HJWM
from m3_hjwm.actor_critic import ActorCritic
from m3_hjwm.imagination import imagine, lambda_returns


def make_batch(cfg: ModelConfig, b: int = 2, t: int = 5):
    return {
        "obs": torch.randint(0, 256, (b, t, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8),
        "actions": torch.randint(0, cfg.action_dim, (b, t - 1)),
        "rewards": torch.randn(b, t - 1),
        "continues": torch.ones(b, t - 1),
    }


def test_world_model_shapes_and_gradients():
    cfg = ModelConfig(
        image_size=32, token_dim=32, spatial_heads=4, temporal_backend="gru",
        temporal_depth=1, predictor_depth=1, num_modes=3,
    )
    train = TrainConfig(batch_size=2, sequence_length=5, imagination_horizon=4, device="cpu")
    model = M3HJWM(cfg)
    batch = make_batch(cfg)
    out = model(batch, train)
    assert out.context_tokens.shape == (2, 5, model.streams, 32)
    assert out.reward_logits.shape[:2] == (2, 4)
    assert out.continue_logits.shape == (2, 4)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())


def test_transition_indexing():
    """Reward count must exactly equal action count."""
    cfg = ModelConfig(image_size=32, token_dim=32, spatial_heads=4, temporal_backend="gru", temporal_depth=1, predictor_depth=1)
    model = M3HJWM(cfg)
    out = model(make_batch(cfg, b=1, t=4), TrainConfig(device="cpu"))
    assert out.reward_logits.shape[1] == 3
    assert out.predictions.per_sample_loss.numel() == 3


def test_step_sequence_equivalence_gru():
    cfg = ModelConfig(image_size=32, token_dim=32, spatial_heads=4, temporal_backend="gru", temporal_depth=1, predictor_depth=1)
    model = M3HJWM(cfg)
    x = torch.randn(2, 6, model.streams, 32)
    seq, _ = model.temporal.forward_sequence(x)
    state = model.temporal.init_state(2, model.streams, x.device, x.dtype)
    steps = []
    for i in range(x.shape[1]):
        y, state = model.temporal.step(x[:, i], state)
        steps.append(y)
    stepped = torch.stack(steps, 1)
    torch.testing.assert_close(seq, stepped)


def test_reset_isolation():
    cfg = ModelConfig(image_size=32, token_dim=32, spatial_heads=4, temporal_backend="gru", temporal_depth=1, predictor_depth=1)
    model = M3HJWM(cfg)
    state = model.initial_state(2, torch.device("cpu"), torch.float32)
    obs = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    a = torch.zeros(2, dtype=torch.long)
    state = model.observe_step(obs, a, state)
    reset = torch.tensor([True, False])
    after = model.observe_step(obs, a, state, reset)
    fresh = model.initial_state(1, torch.device("cpu"), torch.float32)
    fresh_after = model.observe_step(obs[:1], a[:1], fresh)
    torch.testing.assert_close(after.tokens[:1], fresh_after.tokens, atol=1e-5, rtol=1e-5)


def test_imagination_layout():
    cfg = ModelConfig(image_size=32, token_dim=32, spatial_heads=4, temporal_backend="gru", temporal_depth=1, predictor_depth=1)
    model = M3HJWM(cfg)
    actor = ActorCritic(32, cfg.action_dim, value_ensemble=2)
    state = model.initial_state(2, torch.device("cpu"), torch.float32)
    traj = imagine(model, actor, state, horizon=4)
    assert traj.states.shape == (2, 5, 32)
    assert traj.actions.shape == traj.rewards.shape == traj.continues.shape == (2, 4)
    returns = lambda_returns(traj.rewards, traj.continues, traj.values, .99, .95)
    assert returns.shape == (2, 4)


def run_all():
    test_world_model_shapes_and_gradients()
    test_transition_indexing()
    test_step_sequence_equivalence_gru()
    test_reset_isolation()
    test_imagination_layout()
    print("all smoke tests passed")


if __name__ == "__main__":
    run_all()
