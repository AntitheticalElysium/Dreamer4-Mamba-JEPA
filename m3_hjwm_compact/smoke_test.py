"""Run with: PYTHONPATH=. python smoke_test.py"""
import torch

from model import ModelConfig, M3HJWM
from agent import ActorCritic, imagine, lambda_returns
from train import TrainConfig, world_update, actor_critic_update, estimated_parameter_megabytes


def batch(cfg, b=2, t=5):
    return {
        "obs": torch.randint(0, 256, (b, t, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8),
        "actions": torch.randint(0, cfg.action_dim, (b, t - 1)),
        "rewards": torch.randn(b, t - 1),
        "continues": torch.ones(b, t - 1),
    }


def run():
    cfg = ModelConfig(
        patch_size=16, token_dim=16, registers=1, spatial_heads=2, spatial_depth=1,
        temporal_backend="gru", temporal_depth=1, predictor_depth=1, modes=2,
    )
    train_cfg = TrainConfig(batch_size=1, sequence_length=3, imagination_horizon=2, amp=False)
    world = M3HJWM(cfg)
    agent = ActorCritic(cfg.token_dim, cfg.action_dim, critics=2)

    # Shapes, indexing, and gradient flow.
    output = world(batch(cfg, b=1, t=3))
    assert output.context.shape == (1, 3, world.streams, cfg.token_dim)
    assert output.metrics["reward"].isfinite()
    assert output.loss.isfinite()
    output.loss.backward()
    assert any(p.grad is not None for p in world.online_encoder.parameters())

    # Sequence/step equivalence for the reference recurrent backend.
    x = torch.randn(1, 3, world.streams, cfg.token_dim)
    sequence, _ = world.temporal.sequence(x)
    state = world.temporal.init_state(1, world.streams, x.device, x.dtype)
    steps = []
    for t in range(x.shape[1]):
        y, state = world.temporal.step(x[:, t], state)
        steps.append(y)
    torch.testing.assert_close(sequence, torch.stack(steps, 1))

    # Reset isolation.
    obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    action = torch.zeros(2, dtype=torch.long)
    state = world.initial_state(2, torch.device("cpu"))
    state = world.observe_step(obs, action, state)
    reset_state = world.observe_step(obs, action, state, torch.tensor([True, False]))
    fresh = world.initial_state(1, torch.device("cpu"))
    fresh = world.observe_step(obs[:1], action[:1], fresh)
    torch.testing.assert_close(reset_state.tokens[:1], fresh.tokens, atol=1e-5, rtol=1e-5)

    # Imagination layout.
    start = world.initial_state(1, torch.device("cpu"))
    trajectory = imagine(world, agent, start, horizon=2)
    assert trajectory.states.shape == (1, 3, cfg.token_dim)
    assert trajectory.actions.shape == trajectory.rewards.shape == (1, 2)
    assert lambda_returns(
        trajectory.rewards, trajectory.continues, trajectory.values, 0.99, 0.95
    ).shape == (1, 2)

    # Optimiser smoke.
    world_opt = torch.optim.AdamW(world.parameters(), lr=1e-4)
    actor_opt = torch.optim.AdamW(agent.actor.parameters(), lr=3e-5)
    critic_opt = torch.optim.AdamW(agent.critics.parameters(), lr=3e-5)
    wm_metrics = world_update(world, batch(cfg, b=1, t=3), world_opt, train_cfg)
    # Any cache produced before a world update is stale by construction.
    start = world.initial_state(1, torch.device("cpu"))
    ac_metrics = actor_critic_update(world, agent, start, actor_opt, critic_opt, train_cfg)

    print("all smoke tests passed")
    print("world params MB(fp32):", round(estimated_parameter_megabytes(world), 2))
    print("world metrics:", wm_metrics)
    print("agent metrics:", ac_metrics)


if __name__ == "__main__":
    run()
