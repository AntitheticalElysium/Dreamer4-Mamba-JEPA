from dataclasses import replace

import torch

import artifacts.run_oracle_phase3 as oracle
from d4mj.actor_critic import actor_loss, critic_loss, lambda_returns
from d4mj.agent import Heads
from d4mj.config import Config


def _stream(slot: int, config) -> oracle.OracleStream:
    agent = torch.full((1, 1, config.n_agent, config.d_model), float(slot))
    return oracle.OracleStream(slot, 0, slot, 0, torch.empty(0), None, None, agent)


def test_oracle_trajectory_uses_step_successors_and_truth(monkeypatch):
    config = replace(Config(), device="cpu", horizon=2, actor_batch=2)
    heads = Heads(config)

    def step(stream, action, world, encoder, rng, cfg, seed_base):
        index = stream.index + 1
        agent = torch.full_like(stream.agent, float(10 * index + stream.slot))
        successor = replace(stream, index=index, agent=agent)
        reward = float(100 * index + stream.slot)
        continuation = float(not (stream.slot == 0 and index == 1))
        return successor, reward, continuation, not bool(continuation)

    monkeypatch.setattr(oracle, "_step_stream", step)
    trajectory, streams, metrics = oracle.oracle_trajectory(
        [_stream(0, config), _stream(1, config)],
        None,
        None,
        heads,
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        config,
        0,
    )

    assert trajectory.reward.tolist() == [[100.0, 200.0], [101.0, 201.0]]
    assert trajectory.continuation.tolist() == [[0.0, 1.0], [1.0, 1.0]]
    assert torch.equal(trajectory.agent[:, 1, 0, 0], torch.tensor([10.0, 11.0]))
    assert torch.equal(trajectory.agent[:, 2, 0, 0], torch.tensor([20.0, 21.0]))
    assert [stream.index for stream in streams] == [2, 2]
    assert metrics["terminal"] == 0.25


def test_oracle_actor_gradient_does_not_use_outcome_heads(monkeypatch):
    config = replace(Config(), device="cpu", horizon=2, actor_batch=2)
    heads = Heads(config)

    def step(stream, action, world, encoder, rng, cfg, seed_base):
        successor = replace(stream, index=stream.index + 1, agent=stream.agent + 0.1)
        return successor, float(action == 0), 1.0, False

    monkeypatch.setattr(oracle, "_step_stream", step)
    trajectory, _, _ = oracle.oracle_trajectory(
        [_stream(0, config), _stream(1, config)],
        None,
        None,
        heads,
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        config,
        0,
    )
    returns = lambda_returns(trajectory, config)
    prior = torch.zeros(2, 2, config.n_actions)
    loss = actor_loss(trajectory, returns, prior, config)
    loss = loss + critic_loss(
        heads(trajectory.agent[:, :-1])["value"], returns, heads.centers
    )
    loss.backward()

    assert heads.policy.weight.grad is not None
    assert heads.value.weight.grad is not None
    assert heads.reward.weight.grad is None
    assert heads.continuation.weight.grad is None
    assert all(parameter.grad is None for parameter in heads.model_body.parameters())


def test_oracle_trajectory_supports_horizon_sixteen(monkeypatch):
    config = replace(Config(), device="cpu", horizon=16, actor_batch=2)
    heads = Heads(config)
    calls = 0

    def step(stream, action, world, encoder, rng, cfg, seed_base):
        nonlocal calls
        calls += 1
        successor = replace(stream, index=stream.index + 1, agent=stream.agent + 0.1)
        return successor, float(action == 0), 1.0, False

    monkeypatch.setattr(oracle, "_step_stream", step)
    trajectory, streams, _ = oracle.oracle_trajectory(
        [_stream(0, config), _stream(1, config)],
        None,
        None,
        heads,
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        config,
        0,
    )

    assert trajectory.action.shape == (2, 16)
    assert trajectory.agent.shape[1] == 17
    assert [stream.index for stream in streams] == [16, 16]
    assert calls == 32
