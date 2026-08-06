import torch
import torch.nn.functional as F

from d4mj.actor_critic import actor_loss, critic_loss, lambda_returns
from d4mj.agent import _centers
from d4mj.imagination import Trajectory


def trajectory(reward, value, continuation, logits=None, action=None):
    horizon = reward.shape[1]
    if logits is None:
        logits = torch.zeros(reward.shape[0], horizon, 4)
    if action is None:
        action = torch.zeros(reward.shape[0], horizon, dtype=torch.long)
    return Trajectory(
        agent=torch.zeros(reward.shape[0], horizon + 1, 2, 8),
        action=action,
        logits=logits,
        reward=reward,
        continuation=continuation,
        value=value,
    )


def test_returns_use_the_next_index_convention(config):
    """G_t = r_{t+1} + gamma c_{t+1} [(1-lam) v_{t+1} + lam G_{t+1}]. Equation 10's
    printed same-index form shifts every critic target by one step."""
    reward = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[10.0, 20.0, 30.0]])
    continuation = torch.ones(1, 2)
    got = lambda_returns(trajectory(reward, value, continuation), config)

    last = reward[0, 1] + config.gamma * ((1 - config.lam) * value[0, 2] + config.lam * value[0, 2])
    first = reward[0, 0] + config.gamma * ((1 - config.lam) * value[0, 1] + config.lam * last)
    assert torch.allclose(got, torch.tensor([[first, last]]), atol=1e-5)


def test_termination_stops_the_bootstrap(config):
    """A terminal step must not carry value past itself."""
    reward = torch.tensor([[1.0, 1.0]])
    value = torch.full((1, 3), 50.0)
    dead = lambda_returns(trajectory(reward, value, torch.zeros(1, 2)), config)
    alive = lambda_returns(trajectory(reward, value, torch.ones(1, 2)), config)
    assert torch.allclose(dead, torch.tensor([[1.0, 1.0]]))
    assert (alive > dead).all()


def test_actor_uses_only_the_sign_of_the_advantage(config):
    """PMPO discards magnitude, so scaling the advantage must not move the loss."""
    reward = torch.tensor([[1.0, 1.0]])
    continuation = torch.ones(1, 2)
    logits = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(0))
    prior = torch.zeros(1, 2, 4)

    small = trajectory(reward, torch.tensor([[0.0, 0.0, 0.0]]), continuation, logits)
    big = trajectory(reward * 1000, torch.tensor([[0.0, 0.0, 0.0]]), continuation, logits)
    one = actor_loss(small, lambda_returns(small, config), prior, config)
    other = actor_loss(big, lambda_returns(big, config), prior, config)
    assert torch.allclose(one, other, atol=1e-6)


def test_actor_is_finite_when_the_prior_saturates(config):
    """`prior.log()` on a softmax underflows to -inf once the logits separate, which
    turns the whole actor loss NaN; the prior must enter as log-probabilities."""
    reward = torch.ones(1, 2)
    value = torch.zeros(1, 3)
    prior = torch.tensor([[[80.0, -80.0, -80.0, -80.0]] * 2])
    traj = trajectory(reward, value, torch.ones(1, 2))
    assert torch.isfinite(actor_loss(traj, lambda_returns(traj, config), prior, config))


def test_critic_target_is_symlog_two_hot(config):
    centers = _centers(config)
    returns = torch.tensor([[0.0, 5.0]])
    perfect = torch.log(
        torch.zeros(1, 2, len(centers)).scatter(
            -1, torch.tensor([[[len(centers) // 2]], [[len(centers) // 2]]]).view(1, 2, 1), 1.0
        ).clamp(min=1e-30)
    )
    loss = critic_loss(perfect, returns, centers)
    assert torch.isfinite(loss) and loss > 0


def test_masked_mean_is_zero_on_an_empty_set(config):
    """An all-negative-advantage batch must not divide by zero and surface as a
    silent NaN in the actor gradient."""
    reward = torch.full((1, 2), -1.0)
    value = torch.zeros(1, 3)
    traj = trajectory(reward, value, torch.ones(1, 2))
    assert torch.isfinite(actor_loss(traj, lambda_returns(traj, config), torch.zeros(1, 2, 4), config))
