from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from agent import ActorCritic  # noqa: E402
from data import CrafterAdapter, Episode, EpisodeReplay  # noqa: E402
from model import FuturePredictor, LossConfig, M3HJWM, ModelConfig  # noqa: E402
from train import TrainConfig, actor_critic_update, world_update  # noqa: E402


def tiny_config(**overrides) -> ModelConfig:
    values = dict(
        image_size=32,
        patch_size=16,
        token_dim=16,
        registers=1,
        spatial_heads=2,
        spatial_depth=1,
        temporal_backend="gru",
        temporal_depth=1,
        predictor_depth=1,
        predictor="mixture",
        modes=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def random_batch(cfg: ModelConfig, batch: int = 1, observations: int = 4):
    return {
        "obs": torch.randint(
            0,
            256,
            (batch, observations, 3, cfg.image_size, cfg.image_size),
            dtype=torch.uint8,
        ),
        "actions": torch.randint(0, cfg.action_dim, (batch, observations - 1)),
        "rewards": torch.randn(batch, observations - 1),
        "continues": torch.ones(batch, observations - 1),
    }


def test_no_future_observation_leakage():
    torch.manual_seed(7)
    cfg = tiny_config()
    model = M3HJWM(cfg).eval()
    first = random_batch(cfg)
    second = {name: value.clone() for name, value in first.items()}
    second["obs"][:, 3] = 255 - second["obs"][:, 3]

    torch.manual_seed(11)
    out_a = model(first)
    torch.manual_seed(11)
    out_b = model(second)

    # Changing o_3 cannot alter c_0, c_1, or c_2.
    torch.testing.assert_close(out_a.context[:, :3], out_b.context[:, :3])


def test_action_t_is_not_visible_in_context_t():
    torch.manual_seed(13)
    cfg = tiny_config()
    model = M3HJWM(cfg).eval()
    first = random_batch(cfg)
    second = {name: value.clone() for name, value in first.items()}
    second["actions"][:, 1] = (second["actions"][:, 1] + 1) % cfg.action_dim

    torch.manual_seed(17)
    out_a = model(first)
    torch.manual_seed(17)
    out_b = model(second)

    # a_1 is an explicit input to the predictor for o_2, but only becomes the
    # temporal model's previous action at c_2. It cannot leak into c_0 or c_1.
    torch.testing.assert_close(out_a.context[:, :2], out_b.context[:, :2])


def test_action_t_affects_reward_t_plus_1_but_not_earlier_reward():
    torch.manual_seed(15)
    cfg = tiny_config(predictor="deterministic")
    model = M3HJWM(cfg).eval()
    first = random_batch(cfg, observations=4)
    first["actions"].zero_()
    second = {name: value.clone() for name, value in first.items()}
    # a_1 causes the second recorded transition reward r_2.
    second["actions"][:, 1] = 1

    torch.manual_seed(21)
    out_a = model(first)
    torch.manual_seed(21)
    out_b = model(second)
    torch.testing.assert_close(out_a.reward_logits[:, 0], out_b.reward_logits[:, 0])
    assert not torch.equal(out_a.reward_logits[:, 1], out_b.reward_logits[:, 1])


def test_target_encoder_is_a_disjoint_frozen_full_copy():
    model = M3HJWM(tiny_config())
    online = dict(model.online_encoder.named_parameters())
    target = dict(model.target_encoder.model.named_parameters())
    assert online.keys() == target.keys()
    for name in online:
        assert not target[name].requires_grad
        assert online[name].data_ptr() != target[name].data_ptr()
        torch.testing.assert_close(online[name], target[name])


def test_target_encoder_stays_in_eval_mode():
    model = M3HJWM(tiny_config())
    model.train()
    assert not model.target_encoder.model.training


def test_router_conditions_on_action():
    torch.manual_seed(19)
    cfg = tiny_config()
    predictor = FuturePredictor(cfg)
    context = torch.randn(1, 5, cfg.token_dim).expand(2, -1, -1).clone()
    horizon = torch.ones(2, dtype=torch.long)
    actions = torch.tensor([0, 1])
    _, logits = predictor.all_predictions(context, actions, horizon)
    assert not torch.equal(logits[0], logits[1])


def test_balance_loss_has_a_gradient_path_to_predictor_heads():
    torch.manual_seed(23)
    cfg = tiny_config(modes=3)
    predictor = FuturePredictor(cfg)
    context = torch.randn(8, 5, cfg.token_dim)
    target = torch.randn_like(context)
    action = torch.randint(0, cfg.action_dim, (8,))
    horizon = torch.ones(8, dtype=torch.long)
    output = predictor(context, action, horizon, target)
    assert output.balance_loss.requires_grad
    (output.regression + output.balance_loss).backward()
    assert predictor.mode_embed.grad is not None
    assert predictor.mode_embed.grad.abs().sum() > 0


def test_reliability_auxiliaries_are_shadow_only_by_default():
    weights = LossConfig()
    assert weights.manifold == 0.0
    assert weights.energy == 0.0


class _FakeCrafter:
    def __init__(self, result):
        self.result = result

    def step(self, action):
        return self.result


@pytest.mark.parametrize(
    "result, expected",
    [
        ((np.zeros((64, 64, 3), np.uint8), 0.0, True, {"discount": 1.0}), 1.0),
        ((np.zeros((64, 64, 3), np.uint8), 0.0, True, {"discount": 0.0}), 0.0),
        (
            (
                np.zeros((64, 64, 3), np.uint8),
                0.0,
                False,
                True,
                {"discount": 1.0},
            ),
            1.0,
        ),
    ],
)
def test_crafter_continuation_uses_environment_discount(result, expected):
    adapter = CrafterAdapter.__new__(CrafterAdapter)
    adapter.env = _FakeCrafter(result)
    _, _, continuation, _ = adapter.step(0)
    assert continuation == expected


def test_replay_rejects_misaligned_transition_arrays():
    replay = EpisodeReplay()
    episode = Episode(
        obs=np.zeros((3, 3, 8, 8), np.uint8),
        actions=np.zeros(2, np.int64),
        rewards=np.zeros(1, np.float32),
        continues=np.ones(2, np.float32),
    )
    with pytest.raises(ValueError, match="actions/rewards/continues"):
        replay.add(episode)


def test_world_update_invalidates_pre_update_recurrent_state():
    cfg = tiny_config()
    model = M3HJWM(cfg)
    state = model.initial_state(1, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    world_update(
        model,
        random_batch(cfg, observations=3),
        optimizer,
        TrainConfig(batch_size=1, sequence_length=3, amp=False),
    )
    with pytest.raises(RuntimeError, match="stale"):
        model.imagine_step(state, torch.zeros(1, dtype=torch.long))


def test_actor_critic_update_does_not_backpropagate_into_world_model():
    torch.manual_seed(29)
    cfg = tiny_config()
    model = M3HJWM(cfg)
    actor_critic = ActorCritic(cfg.token_dim, cfg.action_dim, critics=2)
    state = model.initial_state(1, torch.device("cpu"))
    obs = torch.randint(0, 256, (1, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8)
    state = model.observe_step(obs, torch.zeros(1, dtype=torch.long), state)
    for parameter in model.parameters():
        parameter.grad = None

    actor_optimizer = torch.optim.AdamW(actor_critic.actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.AdamW(actor_critic.critics.parameters(), lr=1e-4)
    actor_critic_update(
        model,
        actor_critic,
        state,
        actor_optimizer,
        critic_optimizer,
        TrainConfig(imagination_horizon=2, amp=False),
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_auto_temporal_backend_never_silently_substitutes_an_architecture():
    with pytest.raises(ValueError, match="explicit"):
        M3HJWM(tiny_config(temporal_backend="auto"))
