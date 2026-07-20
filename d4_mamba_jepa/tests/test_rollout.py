import importlib.util

import pytest
import torch

from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.rollout import (
    categorical_random_shooting,
    sample_next_packed,
    score_action_plans,
    shortcut_schedule,
)
from d4_mamba_jepa.tests.test_baseline import tiny_config


def test_shortcut_schedule_contract():
    schedule = shortcut_schedule(4, 4)
    assert schedule["K"] == 4
    assert schedule["e"] == 2
    assert schedule["tau"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert schedule["tau_index"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="power of two"):
        shortcut_schedule(4, 3)


def test_transformer_cached_and_uncached_one_step_match():
    torch.manual_seed(47)
    cfg = tiny_config()
    world = D4LiteWorld(cfg).eval()
    context = torch.randn(2, 3, cfg.n_spatial, cfg.d_spatial)
    actions = torch.tensor([[-1, 0, 1, 2], [-1, 3, 4, 5]])
    schedule = shortcut_schedule(cfg.k_max, cfg.k_max)
    first_generator = torch.Generator().manual_seed(101)
    second_generator = torch.Generator().manual_seed(101)
    uncached_z, uncached_h = sample_next_packed(
        world,
        past_packed=context,
        led_to_actions=actions,
        schedule=schedule,
        use_cache=False,
        generator=first_generator,
    )
    cached_z, cached_h = sample_next_packed(
        world,
        past_packed=context,
        led_to_actions=actions,
        schedule=schedule,
        use_cache=True,
        generator=second_generator,
    )
    torch.testing.assert_close(uncached_z, cached_z, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(uncached_h, cached_h, atol=1e-5, rtol=1e-5)


def test_random_shooting_covers_actions_and_returns_finite_scores():
    torch.manual_seed(53)
    cfg = tiny_config()
    world = D4LiteWorld(cfg).eval()
    context = torch.randn(1, 2, cfg.n_spatial, cfg.d_spatial)
    actions = torch.tensor([[-1, 0]])
    schedule = shortcut_schedule(cfg.k_max, cfg.k_max)
    generator = torch.Generator().manual_seed(103)
    result = categorical_random_shooting(
        world,
        context_packed=context,
        context_led_to_actions=actions,
        horizon=2,
        candidates=34,
        schedule=schedule,
        use_cache=True,
        generator=generator,
    )
    assert set(result.plans[:, 0].tolist()) == set(range(cfg.n_actions))
    assert 0 <= result.action < cfg.n_actions
    assert torch.isfinite(result.scores).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official mamba_ssm and CUDA are required",
)
def test_mamba_cached_and_uncached_one_step_match():
    torch.manual_seed(59)
    cfg = tiny_config(temporal_backend="mamba2")
    world = D4LiteWorld(cfg).cuda().to(torch.bfloat16).eval()
    context = torch.randn(
        1,
        3,
        cfg.n_spatial,
        cfg.d_spatial,
        device="cuda",
        dtype=torch.bfloat16,
    )
    actions = torch.tensor([[-1, 0, 1, 2]], device="cuda")
    schedule = shortcut_schedule(cfg.k_max, cfg.k_max)
    first_generator = torch.Generator(device="cuda").manual_seed(107)
    second_generator = torch.Generator(device="cuda").manual_seed(107)
    uncached_z, uncached_h = sample_next_packed(
        world,
        past_packed=context,
        led_to_actions=actions,
        schedule=schedule,
        use_cache=False,
        generator=first_generator,
    )
    cached_z, cached_h = sample_next_packed(
        world,
        past_packed=context,
        led_to_actions=actions,
        schedule=schedule,
        use_cache=True,
        generator=second_generator,
    )
    torch.testing.assert_close(
        uncached_z.float(), cached_z.float(), atol=0.08, rtol=0.08
    )
    torch.testing.assert_close(
        uncached_h.float(), cached_h.float(), atol=0.08, rtol=0.08
    )
