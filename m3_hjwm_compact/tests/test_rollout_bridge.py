"""Contracts for the optional V-JEPA-2-AC-shaped rollout bridge."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402


def config(**overrides):
    values = dict(
        image_size=32,
        patch_size=16,
        token_dim=16,
        registers=1,
        spatial_heads=2,
        spatial_depth=1,
        temporal_backend="gru",
        temporal_depth=1,
        predictor="deterministic",
        predictor_depth=1,
        mask_ratio=0.0,
        rollout_steps=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def batch(cfg, batch_size=2, observations=4):
    generator = torch.Generator().manual_seed(17)
    return {
        "obs": torch.randint(
            0,
            256,
            (batch_size, observations, 3, cfg.image_size, cfg.image_size),
            dtype=torch.uint8,
            generator=generator,
        ),
        "actions": torch.randint(
            0,
            cfg.action_dim,
            (batch_size, observations - 1),
            generator=generator,
        ),
        "rewards": torch.zeros(batch_size, observations - 1),
        "continues": torch.ones(batch_size, observations - 1),
    }


def rollout_only():
    return LossConfig(
        jepa=0.0,
        mode_router=0.0,
        mode_balance=0.0,
        reward=0.0,
        continuation=0.0,
        variance=0.0,
        covariance=0.0,
        rollout=1.0,
        manifold=0.0,
        energy=0.0,
    )


def test_rollout_bridge_is_on_by_default():
    # 2026-07-15: the "opt-in pending Phase B/D evidence" condition was met
    # (S3-B' passed 3/3 seeds at both scales); defaults now encode the
    # validated recipe.
    assert LossConfig().rollout > 0


def test_rollout_uses_generated_middle_state_and_final_real_target_only():
    cfg = config()
    torch.manual_seed(19)
    model = M3HJWM(cfg).eval()
    original = batch(cfg)

    # With T_roll=2 and four observations, o_0,o_1 form the real prefix,
    # o_2 must be generated, and o_3 is the final target.
    changed_middle = {key: value.clone() for key, value in original.items()}
    changed_middle["obs"][:, 2] = 255 - changed_middle["obs"][:, 2]
    changed_final = {key: value.clone() for key, value in original.items()}
    changed_final["obs"][:, 3] = 255 - changed_final["obs"][:, 3]

    first = model(original, rollout_only()).metrics["rollout"]
    middle = model(changed_middle, rollout_only()).metrics["rollout"]
    final = model(changed_final, rollout_only()).metrics["rollout"]
    torch.testing.assert_close(first, middle)
    assert not torch.allclose(first, final)


def test_rollout_gradient_crosses_predictor_temporal_composition_but_stops_at_encoder():
    cfg = config()
    torch.manual_seed(23)
    model = M3HJWM(cfg)
    output = model(batch(cfg), rollout_only())
    output.loss.backward()

    temporal_grad = sum(
        (parameter.grad.abs().sum() for parameter in model.temporal.parameters() if parameter.grad is not None),
        torch.tensor(0.0),
    )
    predictor_grad = sum(
        (parameter.grad.abs().sum() for parameter in model.future.parameters() if parameter.grad is not None),
        torch.tensor(0.0),
    )
    assert float(temporal_grad) > 0
    assert float(predictor_grad) > 0
    # Zero-weight teacher-forced terms may materialize exact-zero gradients, but
    # the rollout auxiliary itself must not update the visual encoder.
    assert all(
        parameter.grad is None or not bool(parameter.grad.abs().any())
        for parameter in model.online_encoder.parameters()
    )


def test_rollout_loss_decreases_on_a_fixed_tiny_batch():
    cfg = config()
    torch.manual_seed(29)
    model = M3HJWM(cfg)
    data = batch(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        output = model(data, rollout_only())
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
        model.mark_parameters_updated()
        losses.append(float(output.metrics["rollout"]))
    assert sum(losses[-5:]) / 5 < 0.6 * (sum(losses[:5]) / 5)
