"""Anti-collapse regularization tests (VICReg terms, arXiv:2105.04906).

Motivated by the 2026-07-12 rank-collapse controls: every unregularized variant of
the JEPA objective collapsed to effective rank ~3 within 300 updates; the VICReg
variance+covariance terms held rank at ~16-19. These tests pin the formula, the
gradient path to the encoder, the defaults, and the mask_ratio=0 control switch.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (  # noqa: E402
    LossConfig,
    M3HJWM,
    ModelConfig,
    variance_covariance_losses,
)


def small_config(**overrides) -> ModelConfig:
    defaults = dict(temporal_backend="gru", predictor="deterministic")
    defaults.update(overrides)
    return ModelConfig(**defaults)


def batch(b=2, t=3, cfg=None):
    cfg = cfg or small_config()
    torch.manual_seed(5)
    return {
        "obs": torch.randint(0, 255, (b, t, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8),
        "actions": torch.randint(0, cfg.action_dim, (b, t - 1)),
        "rewards": torch.randn(b, t - 1),
        "continues": torch.ones(b, t - 1),
    }


def test_collapsed_embeddings_are_maximally_penalized():
    constant = torch.ones(64, 7, 16)
    variance, covariance = variance_covariance_losses(constant)
    # std == 0 in every dim -> hinge saturates at gamma - sqrt(eps) = 1 - 0.01
    assert variance.item() == pytest.approx(0.99, abs=1e-3)
    assert covariance.item() == pytest.approx(0.0, abs=1e-6)


def test_whitened_embeddings_are_not_penalized():
    torch.manual_seed(0)
    x = torch.randn(4096, 16)
    x = (x - x.mean(0)) / x.std(0)
    variance, covariance = variance_covariance_losses(x)
    assert variance.item() < 0.02
    assert covariance.item() < 0.02


def test_anti_collapse_is_on_by_default_and_reported():
    weights = LossConfig()
    assert weights.variance > 0 and weights.covariance > 0
    model = M3HJWM(small_config())
    output = model(batch())
    assert "variance" in output.metrics and "covariance" in output.metrics
    assert float(output.metrics["variance"]) > 0


def test_variance_loss_reaches_encoder_parameters():
    model = M3HJWM(small_config())
    silent = LossConfig(
        jepa=0.0, mode_router=0.0, mode_balance=0.0, reward=0.0,
        continuation=0.0, variance=1.0, covariance=0.04,
    )
    output = model(batch(), silent)
    output.loss.backward()
    stem_grads = [
        p.grad.abs().sum()
        for p in model.online_encoder.stem.parameters()
        if p.grad is not None
    ]
    assert stem_grads and float(sum(stem_grads)) > 0


def test_mask_ratio_zero_disables_stochastic_masking():
    data = batch()
    masked = M3HJWM(small_config())
    torch.manual_seed(11)
    first = masked(data).metrics["jepa"]
    second = masked(data).metrics["jepa"]
    assert not torch.allclose(first, second), "masked objective should differ across random masks"

    unmasked = M3HJWM(small_config(mask_ratio=0.0))
    torch.manual_seed(11)
    first = unmasked(data).metrics["jepa"]
    second = unmasked(data).metrics["jepa"]
    assert torch.allclose(first, second), "unmasked objective must be deterministic"


def test_mask_ratio_bounds_validated():
    with pytest.raises(ValueError):
        ModelConfig(temporal_backend="gru", mask_ratio=1.0).validate()
    with pytest.raises(ValueError):
        ModelConfig(temporal_backend="gru", mask_ratio=-0.1).validate()
