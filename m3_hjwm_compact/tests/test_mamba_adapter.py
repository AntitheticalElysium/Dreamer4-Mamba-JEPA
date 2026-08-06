from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official mamba_ssm and CUDA are required",
)


def mamba2_model() -> M3HJWM:
    cfg = ModelConfig(
        image_size=32,
        patch_size=16,
        token_dim=64,
        registers=1,
        spatial_heads=4,
        spatial_depth=1,
        temporal_backend="mamba2",
        temporal_depth=1,
        mamba_d_state=32,
        mamba_headdim=16,
        predictor="deterministic",
        predictor_depth=1,
    )
    return M3HJWM(cfg).cuda().to(torch.bfloat16)


def test_mamba2_sequence_step_equivalence_and_cache_layout():
    torch.manual_seed(31)
    model = mamba2_model().eval()
    temporal = model.temporal
    x = torch.randn(
        2,
        8,
        model.streams,
        model.cfg.token_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        sequence, _ = temporal.sequence(x)
        state = temporal.init_state(2, model.streams, x.device, x.dtype)
        assert len(state.cache) == model.cfg.temporal_depth
        conv_state, ssm_state = state.cache[0]
        assert conv_state.shape[0] == 2 * model.streams
        assert ssm_state.shape[0] == 2 * model.streams
        outputs = []
        for index in range(x.shape[1]):
            output, state = temporal.step(x[:, index], state)
            outputs.append(output)
        stepped = torch.stack(outputs, 1)
        torch.cuda.synchronize()
    torch.testing.assert_close(sequence.float(), stepped.float(), atol=0.05, rtol=0.05)
    assert torch.isfinite(stepped).all()


def test_mamba2_reset_isolation():
    torch.manual_seed(37)
    model = mamba2_model().eval()
    temporal = model.temporal
    x0 = torch.randn(2, model.streams, 64, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn_like(x0)
    with torch.no_grad():
        state = temporal.init_state(2, model.streams, x0.device, x0.dtype)
        _, state = temporal.step(x0, state)
        reset_output, _ = temporal.step(x1, state, torch.tensor([True, False], device="cuda"))

        fresh = temporal.init_state(1, model.streams, x0.device, x0.dtype)
        fresh_output, _ = temporal.step(x1[:1], fresh)
        torch.cuda.synchronize()
    torch.testing.assert_close(
        reset_output[:1].float(), fresh_output.float(), atol=0.05, rtol=0.05
    )


def test_mamba2_mixed_precision_sequence_gradients_are_finite():
    torch.manual_seed(41)
    model = mamba2_model().train()
    x = torch.randn(
        2,
        8,
        model.streams,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output, _ = model.temporal.sequence(x)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.temporal.parameters()
    )


def test_mamba2_fp32_deployment_cache_matches_autocast_sequence():
    """Production initial_state defaults to fp32 even when calls use bf16 AMP."""
    torch.manual_seed(43)
    model = mamba2_model().float().eval()
    x = torch.randn(2, 8, model.streams, 64, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sequence, _ = model.temporal.sequence(x)
        state = model.temporal.init_state(2, model.streams, x.device, torch.float32)
        outputs = []
        for index in range(x.shape[1]):
            output, state = model.temporal.step(x[:, index], state)
            outputs.append(output)
        stepped = torch.stack(outputs, 1)
    torch.testing.assert_close(sequence.float(), stepped.float(), atol=0.05, rtol=0.05)


def test_mamba2_rollout_bridge_is_finite_and_differentiable():
    torch.manual_seed(47)
    model = mamba2_model().float().train()
    cfg = model.cfg
    data = {
        "obs": torch.randint(
            0, 256, (2, 4, 3, cfg.image_size, cfg.image_size),
            dtype=torch.uint8, device="cuda",
        ),
        "actions": torch.randint(0, cfg.action_dim, (2, 3), device="cuda"),
        "rewards": torch.zeros(2, 3, device="cuda"),
        "continues": torch.ones(2, 3, device="cuda"),
    }
    weights = LossConfig(
        jepa=0.0, mode_router=0.0, mode_balance=0.0, reward=0.0,
        continuation=0.0, variance=0.0, covariance=0.0, rollout=1.0,
        manifold=0.0, energy=0.0,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(data, weights)
    output.loss.backward()
    assert torch.isfinite(output.metrics["rollout"])
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.temporal.parameters()
    )
