from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import M3HJWM, ModelConfig  # noqa: E402


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
