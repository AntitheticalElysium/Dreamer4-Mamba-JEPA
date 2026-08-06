from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import TemporalState  # noqa: E402
from long_context_scale import (  # noqa: E402
    ProjectedGlobalGRUTemporal,
    ProjectedGlobalMambaTemporal,
    matched_gru_hidden,
    projected_gru_parameter_count,
    temporal_parameter_count,
)


def _clone_state(state: TemporalState) -> TemporalState:
    cache = state.cache
    if isinstance(cache, list):
        cache = [
            tuple(value.clone() for value in item)
            if isinstance(item, tuple) else item.clone()
            for item in cache
        ]
    else:
        cache = copy.deepcopy(cache)
    return TemporalState(cache, state.output.clone())


def _sequence_and_steps(core, x, state_dtype, autocast: bool):
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if autocast else torch.autocast("cuda", enabled=False)
    )
    with torch.no_grad(), context:
        sequence, _ = core.sequence(x)
        state = core.init_state(x.shape[0], x.shape[2], x.device, state_dtype)
        outputs = []
        for index in range(x.shape[1]):
            output, state = core.step(x[:, index], state)
            outputs.append(output)
    return sequence, torch.stack(outputs, 1), state


def test_projected_gru_count_formula_and_large_parameter_match():
    for dim, hidden, depth in ((16, 31, 1), (32, 48, 3), (64, 524, 2)):
        core = ProjectedGlobalGRUTemporal(dim=dim, hidden=hidden, depth=depth)
        assert temporal_parameter_count(core) == \
            projected_gru_parameter_count(dim, hidden, depth)

    if importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("official mamba_ssm is unavailable")
    mamba = ProjectedGlobalMambaTemporal()
    target = temporal_parameter_count(mamba)
    hidden = matched_gru_hidden(target)
    gru = ProjectedGlobalGRUTemporal(hidden=hidden)
    relative_error = abs(temporal_parameter_count(gru) - target) / target
    assert hidden == 524
    assert relative_error <= 0.005


def test_large_gru_t128_sequence_step_and_reset_isolation():
    torch.manual_seed(1601)
    core = ProjectedGlobalGRUTemporal(hidden=96, depth=2).eval()
    x = torch.randn(2, 128, 5, 64)
    sequence, stepped, state = _sequence_and_steps(
        core, x, torch.float32, autocast=False)
    torch.testing.assert_close(sequence, stepped, atol=1e-6, rtol=1e-6)

    probe = torch.randn(2, 5, 64)
    reset_output, reset_state = core.step(
        probe, _clone_state(state), torch.tensor([True, False]))
    plain_output, plain_state = core.step(probe, _clone_state(state))
    fresh = core.init_state(1, 5, probe.device, probe.dtype)
    fresh_output, _ = core.step(probe[:1], fresh)
    torch.testing.assert_close(reset_output[:1], fresh_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reset_output[1], plain_output[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reset_state.cache[0][1], plain_state.cache[0][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 AMP requires CUDA")
def test_actual_large_gru_t128_bf16_sequence_step_and_gradients():
    torch.manual_seed(1604)
    core = ProjectedGlobalGRUTemporal(hidden=524, depth=2).cuda().train()
    x = torch.randn(1, 128, 5, 64, device="cuda", requires_grad=True)
    sequence, stepped, _ = _sequence_and_steps(
        core.eval(), x.detach(), torch.float32, autocast=True)
    torch.testing.assert_close(
        sequence.float(), stepped.float(), atol=2e-3, rtol=2e-3)
    core.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = core.sequence(x)
        loss = output.float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in core.parameters()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official Mamba-2 CUDA kernels are required",
)
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_amp"])
def test_large_mamba_t128_sequence_step_and_cache_isolation(autocast):
    torch.manual_seed(1602)
    device = torch.device("cuda")
    # The actual width/depth/state configuration, with fewer spatial streams;
    # pooling makes stream count irrelevant to the recurrent calculation.
    core = ProjectedGlobalMambaTemporal().to(device).eval()
    x = torch.randn(2, 128, 5, 64, device=device)
    sequence, stepped, state = _sequence_and_steps(
        core, x, torch.float32, autocast=autocast)
    tolerance = 0.06 if autocast else 3e-3
    torch.testing.assert_close(
        sequence.float(), stepped.float(), atol=tolerance, rtol=tolerance)
    assert torch.isfinite(sequence).all() and torch.isfinite(stepped).all()

    # Official kernels update cache tensors in place. Branching therefore
    # requires deep clones; verify both the clone and row-local reset contract.
    probe = torch.randn(2, 5, 64, device=device)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if autocast else torch.autocast("cuda", enabled=False)
    )
    with torch.no_grad(), context:
        reset_output, reset_state = core.step(
            probe, _clone_state(state), torch.tensor([True, False], device=device))
        plain_output, plain_state = core.step(probe, _clone_state(state))
        fresh = core.init_state(1, 5, device, torch.float32)
        fresh_output, _ = core.step(probe[:1], fresh)
    torch.testing.assert_close(
        reset_output[:1].float(), fresh_output.float(), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        reset_output[1].float(), plain_output[1].float(),
        atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        reset_state.cache[0][1][1].float(), plain_state.cache[0][1][1].float(),
        atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official Mamba-2 CUDA kernels are required",
)
def test_large_mamba_t128_bf16_gradients_are_finite():
    torch.manual_seed(1603)
    core = ProjectedGlobalMambaTemporal().cuda().train()
    x = torch.randn(1, 128, 5, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = core.sequence(x)
        loss = output.float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in core.parameters()
    )


def test_registered_scale_interaction_readout_uses_environment_signs():
    from run_long_context_scale import _registered_readout

    def result(separation, patch, retrieval, per_env):
        return {
            "step": 2_000,
            "separation_all": separation,
            "separation_patch": patch,
            "retrieval_tie": retrieval,
            "per_env_seed": {
                str(seed): {"separation_all": value}
                for seed, value in zip((111, 112, 113, 114), per_env)
            },
        }

    values = {
        "LS-G64": result(0.0100, 0.0100, 0.30, [0.010] * 4),
        "LS-M64": result(0.0110, 0.0110, 0.31, [0.011] * 4),
        "LL-G": result(0.0100, 0.0100, 0.30, [0.010] * 4),
        "LL-M": result(0.0120, 0.0120, 0.32,
                       [0.012, 0.012, 0.012, 0.009]),
    }
    report = {
        "arms": {
            arm: {
                "monitor": [entry],
                "peak_vram_reserved_mib": 400.0,
            }
            for arm, entry in values.items()
        }
    }
    readout = _registered_readout(report, 2_000)
    assert readout["delta_small"] == pytest.approx(0.001)
    assert readout["delta_large"] == pytest.approx(0.002)
    assert readout["interaction"] == pytest.approx(0.001)
    assert sum(
        value > 0
        for value in readout["delta_large_by_environment_seed"].values()
    ) == 3
    assert readout["licenses_confirmatory_replication"]
