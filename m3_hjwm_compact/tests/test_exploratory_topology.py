"""Contracts for the 2026-07-17 exploratory topology/conditioning screen."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))


def test_flattened_gru_sequence_step_equivalence_and_no_bypass():
    from exploratory_topology import FlattenedGRUTemporal

    torch.manual_seed(3)
    core = FlattenedGRUTemporal(dim=8, streams=5, hidden=32, depth=2)
    x = torch.randn(2, 6, 5, 8)
    seq, _ = core.sequence(x)
    state = core.init_state(2, 5, x.device, x.dtype)
    outs = []
    for t in range(6):
        y, state = core.step(x[:, t], state)
        outs.append(y)
    torch.testing.assert_close(seq, torch.stack(outs, 1), atol=1e-5, rtol=1e-5)
    # NO dense bypass: zeroing the output projection must zero the output
    # entirely (in the pooled arms, out = x + proj(h) would leave x intact).
    with torch.no_grad():
        core.out_proj.weight.zero_()
        core.out_proj.bias.zero_()
    y, _ = core.step(x[:, 0], core.init_state(2, 5, x.device, x.dtype))
    assert torch.count_nonzero(y) == 0, "flattened core must not pass input through"


def test_flattened_gru_parameter_count_is_exact():
    from exploratory_topology import (
        FlattenedGRUTemporal, flattened_gru_parameter_count)

    core = FlattenedGRUTemporal(dim=8, streams=5, hidden=32, depth=2)
    actual = sum(p.numel() for p in core.parameters())
    assert actual == flattened_gru_parameter_count(8, 5, 32, 2)


def test_flattened_mamba_contract():
    if not torch.cuda.is_available():
        pytest.skip("official Mamba kernels need CUDA")
    from exploratory_topology import FlattenedMambaTemporal

    device = torch.device("cuda")
    torch.manual_seed(4)
    core = FlattenedMambaTemporal(dim=8, streams=5, hidden=64, depth=2,
                                  d_state=64, headdim=32).to(device)
    x = torch.randn(2, 6, 5, 8, device=device)
    seq, _ = core.sequence(x)
    state = core.init_state(2, 5, device, torch.float32)
    outs = []
    for t in range(6):
        y, state = core.step(x[:, t], state)
        outs.append(y)
    assert torch.allclose(seq, torch.stack(outs, 1), atol=2e-2), \
        f"seq/step divergence {(seq - torch.stack(outs, 1)).abs().max():.4f}"
    with torch.no_grad():
        core.out_proj.weight.zero_()
        core.out_proj.bias.zero_()
    y, _ = core.step(x[:, 0], core.init_state(2, 5, device, torch.float32))
    assert torch.count_nonzero(y) == 0


def test_adaln_block_is_identity_at_init_and_trains():
    from exploratory_topology import ConditionalSpatialBlock

    torch.manual_seed(5)
    block = ConditionalSpatialBlock(dim=16, heads=2)
    x = torch.randn(3, 7, 16)
    c = torch.randn(3, 16)
    out = block(x, c)
    torch.testing.assert_close(out, x)   # zero-init gates => exact identity
    loss = (block(x, c + 1.0) - torch.randn_like(x)).pow(2).mean()
    loss.backward()
    grad = block.adaLN[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_exploratory_worlds_build_and_forward():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (encoder checkpoint + mamba kernels)")
    from exploratory_topology import build_exploratory_world

    device = torch.device("cuda")
    batch = {
        "obs": torch.randint(0, 255, (2, 4, 3, 64, 64), dtype=torch.uint8,
                             device=device),
        "actions": torch.randint(0, 17, (2, 3), device=device),
        "rewards": torch.randn(2, 3, device=device),
        "continues": torch.ones(2, 3, device=device),
    }
    temporal_params = {}
    for arm in ("X-FLG", "X-FLM", "X-ADA"):
        world = build_exploratory_world(arm, seed=505, device=device)
        out = world(batch)
        assert torch.isfinite(out.loss), arm
        temporal_params[arm] = sum(
            p.numel() for n, p in world.named_parameters()
            if n.startswith("temporal."))
        del world
        torch.cuda.empty_cache()
    gap = abs(temporal_params["X-FLG"] - temporal_params["X-FLM"])
    assert gap / temporal_params["X-FLM"] < 0.005, \
        f"flattened arms not parameter-matched: {temporal_params}"
