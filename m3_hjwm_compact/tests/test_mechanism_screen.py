"""Contracts for the 2026-07-17 mechanism screen + companion audit item 7
(actual-shape BF16 regression)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))


def test_bypass_variant_restores_input_path():
    from mechanism_screen import BypassFlattenedGRUTemporal

    torch.manual_seed(7)
    core = BypassFlattenedGRUTemporal(dim=8, streams=5, hidden=32, depth=2)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        core.out_proj.weight.zero_()
        core.out_proj.bias.zero_()
    y, _ = core.step(x, core.init_state(2, 5, x.device, x.dtype))
    torch.testing.assert_close(y, x)   # bypass present: zeroed proj => y == x
    # sequence path inherits the bypass through the overridden step
    seq, _ = core.sequence(x[:, None].expand(2, 3, 5, 8).contiguous())
    torch.testing.assert_close(seq[:, 0], x)


def test_feedforward_control_has_no_temporal_memory():
    from mechanism_screen import (
        FullGridFeedforward, feedforward_parameter_count)

    torch.manual_seed(8)
    core = FullGridFeedforward(dim=8, streams=5, hidden=32, depth=2)
    assert sum(p.numel() for p in core.parameters()) == \
        feedforward_parameter_count(8, 5, 32, 2)
    a = torch.randn(2, 5, 8)
    b = torch.randn(2, 5, 8)
    state = core.init_state(2, 5, a.device, a.dtype)
    # walk two different histories, then present the same current input
    _, s1 = core.step(a, state)
    _, s2 = core.step(b, core.init_state(2, 5, a.device, a.dtype))
    y1, _ = core.step(a, s1)
    y2, _ = core.step(a, s2)
    torch.testing.assert_close(y1, y2)   # output independent of history
    seq, _ = core.sequence(torch.stack([a, b], 1))
    y_a, _ = core.step(a, core.init_state(2, 5, a.device, a.dtype))
    torch.testing.assert_close(seq[:, 0], y_a)


def test_mechanism_worlds_build_param_matched_and_finite():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (encoder checkpoint)")
    from mechanism_screen import build_mechanism_world
    from exploratory_topology import build_exploratory_world

    device = torch.device("cuda")
    batch = {
        "obs": torch.randint(0, 255, (2, 4, 3, 64, 64), dtype=torch.uint8,
                             device=device),
        "actions": torch.randint(0, 17, (2, 3), device=device),
        "rewards": torch.randn(2, 3, device=device),
        "continues": torch.ones(2, 3, device=device),
    }
    target = sum(p.numel() for n, p in
                 build_exploratory_world("X-FLG", 505, device).named_parameters()
                 if n.startswith("temporal."))
    torch.cuda.empty_cache()
    for arm in ("MS-PC", "MS-FB", "MS-FF"):
        world = build_mechanism_world(arm, seed=505, device=device)
        out = world(batch)
        assert torch.isfinite(out.loss), arm
        temporal = sum(p.numel() for n, p in world.named_parameters()
                       if n.startswith("temporal."))
        assert abs(temporal - target) / target < 0.01, \
            f"{arm} not parameter-matched: {temporal} vs {target}"
        del world
        torch.cuda.empty_cache()


def test_actual_shape_bf16_sequence_step_discrepancy_pinned():
    """Companion audit item 7: the tiny-fp32 contract test does not bound the
    REAL model shape under BF16 autocast. Pin the actual-shape discrepancy
    (its measured values: mean 0.00365, max 0.02344)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from exploratory_topology import FlattenedMambaTemporal

    device = torch.device("cuda")
    torch.manual_seed(11)
    core = FlattenedMambaTemporal(dim=64, streams=66).to(device)
    x = torch.randn(2, 16, 66, 64, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seq, _ = core.sequence(x)
        state = core.init_state(2, 66, device, torch.float32)
        outs = []
        for t in range(16):
            y, state = core.step(x[:, t], state)
            outs.append(y)
    diff = (seq.float() - torch.stack(outs, 1).float()).abs()
    assert torch.isfinite(seq).all() and torch.isfinite(diff).all()
    assert float(diff.mean()) < 0.01, f"BF16 mean discrepancy {diff.mean():.5f}"
    assert float(diff.max()) < 0.05, f"BF16 max discrepancy {diff.max():.5f}"
