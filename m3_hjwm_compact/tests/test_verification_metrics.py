"""Regression tests for Phase B controls that previously produced false passes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "reviews" / "artifacts"
sys.path[:0] = [str(ROOT), str(ROOT / "verification"), str(ARTIFACTS)]

from representation_control import (  # noqa: E402
    changed_patch_mask,
    inventory_probe,
    local_view_labels,
    patch_change_scores,
    target_statistics,
)
from phase_d_backend import changed_window_summary  # noqa: E402


def test_target_statistics_reject_position_only_diversity():
    torch.manual_seed(3)
    codebook = torch.randn(66, 64)
    tokens = codebook[None].expand(128, -1, -1).contiguous()
    stats = target_statistics(tokens, registers=2)
    assert stats["target_flat_effective_rank_covariance"] > 20
    assert stats["target_fixed_stream_variance"] == pytest.approx(0.0, abs=1e-10)
    assert stats["target_observation_variance_fraction"] == pytest.approx(0.0, abs=1e-10)
    assert stats["target_stream_effective_rank_mean"] == pytest.approx(1.0, abs=1e-3)


def test_target_statistics_supports_no_register_tokens():
    torch.manual_seed(4)
    tokens = torch.randn(32, 64, 16)
    stats = target_statistics(tokens, registers=0)
    assert np.isfinite(stats["target_register_pool_effective_rank"])
    assert np.isfinite(stats["target_patch_pool_effective_rank"])


def test_changed_patch_mask_depends_only_on_raw_rgb_change():
    current = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    future = current.clone()
    future[0, :, :4, :4] = 255
    future[1, :, 4:, 4:] = 64
    scores = patch_change_scores(current, future, patch_size=4)
    mask = changed_patch_mask(scores, quantile=0.0)
    assert scores.shape == (2, 4)
    assert bool(mask[0, 0])
    assert not bool(mask[0, 1:].any())
    assert bool(mask[1, 3])


def test_changed_window_summary_is_paired_material_and_fail_closed():
    pred = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    copy = pred + 0.1
    changed = torch.ones_like(pred, dtype=torch.bool)
    summary = changed_window_summary(pred, copy, changed, seed=7)
    assert summary["paired_window_margin_bootstrap_95"] == pytest.approx([0.1, 0.1])
    assert summary["beats_copy_changed"]

    empty = changed_window_summary(pred, copy, torch.zeros_like(changed), seed=7)
    assert empty["valid_windows"] == 0
    assert not empty["beats_copy_changed"]
    assert np.isnan(empty["paired_window_margin_mean"])


def test_local_view_labels_follow_crafter_render_geometry():
    semantic = np.arange(64 * 64, dtype=np.int64).reshape(1, 64, 64)
    player = np.array([[32, 32]], dtype=np.int64)
    labels, valid = local_view_labels(semantic, player, grid=8)
    # Patch center (row=4,col=4) lies in rendered tile (x=0,y=0), i.e.
    # world position player + (-4,-3). Last two token rows are HUD and invalid.
    assert labels[0, 0, 0] == semantic[0, 28, 29]
    assert labels[0, 5, 7] == semantic[0, 36, 35]
    assert bool(valid[0, :6].all())
    assert not bool(valid[0, 6:].any())


def test_inventory_probe_is_invariant_to_encoder_feature_scale():
    torch.manual_seed(5)
    samples, streams, dim = 200, 3, 8
    features = torch.randn(samples, dim)
    tokens = torch.randn(samples, streams, dim) * 0.01
    tokens[:, 0] = features
    labels = torch.stack(
        [features[:, 0], 2 * features[:, 1], features[:, 2] - features[:, 3]], dim=1
    )
    inventory = torch.cat([labels, torch.zeros(samples, 7)], dim=1).numpy().astype(np.float32)
    first = inventory_probe(tokens, inventory, registers=1)
    second = inventory_probe(tokens * 100.0, inventory, registers=1)
    assert first["inventory_r2_mean_varying"] == pytest.approx(
        second["inventory_r2_mean_varying"], abs=1e-8
    )
