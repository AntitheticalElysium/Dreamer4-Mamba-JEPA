"""Tests for the representation oracle (probe machinery; no craftax/JAX)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from d4_mamba_jepa.craftax_oracle import (
    ProbeData,
    PROBE_ONLY_SUFFIX,
    _cnn_predict,
    _episode_three_way,
    _mlp_predict,
    _within_episode_index,
    audit_probe_machinery,
    load_probe_data,
    preserved_modes,
    save_probe_data,
    timestep_features,
)
from d4_mamba_jepa.data import load_episode_replay
from d4_mamba_jepa.oracle_metrics import mean_r2


def test_audit_probe_machinery_passes():
    result = audit_probe_machinery(seed=0)
    assert result["pass"] is True
    assert result["perfect_r2"] > 0.95
    assert abs(result["constant_r2"]) < 0.05
    assert result["misaligned_r2"] < 0.1
    assert result["timestep_shift_r2"] < 0.5   # off-by-one is detected


def test_within_episode_index_and_timestep_features():
    ep = np.array([0, 0, 0, 1, 1])
    assert _within_episode_index(ep).tolist() == [0, 1, 2, 0, 1]
    feats = timestep_features(ep)
    assert feats.shape == (5, 2)
    assert (feats[:, 1] == feats[:, 0] ** 2).all()  # second column is the square


def test_episode_three_way_disjoint():
    ep = np.repeat(np.arange(12), 10)
    train, val, test = _episode_three_way(ep, seed=1)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert (train | val | test).all()
    assert set(ep[train]).isdisjoint(set(ep[test]))
    assert set(ep[val]).isdisjoint(set(ep[test]))


def test_preserved_modes_restores_heterogeneous_mode_map():
    """Mode-map safety: a diagnostic must restore the EXACT prior mode map."""
    module = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    module[0].train()
    module[1].eval()   # deliberately heterogeneous
    module[2].train()
    before = {n: m.training for n, m in module.named_modules()}
    with preserved_modes(module):
        module.eval()  # flips everything to eval inside the block
        assert not module[0].training
    after = {n: m.training for n, m in module.named_modules()}
    assert after == before  # heterogeneous map restored exactly


def _probe(n=12):
    return ProbeData(
        frames=np.zeros((n, 3, 64, 64), dtype=np.uint8),
        vitals=np.zeros((n, 4), dtype=np.float32),
        inventory=np.zeros((n, 12), dtype=np.float32),
        achievements=np.zeros((n, 22), dtype=bool),
        episode_id=np.arange(n, dtype=np.int64),
    )


def test_probe_only_marker_and_replay_loader_rejects(tmp_path):
    with pytest.raises(ValueError):
        save_probe_data(tmp_path / "bad.pt", _probe())        # suffix enforced
    path = save_probe_data(tmp_path / ("rep" + PROBE_ONLY_SUFFIX), _probe())
    assert load_probe_data(path).frames.shape == (12, 3, 64, 64)
    # Bidirectional isolation: the TRAINING replay loader must reject it even if
    # a hash is supplied (it is rejected before the hash is checked here because
    # the marker is inspected after load; supply the real hash to reach it).
    import hashlib
    real_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="probe-only"):
        load_episode_replay(path, expected_sha256=real_hash)


def test_mlp_probe_recovers_nonlinear_signal():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 8)).astype(np.float32)
    y = (np.sin(x[:, :1]) + (x[:, 1:2] ** 2)).astype(np.float32)  # nonlinear
    tr, va, te = slice(0, 240), slice(240, 320), slice(320, n)
    # device="cpu" keeps the probe deterministic in-suite (cuDNN conv/matmul on
    # CUDA is nondeterministic and makes threshold assertions flaky).
    pred = _mlp_predict(x[tr], y[tr], x[va], y[va], x[te], seed=0, max_steps=300, device="cpu")
    assert mean_r2(y[te], pred) > 0.7
    # noise features -> no recovery
    noise = rng.normal(size=(n, 8)).astype(np.float32)
    pred_n = _mlp_predict(noise[tr], y[tr], noise[va], y[va], noise[te], seed=0, max_steps=300, device="cpu")
    assert mean_r2(y[te], pred_n) < 0.3


def test_cnn_probe_recovers_signal_from_frames():
    rng = np.random.default_rng(1)
    n = 300
    frames = rng.integers(0, 256, size=(n, 3, 32, 32)).astype(np.uint8)
    # Encode a per-sample scalar z into a top-left red patch; the CNN must read
    # the patch back out. (A recoverable signal, unlike random-pixel statistics.)
    z = rng.normal(size=(n, 1)).astype(np.float32)
    patch = np.clip(z * 60 + 128, 0, 255).astype(np.uint8)
    frames[:, 0, :16, :16] = patch[:, :, None]  # strong, unambiguous signal
    y = z
    tr, va, te = slice(0, 180), slice(180, 240), slice(240, n)
    pred = _cnn_predict(frames[tr], y[tr], frames[va], y[va], frames[te], seed=0, max_steps=500, device="cpu")
    assert mean_r2(y[te], pred) > 0.5
