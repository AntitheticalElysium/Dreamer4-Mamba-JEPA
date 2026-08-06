"""Stage-A integration contracts: production full-grid backends, checkpoint
module, planner-shaped state handling (2026-07-18 sprint)."""
import dataclasses
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))

from model import M3HJWM, ModelConfig  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[2] / "reviews" / "artifacts"


def test_topology_backend_axes_validate():
    ModelConfig(temporal_topology="full_grid", temporal_backend="mamba2",
                dense_bypass=False).validate()
    ModelConfig(temporal_topology="full_grid", temporal_backend="gru",
                dense_bypass=True).validate()
    with pytest.raises(ValueError):
        ModelConfig(temporal_topology="full_grid",
                    temporal_backend="global_gru").validate()
    with pytest.raises(ValueError):
        ModelConfig(temporal_topology="pooled", temporal_backend="gru",
                    dense_bypass=True).validate()


def test_fullgrid_gru_matches_verification_class_and_bypass_flag():
    from exploratory_topology import FlattenedGRUTemporal
    from model import FullGridGRUTemporal

    torch.manual_seed(2)
    prod = FullGridGRUTemporal(dim=8, streams=5, hidden=32, depth=2)
    torch.manual_seed(2)
    verif = FlattenedGRUTemporal(dim=8, streams=5, hidden=32, depth=2)
    assert set(prod.state_dict()) == set(verif.state_dict())
    prod.load_state_dict(verif.state_dict(), strict=True)
    x = torch.randn(2, 4, 5, 8)
    a, _ = prod.sequence(x)
    b, _ = verif.sequence(x)
    torch.testing.assert_close(a, b)
    # bypass flag: zero projection -> passthrough iff bypass
    with torch.no_grad():
        prod.out_proj.weight.zero_()
        prod.out_proj.bias.zero_()
    y, _ = prod.step(x[:, 0], prod.init_state(2, 5, x.device, x.dtype))
    assert torch.count_nonzero(y) == 0
    prod.bypass = True
    y, _ = prod.step(x[:, 0], prod.init_state(2, 5, x.device, x.dtype))
    torch.testing.assert_close(y, x[:, 0])


@pytest.mark.slow
def test_production_world_loads_verification_checkpoint_and_matches():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from checkpoint import sprint_candidate_config
    from exploratory_topology import build_exploratory_world

    device = torch.device("cuda")
    ckpt = torch.load(ARTIFACTS / "xtopo_X-FLM_s505_16000.pt", weights_only=False)
    prod = M3HJWM(sprint_candidate_config("mamba2")).to(device)
    prod.load_state_dict(ckpt["state_dict"], strict=True)   # key-exact port
    verif = build_exploratory_world("X-FLM", 505, device)
    verif.load_state_dict(ckpt["state_dict"], strict=True)
    prod.eval(); verif.eval()
    torch.manual_seed(3)
    batch = {
        "obs": torch.randint(0, 255, (2, 4, 3, 64, 64), dtype=torch.uint8,
                             device=device),
        "actions": torch.randint(0, 17, (2, 3), device=device),
        "rewards": torch.randn(2, 3, device=device),
        "continues": torch.ones(2, 3, device=device),
    }
    with torch.no_grad():
        a = prod(batch)
        b = verif(batch)
    torch.testing.assert_close(a.loss, b.loss)
    for key in a.metrics:
        torch.testing.assert_close(a.metrics[key], b.metrics[key])


def test_checkpoint_roundtrip_and_drift_rejection(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (mamba constructor for full_grid)")
    from checkpoint import (load_world_checkpoint, save_world_checkpoint,
                            sprint_candidate_config)
    from model import LossConfig

    from checkpoint import derived_encoder_digest

    device = torch.device("cuda")
    cfg = sprint_candidate_config("gru")
    torch.manual_seed(4)
    world = M3HJWM(cfg).to(device)
    path = tmp_path / "w.pt"
    digest = save_world_checkpoint(path, world, LossConfig())
    loaded, payload = load_world_checkpoint(path, device, expect_config=cfg,
                                            expect_sha256=digest)
    for (ka, va), (kb, vb) in zip(sorted(world.state_dict().items()),
                                  sorted(loaded.state_dict().items())):
        assert ka == kb
        torch.testing.assert_close(va.cpu(), vb.cpu())
    # provenance is DERIVED from actual encoder state, never caller-supplied
    assert payload["provenance"]["encoder_state_sha256"] == \
        derived_encoder_digest(loaded)
    # config drift must be rejected
    drifted = dataclasses.replace(cfg, flat_gru_hidden=100)
    with pytest.raises(RuntimeError, match="config drift"):
        load_world_checkpoint(path, device, expect_config=drifted)
    # hash tampering must be rejected
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_world_checkpoint(path, device, expect_sha256="0" * 64)


def test_planner_candidate_cache_isolation_and_prefix_reconstruction():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from checkpoint import sprint_candidate_config

    device = torch.device("cuda")
    torch.manual_seed(5)
    world = M3HJWM(sprint_candidate_config("mamba2")).to(device).eval()
    core = world.temporal.impl
    x = torch.randn(1, 6, 66, 64, device=device)
    # real-prefix reconstruction: sequence output == stepwise output
    seq, _ = core.sequence(x)
    state = core.init_state(1, 66, device, torch.float32)
    outs = []
    for t in range(6):
        y, state = core.step(x[:, t], state)
        outs.append(y)
    assert torch.allclose(seq, torch.stack(outs, 1), atol=2e-2)

    def clone_state(s):
        from model import TemporalState
        return TemporalState([tuple(t.clone() for t in c) for c in s.cache],
                             s.output.clone())

    # candidate isolation: stepping clone A must not disturb clone B
    base = clone_state(state)
    a_state = clone_state(base)
    b_state = clone_state(base)
    b_before = [t.clone() for c in b_state.cache for t in c]
    _, a_state = core.step(x[:, 0] + 1.0, a_state)
    b_after = [t for c in b_state.cache for t in c]
    for t0, t1 in zip(b_before, b_after):
        torch.testing.assert_close(t0, t1)
    # different candidate actions from the same base diverge
    ya, _ = core.step(x[:, 0] + 1.0, clone_state(base))
    yb, _ = core.step(x[:, 0] - 1.0, clone_state(base))
    assert not torch.allclose(ya, yb)


def test_reward_continuation_equal_batched_vs_recurrent():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from checkpoint import sprint_candidate_config

    device = torch.device("cuda")
    torch.manual_seed(6)
    world = M3HJWM(sprint_candidate_config("gru")).to(device).eval()
    core = world.temporal.impl
    x = torch.randn(2, 5, 66, 64, device=device)
    seq, _ = core.sequence(x)
    state = core.init_state(2, 66, device, torch.float32)
    outs = []
    for t in range(5):
        y, state = core.step(x[:, t], state)
        outs.append(y)
    stepped = torch.stack(outs, 1)
    torch.testing.assert_close(seq, stepped, atol=1e-5, rtol=1e-5)
    r_seq = world.reward(world.pool(seq))
    r_step = world.reward(world.pool(stepped))
    torch.testing.assert_close(r_seq, r_step, atol=1e-5, rtol=1e-5)
    c_seq = world.continuation(world.pool(seq))
    c_step = world.continuation(world.pool(stepped))
    torch.testing.assert_close(c_seq, c_step, atol=1e-5, rtol=1e-5)


def test_frozen_encoder_contract_is_executable():
    """2026-07-18 companion BLOCKER 2: 'frozen' must be an enforced invariant
    — EMA raises, encoder params leave the trainable set, drift is caught."""
    from model import assert_encoder_frozen, enforce_frozen_encoder

    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic",
                      mask_ratio=0.0)
    torch.manual_seed(9)
    world = enforce_frozen_encoder(M3HJWM(cfg))
    with pytest.raises(RuntimeError, match="frozen-encoder contract"):
        world.update_target()
    assert all(not p.requires_grad for p in world.online_encoder.parameters())
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    assert_encoder_frozen(world, optimizer)   # clean state passes
    first = next(world.online_encoder.parameters())
    saved = first.detach().clone()
    with torch.no_grad():
        first.add_(1.0)
    with pytest.raises(RuntimeError, match="drifted"):
        assert_encoder_frozen(world, optimizer)
    with torch.no_grad():
        first.copy_(saved)   # bit-exact restore (add/sub round-trip is not)
    bad_optimizer = torch.optim.AdamW(world.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="in optimizer"):
        assert_encoder_frozen(world, bad_optimizer)


def test_planner_batch_step_feasible_at_128_candidates():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from checkpoint import sprint_candidate_config

    device = torch.device("cuda")
    torch.manual_seed(7)
    world = M3HJWM(sprint_candidate_config("mamba2")).to(device).eval()
    core = world.temporal.impl
    state = core.init_state(128, 66, device, torch.float32)
    x = torch.randn(128, 66, 64, device=device)
    with torch.no_grad():
        for _ in range(3):
            _, state = core.step(x, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(8):
            y, state = core.step(x, state)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 8 * 1e3
    assert torch.isfinite(y).all()
    assert ms < 100, f"planner-batch step too slow: {ms:.1f} ms"
