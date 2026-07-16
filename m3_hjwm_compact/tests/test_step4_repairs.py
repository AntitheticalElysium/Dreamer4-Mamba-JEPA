"""Regression tests for the 2026-07-14 companion repairs (pre-step-4)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))


def test_candidate_specific_masks_can_flip_argmin_common_mask_cannot():
    """Synthetic demonstration of companion finding #1: per-target masks make
    retrieval columns incomparable; a common mask cannot reorder a dominated
    candidate."""
    # token-space distances: prediction is uniformly closer to target 0
    d = np.array([[0.10, 0.20], [0.30, 0.20]])   # d[token, target]
    mask_t0 = np.array([True, False])            # target-0 mask sees token 0
    mask_t1 = np.array([False, True])            # target-1 mask sees token 1
    per_target = [d[mask_t0, 0].mean(), d[mask_t1, 1].mean()]   # [0.10, 0.20]
    # flip case: swap which tokens each mask selects
    flipped = [d[mask_t1, 0].mean(), d[mask_t0, 1].mean()]      # [0.30, 0.20]
    assert int(np.argmin(per_target)) != int(np.argmin(flipped)), \
        "mask choice alone flipped the winner - the flaw is real"
    common = mask_t0 | mask_t1
    a = d[common, 0].mean()
    b = d[common, 1].mean()
    assert (a < b) == (d[:, 0].mean() < d[:, 1].mean()), \
        "common mask must preserve the ordering direction of the full comparison"


@pytest.mark.slow
def test_canonical_fork_is_bit_exact_including_night():
    """Companion finding #2: identity-set chunk iteration breaks repeatability.
    Canonicalized forks must be bit-exact on repeat, including night states."""
    crafter = pytest.importorskip("crafter")
    from crafter_canonical import canonical_snapshot, run_branches_canonical

    def signature(info):
        inv = info.get("inventory", {})
        return {"inventory": tuple(sorted(inv.items()))}

    env = crafter.Env(seed=5, length=10_000)
    rng = np.random.default_rng(5)
    env.reset()
    checked = 0
    steps = 0
    while checked < 4 and steps < 2_000:
        obs, _, done, _ = env.step(int(rng.integers(env.action_space.n)))
        steps += 1
        if done:
            env.reset()
            continue
        # check both day and night states, biased to night (harder case)
        is_night = float(env._world.daylight) < 0.5
        if steps % 40 == 0 and (is_night or checked < 2):
            snapshot = canonical_snapshot(env)
            suffix = [int(rng.integers(env.action_space.n)) for _ in range(8)]
            run_branches_canonical(
                snapshot, suffix, base_seed=90_000 + steps, branches=2,
                suffix_len=8, task_signature=signature, verify_repeat=True)
            checked += 1
    assert checked >= 3, "regression did not reach enough probe states"


def test_global_mamba_contract_and_dropin():
    if not torch.cuda.is_available():
        pytest.skip("official Mamba kernels need CUDA")
    from model import M3HJWM, ModelConfig

    device = torch.device("cuda")
    cfg = ModelConfig(temporal_backend="global_mamba2", predictor="deterministic",
                      mask_ratio=0.0)
    torch.manual_seed(9)
    world = M3HJWM(cfg).to(device)
    core = world.temporal.impl
    x = torch.randn(3, 6, 7, 64, device=device)
    seq_out, _ = core.sequence(x)
    state = core.init_state(3, 7, device, torch.float32)
    outs = []
    for t in range(6):
        y, state = core.step(x[:, t], state)
        outs.append(y)
    step_out = torch.stack(outs, 1)
    assert torch.allclose(seq_out, step_out, atol=2e-2), \
        f"sequence/step divergence {(seq_out - step_out).abs().max():.4f}"
    # reset isolation. Official Mamba kernels mutate caches IN PLACE, so each
    # comparison branch needs its own deep-cloned state (same reason imagine()
    # clones caches before stepping).
    from model import TemporalState

    def clone_state(s):
        return TemporalState(
            [tuple(t.clone() for t in c) for c in s.cache], s.output.clone())

    state_a = core.init_state(2, 7, device, torch.float32)
    _, state_a = core.step(x[:2, 0], state_a)
    _, state_r = core.step(x[:2, 1], clone_state(state_a),
                           reset=torch.tensor([True, False], device=device))
    _, state_n = core.step(x[:2, 1], clone_state(state_a))
    assert torch.allclose(state_r.cache[0][1][1], state_n.cache[0][1][1], atol=1e-5),         "non-reset row must be unaffected"
    assert not torch.allclose(state_r.cache[0][1][0], state_n.cache[0][1][0]),         "reset row must differ from the non-reset run"
    # drop-in world forward
    batch = {
        "obs": torch.randint(0, 255, (2, 4, 3, 64, 64), dtype=torch.uint8, device=device),
        "actions": torch.randint(0, cfg.action_dim, (2, 3), device=device),
        "rewards": torch.randn(2, 3, device=device),
        "continues": torch.ones(2, 3, device=device),
    }
    out = world(batch)
    assert torch.isfinite(out.loss)
