"""Regression tests for the 2026-07-14 companion repairs (pre-step-4)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))


def test_candidate_specific_masks_can_flip_argmin_common_mask_cannot():
    """Synthetic demonstration of companion finding #1 (fixture corrected
    2026-07-15: the original fixture's column means tied at 0.20, so the
    common-mask assertion compared False == False and proved nothing).

    Here target 0 is strictly POINTWISE closer at every token, so every
    possible common mask must select it; candidate-specific masks can still
    flip the argmin by scoring each target on different tokens."""
    d = np.array([[0.10, 0.20], [0.30, 0.35]])   # d[token, target]
    assert (d[:, 0] < d[:, 1]).all(), "fixture: target 0 pointwise dominant"
    # per-target masks score each target on its own tokens -> incomparable
    per_target = [d[[0], 0].mean(), d[[1], 1].mean()]   # [0.10, 0.35] -> 0
    flipped = [d[[1], 0].mean(), d[[0], 1].mean()]      # [0.30, 0.20] -> 1
    assert int(np.argmin(per_target)) == 0
    assert int(np.argmin(flipped)) == 1, \
        "mask choice alone flipped the winner - the flaw is real"
    # ANY common mask preserves the dominated ordering, strictly
    for common in ([True, False], [False, True], [True, True]):
        m = np.array(common)
        assert d[m, 0].mean() < d[m, 1].mean(), \
            f"common mask {common} failed to preserve the pointwise order"


def _mk_rows(value, separation=0.05, env_seeds=range(16), per_env=2):
    return [{"anchor": e * 100 + i, "env_seed": e, "night": False,
             "retrieval_all": value, "retrieval_changed": value,
             "separation_all": separation}
            for e in env_seeds for i in range(per_env)]


def test_family_gates_use_per_seed_majority_not_pooled():
    """Companion 2026-07-16 critical finding: the registered rule is per-
    training-seed gates + 2/3 majority; pooling all seeds' rows lets one bad
    seed sink the family (2x 30% + 1x 20% pooled to 26.67% failed G-a)."""
    from step4_runner import gate_decisions_per_seed

    rows_by_seed = {101: _mk_rows(0.30), 202: _mk_rows(0.30), 303: _mk_rows(0.20)}
    controls = {s: _mk_rows(0.25, separation=0.0) for s in rows_by_seed}
    out = gate_decisions_per_seed(rows_by_seed, controls)
    assert out["per_seed"]["101"]["G_a"] and out["per_seed"]["202"]["G_a"]
    assert not out["per_seed"]["303"]["G_a"]
    assert out["majority"]["G_a"], "2/3 seeds passing must pass the majority gate"
    assert out["majority"]["all_gates"]
    # env clustering preserved WITHIN each model (16 env clusters, not 48)
    assert out["per_seed"]["101"]["retrieval_all"]["n_seeds"] == 16
    # pooled counter-example: concatenating all rows fails the same threshold
    pooled = np.mean([r["retrieval_all"]
                      for rows in rows_by_seed.values() for r in rows])
    assert pooled < 0.27, "fixture must demonstrate the pooled-rule failure"


def test_shared_state_digest_covers_buffers_and_excludes_temporal():
    """Companion 2026-07-16 finding: the old digest hashed parameters only —
    a mutated shared BUFFER (e.g. positional embedding) went undetected."""
    from step4_runner import shared_state_digest

    class Sub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(3))
            self.register_buffer("pos", torch.zeros(3))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Sub()
            self.temporal = Sub()

    m = Toy()
    base = shared_state_digest(m)
    with torch.no_grad():
        m.encoder.pos += 1.0
    assert shared_state_digest(m) != base, "shared buffer mutation must change digest"
    with torch.no_grad():
        m.encoder.pos -= 1.0
    assert shared_state_digest(m) == base
    with torch.no_grad():
        m.temporal.w += 1.0
        m.temporal.pos += 1.0
    assert shared_state_digest(m) == base, "temporal-core state must be excluded"


def test_replay_digest_covers_every_batch_tensor():
    """Companion 2026-07-16 finding: the old digest hashed only actions/
    rewards/continues — batches differing in observations collided."""
    import hashlib

    from step4_runner import hash_batch

    def digest(batch):
        h = hashlib.sha256()
        hash_batch(h, batch)
        return h.hexdigest()

    torch.manual_seed(0)
    base = {"obs": torch.randint(0, 255, (2, 4, 3, 8, 8), dtype=torch.uint8),
            "actions": torch.randint(0, 17, (2, 3)),
            "previous_actions": torch.randint(0, 17, (2, 3)),
            "rewards": torch.randn(2, 3), "continues": torch.ones(2, 3)}
    same = {k: v.clone() for k, v in base.items()}
    assert digest(base) == digest(same)
    for key in base:
        changed = {k: v.clone() for k, v in base.items()}
        changed[key] = changed[key] + 1
        assert digest(changed) != digest(base), f"{key} not covered by digest"


def _gates(passed: bool):
    return {"majority": {"G_a": passed, "G_b": passed, "G_c": passed,
                         "G_d": passed, "all_gates": passed}}


def test_backend_winner_requires_valid_families():
    """Companion 2026-07-16 launch blocker 1: a backend must never be
    declared the winner when neither family passed its validity gates
    (20% vs 21% retrieval: Mamba consistently 'better', both invalid)."""
    from step4_runner import overall_decision

    win = {"verdict": "M2_mamba2_wins"}
    out = overall_decision({"M1_gru64": _gates(False), "M2_mamba2": _gates(False)}, win)
    assert out["decision"] == "no_valid_family"
    out = overall_decision({"M1_gru64": _gates(True), "M2_mamba2": _gates(False)}, win)
    assert out["decision"] == "single_valid_family:M1_gru64"
    # both valid + parity -> pre-registered GRU-64 engineering tie-break
    out = overall_decision({"M1_gru64": _gates(True), "M2_mamba2": _gates(True)},
                           {"verdict": "parity"})
    assert out["decision"] == "parity_tiebreak:M1_gru64"
    # a valid Mamba win is NOT attributed to the backend before GRU-72
    out = overall_decision({"M1_gru64": _gates(True), "M2_mamba2": _gates(True)}, win)
    assert out["decision"] == "mamba_wins_pending_gru72"
    out = overall_decision({"M1_gru64": _gates(True), "M2_mamba2": _gates(True)},
                           win, gru72_verdict={"verdict": "parity"})
    assert out["decision"] == "mamba_win_explained_by_capacity"
    out = overall_decision({"M1_gru64": _gates(True), "M2_mamba2": _gates(True)},
                           win, gru72_verdict={"verdict": "M2_mamba2_wins"})
    assert out["decision"] == "mamba_wins_backend_attributed"


def test_resume_rejects_tampered_checkpoint():
    """Companion 2026-07-16 launch blocker 2: a checkpoint whose weights were
    altered while all small metadata was preserved must be rejected (the old
    resume compared metadata only, so tampered weights loaded silently)."""
    import hashlib

    from step4_runner import checkpoint_resume_errors

    expected = {
        "arm": {"name": "M1_gru64_s101", "backend": "global_gru",
                "global_hidden": 64, "seed": 101, "shuffled": False, "steps": 30},
        "model_config": {"temporal_backend": "global_gru"},
        "loss_config": {"rollout": 1.0},
        "source_digest": "src", "encoder_sha256": "enc",
        "replay_file_sha256": "rep",
        "versions": {"python": "3.12.12", "torch": "2.12.1", "cuda": "13.0",
                     "numpy": "2.1.0", "mamba_ssm": "2.3.2", "crafter": "1.8.3"},
    }
    good_ckpt = {"state_dict": {"w": torch.zeros(4)},
                 "provenance": {"arm": dict(expected["arm"]),
                                "model_config": dict(expected["model_config"]),
                                "loss_config": dict(expected["loss_config"]),
                                "source_digest": "src", "encoder_sha256": "enc",
                                "replay_file_sha256": "rep",
                                "versions": dict(expected["versions"])}}

    def fake_hash(ckpt):
        return hashlib.sha256(repr(
            [(k, v.numpy().tobytes()) for k, v in ckpt["state_dict"].items()]
        ).encode()).hexdigest()

    recorded = fake_hash(good_ckpt)
    assert checkpoint_resume_errors(good_ckpt, recorded, recorded, expected) == []
    # tamper ONE weight, preserve every piece of metadata
    tampered = {"state_dict": {"w": torch.ones(4)},
                "provenance": good_ckpt["provenance"]}
    errs = checkpoint_resume_errors(tampered, fake_hash(tampered), recorded, expected)
    assert any("hash" in e for e in errs), "weight tamper must be rejected"
    # environment drift must be rejected
    drifted = {"state_dict": good_ckpt["state_dict"],
               "provenance": {**good_ckpt["provenance"],
                              "versions": {**expected["versions"], "torch": "9.9"}}}
    errs = checkpoint_resume_errors(drifted, recorded, recorded, expected)
    assert any("environment drift: torch" in e for e in errs)
    # config tamper must be rejected
    reconfigured = {"state_dict": good_ckpt["state_dict"],
                    "provenance": {**good_ckpt["provenance"],
                                   "model_config": {"temporal_backend": "gru",
                                                    "rollout_steps": 999}}}
    errs = checkpoint_resume_errors(reconfigured, recorded, recorded, expected)
    assert any("ModelConfig" in e for e in errs)
    # missing report hash must be rejected
    errs = checkpoint_resume_errors(good_ckpt, recorded, None, expected)
    assert any("no checkpoint hash" in e for e in errs)


def test_state_digest_hashes_raw_bytes_not_float32():
    """Companion 2026-07-16: casting to float32 before hashing collided on
    int64 values above 2^24."""
    from step4_runner import shared_state_digest

    class Holder(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("v", torch.tensor([value], dtype=torch.int64))

    a = shared_state_digest(Holder(2 ** 24))
    b = shared_state_digest(Holder(2 ** 24 + 1))
    assert a != b, "int64 2^24 vs 2^24+1 must not collide"


def test_anchor_strata_discriminate_pixel_from_task():
    """Strata must actually discriminate: pixel- and task-effectiveness are
    distinct notions and each must be detectable independently."""
    from step4_runner import anchor_strata

    frames = np.zeros((2, 3, 3, 8, 8), dtype=np.uint8)   # [branch, T, C, H, W]
    outcome = [{"reward_sum": 0.0, "terminated": False}] * 2

    def anchor(pixel_diff, task_diff, night=False):
        alt_frames = frames.copy()
        if pixel_diff:
            alt_frames[0, -1, 0, 0, 0] = 255
        alt_out = ([{"reward_sum": 1.0, "terminated": False}] * 2
                   if task_diff else outcome)
        return {"night": night,
                "branches": {"true": {"frames": frames, "outcomes": outcome},
                             "alt0": {"frames": alt_frames, "outcomes": alt_out}}}

    strata = anchor_strata([anchor(True, False), anchor(False, True),
                            anchor(False, False, night=True)])
    assert strata[0] == {"night": False, "pixel_effective": True,
                         "task_effective": False}
    assert strata[1] == {"night": False, "pixel_effective": False,
                         "task_effective": True}
    assert strata[2] == {"night": True, "pixel_effective": False,
                         "task_effective": False}


@pytest.mark.slow
def test_collector_is_end_to_end_repeatable_one_seed():
    """Companion critical finding (2026-07-15): canonicalizing only branch
    snapshots leaves live anchor DISCOVERY nondeterministic (identity-hashed
    chunk sets). After canonicalizing the live env at every reset, two full
    collection runs on one seed must produce identical anchor sequences."""
    pytest.importorskip("crafter")
    import hashlib

    from collect_final_79_94 import collect

    def digest(anchors):
        h = hashlib.sha256()
        for a in anchors:
            h.update(np.asarray(a["player_pos"]).tobytes())
            h.update(a["obs_hist"].tobytes())
            h.update(np.asarray(a["act_hist"]).tobytes())
            h.update(repr(sorted(a["suffixes"].items())).encode())
            h.update(str(a["night"]).encode())
            for name in sorted(a["branches"]):
                h.update(a["branches"][name]["frames"].tobytes())
                h.update(a["branches"][name]["positions"].tobytes())
        return h.hexdigest()

    run1 = collect(seeds=(79,), verify_repeat=False)
    run2 = collect(seeds=(79,), verify_repeat=False)
    assert len(run1) == len(run2) == 12
    assert digest(run1) == digest(run2), \
        "collector is not end-to-end deterministic"


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
