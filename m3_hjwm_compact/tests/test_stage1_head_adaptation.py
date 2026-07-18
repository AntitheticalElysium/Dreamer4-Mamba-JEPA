"""Stage-1 task-head adaptation indexing, sampling, and resume contracts."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))


def synthetic_episode(length: int = 12) -> dict:
    transitions = length - 1
    return {
        "obs": np.arange(length * 3, dtype=np.uint8).reshape(length, 3, 1, 1),
        "actions": np.arange(transitions, dtype=np.int64) % 17,
        "rewards": np.zeros(transitions, dtype=np.float32),
        "continues": np.ones(transitions, dtype=np.float32),
    }


def test_stage1_window_event_and_transition_alignment():
    """H2 must inspect exactly the two generated-target transitions."""
    from stage1_head_adaptation import make_batch, window_index

    episode = synthetic_episode()
    episode["rewards"][7] = 1.0
    uniform, event = window_index([episode])
    assert uniform == [(0, 0), (0, 1), (0, 2)]
    assert event == [(0, 0)]

    batch = make_batch([episode], [(0, 1)], torch.device("cpu"))
    assert batch["obs"].shape == (1, 10, 3, 1, 1)
    assert batch["actions"][0].tolist() == episode["actions"][1:10].tolist()
    assert batch["rewards"][0].tolist() == episode["rewards"][1:10].tolist()
    assert batch["previous_actions"][0, 0].item() == episode["actions"][0]
    assert batch["previous_actions"][0, 1:].tolist() == \
        episode["actions"][1:10].tolist()


def test_equal_update_schedules_match_stage1_sampling(monkeypatch):
    """R1/R2 must consume the exact H1/H2 replay schedules."""
    import stage1b_equal_update_control as control
    from stage1_head_adaptation import window_index

    episode = synthetic_episode()
    episode["rewards"][7] = 1.0
    train = [episode]
    monkeypatch.setattr(control, "UPDATES", 3)
    monkeypatch.setattr(control, "BATCH", 4)

    uniform, event = window_index(train)
    for arm in ("R1", "R2"):
        actual, _ = control.build_schedule(train, arm, seed=505)
        rng = np.random.default_rng(10_505)
        expected = []
        for _ in range(3):
            if arm == "R2":
                expected.extend(
                    [uniform[int(rng.integers(len(uniform)))] for _ in range(2)]
                    + [event[int(rng.integers(len(event)))] for _ in range(2)]
                )
            else:
                expected.extend(
                    uniform[int(rng.integers(len(uniform)))]
                    for _ in range(4)
                )
        assert actual == expected


def test_depth_ceiling_batch_and_target_alignment(monkeypatch):
    import stage1c_head_depth_ceiling as ceiling

    episode = synthetic_episode(length=17)
    episode["rewards"] = np.arange(16, dtype=np.float32)
    episode["continues"] = np.arange(16, dtype=np.float32) + 100
    batch = ceiling.make_batch(
        [episode], [(0, 1)], torch.device("cpu"))
    assert batch["obs"].shape == (1, 8, 3, 1, 1)
    assert batch["actions"][0].tolist() == episode["actions"][1:16].tolist()
    assert batch["rewards"][0].tolist() == episode["rewards"][1:16].tolist()
    assert batch["previous_actions"][0, 0].item() == episode["actions"][0]
    assert batch["previous_actions"][0, 1:].tolist() == \
        episode["actions"][1:8].tolist()

    class FakeWorld:
        def initial_state(self, batch_size, device):
            return SimpleNamespace(
                tokens=torch.zeros(batch_size, 1, 1, device=device))

        def observe_step(self, observation, previous_action, state):
            return SimpleNamespace(tokens=state.tokens + 1)

        def imagine_step(self, state, action, deterministic_mode):
            assert deterministic_mode
            next_state = SimpleNamespace(tokens=state.tokens + 1)
            return next_state, None, None, None

        def pool(self, tokens):
            return tokens[:, 0]

    monkeypatch.setattr(ceiling, "BATCH", 1)
    _, rewards2, continues2 = ceiling.depth_contexts_and_targets(
        FakeWorld(), batch, depth=2)
    _, rewards8, continues8 = ceiling.depth_contexts_and_targets(
        FakeWorld(), batch, depth=8)
    assert rewards2[0].tolist() == episode["rewards"][1:10].tolist()
    assert continues2[0].tolist() == episode["continues"][1:10].tolist()
    assert rewards8[0].tolist() == episode["rewards"][1:16].tolist()
    assert continues8[0].tolist() == episode["continues"][1:16].tolist()


def test_stage1_freeze_contract_leaves_only_shared_task_heads_trainable():
    from model import M3HJWM, ModelConfig, assert_encoder_frozen
    from stage1_head_adaptation import freeze_world_except_heads

    world = M3HJWM(ModelConfig(
        temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    freeze_world_except_heads(world)
    trainable_names = {
        name for name, parameter in world.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith("reward.") or name.startswith("continuation.")
        for name in trainable_names
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in world.parameters()
         if parameter.requires_grad],
        lr=1e-3,
    )
    assert_encoder_frozen(world, optimizer)


def test_resume_restores_explicit_numpy_generator_and_torch_rng():
    from checkpoint import restore_optimizer_and_rng

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    numpy_rng = np.random.default_rng(123)
    _ = numpy_rng.integers(1000)
    numpy_state = copy.deepcopy(numpy_rng.bit_generator.state)
    expected_numpy = numpy_rng.integers(1000, size=8)

    torch.manual_seed(456)
    torch_state = torch.get_rng_state()
    expected_torch = torch.rand(8)
    payload = {
        "optimizer": optimizer.state_dict(),
        "rng": {
            "torch_cpu": torch_state,
            "torch_cuda": None,
            "numpy": numpy_state,
        },
    }

    _ = numpy_rng.integers(1000, size=8)
    _ = torch.rand(8)
    restore_optimizer_and_rng(payload, optimizer, numpy_rng=numpy_rng)
    np.testing.assert_array_equal(
        numpy_rng.integers(1000, size=8), expected_numpy)
    torch.testing.assert_close(torch.rand(8), expected_torch)

    before_failed_restore = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="pass numpy_rng"):
        restore_optimizer_and_rng(payload, optimizer)
    torch.testing.assert_close(
        torch.get_rng_state(), before_failed_restore)
