"""Permanent isolation, alignment, and routing checks for Stage-2D."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import M3HJWM, ModelConfig  # noqa: E402
from stage2d_analysis import evaluate_gates  # noqa: E402
from stage2d_reward_head import (  # noqa: E402
    ARMS,
    BATCH,
    EXPECTED_BASE_FULL_DIGEST,
    EXPECTED_NONREWARD_DIGEST,
    EXPECTED_REWARD_DIGEST,
    UPDATES,
    build_schedule,
    context_index_contract,
    dev_contract,
    freeze_reward_only,
    load_registered_base,
    reward_contexts,
    selected_state_digest,
)


def indexed_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    observations = 10
    obs = torch.arange(observations, dtype=torch.uint8)[
        None, :, None, None, None
    ].repeat(batch_size, 1, 1, 1, 1)
    actions = torch.arange(observations - 1)[None].repeat(
        batch_size, 1
    )
    rewards = (
        100 + torch.arange(observations - 1, dtype=torch.float32)
    )[None].repeat(batch_size, 1)
    previous = torch.full(
        (batch_size, observations), -1, dtype=torch.long
    )
    previous[:, 1:] = actions
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "continues": torch.ones(batch_size, observations - 1),
        "previous_actions": previous,
    }


class IndexWorld:
    """State values expose whether contexts were observed or imagined."""

    def initial_state(self, batch: int, device):
        return SimpleNamespace(
            tokens=torch.zeros(batch, 1, 1, device=device)
        )

    def observe_step(self, observation, previous_action, state):
        del previous_action, state
        value = observation.flatten(1)[:, 0].float()
        return SimpleNamespace(tokens=value[:, None, None])

    def imagine_step(self, state, action, deterministic_mode):
        assert deterministic_mode
        value = state.tokens[:, 0, 0] + 100.0 + action.float()
        next_state = SimpleNamespace(tokens=value[:, None, None])
        return next_state, None, None, None

    @staticmethod
    def pool(tokens):
        return tokens[:, 0]


def tiny_world_and_batch():
    cfg = ModelConfig(
        image_size=32,
        patch_size=16,
        token_dim=16,
        registers=1,
        spatial_heads=2,
        spatial_depth=1,
        temporal_backend="gru",
        temporal_depth=1,
        predictor="deterministic",
        predictor_depth=1,
        mask_ratio=0.0,
        rollout_steps=2,
    )
    generator = torch.Generator().manual_seed(1818)
    obs = torch.randint(
        0,
        256,
        (2, 10, 3, 32, 32),
        dtype=torch.uint8,
        generator=generator,
    )
    actions = torch.randint(
        0, cfg.action_dim, (2, 9), generator=generator
    )
    previous = torch.full((2, 10), -1, dtype=torch.long)
    previous[:, 1:] = actions
    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": torch.tensor([
            [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        "continues": torch.ones(2, 9),
        "previous_actions": previous,
    }
    return M3HJWM(cfg), batch


def test_context_contract_uses_same_nine_transition_labels():
    real = context_index_contract("D-R")
    generated = context_index_contract("D-G")
    assert [item["reward_index"] for item in real] == list(range(9))
    assert [item["reward_index"] for item in generated] == list(range(9))
    assert real[:7] == generated[:7]
    assert [item["observation_index"] for item in real] == list(
        range(1, 10)
    )
    assert [item["action_index"] for item in generated[-2:]] == [7, 8]
    assert [item["depth"] for item in generated[-2:]] == [1, 2]


def test_reward_contexts_share_real_prefix_and_only_replace_final_two():
    batch = indexed_batch()
    world = IndexWorld()
    real = reward_contexts(world, batch, "D-R")
    generated = reward_contexts(world, batch, "D-G")
    torch.testing.assert_close(real[:, :7], generated[:, :7])
    assert not torch.equal(real[:, -2:], generated[:, -2:])
    torch.testing.assert_close(
        real[0, :, 0], torch.arange(1, 10, dtype=torch.float32)
    )
    assert batch["rewards"][0].tolist() == [
        float(value) for value in range(100, 109)
    ]


def test_reward_only_backward_and_step_preserve_all_nonreward_state():
    world, batch = tiny_world_and_batch()
    names = freeze_reward_only(world)
    assert len(names) == 6
    assert all(name.startswith("reward.") for name in names)
    before_reward = selected_state_digest(world, reward=True)
    before_nonreward = selected_state_digest(world, reward=False)

    with torch.no_grad():
        real_contexts = reward_contexts(world, batch, "D-R")
        contexts = reward_contexts(world, batch, "D-G")
    assert torch.equal(real_contexts[:, :7], contexts[:, :7])
    loss = world.reward.loss(
        world.reward(contexts), batch["rewards"]
    ).mean()
    loss.backward()
    for name, parameter in world.named_parameters():
        if name.startswith("reward."):
            assert parameter.grad is not None
        else:
            assert parameter.grad is None

    optimizer = torch.optim.AdamW(
        [parameter for parameter in world.parameters()
         if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.step()
    world.mark_parameters_updated()
    assert selected_state_digest(world, reward=True) != before_reward
    assert selected_state_digest(world, reward=False) == before_nonreward


def test_schedule_is_uniform_deterministic_and_matched():
    episode = {
        "obs": np.zeros((12, 3, 1, 1), dtype=np.uint8),
        "actions": np.zeros(11, dtype=np.int64),
        "rewards": np.zeros(11, dtype=np.float32),
        "continues": np.ones(11, dtype=np.float32),
    }
    first, first_digest = build_schedule([episode, copy.deepcopy(episode)])
    second, second_digest = build_schedule(
        [episode, copy.deepcopy(episode)]
    )
    assert len(first) == UPDATES * BATCH
    assert first == second
    assert first_digest == second_digest
    assert set(first).issubset({
        (episode_index, start)
        for episode_index in range(2)
        for start in range(3)
    })


def test_dev_contract_never_indexes_final():
    class FinalTrap(dict):
        def __getitem__(self, key):
            if key == "final":
                raise AssertionError("FINAL tier was accessed")
            return super().__getitem__(key)

    dev = {"natural": 1, "terminal": 2, "bundle": 3}
    manifest = FinalTrap(dev=dev, final={"forbidden": True})
    assert dev_contract(manifest) is dev


def test_registered_cl_checkpoint_loads_with_exact_state_digests():
    world, _ = load_registered_base(torch.device("cpu"))
    assert selected_state_digest(
        world, reward=None
    ) == EXPECTED_BASE_FULL_DIGEST
    assert selected_state_digest(
        world, reward=True
    ) == EXPECTED_REWARD_DIGEST
    assert selected_state_digest(
        world, reward=False
    ) == EXPECTED_NONREWARD_DIGEST


def _metric(delta: float, low: float | None = None,
            high: float | None = None):
    return {
        "delta": delta,
        "ci95": [
            delta - 0.01 if low is None else low,
            delta + 0.01 if high is None else high,
        ],
    }


def passing_analysis():
    points = {
        "A": {
            "event_auroc": 0.60,
            "event_average_precision": 0.20,
            "reward_pearson": 0.10,
            "decoded_abs_event_mean": 0.05,
            "mae_event": 0.50,
        },
        "C-L": {
            "event_auroc": 0.55,
            "event_average_precision": 0.15,
            "reward_pearson": 0.02,
            "decoded_abs_event_mean": 0.01,
            "mae_event": 0.55,
        },
        "D-R": {
            "event_auroc": 0.70,
            "event_average_precision": 0.22,
            "reward_pearson": 0.20,
            "decoded_abs_event_mean": 0.10,
            "mae_event": 0.40,
        },
        "D-G": {
            "event_auroc": 0.80,
            "event_average_precision": 0.25,
            "reward_pearson": 0.30,
            "decoded_abs_event_mean": 0.15,
            "mae_event": 0.35,
        },
    }
    contrasts = {
        name: {
            "event_auroc": _metric(0.10, 0.05, 0.15),
            "reward_pearson": _metric(0.10, 0.05, 0.15),
        }
        for name in (
            "generated_effect",
            "real_extra_fit",
            "generated_vs_latent",
            "real_vs_baseline",
            "generated_vs_baseline",
        )
    }
    analysis = {
        "reward": {
            "k8": {"points": points, "contrasts": contrasts},
        },
        "ranking": {
            "contrasts": {
                "generated_effect": {
                    "chosen_minus_random": _metric(0.01, -0.02, 0.04),
                    "regret": _metric(-0.01, -0.04, 0.02),
                },
                "real_extra_fit": {
                    "chosen_minus_random": _metric(0.20, 0.10, 0.30),
                    "regret": _metric(-0.20, -0.30, -0.10),
                },
                "generated_vs_latent": {
                    "chosen_minus_random": _metric(0.21, 0.11, 0.31),
                    "regret": _metric(-0.21, -0.31, -0.11),
                },
                "real_vs_baseline": {
                    "chosen_minus_random": _metric(0.00, -0.05, 0.05),
                    "regret": _metric(0.00, -0.05, 0.05),
                },
                "generated_vs_baseline": {
                    "chosen_minus_random": _metric(0.01, -0.04, 0.06),
                    "regret": _metric(-0.01, -0.06, 0.04),
                },
            },
        },
        "zero_suffix": {
            "contrasts": {
                "generated_effect": {
                    "zero_suffix_abs_predicted_sum": _metric(
                        0.0, -0.01, 0.01
                    ),
                },
                "real_vs_baseline": {
                    "zero_suffix_abs_predicted_sum": _metric(
                        0.01, 0.005, 0.015
                    ),
                },
                "generated_vs_baseline": {
                    "zero_suffix_abs_predicted_sum": _metric(
                        0.01, 0.005, 0.015
                    ),
                },
            },
        },
    }
    for depth in (0, 1):
        analysis["reward"][f"k{depth}"] = {
            "contrasts": {
                name: {
                    "event_auroc": _metric(0.0, -0.02, 0.02),
                    "reward_pearson": _metric(0.0, -0.02, 0.02),
                    "mae_zero": _metric(-0.001, -0.002, 0.0),
                }
                for name in (
                    "real_vs_baseline",
                    "generated_vs_baseline",
                )
            },
        }
    return analysis


def passing_report():
    exact = {
        "continuation_predictions": {
            f"k{depth}": True for depth in (0, 1, 2, 4, 8)
        },
        "latent_errors": {
            f"k{depth}": True for depth in (1, 2, 4, 8)
        },
    }
    return {
        "isolation": {
            arm: {
                "nonreward_digest_unchanged": True,
                "raw_identity": copy.deepcopy(exact),
            }
            for arm in ARMS
        },
        "arms": {
            arm: {
                "state_digest_before": {
                    "nonreward": EXPECTED_NONREWARD_DIGEST
                },
                "state_digest_after": {
                    "nonreward": EXPECTED_NONREWARD_DIGEST
                },
            }
            for arm in ARMS
        },
    }


def test_stage2d_gates_accept_only_isolated_calibrated_candidates():
    result = evaluate_gates(passing_analysis(), passing_report())
    assert result["I_isolation"]["pass"]
    assert result["M_generated_state"]["pass"]
    assert result["C_candidates"]["D-R"]["pass"]
    assert result["C_candidates"]["D-G"]["pass"]
    assert not result["planner_go"]


def test_stage2d_gate_rejects_false_reward_even_with_better_auc():
    analysis = passing_analysis()
    bad = _metric(0.03, 0.025, 0.04)
    analysis["zero_suffix"]["contrasts"][
        "generated_vs_baseline"
    ]["zero_suffix_abs_predicted_sum"] = bad
    result = evaluate_gates(analysis, passing_report())
    assert result["M_generated_state"]["pass"]
    assert result["C_candidates"]["D-R"]["pass"]
    assert not result["C_candidates"]["D-G"]["pass"]
    assert result["route"].startswith("REAL_EXTRA_FIT_SUPPORTED")


def test_stage2d_gate_invalidates_any_frozen_output_drift():
    report = passing_report()
    report["isolation"]["D-G"]["raw_identity"][
        "latent_errors"
    ]["k8"] = False
    result = evaluate_gates(passing_analysis(), report)
    assert not result["I_isolation"]["pass"]
    assert result["route"].startswith("INVALID_IMPLEMENTATION")
