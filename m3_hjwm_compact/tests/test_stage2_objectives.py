"""Permanent Stage-2 objective, routing, evaluation, and gate contracts."""
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

from model import M3HJWM, ModelConfig  # noqa: E402
from stage2_evaluation import paired_analysis  # noqa: E402
from stage2_objectives import (  # noqa: E402
    GeneratedLossWeights,
    generated_step_components,
    weighted_generated_loss,
)


class FakeReward:
    def loss(self, logits, targets):
        return (logits - targets).pow(2)


class FakeWorld:
    """Index-visible world with differentiable scalar task predictions."""

    def __init__(self):
        self.observed = []
        self.imagined_actions = []
        self.target_observations = []
        self.reward = FakeReward()

    def initial_state(self, batch_size, device):
        return SimpleNamespace(value=torch.zeros(batch_size, device=device))

    def observe_step(self, observation, previous_action, state):
        self.observed.append((
            observation.flatten(1)[:, 0].tolist(),
            previous_action.tolist(),
        ))
        return SimpleNamespace(value=state.value + 1)

    def imagine_step(self, state, action, deterministic_mode):
        assert deterministic_mode
        self.imagined_actions.append(action.tolist())
        value = action.float()
        selected = torch.stack((value + 1.0, torch.ones_like(value)), -1)
        selected = selected[:, None]
        prediction = SimpleNamespace(selected=selected)
        return (
            SimpleNamespace(value=state.value + 1),
            value,
            value * 0.1,
            prediction,
        )

    def target_encoder(self, observation):
        value = observation.flatten(1)[:, 0].float()
        self.target_observations.append(value.tolist())
        return torch.stack((value + 1.0, torch.ones_like(value)), -1)[:, None]


def indexed_batch(batch_size=2, observations=10):
    obs = torch.arange(observations, dtype=torch.uint8)[
        None, :, None, None, None
    ].repeat(batch_size, 1, 1, 1, 1)
    actions = torch.arange(observations - 1)[None].repeat(batch_size, 1)
    rewards = (
        100 + torch.arange(observations - 1, dtype=torch.float32)
    )[None].repeat(batch_size, 1)
    continues = torch.ones(batch_size, observations - 1)
    previous = torch.full((batch_size, observations), -1, dtype=torch.long)
    previous[:, 1:] = actions
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "continues": continues,
        "previous_actions": previous,
    }


def tiny_config():
    return ModelConfig(
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


def tiny_batch():
    generator = torch.Generator().manual_seed(1807)
    observations = 4
    return {
        "obs": torch.randint(
            0, 256, (2, observations, 3, 32, 32),
            dtype=torch.uint8, generator=generator,
        ),
        "actions": torch.randint(
            0, 17, (2, observations - 1), generator=generator
        ),
        "rewards": torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        ),
        "continues": torch.ones(2, observations - 1),
        "previous_actions": torch.tensor(
            [[-1, 1, 2, 3], [-1, 4, 5, 6]]
        ),
    }


def gradients_are_zero(module):
    return all(
        parameter.grad is None or not bool(parameter.grad.abs().any())
        for parameter in module.parameters()
    )


def gradients_are_nonzero(module):
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in module.parameters()
    )


def test_generated_transition_and_target_alignment():
    world = FakeWorld()
    batch = indexed_batch()
    generated_step_components(world, batch, prefix=8, steps=2)

    assert len(world.observed) == 8
    assert world.observed[0][1] == [-1, -1]
    assert world.observed[7][1] == [6, 6]
    assert world.imagined_actions == [[7, 7], [8, 8]]
    assert world.target_observations == [[8.0, 8.0], [9.0, 9.0]]


def test_post_terminal_mask_excludes_every_k2_component():
    first = indexed_batch()
    first["continues"][:, 7] = 0.0
    second = copy.deepcopy(first)
    second["obs"][:, 9] = 255
    second["actions"][:, 8] = 16
    second["rewards"][:, 8] = -999.0
    second["continues"][:, 8] = 0.0

    components_first = generated_step_components(
        FakeWorld(), first, prefix=8, steps=2
    )
    components_second = generated_step_components(
        FakeWorld(), second, prefix=8, steps=2
    )
    for name in components_first:
        torch.testing.assert_close(
            components_first[name], components_second[name]
        )


def test_component_sum_matches_committed_stage2_combined_loss():
    from stage2_ab import per_step_generated_loss

    batch = indexed_batch()
    old = per_step_generated_loss(FakeWorld(), batch, torch.device("cpu"))
    components = generated_step_components(
        FakeWorld(), batch, prefix=8, steps=2
    )
    new = weighted_generated_loss(
        components,
        GeneratedLossWeights(
            latent=1.0, reward=1.0, continuation=1.0
        ),
    )
    torch.testing.assert_close(old, new)


def test_generated_latent_only_has_no_task_head_gradient():
    world = M3HJWM(tiny_config())
    components = generated_step_components(
        world, tiny_batch(), prefix=2, steps=2
    )
    loss = weighted_generated_loss(
        components,
        GeneratedLossWeights(
            latent=1.0, reward=0.0, continuation=0.0
        ),
    )
    loss.backward()
    assert gradients_are_nonzero(world.future)
    assert gradients_are_nonzero(world.temporal)
    assert gradients_are_zero(world.reward)
    assert gradients_are_zero(world.continuation)


def test_generated_latent_reward_has_no_continuation_gradient():
    world = M3HJWM(tiny_config())
    components = generated_step_components(
        world, tiny_batch(), prefix=2, steps=2
    )
    loss = weighted_generated_loss(
        components,
        GeneratedLossWeights(
            latent=1.0, reward=0.1, continuation=0.0
        ),
    )
    loss.backward()
    assert gradients_are_nonzero(world.future)
    assert gradients_are_nonzero(world.temporal)
    assert gradients_are_nonzero(world.reward)
    assert gradients_are_zero(world.continuation)


def test_stage2c_arm_specs_are_decoupled_and_compute_matched():
    from stage2c_decoupled import (
        ARM_SPECS,
        GENERATED_REWARD_WEIGHT,
    )

    assert set(ARM_SPECS) == {"C-L", "C-LR"}
    assert ARM_SPECS["C-L"].latent == ARM_SPECS["C-LR"].latent == 1.0
    assert ARM_SPECS["C-L"].continuation == 0.0
    assert ARM_SPECS["C-LR"].continuation == 0.0
    assert ARM_SPECS["C-L"].reward == 0.0
    assert ARM_SPECS["C-LR"].reward == GENERATED_REWARD_WEIGHT == 0.10


def synthetic_episode(observations=17):
    transitions = observations - 1
    return {
        "obs": np.zeros((observations, 3, 1, 1), dtype=np.uint8),
        "actions": np.arange(transitions, dtype=np.int64) % 17,
        "rewards": np.zeros(transitions, dtype=np.float32),
        "continues": np.ones(transitions, dtype=np.float32),
    }


def test_stage2c_uniform_distribution_rejects_generated_terminals():
    from stage2c_decoupled import training_distribution

    episode = synthetic_episode()
    episode["continues"][-1] = 0.0
    output = training_distribution([episode], [(0, 0)])
    assert output["terminal_fraction"] == 0.0

    episode["continues"][7] = 0.0
    with pytest.raises(RuntimeError, match="post-terminal"):
        training_distribution([episode], [(0, 0)])


def _ranking_row(env_seed, prediction):
    return {
        "env_seed": env_seed,
        "differs": True,
        "actual": {"zero": 0.0, "event": 1.0},
        "j_sum": {"zero": prediction, "event": 1.0 + prediction},
        "j_gated": {"zero": prediction / 2, "event": 1.0 + prediction},
        "chosen_minus_random": 0.5,
        "regret": 0.0,
    }


def _raw_arm(reward_shift, latent_shift, terminal_shift):
    actual_reward = np.tile([0.0, 0.0, 1.0, -1.0], 4)
    reward = actual_reward * (0.5 + reward_shift)
    actual_continue = np.tile([1.0, 1.0, 1.0, 0.0], 4)
    predicted_continue = np.clip(
        actual_continue * 0.9 + (1 - actual_continue) * 0.1
        - terminal_shift,
        0.001,
        0.999,
    )
    return {
        "reward_predictions": {
            f"k{depth}": reward.tolist()
            for depth in (0, 1, 2, 4, 8)
        },
        "continuation_predictions": {
            f"k{depth}": predicted_continue.tolist()
            for depth in (0, 1, 2, 4, 8)
        },
        "latent_errors": {
            f"k{depth}": (
                np.full(len(actual_reward), 0.1 + latent_shift)
            ).tolist()
            for depth in (1, 2, 4, 8)
        },
        "ranking_rows": [
            _ranking_row(seed, reward_shift)
            for seed in range(4)
        ],
    }


def test_paired_analysis_preserves_arm_pairing_and_directions():
    actual_reward = np.tile([0.0, 0.0, 1.0, -1.0], 4)
    actual_continue = np.tile([1.0, 1.0, 1.0, 0.0], 4)
    clusters = np.repeat(np.arange(4), 4)
    raw = {
        "A": _raw_arm(0.0, 0.0, 0.0),
        "C-L": _raw_arm(0.0, -0.02, 0.0),
        "C-LR": _raw_arm(0.1, -0.02, 0.0),
    }
    result = paired_analysis(
        raw,
        reward_actual=actual_reward,
        reward_clusters=clusters,
        continue_actual=actual_continue,
        continue_clusters=clusters,
        latent_clusters=clusters,
        contrasts={
            "latent_vs_base": {"C-L": 1.0, "A": -1.0},
            "reward_increment": {"C-LR": 1.0, "C-L": -1.0},
            "candidate_vs_base": {"C-LR": 1.0, "A": -1.0},
        },
        draws=20,
    )
    assert result["latent"]["k2"]["contrasts"]["latent_vs_base"][
        "cosine_error"
    ]["delta"] < 0
    assert result["reward"]["k8"]["contrasts"]["reward_increment"][
        "decoded_abs_event_mean"
    ]["delta"] > 0
    assert result["zero_suffix"]["contrasts"]["candidate_vs_base"][
        "zero_suffix_abs_predicted_sum"
    ]["delta"] > 0


def _metric(delta, low=None, high=None):
    low = delta - 0.001 if low is None else low
    high = delta + 0.001 if high is None else high
    return {"delta": delta, "ci95": [low, high]}


def passing_gate_fixture():
    reward = {}
    continuation = {}
    latent = {}
    for depth in (0, 1, 2, 4, 8):
        reward[f"k{depth}"] = {
            "points": {
                "A": {
                    "event_auroc": 0.6,
                    "event_average_precision": 0.2,
                    "reward_pearson": 0.1,
                    "decoded_abs_event_mean": 0.05,
                },
                "C-L": {
                    "event_auroc": 0.61,
                    "event_average_precision": 0.21,
                    "reward_pearson": 0.11,
                    "decoded_abs_event_mean": 0.06,
                },
                "C-LR": {
                    "event_auroc": 0.7,
                    "event_average_precision": 0.3,
                    "reward_pearson": 0.2,
                    "decoded_abs_event_mean": 0.15,
                },
            },
            "contrasts": {
                "candidate_vs_base": {
                    "event_auroc": _metric(0.1, 0.05, 0.15),
                    "reward_pearson": _metric(0.1, 0.05, 0.15),
                },
                "reward_increment": {
                    "event_auroc": _metric(0.09, 0.04, 0.14),
                    "reward_pearson": _metric(0.09, 0.04, 0.14),
                },
            },
        }
        continuation[f"k{depth}"] = {
            "points": {
                "A": {"predicted_termination_nonterminal_mean": 0.005},
                "C-L": {"predicted_termination_nonterminal_mean": 0.006},
                "C-LR": {"predicted_termination_nonterminal_mean": 0.006},
            },
            "contrasts": {
                "latent_vs_base": {
                    "brier_skill": _metric(0.01, -0.01, 0.03)
                },
                "candidate_vs_base": {
                    "brier_skill": _metric(0.01, -0.01, 0.03)
                },
            },
        }
        if depth:
            latent[f"k{depth}"] = {
                "contrasts": {
                    "latent_vs_base": {
                        "cosine_error": _metric(
                            -0.01, -0.02, -0.005
                        )
                    },
                    "reward_increment": {
                        "cosine_error": _metric(
                            0.0, -0.001, 0.001
                        )
                    },
                    "candidate_vs_base": {
                        "cosine_error": _metric(
                            -0.01, -0.02, -0.005
                        )
                    },
                }
            }
    ranking = {
        "contrasts": {
            "latent_vs_base": {
                "chosen_minus_random": _metric(0.0, -0.01, 0.01),
                "regret": _metric(0.0, -0.01, 0.01),
            },
            "candidate_vs_base": {
                "chosen_minus_random": _metric(0.0, -0.01, 0.01),
                "regret": _metric(0.0, -0.01, 0.01),
            },
        }
    }
    zero = {
        "contrasts": {
            "latent_vs_base": {
                "zero_suffix_abs_predicted_sum": _metric(
                    0.0, -0.01, 0.01
                )
            },
            "candidate_vs_base": {
                "zero_suffix_abs_predicted_sum": _metric(
                    0.01, 0.0, 0.019
                )
            },
        }
    }
    return {
        "reward": reward,
        "continuation": continuation,
        "latent": latent,
        "ranking": ranking,
        "zero_suffix": zero,
    }


def test_stage2c_gate_directions_and_false_reward_budget():
    from stage2c_analysis import evaluate_gates

    passing = passing_gate_fixture()
    verdict = evaluate_gates(passing)
    assert verdict["G1_generated_latent"]["pass"]
    assert verdict["G2_generated_reward"]["pass"]
    assert verdict["overall_pass"]

    failing = copy.deepcopy(passing)
    failing["zero_suffix"]["contrasts"]["candidate_vs_base"][
        "zero_suffix_abs_predicted_sum"
    ] = _metric(0.03, 0.02, 0.04)
    verdict = evaluate_gates(failing)
    assert verdict["G1_generated_latent"]["pass"]
    assert not verdict["G2_generated_reward"]["pass"]
    assert not verdict["overall_pass"]


def test_stage2c_runner_never_indexes_final_manifest_tier():
    source = (
        COMPACT_ROOT / "verification" / "stage2c_decoupled.py"
    ).read_text()
    assert 'manifest["final"]' not in source
