"""Permanent math, split, identity, and gate checks for Stage-2E."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import decode_two_hot, symexp, two_hot  # noqa: E402
from stage2e_analysis import evaluate_gates  # noqa: E402
from stage2e_calibration import (  # noqa: E402
    ARM_ORDER,
    ZERO_BIN,
    CalibrationSpec,
    calibrate_logits,
    decode_calibrated,
    fit_calibrator,
    select_calibrator,
)
from stage2e_evaluate import (  # noqa: E402
    assert_identity,
    dev_contract,
    dev_reward_outputs,
)
import stage2e_fit_calibration as fit_module  # noqa: E402


LOW = -20.0
HIGH = 20.0


def test_zero_reward_is_exact_center_bin_and_targets_sum_to_one():
    rewards = torch.tensor([0.0, -0.7, -0.1, 0.1, 1.0, 1.1])
    targets = two_hot(rewards, 255, LOW, HIGH)
    assert ZERO_BIN == 127
    assert targets[0, ZERO_BIN] == 1.0
    assert torch.count_nonzero(targets[0]) == 1
    torch.testing.assert_close(
        targets.sum(-1), torch.ones(len(rewards))
    )


def test_identity_calibration_is_bit_exact():
    generator = torch.Generator().manual_seed(1821)
    logits = torch.randn(7, 255, generator=generator)
    identity = CalibrationSpec("E-I")
    assert torch.equal(calibrate_logits(logits, identity), logits)
    expected = decode_two_hot(logits, LOW, HIGH)
    observed = decode_calibrated(
        logits, identity, low=LOW, high=HIGH
    )
    assert torch.equal(observed, expected)


def test_dev_decode_preserves_canonical_cuda_operator():
    if not torch.cuda.is_available():
        return
    generator = torch.Generator().manual_seed(2718)
    logits = torch.randn(31, 255, generator=generator).cuda()
    rewards = torch.randn(31, generator=generator)
    expected = decode_two_hot(logits, LOW, HIGH).float().cpu()
    observed, _ = dev_reward_outputs(
        logits.cpu(),
        rewards,
        CalibrationSpec("E-I"),
        torch.device("cuda"),
        low=LOW,
        high=HIGH,
    )
    assert torch.equal(torch.from_numpy(observed), expected)


def test_zero_bias_changes_only_center_and_temperature_is_positive():
    logits = torch.zeros(3, 255)
    spec = CalibrationSpec("E-Z", zero_bias=1.25)
    calibrated = calibrate_logits(logits, spec)
    assert calibrated[:, ZERO_BIN].tolist() == [1.25, 1.25, 1.25]
    without_center = calibrated.clone()
    without_center[:, ZERO_BIN] = 0
    assert torch.count_nonzero(without_center) == 0
    assert CalibrationSpec(
        "E-T", log_temperature=-3.0
    ).temperature > 0


def test_full_batch_zero_bias_fit_is_deterministic_and_improves_nll():
    logits = torch.zeros(100, 255)
    rewards = torch.zeros(100)
    rewards[-10:] = 1.0
    first, first_info = fit_calibrator(
        "E-Z", logits, rewards, low=LOW, high=HIGH
    )
    second, second_info = fit_calibrator(
        "E-Z", logits, rewards, low=LOW, high=HIGH
    )
    assert first == second
    assert first_info == second_info
    assert first.zero_bias > 0
    assert first_info["final_nll"] < first_info["initial_nll"]


def test_selection_uses_cal_nll_and_registered_tie_order():
    fits = {
        arm: {"optimization": {"final_nll": 1.0}}
        for arm in ARM_ORDER
    }
    assert select_calibrator(fits) == "E-I"
    fits["E-TZ"]["optimization"]["final_nll"] = 0.9
    assert select_calibrator(fits) == "E-TZ"


def test_local_and_dreamerv3_decode_operators_diverge_under_uncertainty():
    logits = torch.full((1, 255), -20.0)
    logits[0, ZERO_BIN - 2] = 0.0
    logits[0, ZERO_BIN + 3] = 0.0
    local = decode_two_hot(logits, LOW, HIGH)
    support_symlog = torch.linspace(LOW, HIGH, 255)
    dreamerv3_style = (
        logits.softmax(-1) * symexp(support_symlog)
    ).sum(-1)
    assert not torch.allclose(local, dreamerv3_style)


def test_fit_module_has_no_dev_or_final_artifact_dependency():
    source = Path(fit_module.__file__).read_text().lower()
    for forbidden in (
        "stage2_eval_bundles",
        "stage2c_raw",
        "stage2_dev",
        "manifest[\"dev\"]",
        "manifest['dev']",
        "manifest[\"final\"]",
        "manifest['final']",
    ):
        assert forbidden not in source


def test_evaluator_dev_contract_does_not_index_final():
    class FinalTrap(dict):
        def __getitem__(self, key):
            if key == "final":
                raise AssertionError("FINAL tier accessed")
            return super().__getitem__(key)

    dev = {"natural": 1, "terminal": 2, "bundle": 3}
    manifest = FinalTrap(dev=dev, final={"forbidden": True})
    assert dev_contract(manifest) is dev


def test_identity_assertion_rejects_any_prediction_or_ranking_drift():
    committed = {
        "reward_predictions": {"k0": [0.0], "k8": [0.1]},
        "ranking_rows": [{"env_seed": 1}],
    }
    assert_identity(
        copy.deepcopy(committed["reward_predictions"]),
        copy.deepcopy(committed["ranking_rows"]),
        committed,
    )
    bad = copy.deepcopy(committed["reward_predictions"])
    bad["k8"][0] += 1e-8
    try:
        assert_identity(
            bad, copy.deepcopy(committed["ranking_rows"]), committed
        )
    except RuntimeError as error:
        assert "prediction drift" in str(error)
    else:
        raise AssertionError("identity drift was accepted")


def _metric(delta, low=None, high=None):
    return {
        "delta": delta,
        "ci95": [
            delta - 0.01 if low is None else low,
            delta + 0.01 if high is None else high,
        ],
    }


def passing_gate_fixture():
    selected = "E-Z"
    selected_vs_clr = f"{selected}_vs_clr"
    selected_vs_a = f"{selected}_vs_a"
    points = {
        "A": {
            "event_auroc": 0.70,
            "event_average_precision": 0.20,
            "reward_pearson": 0.20,
            "mae_event": 0.40,
        },
        "C-LR": {
            "event_auroc": 0.75,
            "event_average_precision": 0.25,
            "reward_pearson": 0.25,
            "mae_event": 0.35,
        },
        selected: {
            "event_auroc": 0.76,
            "event_average_precision": 0.26,
            "reward_pearson": 0.26,
            "mae_event": 0.34,
        },
    }
    paired = {
        "zero_suffix": {
            "contrasts": {
                selected_vs_a: {
                    "zero_suffix_abs_predicted_sum": _metric(
                        0.01, 0.005, 0.015
                    ),
                },
            },
        },
        "ranking": {
            "contrasts": {
                selected_vs_clr: {
                    "chosen_minus_random": _metric(
                        0.0, -0.02, 0.02
                    ),
                    "regret": _metric(0.0, -0.02, 0.02),
                },
                selected_vs_a: {
                    "chosen_minus_random": _metric(
                        0.0, -0.02, 0.02
                    ),
                    "regret": _metric(0.0, -0.02, 0.02),
                },
            },
        },
        "reward": {
            "k8": {
                "points": points,
                "contrasts": {
                    selected_vs_clr: {
                        "event_auroc": _metric(
                            0.01, -0.01, 0.03
                        ),
                        "reward_pearson": _metric(
                            0.01, -0.01, 0.03
                        ),
                        "mae_event": _metric(
                            -0.01, -0.03, 0.01
                        ),
                    },
                },
            },
        },
    }
    for depth in (0, 1):
        paired["reward"][f"k{depth}"] = {
            "contrasts": {
                selected_vs_a: {
                    "event_auroc": _metric(0.0, -0.02, 0.02),
                    "reward_pearson": _metric(0.0, -0.02, 0.02),
                    "mae_zero": _metric(-0.001, -0.002, 0.0),
                },
            },
        }
    report = {
        "identity": {
            "reward_predictions_exact": True,
            "ranking_rows_exact": True,
        },
        "frozen_outputs": {
            "continuation_reused_exact": True,
            "latent_reused_exact": True,
        },
        "state_digest_before": "same",
        "state_digest_after": "same",
    }
    fit = {
        "selected_arm": selected,
        "fits": {
            "E-I": {"optimization": {"final_nll": 0.20}},
            selected: {"optimization": {"final_nll": 0.19}},
        },
    }
    return paired, report, fit


def test_gate_accepts_only_cal_improved_safe_candidate():
    paired, report, fit = passing_gate_fixture()
    result = evaluate_gates(paired, report, fit)
    assert result["pass"]
    assert not result["planner_go"]


def test_gate_rejects_dev_false_reward_or_ranking_harm():
    paired, report, fit = passing_gate_fixture()
    paired["zero_suffix"]["contrasts"]["E-Z_vs_a"][
        "zero_suffix_abs_predicted_sum"
    ] = _metric(0.03, 0.02, 0.04)
    result = evaluate_gates(paired, report, fit)
    assert not result["pass"]
    assert result["route"].startswith(
        "REJECT_GLOBAL_CATEGORICAL_CALIBRATION"
    )

    paired, report, fit = passing_gate_fixture()
    paired["ranking"]["contrasts"]["E-Z_vs_clr"][
        "chosen_minus_random"
    ] = _metric(-0.05, -0.08, -0.02)
    result = evaluate_gates(paired, report, fit)
    assert not result["pass"]
