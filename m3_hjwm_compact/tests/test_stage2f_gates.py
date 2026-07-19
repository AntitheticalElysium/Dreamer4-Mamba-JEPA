"""Permanent split, provenance, and decision tests for Stage-2F."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from stage2f_analysis import evaluate_gates  # noqa: E402
from stage2f_evaluate import (  # noqa: E402
    EXPECTED_STATIC_SHA256,
    assert_reference_exact,
    dev_contract,
)


def metric(delta: float, low: float = -0.01,
           high: float = 0.01) -> dict:
    return {"delta": delta, "ci95": [low, high]}


def passing_fixture() -> tuple[dict, dict, dict]:
    contrasts = {
        "F-DZ_vs_F-LZ": {
            "chosen_minus_random": metric(0.0),
            "regret": metric(0.0),
        },
        "F-DZ_vs_F-R": {
            "chosen_minus_random": metric(0.0),
            "regret": metric(0.0),
        },
        "F-DZ_vs_A": {
            "chosen_minus_random": metric(0.0),
            "regret": metric(0.0),
        },
    }
    paired = {
        "zero_suffix": {
            "contrasts": {
                "F-DZ_vs_F-LZ": {
                    "zero_suffix_abs_predicted_sum": metric(
                        -0.02, -0.03, -0.01
                    )
                },
                "F-DZ_vs_A": {
                    "zero_suffix_abs_predicted_sum": metric(
                        0.01, 0.0, 0.015
                    )
                },
            }
        },
        "ranking": {"contrasts": contrasts},
        "reward": {},
        "continuation": {},
        "latent": {},
    }
    points = {
        "F-R": {
            "event_auroc": 0.70,
            "event_average_precision": 0.20,
            "reward_pearson": 0.20,
            "mae_event": 0.40,
        },
        "F-DZ": {
            "event_auroc": 0.71,
            "event_average_precision": 0.21,
            "reward_pearson": 0.21,
            "mae_event": 0.39,
        },
    }
    paired["reward"]["k8"] = {
        "points": points,
        "contrasts": {
            "F-DZ_vs_F-LZ": {
                "event_auroc": metric(0.01),
                "reward_pearson": metric(0.01),
                "mae_event": metric(-0.01),
            },
            "F-DZ_vs_F-R": {
                "event_auroc": metric(0.01),
                "reward_pearson": metric(0.01),
                "mae_event": metric(-0.01),
            },
        },
    }
    for depth in (0, 1):
        paired["reward"][f"k{depth}"] = {
            "contrasts": {
                "F-DZ_vs_A": {
                    "event_auroc": metric(0.0),
                    "reward_pearson": metric(0.0),
                    "mae_zero": metric(0.001),
                }
            }
        }
    for depth in (1, 2, 4, 8):
        key = f"k{depth}"
        paired["latent"][key] = {"contrasts": {}}
        paired["continuation"][key] = {"contrasts": {}}
        for contrast in ("F-DZ_vs_F-LZ", "F-DZ_vs_F-R"):
            paired["latent"][key]["contrasts"][contrast] = {
                "cosine_error": metric(0.0)
            }
            paired["continuation"][key]["contrasts"][contrast] = {
                "terminal_auroc": metric(0.0),
                "brier_skill": metric(0.0),
            }
    report = {
        "reference_exact": True,
        "arms": {
            "F-R": {
                "operator": "local_symlog",
                "state_digest_before": "r",
                "state_digest_after": "r",
            },
            "F-LZ": {
                "operator": "local_symlog",
                "state_digest_before": "l",
                "state_digest_after": "l",
            },
            "F-DZ": {
                "operator": "dreamerv3_symexp",
                "state_digest_before": "d",
                "state_digest_after": "d",
            },
        },
    }
    train_report = {
        "arms": {
            "F-LZ": {"initial_digest": "same"},
            "F-DZ": {"initial_digest": "same"},
        }
    }
    return paired, report, train_report


def test_evaluator_dev_contract_never_indexes_final():
    class FinalTrap(dict):
        def __getitem__(self, key):
            if key == "final":
                raise AssertionError("FINAL accessed")
            return super().__getitem__(key)

    dev = {"natural": 1, "terminal": 2, "bundle": 3}
    manifest = FinalTrap(dev=dev, final={"forbidden": True})
    assert dev_contract(manifest) is dev


def test_evaluator_static_hashes_match_sealed_artifacts():
    for path, expected in EXPECTED_STATIC_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_reference_assertion_is_exact_and_fails_closed():
    value = {
        "reward_predictions": {"k0": [0.0]},
        "continuation_predictions": {"k0": [1.0]},
        "latent_errors": {"k1": [0.1]},
        "ranking_rows": [{"env_seed": 1}],
    }
    assert_reference_exact(value.copy(), value)
    changed = {
        **value,
        "reward_predictions": {"k0": [1e-12]},
    }
    try:
        assert_reference_exact(changed, value)
    except RuntimeError as error:
        assert "reward_predictions" in str(error)
    else:
        raise AssertionError("reference drift was accepted")


def test_gate_accepts_only_valid_mechanism_and_operational_candidate():
    paired, report, train = passing_fixture()
    gate = evaluate_gates(paired, report, train)
    assert gate["valid"]
    assert gate["mechanism_pass"]
    assert gate["operational_pass"]
    assert not gate["planner_go"]
    assert gate["route"].startswith("FRESH_REPLICATION_REQUIRED")


def test_gate_rejects_operator_without_false_reward_improvement():
    paired, report, train = passing_fixture()
    paired["zero_suffix"]["contrasts"]["F-DZ_vs_F-LZ"][
        "zero_suffix_abs_predicted_sum"
    ] = metric(0.0, -0.01, 0.01)
    gate = evaluate_gates(paired, report, train)
    assert not gate["mechanism_pass"]
    assert gate["operational_pass"]
    assert gate["route"].startswith("REJECT_REWARD_OPERATOR_SEARCH")


def test_gate_rejects_unsafe_operational_candidate():
    paired, report, train = passing_fixture()
    paired["zero_suffix"]["contrasts"]["F-DZ_vs_A"][
        "zero_suffix_abs_predicted_sum"
    ] = metric(0.03, 0.02, 0.04)
    gate = evaluate_gates(paired, report, train)
    assert gate["mechanism_pass"]
    assert not gate["operational_pass"]
    assert gate["route"].startswith(
        "OPERATOR_CAUSAL_BUT_INSUFFICIENT"
    )
