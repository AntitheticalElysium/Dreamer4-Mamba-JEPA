"""Permanent split, provenance, and decision tests for Stage-2G."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from stage2g_analysis import evaluate_gates  # noqa: E402
from stage2g_evaluate import (  # noqa: E402
    ARM_CHECKPOINTS,
    EXPECTED_ARM_TRAINING,
    EXPECTED_ENCODER_SHA256,
    EXPECTED_STATIC_SHA256,
    EXPECTED_TRAINING_CONTRACT,
    PROTOCOL,
    TRAIN_REPORT,
    assert_checkpoint_contract,
    assert_reference_exact,
    assert_training_contract,
    dev_contract,
)

EXPECTED_OUTCOME_SHA256 = {
    "stage2g_eval_report.json": (
        "8a294c59836d3515ffc6a5d680fa3de7fcc605080966e9f4a5f2a61bb6790f37"
    ),
    "stage2g_eval_raw.json": (
        "ebfc2cbe0e04ee3e579b80d2eda7686e5a4a10eea7f555889d6de866c480f574"
    ),
    "stage2g_analysis.json": (
        "5036e11d2a0b5c30d6e417a10826df085826bf0be757be630110226cf0edac57"
    ),
}


def metric(
    delta: float = 0.0,
    low: float = -0.01,
    high: float = 0.01,
) -> dict:
    return {"delta": delta, "ci95": [low, high]}


def passing_fixture() -> tuple[dict, dict, dict]:
    arms = ("A", "C-L", "C-LR", "G-LA", "G-LRA")
    factorial = ("G-LA_vs_C-L", "G-LRA_vs_C-LR")
    all_contrasts = (
        *factorial,
        "G-LRA_vs_G-LA",
        "G-LA_vs_A",
        "G-LRA_vs_A",
        "G-LA_vs_C-LR",
    )
    ranking_points = {
        "A": {"chosen_minus_random": 0.20, "regret": 0.40},
        "C-L": {"chosen_minus_random": 0.15, "regret": 0.50},
        "C-LR": {"chosen_minus_random": 0.25, "regret": 0.30},
        "G-LA": {"chosen_minus_random": 0.21, "regret": 0.39},
        "G-LRA": {"chosen_minus_random": 0.26, "regret": 0.29},
    }
    paired = {
        "zero_suffix": {
            "contrasts": {
                name: {
                    "zero_suffix_abs_predicted_sum": metric()
                }
                for name in all_contrasts
            }
        },
        "ranking": {
            "points": ranking_points,
            "contrasts": {
                name: {
                    "chosen_minus_random": metric(),
                    "regret": metric(),
                }
                for name in all_contrasts
            },
        },
        "reward": {},
        "continuation": {},
        "latent": {},
    }
    for name in ("G-LA_vs_A", "G-LRA_vs_A"):
        paired["zero_suffix"]["contrasts"][name][
            "zero_suffix_abs_predicted_sum"
        ] = metric(0.01, 0.0, 0.015)

    reward_points = {
        "A": {
            "event_auroc": 0.65,
            "event_average_precision": 0.18,
            "reward_pearson": 0.15,
            "mae_event": 0.45,
        },
        "C-L": {
            "event_auroc": 0.60,
            "event_average_precision": 0.15,
            "reward_pearson": 0.10,
            "mae_event": 0.50,
        },
        "C-LR": {
            "event_auroc": 0.70,
            "event_average_precision": 0.20,
            "reward_pearson": 0.20,
            "mae_event": 0.40,
        },
        "G-LA": {
            "event_auroc": 0.71,
            "event_average_precision": 0.21,
            "reward_pearson": 0.21,
            "mae_event": 0.39,
        },
        "G-LRA": {
            "event_auroc": 0.72,
            "event_average_precision": 0.22,
            "reward_pearson": 0.22,
            "mae_event": 0.38,
        },
    }
    paired["reward"]["k8"] = {
        "points": reward_points,
        "contrasts": {
            name: {
                "event_auroc": metric(0.01),
                "reward_pearson": metric(0.01),
                "mae_event": metric(-0.01),
            }
            for name in all_contrasts
        },
    }
    for depth in (0, 1):
        paired["reward"][f"k{depth}"] = {
            "contrasts": {
                name: {
                    "event_auroc": metric(),
                    "reward_pearson": metric(),
                    "mae_zero": metric(0.001),
                }
                for name in ("G-LA_vs_A", "G-LRA_vs_A")
            }
        }
    for depth in (1, 2, 4, 8):
        key = f"k{depth}"
        paired["latent"][key] = {
            "contrasts": {
                name: {"cosine_error": metric()}
                for name in factorial
            }
        }
        paired["continuation"][key] = {
            "contrasts": {
                name: {
                    "terminal_auroc": metric(),
                    "brier_skill": metric(),
                }
                for name in factorial
            }
        }

    train = {
        **EXPECTED_TRAINING_CONTRACT,
        "arms": {},
    }
    report_arms = {
        arm: {
            "state_digest_before": arm,
            "state_digest_after": arm,
        }
        for arm in arms
    }
    for arm, expected in EXPECTED_ARM_TRAINING.items():
        train["arms"][arm] = {
            **expected,
            "updates": EXPECTED_TRAINING_CONTRACT["updates"],
        }
        report_arms[arm] = {
            "state_digest_before": expected["world_final_digest"],
            "state_digest_after": expected["world_final_digest"],
            "checkpoint_contract_exact": True,
            "encoder_state_sha256": EXPECTED_ENCODER_SHA256,
            "checkpoint_sha256": EXPECTED_STATIC_SHA256[
                ARM_CHECKPOINTS[arm][0]
            ],
        }
    report = {
        "references_exact": {
            "A": True,
            "C-L": True,
            "C-LR": True,
        },
        "training_contract_exact": True,
        "arms": report_arms,
    }
    return paired, report, train


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


def test_committed_outcome_artifacts_are_byte_exact():
    for name, expected in EXPECTED_OUTCOME_SHA256.items():
        path = ARTIFACTS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_committed_outcome_chain_and_rejection_are_exact():
    report = json.loads(
        (ARTIFACTS / "stage2g_eval_report.json").read_text()
    )
    analysis = json.loads(
        (ARTIFACTS / "stage2g_analysis.json").read_text()
    )
    assert report["head"] == (
        "0a0e7904aa5aa436f46e1e0e8e866048f94945d3"
    )
    assert report["raw_sha256"] == EXPECTED_OUTCOME_SHA256[
        "stage2g_eval_raw.json"
    ]
    assert analysis["provenance"]["report_sha256"] == (
        EXPECTED_OUTCOME_SHA256["stage2g_eval_report.json"]
    )
    assert analysis["provenance"]["raw_sha256"] == (
        EXPECTED_OUTCOME_SHA256["stage2g_eval_raw.json"]
    )
    assert analysis["provenance"]["train_report_sha256"] == (
        EXPECTED_STATIC_SHA256[TRAIN_REPORT]
    )
    gate = analysis["gate"]
    assert gate["valid"] is True
    assert gate["mechanism_pass"] == {
        "G-LA": False,
        "G-LRA": False,
    }
    assert gate["operational_pass"] == {
        "G-LA": False,
        "G-LRA": False,
    }
    assert gate["route"] == (
        "REJECT_LOCAL_EVENT_SIGN_AUXILIARY; consider separately "
        "registered conditioning or contrastive control"
    )
    assert gate["planner_go"] is False


def test_training_contract_accepts_only_sealed_values():
    report = json.loads(TRAIN_REPORT.read_text())
    assert_training_contract(report)
    changed = copy.deepcopy(report)
    changed["world_initial_digest"] = "wrong"
    try:
        assert_training_contract(changed)
    except RuntimeError as error:
        assert "world_initial_digest" in str(error)
    else:
        raise AssertionError("incorrect shared initialization was accepted")


def test_checkpoint_contract_accepts_sealed_payload_and_fails_closed():
    report = json.loads(TRAIN_REPORT.read_text())
    path = ARM_CHECKPOINTS["G-LA"][0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_checkpoint_contract("G-LA", payload, report)

    changed = {
        **payload,
        "extra": {
            **payload["extra"],
            "auxiliary_schedule_sha256": "wrong",
        },
    }
    try:
        assert_checkpoint_contract("G-LA", changed, report)
    except RuntimeError as error:
        assert "auxiliary_schedule_sha256" in str(error)
    else:
        raise AssertionError("checkpoint schedule drift was accepted")


def test_reference_assertion_is_exact_and_fails_closed():
    value = {
        "reward_predictions": {"k0": [0.0]},
        "continuation_predictions": {"k0": [1.0]},
        "latent_errors": {"k1": [0.1]},
        "ranking_rows": [{"env_seed": 1}],
    }
    assert_reference_exact("A", value.copy(), value)
    changed = {
        **value,
        "reward_predictions": {"k0": [1e-12]},
    }
    try:
        assert_reference_exact("A", changed, value)
    except RuntimeError as error:
        assert "reward_predictions" in str(error)
    else:
        raise AssertionError("reference drift was accepted")


def test_gate_accepts_only_valid_mechanisms_and_candidates():
    paired, report, train = passing_fixture()
    gate = evaluate_gates(paired, report, train)
    assert gate["valid"]
    assert gate["mechanism_pass"] == {
        "G-LA": True,
        "G-LRA": True,
    }
    assert gate["operational_pass"] == {
        "G-LA": True,
        "G-LRA": True,
    }
    assert not gate["planner_go"]
    assert gate["route"].startswith("FRESH_REPLICATION_REQUIRED")


def test_gate_keeps_factorial_mechanism_verdicts_separate():
    paired, report, train = passing_fixture()
    paired["zero_suffix"]["contrasts"]["G-LA_vs_C-L"][
        "zero_suffix_abs_predicted_sum"
    ] = metric(0.03, 0.02, 0.04)
    gate = evaluate_gates(paired, report, train)
    assert gate["mechanism_pass"] == {
        "G-LA": False,
        "G-LRA": True,
    }


def test_gate_rejects_unsafe_candidate_without_erasing_other_arm():
    paired, report, train = passing_fixture()
    paired["zero_suffix"]["contrasts"]["G-LA_vs_A"][
        "zero_suffix_abs_predicted_sum"
    ] = metric(0.03, 0.02, 0.04)
    gate = evaluate_gates(paired, report, train)
    assert gate["operational_pass"] == {
        "G-LA": False,
        "G-LRA": True,
    }
    assert gate["mechanism_pass"]["G-LA"]


def test_gate_fails_validity_on_shared_initialization_drift():
    paired, report, train = passing_fixture()
    train["world_initial_digest"] = "wrong"
    gate = evaluate_gates(paired, report, train)
    assert not gate["valid"]
    assert not any(gate["mechanism_pass"].values())
    assert not any(gate["operational_pass"].values())
    assert gate["route"].startswith("INVALID_STAGE2G")


def test_protocol_is_the_registered_stage2g_record():
    assert PROTOCOL == (
        "reviews/2026-07-19-stage2g-shared-reward-relevance-protocol.md"
    )
