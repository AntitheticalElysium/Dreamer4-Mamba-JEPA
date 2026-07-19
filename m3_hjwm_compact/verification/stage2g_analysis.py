"""Paired Stage-2G mechanism and operational gates."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from fork_oracle_v2 import sha256_file  # noqa: E402
from stage2_evaluation import paired_analysis  # noqa: E402
from stage2g_evaluate import (  # noqa: E402
    ARM_CHECKPOINTS,
    ARTIFACTS,
    EXPECTED_ARM_TRAINING,
    EXPECTED_ENCODER_SHA256,
    EXPECTED_STATIC_SHA256,
    EXPECTED_TRAINING_CONTRACT,
    PROTOCOL,
    RAW_PATH,
    REPORT_PATH,
    TRAIN_REPORT,
)


OUTPUT = ARTIFACTS / "stage2g_analysis.json"
BOOTSTRAP_DRAWS = 2_000
CONTRASTS = {
    "G-LA_vs_C-L": {"G-LA": 1.0, "C-L": -1.0},
    "G-LRA_vs_C-LR": {"G-LRA": 1.0, "C-LR": -1.0},
    "G-LRA_vs_G-LA": {"G-LRA": 1.0, "G-LA": -1.0},
    "G-LA_vs_A": {"G-LA": 1.0, "A": -1.0},
    "G-LRA_vs_A": {"G-LRA": 1.0, "A": -1.0},
    "G-LA_vs_C-LR": {"G-LA": 1.0, "C-LR": -1.0},
}
MECHANISM_CONTRASTS = {
    "G-LA": ("C-L", "G-LA_vs_C-L"),
    "G-LRA": ("C-LR", "G-LRA_vs_C-LR"),
}
OPERATIONAL_CONTRASTS = {
    "G-LA": {
        "factorial_reference": "C-L",
        "factorial_contrast": "G-LA_vs_C-L",
        "versus_a": "G-LA_vs_A",
        "versus_clr": "G-LA_vs_C-LR",
    },
    "G-LRA": {
        "factorial_reference": "C-LR",
        "factorial_contrast": "G-LRA_vs_C-LR",
        "versus_a": "G-LRA_vs_A",
        "versus_clr": "G-LRA_vs_C-LR",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(name: str, passed: bool, **evidence) -> dict:
    return {
        "name": name,
        "pass": bool(passed),
        "evidence": evidence,
    }


def _not_significantly_lower(metric: dict) -> bool:
    return metric["ci95"][1] >= 0.0


def _not_significantly_higher(metric: dict) -> bool:
    return metric["ci95"][0] <= 0.0


def _temporal_safety(
    paired: dict,
    contrast: str,
) -> tuple[bool, dict]:
    evidence = {}
    passed = True
    for depth in (1, 2, 4, 8):
        key = f"k{depth}"
        latent = paired["latent"][key]["contrasts"][contrast][
            "cosine_error"
        ]
        continuation = paired["continuation"][key]["contrasts"][
            contrast
        ]
        depth_pass = (
            _not_significantly_higher(latent)
            and _not_significantly_lower(
                continuation["terminal_auroc"]
            )
            and _not_significantly_lower(
                continuation["brier_skill"]
            )
        )
        evidence[key] = {
            "pass": depth_pass,
            "latent_cosine_error": latent,
            "terminal_auroc": continuation["terminal_auroc"],
            "brier_skill": continuation["brier_skill"],
        }
        passed = passed and depth_pass
    return passed, evidence


def _mechanism_gate(
    paired: dict,
    candidate: str,
    reference: str,
    contrast: str,
) -> dict:
    conditions = []
    k8 = paired["reward"]["k8"]
    candidate_point = k8["points"][candidate]
    reference_point = k8["points"][reference]
    reward = k8["contrasts"][contrast]
    conditions.append(_condition(
        f"{candidate} K8 AUROC/Pearson improve versus {reference}",
        candidate_point["event_auroc"]
        > reference_point["event_auroc"]
        and candidate_point["reward_pearson"]
        > reference_point["reward_pearson"]
        and _not_significantly_lower(reward["event_auroc"])
        and _not_significantly_lower(reward["reward_pearson"]),
        candidate={
            "event_auroc": candidate_point["event_auroc"],
            "reward_pearson": candidate_point["reward_pearson"],
        },
        reference={
            "event_auroc": reference_point["event_auroc"],
            "reward_pearson": reference_point["reward_pearson"],
        },
        contrasts={
            "event_auroc": reward["event_auroc"],
            "reward_pearson": reward["reward_pearson"],
        },
    ))
    conditions.append(_condition(
        f"{candidate} K8 event MAE has no harm versus {reference}",
        _not_significantly_higher(reward["mae_event"]),
        value=reward["mae_event"],
    ))
    zero = paired["zero_suffix"]["contrasts"][contrast][
        "zero_suffix_abs_predicted_sum"
    ]
    conditions.append(_condition(
        f"{candidate} false reward has no harm versus {reference}",
        zero["delta"] <= 0.02 and _not_significantly_higher(zero),
        value=zero,
    ))
    candidate_rank = paired["ranking"]["points"][candidate]
    reference_rank = paired["ranking"]["points"][reference]
    ranking = paired["ranking"]["contrasts"][contrast]
    conditions.append(_condition(
        f"{candidate} ranking improves versus {reference}",
        candidate_rank["chosen_minus_random"]
        > reference_rank["chosen_minus_random"]
        and _not_significantly_lower(
            ranking["chosen_minus_random"]
        )
        and _not_significantly_higher(ranking["regret"]),
        candidate=candidate_rank,
        reference=reference_rank,
        contrasts=ranking,
    ))
    temporal_pass, temporal = _temporal_safety(paired, contrast)
    conditions.append(_condition(
        f"{candidate} latent/continuation safety versus {reference}",
        temporal_pass,
        depths=temporal,
    ))
    return {
        "candidate": candidate,
        "reference": reference,
        "contrast": contrast,
        "pass": all(item["pass"] for item in conditions),
        "conditions": conditions,
    }


def _operational_gate(
    paired: dict,
    candidate: str,
    spec: dict,
) -> dict:
    conditions = []
    versus_a = spec["versus_a"]
    versus_clr = spec["versus_clr"]
    zero_a = paired["zero_suffix"]["contrasts"][versus_a][
        "zero_suffix_abs_predicted_sum"
    ]
    conditions.append(_condition(
        f"{candidate} false reward stays within A + .02",
        zero_a["delta"] <= 0.02 and zero_a["ci95"][1] <= 0.02,
        value=zero_a,
    ))

    ranking_points = paired["ranking"]["points"]
    candidate_rank = ranking_points[candidate][
        "chosen_minus_random"
    ]
    minimum_reference = min(
        ranking_points["A"]["chosen_minus_random"],
        ranking_points["C-LR"]["chosen_minus_random"],
    )
    ranking_a = paired["ranking"]["contrasts"][versus_a]
    ranking_clr = paired["ranking"]["contrasts"][versus_clr]
    conditions.append(_condition(
        f"{candidate} preserves A/C-LR ranking",
        candidate_rank > 0
        and candidate_rank >= minimum_reference
        and _not_significantly_lower(
            ranking_a["chosen_minus_random"]
        )
        and _not_significantly_higher(ranking_a["regret"])
        and _not_significantly_lower(
            ranking_clr["chosen_minus_random"]
        )
        and _not_significantly_higher(ranking_clr["regret"]),
        candidate=candidate_rank,
        minimum_reference=minimum_reference,
        versus_a=ranking_a,
        versus_clr=ranking_clr,
    ))

    k8 = paired["reward"]["k8"]
    candidate_point = k8["points"][candidate]
    clr_point = k8["points"]["C-LR"]
    conditions.append(_condition(
        f"{candidate} K8 reward points preserve C-LR",
        candidate_point["event_auroc"] >= clr_point["event_auroc"]
        and candidate_point["event_average_precision"]
        >= clr_point["event_average_precision"]
        and candidate_point["reward_pearson"]
        >= clr_point["reward_pearson"]
        and candidate_point["mae_event"] <= clr_point["mae_event"],
        candidate={
            name: candidate_point[name]
            for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
        reference={
            name: clr_point[name]
            for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
    ))
    k8_delta = k8["contrasts"][versus_clr]
    conditions.append(_condition(
        f"{candidate} K8 reward contrasts have no C-LR harm",
        _not_significantly_lower(k8_delta["event_auroc"])
        and _not_significantly_lower(k8_delta["reward_pearson"])
        and _not_significantly_higher(k8_delta["mae_event"]),
        event_auroc=k8_delta["event_auroc"],
        reward_pearson=k8_delta["reward_pearson"],
        mae_event=k8_delta["mae_event"],
    ))
    for depth in (0, 1):
        shallow = paired["reward"][f"k{depth}"]["contrasts"][
            versus_a
        ]
        conditions.append(_condition(
            f"{candidate} K{depth} reward safety versus A",
            _not_significantly_lower(shallow["event_auroc"])
            and _not_significantly_lower(shallow["reward_pearson"])
            and shallow["mae_zero"]["delta"] <= 0.005
            and _not_significantly_higher(shallow["mae_zero"]),
            event_auroc=shallow["event_auroc"],
            reward_pearson=shallow["reward_pearson"],
            mae_zero=shallow["mae_zero"],
        ))
    temporal_pass, temporal = _temporal_safety(
        paired, spec["factorial_contrast"]
    )
    conditions.append(_condition(
        f"{candidate} latent/continuation factorial safety",
        temporal_pass,
        reference=spec["factorial_reference"],
        depths=temporal,
    ))
    return {
        "candidate": candidate,
        "pass": all(item["pass"] for item in conditions),
        "conditions": conditions,
    }


def evaluate_gates(
    paired: dict,
    report: dict,
    train_report: dict,
) -> dict:
    validity = []
    validity.append(_condition(
        "A/C-L/C-LR exactly reproduce committed Stage-2C",
        set(report.get("references_exact", {}))
        == {"A", "C-L", "C-LR"}
        and all(report["references_exact"].values()),
        references=report.get("references_exact"),
    ))
    states_exact = all(
        block["state_digest_before"] == block["state_digest_after"]
        for block in report["arms"].values()
    )
    validity.append(_condition(
        "all evaluated worlds remain bit-identical",
        states_exact,
        states={
            arm: {
                "before": block["state_digest_before"],
                "after": block["state_digest_after"],
            }
            for arm, block in report["arms"].items()
        },
    ))
    factorial_exact = (
        set(report["arms"])
        == {"A", "C-L", "C-LR", "G-LA", "G-LRA"}
        and report.get("training_contract_exact") is True
        and set(train_report.get("arms", {}))
        == set(EXPECTED_ARM_TRAINING)
        and all(
            train_report.get(key) == expected
            for key, expected in EXPECTED_TRAINING_CONTRACT.items()
        )
        and train_report["arms"]["G-LA"][
            "generated_reward_weight"
        ] == 0.0
        and train_report["arms"]["G-LRA"][
            "generated_reward_weight"
        ] == 0.10
        and all(
            train_report["arms"][arm]["updates"]
            == EXPECTED_TRAINING_CONTRACT["updates"]
            and train_report["arms"][arm]["world_final_digest"]
            == expected["world_final_digest"]
            and train_report["arms"][arm]["auxiliary_final_digest"]
            == expected["auxiliary_final_digest"]
            and report["arms"][arm].get("checkpoint_contract_exact")
            is True
            and report["arms"][arm]["state_digest_before"]
            == expected["world_final_digest"]
            and report["arms"][arm]["encoder_state_sha256"]
            == EXPECTED_ENCODER_SHA256
            and report["arms"][arm]["checkpoint_sha256"]
            == EXPECTED_STATIC_SHA256[
                ARM_CHECKPOINTS[arm][0]
            ]
            for arm, expected in EXPECTED_ARM_TRAINING.items()
        )
    )
    validity.append(_condition(
        "complete registered factorial and training provenance",
        factorial_exact,
        evaluation_arms=sorted(report["arms"]),
        generated_reward_weights={
            arm: train_report["arms"][arm][
                "generated_reward_weight"
            ]
            for arm in ("G-LA", "G-LRA")
        },
        world_initial_digest=train_report.get("world_initial_digest"),
        auxiliary_initial_digest=train_report.get(
            "auxiliary_initial_digest"
        ),
        schedules={
            key: train_report.get(key)
            for key in (
                "base_schedule_sha256",
                "auxiliary_schedule_sha256",
                "probe_sha256",
            )
        },
        candidate_states={
            arm: report["arms"][arm]["state_digest_before"]
            for arm in ("G-LA", "G-LRA")
        },
    ))
    valid = all(item["pass"] for item in validity)

    mechanisms = {
        candidate: _mechanism_gate(
            paired, candidate, reference, contrast
        )
        for candidate, (reference, contrast)
        in MECHANISM_CONTRASTS.items()
    }
    operational = {
        candidate: _operational_gate(paired, candidate, spec)
        for candidate, spec in OPERATIONAL_CONTRASTS.items()
    }
    mechanism_pass = {
        candidate: valid and block["pass"]
        for candidate, block in mechanisms.items()
    }
    operational_pass = {
        candidate: valid and block["pass"]
        for candidate, block in operational.items()
    }

    if not valid:
        route = "INVALID_STAGE2G; repair and rerun"
    elif any(operational_pass.values()):
        route = (
            "FRESH_REPLICATION_REQUIRED; no planner until matched seeds "
            "and a new tier pass"
        )
    elif any(mechanism_pass.values()):
        route = (
            "RELEVANCE_CAUSAL_BUT_INSUFFICIENT; no planner and no DEV tuning"
        )
    else:
        route = (
            "REJECT_LOCAL_EVENT_SIGN_AUXILIARY; consider separately "
            "registered conditioning or contrastive control"
        )
    return {
        "valid": valid,
        "validity_conditions": validity,
        "mechanism": mechanisms,
        "mechanism_pass": mechanism_pass,
        "operational": operational,
        "operational_pass": operational_pass,
        "route": route,
        "planner_go": False,
    }


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    train_report = json.loads(TRAIN_REPORT.read_text())
    if report["protocol"] != PROTOCOL:
        raise RuntimeError("Stage-2G protocol drift")
    if report["raw_sha256"] != sha256_file(RAW_PATH):
        raise RuntimeError("Stage-2G raw/report mismatch")
    if report["train_report_sha256"] != sha256_file(TRAIN_REPORT):
        raise RuntimeError("Stage-2G train/evaluation chain drift")
    if set(raw["arms"]) != {"A", "C-L", "C-LR", "G-LA", "G-LRA"}:
        raise RuntimeError("Stage-2G evaluation arms incomplete")

    targets = raw["targets"]
    paired = paired_analysis(
        raw["arms"],
        reward_actual=np.asarray(
            targets["reward_actual"], dtype=np.float32
        ),
        reward_clusters=np.asarray(
            targets["reward_episode"], dtype=np.int64
        ),
        continue_actual=np.asarray(
            targets["continue_actual"], dtype=np.float32
        ),
        continue_clusters=np.asarray(
            targets["continue_episode"], dtype=np.int64
        ),
        latent_clusters=np.asarray(
            targets["reward_episode"], dtype=np.int64
        ),
        contrasts=CONTRASTS,
        draws=BOOTSTRAP_DRAWS,
    )
    output = {
        "format": "stage2g_analysis_v1",
        "protocol": PROTOCOL,
        "provenance": {
            "script_sha256": _sha(Path(__file__)),
            "report_sha256": sha256_file(REPORT_PATH),
            "raw_sha256": sha256_file(RAW_PATH),
            "train_report_sha256": sha256_file(TRAIN_REPORT),
        },
        "paired": paired,
    }
    output["gate"] = evaluate_gates(
        paired, report, train_report
    )
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(
        f"{OUTPUT}: valid={output['gate']['valid']} "
        f"mechanism={output['gate']['mechanism_pass']} "
        f"operational={output['gate']['operational_pass']} "
        f"route={output['gate']['route']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
