"""Paired Stage-2F mechanism and operational gates."""
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
from stage2f_evaluate import (  # noqa: E402
    ARTIFACTS,
    PROTOCOL,
    RAW_PATH,
    REPORT_PATH,
    TRAIN_REPORT,
)


OUTPUT = ARTIFACTS / "stage2f_analysis.json"
BOOTSTRAP_DRAWS = 2_000
CONTRASTS = {
    "F-LZ_vs_F-R": {"F-LZ": 1.0, "F-R": -1.0},
    "F-DZ_vs_F-LZ": {"F-DZ": 1.0, "F-LZ": -1.0},
    "F-DZ_vs_F-R": {"F-DZ": 1.0, "F-R": -1.0},
    "F-DZ_vs_A": {"F-DZ": 1.0, "A": -1.0},
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


def evaluate_gates(
    paired: dict,
    report: dict,
    train_report: dict,
) -> dict:
    validity = []
    validity.append(_condition(
        "F-R exactly reconstructs committed C-LR",
        report.get("reference_exact") is True,
        reference_exact=report.get("reference_exact"),
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
    operator_exact = (
        report["arms"]["F-R"]["operator"] == "local_symlog"
        and report["arms"]["F-LZ"]["operator"] == "local_symlog"
        and report["arms"]["F-DZ"]["operator"]
        == "dreamerv3_symexp"
        and train_report["arms"]["F-LZ"]["initial_digest"]
        == train_report["arms"]["F-DZ"]["initial_digest"]
    )
    validity.append(_condition(
        "operator labels and zero-initialized branch state are exact",
        operator_exact,
        operators={
            arm: block["operator"]
            for arm, block in report["arms"].items()
        },
        zero_initial={
            arm: train_report["arms"][arm]["initial_digest"]
            for arm in ("F-LZ", "F-DZ")
        },
    ))
    valid = all(item["pass"] for item in validity)

    mechanism = []
    versus_local_zero = "F-DZ_vs_F-LZ"
    zero = paired["zero_suffix"]["contrasts"][
        versus_local_zero
    ]["zero_suffix_abs_predicted_sum"]
    mechanism.append(_condition(
        "DreamerV3 operator significantly lowers zero-suffix false reward",
        zero["delta"] < 0 and zero["ci95"][1] < 0,
        value=zero,
    ))
    reward = paired["reward"]["k8"]["contrasts"][
        versus_local_zero
    ]
    mechanism.append(_condition(
        "DreamerV3 operator has no significant K8 reward harm",
        _not_significantly_lower(reward["event_auroc"])
        and _not_significantly_lower(reward["reward_pearson"])
        and _not_significantly_higher(reward["mae_event"]),
        event_auroc=reward["event_auroc"],
        reward_pearson=reward["reward_pearson"],
        mae_event=reward["mae_event"],
    ))
    ranking = paired["ranking"]["contrasts"][versus_local_zero]
    mechanism.append(_condition(
        "DreamerV3 operator has no significant ranking harm",
        _not_significantly_lower(
            ranking["chosen_minus_random"]
        )
        and _not_significantly_higher(ranking["regret"]),
        values=ranking,
    ))
    temporal_pass, temporal_evidence = _temporal_safety(
        paired, versus_local_zero
    )
    mechanism.append(_condition(
        "DreamerV3 operator has no latent/continuation harm",
        temporal_pass,
        depths=temporal_evidence,
    ))
    mechanism_pass = valid and all(
        item["pass"] for item in mechanism
    )

    operational = []
    versus_a = "F-DZ_vs_A"
    versus_reference = "F-DZ_vs_F-R"
    zero_a = paired["zero_suffix"]["contrasts"][versus_a][
        "zero_suffix_abs_predicted_sum"
    ]
    operational.append(_condition(
        "F-DZ zero-suffix false reward stays within A + .02",
        zero_a["delta"] <= 0.02 and zero_a["ci95"][1] <= 0.02,
        value=zero_a,
    ))
    for contrast, label in (
        (versus_reference, "F-R"),
        (versus_a, "A"),
    ):
        value = paired["ranking"]["contrasts"][contrast]
        operational.append(_condition(
            f"F-DZ ranking is not significantly worse than {label}",
            _not_significantly_lower(
                value["chosen_minus_random"]
            )
            and _not_significantly_higher(value["regret"]),
            values=value,
        ))

    k8 = paired["reward"]["k8"]
    candidate = k8["points"]["F-DZ"]
    reference = k8["points"]["F-R"]
    operational.append(_condition(
        "F-DZ K8 reward points preserve F-R",
        candidate["event_auroc"] >= reference["event_auroc"]
        and candidate["event_average_precision"]
        >= reference["event_average_precision"]
        and candidate["reward_pearson"]
        >= reference["reward_pearson"]
        and candidate["mae_event"] <= reference["mae_event"],
        candidate={
            name: candidate[name]
            for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
        reference={
            name: reference[name]
            for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
    ))
    k8_delta = k8["contrasts"][versus_reference]
    operational.append(_condition(
        "F-DZ K8 reward contrasts show no significant harm",
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
        operational.append(_condition(
            f"F-DZ K{depth} reward safety versus A",
            _not_significantly_lower(shallow["event_auroc"])
            and _not_significantly_lower(shallow["reward_pearson"])
            and shallow["mae_zero"]["delta"] <= 0.005
            and _not_significantly_higher(shallow["mae_zero"]),
            event_auroc=shallow["event_auroc"],
            reward_pearson=shallow["reward_pearson"],
            mae_zero=shallow["mae_zero"],
        ))
    temporal_pass, temporal_evidence = _temporal_safety(
        paired, versus_reference
    )
    operational.append(_condition(
        "F-DZ latent/continuation safety versus F-R",
        temporal_pass,
        depths=temporal_evidence,
    ))
    operational_pass = valid and all(
        item["pass"] for item in operational
    )

    if not valid:
        route = "INVALID_STAGE2F; repair and rerun"
    elif mechanism_pass and operational_pass:
        route = (
            "FRESH_REPLICATION_REQUIRED; no planner until new tier and "
            "matched seeds pass"
        )
    elif mechanism_pass:
        route = (
            "OPERATOR_CAUSAL_BUT_INSUFFICIENT; no planner and no DEV tuning"
        )
    else:
        route = (
            "REJECT_REWARD_OPERATOR_SEARCH; route to separately registered "
            "reward-relevant representation/action-conditioning control"
        )
    return {
        "valid": valid,
        "validity_conditions": validity,
        "mechanism_pass": mechanism_pass,
        "mechanism_conditions": mechanism,
        "operational_pass": operational_pass,
        "operational_conditions": operational,
        "route": route,
        "planner_go": False,
    }


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    train_report = json.loads(TRAIN_REPORT.read_text())
    if report["protocol"] != PROTOCOL:
        raise RuntimeError("Stage-2F protocol drift")
    if report["raw_sha256"] != sha256_file(RAW_PATH):
        raise RuntimeError("Stage-2F raw/report hash mismatch")
    if report["train_report_sha256"] != sha256_file(TRAIN_REPORT):
        raise RuntimeError("Stage-2F training report drift")
    if set(raw["arms"]) != {"A", "F-R", "F-LZ", "F-DZ"}:
        raise RuntimeError("Stage-2F evaluation arms incomplete")

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
        "format": "stage2f_analysis_v1",
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
