"""Paired analysis and executable diagnostic gates for Stage-2D."""
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
from stage2d_reward_head import (  # noqa: E402
    ARMS,
    ARTIFACTS,
    EXPECTED_NONREWARD_DIGEST,
    PROTOCOL,
    RAW_PATH,
    REPORT_PATH,
)


OUTPUT = ARTIFACTS / "stage2d_analysis.json"
BOOTSTRAP_DRAWS = 2_000
CONTRASTS = {
    "generated_effect": {"D-G": 1.0, "D-R": -1.0},
    "real_extra_fit": {"D-R": 1.0, "C-L": -1.0},
    "generated_vs_latent": {"D-G": 1.0, "C-L": -1.0},
    "real_vs_baseline": {"D-R": 1.0, "A": -1.0},
    "generated_vs_baseline": {"D-G": 1.0, "A": -1.0},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(name: str, passed: bool, **evidence) -> dict:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def _not_significantly_lower(metric: dict) -> bool:
    """Higher-is-better metric has no significant harmful delta."""
    return metric["ci95"][1] >= 0.0


def _not_significantly_higher(metric: dict) -> bool:
    """Lower-is-better metric has no significant harmful delta."""
    return metric["ci95"][0] <= 0.0


def _candidate_gate(
    analysis: dict,
    arm: str,
) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    versus_latent = (
        "real_extra_fit" if arm == "D-R" else "generated_vs_latent"
    )
    versus_baseline = (
        "real_vs_baseline"
        if arm == "D-R" else "generated_vs_baseline"
    )
    conditions = []

    reward_k8 = analysis["reward"]["k8"]
    candidate = reward_k8["points"][arm]
    latent = reward_k8["points"]["C-L"]
    baseline = reward_k8["points"]["A"]
    contrast_latent = reward_k8["contrasts"][versus_latent]
    conditions.append(_condition(
        f"{arm} K8 reward points improve versus C-L",
        candidate["event_auroc"] > latent["event_auroc"]
        and candidate["event_average_precision"]
        > latent["event_average_precision"]
        and candidate["reward_pearson"] > latent["reward_pearson"]
        and candidate["decoded_abs_event_mean"]
        > latent["decoded_abs_event_mean"]
        and candidate["mae_event"] < latent["mae_event"],
        candidate={
            name: candidate[name] for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "decoded_abs_event_mean",
                "mae_event",
            )
        },
        latent_arm={
            name: latent[name] for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "decoded_abs_event_mean",
                "mae_event",
            )
        },
    ))
    conditions.append(_condition(
        f"{arm} K8 AUROC or Pearson CI improves versus C-L",
        contrast_latent["event_auroc"]["ci95"][0] > 0.0
        or contrast_latent["reward_pearson"]["ci95"][0] > 0.0,
        event_auroc=contrast_latent["event_auroc"],
        reward_pearson=contrast_latent["reward_pearson"],
    ))
    conditions.append(_condition(
        f"{arm} K8 average precision reaches A",
        candidate["event_average_precision"]
        >= baseline["event_average_precision"],
        candidate=candidate["event_average_precision"],
        baseline=baseline["event_average_precision"],
    ))

    ranking_latent = analysis["ranking"]["contrasts"][versus_latent]
    conditions.append(_condition(
        f"{arm} ranking restores versus C-L with paired confidence",
        ranking_latent["chosen_minus_random"]["ci95"][0] > 0.0
        and ranking_latent["regret"]["ci95"][1] < 0.0,
        values=ranking_latent,
    ))
    ranking_base = analysis["ranking"]["contrasts"][versus_baseline]
    conditions.append(_condition(
        f"{arm} ranking is not significantly worse than A",
        _not_significantly_lower(
            ranking_base["chosen_minus_random"]
        )
        and _not_significantly_higher(ranking_base["regret"]),
        values=ranking_base,
    ))

    zero = analysis["zero_suffix"]["contrasts"][versus_baseline][
        "zero_suffix_abs_predicted_sum"
    ]
    conditions.append(_condition(
        f"{arm} zero-suffix absolute reward stays within A + .02",
        zero["delta"] <= 0.02 and zero["ci95"][1] <= 0.02,
        value=zero,
    ))

    for depth in (0, 1):
        shallow = analysis["reward"][f"k{depth}"]["contrasts"][
            versus_baseline
        ]
        conditions.append(_condition(
            f"{arm} K{depth} AUROC/Pearson safety versus A",
            _not_significantly_lower(shallow["event_auroc"])
            and _not_significantly_lower(shallow["reward_pearson"]),
            event_auroc=shallow["event_auroc"],
            reward_pearson=shallow["reward_pearson"],
        ))
        conditions.append(_condition(
            f"{arm} K{depth} zero-reward MAE safety versus A",
            shallow["mae_zero"]["delta"] <= 0.005
            and _not_significantly_higher(shallow["mae_zero"]),
            mae_zero=shallow["mae_zero"],
        ))

    return {
        "pass": all(item["pass"] for item in conditions),
        "conditions": conditions,
    }


def evaluate_gates(
    analysis: dict,
    report: dict,
) -> dict:
    isolation_conditions = []
    for arm in ARMS:
        block = report["isolation"][arm]
        raw_identity = block["raw_identity"]
        isolation_conditions.append(_condition(
            f"{arm} non-reward state and frozen outputs are exact",
            block["nonreward_digest_unchanged"]
            and report["arms"][arm]["state_digest_before"]["nonreward"]
            == EXPECTED_NONREWARD_DIGEST
            and report["arms"][arm]["state_digest_after"]["nonreward"]
            == EXPECTED_NONREWARD_DIGEST
            and all(
                all(depths.values())
                for depths in raw_identity.values()
            ),
            block=block,
            before=report["arms"][arm]["state_digest_before"],
            after=report["arms"][arm]["state_digest_after"],
        ))
    isolation = {
        "pass": all(item["pass"] for item in isolation_conditions),
        "conditions": isolation_conditions,
    }

    k8 = analysis["reward"]["k8"]
    generated = k8["points"]["D-G"]
    real = k8["points"]["D-R"]
    effect = k8["contrasts"]["generated_effect"]
    ranking_effect = analysis["ranking"]["contrasts"][
        "generated_effect"
    ]
    zero_effect = analysis["zero_suffix"]["contrasts"][
        "generated_effect"
    ]["zero_suffix_abs_predicted_sum"]
    mechanism_conditions = [
        _condition(
            "D-G K8 AUROC/Pearson/magnitude points improve over D-R",
            generated["event_auroc"] > real["event_auroc"]
            and generated["reward_pearson"] > real["reward_pearson"]
            and generated["decoded_abs_event_mean"]
            > real["decoded_abs_event_mean"],
            generated={
                name: generated[name] for name in (
                    "event_auroc",
                    "reward_pearson",
                    "decoded_abs_event_mean",
                )
            },
            real={
                name: real[name] for name in (
                    "event_auroc",
                    "reward_pearson",
                    "decoded_abs_event_mean",
                )
            },
        ),
        _condition(
            "D-G K8 AUROC or Pearson paired CI improves over D-R",
            effect["event_auroc"]["ci95"][0] > 0.0
            or effect["reward_pearson"]["ci95"][0] > 0.0,
            event_auroc=effect["event_auroc"],
            reward_pearson=effect["reward_pearson"],
        ),
        _condition(
            "D-G ranking is not significantly worse than D-R",
            _not_significantly_lower(
                ranking_effect["chosen_minus_random"]
            )
            and _not_significantly_higher(
                ranking_effect["regret"]
            ),
            values=ranking_effect,
        ),
        _condition(
            "D-G zero-suffix reward is not significantly worse than D-R",
            _not_significantly_higher(zero_effect),
            value=zero_effect,
        ),
    ]
    mechanism = {
        "pass": all(item["pass"] for item in mechanism_conditions),
        "conditions": mechanism_conditions,
    }
    candidates = {
        arm: _candidate_gate(analysis, arm)
        for arm in ARMS
    }

    if not isolation["pass"]:
        route = "INVALID_IMPLEMENTATION; repair and rerun Stage-2D"
    elif not any(block["pass"] for block in candidates.values()):
        route = (
            "STOP_HEAD_ADAPTATION; diagnose reward parameterization and "
            "calibration before more world training"
        )
    elif candidates["D-R"]["pass"] and (
        not candidates["D-G"]["pass"] or not mechanism["pass"]
    ):
        route = (
            "REAL_EXTRA_FIT_SUPPORTED; generated-state exposure unsupported; "
            "fresh evaluation and separate continuation work required"
        )
    elif candidates["D-G"]["pass"] and mechanism["pass"]:
        route = (
            "GENERATED_STATE_SHIFT_SUPPORTED; replicate isolated head on "
            "fresh evaluation before any planner gate"
        )
    else:
        route = (
            "SEPARATED_HEAD_PROMISING_BUT_MECHANISM_UNRESOLVED; fresh "
            "evaluation and separate continuation work required"
        )
    return {
        "I_isolation": isolation,
        "M_generated_state": mechanism,
        "C_candidates": candidates,
        "route": route,
        "planner_go": False,
    }


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if report["protocol"] != PROTOCOL:
        raise RuntimeError("Stage-2D protocol drift")
    if report["hashes"]["raw"] != sha256_file(RAW_PATH):
        raise RuntimeError("Stage-2D raw artifact hash drift")
    expected_arms = {"A", "C-L", "D-R", "D-G"}
    if set(raw["arms"]) != expected_arms:
        raise RuntimeError("Stage-2D raw arms incomplete")
    for arm in ARMS:
        info = report["arms"][arm]
        if sha256_file(Path(info["checkpoint"])) != info[
            "checkpoint_sha256"
        ]:
            raise RuntimeError(f"{arm} head checkpoint hash drift")

    targets = raw["targets"]
    result = paired_analysis(
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
        "protocol": PROTOCOL,
        "provenance": {
            "script_sha256": _sha(Path(__file__)),
            "report_sha256": sha256_file(REPORT_PATH),
            "raw_sha256": sha256_file(RAW_PATH),
            "head_checkpoint_sha256": report["hashes"][
                "head_checkpoints"
            ],
        },
        "paired": result,
    }
    output["gates"] = evaluate_gates(result, report)
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(
        f"{OUTPUT}: route={output['gates']['route']} "
        f"candidates="
        f"{ {arm: block['pass'] for arm, block in output['gates']['C_candidates'].items()} }"
    )


if __name__ == "__main__":
    main()
