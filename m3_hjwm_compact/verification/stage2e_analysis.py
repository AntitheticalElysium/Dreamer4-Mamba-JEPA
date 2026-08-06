"""Paired Stage-2E analysis; gate only the CAL-selected arm."""
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
from stage2e_calibration import ARM_ORDER  # noqa: E402
from stage2e_evaluate import (  # noqa: E402
    ARTIFACTS,
    FIT_PATH,
    PROTOCOL,
    RAW_PATH,
    REPORT_PATH,
)


OUTPUT = ARTIFACTS / "stage2e_analysis.json"
BOOTSTRAP_DRAWS = 2_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(name: str, passed: bool, **evidence) -> dict:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def _not_significantly_lower(metric: dict) -> bool:
    return metric["ci95"][1] >= 0.0


def _not_significantly_higher(metric: dict) -> bool:
    return metric["ci95"][0] <= 0.0


def build_contrasts() -> dict[str, dict[str, float]]:
    output = {}
    for arm in ARM_ORDER:
        output[f"{arm}_vs_clr"] = {arm: 1.0, "C-LR": -1.0}
        output[f"{arm}_vs_a"] = {arm: 1.0, "A": -1.0}
    return output


def evaluate_gates(
    paired: dict,
    report: dict,
    fit: dict,
) -> dict:
    selected = fit["selected_arm"]
    versus_clr = f"{selected}_vs_clr"
    versus_a = f"{selected}_vs_a"
    conditions = []

    identity = report["identity"]
    isolation_pass = (
        identity.get("reward_predictions_exact") is True
        and identity.get("ranking_rows_exact") is True
        and report["frozen_outputs"].get(
            "continuation_reused_exact"
        ) is True
        and report["frozen_outputs"].get(
            "latent_reused_exact"
        ) is True
        and report["state_digest_before"]
        == report["state_digest_after"]
    )
    conditions.append(_condition(
        "identity and frozen-world invariants are exact",
        isolation_pass,
        identity=identity,
        state_before=report["state_digest_before"],
        state_after=report["state_digest_after"],
    ))

    identity_nll = fit["fits"]["E-I"]["optimization"]["final_nll"]
    selected_nll = fit["fits"][selected]["optimization"]["final_nll"]
    conditions.append(_condition(
        "CAL-selected NLL improves identity by at least 1e-6",
        selected_nll <= identity_nll - 1e-6,
        selected=selected,
        identity_nll=identity_nll,
        selected_nll=selected_nll,
        improvement=identity_nll - selected_nll,
    ))

    zero = paired["zero_suffix"]["contrasts"][versus_a][
        "zero_suffix_abs_predicted_sum"
    ]
    conditions.append(_condition(
        "selected zero-suffix absolute reward stays within A + .02",
        zero["delta"] <= 0.02 and zero["ci95"][1] <= 0.02,
        value=zero,
    ))

    for contrast_name, label in (
        (versus_clr, "C-LR"),
        (versus_a, "A"),
    ):
        ranking = paired["ranking"]["contrasts"][contrast_name]
        conditions.append(_condition(
            f"selected ranking not significantly worse than {label}",
            _not_significantly_lower(
                ranking["chosen_minus_random"]
            )
            and _not_significantly_higher(ranking["regret"]),
            values=ranking,
        ))

    reward_k8 = paired["reward"]["k8"]
    selected_points = reward_k8["points"][selected]
    clr_points = reward_k8["points"]["C-LR"]
    conditions.append(_condition(
        "selected K8 AUROC/AP/Pearson/event-MAE points preserve C-LR",
        selected_points["event_auroc"] >= clr_points["event_auroc"]
        and selected_points["event_average_precision"]
        >= clr_points["event_average_precision"]
        and selected_points["reward_pearson"]
        >= clr_points["reward_pearson"]
        and selected_points["mae_event"] <= clr_points["mae_event"],
        selected={
            name: selected_points[name] for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
        baseline={
            name: clr_points[name] for name in (
                "event_auroc",
                "event_average_precision",
                "reward_pearson",
                "mae_event",
            )
        },
    ))
    k8_delta = reward_k8["contrasts"][versus_clr]
    conditions.append(_condition(
        "selected K8 paired metrics show no significant harm versus C-LR",
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
            f"selected K{depth} AUROC/Pearson safety versus A",
            _not_significantly_lower(shallow["event_auroc"])
            and _not_significantly_lower(shallow["reward_pearson"]),
            event_auroc=shallow["event_auroc"],
            reward_pearson=shallow["reward_pearson"],
        ))
        conditions.append(_condition(
            f"selected K{depth} zero-MAE safety versus A",
            shallow["mae_zero"]["delta"] <= 0.005
            and _not_significantly_higher(shallow["mae_zero"]),
            mae_zero=shallow["mae_zero"],
        ))

    passed = all(item["pass"] for item in conditions)
    if not isolation_pass:
        route = "INVALID_STAGE2E; repair and rerun"
    elif passed:
        route = (
            "FRESH_REPLICATION_REQUIRED; no planner until new tier and "
            "matched world seeds pass"
        )
    else:
        route = (
            "REJECT_GLOBAL_CATEGORICAL_CALIBRATION; do not sweep DEV "
            "thresholds"
        )
    return {
        "selected_arm": selected,
        "pass": passed,
        "conditions": conditions,
        "route": route,
        "planner_go": False,
    }


def main() -> None:
    fit = json.loads(FIT_PATH.read_text())
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if report["protocol"] != PROTOCOL:
        raise RuntimeError("Stage-2E protocol drift")
    if report["fit_sha256"] != sha256_file(FIT_PATH):
        raise RuntimeError("Stage-2E fit hash drift")
    if report["raw_sha256"] != sha256_file(RAW_PATH):
        raise RuntimeError("Stage-2E raw hash drift")
    expected = {"A", "C-LR", *ARM_ORDER}
    if set(raw["arms"]) != expected:
        raise RuntimeError("Stage-2E raw arms incomplete")

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
        contrasts=build_contrasts(),
        draws=BOOTSTRAP_DRAWS,
    )
    output = {
        "protocol": PROTOCOL,
        "provenance": {
            "script_sha256": _sha(Path(__file__)),
            "fit_sha256": sha256_file(FIT_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
            "raw_sha256": sha256_file(RAW_PATH),
        },
        "paired": paired,
    }
    output["gate"] = evaluate_gates(paired, report, fit)
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(
        f"{OUTPUT}: selected={output['gate']['selected_arm']} "
        f"pass={output['gate']['pass']} "
        f"route={output['gate']['route']}"
    )


if __name__ == "__main__":
    main()
