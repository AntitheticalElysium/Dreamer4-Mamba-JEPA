"""Paired analysis and executable acceptance decision for Stage-2C."""
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
from stage2c_decoupled import (  # noqa: E402
    ARTIFACTS,
    PROTOCOL,
    RAW_PATH,
    REPORT_PATH,
)


OUTPUT = ARTIFACTS / "stage2c_analysis.json"
BOOTSTRAP_DRAWS = 2_000
CONTRASTS = {
    "latent_vs_base": {"C-L": 1.0, "A": -1.0},
    "reward_increment": {"C-LR": 1.0, "C-L": -1.0},
    "candidate_vs_base": {"C-LR": 1.0, "A": -1.0},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _not_significantly_lower(block: dict) -> bool:
    """Higher-is-better delta is not significantly harmful."""
    return block["ci95"][1] >= 0.0


def _not_significantly_higher(block: dict) -> bool:
    """Lower-is-better delta is not significantly harmful."""
    return block["ci95"][0] <= 0.0


def _condition(name: str, passed: bool, **evidence) -> dict:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def evaluate_gates(analysis: dict) -> dict:
    g1 = []
    latent_g1 = {
        depth: analysis["latent"][f"k{depth}"]["contrasts"][
            "latent_vs_base"
        ]["cosine_error"]
        for depth in (1, 2, 4, 8)
    }
    g1.append(_condition(
        "C-L improves K1 and K2 latent point estimates",
        all(latent_g1[depth]["delta"] < 0 for depth in (1, 2)),
        values={f"k{depth}": latent_g1[depth] for depth in (1, 2)},
    ))
    g1.append(_condition(
        "at least one K1/K2 latent CI excludes zero in the good direction",
        any(latent_g1[depth]["ci95"][1] < 0 for depth in (1, 2)),
        values={f"k{depth}": latent_g1[depth] for depth in (1, 2)},
    ))
    g1.append(_condition(
        "K4/K8 latent harm absent and point delta <= .005",
        all(
            _not_significantly_higher(latent_g1[depth])
            and latent_g1[depth]["delta"] <= 0.005
            for depth in (4, 8)
        ),
        values={f"k{depth}": latent_g1[depth] for depth in (4, 8)},
    ))

    for depth in (0, 1, 2, 4, 8):
        block = analysis["continuation"][f"k{depth}"]
        delta = block["contrasts"]["latent_vs_base"]
        points = block["points"]
        brier = delta["brier_skill"]
        candidate_terminal = points["C-L"][
            "predicted_termination_nonterminal_mean"
        ]
        baseline_terminal = points["A"][
            "predicted_termination_nonterminal_mean"
        ]
        ceiling = max(baseline_terminal + 0.01, 0.02)
        g1.append(_condition(
            f"C-L continuation safety K{depth}",
            _not_significantly_lower(brier)
            and candidate_terminal <= ceiling,
            brier_skill=brier,
            candidate_nonterminal_termination=candidate_terminal,
            baseline_nonterminal_termination=baseline_terminal,
            ceiling=ceiling,
        ))

    ranking_g1 = analysis["ranking"]["contrasts"]["latent_vs_base"]
    zero_g1 = analysis["zero_suffix"]["contrasts"]["latent_vs_base"][
        "zero_suffix_abs_predicted_sum"
    ]
    g1.append(_condition(
        "C-L ranking not significantly worse",
        _not_significantly_lower(ranking_g1["chosen_minus_random"])
        and _not_significantly_higher(ranking_g1["regret"]),
        values=ranking_g1,
    ))
    g1.append(_condition(
        "C-L absolute zero-suffix reward not significantly worse",
        _not_significantly_higher(zero_g1),
        value=zero_g1,
    ))

    g2 = []
    reward_k8 = analysis["reward"]["k8"]
    points_a = reward_k8["points"]["A"]
    points_l = reward_k8["points"]["C-L"]
    points_lr = reward_k8["points"]["C-LR"]
    total = reward_k8["contrasts"]["candidate_vs_base"]
    increment = reward_k8["contrasts"]["reward_increment"]
    point_metrics = (
        "event_auroc",
        "event_average_precision",
        "reward_pearson",
        "decoded_abs_event_mean",
    )
    g2.append(_condition(
        "C-LR K8 reward points improve versus A",
        all(points_lr[name] > points_a[name] for name in point_metrics),
        baseline={name: points_a[name] for name in point_metrics},
        candidate={name: points_lr[name] for name in point_metrics},
    ))
    g2.append(_condition(
        "C-LR K8 reward increment improves AUROC and magnitude versus C-L",
        points_lr["event_auroc"] > points_l["event_auroc"]
        and points_lr["decoded_abs_event_mean"]
        > points_l["decoded_abs_event_mean"],
        latent_arm={
            "event_auroc": points_l["event_auroc"],
            "decoded_abs_event_mean": points_l["decoded_abs_event_mean"],
        },
        candidate={
            "event_auroc": points_lr["event_auroc"],
            "decoded_abs_event_mean": points_lr["decoded_abs_event_mean"],
        },
    ))
    g2.append(_condition(
        "C-LR versus A K8 AUROC paired CI excludes zero",
        total["event_auroc"]["ci95"][0] > 0.0,
        value=total["event_auroc"],
    ))

    reward_k1 = analysis["reward"]["k1"]["contrasts"][
        "candidate_vs_base"
    ]
    g2.append(_condition(
        "C-LR K1 AUROC/Pearson not significantly worse",
        _not_significantly_lower(reward_k1["event_auroc"])
        and _not_significantly_lower(reward_k1["reward_pearson"]),
        values={
            "event_auroc": reward_k1["event_auroc"],
            "reward_pearson": reward_k1["reward_pearson"],
        },
    ))

    zero_g2 = analysis["zero_suffix"]["contrasts"][
        "candidate_vs_base"
    ]["zero_suffix_abs_predicted_sum"]
    g2.append(_condition(
        "C-LR zero-suffix absolute reward delta <= .02 including CI",
        zero_g2["delta"] <= 0.02 and zero_g2["ci95"][1] <= 0.02,
        value=zero_g2,
    ))

    for depth in (1, 2, 4, 8):
        versus_l = analysis["latent"][f"k{depth}"]["contrasts"][
            "reward_increment"
        ]["cosine_error"]
        versus_a = analysis["latent"][f"k{depth}"]["contrasts"][
            "candidate_vs_base"
        ]["cosine_error"]
        g2.append(_condition(
            f"C-LR latent safety K{depth}",
            _not_significantly_higher(versus_l)
            and versus_a["delta"] <= 0.005,
            versus_latent_arm=versus_l,
            versus_baseline=versus_a,
        ))

    for depth in (0, 1, 2, 4, 8):
        block = analysis["continuation"][f"k{depth}"]
        delta = block["contrasts"]["candidate_vs_base"]
        points = block["points"]
        brier = delta["brier_skill"]
        candidate_terminal = points["C-LR"][
            "predicted_termination_nonterminal_mean"
        ]
        baseline_terminal = points["A"][
            "predicted_termination_nonterminal_mean"
        ]
        ceiling = max(baseline_terminal + 0.01, 0.02)
        g2.append(_condition(
            f"C-LR continuation safety K{depth}",
            _not_significantly_lower(brier)
            and candidate_terminal <= ceiling,
            brier_skill=brier,
            candidate_nonterminal_termination=candidate_terminal,
            baseline_nonterminal_termination=baseline_terminal,
            ceiling=ceiling,
        ))

    ranking_g2 = analysis["ranking"]["contrasts"]["candidate_vs_base"]
    g2.append(_condition(
        "C-LR ranking not significantly worse",
        _not_significantly_lower(ranking_g2["chosen_minus_random"])
        and _not_significantly_higher(ranking_g2["regret"]),
        values=ranking_g2,
    ))

    g1_pass = all(item["pass"] for item in g1)
    g2_pass = all(item["pass"] for item in g2)
    if not g1_pass:
        route = (
            "STOP_FULL_WORLD_GENERATED_EXPANSION; do not transfer to Mamba "
            "or add task curricula"
        )
    elif not g2_pass:
        route = (
            "LATENT_DIAGNOSTIC_ONLY; reject generated reward through shared "
            "dynamics and investigate separated task adaptation"
        )
    else:
        route = (
            "REPLICATE_GRU_606_707_THEN_MATCHED_MAMBA; FINAL and planner "
            "remain untouched pending replication"
        )
    return {
        "G1_generated_latent": {
            "pass": g1_pass,
            "conditions": g1,
        },
        "G2_generated_reward": {
            "pass": g2_pass,
            "conditions": g2,
        },
        "overall_pass": g1_pass and g2_pass,
        "route": route,
    }


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if report["protocol"] != PROTOCOL:
        raise RuntimeError("Stage-2C protocol drift")
    if report["hashes"]["raw"] != sha256_file(RAW_PATH):
        raise RuntimeError("Stage-2C raw artifact hash drift")
    expected_arms = {"A", "C-L", "C-LR"}
    if set(raw["arms"]) != expected_arms:
        raise RuntimeError("Stage-2C raw arms incomplete")
    for arm in ("C-L", "C-LR"):
        path = Path(report["arms"][arm]["checkpoint"])
        if sha256_file(path) != report["arms"][arm]["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} checkpoint hash drift")

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
            "checkpoint_sha256": report["hashes"]["checkpoints"],
        },
        "paired": result,
    }
    output["gates"] = evaluate_gates(result)
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(
        f"{OUTPUT}: overall={output['gates']['overall_pass']} "
        f"route={output['gates']['route']}"
    )


if __name__ == "__main__":
    main()
