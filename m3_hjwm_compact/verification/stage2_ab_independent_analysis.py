"""Reproducible independent analysis of the committed Stage-2 A/B result."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import (  # noqa: E402
    load_world_checkpoint,
    sprint_candidate_config,
)
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import enforce_frozen_encoder  # noqa: E402
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from stage1b_equal_update_analysis import (  # noqa: E402
    reward_bootstrap_metrics,
    reward_metrics,
)
from stage2_ab import (  # noqa: E402
    BATCH,
    K_GEN,
    MANIFEST,
    PREFIX,
    SEED,
    TERMINAL_MIX,
    UPDATES,
    build_schedule,
    make_batch,
    terminal_pool_10,
)
from stage2_evaluation import (  # noqa: E402
    _bootstrap_metric_contrasts,
    evaluate_arm,
    paired_analysis,
)
from stage2c_decoupled import (  # noqa: E402
    component_gradient_diagnostic,
    training_distribution,
)
from step3_temporal import load_scaled_data  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
STAGE2_REPORT = ARTIFACTS / "stage2_ab_report.json"
RAW_PATH = ARTIFACTS / "stage2_ab_independent_raw.json"
OUTPUT = ARTIFACTS / "stage2_ab_independent_analysis.json"
CHECKPOINTS = {
    "A": ARTIFACTS / "stage2_armA_s505.pt",
    "B": ARTIFACTS / "stage2_armB_s505.pt",
}
CONTRASTS = {"B_minus_A": {"B": 1.0, "A": -1.0}}
BOOTSTRAP_DRAWS = 2_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generated_distribution(
    train: list[dict],
    picks: list[tuple[int, int]],
) -> dict:
    per_depth_reward = [[] for _ in range(K_GEN)]
    per_depth_continue = [[] for _ in range(K_GEN)]
    for episode, start in picks:
        for depth in range(K_GEN):
            transition = start + PREFIX - 1 + depth
            per_depth_reward[depth].append(
                train[episode]["rewards"][transition]
            )
            per_depth_continue[depth].append(
                train[episode]["continues"][transition]
            )
    output = {"depths": {}}
    for depth, (reward, cont) in enumerate(
        zip(per_depth_reward, per_depth_continue), start=1
    ):
        reward = np.asarray(reward, dtype=np.float32)
        cont = np.asarray(cont, dtype=np.float32)
        output["depths"][f"k{depth}"] = {
            "labels": len(reward),
            "event_fraction": float(np.mean(np.abs(reward) > 1e-6)),
            "terminal_fraction": float(np.mean(cont < 0.5)),
            "mean_reward": float(reward.mean()),
        }
    reward = np.concatenate([
        np.asarray(value, dtype=np.float32)
        for value in per_depth_reward
    ])
    cont = np.concatenate([
        np.asarray(value, dtype=np.float32)
        for value in per_depth_continue
    ])
    output["overall"] = {
        "labels": len(reward),
        "event_fraction": float(np.mean(np.abs(reward) > 1e-6)),
        "terminal_fraction": float(np.mean(cont < 0.5)),
        "mean_reward": float(reward.mean()),
    }
    return output


def realized_terminal_picks(
    pool: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(50_000 + SEED)
    picks = []
    for update in range(UPDATES):
        if update % int(1 / TERMINAL_MIX) == 0:
            picks.extend(
                pool[int(rng.integers(len(pool)))]
                for _ in range(BATCH)
            )
    return picks


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("independent Stage-2 audit requires CUDA")
    device = torch.device("cuda")
    report = json.loads(STAGE2_REPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    dev = manifest["dev"]
    for key in ("natural", "terminal", "bundle"):
        if sha256_file(Path(dev[key]["path"])) != dev[key]["sha256"]:
            raise RuntimeError(f"dev {key} hash drift")

    natural_episodes = torch.load(
        Path(dev["natural"]["path"]), weights_only=False
    )
    terminal_episodes = torch.load(
        Path(dev["terminal"]["path"]), weights_only=False
    )
    anchors = torch.load(Path(dev["bundle"]["path"]), weights_only=False)
    natural_rows = target_rows(natural_episodes)
    natural_arrays = window_arrays(natural_episodes, natural_rows)
    terminal_rows = target_rows(terminal_episodes)
    terminal_arrays = window_arrays(terminal_episodes, terminal_rows)
    actual_continue = continuation_targets(
        terminal_episodes, terminal_rows
    )

    raw = {
        "targets": {
            "reward_actual": natural_arrays["rewards"].tolist(),
            "reward_episode": natural_arrays["episodes"].tolist(),
            "reward_transition": natural_arrays["transitions"].tolist(),
            "reward_continue": continuation_targets(
                natural_episodes, natural_rows
            ).tolist(),
            "continue_actual": actual_continue.tolist(),
            "continue_episode": terminal_arrays["episodes"].tolist(),
            "continue_transition": terminal_arrays[
                "transitions"
            ].tolist(),
        },
        "arms": {},
    }
    points = {}
    for arm, path in CHECKPOINTS.items():
        expected = report["arms"][arm]["checkpoint_sha256"]
        world, _ = load_world_checkpoint(
            path,
            device,
            expect_config=sprint_candidate_config("gru"),
            expect_sha256=expected,
        )
        enforce_frozen_encoder(world)
        world.eval()
        points[arm], raw["arms"][arm] = evaluate_arm(
            world,
            natural_arrays,
            terminal_arrays,
            actual_continue,
            anchors,
            device,
        )
        del world
        torch.cuda.empty_cache()

    RAW_PATH.write_text(json.dumps(raw))
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

    nonterminal = np.asarray(
        targets["reward_continue"], dtype=np.float32
    ) > 0.5
    filtered_predictions = {
        arm: np.asarray(
            raw["arms"][arm]["reward_predictions"]["k8"]
        )[nonterminal]
        for arm in ("A", "B")
    }
    filtered_actual = np.asarray(
        targets["reward_actual"], dtype=np.float32
    )[nonterminal]
    filtered_clusters = np.asarray(
        targets["reward_episode"], dtype=np.int64
    )[nonterminal]
    terminal_excluded = {
        "points": {
            arm: reward_metrics(values, filtered_actual)
            for arm, values in filtered_predictions.items()
        },
        "contrasts": _bootstrap_metric_contrasts(
            filtered_predictions,
            filtered_actual,
            filtered_clusters,
            reward_bootstrap_metrics,
            CONTRASTS,
            seed=28_808,
            draws=BOOTSTRAP_DRAWS,
        ),
        "excluded_rows": int((~nonterminal).sum()),
    }

    train, _ = load_scaled_data()
    schedule, schedule_digest = build_schedule(train)
    pool = terminal_pool_10(train)
    terminal_realized = realized_terminal_picks(pool)
    first_natural = make_batch(train, schedule[:BATCH], device)
    first_terminal = make_batch(
        train, pool[:BATCH], device, window=PREFIX + K_GEN
    )
    gradient = {
        "natural": component_gradient_diagnostic(
            first_natural, device
        ),
        "terminal": component_gradient_diagnostic(
            first_terminal, device
        ),
    }

    output = {
        "review": "reviews/2026-07-18-stage2-independent-audit.md",
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "hashes": {
            "stage2_report": sha256_file(STAGE2_REPORT),
            "manifest": sha256_file(MANIFEST),
            "raw": sha256_file(RAW_PATH),
            "checkpoints": {
                arm: sha256_file(path)
                for arm, path in CHECKPOINTS.items()
            },
        },
        "schedule_sha256": schedule_digest,
        "point_reproduction": points,
        "paired": paired,
        "k8_reward_excluding_terminal_targets": terminal_excluded,
        "training_distribution": {
            "main": training_distribution(train, schedule),
            "terminal_pool_windows": len(pool),
            "terminal_pool": generated_distribution(train, pool),
            "realized_terminal_samples": len(terminal_realized),
            "realized_terminal": generated_distribution(
                train, terminal_realized
            ),
        },
        "component_gradient_diagnostic": gradient,
        "verdict": {
            "implementation_indexing": "PASS",
            "full_registered_acceptance": "FAIL",
            "broad_world_repair_claim": "REFUTED",
            "narrow_k8_event_discrimination": "SUPPORTED",
            "replication_mamba_planner_final": "NO_GO",
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"{OUTPUT} {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
