"""Evaluate committed Stage-2E categorical scalars on spent DEV only."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import load_world_checkpoint, sprint_candidate_config  # noqa: E402
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import enforce_frozen_encoder  # noqa: E402
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import HORIZONS, target_rows, window_arrays  # noqa: E402
from phase_e_taskheads import GAMMA, clone_world_state  # noqa: E402
from stage1b_equal_update_analysis import reward_metrics  # noqa: E402
from stage2d_reward_head import selected_state_digest  # noqa: E402
from stage2e_calibration import (  # noqa: E402
    ARM_ORDER,
    CalibrationSpec,
    calibration_nll,
    collect_same_target_logits,
    decode_calibrated,
    select_calibrator,
)
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
PROTOCOL = "reviews/2026-07-18-stage2e-categorical-calibration-protocol.md"
CHECKPOINT = ARTIFACTS / "stage2c_clr_s505.pt"
FIT_PATH = ARTIFACTS / "stage2e_calibration_fit.json"
MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
STAGE2C_RAW = ARTIFACTS / "stage2c_raw.json"
REPORT_PATH = ARTIFACTS / "stage2e_report.json"
RAW_PATH = ARTIFACTS / "stage2e_raw.json"

EXPECTED_FIT_SHA256 = (
    "6c9f436fb64e1c6b92fa9cc3b351e24b4a49063cc430145f29a519a874351a0d"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae"
)
EXPECTED_STATE_DIGEST = (
    "93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49"
)
EXPECTED_MANIFEST_SHA256 = (
    "0b909b886e86bb221e9bd500da88bd38a7871c7e0534ccd159d2cf3c1b6c2bd4"
)
EXPECTED_STAGE2C_RAW_SHA256 = (
    "e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5"
)
EXPECTED_DEV_SHA256 = {
    "natural": (
        "5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e"
    ),
    "terminal": (
        "14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf"
    ),
    "bundle": (
        "d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8"
    ),
}


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_contract(manifest: dict) -> dict:
    """Expose only the spent evaluation tier for a permanent access test."""
    return manifest["dev"]


def load_specs(fit: dict) -> dict[str, CalibrationSpec]:
    if fit.get("format") != "stage2e_calibration_fit_v1":
        raise RuntimeError("invalid Stage-2E CAL artifact format")
    specs = {}
    for arm in ARM_ORDER:
        block = fit["fits"][arm]["spec"]
        if block["arm"] != arm:
            raise RuntimeError(f"CAL spec arm mismatch for {arm}")
        specs[arm] = CalibrationSpec(
            arm=arm,
            log_temperature=float(block["log_temperature"]),
            zero_bias=float(block["zero_bias"]),
        )
    if fit["selected_arm"] not in specs:
        raise RuntimeError("CAL-selected arm missing")
    if select_calibrator(fit["fits"]) != fit["selected_arm"]:
        raise RuntimeError("CAL selection does not reproduce from NLLs")
    return specs


def _summarize_ranking_rows(rows: list[dict]) -> dict:
    differing = [row for row in rows if row["differs"]]

    def within_corrs(kind):
        pearson, spearman = [], []
        for row in differing:
            actual = np.array(list(row["actual"].values()))
            predicted = np.array([
                row[kind][name] for name in row["actual"]
            ])
            if actual.std() == 0 or predicted.std() == 0:
                continue
            pearson.append(float(np.corrcoef(predicted, actual)[0, 1]))
            rank_actual = actual.argsort().argsort().astype(float)
            rank_predicted = predicted.argsort().argsort().astype(float)
            spearman.append(float(np.corrcoef(
                rank_predicted, rank_actual
            )[0, 1]))
        return (
            float(np.mean(pearson)) if pearson else None,
            float(np.mean(spearman)) if spearman else None,
        )

    pearson_gated, spearman_gated = within_corrs("j_gated")
    pearson_sum, spearman_sum = within_corrs("j_sum")
    advantage = np.array([
        row["chosen_minus_random"] for row in differing
    ])
    by_environment = {}
    for row in differing:
        by_environment.setdefault(row["env_seed"], []).append(
            row["chosen_minus_random"]
        )
    rng = np.random.default_rng(0)
    clusters = [
        np.asarray(values) for values in by_environment.values()
    ]
    bootstrap = np.empty(10_000)
    for index in range(10_000):
        picked = rng.integers(len(clusters), size=len(clusters))
        bootstrap[index] = float(np.mean(np.concatenate([
            clusters[value] for value in picked
        ])))
    return {
        "n_differing": len(differing),
        "pearson_gated": pearson_gated,
        "spearman_gated": spearman_gated,
        "pearson_sum": pearson_sum,
        "spearman_sum": spearman_sum,
        "chosen_minus_random_mean": (
            float(advantage.mean()) if len(advantage) else None
        ),
        "chosen_minus_random_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "regret_mean": (
            float(np.mean([row["regret"] for row in differing]))
            if differing else None
        ),
    }


@torch.no_grad()
def calibrated_ranking_metrics(
    world,
    anchors: list[dict],
    device: torch.device,
    spec: CalibrationSpec,
) -> dict:
    rows = []
    for anchor in anchors:
        obs = torch.from_numpy(anchor["obs_hist"][None]).to(device)
        action_history = anchor["act_hist"]
        state = world.initial_state(1, device)
        for index in range(len(action_history)):
            previous = torch.tensor(
                [int(action_history[index])], device=device
            )
            with _autocast(device):
                state = world.observe_step(
                    obs[:, index], previous, state
                )
        gated, raw, actual = {}, {}, {}
        for name, suffix in anchor["suffixes"].items():
            branch_state = clone_world_state(state)
            alive = torch.ones(1, device=device)
            total_gated = torch.zeros(1, device=device)
            total_raw = torch.zeros(1, device=device)
            for depth, action_value in enumerate(suffix):
                action = torch.tensor(
                    [int(action_value)], device=device
                )
                with _autocast(device):
                    branch_state, reward_logits, continue_logits, _ = (
                        world.imagine_step(
                            branch_state,
                            action,
                            deterministic_mode=True,
                        )
                    )
                reward = decode_calibrated(
                    reward_logits.float(),
                    spec,
                    low=world.cfg.reward_low,
                    high=world.cfg.reward_high,
                )
                total_gated = (
                    total_gated
                    + (GAMMA ** depth) * alive * reward
                )
                total_raw = total_raw + reward
                alive = alive * torch.sigmoid(continue_logits.float())
            gated[name] = float(total_gated)
            raw[name] = float(total_raw)
            actual[name] = float(np.mean([
                outcome["reward_sum"]
                for outcome in anchor["branches"][name]["outcomes"]
            ]))
        names = list(anchor["suffixes"])
        actual_values = np.array([actual[name] for name in names])
        gated_values = np.array([gated[name] for name in names])
        chosen = names[int(gated_values.argmax())]
        rows.append({
            "env_seed": int(anchor["env_seed"]),
            "night": bool(anchor["night"]),
            "j_gated": gated,
            "j_sum": raw,
            "actual": actual,
            "differs": bool(actual_values.std() > 1e-9),
            "chosen_minus_random": float(
                actual[chosen] - actual_values.mean()
            ),
            "regret": float(actual_values.max() - actual[chosen]),
        })
    return {
        "rows": rows,
        **_summarize_ranking_rows(rows),
    }


def assert_identity(
    predictions: dict[str, list[float]],
    ranking_rows: list[dict],
    committed: dict,
) -> None:
    for key, expected in committed["reward_predictions"].items():
        if not np.array_equal(
            np.asarray(predictions[key]), np.asarray(expected)
        ):
            raise RuntimeError(f"E-I reward prediction drift at {key}")
    if ranking_rows != committed["ranking_rows"]:
        raise RuntimeError("E-I fork rows differ from committed C-LR")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2E DEV evaluation requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit CAL artifact and evaluator pin before DEV:\n"
            + "\n".join(dirty)
        )
    if EXPECTED_FIT_SHA256.startswith("__"):
        raise RuntimeError("pin committed CAL artifact hash before DEV")

    static = {
        CHECKPOINT: EXPECTED_CHECKPOINT_SHA256,
        FIT_PATH: EXPECTED_FIT_SHA256,
        MANIFEST: EXPECTED_MANIFEST_SHA256,
        STAGE2C_RAW: EXPECTED_STAGE2C_RAW_SHA256,
    }
    for path, expected in static.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"static artifact drift: {path}")
    fit = json.loads(FIT_PATH.read_text())
    specs = load_specs(fit)
    manifest = json.loads(MANIFEST.read_text())
    dev = dev_contract(manifest)
    for key, expected in EXPECTED_DEV_SHA256.items():
        if dev[key]["sha256"] != expected:
            raise RuntimeError(f"DEV manifest drift for {key}")
        if sha256_file(Path(dev[key]["path"])) != expected:
            raise RuntimeError(f"DEV artifact drift for {key}")

    world, _ = load_world_checkpoint(
        CHECKPOINT,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    enforce_frozen_encoder(world)
    world.eval()
    before = selected_state_digest(world, reward=None)
    if before != EXPECTED_STATE_DIGEST:
        raise RuntimeError("C-LR state digest drift")

    natural_episodes = torch.load(
        Path(dev["natural"]["path"]), weights_only=False
    )
    terminal_episodes = torch.load(
        Path(dev["terminal"]["path"]), weights_only=False
    )
    anchors = torch.load(
        Path(dev["bundle"]["path"]), weights_only=False
    )
    natural_arrays = window_arrays(
        natural_episodes, target_rows(natural_episodes)
    )
    terminal_rows = target_rows(terminal_episodes)
    terminal_arrays = window_arrays(terminal_episodes, terminal_rows)
    actual_continue = continuation_targets(
        terminal_episodes, terminal_rows
    )
    del terminal_arrays
    stage2c = json.loads(STAGE2C_RAW.read_text())
    targets = stage2c["targets"]
    actual = np.asarray(targets["reward_actual"], dtype=np.float32)
    if not np.array_equal(natural_arrays["rewards"], actual):
        raise RuntimeError("Stage-2E natural targets differ from Stage-2C")
    if not np.array_equal(
        actual_continue,
        np.asarray(targets["continue_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2E continuation targets differ")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits_by_depth = collect_same_target_logits(
        world, natural_arrays, device
    )
    report = {
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "fit_sha256": EXPECTED_FIT_SHA256,
        "selected_arm": fit["selected_arm"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "state_digest_before": before,
        "arms": {},
        "identity": {},
        "frozen_outputs": {
            "continuation_reused_exact": True,
            "latent_reused_exact": True,
        },
    }
    raw = {
        "targets": targets,
        "arms": {
            "A": stage2c["arms"]["A"],
            "C-LR": stage2c["arms"]["C-LR"],
        },
    }
    for arm in ARM_ORDER:
        spec = specs[arm]
        predictions = {}
        metrics = {}
        for depth in HORIZONS:
            key = f"k{depth}"
            logits = logits_by_depth[key]
            rewards = torch.from_numpy(actual)
            decoded = decode_calibrated(
                logits,
                spec,
                low=world.cfg.reward_low,
                high=world.cfg.reward_high,
            ).numpy()
            predictions[key] = decoded.tolist()
            metrics[key] = {
                **reward_metrics(decoded, actual),
                "nll": float(calibration_nll(
                    logits,
                    rewards,
                    spec,
                    low=world.cfg.reward_low,
                    high=world.cfg.reward_high,
                ).mean()),
            }
        ranking = calibrated_ranking_metrics(
            world, anchors, device, spec
        )
        ranking_rows = ranking.pop("rows")
        arm_raw = {
            "reward_predictions": predictions,
            "continuation_predictions": copy.deepcopy(
                stage2c["arms"]["C-LR"][
                    "continuation_predictions"
                ]
            ),
            "latent_errors": copy.deepcopy(
                stage2c["arms"]["C-LR"]["latent_errors"]
            ),
            "ranking_rows": ranking_rows,
        }
        if arm == "E-I":
            assert_identity(
                predictions,
                ranking_rows,
                stage2c["arms"]["C-LR"],
            )
            report["identity"] = {
                "reward_predictions_exact": True,
                "ranking_rows_exact": True,
            }
        raw["arms"][arm] = arm_raw
        report["arms"][arm] = {
            "spec": spec.to_dict(),
            "reward_depth": metrics,
            "ranking": ranking,
        }
        print(
            f"[{arm}] K8 AUC={metrics['k8']['event_auroc']:.4f} "
            f"Pearson={metrics['k8']['reward_pearson']:.4f} "
            f"rank={ranking['chosen_minus_random_mean']:.4f}",
            flush=True,
        )

    after = selected_state_digest(world, reward=None)
    if after != before:
        raise RuntimeError("world state changed during Stage-2E DEV")
    report["state_digest_after"] = after
    report["wall_seconds"] = time.perf_counter() - started
    report["peak_allocated_mib"] = (
        torch.cuda.max_memory_allocated() / 2**20
    )
    report["peak_reserved_mib"] = (
        torch.cuda.max_memory_reserved() / 2**20
    )
    RAW_PATH.write_text(json.dumps(raw))
    report["raw_sha256"] = sha256_file(RAW_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2E DEV complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
