"""Evaluate sealed Stage-2F checkpoints on spent DEV; never access FINAL."""
from __future__ import annotations

from dataclasses import replace
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

from checkpoint import (  # noqa: E402
    load_world_checkpoint,
    sprint_candidate_config,
)
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import enforce_frozen_encoder  # noqa: E402
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_evaluation import evaluate_arm  # noqa: E402
from stage2f_reward_operator import PROTOCOL  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
STAGE2C_RAW = ARTIFACTS / "stage2c_raw.json"
REFERENCE = ARTIFACTS / "stage2c_clr_s505.pt"
TRAIN_REPORT = ARTIFACTS / "stage2f_train_report.json"
TRAIN_RAW = ARTIFACTS / "stage2f_train_raw.json"
REPORT_PATH = ARTIFACTS / "stage2f_eval_report.json"
RAW_PATH = ARTIFACTS / "stage2f_eval_raw.json"

EXPECTED_STATIC_SHA256 = {
    MANIFEST: (
        "0b909b886e86bb221e9bd500da88bd38a7871c7e0534ccd159d2cf3c1b6c2bd4"
    ),
    STAGE2C_RAW: (
        "e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5"
    ),
    REFERENCE: (
        "60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae"
    ),
    TRAIN_REPORT: (
        "a602155d14badfc370a94cc922cc584d7fe1093f789b2e58de3b7f64928d4f08"
    ),
    TRAIN_RAW: (
        "9fdb9f318bbb847fafd18852ea7a43bdc96ff28abbb1339b1474fe8631759c87"
    ),
    ARTIFACTS / "stage2f_flz_s505.pt": (
        "e6b448b1cfa6415080ee0148618c84846916250107f5a6f6f6a44a1530511743"
    ),
    ARTIFACTS / "stage2f_fdz_s505.pt": (
        "171c3826f6f9c5791b3ef03476fc5e8014fd99437dd20aed7bd63706a73671cb"
    ),
}
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
ARM_CHECKPOINTS = {
    "F-R": (
        REFERENCE,
        "local_symlog",
        "93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49",
    ),
    "F-LZ": (
        ARTIFACTS / "stage2f_flz_s505.pt",
        "local_symlog",
        "e739308a4bfca57e5838fef2eb40f1f0f0b9b75648612add9e69c1cc44720645",
    ),
    "F-DZ": (
        ARTIFACTS / "stage2f_fdz_s505.pt",
        "dreamerv3_symexp",
        "d3af89bcc7a6df87581cd02a40b9dddd35ce4a1f71b83811ae38e170ddb114e7",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_contract(manifest: dict) -> dict:
    return manifest["dev"]


def assert_reference_exact(observed: dict, committed: dict) -> None:
    if observed != committed:
        for key in (
            "reward_predictions",
            "continuation_predictions",
            "latent_errors",
            "ranking_rows",
        ):
            if observed.get(key) != committed.get(key):
                raise RuntimeError(
                    f"F-R reference reconstruction drift at {key}"
                )
        raise RuntimeError("F-R reference reconstruction drift")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2F DEV evaluation requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2F checkpoints and evaluator before DEV:\n"
            + "\n".join(dirty)
        )
    for path, expected in EXPECTED_STATIC_SHA256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"static artifact drift: {path}")

    train_report = json.loads(TRAIN_REPORT.read_text())
    if train_report["raw_sha256"] != sha256_file(TRAIN_RAW):
        raise RuntimeError("Stage-2F training raw/report mismatch")
    for arm in ("F-LZ", "F-DZ"):
        path, operator, digest = ARM_CHECKPOINTS[arm]
        block = train_report["arms"][arm]
        if block["checkpoint_sha256"] != sha256_file(path):
            raise RuntimeError(f"{arm} training checkpoint hash drift")
        if block["operator"] != operator:
            raise RuntimeError(f"{arm} training operator drift")
        if block["final_digest"] != digest:
            raise RuntimeError(f"{arm} training state digest drift")

    manifest = json.loads(MANIFEST.read_text())
    dev = dev_contract(manifest)
    for key, expected in EXPECTED_DEV_SHA256.items():
        if dev[key]["sha256"] != expected:
            raise RuntimeError(f"DEV manifest drift for {key}")
        if sha256_file(Path(dev[key]["path"])) != expected:
            raise RuntimeError(f"DEV artifact drift for {key}")

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

    stage2c = json.loads(STAGE2C_RAW.read_text())
    targets = stage2c["targets"]
    if not np.array_equal(
        natural_arrays["rewards"],
        np.asarray(targets["reward_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2F natural target drift")
    if not np.array_equal(
        actual_continue,
        np.asarray(targets["continue_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2F continuation target drift")

    report = {
        "format": "stage2f_eval_report_v1",
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "static_sha256": {
            str(path): digest
            for path, digest in EXPECTED_STATIC_SHA256.items()
        },
        "dev_sha256": EXPECTED_DEV_SHA256,
        "train_report_sha256": EXPECTED_STATIC_SHA256[TRAIN_REPORT],
        "arms": {},
        "reference_exact": False,
    }
    raw = {
        "format": "stage2f_eval_raw_v1",
        "targets": targets,
        "arms": {
            "A": copy.deepcopy(stage2c["arms"]["A"]),
        },
    }

    for arm, (path, operator, expected_state) in (
        ARM_CHECKPOINTS.items()
    ):
        expected_config = replace(
            sprint_candidate_config("gru"),
            reward_operator=operator,
        )
        world, payload = load_world_checkpoint(
            path,
            device,
            expect_config=expected_config,
            expect_sha256=EXPECTED_STATIC_SHA256[path],
        )
        enforce_frozen_encoder(world)
        world.eval()
        before = state_digest(world, exclude_heads=False)
        if before != expected_state:
            raise RuntimeError(f"{arm} state digest drift before DEV")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        point, arm_raw = evaluate_arm(
            world,
            natural_arrays,
            terminal_arrays,
            actual_continue,
            anchors,
            device,
        )
        after = state_digest(world, exclude_heads=False)
        if after != before:
            raise RuntimeError(f"{arm} state changed during DEV")
        if arm == "F-R":
            assert_reference_exact(
                arm_raw, stage2c["arms"]["C-LR"]
            )
            report["reference_exact"] = True

        raw["arms"][arm] = arm_raw
        report["arms"][arm] = {
            "operator": operator,
            "checkpoint": str(path),
            "checkpoint_sha256": EXPECTED_STATIC_SHA256[path],
            "checkpoint_head": payload["provenance"]["head"],
            "state_digest_before": before,
            "state_digest_after": after,
            "metrics": point,
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20
            ),
        }
        print(
            f"[{arm}] K8 AUC="
            f"{point['reward_depth']['k8']['event_auroc']:.4f} "
            f"Pearson="
            f"{point['reward_depth']['k8']['reward_pearson']:.4f} "
            f"rank={point['ranking']['chosen_minus_random_mean']:.4f}",
            flush=True,
        )
        del world
        torch.cuda.empty_cache()

    if not report["reference_exact"]:
        raise RuntimeError("F-R exact reference control did not run")
    RAW_PATH.write_text(json.dumps(raw))
    report["raw_sha256"] = sha256_file(RAW_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2F DEV complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
