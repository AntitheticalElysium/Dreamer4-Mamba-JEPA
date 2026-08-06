"""Evaluate sealed Stage-2G worlds on spent DEV; never access FINAL."""
from __future__ import annotations

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
from stage2g_relevance import PROTOCOL  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
STAGE2C_RAW = ARTIFACTS / "stage2c_raw.json"
PREFLIGHT = ARTIFACTS / "stage2g_preflight.json"
TRAIN_REPORT = ARTIFACTS / "stage2g_train_report.json"
TRAIN_RAW = ARTIFACTS / "stage2g_train_raw.json"
REPORT_PATH = ARTIFACTS / "stage2g_eval_report.json"
RAW_PATH = ARTIFACTS / "stage2g_eval_raw.json"

EXPECTED_STATIC_SHA256 = {
    MANIFEST: (
        "0b909b886e86bb221e9bd500da88bd38a7871c7e0534ccd159d2cf3c1b6c2bd4"
    ),
    STAGE2C_RAW: (
        "e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5"
    ),
    PREFLIGHT: (
        "5551ead595a0d1ae71d4e479918176439e1a1405cbcdb11b07d9159919f5b97d"
    ),
    TRAIN_REPORT: (
        "4cc81e774c9d7ab21fa667b03ce12d47ec9ef20a4a82c35f0a90184c5f2e8e60"
    ),
    TRAIN_RAW: (
        "87637ab2ed4df4d77f06f661d6449c2bf87b3aef5b868dff68a62bf8c7290876"
    ),
    ARTIFACTS / "stage2_armA_s505.pt": (
        "fcbc9407a36faf59e32ec1425c2fbee7a5e5a21ea73cb13170a828e4e9c6d1f2"
    ),
    ARTIFACTS / "stage2c_cl_s505.pt": (
        "227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1"
    ),
    ARTIFACTS / "stage2c_clr_s505.pt": (
        "60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae"
    ),
    ARTIFACTS / "stage2g_gla_s505.pt": (
        "c7c909654b6eda45149e080417da2c1fb0637120b9c725b3e0ff2482392336e5"
    ),
    ARTIFACTS / "stage2g_glra_s505.pt": (
        "40cdbf59b23b9878e2ec1660e795babf3b0254d99dbcd889939135f84c0f7823"
    ),
}
EXPECTED_TRAINING_CONTRACT = {
    "preflight_sha256": EXPECTED_STATIC_SHA256[PREFLIGHT],
    "replay_sha256": (
        "c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad"
    ),
    "base_schedule_sha256": (
        "427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0"
    ),
    "auxiliary_schedule_sha256": (
        "d109da9a1c8950ec929dd5dcdf5873e871f78c40a26cf0b5a5413e22d1550f1b"
    ),
    "probe_sha256": (
        "9c4c2b80017e6b4e687fc3c44c91e954021a4a2ef828e1522a55f3eebe5d0fae"
    ),
    "lambda_aux": 0.19708130570134666,
    "updates": 16_000,
    "batch": 4,
    "world_initial_digest": (
        "55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260"
    ),
    "auxiliary_initial_digest": (
        "79070c71c08dccb9c118664216eae15ab06c1a37a4e7a3514595c7bd6c8ec107"
    ),
}
EXPECTED_ENCODER_SHA256 = (
    "3cc79446d18aaeea3f8c022e20f8d2b63db1bf33f5e7f7f3bf9ef759d3f825cc"
)
EXPECTED_ARM_TRAINING = {
    "G-LA": {
        "generated_reward_weight": 0.0,
        "world_final_digest": (
            "f0ebb0346bfa66162a71252d202093c41d70fa1d76d2e0da689f16a0bab56d1a"
        ),
        "auxiliary_final_digest": (
            "9c5d328789b7744bc9ea1816533d09dab409d008d632f2dad48e395c76d525d8"
        ),
    },
    "G-LRA": {
        "generated_reward_weight": 0.1,
        "world_final_digest": (
            "a0e2cb7817de699cdf06307ac6e46d00c5478ecb925cdfbee4ee6d9ee1394cb8"
        ),
        "auxiliary_final_digest": (
            "7e98d18d77a92c45268ed277441b799b55f50cd359b3c91ffbb0e2ebd45e002f"
        ),
    },
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
    "A": (
        ARTIFACTS / "stage2_armA_s505.pt",
        "6467e3194ce82844ec15447035fb75e606763e83d0c76052848d77578124e0e4",
    ),
    "C-L": (
        ARTIFACTS / "stage2c_cl_s505.pt",
        "a0cf4ec132a9e023ecf71fa63d7f1f8e17dd00d6080684f1e7b6962844b8c1c9",
    ),
    "C-LR": (
        ARTIFACTS / "stage2c_clr_s505.pt",
        "93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49",
    ),
    "G-LA": (
        ARTIFACTS / "stage2g_gla_s505.pt",
        "f0ebb0346bfa66162a71252d202093c41d70fa1d76d2e0da689f16a0bab56d1a",
    ),
    "G-LRA": (
        ARTIFACTS / "stage2g_glra_s505.pt",
        "a0e2cb7817de699cdf06307ac6e46d00c5478ecb925cdfbee4ee6d9ee1394cb8",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_contract(manifest: dict) -> dict:
    return manifest["dev"]


def assert_reference_exact(
    arm: str,
    observed: dict,
    committed: dict,
) -> None:
    if observed != committed:
        for key in (
            "reward_predictions",
            "continuation_predictions",
            "latent_errors",
            "ranking_rows",
        ):
            if observed.get(key) != committed.get(key):
                raise RuntimeError(
                    f"{arm} reference reconstruction drift at {key}"
                )
        raise RuntimeError(f"{arm} reference reconstruction drift")


def assert_training_contract(train_report: dict) -> None:
    if train_report.get("protocol") != PROTOCOL:
        raise RuntimeError("Stage-2G training protocol drift")
    for key, expected in EXPECTED_TRAINING_CONTRACT.items():
        if train_report.get(key) != expected:
            raise RuntimeError(f"Stage-2G training contract drift at {key}")
    if set(train_report.get("arms", {})) != set(EXPECTED_ARM_TRAINING):
        raise RuntimeError("Stage-2G training arms drift")
    for arm, expected in EXPECTED_ARM_TRAINING.items():
        block = train_report["arms"][arm]
        for key, value in expected.items():
            if block.get(key) != value:
                raise RuntimeError(
                    f"Stage-2G {arm} training contract drift at {key}"
                )
        if block.get("updates") != EXPECTED_TRAINING_CONTRACT["updates"]:
            raise RuntimeError(f"Stage-2G {arm} update count drift")


def assert_checkpoint_contract(
    arm: str,
    payload: dict,
    train_report: dict,
) -> None:
    expected = EXPECTED_ARM_TRAINING[arm]
    extra = payload.get("extra", {})
    exact = {
        "protocol": PROTOCOL,
        "arm": arm,
        "seed": 505,
        "updates": EXPECTED_TRAINING_CONTRACT["updates"],
        "generated_reward_weight": expected["generated_reward_weight"],
        "lambda_aux": EXPECTED_TRAINING_CONTRACT["lambda_aux"],
        "preflight_sha256": EXPECTED_TRAINING_CONTRACT[
            "preflight_sha256"
        ],
        "base_schedule_sha256": EXPECTED_TRAINING_CONTRACT[
            "base_schedule_sha256"
        ],
        "auxiliary_schedule_sha256": EXPECTED_TRAINING_CONTRACT[
            "auxiliary_schedule_sha256"
        ],
        "probe_sha256": EXPECTED_TRAINING_CONTRACT["probe_sha256"],
        "world_initial_digest": EXPECTED_TRAINING_CONTRACT[
            "world_initial_digest"
        ],
        "world_final_digest": expected["world_final_digest"],
        "auxiliary_initial_digest": EXPECTED_TRAINING_CONTRACT[
            "auxiliary_initial_digest"
        ],
        "auxiliary_final_digest": expected["auxiliary_final_digest"],
    }
    for key, value in exact.items():
        if extra.get(key) != value:
            raise RuntimeError(
                f"Stage-2G {arm} checkpoint contract drift at {key}"
            )
    if (
        payload.get("provenance", {}).get("encoder_state_sha256")
        != EXPECTED_ENCODER_SHA256
    ):
        raise RuntimeError(f"Stage-2G {arm} encoder provenance drift")
    block = train_report["arms"][arm]
    if (
        block["world_final_digest"] != extra["world_final_digest"]
        or block["auxiliary_final_digest"]
        != extra["auxiliary_final_digest"]
    ):
        raise RuntimeError(f"Stage-2G {arm} report/checkpoint drift")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2G DEV evaluation requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2G checkpoints and evaluator before DEV:\n"
            + "\n".join(dirty)
        )
    for path, expected in EXPECTED_STATIC_SHA256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"static artifact drift: {path}")

    train_report = json.loads(TRAIN_REPORT.read_text())
    if train_report["raw_sha256"] != sha256_file(TRAIN_RAW):
        raise RuntimeError("Stage-2G training raw/report mismatch")
    assert_training_contract(train_report)
    for arm in ("G-LA", "G-LRA"):
        path, expected_state = ARM_CHECKPOINTS[arm]
        block = train_report["arms"][arm]
        if block["checkpoint_sha256"] != sha256_file(path):
            raise RuntimeError(f"{arm} checkpoint hash drift")
        if block["world_final_digest"] != expected_state:
            raise RuntimeError(f"{arm} state digest drift")

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
        raise RuntimeError("Stage-2G natural target drift")
    if not np.array_equal(
        actual_continue,
        np.asarray(targets["continue_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2G continuation target drift")

    report = {
        "format": "stage2g_eval_report_v1",
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
        "training_contract_exact": True,
        "arms": {},
        "references_exact": {},
    }
    raw = {
        "format": "stage2g_eval_raw_v1",
        "targets": targets,
        "arms": {},
    }

    for arm, (path, expected_state) in ARM_CHECKPOINTS.items():
        world, payload = load_world_checkpoint(
            path,
            device,
            expect_config=sprint_candidate_config("gru"),
            expect_sha256=EXPECTED_STATIC_SHA256[path],
        )
        if arm in EXPECTED_ARM_TRAINING:
            assert_checkpoint_contract(arm, payload, train_report)
        enforce_frozen_encoder(world)
        world.eval()
        before = state_digest(world, exclude_heads=False)
        if before != expected_state:
            raise RuntimeError(f"{arm} state drift before DEV")

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
        if arm in ("A", "C-L", "C-LR"):
            assert_reference_exact(
                arm, arm_raw, stage2c["arms"][arm]
            )
            report["references_exact"][arm] = True

        raw["arms"][arm] = arm_raw
        report["arms"][arm] = {
            "checkpoint": str(path),
            "checkpoint_sha256": EXPECTED_STATIC_SHA256[path],
            "checkpoint_head": payload["provenance"]["head"],
            "encoder_state_sha256": payload["provenance"][
                "encoder_state_sha256"
            ],
            "checkpoint_contract_exact": arm in EXPECTED_ARM_TRAINING,
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

    if set(report["references_exact"]) != {"A", "C-L", "C-LR"}:
        raise RuntimeError("not all Stage-2C references reproduced")
    RAW_PATH.write_text(json.dumps(raw))
    report["raw_sha256"] = sha256_file(RAW_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2G DEV complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
