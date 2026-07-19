"""Train Stage-2F F-LZ/F-DZ without importing any evaluation artifact."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import (  # noqa: E402
    load_world_checkpoint,
    save_world_checkpoint,
    sprint_candidate_config,
)
from fork_oracle_v2 import sha256_file  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import build_schedule  # noqa: E402
from stage2f_preflight import OUTPUT as PREFLIGHT_PATH  # noqa: E402
from stage2f_reward_operator import (  # noqa: E402
    FULL_UPDATES,
    LOCAL_FINGERPRINT,
    PROTOCOL,
    assert_same_initial_state,
    build_operator_world,
    train_world,
)
from step3_temporal import (  # noqa: E402
    TRAIN_40K_CACHE,
    load_scaled_data,
)
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
REPORT_PATH = ARTIFACTS / "stage2f_train_report.json"
RAW_PATH = ARTIFACTS / "stage2f_train_raw.json"
EXPECTED_PREFLIGHT_SHA256 = "__PIN_AFTER_PREFLIGHT_COMMIT__"
ARMS = {
    "F-LZ": "local_symlog",
    "F-DZ": "dreamerv3_symexp",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_checkpoint(
    path: Path,
    world,
    histories: dict,
    *,
    extra: dict,
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = save_world_checkpoint(
        temporary,
        world,
        world_loss_config(),
        loss_histories=histories,
        extra=extra,
    )
    temporary.replace(path)
    if sha256_file(path) != digest:
        raise RuntimeError("checkpoint digest changed after atomic rename")
    return digest


def world_loss_config():
    from model import frozen_dynamics_recipe
    return frozen_dynamics_recipe()


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2F training requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit preflight artifact and hash pin before training:\n"
            + "\n".join(dirty)
        )
    if EXPECTED_PREFLIGHT_SHA256.startswith("__"):
        raise RuntimeError("pin committed Stage-2F preflight before training")
    if sha256_file(PREFLIGHT_PATH) != EXPECTED_PREFLIGHT_SHA256:
        raise RuntimeError("Stage-2F preflight artifact drift")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if not preflight["local_regression"]["exact"]:
        raise RuntimeError("Stage-2F local regression did not pass")

    train, _ = load_scaled_data()
    if sha256_file(TRAIN_40K_CACHE) != preflight["replay"]["sha256"]:
        raise RuntimeError("training replay drift")
    schedule, schedule_digest = build_schedule(train)
    if schedule_digest != LOCAL_FINGERPRINT["schedule_sha256"]:
        raise RuntimeError("training schedule drift")

    initial_worlds = {
        arm: build_operator_world(
            device, operator, zero_output=True
        )
        for arm, operator in ARMS.items()
    }
    assert_same_initial_state(
        initial_worlds["F-LZ"], initial_worlds["F-DZ"]
    )
    initial_digest = state_digest(
        initial_worlds["F-LZ"], exclude_heads=False
    )
    if initial_digest != preflight["zero_initial_digest"]:
        raise RuntimeError("full-run zero-init digest differs from preflight")
    del initial_worlds
    torch.cuda.empty_cache()

    report = {
        "format": "stage2f_train_report_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "replay_sha256": preflight["replay"]["sha256"],
        "schedule_sha256": schedule_digest,
        "updates": FULL_UPDATES,
        "batch": 4,
        "zero_initial_digest": initial_digest,
        "arms": {},
    }
    raw = {
        "format": "stage2f_train_raw_v1",
        "arms": {},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    for arm, operator in ARMS.items():
        world = build_operator_world(
            device, operator, zero_output=True
        )
        arm_initial = state_digest(world, exclude_heads=False)
        if arm_initial != initial_digest:
            raise RuntimeError(f"{arm} initial digest drift")
        world, info = train_world(
            world,
            train,
            schedule,
            updates=FULL_UPDATES,
            progress_name=arm,
        )
        checkpoint = ARTIFACTS / (
            f"stage2f_{arm.lower().replace('-', '')}_s505.pt"
        )
        histories = info.pop("histories")
        checkpoint_sha256 = _atomic_checkpoint(
            checkpoint,
            world,
            histories,
            extra={
                "protocol": PROTOCOL,
                "arm": arm,
                "operator": operator,
                "zero_reward_output": True,
                "seed": 505,
                "updates": FULL_UPDATES,
                "initial_digest": arm_initial,
                "final_digest": info["final_digest"],
                "schedule_sha256": schedule_digest,
                "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
            },
        )
        expected_config = replace(
            sprint_candidate_config("gru"),
            reward_operator=operator,
        )
        loaded, payload = load_world_checkpoint(
            checkpoint,
            device,
            expect_config=expected_config,
            expect_sha256=checkpoint_sha256,
        )
        loaded_digest = state_digest(loaded, exclude_heads=False)
        if loaded_digest != info["final_digest"]:
            raise RuntimeError(f"{arm} checkpoint state round-trip drift")
        if payload["extra"]["operator"] != operator:
            raise RuntimeError(f"{arm} checkpoint operator provenance drift")

        raw["arms"][arm] = {"histories": histories}
        RAW_PATH.write_text(json.dumps(raw))
        report["arms"][arm] = {
            "operator": operator,
            "zero_reward_output": True,
            "initial_digest": arm_initial,
            **info,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "roundtrip_state_digest": loaded_digest,
            "checkpoint_model_config": payload["model_config"],
        }
        report["raw_sha256"] = sha256_file(RAW_PATH)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(
            f"[{arm}] complete checkpoint={checkpoint_sha256}",
            flush=True,
        )
        del world, loaded
        torch.cuda.empty_cache()

    if set(report["arms"]) != set(ARMS):
        raise RuntimeError("Stage-2F training arms incomplete")
    report["raw_sha256"] = sha256_file(RAW_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2F training complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
