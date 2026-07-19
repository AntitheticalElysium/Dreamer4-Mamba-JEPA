"""Train the two sealed Stage-2G arms without importing evaluation code."""
from __future__ import annotations

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
from model import frozen_dynamics_recipe  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import build_fresh_world, build_schedule  # noqa: E402
from stage2g_preflight import OUTPUT as PREFLIGHT_PATH  # noqa: E402
from stage2g_relevance import (  # noqa: E402
    ARM_REWARD_WEIGHTS,
    FULL_UPDATES,
    PROTOCOL,
    build_auxiliary_contract,
    build_relevance_heads,
    module_state_digest,
    relevance_pools,
    train_relevance_world,
)
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
REPORT_PATH = ARTIFACTS / "stage2g_train_report.json"
RAW_PATH = ARTIFACTS / "stage2g_train_raw.json"
EXPECTED_PREFLIGHT_SHA256 = (
    "5551ead595a0d1ae71d4e479918176439e1a1405cbcdb11b07d9159919f5b97d"
)


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
        frozen_dynamics_recipe(),
        loss_histories=histories,
        extra=extra,
    )
    temporary.replace(path)
    if sha256_file(path) != digest:
        raise RuntimeError("checkpoint changed after atomic rename")
    return digest


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2G training requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit and pin the Stage-2G training runner first:\n"
            + "\n".join(dirty)
        )
    if sha256_file(PREFLIGHT_PATH) != EXPECTED_PREFLIGHT_SHA256:
        raise RuntimeError("Stage-2G preflight artifact drift")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight["status"] != "passed":
        raise RuntimeError("Stage-2G preflight did not pass")
    if not preflight["local_regression"]["exact"]:
        raise RuntimeError("Stage-2G local regression did not pass")
    if sha256_file(TRAIN_40K_CACHE) != preflight["replay"]["sha256"]:
        raise RuntimeError("Stage-2G replay drift")

    train, _ = load_scaled_data()
    base_schedule, base_schedule_sha256 = build_schedule(train)
    if base_schedule_sha256 != preflight["base_schedule_sha256"]:
        raise RuntimeError("Stage-2G base schedule drift")
    auxiliary_schedule, probe, contract = build_auxiliary_contract(
        relevance_pools(train), updates=FULL_UPDATES
    )
    for name in ("schedule_sha256", "probe_sha256"):
        if contract[name] != preflight["auxiliary_contract"][name]:
            raise RuntimeError(f"Stage-2G {name} drift")
    coefficient = preflight["gradient_registration"]["lambda_aux"]
    if coefficient != (
        0.10
        * preflight["gradient_registration"][
            "raw_generated_reward_rms"
        ]
        / preflight["gradient_registration"]["raw_auxiliary_rms"]
    ):
        raise RuntimeError("Stage-2G coefficient formula drift")

    initial_world = build_fresh_world(device)
    initial_heads = build_relevance_heads(
        initial_world.cfg.token_dim, device
    )
    world_initial_digest = state_digest(
        initial_world, exclude_heads=False
    )
    auxiliary_initial_digest = module_state_digest(initial_heads)
    if world_initial_digest != preflight["fresh_world_digest"]:
        raise RuntimeError("Stage-2G fresh world drift")
    if (
        auxiliary_initial_digest
        != preflight["auxiliary_initial_digest"]
    ):
        raise RuntimeError("Stage-2G auxiliary initialization drift")
    del initial_world, initial_heads
    torch.cuda.empty_cache()

    report = {
        "format": "stage2g_train_report_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "replay_sha256": preflight["replay"]["sha256"],
        "base_schedule_sha256": base_schedule_sha256,
        "auxiliary_schedule_sha256": contract["schedule_sha256"],
        "probe_sha256": contract["probe_sha256"],
        "lambda_aux": coefficient,
        "updates": FULL_UPDATES,
        "batch": 4,
        "world_initial_digest": world_initial_digest,
        "auxiliary_initial_digest": auxiliary_initial_digest,
        "arms": {},
    }
    raw = {
        "format": "stage2g_train_raw_v1",
        "arms": {},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    for arm, reward_weight in ARM_REWARD_WEIGHTS.items():
        world = build_fresh_world(device)
        heads = build_relevance_heads(world.cfg.token_dim, device)
        if (
            state_digest(world, exclude_heads=False)
            != world_initial_digest
        ):
            raise RuntimeError(f"{arm} world initialization drift")
        if module_state_digest(heads) != auxiliary_initial_digest:
            raise RuntimeError(f"{arm} auxiliary initialization drift")

        world, heads, info = train_relevance_world(
            world,
            heads,
            train,
            base_schedule,
            auxiliary_schedule,
            probe,
            arm=arm,
            lambda_aux=coefficient,
            updates=FULL_UPDATES,
            probe_updates=(0, FULL_UPDATES),
        )
        histories = info.pop("histories")
        auxiliary_state = {
            name: tensor.detach().cpu()
            for name, tensor in heads.state_dict().items()
        }
        checkpoint = ARTIFACTS / (
            f"stage2g_{arm.lower().replace('-', '')}_s505.pt"
        )
        checkpoint_sha256 = _atomic_checkpoint(
            checkpoint,
            world,
            histories,
            extra={
                "protocol": PROTOCOL,
                "arm": arm,
                "seed": 505,
                "updates": FULL_UPDATES,
                "generated_reward_weight": reward_weight,
                "lambda_aux": coefficient,
                "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
                "base_schedule_sha256": base_schedule_sha256,
                "auxiliary_schedule_sha256": contract[
                    "schedule_sha256"
                ],
                "probe_sha256": contract["probe_sha256"],
                "world_initial_digest": world_initial_digest,
                "world_final_digest": info["world_final_digest"],
                "auxiliary_initial_digest": auxiliary_initial_digest,
                "auxiliary_final_digest": info[
                    "auxiliary_final_digest"
                ],
                "auxiliary_state_dict": auxiliary_state,
            },
        )
        loaded_world, payload = load_world_checkpoint(
            checkpoint,
            device,
            expect_config=sprint_candidate_config("gru"),
            expect_sha256=checkpoint_sha256,
        )
        if (
            state_digest(loaded_world, exclude_heads=False)
            != info["world_final_digest"]
        ):
            raise RuntimeError(f"{arm} world checkpoint drift")
        loaded_heads = build_relevance_heads(
            loaded_world.cfg.token_dim, device
        )
        loaded_heads.load_state_dict(
            payload["extra"]["auxiliary_state_dict"], strict=True
        )
        if (
            module_state_digest(loaded_heads)
            != info["auxiliary_final_digest"]
        ):
            raise RuntimeError(f"{arm} auxiliary checkpoint drift")
        if payload["extra"]["generated_reward_weight"] != reward_weight:
            raise RuntimeError(f"{arm} reward-factor provenance drift")

        raw["arms"][arm] = {"histories": histories}
        RAW_PATH.write_text(json.dumps(raw))
        report["arms"][arm] = {
            "generated_reward_weight": reward_weight,
            **info,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "roundtrip_world_digest": state_digest(
                loaded_world, exclude_heads=False
            ),
            "roundtrip_auxiliary_digest": module_state_digest(
                loaded_heads
            ),
            "checkpoint_model_config": payload["model_config"],
        }
        report["raw_sha256"] = sha256_file(RAW_PATH)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(
            f"[{arm}] complete checkpoint={checkpoint_sha256}",
            flush=True,
        )
        del world, heads, loaded_world, loaded_heads
        torch.cuda.empty_cache()

    if set(report["arms"]) != set(ARM_REWARD_WEIGHTS):
        raise RuntimeError("Stage-2G training arms incomplete")
    report["raw_sha256"] = sha256_file(RAW_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2G training complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
