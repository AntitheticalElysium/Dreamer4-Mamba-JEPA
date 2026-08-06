"""Training-only preflight for the registered Stage-2F operator arms."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

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
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import build_schedule  # noqa: E402
from stage2f_reward_operator import (  # noqa: E402
    LOCAL_FINGERPRINT,
    PROTOCOL,
    SMOKE_UPDATES,
    assert_same_initial_state,
    build_operator_world,
    train_world,
    validate_probe_limits,
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
OUTPUT = ARTIFACTS / "stage2f_preflight.json"
REFERENCE = ARTIFACTS / "stage2c_clr_s505.pt"
EXPECTED_REFERENCE_SHA256 = (
    "60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae"
)
EXPECTED_REFERENCE_STATE = (
    "93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49"
)
SOURCE_REPOS = {
    "dreamer_cdp": (
        REPO_ROOT / "third_party/sources/fmi-basel__Dreamer-CDP",
        "a851fa3e3d70b624b094ee1810ad4bb602346092",
    ),
    "drama": (
        REPO_ROOT / "third_party/sources/realwenlongwang__Drama",
        "a50bd54c34e77d1d13e988a031733a47817098e2",
    ),
    "dreamer4_jax_unofficial": (
        REPO_ROOT / "third_party/sources/edwhu__dreamer4-jax",
        "8144b940d801971f12ec5633553b95001e555949",
    ),
}
SOURCE_FILES = (
    "third_party/sources/fmi-basel__Dreamer-CDP/dreamerv3/configs.yaml",
    "third_party/sources/fmi-basel__Dreamer-CDP/embodied/jax/heads.py",
    "third_party/sources/fmi-basel__Dreamer-CDP/embodied/jax/outs.py",
    "third_party/sources/fmi-basel__Dreamer-CDP/embodied/jax/nets.py",
    "third_party/sources/realwenlongwang__Drama/"
    "sub_models/functions_losses.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commits() -> dict[str, str]:
    output = {}
    for name, (path, expected) in SOURCE_REPOS.items():
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(
                f"{name} source drift: {actual} != {expected}"
            )
        if subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout:
            raise RuntimeError(f"{name} source tree is dirty")
        output[name] = actual
    return output


def _compact_smoke(info: dict) -> dict:
    return {
        key: value
        for key, value in info.items()
        if key != "histories"
    } | {"histories": info["histories"]}


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2F preflight requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2F implementation before preflight:\n"
            + "\n".join(dirty)
        )
    source_commits = _source_commits()
    if sha256_file(REFERENCE) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("C-LR reference checkpoint drift")

    train, _ = load_scaled_data()
    schedule, schedule_digest = build_schedule(train)
    if schedule_digest != LOCAL_FINGERPRINT["schedule_sha256"]:
        raise RuntimeError("Stage-2F schedule drift")
    started = time.perf_counter()

    legacy, _ = load_world_checkpoint(
        REFERENCE,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=EXPECTED_REFERENCE_SHA256,
    )
    enforce_frozen_encoder(legacy)
    if legacy.cfg.reward_operator != "local_symlog":
        raise RuntimeError("legacy checkpoint did not normalize to local")
    if state_digest(legacy, exclude_heads=False) != EXPECTED_REFERENCE_STATE:
        raise RuntimeError("legacy C-LR state drift")
    del legacy
    torch.cuda.empty_cache()

    local = build_operator_world(
        device, "local_symlog", zero_output=False
    )
    local_initial = state_digest(local, exclude_heads=False)
    local, local_info = train_world(
        local,
        train,
        schedule,
        updates=SMOKE_UPDATES,
        progress_name="F-local-regression",
    )
    local_actual = {
        "schedule_sha256": schedule_digest,
        "init_digest": local_initial,
        "final_digest": local_info["final_digest"],
        "history_sha256": local_info["history_sha256"],
    }
    if local_actual != LOCAL_FINGERPRINT:
        raise RuntimeError(
            "historical local training fingerprint changed:\n"
            + json.dumps(
                {"expected": LOCAL_FINGERPRINT,
                 "actual": local_actual},
                indent=2,
            )
        )
    del local
    torch.cuda.empty_cache()

    initial_local = build_operator_world(
        device, "local_symlog", zero_output=True
    )
    initial_dreamer = build_operator_world(
        device, "dreamerv3_symexp", zero_output=True
    )
    assert_same_initial_state(initial_local, initial_dreamer)
    zero_initial_digest = state_digest(
        initial_local, exclude_heads=False
    )
    if zero_initial_digest != state_digest(
        initial_dreamer, exclude_heads=False
    ):
        raise RuntimeError("zero-initialized operator state digests differ")
    del initial_local, initial_dreamer
    torch.cuda.empty_cache()

    smokes = {}
    for arm, operator in (
        ("F-LZ", "local_symlog"),
        ("F-DZ", "dreamerv3_symexp"),
    ):
        world = build_operator_world(
            device, operator, zero_output=True
        )
        initial = state_digest(world, exclude_heads=False)
        if initial != zero_initial_digest:
            raise RuntimeError(f"{arm} zero-init digest drift")
        world, info = train_world(
            world,
            train,
            schedule,
            updates=SMOKE_UPDATES,
            progress_name=arm,
            probe_updates=(0, 1, 16, 64),
        )
        validate_probe_limits(info)
        smokes[arm] = {
            "operator": operator,
            "initial_digest": initial,
            **_compact_smoke(info),
        }
        del world
        torch.cuda.empty_cache()

    report = {
        "format": "stage2f_preflight_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "source_commits": source_commits,
        "source_file_sha256": {
            path: _sha(REPO_ROOT / path) for path in SOURCE_FILES
        },
        "replay": {
            "path": str(TRAIN_40K_CACHE),
            "sha256": sha256_file(TRAIN_40K_CACHE),
            "episodes": len(train),
        },
        "reference": {
            "path": str(REFERENCE),
            "sha256": EXPECTED_REFERENCE_SHA256,
            "state_digest": EXPECTED_REFERENCE_STATE,
            "legacy_operator": "local_symlog",
        },
        "schedule_sha256": schedule_digest,
        "local_regression": {
            "expected": LOCAL_FINGERPRINT,
            "actual": local_actual,
            "exact": True,
        },
        "zero_initial_digest": zero_initial_digest,
        "zero_initial_state_exact": True,
        "smokes": smokes,
        "wall_seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(
        f"Stage-2F preflight passed; wrote {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
