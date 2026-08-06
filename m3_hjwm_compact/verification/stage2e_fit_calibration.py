"""Fit and select Stage-2E categorical scalars on the CAL artifact only."""
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

from checkpoint import load_world_checkpoint, sprint_candidate_config  # noqa: E402
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import enforce_frozen_encoder  # noqa: E402
from phase_e_same_target import HORIZONS, target_rows, window_arrays  # noqa: E402
from stage1b_equal_update_analysis import reward_metrics  # noqa: E402
from stage2d_reward_head import selected_state_digest  # noqa: E402
from stage2e_calibration import (  # noqa: E402
    ARM_ORDER,
    calibration_nll,
    collect_same_target_logits,
    decode_calibrated,
    fit_calibrator,
    select_calibrator,
    tensor_sha256,
)
from step3_temporal import HELDOUT_20_CACHE  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
PROTOCOL = "reviews/2026-07-18-stage2e-categorical-calibration-protocol.md"
CHECKPOINT = ARTIFACTS / "stage2c_clr_s505.pt"
OUTPUT = ARTIFACTS / "stage2e_calibration_fit.json"

EXPECTED_CHECKPOINT_SHA256 = (
    "60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae"
)
EXPECTED_STATE_DIGEST = (
    "93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49"
)
EXPECTED_CAL_SHA256 = (
    "709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad"
)
EXPECTED_TARGETS = 3_262
EXPECTED_EVENTS = 140


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2E CAL collection requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2E implementation before CAL fitting:\n"
            + "\n".join(dirty)
        )
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("C-LR checkpoint drift")
    if sha256_file(HELDOUT_20_CACHE) != EXPECTED_CAL_SHA256:
        raise RuntimeError("CAL artifact drift")

    world, payload = load_world_checkpoint(
        CHECKPOINT,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    enforce_frozen_encoder(world)
    world.eval()
    before = selected_state_digest(world, reward=None)
    if before != EXPECTED_STATE_DIGEST:
        raise RuntimeError("C-LR full-state digest drift")

    episodes = torch.load(HELDOUT_20_CACHE, weights_only=False)
    rows = target_rows(episodes)
    arrays = window_arrays(episodes, rows)
    actual = np.asarray(arrays["rewards"], dtype=np.float32)
    events = int(np.sum(np.abs(actual) > 1e-6))
    if len(rows) != EXPECTED_TARGETS or events != EXPECTED_EVENTS:
        raise RuntimeError(
            f"CAL construction drift: rows={len(rows)}, events={events}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits_by_depth = collect_same_target_logits(
        world, arrays, device
    )
    after = selected_state_digest(world, reward=None)
    if after != before:
        raise RuntimeError("world state changed during CAL collection")

    logits = torch.cat([
        logits_by_depth[f"k{depth}"] for depth in HORIZONS
    ])
    repeated_rewards = torch.from_numpy(
        np.tile(actual, len(HORIZONS))
    )
    fits = {}
    for arm in ARM_ORDER:
        spec, optimization = fit_calibrator(
            arm,
            logits,
            repeated_rewards,
            low=world.cfg.reward_low,
            high=world.cfg.reward_high,
        )
        depth_metrics = {}
        for depth in HORIZONS:
            key = f"k{depth}"
            depth_logits = logits_by_depth[key]
            rewards = torch.from_numpy(actual)
            decoded = decode_calibrated(
                depth_logits,
                spec,
                low=world.cfg.reward_low,
                high=world.cfg.reward_high,
            ).numpy()
            nll = calibration_nll(
                depth_logits,
                rewards,
                spec,
                low=world.cfg.reward_low,
                high=world.cfg.reward_high,
            )
            depth_metrics[key] = {
                **reward_metrics(decoded, actual),
                "nll": float(nll.mean()),
            }
        fits[arm] = {
            "spec": spec.to_dict(),
            "optimization": optimization,
            "depth_metrics": depth_metrics,
        }
        print(
            f"[{arm}] T={spec.temperature:.8g} "
            f"zero_bias={spec.zero_bias:.8g} "
            f"NLL={optimization['final_nll']:.9f}",
            flush=True,
        )
    selected = select_calibrator(fits)
    elapsed = time.perf_counter() - started
    report = {
        "format": "stage2e_calibration_fit_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_head": payload["provenance"]["head"],
            "state_digest_before": before,
            "state_digest_after": after,
        },
        "calibration_data": {
            "path": str(HELDOUT_20_CACHE),
            "sha256": EXPECTED_CAL_SHA256,
            "episodes": len(episodes),
            "targets": len(rows),
            "events": events,
            "horizons": list(HORIZONS),
            "examples": len(repeated_rewards),
            "actual_reward_sha256": tensor_sha256(
                torch.from_numpy(actual)
            ),
            "logit_sha256": {
                key: tensor_sha256(value)
                for key, value in logits_by_depth.items()
            },
        },
        "fits": fits,
        "selection_rule": (
            "lowest finite CAL NLL; ties within 1e-10 use "
            "E-I,E-T,E-Z,E-TZ order"
        ),
        "selected_arm": selected,
        "wall_seconds": elapsed,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"selected {selected}; wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
