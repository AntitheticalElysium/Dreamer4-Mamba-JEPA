"""Equal-update real-state controls for Stage-1 head adaptation.

Protocol:
reviews/2026-07-18-stage1b-equal-update-control-protocol.md

R1/R2 match H1/H2's update budget, labels, schedules, optimizer, and
trainable parameters. Only the final two contexts are teacher-forced rather
than generated. H1/H2 are loaded from their committed head checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from fork_oracle_v2 import sha256_file  # noqa: E402
from model import assert_encoder_frozen  # noqa: E402
from phase_e_taskheads import ranking_metrics  # noqa: E402
import phase_e_continuation_depth as cont_depth  # noqa: E402
import phase_e_same_target as same_target  # noqa: E402
from stage1_head_adaptation import (  # noqa: E402
    ARTIFACTS,
    BATCH,
    BUNDLE,
    CHECKPOINTS,
    LR,
    MANIFEST,
    NATURAL,
    PREFIX,
    TERMINAL,
    UPDATES,
    WINDOW,
    freeze_world_except_heads,
    load_base,
    make_batch,
    window_index,
)
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402

PROTOCOL = REPO_ROOT / \
    "reviews/2026-07-18-stage1b-equal-update-control-protocol.md"
REPORT_PATH = ARTIFACTS / "stage1b_equal_update_report.json"
RAW_PATH = ARTIFACTS / "stage1b_equal_update_raw.json"
ARMS = ("R1", "R2")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(world, exclude_heads: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(world.state_dict().items()):
        if exclude_heads and (
            name.startswith("reward.") or name.startswith("continuation.")
        ):
            continue
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1)
            .view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def build_schedule(train, arm: str, seed: int) -> tuple[list[tuple[int, int]], str]:
    """Reproduce Stage 1's natural or event-focused schedule exactly."""
    if arm not in ("R1", "R2"):
        raise ValueError(arm)
    uniform, event = window_index(train)
    rng = np.random.default_rng(10_000 + seed)
    schedule = []
    for _ in range(UPDATES):
        if arm == "R2":
            half = BATCH // 2
            picks = (
                [uniform[int(rng.integers(len(uniform)))] for _ in range(half)]
                + [event[int(rng.integers(len(event)))] for _ in range(half)]
            )
        else:
            picks = [
                uniform[int(rng.integers(len(uniform)))]
                for _ in range(BATCH)
            ]
        schedule.extend(picks)
    array = np.asarray(schedule, dtype=np.int64)
    return schedule, hashlib.sha256(array.tobytes()).hexdigest()


def train_real_heads(world, arm: str, seed: int, train, device):
    """Train heads on nine teacher-forced post-transition contexts."""
    freeze_world_except_heads(world)
    named_trainable = [
        name for name, parameter in world.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for parameter in world.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    schedule, schedule_digest = build_schedule(train, arm, seed)
    before_nonhead = state_digest(world, exclude_heads=True)
    losses = []
    world.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for update in range(UPDATES):
        offset = update * BATCH
        batch = make_batch(
            train, schedule[offset:offset + BATCH], device)
        state = world.initial_state(BATCH, device)
        pooled = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for time_index in range(WINDOW):
                state = world.observe_step(
                    batch["obs"][:, time_index],
                    batch["previous_actions"][:, time_index],
                    state,
                )
                if time_index >= 1:
                    pooled.append(world.pool(state.tokens))
            pooled = torch.stack(pooled, dim=1)
            reward_logits = world.reward(pooled)
            continue_logits = world.continuation(pooled)
            loss = (
                world.reward.loss(
                    reward_logits, batch["rewards"]).mean()
                + F.binary_cross_entropy_with_logits(
                    continue_logits, batch["continues"])
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    elapsed = time.perf_counter() - started
    assert_encoder_frozen(world, optimizer)
    after_nonhead = state_digest(world, exclude_heads=True)
    if before_nonhead != after_nonhead:
        raise RuntimeError("non-head state changed in equal-update control")
    world.eval()
    return {
        "loss_first_last": [
            float(np.mean(losses[:100])),
            float(np.mean(losses[-100:])),
        ],
        "schedule_sha256": schedule_digest,
        "base_nonhead_sha256": before_nonhead,
        "after_nonhead_sha256": after_nonhead,
        "trainable_names": named_trainable,
        "optimizer": {
            "class": "AdamW",
            "lr": LR,
            "defaults": {
                key: value for key, value in optimizer.defaults.items()
                if isinstance(value, (str, int, float, bool, type(None), tuple))
            },
        },
        "wall_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def save_heads(path: Path, world, arm: str, base: str,
               metadata: dict) -> str:
    payload = {
        "format": "stage1b_equal_update_heads_v1",
        "reward": {
            key: value.detach().cpu()
            for key, value in world.reward.state_dict().items()
        },
        "continuation": {
            key: value.detach().cpu()
            for key, value in world.continuation.state_dict().items()
        },
        "arm": arm,
        "base": base,
        "metadata": metadata,
    }
    torch.save(payload, path)
    return sha256_file(path)


def load_adapted_heads(world, path: Path) -> None:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    world.reward.load_state_dict(payload["reward"], strict=True)
    world.continuation.load_state_dict(
        payload["continuation"], strict=True)


@torch.no_grad()
def evaluate_with_raw(world, natural_arrays, continuation_arrays,
                      actual_continue, anchors, device):
    reward = same_target.evaluate_world(world, natural_arrays, device)
    continuation = cont_depth.evaluate_world(
        world, continuation_arrays, actual_continue, device)
    ranking = ranking_metrics(world, anchors, device)
    metrics = {
        "reward_depth": reward["metrics"],
        "continuation_depth": continuation["metrics"],
        "ranking": {key: value for key, value in ranking.items()
                    if key != "rows"},
    }
    raw = {
        "reward_predictions": reward["predictions"],
        "continuation_predictions": continuation["predictions"],
        "ranking_rows": ranking["rows"],
    }
    return metrics, raw


def dirty_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def main() -> None:
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    paths = {"natural": NATURAL, "terminal": TERMINAL, "bundle": BUNDLE}
    for key, path in paths.items():
        if sha256_file(path) != manifest[key]["sha256"]:
            raise RuntimeError(f"{key} artifact drift")

    natural_episodes = torch.load(NATURAL, weights_only=False)
    terminal_episodes = torch.load(TERMINAL, weights_only=False)
    anchors = torch.load(BUNDLE, weights_only=False)
    natural_rows = same_target.target_rows(natural_episodes)
    natural_arrays = same_target.window_arrays(
        natural_episodes, natural_rows)
    continuation_rows = same_target.target_rows(terminal_episodes)
    continuation_arrays = same_target.window_arrays(
        terminal_episodes, continuation_rows)
    actual_continue = cont_depth.continuation_targets(
        terminal_episodes, continuation_rows)
    train, _ = load_scaled_data()

    provenance = {
        "protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL),
        "script_sha256": file_sha256(Path(__file__)),
        "head": git_head(),
        "source_digest": source_digest(),
        "dirty_status_at_start": dirty_status(),
        "versions": software_versions(),
        "data_sha256": {
            **{key: manifest[key]["sha256"] for key in paths},
            "replay": sha256_file(TRAIN_40K_CACHE),
        },
        "contract": {
            "updates": UPDATES,
            "batch": BATCH,
            "window": WINDOW,
            "prefix": PREFIX,
            "lr": LR,
            "arms": ARMS,
        },
    }
    report = {"provenance": provenance, "results": {}}
    raw = {"provenance": provenance, "results": {}}

    for kind, seed in CHECKPOINTS:
        base = f"{kind}_s{seed}"
        base_path = ARTIFACTS / f"xtopo_{kind}_s{seed}_16000.pt"
        for arm in ARMS:
            tag = f"{base}_{arm}"
            world = load_base(kind, seed, device)
            metadata = train_real_heads(
                world, arm, seed, train, device)
            metadata["base_checkpoint_sha256"] = sha256_file(base_path)
            head_path = ARTIFACTS / f"stage1b_heads_{tag}.pt"
            metadata["head_checkpoint_sha256"] = save_heads(
                head_path, world, arm, base, metadata)
            metrics, rows = evaluate_with_raw(
                world, natural_arrays, continuation_arrays,
                actual_continue, anchors, device)
            report["results"][tag] = {**metadata, **metrics}
            raw["results"][tag] = rows
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
            RAW_PATH.write_text(json.dumps(raw, default=str))
            print(
                tag,
                "rank",
                metrics["ranking"]["chosen_minus_random_mean"],
                "rwK8",
                metrics["reward_depth"]["k8"]["event_auroc"],
                "contK8",
                metrics["continuation_depth"]["k8"]["terminal_auroc"],
                flush=True,
            )
            del world
            torch.cuda.empty_cache()

        # Retain raw rows for the committed generated arms under the exact
        # same evaluator so paired factorial deltas can be recomputed without
        # another GPU run.
        for arm in ("H1", "H2"):
            tag = f"{base}_{arm}"
            world = load_base(kind, seed, device)
            load_adapted_heads(
                world, ARTIFACTS / f"stage1_heads_{tag}.pt")
            world.eval()
            metrics, rows = evaluate_with_raw(
                world, natural_arrays, continuation_arrays,
                actual_continue, anchors, device)
            report["results"][tag] = metrics
            raw["results"][tag] = rows
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
            RAW_PATH.write_text(json.dumps(raw, default=str))
            print(tag, "raw evaluation complete", flush=True)
            del world
            torch.cuda.empty_cache()

    raw_hash = sha256_file(RAW_PATH)
    report["provenance"]["raw_sha256"] = raw_hash
    report["provenance"]["dirty_status_at_end"] = dirty_status()
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("stage1b complete", raw_hash)


if __name__ == "__main__":
    main()
