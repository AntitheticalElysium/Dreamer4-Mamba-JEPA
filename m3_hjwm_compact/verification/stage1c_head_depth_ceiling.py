"""Registered Stage-1c head-only K2 versus per-step K8 depth diagnostic."""
from __future__ import annotations

import hashlib
import json
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
from stage1_head_adaptation import (  # noqa: E402
    ARTIFACTS,
    BATCH,
    BUNDLE,
    LR,
    MANIFEST,
    NATURAL,
    TERMINAL,
    UPDATES,
    freeze_world_except_heads,
    load_base,
)
from stage1b_equal_update_control import (  # noqa: E402
    dirty_status,
    evaluate_with_raw,
    state_digest,
)
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402
import phase_e_continuation_depth as cont_depth  # noqa: E402
import phase_e_same_target as same_target  # noqa: E402

PROTOCOL = REPO_ROOT / \
    "reviews/2026-07-18-stage1c-head-depth-ceiling-protocol.md"
STAGE1B_DEPENDENCY = COMPACT_ROOT / \
    "verification/stage1b_equal_update_control.py"
REPORT_PATH = ARTIFACTS / "stage1c_head_depth_report.json"
RAW_PATH = ARTIFACTS / "stage1c_head_depth_raw.json"
SEED = 505
KINDS = ("X-FLM", "X-FLG")
DEPTHS = {"D2": 2, "D8": 8}
PREFIX = 8
WINDOW = PREFIX + max(DEPTHS.values())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_index(train: list[dict]) -> list[tuple[int, int]]:
    output = []
    for episode_index, episode in enumerate(train):
        for start in range(len(episode["obs"]) - WINDOW + 1):
            output.append((episode_index, start))
    return output


def build_schedule(
    train: list[dict],
    seed: int = SEED,
) -> tuple[list[tuple[int, int]], str]:
    uniform = window_index(train)
    rng = np.random.default_rng(20_000 + seed)
    schedule = [
        uniform[int(rng.integers(len(uniform)))]
        for _ in range(UPDATES * BATCH)
    ]
    array = np.asarray(schedule, dtype=np.int64)
    return schedule, hashlib.sha256(array.tobytes()).hexdigest()


def make_batch(train, picks, device) -> dict[str, torch.Tensor]:
    observations, actions, rewards, continues, previous = [], [], [], [], []
    for episode_index, start in picks:
        episode = train[episode_index]
        observations.append(
            episode["obs"][start:start + PREFIX])
        actions.append(
            episode["actions"][start:start + WINDOW - 1])
        rewards.append(
            episode["rewards"][start:start + WINDOW - 1])
        continues.append(
            episode["continues"][start:start + WINDOW - 1])
        item_previous = np.full(PREFIX, -1, dtype=np.int64)
        if start:
            item_previous[0] = episode["actions"][start - 1]
        item_previous[1:] = episode["actions"][
            start:start + PREFIX - 1]
        previous.append(item_previous)

    def to(items, dtype):
        return torch.from_numpy(np.stack(items)).to(
            device=device, dtype=dtype)

    return {
        "obs": to(observations, torch.uint8),
        "actions": to(actions, torch.int64),
        "rewards": to(rewards, torch.float32),
        "continues": to(continues, torch.float32),
        "previous_actions": to(previous, torch.int64),
    }


def depth_contexts_and_targets(world, batch, depth: int):
    state = world.initial_state(BATCH, batch["obs"].device)
    pooled, reward_targets, continue_targets = [], [], []
    for time_index in range(PREFIX):
        state = world.observe_step(
            batch["obs"][:, time_index],
            batch["previous_actions"][:, time_index],
            state,
        )
        if time_index >= 1:
            pooled.append(world.pool(state.tokens))
            reward_targets.append(batch["rewards"][:, time_index - 1])
            continue_targets.append(
                batch["continues"][:, time_index - 1])
    for generated_index in range(depth):
        action_index = PREFIX - 1 + generated_index
        state, _, _, _ = world.imagine_step(
            state,
            batch["actions"][:, action_index],
            deterministic_mode=True,
        )
        pooled.append(world.pool(state.tokens))
        reward_targets.append(batch["rewards"][:, action_index])
        continue_targets.append(batch["continues"][:, action_index])
    return (
        torch.stack(pooled, dim=1),
        torch.stack(reward_targets, dim=1),
        torch.stack(continue_targets, dim=1),
    )


def train_heads(world, depth: int, schedule, train, device) -> dict:
    freeze_world_except_heads(world)
    named_trainable = [
        name for name, parameter in world.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [
        parameter for parameter in world.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled, reward_targets, continue_targets = \
                depth_contexts_and_targets(world, batch, depth)
            reward_logits = world.reward(pooled)
            continue_logits = world.continuation(pooled)
            loss = (
                world.reward.loss(
                    reward_logits, reward_targets).mean()
                + F.binary_cross_entropy_with_logits(
                    continue_logits, continue_targets)
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
        raise RuntimeError("non-head state changed in depth diagnostic")
    world.eval()
    return {
        "loss_first_last": [
            float(np.mean(losses[:100])),
            float(np.mean(losses[-100:])),
        ],
        "base_nonhead_sha256": before_nonhead,
        "after_nonhead_sha256": after_nonhead,
        "trainable_names": named_trainable,
        "optimizer": {
            "class": "AdamW",
            "lr": LR,
            "defaults": {
                key: value
                for key, value in optimizer.defaults.items()
                if isinstance(
                    value,
                    (str, int, float, bool, type(None), tuple),
                )
            },
        },
        "wall_seconds": elapsed,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20),
    }


def save_heads(path: Path, world, arm: str, base: str, metadata: dict) -> str:
    payload = {
        "format": "stage1c_head_depth_v1",
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


def main() -> None:
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    for key, path in (
        ("natural", NATURAL),
        ("terminal", TERMINAL),
        ("bundle", BUNDLE),
    ):
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
    schedule, schedule_digest = build_schedule(train)

    provenance = {
        "protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL),
        "script_sha256": file_sha256(Path(__file__)),
        "untracked_dependency_sha256": {
            str(STAGE1B_DEPENDENCY.relative_to(REPO_ROOT)):
                file_sha256(STAGE1B_DEPENDENCY),
        },
        "head": git_head(),
        "source_digest": source_digest(),
        "dirty_status_at_start": dirty_status(),
        "versions": software_versions(),
        "data_sha256": {
            "natural": manifest["natural"]["sha256"],
            "terminal": manifest["terminal"]["sha256"],
            "bundle": manifest["bundle"]["sha256"],
            "replay": sha256_file(TRAIN_40K_CACHE),
        },
        "contract": {
            "seed": SEED,
            "kinds": KINDS,
            "depths": DEPTHS,
            "updates": UPDATES,
            "batch": BATCH,
            "window": WINDOW,
            "prefix": PREFIX,
            "lr": LR,
            "schedule_sha256": schedule_digest,
            "eligible_windows": len(window_index(train)),
        },
    }
    report = {"provenance": provenance, "results": {}}
    raw = {"provenance": provenance, "results": {}}

    for kind in KINDS:
        base = f"{kind}_s{SEED}"
        base_path = ARTIFACTS / f"xtopo_{kind}_s{SEED}_16000.pt"
        for arm, depth in DEPTHS.items():
            tag = f"{base}_{arm}"
            world = load_base(kind, SEED, device)
            metadata = train_heads(
                world, depth, schedule, train, device)
            metadata["depth"] = depth
            metadata["schedule_sha256"] = schedule_digest
            metadata["base_checkpoint_sha256"] = sha256_file(
                base_path)
            head_path = ARTIFACTS / f"stage1c_heads_{tag}.pt"
            metadata["head_checkpoint_sha256"] = save_heads(
                head_path, world, arm, base, metadata)
            metrics, rows = evaluate_with_raw(
                world,
                natural_arrays,
                continuation_arrays,
                actual_continue,
                anchors,
                device,
            )
            report["results"][tag] = {**metadata, **metrics}
            raw["results"][tag] = rows
            REPORT_PATH.write_text(
                json.dumps(report, indent=2, default=str))
            RAW_PATH.write_text(json.dumps(raw, default=str))
            print(
                tag,
                "rank",
                metrics["ranking"]["chosen_minus_random_mean"],
                "rwK8",
                metrics["reward_depth"]["k8"]["event_auroc"],
                "contK8",
                metrics["continuation_depth"]["k8"][
                    "terminal_auroc"],
                flush=True,
            )
            del world
            torch.cuda.empty_cache()

    report["provenance"]["raw_sha256"] = sha256_file(RAW_PATH)
    report["provenance"]["dirty_status_at_end"] = dirty_status()
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str))
    print("stage1c complete", report["provenance"]["raw_sha256"])


if __name__ == "__main__":
    main()
