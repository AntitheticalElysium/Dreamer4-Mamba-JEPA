"""Stage-2D frozen-trunk reward-head state factorial.

Protocol:
reviews/2026-07-18-stage2d-separated-reward-head-protocol.md

Only ``world.reward`` may change. D-R and D-G use identical natural replay
windows and reward labels; the final two reward-head inputs are respectively
teacher-forced or generated.
"""
from __future__ import annotations

from contextlib import nullcontext
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
from model import assert_encoder_frozen, enforce_frozen_encoder  # noqa: E402
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from stage1_head_adaptation import make_batch  # noqa: E402
from stage2_evaluation import evaluate_arm  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
PROTOCOL = "reviews/2026-07-18-stage2d-separated-reward-head-protocol.md"
MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
BASE = ARTIFACTS / "stage2c_cl_s505.pt"
STAGE2C_REPORT = ARTIFACTS / "stage2c_report.json"
STAGE2C_RAW = ARTIFACTS / "stage2c_raw.json"
REPORT_PATH = ARTIFACTS / "stage2d_report.json"
RAW_PATH = ARTIFACTS / "stage2d_raw.json"

EXPECTED_MANIFEST_SHA256 = (
    "0b909b886e86bb221e9bd500da88bd38a7871c7e0534ccd159d2cf3c1b6c2bd4"
)
EXPECTED_BASE_SHA256 = (
    "227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1"
)
EXPECTED_BASE_FULL_DIGEST = (
    "a0cf4ec132a9e023ecf71fa63d7f1f8e17dd00d6080684f1e7b6962844b8c1c9"
)
EXPECTED_REWARD_DIGEST = (
    "091e08894efc407b7b8d1cd2b4af375adadf340c4f603c53ff3e23a9fa8ac7f3"
)
EXPECTED_NONREWARD_DIGEST = (
    "c44815c4236b748fb4f95d0f82a14671aa7ddbc462d6cd3c62cf388e5686c6c5"
)
EXPECTED_REPLAY_SHA256 = (
    "c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad"
)
EXPECTED_STAGE2C_REPORT_SHA256 = (
    "b73360a52bb137ef939a45c55f247fd0091011273fd6b1c1b8594201101706fc"
)
EXPECTED_STAGE2C_RAW_SHA256 = (
    "e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5"
)
EXPECTED_SCHEDULE_SHA256 = (
    "d8ed746758296f365282823eba8595751b407d616c96b93e8f8417904126fc4c"
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

ARMS = ("D-R", "D-G")
SEED = 505
SCHEDULE_SEED = 10_000 + SEED
UPDATES = 3_000
BATCH = 8
WINDOW = 10
PREFIX = 8
LR = 1e-3


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_state_digest(world, *, reward: bool | None = None) -> str:
    """Hash selected state-dict tensors with stable names and byte order.

    ``reward=True`` selects only ``reward.*``; ``False`` selects every other
    tensor; ``None`` selects the full state.
    """
    digest = hashlib.sha256()
    selected = 0
    for name, tensor in sorted(world.state_dict().items()):
        is_reward = name.startswith("reward.")
        if reward is not None and is_reward != reward:
            continue
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1)
            .view(torch.uint8).numpy().tobytes()
        )
        selected += 1
    if selected == 0:
        raise RuntimeError("state-digest selection was empty")
    return digest.hexdigest()


def window_index(train: list[dict]) -> list[tuple[int, int]]:
    """Every unpadded ten-observation window, with no event filtering."""
    output = []
    for episode_index, episode in enumerate(train):
        observations = len(episode["obs"])
        output.extend(
            (episode_index, start)
            for start in range(observations - WINDOW + 1)
        )
    if not output:
        raise RuntimeError("training replay has no eligible Stage-2D window")
    return output


def build_schedule(train: list[dict]) -> tuple[list[tuple[int, int]], str]:
    eligible = window_index(train)
    rng = np.random.default_rng(SCHEDULE_SEED)
    schedule = [
        eligible[int(rng.integers(len(eligible)))]
        for _ in range(UPDATES * BATCH)
    ]
    digest = hashlib.sha256(
        np.asarray(schedule, dtype=np.int64).tobytes()
    ).hexdigest()
    return schedule, digest


def context_index_contract(arm: str) -> list[dict[str, int | str]]:
    """Symbolic post-transition alignment used by tests and provenance."""
    if arm not in ARMS:
        raise ValueError(f"unknown Stage-2D arm {arm}")
    output = [
        {
            "kind": "real",
            "observation_index": time_index,
            "reward_index": time_index - 1,
        }
        for time_index in range(1, PREFIX)
    ]
    if arm == "D-R":
        output.extend(
            {
                "kind": "real",
                "observation_index": time_index,
                "reward_index": time_index - 1,
            }
            for time_index in range(PREFIX, WINDOW)
        )
    else:
        output.extend(
            {
                "kind": "generated",
                "action_index": PREFIX - 1 + generated_index,
                "reward_index": PREFIX - 1 + generated_index,
                "depth": generated_index + 1,
            }
            for generated_index in range(2)
        )
    return output


def reward_contexts(world, batch: dict[str, torch.Tensor], arm: str):
    """Return nine aligned frozen contexts for reward-head fitting."""
    if arm not in ARMS:
        raise ValueError(f"unknown Stage-2D arm {arm}")
    batch_size = batch["obs"].shape[0]
    device = batch["obs"].device
    state = world.initial_state(batch_size, device)
    contexts = []

    for time_index in range(PREFIX):
        state = world.observe_step(
            batch["obs"][:, time_index],
            batch["previous_actions"][:, time_index],
            state,
        )
        if time_index >= 1:
            contexts.append(world.pool(state.tokens))

    if arm == "D-R":
        for time_index in range(PREFIX, WINDOW):
            state = world.observe_step(
                batch["obs"][:, time_index],
                batch["previous_actions"][:, time_index],
                state,
            )
            contexts.append(world.pool(state.tokens))
    else:
        for generated_index in range(2):
            action_index = PREFIX - 1 + generated_index
            state, _, _, _ = world.imagine_step(
                state,
                batch["actions"][:, action_index],
                deterministic_mode=True,
            )
            contexts.append(world.pool(state.tokens))

    if len(contexts) != WINDOW - 1:
        raise RuntimeError(
            f"{arm} produced {len(contexts)} contexts, expected {WINDOW - 1}"
        )
    return torch.stack(contexts, dim=1)


def freeze_reward_only(world):
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    for parameter in world.reward.parameters():
        parameter.requires_grad_(True)
    enforce_frozen_encoder(world)
    world.eval()
    world.reward.train()
    names = [
        name for name, parameter in world.named_parameters()
        if parameter.requires_grad
    ]
    if len(names) != 6 or any(
        not name.startswith("reward.") for name in names
    ):
        raise RuntimeError(f"reward-only trainable contract violated: {names}")
    return names


def load_registered_base(device: torch.device):
    if sha256_file(BASE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("C-L base checkpoint hash drift")
    world, payload = load_world_checkpoint(
        BASE,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=EXPECTED_BASE_SHA256,
    )
    if payload["extra"].get("arm") != "C-L":
        raise RuntimeError("registered base is not the C-L arm")
    observed = {
        "full": selected_state_digest(world, reward=None),
        "reward": selected_state_digest(world, reward=True),
        "nonreward": selected_state_digest(world, reward=False),
    }
    expected = {
        "full": EXPECTED_BASE_FULL_DIGEST,
        "reward": EXPECTED_REWARD_DIGEST,
        "nonreward": EXPECTED_NONREWARD_DIGEST,
    }
    if observed != expected:
        raise RuntimeError(
            f"registered C-L state digest drift: {observed} != {expected}"
        )
    enforce_frozen_encoder(world)
    return world.eval(), payload


def training_distribution(
    train: list[dict],
    schedule: list[tuple[int, int]],
) -> dict:
    all_rewards, final_rewards = [], []
    for episode_index, start in schedule:
        rewards = np.asarray(
            train[episode_index]["rewards"][start:start + WINDOW - 1],
            dtype=np.float32,
        )
        all_rewards.extend(rewards.tolist())
        final_rewards.extend(rewards[-2:].tolist())
    all_rewards = np.asarray(all_rewards)
    final_rewards = np.asarray(final_rewards)
    return {
        "sampled_windows": len(schedule),
        "unique_windows": len(set(schedule)),
        "labels_per_window": WINDOW - 1,
        "all_label_event_fraction": float(
            np.mean(np.abs(all_rewards) > 1e-6)
        ),
        "final2_label_event_fraction": float(
            np.mean(np.abs(final_rewards) > 1e-6)
        ),
    }


def _atomic_head_checkpoint(
    path: Path,
    world,
    arm: str,
    metadata: dict,
) -> str:
    payload = {
        "format": "stage2d_reward_head_v1",
        "arm": arm,
        "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "reward": {
            name: tensor.detach().cpu()
            for name, tensor in world.reward.state_dict().items()
        },
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def train_arm(
    arm: str,
    train: list[dict],
    schedule: list[tuple[int, int]],
    schedule_digest: str,
    device: torch.device,
) -> dict:
    if schedule_digest != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Stage-2D schedule drift")
    world, _ = load_registered_base(device)
    trainable_names = freeze_reward_only(world)
    trainable = [
        parameter for parameter in world.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    before = {
        "reward": selected_state_digest(world, reward=True),
        "nonreward": selected_state_digest(world, reward=False),
    }
    losses = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update in range(UPDATES):
        offset = update * BATCH
        batch = make_batch(
            train, schedule[offset:offset + BATCH], device
        )
        with torch.no_grad(), _autocast(device):
            contexts = reward_contexts(world, batch, arm)
        with _autocast(device):
            logits = world.reward(contexts.detach())
            loss = world.reward.loss(
                logits, batch["rewards"]
            ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in world.named_parameters():
            if (
                not name.startswith("reward.")
                and parameter.grad is not None
            ):
                raise RuntimeError(f"non-reward gradient reached {name}")
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        losses.append(float(loss.detach()))
        if update % 1_000 == 999:
            print(
                f"[{arm}] {update + 1}: "
                f"reward_nll={np.mean(losses[-200:]):.6f}",
                flush=True,
            )

    assert_encoder_frozen(world, optimizer)
    after = {
        "reward": selected_state_digest(world, reward=True),
        "nonreward": selected_state_digest(world, reward=False),
    }
    if before["nonreward"] != EXPECTED_NONREWARD_DIGEST:
        raise RuntimeError("base non-reward digest differs from registration")
    if after["nonreward"] != before["nonreward"]:
        raise RuntimeError("non-reward state changed during Stage-2D")
    if after["reward"] == before["reward"]:
        raise RuntimeError("reward head did not change during Stage-2D")

    elapsed = time.perf_counter() - started
    metadata = {
        "arm": arm,
        "updates": UPDATES,
        "batch": BATCH,
        "lr": LR,
        "schedule_sha256": schedule_digest,
        "context_contract": context_index_contract(arm),
        "trainable_names": trainable_names,
        "state_digest_before": before,
        "state_digest_after": after,
        "loss_first100": float(np.mean(losses[:100])),
        "loss_last100": float(np.mean(losses[-100:])),
        "wall_seconds": elapsed,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20
            if device.type == "cuda" else 0.0
        ),
        "optimizer": {
            "class": type(optimizer).__name__,
            "defaults": {
                key: value
                for key, value in optimizer.defaults.items()
                if isinstance(
                    value,
                    (str, int, float, bool, type(None), tuple),
                )
            },
        },
    }
    path = ARTIFACTS / f"stage2d_{arm.lower().replace('-', '')}_s{SEED}.pt"
    digest = _atomic_head_checkpoint(path, world, arm, metadata)
    metadata["checkpoint"] = str(path)
    metadata["checkpoint_sha256"] = digest
    del world, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def load_adapted_arm(
    path: Path,
    expected_sha256: str,
    device: torch.device,
):
    world, _ = load_registered_base(device)
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("Stage-2D head checkpoint hash drift")
    if payload.get("format") != "stage2d_reward_head_v1":
        raise RuntimeError("invalid Stage-2D head checkpoint format")
    if payload.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("Stage-2D head checkpoint base drift")
    world.reward.load_state_dict(payload["reward"], strict=True)
    if (
        selected_state_digest(world, reward=False)
        != EXPECTED_NONREWARD_DIGEST
    ):
        raise RuntimeError("adapted arm changed non-reward state")
    expected_reward = payload["metadata"]["state_digest_after"]["reward"]
    if selected_state_digest(world, reward=True) != expected_reward:
        raise RuntimeError("adapted reward-head digest drift")
    enforce_frozen_encoder(world)
    return world.eval()


def dev_contract(manifest: dict) -> dict:
    """Return only the spent DEV tier; kept separate for a FINAL-access test."""
    return manifest["dev"]


def assert_frozen_outputs_identical(
    candidate_raw: dict,
    base_raw: dict,
) -> dict:
    checked = {}
    for domain in ("continuation_predictions", "latent_errors"):
        checked[domain] = {}
        for depth, base_values in base_raw[domain].items():
            candidate = np.asarray(candidate_raw[domain][depth])
            baseline = np.asarray(base_values)
            equal = bool(np.array_equal(candidate, baseline))
            checked[domain][depth] = equal
            if not equal:
                maximum = float(np.max(np.abs(candidate - baseline)))
                raise RuntimeError(
                    f"frozen {domain}/{depth} changed; max delta {maximum}"
                )
    return checked


def _validate_static_inputs() -> tuple[dict, dict]:
    paths = {
        MANIFEST: EXPECTED_MANIFEST_SHA256,
        STAGE2C_REPORT: EXPECTED_STAGE2C_REPORT_SHA256,
        STAGE2C_RAW: EXPECTED_STAGE2C_RAW_SHA256,
        TRAIN_40K_CACHE: EXPECTED_REPLAY_SHA256,
    }
    for path, expected in paths.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"static artifact drift for {path}: {actual} != {expected}"
            )
    manifest = json.loads(MANIFEST.read_text())
    dev = dev_contract(manifest)
    for key, expected in EXPECTED_DEV_SHA256.items():
        if dev[key]["sha256"] != expected:
            raise RuntimeError(f"manifest DEV hash drift for {key}")
        if sha256_file(Path(dev[key]["path"])) != expected:
            raise RuntimeError(f"DEV artifact drift for {key}")
    return dev, json.loads(STAGE2C_RAW.read_text())


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2D fitting requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2D implementation before fitting:\n"
            + "\n".join(dirty)
        )

    dev, stage2c_raw = _validate_static_inputs()
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

    targets = stage2c_raw["targets"]
    if not np.array_equal(
        natural_arrays["rewards"],
        np.asarray(targets["reward_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2D reward targets differ from Stage-2C")
    if not np.array_equal(
        actual_continue,
        np.asarray(targets["continue_actual"], dtype=np.float32),
    ):
        raise RuntimeError("Stage-2D continuation targets differ from Stage-2C")

    train, _ = load_scaled_data()
    schedule, schedule_digest = build_schedule(train)
    if schedule_digest != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            f"schedule drift: {schedule_digest} != "
            f"{EXPECTED_SCHEDULE_SHA256}"
        )
    distribution = training_distribution(train, schedule)
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "fixed_contract": {
            "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
            "base_full_digest": EXPECTED_BASE_FULL_DIGEST,
            "base_reward_digest": EXPECTED_REWARD_DIGEST,
            "base_nonreward_digest": EXPECTED_NONREWARD_DIGEST,
            "schedule_sha256": schedule_digest,
            "replay_sha256": EXPECTED_REPLAY_SHA256,
            "dev_sha256": EXPECTED_DEV_SHA256,
            "stage2c_report_sha256": EXPECTED_STAGE2C_REPORT_SHA256,
            "stage2c_raw_sha256": EXPECTED_STAGE2C_RAW_SHA256,
        },
        "training_distribution": distribution,
        "arms": {},
        "evaluation": {},
        "isolation": {},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    for arm in ARMS:
        report["arms"][arm] = train_arm(
            arm, train, schedule, schedule_digest, device
        )
        REPORT_PATH.write_text(json.dumps(report, indent=2))

    raw = {
        "targets": targets,
        "arms": {
            "A": stage2c_raw["arms"]["A"],
            "C-L": stage2c_raw["arms"]["C-L"],
        },
    }
    for arm in ARMS:
        info = report["arms"][arm]
        world = load_adapted_arm(
            Path(info["checkpoint"]),
            info["checkpoint_sha256"],
            device,
        )
        point, arm_raw = evaluate_arm(
            world,
            natural_arrays,
            terminal_arrays,
            actual_continue,
            anchors,
            device,
        )
        identity = assert_frozen_outputs_identical(
            arm_raw, stage2c_raw["arms"]["C-L"]
        )
        report["evaluation"][arm] = point
        report["isolation"][arm] = {
            "nonreward_digest_unchanged": (
                info["state_digest_after"]["nonreward"]
                == EXPECTED_NONREWARD_DIGEST
            ),
            "raw_identity": identity,
        }
        raw["arms"][arm] = arm_raw
        print(
            f"[eval {arm}] "
            f"reward-k8={point['reward_depth']['k8']['event_auroc']:.4f} "
            f"pearson-k8={point['reward_depth']['k8']['reward_pearson']:.4f} "
            f"rank={point['ranking']['chosen_minus_random_mean']:.4f}",
            flush=True,
        )
        del world
        torch.cuda.empty_cache()

    RAW_PATH.write_text(json.dumps(raw))
    report["hashes"] = {
        "raw": sha256_file(RAW_PATH),
        "head_checkpoints": {
            arm: report["arms"][arm]["checkpoint_sha256"]
            for arm in ARMS
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2D complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
