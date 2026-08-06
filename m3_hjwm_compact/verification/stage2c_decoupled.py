"""Stage-2C uniform generated-latent/reward factorial.

Protocol: reviews/2026-07-18-stage2c-decoupled-protocol.md

This runner trains only the two pre-registered GRU-505 arms. It reuses the
hash-pinned Stage-2 Arm-A checkpoint as the baseline and never opens the FINAL
evaluation tier.
"""
from __future__ import annotations

import dataclasses
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
    save_world_checkpoint,
    sprint_candidate_config,
)
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import (  # noqa: E402
    assert_encoder_frozen,
    enforce_frozen_encoder,
    frozen_dynamics_recipe,
)
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import (  # noqa: E402
    BATCH,
    PREFIX,
    SEED,
    UPDATES,
    build_fresh_world,
    build_schedule,
    make_batch,
)
from stage2_evaluation import evaluate_arm  # noqa: E402
from stage2_objectives import (  # noqa: E402
    GeneratedLossWeights,
    generated_step_components,
    weighted_generated_loss,
)
from step3_temporal import load_scaled_data  # noqa: E402
from step4_runner import (  # noqa: E402
    git_head,
    software_versions,
    source_digest,
    tracked_dirty,
)


ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
PROTOCOL = "reviews/2026-07-18-stage2c-decoupled-protocol.md"
MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
STAGE2_REPORT = ARTIFACTS / "stage2_ab_report.json"
BASELINE = ARTIFACTS / "stage2_armA_s505.pt"
REPORT_PATH = ARTIFACTS / "stage2c_report.json"
RAW_PATH = ARTIFACTS / "stage2c_raw.json"

EXPECTED_BASELINE_SHA256 = (
    "fcbc9407a36faf59e32ec1425c2fbee7a5e5a21ea73cb13170a828e4e9c6d1f2"
)
EXPECTED_INIT_DIGEST = (
    "55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260"
)
EXPECTED_SCHEDULE_SHA256 = (
    "427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0"
)
GENERATED_STEPS = 2
GENERATED_REWARD_WEIGHT = 0.10
ARM_SPECS = {
    "C-L": GeneratedLossWeights(
        latent=1.0, reward=0.0, continuation=0.0
    ),
    "C-LR": GeneratedLossWeights(
        latent=1.0,
        reward=GENERATED_REWARD_WEIGHT,
        continuation=0.0,
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_distribution(
    train: list[dict],
    schedule: list[tuple[int, int]],
) -> dict:
    """Audit exactly the labels seen by the generated K1/K2 objective."""
    rewards, continues = [], []
    for episode, start in schedule:
        transition = start + PREFIX - 1
        rewards.extend(
            train[episode]["rewards"][
                transition:transition + GENERATED_STEPS
            ].tolist()
        )
        continues.extend(
            train[episode]["continues"][
                transition:transition + GENERATED_STEPS
            ].tolist()
        )
    rewards = np.asarray(rewards, dtype=np.float32)
    continues = np.asarray(continues, dtype=np.float32)
    output = {
        "windows": len(schedule),
        "unique_windows": len(set(schedule)),
        "generated_labels": len(rewards),
        "event_fraction": float(np.mean(np.abs(rewards) > 1e-6)),
        "terminal_fraction": float(np.mean(continues < 0.5)),
        "mean_reward": float(rewards.mean()),
        "minimum_continue": float(continues.min()),
    }
    if output["terminal_fraction"] != 0.0:
        raise RuntimeError(
            "uniform unpadded schedule unexpectedly contains a generated "
            "post-terminal label"
        )
    return output


def _module_gradient_norm(modules) -> float:
    modules = tuple(modules)
    device = next(modules[0].parameters()).device
    squared = torch.zeros((), device=device)
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.float().pow(2).sum()
    return float(squared.sqrt())


def component_gradient_diagnostic(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    """Fixed pre-training scale audit; it cannot alter registered weights."""
    output = {}
    for component_name in ("latent", "reward", "continuation"):
        world = build_fresh_world(device)
        components = generated_step_components(
            world, batch, prefix=PREFIX, steps=GENERATED_STEPS
        )
        world.zero_grad(set_to_none=True)
        components[component_name].backward()
        output[component_name] = {
            "loss": float(components[component_name].detach()),
            "shared_gradient_l2": _module_gradient_norm((
                world.action_input,
                world.temporal,
                world.future,
            )),
            "reward_head_gradient_l2": _module_gradient_norm((world.reward,)),
            "continuation_head_gradient_l2": _module_gradient_norm(
                (world.continuation,)
            ),
        }
        del world
    output["registered_reward_weight"] = GENERATED_REWARD_WEIGHT
    output["scaled_reward_to_latent_gradient_ratio"] = (
        GENERATED_REWARD_WEIGHT
        * output["reward"]["shared_gradient_l2"]
        / output["latent"]["shared_gradient_l2"]
    )
    return output


def validate_fixed_contract(
    train: list[dict],
    schedule_digest: str,
    device: torch.device,
) -> tuple[dict, dict]:
    if schedule_digest != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Stage-2C schedule differs from pinned Arm A")
    if sha256_file(BASELINE) != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("pinned Arm-A checkpoint hash drift")
    stage2 = json.loads(STAGE2_REPORT.read_text())
    if stage2["schedule_sha256"] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Stage-2 report schedule drift")
    if stage2["arms"]["A"]["checkpoint_sha256"] != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("Stage-2 report baseline hash drift")

    fresh = build_fresh_world(device)
    fresh_digest = state_digest(fresh, exclude_heads=False)
    if fresh_digest != EXPECTED_INIT_DIGEST:
        raise RuntimeError(
            f"fresh initialization drift: {fresh_digest}"
        )
    del fresh

    baseline, payload = load_world_checkpoint(
        BASELINE,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=EXPECTED_BASELINE_SHA256,
    )
    if payload["extra"]["init_digest"] != EXPECTED_INIT_DIGEST:
        raise RuntimeError("baseline initial-state digest drift")
    enforce_frozen_encoder(baseline)
    baseline.eval()
    return baseline, {
        "baseline_checkpoint_sha256": EXPECTED_BASELINE_SHA256,
        "baseline_checkpoint_head": payload["provenance"]["head"],
        "baseline_encoder_sha256": payload["provenance"][
            "encoder_state_sha256"
        ],
        "fresh_init_digest": fresh_digest,
        "training_replay_episodes": len(train),
    }


def _atomic_world_checkpoint(
    path: Path,
    world,
    base_weights,
    *,
    histories: dict,
    extra: dict,
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = save_world_checkpoint(
        temporary,
        world,
        base_weights,
        loss_histories=histories,
        extra=extra,
    )
    temporary.replace(path)
    if sha256_file(path) != digest:
        raise RuntimeError("atomic checkpoint digest changed after rename")
    return digest


def train_arm(
    arm: str,
    spec: GeneratedLossWeights,
    schedule: list[tuple[int, int]],
    schedule_digest: str,
    train: list[dict],
    device: torch.device,
) -> dict:
    world = build_fresh_world(device)
    init_digest = state_digest(world, exclude_heads=False)
    if init_digest != EXPECTED_INIT_DIGEST:
        raise RuntimeError(f"{arm} did not branch from the registered init")
    base_weights = frozen_dynamics_recipe()
    trainable_names = [
        name for name, parameter in world.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [
        parameter for parameter in world.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    histories = {
        "jepa": [],
        "base_reward": [],
        "base_continuation": [],
        "base_rollout": [],
        "generated_latent": [],
        "generated_reward": [],
        "generated_continuation": [],
        "generated_weighted": [],
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update in range(UPDATES):
        picks = schedule[update * BATCH:(update + 1) * BATCH]
        batch = make_batch(train, picks, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = world(batch, base_weights)
            components = generated_step_components(
                world, batch, prefix=PREFIX, steps=GENERATED_STEPS
            )
            generated = weighted_generated_loss(components, spec)
            loss = base.loss + generated
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()

        histories["jepa"].append(float(base.metrics["jepa"]))
        histories["base_reward"].append(float(base.metrics["reward"]))
        histories["base_continuation"].append(
            float(base.metrics["continuation"])
        )
        histories["base_rollout"].append(float(base.metrics["rollout"]))
        for name in ("latent", "reward", "continuation"):
            histories[f"generated_{name}"].append(
                float(components[name].detach())
            )
        histories["generated_weighted"].append(float(generated.detach()))
        if update % 4_000 == 3_999:
            print(
                f"[{arm}] {update + 1}: "
                f"jepa={np.mean(histories['jepa'][-500:]):.5f} "
                f"gen_lat={np.mean(histories['generated_latent'][-500:]):.5f} "
                f"gen_rew={np.mean(histories['generated_reward'][-500:]):.5f}",
                flush=True,
            )

    assert_encoder_frozen(world, optimizer)
    minutes = (time.perf_counter() - started) / 60
    final_digest = state_digest(world, exclude_heads=False)
    checkpoint = ARTIFACTS / f"stage2c_{arm.replace('-', '').lower()}_s{SEED}.pt"
    checkpoint_digest = _atomic_world_checkpoint(
        checkpoint,
        world,
        base_weights,
        histories=histories,
        extra={
            "protocol": PROTOCOL,
            "arm": arm,
            "seed": SEED,
            "updates": UPDATES,
            "generated_weights": dataclasses.asdict(spec),
            "init_digest": init_digest,
            "final_digest": final_digest,
            "schedule_sha256": schedule_digest,
            "trainable_names": trainable_names,
        },
    )
    info = {
        "generated_weights": dataclasses.asdict(spec),
        "init_digest": init_digest,
        "final_digest": final_digest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "train_minutes": round(minutes, 2),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 2
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 2**20, 2
        ),
        "last500": {
            name: float(np.mean(values[-500:]))
            for name, values in histories.items()
        },
    }
    del world, optimizer
    torch.cuda.empty_cache()
    return info


def _load_arm(path: Path, digest: str, device: torch.device):
    world, _ = load_world_checkpoint(
        path,
        device,
        expect_config=sprint_candidate_config("gru"),
        expect_sha256=digest,
    )
    enforce_frozen_encoder(world)
    return world.eval()


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2C fitting requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("commit pre-outcome work first:\n" + "\n".join(dirty))

    manifest = json.loads(MANIFEST.read_text())
    # Deliberately access only the spent DEV tier. FINAL stays serialized and
    # untouched until a separately registered planner gate.
    dev = manifest["dev"]
    for key in ("natural", "terminal", "bundle"):
        path = Path(dev[key]["path"])
        if sha256_file(path) != dev[key]["sha256"]:
            raise RuntimeError(f"dev {key} hash mismatch")
    natural_episodes = torch.load(
        Path(dev["natural"]["path"]), weights_only=False
    )
    terminal_episodes = torch.load(
        Path(dev["terminal"]["path"]), weights_only=False
    )
    anchors = torch.load(Path(dev["bundle"]["path"]), weights_only=False)
    natural_arrays = window_arrays(
        natural_episodes, target_rows(natural_episodes)
    )
    terminal_rows = target_rows(terminal_episodes)
    terminal_arrays = window_arrays(terminal_episodes, terminal_rows)
    actual_continue = continuation_targets(
        terminal_episodes, terminal_rows
    )

    train, _ = load_scaled_data()
    schedule, schedule_digest = build_schedule(train)
    baseline, fixed = validate_fixed_contract(
        train, schedule_digest, device
    )
    distribution = training_distribution(train, schedule)
    first_batch = make_batch(train, schedule[:BATCH], device)
    gradient_diagnostic = component_gradient_diagnostic(
        first_batch, device
    )
    del first_batch
    torch.cuda.empty_cache()

    report = {
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "fixed_contract": fixed,
        "hashes": {
            "manifest": sha256_file(MANIFEST),
            "stage2_report": sha256_file(STAGE2_REPORT),
            "baseline": EXPECTED_BASELINE_SHA256,
            "dev": {key: dev[key]["sha256"] for key in
                    ("natural", "terminal", "bundle")},
        },
        "schedule_sha256": schedule_digest,
        "training_distribution": distribution,
        "gradient_diagnostic": gradient_diagnostic,
        "arms": {},
        "evaluation": {},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    for arm, spec in ARM_SPECS.items():
        report["arms"][arm] = train_arm(
            arm, spec, schedule, schedule_digest, train, device
        )
        REPORT_PATH.write_text(json.dumps(report, indent=2))

    targets = {
        "reward_actual": natural_arrays["rewards"].tolist(),
        "reward_episode": natural_arrays["episodes"].tolist(),
        "reward_transition": natural_arrays["transitions"].tolist(),
        "continue_actual": actual_continue.tolist(),
        "continue_episode": terminal_arrays["episodes"].tolist(),
        "continue_transition": terminal_arrays["transitions"].tolist(),
    }
    raw = {"targets": targets, "arms": {}}
    arm_loaders = {
        "A": (BASELINE, EXPECTED_BASELINE_SHA256),
        **{
            arm: (
                Path(info["checkpoint"]),
                info["checkpoint_sha256"],
            )
            for arm, info in report["arms"].items()
        },
    }
    for arm, (path, digest) in arm_loaders.items():
        world = baseline if arm == "A" else _load_arm(path, digest, device)
        point, arm_raw = evaluate_arm(
            world,
            natural_arrays,
            terminal_arrays,
            actual_continue,
            anchors,
            device,
        )
        report["evaluation"][arm] = point
        raw["arms"][arm] = arm_raw
        print(
            f"[eval {arm}] "
            f"latent-k8={point['latent_depth']['k8']['cosine_error']:.5f} "
            f"reward-k8={point['reward_depth']['k8']['event_auroc']:.4f} "
            f"cont-k8={point['continuation_depth']['k8']['brier_skill']:.4f} "
            f"rank={point['ranking']['chosen_minus_random_mean']:.4f}",
            flush=True,
        )
        if arm != "A":
            del world
            torch.cuda.empty_cache()

    RAW_PATH.write_text(json.dumps(raw))
    report["hashes"]["raw"] = sha256_file(RAW_PATH)
    report["hashes"]["checkpoints"] = {
        arm: info["checkpoint_sha256"]
        for arm, info in report["arms"].items()
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Stage-2C complete: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
