"""Training-only Stage-2G provenance, gradient, and 256-update preflight."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from fork_oracle_v2 import sha256_file  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import BATCH, build_fresh_world, build_schedule, make_batch  # noqa: E402
from stage2_objectives import generated_step_components  # noqa: E402
from stage2f_reward_operator import (  # noqa: E402
    LOCAL_FINGERPRINT,
    history_sha256,
    train_world,
)
from stage2g_relevance import (  # noqa: E402
    ARM_REWARD_WEIGHTS,
    FULL_UPDATES,
    GRADIENT_BATCHES,
    PROTOCOL,
    SMOKE_UPDATES,
    WINDOW,
    build_auxiliary_contract,
    build_relevance_heads,
    component_gradient_norms,
    module_state_digest,
    relevance_loss,
    relevance_pools,
    schedule_label_audit,
    train_reference_world,
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
OUTPUT = ARTIFACTS / "stage2g_preflight.json"
EXPECTED_REPLAY_SHA256 = (
    "c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad"
)
EXPECTED_INIT_DIGEST = (
    "55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260"
)
SOURCES = {
    "taco": {
        "root": (
            REPO_ROOT / "third_party/sources/FrankZheng2022__TACO"
        ),
        "commit": (
            "84c38e34f4f9dfd2b059fb6d1356757e8d40712e"
        ),
        "files": ("agents/taco.py",),
    },
    "dbc": {
        "root": (
            REPO_ROOT
            / "third_party/sources/facebookresearch__deep_bisim4control"
        ),
        "commit": (
            "5967b6d0ccfc1032837cbe542f7bc5a96dc02cbb"
        ),
        "files": ("agent/bisim_agent.py", "agent/deepmdp_agent.py"),
    },
}
PAPERS = {
    "deepmdp": (
        "third_party/papers/1906.02736v1.pdf",
        "8d378562cd5c69829af5a7d1e35300b0f81905e6ed5c06e04626603c8a7e2e33",
    ),
    "dbc": (
        "third_party/papers/2006.10742v2.pdf",
        "8322d722f58393f703be8cbc7c24de360a028926db6843feb0e2adc3a392b952",
    ),
    "taco": (
        "third_party/papers/2306.13229v3.pdf",
        "93e21eba6d0721628f69eef345afcffc89d2be32e3e15bca7f867ff2be55733c",
    ),
    "byol_ac": (
        "third_party/papers/2406.02035v1.pdf",
        "38c8e0a756c804bdb0630c0a149179a3d75394fe7de7e3c70ad5c14e0c31ab27",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance() -> dict:
    output = {"papers": {}, "repositories": {}}
    for name, (relative, expected) in PAPERS.items():
        path = REPO_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"{name} paper hash drift")
        output["papers"][name] = {
            "path": relative,
            "sha256": observed,
        }
    for name, spec in SOURCES.items():
        root = spec["root"]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != spec["commit"] or dirty:
            raise RuntimeError(
                f"{name} source is not clean at pinned commit"
            )
        output["repositories"][name] = {
            "commit": commit,
            "clean": True,
            "file_sha256": {
                relative: sha256_file(root / relative)
                for relative in spec["files"]
            },
        }
    return output


def _module_l2(module) -> float:
    device = next(module.parameters()).device
    squared = torch.zeros((), device=device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.float().pow(2).sum()
    return float(squared.sqrt())


def gradient_registration(
    train: list[dict],
    base_schedule: list[tuple[int, int]],
    auxiliary_schedule: list[tuple[int, int]],
    device: torch.device,
) -> dict:
    world = build_fresh_world(device)
    heads = build_relevance_heads(world.cfg.token_dim, device)
    reward_shared = []
    auxiliary_shared = []
    reward_rows = []
    auxiliary_rows = []

    for index in range(GRADIENT_BATCHES):
        base_batch = make_batch(
            train,
            base_schedule[index * BATCH:(index + 1) * BATCH],
            device,
        )
        auxiliary_batch = make_batch(
            train,
            auxiliary_schedule[
                index * BATCH:(index + 1) * BATCH
            ],
            device,
            window=WINDOW,
        )
        world.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            components = generated_step_components(
                world, base_batch, prefix=8, steps=2
            )
        components["reward"].backward()
        reward_norms = component_gradient_norms(world, heads)
        reward_norms["action_input"] = _module_l2(
            world.action_input
        )
        reward_norms["future"] = _module_l2(world.future)
        reward_norms["temporal"] = _module_l2(world.temporal)
        reward_shared.append(reward_norms["shared"])
        reward_rows.append(reward_norms)

        world.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            auxiliary = relevance_loss(
                world, heads, auxiliary_batch
            )
        auxiliary["loss"].backward()
        auxiliary_norms = component_gradient_norms(world, heads)
        auxiliary_norms["action_input"] = _module_l2(
            world.action_input
        )
        auxiliary_norms["future"] = _module_l2(world.future)
        auxiliary_norms["temporal"] = _module_l2(world.temporal)
        auxiliary_shared.append(auxiliary_norms["shared"])
        auxiliary_rows.append(auxiliary_norms)

    reward_rms = float(np.sqrt(np.mean(
        np.square(reward_shared, dtype=np.float64)
    )))
    auxiliary_rms = float(np.sqrt(np.mean(
        np.square(auxiliary_shared, dtype=np.float64)
    )))
    if (
        not np.isfinite(reward_rms)
        or not np.isfinite(auxiliary_rms)
        or reward_rms <= 0
        or auxiliary_rms <= 0
    ):
        raise RuntimeError("invalid Stage-2G gradient registration")
    coefficient = 0.10 * reward_rms / auxiliary_rms
    if not 0.01 <= coefficient <= 10.0:
        raise RuntimeError(
            f"auxiliary coefficient {coefficient} outside protocol"
        )

    for row in auxiliary_rows:
        for name in ("action_input", "future", "temporal",
                     "shared", "auxiliary_heads"):
            if row[name] <= 0:
                raise RuntimeError(
                    f"auxiliary has no gradient in {name}"
                )
        for name in (
            "reward_head",
            "continuation_head",
            "online_encoder",
            "target_encoder",
        ):
            if row[name] != 0.0:
                raise RuntimeError(
                    f"auxiliary leaked gradient into {name}"
                )

    world.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    detached_batch = make_batch(
        train, auxiliary_schedule[:BATCH], device, window=WINDOW
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        detached = relevance_loss(
            world, heads, detached_batch, detach_world=True
        )
    detached["loss"].backward()
    detached_norms = component_gradient_norms(world, heads)
    if detached_norms["shared"] != 0.0:
        raise RuntimeError("detached auxiliary reaches shared world")
    if detached_norms["auxiliary_heads"] <= 0:
        raise RuntimeError("detached auxiliary cannot train its heads")

    return {
        "batches": GRADIENT_BATCHES,
        "raw_generated_reward_shared_l2": reward_shared,
        "raw_auxiliary_shared_l2": auxiliary_shared,
        "raw_generated_reward_rms": reward_rms,
        "raw_auxiliary_rms": auxiliary_rms,
        "target_generated_reward_weight": 0.10,
        "lambda_aux": coefficient,
        "reward_routes": reward_rows,
        "auxiliary_routes": auxiliary_rows,
        "detached_routes": detached_norms,
    }


def validate_smoke(arm: str, info: dict, reference: dict) -> None:
    if info["world_final_digest"] == reference["final_digest"]:
        raise RuntimeError(f"{arm} auxiliary left world bit-identical")
    before = info["probes"]["u0"]
    after = info["probes"][f"u{SMOKE_UPDATES}"]
    if not after["loss"] < before["loss"]:
        raise RuntimeError(f"{arm} auxiliary probe loss did not fall")
    for metric in ("event_auroc", "sign_auroc"):
        if after[metric] is None or after[metric] <= 0.55:
            raise RuntimeError(
                f"{arm} {metric} did not exceed smoke threshold"
            )
    if after["decoded_absolute_maximum"] >= 100.0:
        raise RuntimeError(f"{arm} decoded smoke reward exceeds 100")
    if info["peak_reserved_mib"] >= 5500:
        raise RuntimeError(f"{arm} smoke exceeds VRAM budget")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2G preflight requires CUDA")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError(
            "commit Stage-2G implementation before preflight:\n"
            + "\n".join(dirty)
        )
    if sha256_file(TRAIN_40K_CACHE) != EXPECTED_REPLAY_SHA256:
        raise RuntimeError("training replay drift")
    sources = source_provenance()

    train, _ = load_scaled_data()
    base_schedule, base_schedule_sha256 = build_schedule(train)
    if base_schedule_sha256 != LOCAL_FINGERPRINT["schedule_sha256"]:
        raise RuntimeError("base schedule drift")
    pools = relevance_pools(train)
    auxiliary_schedule, probe, auxiliary_contract = (
        build_auxiliary_contract(pools, updates=FULL_UPDATES)
    )
    auxiliary_contract["schedule_labels"] = schedule_label_audit(
        train, auxiliary_schedule
    )
    auxiliary_contract["probe_labels"] = schedule_label_audit(
        train, probe
    )
    for name in ("schedule_labels", "probe_labels"):
        audit = auxiliary_contract[name]
        if audit["event_window_fraction"] != 0.5:
            raise RuntimeError(f"{name} is not event balanced")
        if audit["positive_window_fraction"] != 0.25:
            raise RuntimeError(f"{name} is not positive balanced")
        if audit["negative_window_fraction"] != 0.25:
            raise RuntimeError(f"{name} is not negative balanced")
        if audit["terminal_row_fraction"] != 0.0:
            raise RuntimeError(f"{name} contains terminal labels")

    fresh = build_fresh_world(device)
    fresh_digest = state_digest(fresh, exclude_heads=False)
    if fresh_digest != EXPECTED_INIT_DIGEST:
        raise RuntimeError("Stage-2G fresh world initialization drift")
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    heads = build_relevance_heads(fresh.cfg.token_dim, device)
    if not torch.equal(cpu_rng_before, torch.get_rng_state()):
        raise RuntimeError("auxiliary initialization changed CPU RNG")
    if not torch.equal(cuda_rng_before, torch.cuda.get_rng_state()):
        raise RuntimeError("auxiliary initialization changed CUDA RNG")
    auxiliary_initial_digest = module_state_digest(heads)
    del fresh, heads
    torch.cuda.empty_cache()

    gradients = gradient_registration(
        train, base_schedule, auxiliary_schedule, device
    )
    coefficient = gradients["lambda_aux"]

    regression = build_fresh_world(device)
    regression, regression_info = train_world(
        regression,
        train,
        base_schedule,
        updates=64,
        progress_name="G-local-regression",
    )
    regression_exact = (
        regression_info["final_digest"]
        == LOCAL_FINGERPRINT["final_digest"]
        and history_sha256(regression_info["histories"])
        == LOCAL_FINGERPRINT["history_sha256"]
    )
    if not regression_exact:
        raise RuntimeError("Stage-2G changed historical C-LR path")
    del regression
    torch.cuda.empty_cache()

    references = {}
    for arm, reward_weight in ARM_REWARD_WEIGHTS.items():
        reference_world = build_fresh_world(device)
        reference_world, reference_info = train_reference_world(
            reference_world,
            train,
            base_schedule,
            reward_weight=reward_weight,
            updates=SMOKE_UPDATES,
        )
        references[arm] = reference_info
        del reference_world
        torch.cuda.empty_cache()

    smokes = {}
    for arm in ARM_REWARD_WEIGHTS:
        world = build_fresh_world(device)
        arm_heads = build_relevance_heads(world.cfg.token_dim, device)
        if state_digest(world, exclude_heads=False) != fresh_digest:
            raise RuntimeError(f"{arm} fresh world differs")
        if module_state_digest(arm_heads) != auxiliary_initial_digest:
            raise RuntimeError(f"{arm} auxiliary initialization differs")
        world, arm_heads, info = train_relevance_world(
            world,
            arm_heads,
            train,
            base_schedule,
            auxiliary_schedule,
            probe,
            arm=arm,
            lambda_aux=coefficient,
            updates=SMOKE_UPDATES,
            probe_updates=(0, SMOKE_UPDATES),
        )
        validate_smoke(arm, info, references[arm])
        smokes[arm] = info
        del world, arm_heads
        torch.cuda.empty_cache()

    output = {
        "format": "stage2g_preflight_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha(REPO_ROOT / PROTOCOL),
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _sha(Path(__file__)),
        "versions": software_versions(),
        "sources": sources,
        "replay": {
            "path": str(TRAIN_40K_CACHE),
            "sha256": EXPECTED_REPLAY_SHA256,
        },
        "base_schedule_sha256": base_schedule_sha256,
        "auxiliary_contract": auxiliary_contract,
        "fresh_world_digest": fresh_digest,
        "auxiliary_initial_digest": auxiliary_initial_digest,
        "gradient_registration": gradients,
        "local_regression": {
            "exact": regression_exact,
            "final_digest": regression_info["final_digest"],
            "history_sha256": history_sha256(
                regression_info["histories"]
            ),
        },
        "references_256": references,
        "smokes_256": smokes,
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(
        f"{OUTPUT}: lambda_aux={coefficient:.9f}; "
        "both 256-update smokes pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
