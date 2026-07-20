"""Matched real-Crafter Transformer versus Mamba feasibility runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .checkpoint import (
    file_sha256,
    implementation_sha256,
    load_checkpoint,
    load_tokenizer_checkpoint,
    save_checkpoint,
)
from .config import D4LiteConfig
from .crafter_preflight import (
    DEV_REPLAY,
    DEV_REPLAY_SHA256,
    TRAIN_REPLAY,
    TRAIN_REPLAY_SHA256,
    _aligned_eval_batches,
    _fixed_eval_batches,
    _to_device,
    evaluate_world,
)
from .data import SequenceBatch, load_episode_replay, replay_sample_to_sequence
from .model import D4LiteWorld
from .objectives import optimizer_groups
from .source import source_report
from .training import WorldLossNormalizer, world_loss


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_CHECKPOINT = (
    REPO_ROOT / "outputs/d4_mamba_jepa/preflight_t_base_5k/tokenizer_t_base.pt"
)
TOKENIZER_SHA256 = (
    "91a210dc8c76fa29793599ced04190438d776a0c1a757b674691272eeb58b22c"
)
FORMAT = "d4_mamba_jepa_stage_m1_v1"
INIT_SEED = 20260721
TRAINING_NOISE_SEED = 20260722
SCHEDULE_SEED = 20260723
EVAL_SEED = 20260724
DETERMINISM_SOURCE = (
    REPO_ROOT
    / "third_party/sources/state-spaces__mamba/mamba_ssm/utils/determinism.py"
)
DETERMINISM_SOURCE_SHA256 = (
    "cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _self_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _tensor_digest(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _rng_digest() -> str:
    states = [("torch_cpu", torch.get_rng_state())]
    if torch.cuda.is_available():
        states.extend(
            (f"torch_cuda_{index}", state)
            for index, state in enumerate(torch.cuda.get_rng_state_all())
        )
    return _tensor_digest(states)


def configure_determinism(device: torch.device) -> dict:
    """Enable the deterministic contract required by official Mamba kernels."""
    if device.type == "cuda" and torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before the Stage-M1 determinism contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not DETERMINISM_SOURCE.is_file():
        raise RuntimeError(
            f"missing pinned Mamba determinism source: {DETERMINISM_SOURCE}"
        )
    pinned_sha256 = file_sha256(DETERMINISM_SOURCE)
    if pinned_sha256 != DETERMINISM_SOURCE_SHA256:
        raise RuntimeError(
            "pinned Mamba determinism source drift: "
            f"{pinned_sha256} != {DETERMINISM_SOURCE_SHA256}"
        )
    import inspect
    import mamba_ssm.utils.determinism as mamba_determinism

    installed_path = Path(
        inspect.getsourcefile(mamba_determinism) or ""
    ).resolve()
    if not installed_path.is_file():
        raise RuntimeError("cannot locate installed Mamba determinism source")
    installed_sha256 = file_sha256(installed_path)
    if installed_sha256 != DETERMINISM_SOURCE_SHA256:
        raise RuntimeError(
            "installed Mamba determinism source drift: "
            f"{installed_sha256} != {DETERMINISM_SOURCE_SHA256}"
        )
    if not mamba_determinism.use_deterministic_mode():
        raise RuntimeError("official Mamba deterministic mode is not active")
    return {
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "triton_cache_autotuning": os.environ["TRITON_CACHE_AUTOTUNING"],
        "mamba_deterministic_mode": (
            mamba_determinism.use_deterministic_mode()
        ),
        "pinned_source": str(DETERMINISM_SOURCE),
        "installed_source": str(installed_path),
        "source_sha256": installed_sha256,
    }


def _is_dynamics_temporal(name: str) -> bool:
    return (
        name.startswith("dynamics.transformer.layers.")
        and ".time." in name
    )


def verify_shared_initialization(
    transformer: D4LiteWorld,
    mamba: D4LiteWorld,
) -> dict:
    """Require every non-temporal state tensor to be bit-identical."""
    left = transformer.state_dict()
    right = mamba.state_dict()
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    unexpected = [
        name
        for name in left_only + right_only
        if not _is_dynamics_temporal(name)
    ]
    if unexpected:
        raise RuntimeError(
            f"non-temporal state keys differ across backends: {unexpected}"
        )
    shared_names = sorted(set(left).intersection(right))
    mismatched = [
        name
        for name in shared_names
        if not torch.equal(left[name], right[name])
    ]
    if mismatched:
        raise RuntimeError(
            f"shared initialization differs at {mismatched[:8]}"
        )
    if not left_only or not right_only:
        raise RuntimeError("backend-specific temporal state was not isolated")
    return {
        "shared_tensors": len(shared_names),
        "shared_parameters_and_buffers": sum(
            left[name].numel() for name in shared_names
        ),
        "shared_sha256": _tensor_digest(
            (name, left[name]) for name in shared_names
        ),
        "transformer_temporal_keys": left_only,
        "mamba_temporal_keys": right_only,
    }


@dataclass(frozen=True)
class WindowSchedule:
    """Hashable episode/window schedule materialized identically for both arms."""

    entries: np.ndarray  # int64 [updates,batch,2] = episode index, start
    sequence_length: int

    @classmethod
    def generate(
        cls,
        replay,
        *,
        updates: int,
        batch_size: int,
        sequence_length: int,
        seed: int,
    ) -> "WindowSchedule":
        if updates < 1 or batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        valid = np.asarray(
            [
                index
                for index, episode in enumerate(replay.episodes)
                if len(episode.obs) >= sequence_length
            ],
            dtype=np.int64,
        )
        if not len(valid):
            raise RuntimeError("no episode can supply the requested sequence")
        rng = np.random.default_rng(seed)
        entries = np.empty((updates, batch_size, 2), dtype=np.int64)
        for update in range(updates):
            for row in range(batch_size):
                episode_index = int(valid[int(rng.integers(len(valid)))])
                episode = replay.episodes[episode_index]
                start = int(
                    rng.integers(0, len(episode.obs) - sequence_length + 1)
                )
                entries[update, row] = (episode_index, start)
        return cls(entries=entries, sequence_length=sequence_length)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            np.asarray(
                [*self.entries.shape, self.sequence_length],
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(self.entries.tobytes())
        return digest.hexdigest()

    def materialize(self, replay, update: int) -> SequenceBatch:
        if not 0 <= update < self.entries.shape[0]:
            raise IndexError(update)
        observations = []
        actions = []
        rewards = []
        continues = []
        previous_actions = []
        length = self.sequence_length
        for episode_index, start in self.entries[update]:
            episode = replay.episodes[int(episode_index)]
            start = int(start)
            stop = start + length
            observations.append(episode.obs[start:stop])
            action_slice = episode.actions[start:stop - 1]
            actions.append(action_slice)
            rewards.append(episode.rewards[start:stop - 1])
            continues.append(episode.continues[start:stop - 1])
            previous = np.full(length, -1, dtype=np.int64)
            if start > 0:
                previous[0] = episode.actions[start - 1]
            previous[1:] = action_slice
            previous_actions.append(previous)
        sample = {
            "obs": torch.from_numpy(np.stack(observations)),
            "actions": torch.from_numpy(np.stack(actions)),
            "rewards": torch.from_numpy(np.stack(rewards)),
            "continues": torch.from_numpy(np.stack(continues)),
            "previous_actions": torch.from_numpy(np.stack(previous_actions)),
        }
        return replay_sample_to_sequence(sample)


def _build_world(
    *,
    cfg: D4LiteConfig,
    tokenizer,
    seed: int,
) -> D4LiteWorld:
    torch.manual_seed(seed)
    world = D4LiteWorld(cfg)
    world.encoder.load_state_dict(tokenizer.encoder.state_dict(), strict=True)
    world.decoder.load_state_dict(tokenizer.decoder.state_dict(), strict=True)
    world.freeze_tokenizer()
    return world


def _history_mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
    }


def _train_arm(
    *,
    world: D4LiteWorld,
    replay,
    schedule: WindowSchedule,
    evaluation: dict[str, list[SequenceBatch]],
    output_dir: Path,
    device: torch.device,
    learning_rate: float,
    warmup_steps: int,
    bootstrap_start: int,
    training_noise_seed: int,
    runner_sha256: str,
    shared_initialization_sha256: str,
    determinism: dict,
) -> dict:
    arm = world.cfg.arm_id
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    world = world.to(device)
    normalizer = WorldLossNormalizer().to(device)
    groups = optimizer_groups(world, learning_rate)
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    before = {
        name: evaluate_world(
            world,
            batches,
            cfg=world.cfg,
            device=device,
            seed=EVAL_SEED + offset,
        )
        for offset, (name, batches) in enumerate(evaluation.items())
    }

    torch.manual_seed(training_noise_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_noise_seed)
    world.train()
    world.encoder.eval()
    world.decoder.eval()
    history: list[dict[str, float]] = []
    bootstrap_rows = max(
        0, min(schedule.entries.shape[1] - 1, round(0.25 * schedule.entries.shape[1]))
    )
    started = time.perf_counter()
    for update in range(schedule.entries.shape[0]):
        batch = _to_device(schedule.materialize(replay, update), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, metrics = world_loss(
                world,
                batch,
                normalizer=normalizer,
                global_step=update,
                bootstrap_rows=bootstrap_rows,
                bootstrap_start=bootstrap_start,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{arm} non-finite loss at update {update}")
        loss.backward()
        trainable = [
            parameter
            for group in groups
            for parameter in group["params"]
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(
                f"{arm} non-finite gradient norm at update {update}"
            )
        if update < warmup_steps:
            fraction = float(update + 1) / warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * fraction
        optimizer.step()
        history.append(
            {
                "loss/flow": float(metrics["loss/flow"].item()),
                "loss/reward": float(metrics["loss/reward"].item()),
                "loss/continuation": float(
                    metrics["loss/continuation"].item()
                ),
                "loss/total": float(metrics["loss/total"].item()),
            }
        )
        if (update + 1) % 1_000 == 0:
            print(
                f"[{arm}] update {update + 1}/{schedule.entries.shape[0]} "
                f"flow={history[-1]['loss/flow']:.6f} "
                f"reward={history[-1]['loss/reward']:.6f} "
                f"continuation={history[-1]['loss/continuation']:.6f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    post_training_rng_sha256 = _rng_digest()
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    after = {
        name: evaluate_world(
            world,
            batches,
            cfg=world.cfg,
            device=device,
            seed=EVAL_SEED + offset,
        )
        for offset, (name, batches) in enumerate(evaluation.items())
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{arm.lower().replace('-', '_')}.pt"
    checkpoint_sha256 = save_checkpoint(
        checkpoint_path,
        world=world,
        normalizer=normalizer,
        optimizer=optimizer,
        step=schedule.entries.shape[0],
        extra={
            "format": FORMAT,
            "runner_sha256": runner_sha256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "schedule_sha256": schedule.sha256,
            "shared_initialization_sha256": shared_initialization_sha256,
            "training_noise_seed": training_noise_seed,
            "post_training_rng_sha256": post_training_rng_sha256,
            "determinism": determinism,
        },
    )
    final_world_sha256 = _tensor_digest(world.state_dict().items())
    window = min(20, len(history))
    result = {
        "arm": arm,
        "config": asdict(world.cfg),
        "parameters": {
            "total": sum(parameter.numel() for parameter in world.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in world.parameters()
                if parameter.requires_grad
            ),
        },
        "training": {
            "updates": schedule.entries.shape[0],
            "seconds": training_seconds,
            "updates_per_second": schedule.entries.shape[0] / training_seconds,
            "peak_vram_bytes": peak_vram,
            "bootstrap_rows": bootstrap_rows,
            "first_20": _history_mean(history[:window]),
            "last_20": _history_mean(history[-window:]),
            "post_training_rng_sha256": post_training_rng_sha256,
        },
        "evaluation": {"before": before, "after": after},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "world_state_sha256": final_world_sha256,
        },
    }

    # The on-disk artifact must reconstruct under strict source/config/state
    # checks before this arm can enter the comparison.
    del optimizer
    world = world.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loaded, _, payload = load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_config=world.cfg,
        expected_sha256=checkpoint_sha256,
    )
    if _tensor_digest(loaded.state_dict().items()) != final_world_sha256:
        raise RuntimeError(f"{arm} strict checkpoint state digest mismatch")
    result["checkpoint"]["strict_reload_step"] = int(payload["step"])
    return result


def _gate(arms: dict[str, dict]) -> dict:
    transformer = arms["T-BASE"]["evaluation"]["after"]["uniform"]
    mamba = arms["M-BASE"]["evaluation"]["after"]["uniform"]
    t_flow = transformer["raw_losses"]["flow/flow_mse"]
    m_flow = mamba["raw_losses"]["flow/flow_mse"]
    t_reward = transformer["generated_k4_reward"]
    m_reward = mamba["generated_k4_reward"]
    checks = {
        "flow_mse_at_most_1_25x_transformer": m_flow <= 1.25 * t_flow,
        "action_shuffle_ratio_at_least_1_05": (
            mamba["paired_action_shuffle"]["shuffled_over_true"] >= 1.05
        ),
        "correct_action_mean_advantage_positive": (
            mamba["one_step_action_conditioning"]["wrong_minus_actual_mse"] > 0
        ),
        "reward_auroc_no_more_than_0_05_below_transformer": (
            m_reward["event_auroc_abs_prediction"]
            >= t_reward["event_auroc_abs_prediction"] - 0.05
        ),
        "false_reward_no_more_than_2x_transformer": (
            m_reward["zero_target_mean_abs_prediction"]
            <= 2.0 * t_reward["zero_target_mean_abs_prediction"]
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "ratios_and_deltas": {
            "mamba_over_transformer_flow_mse": m_flow / t_flow,
            "mamba_minus_transformer_reward_auroc": (
                m_reward["event_auroc_abs_prediction"]
                - t_reward["event_auroc_abs_prediction"]
            ),
            "mamba_over_transformer_false_reward": (
                m_reward["zero_target_mean_abs_prediction"]
                / max(t_reward["zero_target_mean_abs_prediction"], 1e-12)
            ),
        },
        "next": (
            "RUN_MATCHED_BASE_VS_CDP_FACTORIAL"
            if all(checks.values())
            else "STOP_AND_LOCALIZE_MAMBA_BACKEND"
        ),
    }


def run(
    *,
    output_dir: Path,
    updates: int,
    batch_size: int,
    device: torch.device,
    learning_rate: float,
) -> dict:
    determinism = configure_determinism(device)
    runner_sha256 = _self_sha256()
    base_cfg = D4LiteConfig()
    tokenizer, tokenizer_payload = load_tokenizer_checkpoint(
        TOKENIZER_CHECKPOINT,
        device=torch.device("cpu"),
        expected_config=base_cfg,
        expected_sha256=TOKENIZER_SHA256,
        training_mask=False,
    )
    train_replay = load_episode_replay(
        TRAIN_REPLAY, expected_sha256=TRAIN_REPLAY_SHA256
    )
    dev_replay = load_episode_replay(
        DEV_REPLAY, expected_sha256=DEV_REPLAY_SHA256
    )
    schedule = WindowSchedule.generate(
        train_replay,
        updates=updates,
        batch_size=batch_size,
        sequence_length=base_cfg.sequence_length,
        seed=SCHEDULE_SEED,
    )
    evaluation = {
        "uniform": _fixed_eval_batches(
            dev_replay,
            cfg=base_cfg,
            count=16,
            batch_size=32,
            seed=EVAL_SEED,
        ),
        "reward_event_aligned": _aligned_eval_batches(
            dev_replay,
            target="reward_event",
            context=8,
            batch_size=32,
        ),
        "terminal_aligned": _aligned_eval_batches(
            dev_replay,
            target="terminal",
            context=8,
            batch_size=32,
        ),
    }

    transformer_cfg = replace(base_cfg, temporal_backend="transformer")
    mamba_cfg = replace(base_cfg, temporal_backend="mamba2")
    transformer = _build_world(
        cfg=transformer_cfg, tokenizer=tokenizer, seed=INIT_SEED
    )
    mamba = _build_world(cfg=mamba_cfg, tokenizer=tokenizer, seed=INIT_SEED)
    shared = verify_shared_initialization(transformer, mamba)

    arms = {}
    for world in (transformer, mamba):
        result = _train_arm(
            world=world,
            replay=train_replay,
            schedule=schedule,
            evaluation=evaluation,
            output_dir=output_dir,
            device=device,
            learning_rate=learning_rate,
            warmup_steps=1_000,
            bootstrap_start=10_000,
            training_noise_seed=TRAINING_NOISE_SEED,
            runner_sha256=runner_sha256,
            shared_initialization_sha256=shared["shared_sha256"],
            determinism=determinism,
        )
        arms[result["arm"]] = result

    rng_match = (
        arms["T-BASE"]["training"]["post_training_rng_sha256"]
        == arms["M-BASE"]["training"]["post_training_rng_sha256"]
    )
    if not rng_match:
        raise RuntimeError("paired arms consumed different Torch RNG streams")
    decision = _gate(arms)
    payload = {
        "format": FORMAT,
        "status": "completed",
        "claim_boundary": (
            "one-seed matched backend feasibility screen on spent diagnostic "
            "data; not a final architecture or long-context claim"
        ),
        "provenance": {
            "runner_sha256": runner_sha256,
            "core_implementation_sha256": implementation_sha256(),
            "sources": source_report(),
            "tokenizer": {
                "path": str(TOKENIZER_CHECKPOINT),
                "sha256": file_sha256(TOKENIZER_CHECKPOINT),
                "step": int(tokenizer_payload["step"]),
            },
            "train_replay_sha256": TRAIN_REPLAY_SHA256,
            "dev_replay_sha256": DEV_REPLAY_SHA256,
            "determinism": determinism,
        },
        "pairing": {
            "initialization_seed": INIT_SEED,
            "training_noise_seed": TRAINING_NOISE_SEED,
            "schedule_seed": SCHEDULE_SEED,
            "evaluation_seed": EVAL_SEED,
            "schedule_sha256": schedule.sha256,
            "schedule_shape": list(schedule.entries.shape),
            "shared_initialization": shared,
            "post_training_rng_match": rng_match,
        },
        "arms": arms,
        "decision": decision,
    }
    report_path = output_dir / "report.json"
    _atomic_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/d4_mamba_jepa/stage_m1",
    )
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    payload = run(
        output_dir=args.output_dir,
        updates=args.updates,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        learning_rate=args.learning_rate,
    )
    summary = {
        "status": payload["status"],
        "pairing": payload["pairing"],
        "decision": payload["decision"],
        "arms": {
            name: {
                "training": arm["training"],
                "after": arm["evaluation"]["after"],
                "checkpoint": arm["checkpoint"],
            }
            for name, arm in payload["arms"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
