"""Prepare fixed TRAIN geometry and held-out logged terminal transitions."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    data_digests,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import (
    atomic_torch,
    compact_records,
    fit_fatal_direction,
    regularized_precision,
    terminal_pair_rows,
)
from artifacts.run_stage_a import ARCHIVE, SUPPORT
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import episode_splits, load_episodes, sample_batch
from d4mj.expert import load_archive
from d4mj.representation import Encoder
from d4mj.train import _cache_digest, _generators, cache_latents


def split_pools(config: Config, expert: int) -> tuple[dict[str, list], dict[str, list]]:
    pools = {"expert": load_archive(ARCHIVE, config, limit=expert)}
    if SUPPORT.exists():
        pools["support"] = load_episodes(SUPPORT)
    train, dev = {}, {}
    for index, (name, episodes) in enumerate(pools.items()):
        first, second, _ = episode_splits(len(episodes), config.seed + index)
        train[name] = [episodes[value] for value in first.tolist()]
        dev[name] = [episodes[value] for value in second.tolist()]
    return train, dev


def pool_summary(pools: dict[str, list]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "episodes": len(episodes),
            "transitions": sum(len(episode) for episode in episodes),
            "terminal_episodes": sum(bool(episode.terminated.any()) for episode in episodes),
            "terminal_transitions": sum(int(episode.terminated.sum()) for episode in episodes),
            "truncated_episodes": sum(bool(episode.truncated.any()) for episode in episodes),
        }
        for name, episodes in pools.items()
    }


def covariance_samples(
    episodes: list,
    config: Config,
    count: int,
    schedule_cycle: int,
) -> Tensor:
    """Sample targets in the teacher/rollout proportions of the Direct objective."""
    sampler = torch.Generator().manual_seed(config.seed + 8300)
    schedule = torch.Generator().manual_seed(config.seed + 8301)
    choice = torch.Generator().manual_seed(config.seed + 8302)
    selected = []
    while sum(len(value) for value in selected) < count:
        schedule_step = int(torch.randint(schedule_cycle, (1,), generator=schedule))
        batch = sample_batch(
            episodes, sampler, config, schedule_step, schedule_cycle
        )
        rows, length = batch.latents.shape[:2]
        teacher_index = torch.randint(1, length, (rows,), generator=choice)
        rollout_index = length - 1 - torch.randint(2, (rows,), generator=choice)
        row = torch.arange(rows)
        selected.append(batch.latents[row, teacher_index])
        selected.append(batch.latents[row, rollout_index])
    return torch.cat(selected)[:count]


def metadata_cache(episodes: list) -> list:
    """Mark episodes cached without retaining their 512-D latent payload."""
    return [
        replace(
            episode,
            latents=torch.empty(len(episode) + 1, 0),
            latent_digest="sampler-exposure-only",
        )
        for episode in episodes
    ]


def sampler_exposure(
    episodes: list,
    config: Config,
    *,
    steps: int,
    schedule_cycle: int,
    milestones: tuple[int, ...],
    checkpoint: Path,
    contract: dict,
) -> dict:
    """Replay the exact production sampler and count targets the loss actually scores."""
    sampler, _ = _generators(config, 1)
    state = {
        "step": 0,
        "terminal_batches": 0,
        "teacher_targets": 0,
        "teacher_terminal_targets": 0,
        "rollout_targets": 0,
        "rollout_terminal_targets": 0,
        "effective_target_mass": 0.0,
        "effective_terminal_mass": 0.0,
        "short_batches": 0,
        "long_batches": 0,
        "milestones": {},
    }
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("sampler-exposure contract changed")
        state.update(saved["state"])
        sampler.set_state(saved["sampler"])

    for step in range(int(state["step"]), steps):
        schedule_step = step % schedule_cycle
        batch = sample_batch(
            episodes, sampler, config, schedule_step, schedule_cycle
        )
        terminal = batch.terminated.bool()
        rows, length = terminal.shape
        teacher_terminal = int(terminal[:, 1:].sum())
        rollout_terminal = int(terminal[:, -2:].sum())
        state["terminal_batches"] += int(bool(terminal[:, 1:].any()))
        state["teacher_targets"] += rows * (length - 1)
        state["teacher_terminal_targets"] += teacher_terminal
        state["rollout_targets"] += rows * 2
        state["rollout_terminal_targets"] += rollout_terminal
        state["effective_target_mass"] += 2.0
        state["effective_terminal_mass"] += (
            teacher_terminal / (rows * (length - 1))
            + 0.5 * rollout_terminal / rows
        )
        state[
            "long_batches" if length == config.sequence_long else "short_batches"
        ] += 1
        completed = step + 1
        state["step"] = completed
        if completed in milestones:
            state["milestones"][str(completed)] = {
                key: value
                for key, value in state.items()
                if key not in ("milestones", "step")
            } | {
                "effective_terminal_fraction": (
                    state["effective_terminal_mass"]
                    / state["effective_target_mass"]
                )
            }
        if completed % 1000 == 0 or completed == steps:
            atomic_torch(
                checkpoint,
                {
                    "contract": contract,
                    "state": state,
                    "sampler": sampler.get_state(),
                },
            )
            print(f"sampler exposure: {completed}/{steps}", flush=True)
    return state | {
        "effective_terminal_fraction": (
            state["effective_terminal_mass"] / state["effective_target_mass"]
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--covariance-samples", type=int, default=8192)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--probe-seeds", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=800)
    parser.add_argument("--exposure-steps", type=int, default=80000)
    parser.add_argument("--schedule-cycle", type=int, default=20000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    train_pools, dev_pools = split_pools(base, args.expert)
    train = [episode for episodes in train_pools.values() for episode in episodes]
    contract = {
        "version": "phase1b-archive-geometry-preparation-v1",
        "phase1a": file_digest(args.phase1a),
        "data": data_digests(),
        "implementation": implementation_digests(
            Path(__file__), Path("artifacts/phase1b_geometry_common.py")
        ),
        "expert": args.expert,
        "covariance_samples": args.covariance_samples,
        "shrinkage": args.shrinkage,
        "probe_seeds": args.probe_seeds,
        "probe_steps": args.probe_steps,
        "exposure_steps": args.exposure_steps,
        "schedule_cycle": args.schedule_cycle,
        "split": "the production whole-episode split, expert and support separately",
    }
    prepared_path = args.out / "prepared.pt"
    if prepared_path.exists():
        prepared = torch.load(prepared_path, weights_only=False)
        if prepared["contract"] != contract:
            raise ValueError("archive-geometry preparation contract changed")
        print(f"already complete: {prepared_path}", flush=True)
        return

    encoder = Encoder(base).to(base.device)
    load(args.phase1a, base, part0=encoder)
    encoder.eval()
    cache_digest = _cache_digest(encoder, base)
    statistics_path = args.out / "training_statistics.pt"
    if statistics_path.exists():
        statistics = torch.load(statistics_path, weights_only=False)
        if statistics["contract"] != contract:
            raise ValueError("training-statistics contract changed")
    else:
        print("caching TRAIN latents for direction and covariance", flush=True)
        cached_train = cache_latents(encoder, train, base)
        direction_rows, train_records = terminal_pair_rows(cached_train, "train")
        seeds = [config.seed + 8400 + value for value in range(args.probe_seeds)]
        direction, means, direction_report = fit_fatal_direction(
            direction_rows, config, seeds=seeds, steps=args.probe_steps
        )
        samples = covariance_samples(
            cached_train, config, args.covariance_samples, args.schedule_cycle
        )
        precision, covariance_report = regularized_precision(
            samples, args.shrinkage
        )
        covariance = covariance_report.pop("covariance")
        trace = float(torch.trace(covariance))
        direction_variance = float(direction @ covariance @ direction)
        statistics = {
            "contract": contract,
            "direction": direction,
            "action_means": means,
            "covariance": covariance,
            "precision": precision,
            "report": {
                "fatal_direction": direction_report,
                "covariance": covariance_report,
                "direction_variance": direction_variance,
                "direction_variance_share": direction_variance / trace,
                "direction_variance_relative_to_isotropic": (
                    direction_variance * direction.numel() / trace
                ),
            },
            "train_records": compact_records(train_records, config.sequence_long),
        }
        atomic_torch(statistics_path, statistics)
        del cached_train, direction_rows, train_records, samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []
    evaluation_report = {}
    for name, episodes in dev_pools.items():
        terminal = [episode for episode in episodes if bool(episode.terminated.any())]
        print(f"caching {name} DEV terminal episodes: {len(terminal)}", flush=True)
        cached = cache_latents(encoder, terminal, base)
        rows, pool_records = terminal_pair_rows(cached, name)
        offset = len(records)
        records.extend(pool_records)
        evaluation_report[name] = {
            "pairs": len(pool_records),
            "examples": len(rows["target"]),
            "same_action_safe_pairs": int(rows["same_action_safe_pairs"]),
            "group_offset": offset,
        }
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    exposure_contract = {
        "preparation": contract,
        "milestones": [5000, 20000, 80000],
    }
    exposure = sampler_exposure(
        metadata_cache(train),
        config,
        steps=args.exposure_steps,
        schedule_cycle=args.schedule_cycle,
        milestones=(5000, 20000, 80000),
        checkpoint=args.out / "sampler_exposure.pt",
        contract=exposure_contract,
    )
    report = {
        "contract": contract,
        "cache_digest": cache_digest,
        "train": pool_summary(train_pools),
        "dev": pool_summary(dev_pools),
        "evaluation": evaluation_report,
        "training_geometry": statistics["report"],
        "sampler_exposure": exposure,
    }
    prepared = {
        "contract": contract,
        "cache_digest": cache_digest,
        "direction": statistics["direction"],
        "action_means": statistics["action_means"],
        "covariance": statistics["covariance"],
        "precision": statistics["precision"],
        "train_records": statistics["train_records"],
        "records": records,
        "report": report,
    }
    atomic_torch(prepared_path, prepared)
    atomic_json(args.out / "preparation_report.json", report)
    print(f"complete: {prepared_path}", flush=True)


if __name__ == "__main__":
    main()
