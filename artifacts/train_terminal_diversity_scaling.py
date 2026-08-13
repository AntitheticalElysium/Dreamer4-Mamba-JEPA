"""Phase-1B terminal-tail diversity scaling at fixed exposure."""

from __future__ import annotations

import argparse
import gc
from collections import Counter, defaultdict
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
    state_digest,
)
from artifacts.train_phase1b_geometry_factorial import (
    admit_terminal_batch,
    combine_strata,
    reference_contract,
    tensor_digest,
)
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import Batch, _terminal_start, _window, sample_batch
from d4mj.train import (
    _balance,
    _generators,
    _share_initialisation,
    _to,
    _update,
    generator_state,
    optimizer,
)
from d4mj.transition import World, transition_loss


def terminal_metadata(episodes: list, config: Config) -> dict[int, dict]:
    metadata = {}
    for index, episode in enumerate(episodes):
        terminals = episode.terminated.nonzero().flatten()
        if not episode.uniform_eligible or not len(terminals):
            continue
        if len(terminals) != 1 or int(terminals[0]) != len(episode) - 1:
            raise ValueError("terminal diversity requires one final terminal per episode")
        if len(episode) + 1 < config.sequence_long:
            continue
        terminal = int(terminals[0])
        metadata[index] = {
            "episode_index": index,
            "pool": "expert" if episode.bc_eligible else "support",
            "fatal_action": int(episode.actions_taken[terminal]),
            "steps": len(episode),
        }
    if not metadata:
        raise ValueError("no terminal episode is eligible at both sequence lengths")
    return metadata


def stratified_terminal_ranking(
    metadata: dict[int, dict], seed: int
) -> list[int]:
    """A nested proportional ordering over pool-by-fatal-action strata."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in metadata.items():
        groups[(row["pool"], row["fatal_action"])].append(index)
    rng = torch.Generator().manual_seed(seed)
    tie = {key: float(torch.rand((), generator=rng)) for key in groups}
    for key, values in groups.items():
        order = torch.randperm(len(values), generator=rng).tolist()
        groups[key] = [values[position] for position in order]

    sizes = {key: len(values) for key, values in groups.items()}
    used = {key: 0 for key in groups}
    total = sum(sizes.values())
    ranking = []
    for position in range(total):
        active = [key for key in groups if used[key] < sizes[key]]
        key = max(
            active,
            key=lambda value: (
                (position + 1) * sizes[value] / total - used[value],
                tie[value],
            ),
        )
        ranking.append(groups[key][used[key]])
        used[key] += 1
    if len(ranking) != len(set(ranking)) or set(ranking) != set(metadata):
        raise AssertionError("terminal ranking is not a permutation")
    return ranking


def balanced_terminal_schedule(
    selected: list[int], draws: int, seed: int
) -> torch.Tensor:
    if not selected or draws < len(selected):
        raise ValueError("draw count must expose every selected terminal episode")
    rng = torch.Generator().manual_seed(seed)
    base = torch.tensor(selected, dtype=torch.long)
    chunks = []
    while sum(len(chunk) for chunk in chunks) < draws:
        chunks.append(base[torch.randperm(len(base), generator=rng)])
    schedule = torch.cat(chunks)[:draws]
    counts = Counter(schedule.tolist())
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("terminal schedule is not balanced")
    return schedule


def terminal_tail_batch(episode, config: Config, step: int, total: int) -> Batch:
    if config.terminal_batch != 1:
        raise ValueError("diversity scaling is defined for one terminal row per update")
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (
        config.long_batch_every > 0
        and (step + 1) % config.long_batch_every == 0
    )
    length = config.sequence_long if long else config.sequence
    if len(episode) + 1 < length:
        raise ValueError("selected terminal episode is too short for the schedule")
    row = _window(episode, _terminal_start(episode, length), length, config)
    stack = {field: value[None] for field, value in row.items()}
    batch = Batch(
        burn_in=0,
        relevant=torch.zeros(1, dtype=torch.bool),
        support=torch.ones(1, dtype=torch.bool),
        **stack,
    )
    if not bool(batch.terminated[:, -1].all()):
        raise AssertionError("terminal-tail batch does not end at the terminal")
    return batch


def train_cell(
    episodes,
    schedule: torch.Tensor,
    config: Config,
    *,
    steps: int,
    milestones: tuple[int, ...],
    reference: dict,
    checkpoint: Path,
    out: Path,
    contract: dict,
) -> dict:
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    initial_digest = state_digest(world)
    optimiser = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, model_rng = _generators(config, 1)
    terminal_model_rng = torch.Generator(device=config.device).manual_seed(
        config.seed + 7101
    )
    streams, meta = {}, {}
    resume = 0
    curve: list[dict] = []
    milestone_digests: dict[str, str] = {}
    terminal_rows = 0
    if checkpoint.exists():
        load(
            checkpoint,
            config,
            part0=world,
            part1=optimiser,
            balance=balance,
            streams=streams,
            meta=meta,
        )
        if meta.get("contract") != contract:
            raise ValueError("terminal-diversity checkpoint contract changed")
        if meta.get("initial_world_sha256") != initial_digest:
            raise ValueError("terminal-diversity initialization changed")
        resume = int(meta["step"])
        curve = list(meta.get("curve", []))
        milestone_digests = dict(meta.get("milestone_digests", {}))
        terminal_rows = int(meta.get("terminal_rows", 0))
        sampler.set_state(streams["sampler"])
        model_rng.set_state(streams["model"])
        terminal_model_rng.set_state(streams["terminal_model"])

    out.mkdir(parents=True, exist_ok=True)
    main_values, terminal_values = [], []
    for step in range(resume, steps):
        ordinary_batch = _to(
            sample_batch(episodes, sampler, config, step, steps), config.device
        )
        terminal_batch = _to(
            terminal_tail_batch(
                episodes[int(schedule[step])], config, step, steps
            ),
            config.device,
        )
        ordinary = transition_loss(
            world, ordinary_batch, model_rng, config, step=step
        )
        terminal = transition_loss(
            world,
            admit_terminal_batch(terminal_batch),
            terminal_model_rng,
            config,
            step=step,
        )
        objective = combine_strata(
            ordinary, terminal, config.terminal_loss_mass
        )
        main_values.append(float(ordinary.detach()))
        terminal_values.append(float(terminal.detach()))
        terminal_rows += 1
        _update(
            optimiser,
            _balance({"dynamics": objective}, balance, config),
            [world],
            config,
            step,
        )
        completed = step + 1
        if completed % 500 == 0:
            curve.append(
                {
                    "step": completed,
                    "main_raw_mean": sum(main_values) / len(main_values),
                    "terminal_raw_mean": sum(terminal_values) / len(terminal_values),
                    "dynamics_rms": balance["dynamics"] ** 0.5,
                }
            )
            main_values.clear()
            terminal_values.clear()
            print(
                f"{contract['cell']['name']} {completed}/{steps} "
                f"main={curve[-1]['main_raw_mean']:.6f} "
                f"tail={curve[-1]['terminal_raw_mean']:.6f}",
                flush=True,
            )
        if completed in milestones:
            milestone_digests[str(completed)] = state_digest(world)
            save(
                out / f"world_{completed:06d}.pt",
                config,
                part0=world,
                experiment=contract,
                step=completed,
            )
        if completed % config.checkpoint_every == 0 or completed == steps:
            save(
                checkpoint,
                config,
                part0=world,
                part1=optimiser,
                balance=balance,
                streams=generator_state(
                    sampler=sampler,
                    model=model_rng,
                    terminal_model=terminal_model_rng,
                ),
                meta={
                    "contract": contract,
                    "step": completed,
                    "initial_world_sha256": initial_digest,
                    "curve": curve,
                    "milestone_digests": milestone_digests,
                    "terminal_rows": terminal_rows,
                },
            )

    final_streams = generator_state(
        sampler=sampler, model=model_rng, terminal_model=terminal_model_rng
    )
    stream_digests = {
        name: tensor_digest(value) for name, value in final_streams.items()
    }
    ordinary_match = (
        stream_digests["sampler"] == reference["sampler_sha256"]
        and stream_digests["model"] == reference["model_sha256"]
    )
    if not ordinary_match:
        raise AssertionError("ordinary Phase-1B streams diverged from production")
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_digest,
        "milestone_world_sha256": milestone_digests,
        "stream_sha256": stream_digests,
        "ordinary_streams_match_reference": ordinary_match,
        "terminal_rows_scored": terminal_rows,
        "training_curve": curve,
    }


def stratum_counts(selected: list[int], metadata: dict[int, dict]) -> dict:
    counts = Counter(
        (metadata[index]["pool"], metadata[index]["fatal_action"])
        for index in selected
    )
    return {
        f"{pool}:action_{action}": counts[pool, action]
        for pool in ("expert", "support")
        for action in range(Config().n_actions)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--reference-phase1b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000)
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(32, 96, 192, 300))
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()

    milestones = tuple(sorted(set(args.milestones)))
    sizes = tuple(sorted(set(args.sizes)))
    if milestones[-1] != args.steps:
        parser.error("the final step must be a milestone")
    if args.replicates < 1:
        parser.error("at least one subset replicate is required")
    config = Config(transition="direct", time_mixer="attention")
    reference = reference_contract(args.reference_phase1b, config, args.steps)
    encoder, episodes = cached_train(args.phase1a, Config(), args.expert)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metadata = terminal_metadata(episodes, config)
    if sizes[-1] != len(metadata) or sizes[0] < 2:
        parser.error(
            f"sizes must end at the common terminal universe {len(metadata)}"
        )

    common = {
        "version": "terminal-diversity-scaling-v1",
        "phase1a": file_digest(args.phase1a),
        "reference_phase1b": file_digest(args.reference_phase1b),
        "data": data_digests(),
        "implementation": implementation_digests(
            Path(__file__), Path("artifacts/train_phase1b_geometry_factorial.py")
        ),
        "steps": args.steps,
        "milestones": milestones,
        "terminal_mass": config.terminal_loss_mass,
        "terminal_draws": args.steps,
        "candidate_universe": len(metadata),
        "candidate_rule": (
            "TRAIN uniform-eligible episode with one final terminal and at least "
            "64 observations; the same universe is usable at every schedule step"
        ),
        "scope": (
            "the subset controls only explicit tail-aligned terminal exposure; "
            "the ordinary production sampler retains the complete TRAIN corpus"
        ),
        "ranking": "nested proportional pool-by-fatal-action deficit ordering",
        "schedule": "randomized balanced cycles; exposure differs by at most one draw",
    }
    reports, selections = {}, {}
    for replicate in range(args.replicates):
        ranking_seed = config.seed + 6200 + replicate
        schedule_seed = config.seed + 6300 + replicate
        ranking = stratified_terminal_ranking(metadata, ranking_seed)
        for size in sizes:
            if size == sizes[-1] and replicate > 0:
                continue
            selected = ranking[:size]
            schedule = balanced_terminal_schedule(
                selected, args.steps, schedule_seed
            )
            name = f"k{size:04d}_r{replicate}"
            counts = Counter(schedule.tolist())
            contract = common | {
                "cell": {
                    "name": name,
                    "unique_terminal_episodes": size,
                    "replicate": replicate,
                },
                "ranking_seed": ranking_seed,
                "schedule_seed": schedule_seed,
                "selected_episode_indices": selected,
                "selected_strata": stratum_counts(selected, metadata),
                "schedule_sha256": tensor_digest(schedule),
                "draws_per_episode": {
                    "minimum": min(counts.values()),
                    "maximum": max(counts.values()),
                },
            }
            cell_out = args.out / name
            report = train_cell(
                episodes,
                schedule,
                config,
                steps=args.steps,
                milestones=milestones,
                reference=reference,
                checkpoint=cell_out / "train.pt",
                out=cell_out,
                contract=contract,
            )
            atomic_json(cell_out / "training_report.json", report)
            reports[name], selections[name] = report, selected
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for replicate in range(args.replicates):
        names = [f"k{size:04d}_r{replicate}" for size in sizes[:-1]]
        for smaller, larger in zip(names, names[1:]):
            if not set(selections[smaller]).issubset(selections[larger]):
                raise AssertionError("terminal subsets are not nested")
    initial = {report["initial_world_sha256"] for report in reports.values()}
    ordinary = {
        (report["stream_sha256"]["sampler"], report["stream_sha256"]["model"])
        for report in reports.values()
    }
    invariants = {
        "same_initialization": len(initial) == 1,
        "same_ordinary_streams": len(ordinary) == 1,
        "ordinary_streams_match_reference": all(
            report["ordinary_streams_match_reference"] for report in reports.values()
        ),
        "fixed_terminal_draws": all(
            report["terminal_rows_scored"] == args.steps for report in reports.values()
        ),
        "nested_within_replicate": True,
    }
    if not all(invariants.values()):
        raise AssertionError(f"terminal-diversity matching failed: {invariants}")
    summary = {
        "version": "terminal-diversity-scaling-training-v1",
        "common": common,
        "cells": list(reports),
        "invariants": invariants,
        "reports": {
            name: str((args.out / name / "training_report.json").resolve())
            for name in reports
        },
    }
    atomic_json(args.out / "training_report.json", summary)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
