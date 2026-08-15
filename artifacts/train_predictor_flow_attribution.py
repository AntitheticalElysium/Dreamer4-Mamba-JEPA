"""Train the matched Direct-topology and Flow-diversity cells."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
    state_digest,
    stored_state_digest,
)
from artifacts.predictor_flow_attribution_common import (
    PREDICTORS,
    SOURCE_FILES,
    make_world,
    parameter_report,
    shared_state_digest,
)
from artifacts.train_phase1b_geometry_factorial import (
    admit_terminal_batch,
    combine_strata,
    tensor_digest,
)
from artifacts.train_terminal_diversity_scaling import (
    balanced_terminal_schedule,
    stratified_terminal_ranking,
    stratum_counts,
    terminal_metadata,
    terminal_tail_batch,
)
from d4mj.checkpoint import FORMAT, load, save
from d4mj.config import Config
from d4mj.data import sample_batch
from d4mj.train import (
    _balance,
    _generators,
    _share_initialisation,
    _to,
    _update,
    generator_state,
    optimizer,
)
from d4mj.transition import transition_loss


RELEVANT_CONTROL_FILES = (
    "d4mj/config.py",
    "d4mj/data.py",
    "d4mj/train.py",
    "d4mj/transition.py",
    "artifacts/train_terminal_diversity_scaling.py",
    "artifacts/train_phase1b_geometry_factorial.py",
)


def completed_reference(path: Path, config: Config, steps: int) -> dict:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    modules = payload.get("modules", {})
    if payload.get("format") != FORMAT or payload.get("config") != asdict(config):
        raise ValueError(f"reference config mismatch: {path}")
    if modules.get("contract") != f"1B:{steps}" or modules.get("step") != steps:
        raise ValueError(f"reference is not a completed {steps}-step Phase 1B: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "world_sha256": stored_state_digest(modules["part0"]),
    }


def validate_direct_control(root: Path, config: Config, steps: int) -> dict:
    report_path = root / "training_report.json"
    report = json.loads(report_path.read_text())
    contract = report["contract"]
    if contract["cell"]["unique_terminal_episodes"] != contract["candidate_universe"]:
        raise ValueError("Direct control is not the full-diversity endpoint")
    if contract["steps"] != steps:
        raise ValueError("Direct control step count differs")
    for name in RELEVANT_CONTROL_FILES:
        if contract["implementation"].get(name) != file_digest(Path(name)):
            raise ValueError(f"Direct control implementation changed: {name}")

    torch.manual_seed(config.seed + 1)
    initial = _share_initialisation(make_world(config, "current"), config)
    if state_digest(initial) != report["initial_world_sha256"]:
        raise ValueError("Direct control initialization no longer reproduces")
    milestones = {}
    for step in contract["milestones"]:
        path = root / f"world_{step:06d}.pt"
        if not path.exists():
            raise ValueError(f"missing Direct control milestone: {path}")
        milestones[str(step)] = str(path.resolve())
    return {
        "report": str(report_path.resolve()),
        "report_sha256": file_digest(report_path),
        "contract": contract,
        "initial_world_sha256": report["initial_world_sha256"],
        "shared_initial_sha256": shared_state_digest(initial),
        "stream_sha256": report["stream_sha256"],
        "milestones": milestones,
        "selected_episode_indices": contract["selected_episode_indices"],
        "schedule_sha256": contract["schedule_sha256"],
    }


def train_cell(
    episodes,
    schedule: torch.Tensor,
    config: Config,
    *,
    predictor: str,
    steps: int,
    milestones: tuple[int, ...],
    checkpoint: Path,
    out: Path,
    contract: dict,
) -> dict:
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(make_world(config, predictor), config).to(config.device)
    initial_digest = state_digest(world)
    shared_digest = shared_state_digest(world)
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
            raise ValueError("attribution checkpoint contract changed")
        if meta.get("initial_world_sha256") != initial_digest:
            raise ValueError("attribution initialization changed")
        resume = int(meta["step"])
        curve = list(meta.get("curve", []))
        milestone_digests = dict(meta.get("milestone_digests", {}))
        sampler.set_state(streams["sampler"])
        model_rng.set_state(streams["model"])
        terminal_model_rng.set_state(streams["terminal_model"])

    out.mkdir(parents=True, exist_ok=True)
    main_values: list[float] = []
    terminal_values: list[float] = []
    for step in range(resume, steps):
        ordinary_batch = _to(
            sample_batch(episodes, sampler, config, step, steps), config.device
        )
        tail_batch = _to(
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
            admit_terminal_batch(tail_batch),
            terminal_model_rng,
            config,
            step=step,
        )
        objective = combine_strata(
            ordinary, terminal, config.terminal_loss_mass
        )
        main_values.append(float(ordinary.detach()))
        terminal_values.append(float(terminal.detach()))
        _update(
            optimiser,
            _balance({"dynamics": objective}, balance, config),
            [world],
            config,
            step,
        )
        completed = step + 1
        report_every = min(500, steps)
        if completed % report_every == 0 or completed == steps:
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
                    "shared_initial_sha256": shared_digest,
                    "curve": curve,
                    "milestone_digests": milestone_digests,
                },
            )

    final_streams = generator_state(
        sampler=sampler, model=model_rng, terminal_model=terminal_model_rng
    )
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_digest,
        "shared_initial_sha256": shared_digest,
        "milestone_world_sha256": milestone_digests,
        "stream_sha256": {
            name: tensor_digest(value) for name, value in final_streams.items()
        },
        "terminal_rows_scored": steps,
        "training_curve": curve,
    }


def cell_contract(
    common: dict,
    name: str,
    transition: str,
    predictor: str,
    selected: list[int],
    metadata: dict,
    schedule: torch.Tensor,
    ranking_seed: int,
    schedule_seed: int,
) -> dict:
    counts = Counter(schedule.tolist())
    return common | {
        "cell": {
            "name": name,
            "transition": transition,
            "predictor": predictor,
            "unique_terminal_episodes": len(selected),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--direct-reference", type=Path, required=True)
    parser.add_argument("--flow-reference", type=Path, required=True)
    parser.add_argument("--direct-control", type=Path)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000)
    )
    parser.add_argument("--flow-small", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    milestones = tuple(sorted(set(args.milestones)))
    if milestones[-1] != args.steps:
        parser.error("the final step must be a milestone")
    if args.smoke and args.direct_control is not None:
        parser.error("smoke mode trains its own current control")

    direct_config = Config(transition="direct", time_mixer="attention")
    flow_config = Config(transition="flow", time_mixer="attention")
    references = {
        "direct": completed_reference(
            args.direct_reference, direct_config, 20_000
        ),
        "flow": completed_reference(args.flow_reference, flow_config, 20_000),
    }
    encoder, episodes = cached_train(
        args.phase1a,
        Config(),
        args.expert,
        support=args.support,
        cache=args.cache,
    )
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = terminal_metadata(episodes, direct_config)
    ranking_seed = direct_config.seed + 6200
    schedule_seed = direct_config.seed + 6300
    ranking = stratified_terminal_ranking(metadata, ranking_seed)
    full = len(ranking)
    flow_small = 1 if args.smoke else args.flow_small
    full_used = 2 if args.smoke else full
    if flow_small >= full_used or full_used > full:
        parser.error("Flow diversity endpoints are outside the terminal universe")

    parameters = parameter_report(direct_config)
    relative_capacity_gap = abs(
        parameters["deep_mlp"]["predictor"]
        - parameters["token_transformer"]["predictor"]
    ) / parameters["token_transformer"]["predictor"]
    if relative_capacity_gap > 0.05:
        raise AssertionError("capacity control is not matched to the token predictor")

    common = {
        "version": "predictor-flow-attribution-training-v1",
        "phase1a": file_digest(args.phase1a),
        "references": references,
        "data": data_digests(args.support),
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/predictor_flow_attribution_common.py"),
            Path("artifacts/train_terminal_diversity_scaling.py"),
            Path("artifacts/train_phase1b_geometry_factorial.py"),
        ),
        "source_files": {str(path): file_digest(path) for path in SOURCE_FILES},
        "online_corroboration": {
            "dino_wm": (
                "https://github.com/gaoyuezhou/dino_wm/blob/main/"
                "models/visual_world_model.py"
            ),
            "role": "corroboration only; the pinned V-JEPA 2-AC snapshot is the code source",
        },
        "steps": args.steps,
        "milestones": milestones,
        "terminal_mass": direct_config.terminal_loss_mass,
        "terminal_draws": args.steps,
        "candidate_universe": full,
        "parameters": parameters,
        "capacity_relative_gap": relative_capacity_gap,
        "fixed_controls": (
            "same encoder latents, ordinary sampler, tail exposure, initialization "
            "of every shared tensor, squared latent loss, teacher forcing, two-step "
            "Direct rollout, optimizer and milestones"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}
    selections: dict[str, list[int]] = {}
    direct_anchor = None
    direct_shared = None
    if args.direct_control is not None:
        control = validate_direct_control(args.direct_control, direct_config, args.steps)
        expected = ranking
        expected_schedule = balanced_terminal_schedule(
            expected, args.steps, schedule_seed
        )
        if control["selected_episode_indices"] != expected:
            raise ValueError("Direct control terminal selection differs")
        if control["schedule_sha256"] != tensor_digest(expected_schedule):
            raise ValueError("Direct control terminal schedule differs")
        name = "direct_current"
        reports[name] = {"external_control": control}
        selections[name] = expected
        direct_anchor = control["stream_sha256"]
        direct_shared = control["shared_initial_sha256"]

    direct_predictors = PREDICTORS if args.direct_control is None else PREDICTORS[1:]
    direct_selected = ranking[:full_used]
    direct_schedule = balanced_terminal_schedule(
        direct_selected, args.steps, schedule_seed
    )
    for predictor in direct_predictors:
        name = f"direct_{predictor}"
        contract = cell_contract(
            common,
            name,
            "direct",
            predictor,
            direct_selected,
            metadata,
            direct_schedule,
            ranking_seed,
            schedule_seed,
        )
        cell_out = args.out / name
        report = train_cell(
            episodes,
            direct_schedule,
            direct_config,
            predictor=predictor,
            steps=args.steps,
            milestones=milestones,
            checkpoint=cell_out / "train.pt",
            out=cell_out,
            contract=contract,
        )
        atomic_json(cell_out / "training_report.json", report)
        reports[name], selections[name] = report, direct_selected
        if direct_anchor is None:
            direct_anchor = report["stream_sha256"]
            direct_shared = report["shared_initial_sha256"]
        if report["stream_sha256"] != direct_anchor:
            raise AssertionError("Direct data/model streams differ across topologies")
        if report["shared_initial_sha256"] != direct_shared:
            raise AssertionError("Direct shared initialization differs across topologies")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flow_reports = []
    flow_anchor = None
    flow_initial = None
    for size in (flow_small, full_used):
        name = f"flow_k{size:04d}"
        selected = ranking[:size]
        schedule = balanced_terminal_schedule(selected, args.steps, schedule_seed)
        contract = cell_contract(
            common,
            name,
            "flow",
            "current",
            selected,
            metadata,
            schedule,
            ranking_seed,
            schedule_seed,
        )
        cell_out = args.out / name
        report = train_cell(
            episodes,
            schedule,
            flow_config,
            predictor="current",
            steps=args.steps,
            milestones=milestones,
            checkpoint=cell_out / "train.pt",
            out=cell_out,
            contract=contract,
        )
        atomic_json(cell_out / "training_report.json", report)
        reports[name], selections[name] = report, selected
        flow_reports.append(name)
        if flow_anchor is None:
            flow_anchor = report["stream_sha256"]
            flow_initial = report["initial_world_sha256"]
        if report["stream_sha256"] != flow_anchor:
            raise AssertionError("Flow streams differ across diversity cells")
        if report["initial_world_sha256"] != flow_initial:
            raise AssertionError("Flow initialization differs across diversity cells")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not set(selections[flow_reports[0]]).issubset(selections[flow_reports[1]]):
        raise AssertionError("Flow diversity subsets are not nested")
    summary = {
        "version": "predictor-flow-attribution-training-summary-v1",
        "common": common,
        "smoke": args.smoke,
        "cells": list(reports),
        "reports": reports,
        "invariants": {
            "direct_same_shared_initialization": True,
            "direct_same_streams": True,
            "direct_fixed_full_terminal_exposure": True,
            "capacity_control_within_five_percent": True,
            "flow_same_initialization": True,
            "flow_same_streams": True,
            "flow_nested_terminal_subsets": True,
            "flow_fixed_terminal_draws": True,
        },
    }
    atomic_json(args.out / "training_report.json", summary)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
