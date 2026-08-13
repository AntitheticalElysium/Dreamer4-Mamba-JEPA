"""One matched cell of the Phase-1B loss-geometry by sampling factorial."""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import asdict, replace
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
from artifacts.phase1b_geometry_common import (
    direct_metric_loss,
    precision_from_covariance,
)
from d4mj.checkpoint import FORMAT, load, save
from d4mj.config import Config
from d4mj.data import sample_batch, sample_terminal_batch
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


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def reference_contract(path: Path, config: Config, steps: int) -> dict:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if payload["format"] != FORMAT or payload["config"] != asdict(config):
        raise ValueError("reference is not a matching Direct-Attention checkpoint")
    modules = payload["modules"]
    if modules.get("contract") != f"1B:{steps}" or modules.get("step") != steps:
        raise ValueError("reference is not the completed production Phase-1B run")
    return {
        "world_sha256": stored_state_digest(modules["part0"]),
        "sampler_sha256": tensor_digest(modules["generators"]["sampler"]),
        "model_sha256": tensor_digest(modules["generators"]["model"]),
    }


def dynamics_loss(
    metric: str,
    world: World,
    batch,
    rng: torch.Generator,
    config: Config,
    precision: torch.Tensor,
) -> torch.Tensor:
    if metric == "ordinary":
        return transition_loss(world, batch, rng, config)
    if metric == "whitened":
        return direct_metric_loss(world, batch, rng, config, precision)
    raise ValueError(f"unknown metric: {metric}")


def admit_terminal_batch(batch):
    if batch.support is None or not bool(batch.support.all()):
        raise ValueError("terminal dynamics requires an all-support tail batch")
    return replace(batch, support=None)


def combine_strata(
    ordinary: torch.Tensor, terminal: torch.Tensor, terminal_mass: float
) -> torch.Tensor:
    if not 0.0 < terminal_mass < 1.0:
        raise ValueError("terminal mass must be in (0, 1)")
    return (1.0 - terminal_mass) * ordinary + terminal_mass * terminal


def train(
    episodes,
    precision: torch.Tensor,
    config: Config,
    *,
    metric: str,
    sampling: str,
    steps: int,
    milestones: tuple[int, ...],
    terminal_mass: float,
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
    terminal_sampler = torch.Generator().manual_seed(config.seed + 6101)
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
            raise ValueError("factorial checkpoint contract changed")
        if meta.get("initial_world_sha256") != initial_digest:
            raise ValueError("factorial initialization changed")
        resume = int(meta["step"])
        curve = list(meta.get("curve", []))
        milestone_digests = dict(meta.get("milestone_digests", {}))
        terminal_rows = int(meta.get("terminal_rows", 0))
        sampler.set_state(streams["sampler"])
        model_rng.set_state(streams["model"])
        terminal_sampler.set_state(streams["terminal_sampler"])
        terminal_model_rng.set_state(streams["terminal_model"])

    out.mkdir(parents=True, exist_ok=True)
    main_values: list[float] = []
    terminal_values: list[float] = []
    for step in range(resume, steps):
        main = _to(
            sample_batch(episodes, sampler, config, step, steps), config.device
        )
        terminal = _to(
            sample_terminal_batch(
                episodes, terminal_sampler, config, step, steps
            ),
            config.device,
        )
        if not bool(terminal.terminated[:, -1].all()):
            raise AssertionError("a terminal stratum row is not tail-aligned")

        ordinary = dynamics_loss(
            metric, world, main, model_rng, config, precision
        )
        objective = ordinary
        main_values.append(float(ordinary.detach()))
        if sampling == "terminal":
            admitted = admit_terminal_batch(terminal)
            tail = dynamics_loss(
                metric,
                world,
                admitted,
                terminal_model_rng,
                config,
                precision,
            )
            objective = combine_strata(ordinary, tail, terminal_mass)
            terminal_values.append(float(tail.detach()))
            terminal_rows += terminal.led_to_action.shape[0]

        _update(
            optimiser,
            _balance({"dynamics": objective}, balance, config),
            [world],
            config,
            step,
        )
        completed = step + 1
        if completed % 500 == 0:
            row = {
                "step": completed,
                "main_raw_mean": sum(main_values) / len(main_values),
                "terminal_raw_mean": (
                    sum(terminal_values) / len(terminal_values)
                    if terminal_values
                    else None
                ),
                "dynamics_rms": balance["dynamics"] ** 0.5,
            }
            curve.append(row)
            main_values.clear()
            terminal_values.clear()
            print(
                f"{metric}/{sampling} {completed}/{steps} "
                f"main={row['main_raw_mean']:.6f} "
                f"tail={row['terminal_raw_mean']}",
                flush=True,
            )

        if completed in milestones:
            digest = state_digest(world)
            milestone_digests[str(completed)] = digest
            save(
                out / f"world_{completed:06d}.pt",
                config,
                part0=world,
                experiment=contract,
                step=completed,
            )

        if completed % config.checkpoint_every == 0 or completed == steps:
            current_streams = generator_state(
                sampler=sampler,
                model=model_rng,
                terminal_sampler=terminal_sampler,
                terminal_model=terminal_model_rng,
            )
            meta = {
                "contract": contract,
                "step": completed,
                "initial_world_sha256": initial_digest,
                "curve": curve,
                "milestone_digests": milestone_digests,
                "terminal_rows": terminal_rows,
            }
            save(
                checkpoint,
                config,
                part0=world,
                part1=optimiser,
                balance=balance,
                streams=current_streams,
                meta=meta,
            )

    final_streams = generator_state(
        sampler=sampler,
        model=model_rng,
        terminal_sampler=terminal_sampler,
        terminal_model=terminal_model_rng,
    )
    stream_digests = {
        name: tensor_digest(state) for name, state in final_streams.items()
    }
    ordinary_streams_match = (
        stream_digests["sampler"] == reference["sampler_sha256"]
        and stream_digests["model"] == reference["model_sha256"]
    )
    if steps == contract["steps"] and not ordinary_streams_match:
        raise AssertionError("ordinary Phase-1B streams diverged from the control")
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_digest,
        "milestone_world_sha256": milestone_digests,
        "stream_sha256": stream_digests,
        "reference": reference,
        "ordinary_streams_match_reference": ordinary_streams_match,
        "terminal_rows_scored": terminal_rows,
        "training_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--reference-phase1b", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cell",
        nargs=2,
        action="append",
        metavar=("METRIC", "SAMPLING"),
        required=True,
    )
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 20_000))
    parser.add_argument("--shrinkage", type=float, default=0.01)
    parser.add_argument("--condition-limit", type=float, default=10_000.0)
    args = parser.parse_args()

    allowed = {"ordinary", "whitened"}, {"ordinary", "terminal"}
    cells = [(metric, sampling) for metric, sampling in args.cell]
    if len(cells) != len(set(cells)):
        parser.error("factorial cells must be unique")
    for metric, sampling in cells:
        if metric not in allowed[0] or sampling not in allowed[1]:
            parser.error(f"invalid factorial cell: {metric}/{sampling}")

    milestones = tuple(sorted(set(args.milestones)))
    if not milestones or milestones[-1] != args.steps:
        parser.error("the final step must be a milestone")
    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    precision, precision_report = precision_from_covariance(
        prepared["covariance"], args.shrinkage
    )
    precision_report.pop("covariance")
    if precision_report["regularized_condition"] > args.condition_limit:
        parser.error("regularized covariance exceeds the condition-number limit")

    config = Config(transition="direct", time_mixer="attention")
    reference = reference_contract(args.reference_phase1b, config, args.steps)
    encoder, episodes = cached_train(args.phase1a, Config(), args.expert)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    common = {
        "phase1a": file_digest(args.phase1a),
        "reference_phase1b": file_digest(args.reference_phase1b),
        "prepared": file_digest(args.prepared),
        "data": data_digests(),
        "implementation": implementation_digests(
            Path(__file__), Path("artifacts/phase1b_geometry_common.py")
        ),
        "steps": args.steps,
        "milestones": milestones,
        "precision": precision_report,
        "condition_limit": args.condition_limit,
        "terminal_seed": config.seed + 6101,
        "invariants": (
            "same initialization, ordinary sampler/model streams, optimizer, "
            "schedule and Phase-1A cache; terminal tails use independent fixed "
            "streams; terminal mass 1/5 gives each of four ordinary rows and one "
            "tail row equal sequence weight"
        ),
    }
    precision = precision.to(config.device)
    reports = {}
    for metric, sampling in cells:
        terminal_mass = (
            config.terminal_loss_mass if sampling == "terminal" else 0.0
        )
        cell_name = f"{metric}_{sampling}"
        cell_out = args.out / cell_name
        contract = {
            "version": "phase1b-geometry-factorial-v1",
            "cell": {"metric": metric, "sampling": sampling},
            **common,
            "terminal_mass": terminal_mass,
            "terminal_successor_raw_coefficient": {
                "short": terminal_mass
                * (1.0 / (config.sequence - 1) + 1.0 / 2.0),
                "long": terminal_mass
                * (1.0 / (config.sequence_long - 1) + 1.0 / 2.0),
            },
        }
        report = train(
            episodes,
            precision,
            config,
            metric=metric,
            sampling=sampling,
            steps=args.steps,
            milestones=milestones,
            terminal_mass=terminal_mass,
            reference=reference,
            checkpoint=cell_out / "train.pt",
            out=cell_out,
            contract=contract,
        )
        atomic_json(cell_out / "training_report.json", report)
        reports[cell_name] = report
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    initial = {report["initial_world_sha256"] for report in reports.values()}
    ordinary_streams = {
        (report["stream_sha256"]["sampler"], report["stream_sha256"]["model"])
        for report in reports.values()
    }
    terminal_streams = {
        (
            report["stream_sha256"]["terminal_sampler"],
            report["stream_sha256"]["terminal_model"],
        )
        for report in reports.values()
    }
    invariants = {
        "same_initialization": len(initial) == 1,
        "same_ordinary_streams": len(ordinary_streams) == 1,
        "same_terminal_streams": len(terminal_streams) == 1,
        "ordinary_streams_match_reference": all(
            report["ordinary_streams_match_reference"]
            for report in reports.values()
        ),
    }
    if not all(invariants.values()):
        raise AssertionError(f"factorial matching failed: {invariants}")
    summary = {
        "version": "phase1b-geometry-factorial-training-v1",
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
