"""Evaluate attribution worlds on fixed held-out logged transitions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from artifacts.evaluate_fatality_direction_delta import archive_current, delta_metrics
from artifacts.evaluate_phase1b_archive_geometry import generated_probe, score_data
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch, finite_json
from artifacts.predictor_flow_attribution_common import load_world
from d4mj.config import Config
from d4mj.state import WorldState
from d4mj.transition import advance, commit_inputs


def _led_to(actions: torch.Tensor, start: int, end: int, config: Config) -> torch.Tensor:
    positions = torch.arange(start, end)
    incoming = positions - 1
    return torch.where(
        incoming >= 0,
        actions[incoming.clamp(min=0)],
        torch.tensor(config.n_actions),
    ).long()


@torch.no_grad()
def predict_record(
    world,
    record: dict,
    config: Config,
    *,
    context: int,
    samples: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    transition = int(record["transitions"][0])
    start = max(0, transition + 1 - context)
    latents = record["latents"]
    actions = record["actions_taken"]
    block = latents[start : transition + 1][None].to(config.device)
    led_to = _led_to(actions, start, transition + 1, config)[None].to(config.device)
    context_rng = torch.Generator(device=config.device).manual_seed(seed)
    committed, conditioning = commit_inputs(block, context_rng, config)
    features, _, memory = world(None, led_to, committed, conditioning)
    action = actions[transition].view(1, 1).to(config.device)
    if config.transition == "direct":
        predicted = world.predict(features[:, -1:], action)[0, 0].cpu()
        return {"reset16": predicted}

    state = WorldState(block[:, -1:], memory, block.shape[1], features[:, -1:])
    predictions = []
    for sample in range(samples):
        rng = torch.Generator(device=config.device).manual_seed(
            seed + 1_000_003 + sample
        )
        successor, _ = advance(world, state, action, rng, config)
        predictions.append(successor.latent[0, 0].cpu())
    stacked = torch.stack(predictions)
    return {
        "reset16_first": stacked[0],
        "reset16_mean": stacked.mean(0),
    }


@torch.no_grad()
def predict_records(
    world,
    records: list[dict],
    config: Config,
    *,
    context: int,
    samples: int,
    label: str = "archive",
) -> dict[str, dict]:
    by_path: dict[str, dict[str, list]] = {}
    for index, record in enumerate(records):
        values = predict_record(
            world,
            record,
            config,
            context=context,
            samples=samples,
            seed=config.seed + 12_000 + index * 101,
        )
        transition = record["transitions"]
        for path, predicted in values.items():
            row = by_path.setdefault(
                path,
                {
                    "predicted": [],
                    "target": [],
                    "label": [],
                    "action": [],
                    "group": [],
                    "pool": [],
                },
            )
            row["predicted"].append(predicted)
            row["target"].append(record["latents"][transition + 1][0])
            row["label"].append(record["labels"][0])
            row["action"].append(record["actions_taken"][transition][0])
            row["group"].append(torch.tensor(record.get("group", index)))
            row["pool"].append(record["pool"])
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"{label}: {index + 1}/{len(records)} records", flush=True)
    packed = {}
    for path, row in by_path.items():
        packed[path] = {
            key: value if key == "pool" else torch.stack(value)
            for key, value in row.items()
        }
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--world",
        nargs=4,
        action="append",
        metavar=("NAME", "TRANSITION", "PREDICTOR", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--flow-samples", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    worlds = [
        (name, transition, predictor, Path(path))
        for name, transition, predictor, path in args.world
    ]
    names = [row[0] for row in worlds]
    if len(names) != len(set(names)):
        parser.error("world names must be unique")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("world names may contain lowercase letters, digits, _ and -")
    if args.flow_samples < 2:
        parser.error("Flow evaluation requires at least two samples")

    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    inputs = {
        "prepared": file_digest(args.prepared),
        "worlds": {name: file_digest(path) for name, _, _, path in worlds},
    }
    contract = {
        "version": "predictor-flow-archive-evaluation-v1",
        "inputs": inputs,
        "specs": {
            name: {"transition": transition, "predictor": predictor}
            for name, transition, predictor, _ in worlds
        },
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/predictor_flow_attribution_common.py"),
            Path("artifacts/evaluate_fatality_direction_delta.py"),
            Path("artifacts/evaluate_phase1b_archive_geometry.py"),
        ),
        "context": args.context,
        "flow_samples": args.flow_samples,
        "flow_randomness": "common sample seeds for every Flow world",
        "bootstraps": args.bootstraps,
        "split": "whole held-out DEV episodes",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("attribution archive contract changed")
    atomic_json(contract_path, contract)

    current = archive_current(prepared["records"])
    results = {}
    reference = None
    for world_index, (name, transition, predictor, path) in enumerate(worlds):
        feature_path = args.out / "features" / f"{name}.pt"
        feature_contract = {
            "contract": contract,
            "world": name,
            "checkpoint": inputs["worlds"][name],
        }
        if feature_path.exists():
            payload = torch.load(feature_path, weights_only=False, map_location="cpu")
            if payload["contract"] != feature_contract:
                raise ValueError(f"archive feature contract changed: {feature_path}")
            dev_paths, train_paths = payload["paths"], payload["train_paths"]
        else:
            print(f"archive extraction: {name}", flush=True)
            config = Config(transition=transition, time_mixer="attention")
            world = load_world(path, config, predictor)
            dev_paths = predict_records(
                world,
                prepared["records"],
                config,
                context=args.context,
                samples=args.flow_samples,
                label=f"{name} DEV",
            )
            train_paths = predict_records(
                world,
                prepared["train_records"],
                config,
                context=args.context,
                samples=args.flow_samples,
                label=f"{name} TRAIN probe",
            )
            atomic_torch(
                feature_path,
                {
                    "contract": feature_contract,
                    "paths": dev_paths,
                    "train_paths": train_paths,
                },
            )
            world.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        config = Config(transition=transition, time_mixer="attention")
        paths = {}
        for path_index, (prediction_path, data) in enumerate(dev_paths.items()):
            identity = (data["target"], data["label"], data["action"], data["group"])
            if reference is None:
                reference = identity
            elif not all(
                torch.equal(first, second)
                for first, second in zip(reference, identity)
            ):
                raise AssertionError(f"{name}/{prediction_path} changed DEV examples")
            scored = score_data(
                data,
                prepared,
                seed=config.seed + 13_000 + world_index * 100 + path_index * 10,
                bootstraps=args.bootstraps,
            )
            fresh = generated_probe(
                train_paths[prediction_path],
                data,
                prepared,
                config,
                seeds=[config.seed + 13_500 + value for value in range(5)],
                steps=prepared["contract"]["probe_steps"],
                bootstraps=args.bootstraps,
                seed=config.seed + 14_000 + world_index * 100 + path_index * 10,
            )
            delta = {}
            for pool_index, pool in enumerate(("combined", "support")):
                mask = torch.ones(len(data["label"]), dtype=torch.bool)
                if pool == "support":
                    mask = torch.tensor([value == pool for value in data["pool"]])
                delta[pool] = delta_metrics(
                    current[mask],
                    data["target"][mask],
                    data["predicted"][mask],
                    data["label"][mask],
                    data["action"][mask],
                    data["group"][mask],
                    prepared["direction"],
                    prepared["action_means"],
                    bootstraps=args.bootstraps,
                    seed=config.seed
                    + 14_500
                    + world_index * 100
                    + path_index * 10
                    + pool_index,
                )
            paths[prediction_path] = {
                "geometry": scored,
                "fresh_probe": fresh,
                "delta": delta,
            }
        results[name] = {
            "transition": transition,
            "predictor": predictor,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["worlds"][name],
            "paths": paths,
        }

    report = finite_json({"contract": contract, "worlds": results})
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
