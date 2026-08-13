"""Evaluate Direct worlds on held-out logged terminal/safe archive transitions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import (
    atomic_torch,
    finite_json,
    fit_fatal_direction,
    geometry_metrics,
    predict_records,
    projection_metrics,
)
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World


PATHS = ("reset16", "reset64", "recurrent")


def _load_world(path: Path, config: Config) -> World:
    world = World(config).to(config.device)
    load(path, config, part0=world)
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return world


def _subset(data: dict, mask: torch.Tensor) -> dict:
    return {
        key: value[mask] if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
        if key != "pool"
    }


def score_data(data: dict, prepared: dict, seed: int, bootstraps: int) -> dict:
    pools = sorted(set(data["pool"]))
    output = {
        "combined": geometry_metrics(
            data,
            prepared["direction"],
            prepared["action_means"],
            prepared["covariance"],
            bootstrap_samples=bootstraps,
            bootstrap_seed=seed,
        )
    }
    pool_tensor = data["pool"]
    for index, pool in enumerate(pools):
        mask = torch.tensor([value == pool for value in pool_tensor])
        output[pool] = geometry_metrics(
            _subset(data, mask),
            prepared["direction"],
            prepared["action_means"],
            prepared["covariance"],
            bootstrap_samples=bootstraps,
            bootstrap_seed=seed + 100 * (index + 1),
        )
    return output


def probe_rows(data: dict) -> dict:
    return {
        "feature": data["predicted"],
        "target": data["label"].float(),
        "action": data["action"].long(),
        "group": data["group"].long(),
        "same_action_safe_pairs": torch.tensor(0),
    }


def generated_probe(
    train: dict,
    dev: dict,
    prepared: dict,
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    bootstraps: int,
    seed: int,
) -> dict:
    direction, means, fit_report = fit_fatal_direction(
        probe_rows(train), config, seeds=seeds, steps=steps
    )
    output = {
        "fit": fit_report,
        "combined": projection_metrics(
            dev["predicted"],
            dev["label"],
            dev["action"],
            dev["group"],
            direction,
            means,
            bootstrap_samples=bootstraps,
            bootstrap_seed=seed,
        ),
    }
    for index, pool in enumerate(sorted(set(dev["pool"]))):
        mask = torch.tensor([value == pool for value in dev["pool"]])
        output[pool] = projection_metrics(
            dev["predicted"][mask],
            dev["label"][mask],
            dev["action"][mask],
            dev["group"][mask],
            direction,
            means,
            bootstrap_samples=bootstraps,
            bootstrap_seed=seed + 100 * (index + 1),
        )
    return output


def whitening_gate(report: dict, prepared: dict) -> dict:
    """Predeclared mechanism gate; DEV facts select whether training may launch."""
    required = ("step_005k", "step_020k", "step_080k")
    missing = [name for name in required if name not in report["worlds"]]
    if missing:
        return {"eligible": False, "reason": f"missing required worlds: {missing}"}

    observed = report["observed"]["combined"]
    observed_lower = observed["target_separation"]["auc_ci95"][0]
    low_variance = (
        prepared["report"]["training_geometry"]
        ["direction_variance_relative_to_isotropic"] < 1.0
    )
    path_checks = {}
    for path in ("reset16", "reset64"):
        early = report["worlds"]["step_005k"]["paths"][path]["combined"]
        middle = report["worlds"]["step_020k"]["paths"][path]["combined"]
        late = report["worlds"]["step_080k"]["paths"][path]["combined"]
        archive_gap = (
            middle["predicted_separation"]["auc_ci95"][1] < observed_lower
        )
        generated_probe_gap = (
            report["worlds"]["step_020k"]["generated_probe"][path]
            ["combined"]["auc_ci95"][1] < observed_lower
        )
        total_improves = late["total_mse"] < early["total_mse"]
        orthogonal_improves = (
            late["orthogonal_mse_per_dimension"]
            < early["orthogonal_mse_per_dimension"]
        )
        direction_stagnates = (
            late["direction_mse_over_target_variance"]
            >= early["direction_mse_over_target_variance"]
        )
        correlation_stagnates = (
            late["target_prediction_projection_correlation"]
            <= early["target_prediction_projection_correlation"]
        )
        path_checks[path] = {
            "archive_generated_below_observed": archive_gap,
            "fresh_generated_probe_below_observed": generated_probe_gap,
            "total_mse_improves": total_improves,
            "orthogonal_mse_improves": orthogonal_improves,
            "directional_error_stagnates": direction_stagnates,
            "directional_correlation_stagnates": correlation_stagnates,
            "geometry_signature": all(
                (
                    archive_gap,
                    generated_probe_gap,
                    total_improves,
                    orthogonal_improves,
                    direction_stagnates,
                    correlation_stagnates,
                )
            ),
        }
    observed_decodable = observed_lower > 0.5
    qualifying = [
        path for path, checks in path_checks.items() if checks["geometry_signature"]
    ]
    eligible = observed_decodable and low_variance and bool(qualifying)
    return {
        "eligible": eligible,
        "observed_direction_decodable": observed_decodable,
        "observed_auc_ci95": observed["target_separation"]["auc_ci95"],
        "low_variance_direction": low_variance,
        "direction_variance_relative_to_isotropic": (
            prepared["report"]["training_geometry"]
            ["direction_variance_relative_to_isotropic"]
        ),
        "qualifying_reset_paths": qualifying,
        "path_checks": path_checks,
        "rule": (
            "launch only when the held-out observed direction is decodable, its TRAIN "
            "variance is below the isotropic average, and at least one production "
            "reset stratum has both fixed-direction and fresh-probe gaps, then shows "
            "lower total/orthogonal error but non-improving directional error and "
            "correlation from 5k to 80k"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--world", nargs=2, action="append", metavar=("NAME", "CHECKPOINT"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    worlds = [(name, Path(path)) for name, path in args.world]
    names = [name for name, _ in worlds]
    if len(names) != len(set(names)):
        parser.error("world names must be unique")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("world names may contain lowercase letters, digits, _ and -")
    prepared = torch.load(args.prepared, weights_only=False)
    inputs = {
        "prepared": file_digest(args.prepared),
        "worlds": {name: file_digest(path) for name, path in worlds},
    }
    contract = {
        "version": "phase1b-archive-geometry-evaluation-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__), Path("artifacts/phase1b_geometry_common.py")
        ),
        "paths": list(PATHS),
        "bootstraps": args.bootstraps,
        "probe_direction": "one fixed TRAIN real-successor direction for every world",
        "generated_probe": (
            "fit independently per checkpoint on TRAIN logged transitions; evaluate "
            "only on whole held-out DEV episodes"
        ),
        "evaluation": "whole held-out episodes; expert and epsilon support reported separately",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("archive-geometry evaluation contract changed")
    else:
        atomic_json(contract_path, contract)

    records = prepared["records"]
    reference = None
    results = {}
    config = Config(transition="direct", time_mixer="attention")
    for world_index, (name, path) in enumerate(worlds):
        feature_path = args.out / "features" / f"{name}.pt"
        feature_contract = {
            "evaluation": contract,
            "world": name,
            "checkpoint": inputs["worlds"][name],
        }
        if feature_path.exists():
            cached = torch.load(feature_path, weights_only=False)
            if cached["contract"] != feature_contract:
                raise ValueError(f"archive feature contract changed: {feature_path}")
            path_data = cached["paths"]
            train_path_data = cached["train_paths"]
        else:
            print(f"evaluating logged transitions: {name}", flush=True)
            world = _load_world(path, config)
            path_data = {
                prediction_path: predict_records(
                    world, records, prediction_path, config
                )
                for prediction_path in PATHS
            }
            train_path_data = {
                prediction_path: predict_records(
                    world, prepared["train_records"], prediction_path, config
                )
                for prediction_path in ("reset16", "reset64")
            }
            atomic_torch(
                feature_path,
                {
                    "contract": feature_contract,
                    "paths": path_data,
                    "train_paths": train_path_data,
                },
            )
            world.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for prediction_path, data in path_data.items():
            identity = (data["target"], data["label"], data["action"], data["group"])
            if reference is None:
                reference = identity
            elif not all(torch.equal(first, second) for first, second in zip(reference, identity)):
                raise AssertionError(f"{name}/{prediction_path} changed held-out examples")
        results[name] = {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["worlds"][name],
            "paths": {
                prediction_path: score_data(
                    data,
                    prepared,
                    seed=config.seed + 8500 + world_index * 100 + path_index * 10,
                    bootstraps=args.bootstraps,
                )
                for path_index, (prediction_path, data) in enumerate(path_data.items())
            },
            "generated_probe": {
                prediction_path: generated_probe(
                    train_path_data[prediction_path],
                    path_data[prediction_path],
                    prepared,
                    config,
                    seeds=[
                        config.seed + 8600 + value
                        for value in range(
                            prepared["contract"]["probe_seeds"]
                        )
                    ],
                    steps=prepared["contract"]["probe_steps"],
                    bootstraps=args.bootstraps,
                    seed=config.seed + 8700 + world_index * 100 + path_index * 10,
                )
                for path_index, prediction_path in enumerate(("reset16", "reset64"))
            },
        }

    observed_data = dict(next(iter(next(iter(results.values()))["paths"].values())))
    del observed_data
    first_feature = torch.load(
        args.out / "features" / f"{names[0]}.pt", weights_only=False
    )["paths"][PATHS[0]]
    observed_raw = dict(first_feature)
    observed_raw["predicted"] = observed_raw["target"]
    observed = score_data(
        observed_raw,
        prepared,
        seed=config.seed + 8900,
        bootstraps=args.bootstraps,
    )
    report = finite_json(
        {
            "contract": contract,
            "preparation": prepared["report"],
            "observed": observed,
            "worlds": results,
        }
    )
    gate = whitening_gate(report, prepared)
    report["whitening_gate"] = gate
    atomic_json(args.out / "report.json", report)
    atomic_json(args.out / "whitening_gate.json", gate)
    print(f"whitening eligible: {gate['eligible']}", flush=True)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
