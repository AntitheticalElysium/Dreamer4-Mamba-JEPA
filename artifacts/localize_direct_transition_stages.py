"""Locate when Direct loses action-conditioned fatality information."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from artifacts.localize_counterfactual import (
    ForkData,
    binary_metrics,
    fit_linear_once,
    flatten,
    latent_error,
    load_models,
)
from artifacts.localize_counterfactual_interaction import (
    action_means,
    report_score,
)
from artifacts.localize_matched_counterfactual import extract_matched_forks
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World

ROOT = Path(__file__).resolve().parent.parent


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _implementation() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "artifacts" / "localize_counterfactual.py",
        ROOT / "artifacts" / "localize_counterfactual_interaction.py",
        ROOT / "artifacts" / "localize_matched_counterfactual.py",
        ROOT / "d4mj" / "agent.py",
        ROOT / "d4mj" / "checkpoint.py",
        ROOT / "d4mj" / "env.py",
        ROOT / "d4mj" / "representation.py",
        ROOT / "d4mj" / "transition.py",
    )
    return {str(path.relative_to(ROOT)): _digest(path) for path in paths}


def _load_stage(path: Path, world_only: bool, config: Config) -> tuple[World, Heads]:
    torch.manual_seed(config.seed + 2)
    world = World(config).to(config.device)
    heads = Heads(config).to(config.device)
    objects = {"part0": world} if world_only else {"part0": world, "part1": heads}
    load(path, config, **objects)
    for module in (world, heads):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return world, heads


def _same_examples(reference: ForkData, candidate: ForkData) -> None:
    for field in ("target", "action", "group"):
        if not torch.equal(getattr(reference, field), getattr(candidate, field)):
            raise AssertionError(f"stage changed fork {field}")


def _reference_error(data: ForkData, path: Path) -> dict[str, float]:
    expected = torch.load(path, weights_only=False)
    errors = {}
    for field in (
        "target",
        "action",
        "group",
        "observed_latent",
        "generated_latent",
        "observed_readout",
        "generated_readout",
    ):
        current = getattr(data, field)
        recorded = expected[field]
        if current.dtype in (torch.long, torch.bool):
            if not torch.equal(current, recorded):
                raise AssertionError(f"paired reference {field} changed")
            errors[field] = 0.0
        else:
            error = float((current - recorded).abs().max())
            if error > 1e-5:
                raise AssertionError(f"paired reference {field} changed: {error:.3e}")
            errors[field] = error
    return errors


def _resumable_linear_probe(
    feature: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
    group: torch.Tensor,
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    checkpoint: Path,
    contract: dict,
) -> torch.Tensor:
    """Existing LO-state-out probe, checkpointed after every completed fold."""
    feature = flatten(feature).cpu()
    target, action, group = target.cpu(), action.cpu(), group.cpu()
    groups = sorted(set(group.tolist()))
    prediction = torch.zeros_like(target)
    complete: set[int] = set()

    if checkpoint.exists():
        stored = torch.load(checkpoint, weights_only=False)
        if stored["contract"] != contract:
            raise ValueError(f"probe contract changed: {checkpoint}")
        prediction.copy_(stored["prediction"])
        complete = set(stored["complete"])

    for test_group in groups:
        if test_group in complete:
            continue
        test_mask = group == test_group
        remaining = [value for value in groups if value != test_group]
        seed_predictions = []

        for seed_index, seed in enumerate(seeds):
            val_group = remaining[seed_index % len(remaining)]
            fit_mask = (group != test_group) & (group != val_group)
            val_mask = group == val_group
            means = action_means(feature, action, fit_mask, config.n_actions)
            centered = feature - means[action]
            seed_predictions.append(
                fit_linear_once(
                    centered[fit_mask],
                    target[fit_mask],
                    centered[val_mask],
                    target[val_mask],
                    centered[test_mask],
                    seed=seed + test_group * 101,
                    device=config.device,
                    steps=steps,
                    lr=3e-3,
                    weight_decay=1e-3,
                )
            )

        prediction[test_mask] = torch.stack(seed_predictions).mean(0)
        complete.add(test_group)
        _atomic_torch(
            checkpoint,
            {
                "contract": contract,
                "prediction": prediction,
                "complete": sorted(complete),
            },
        )
        print(
            f"{contract['stage']} {contract['feature']}: "
            f"{len(complete)}/{len(groups)} folds",
            flush=True,
        )
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--phase1b", type=Path, required=True)
    parser.add_argument(
        "--phase2",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--reference-stage", required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stages = [("phase1b", args.phase1b, True)] + [
        (name, Path(path), False) for name, path in args.phase2
    ]
    names = [name for name, _, _ in stages]
    if len(names) != len(set(names)):
        parser.error("stage names must be unique")
    if args.reference_stage not in names:
        parser.error("--reference-stage must name one supplied stage")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("stage names may contain lowercase letters, digits, _ and -")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "features").mkdir(exist_ok=True)
    (args.out / "probes").mkdir(exist_ok=True)
    inputs = {
        "phase1a": _digest(args.phase1a),
        "trajectory_phase2": _digest(args.trajectory_phase2),
        "forks": _digest(args.forks),
        "reference_features": _digest(args.reference_features),
        "stages": {
            name: {"sha256": _digest(path), "world_only": world_only}
            for name, path, world_only in stages
        },
    }
    contract = {
        "version": "direct-transition-stages-v1",
        "inputs": inputs,
        "implementation": _implementation(),
        "evaluation": "same 100 terminal-opportunity states and all 17 actions",
        "probe": "action-centered leave-one-pre-action-state-out linear",
        "seeds": args.seeds,
        "linear_steps": args.linear_steps,
        "permutations": args.permutations,
    }
    contract_path = args.out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("transition-stage experiment contract changed")
    else:
        _atomic_json(contract_path, contract)

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    saved = torch.load(args.forks, weights_only=False)
    extracted: dict[str, tuple[ForkData, dict]] = {}
    reference_data = None

    for name, path, world_only in stages:
        feature_path = args.out / "features" / f"{name}.pt"
        stage_contract = {
            "experiment": contract["version"],
            "implementation": contract["implementation"],
            "phase1a": inputs["phase1a"],
            "trajectory_phase2": inputs["trajectory_phase2"],
            "forks": inputs["forks"],
            "stage": name,
            "checkpoint": inputs["stages"][name],
        }
        if feature_path.exists():
            payload = torch.load(feature_path, weights_only=False)
            if payload["contract"] != stage_contract:
                raise ValueError(f"feature contract changed: {feature_path}")
            data = ForkData(**payload["data"])
            replay = payload["replay"]
        else:
            print(f"extracting matched forks: {name}", flush=True)
            world, heads = _load_stage(path, world_only, config)
            data, replay = extract_matched_forks(
                saved,
                encoder,
                trajectory_world,
                trajectory_heads,
                world,
                heads,
                config,
                config,
            )
            world.cpu()
            heads.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _atomic_torch(
                feature_path,
                {"contract": stage_contract, "data": vars(data), "replay": replay},
            )

        if reference_data is None:
            reference_data = data
        else:
            _same_examples(reference_data, data)
        extracted[name] = (data, replay)

    reference_error = _reference_error(
        extracted[args.reference_stage][0], args.reference_features
    )
    encoder.cpu()
    trajectory_world.cpu()
    trajectory_heads.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seeds = [config.seed + 4000 + index for index in range(args.seeds)]
    results = {}
    for stage_index, (name, path, world_only) in enumerate(stages):
        data, replay = extracted[name]
        probes = {}
        for feature_index, feature_name in enumerate(
            ("observed_latent", "generated_latent")
        ):
            probe_contract = {
                "version": contract["version"],
                "stage": name,
                "stage_sha256": inputs["stages"][name]["sha256"],
                "feature": feature_name,
                "seeds": seeds,
                "steps": args.linear_steps,
                "lr": 3e-3,
                "weight_decay": 1e-3,
            }
            probability = _resumable_linear_probe(
                getattr(data, feature_name),
                data.target,
                data.action,
                data.group,
                config,
                seeds=seeds,
                steps=args.linear_steps,
                checkpoint=args.out / "probes" / f"{name}.{feature_name}.pt",
                contract=probe_contract,
            )
            probes[feature_name] = {
                "binary": binary_metrics(probability, data.target),
                "same_action": report_score(
                    probability,
                    data.target,
                    data.action,
                    permutations=args.permutations,
                    seed=config.seed + 7000 + stage_index * 10 + feature_index,
                ),
            }

        observed_auc = probes["observed_latent"]["same_action"]["conditional"][
            "pooled_pair_auc"
        ]
        generated_auc = probes["generated_latent"]["same_action"]["conditional"][
            "pooled_pair_auc"
        ]
        results[name] = {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["stages"][name]["sha256"],
            "world_only": world_only,
            "replay": replay,
            "latent_prediction_error": latent_error(data),
            "probes": probes,
            "observed_minus_generated_same_action_auc": observed_auc - generated_auc,
        }

    baseline_generated = results["phase1b"]["probes"]["generated_latent"][
        "same_action"
    ]["conditional"]["pooled_pair_auc"]
    for name in names:
        current = results[name]["probes"]["generated_latent"]["same_action"][
            "conditional"
        ]["pooled_pair_auc"]
        results[name]["generated_auc_change_from_phase1b"] = (
            current - baseline_generated
        )

    report = {
        "contract": contract,
        "reference_validation": {
            "stage": args.reference_stage,
            "max_abs_error": reference_error,
        },
        "stages": results,
    }
    _atomic_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
