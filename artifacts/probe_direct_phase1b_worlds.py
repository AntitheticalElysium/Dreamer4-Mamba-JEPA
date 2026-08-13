"""Probe Direct Phase-1B worlds on fixed matched terminal forks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from artifacts.localize_counterfactual import ForkData, binary_metrics, latent_error, load_models
from artifacts.localize_counterfactual_interaction import report_score
from artifacts.localize_direct_transition_stages import _resumable_linear_probe
from artifacts.localize_matched_counterfactual import extract_matched_forks
from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_world(path: Path, config: Config) -> tuple[World, Heads]:
    world = World(config).to(config.device)
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(config.device)
    load(path, config, part0=world)
    for module in (world, heads):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return world, heads


def _same_examples(reference: ForkData, current: ForkData, reference_replay, replay) -> None:
    for field in ("target", "action", "group", "observed_latent"):
        first, second = getattr(reference, field), getattr(current, field)
        if first.dtype in (torch.bool, torch.long):
            equal = torch.equal(first, second)
        else:
            equal = bool(torch.allclose(first, second, atol=1e-6, rtol=0.0))
        if not equal:
            raise AssertionError(f"world changed matched {field}")
    if replay["trajectory_action_by_group"] != reference_replay["trajectory_action_by_group"]:
        raise AssertionError("world changed the fixed trajectory actions")


def _mse_split(data: ForkData, mask: torch.Tensor) -> dict[str, float | int | None]:
    error = (data.generated_latent - data.observed_latent).pow(2).flatten(1).mean(1)
    target = data.target.bool()
    selected = error[mask]
    dead = mask & target
    alive = mask & ~target
    return {
        "examples": int(mask.sum()),
        "deaths": int(dead.sum()),
        "all": float(selected.mean()),
        "fatal": float(error[dead].mean()) if bool(dead.any()) else None,
        "safe": float(error[alive].mean()) if bool(alive.any()) else None,
    }


def _binary(probability: torch.Tensor, target: torch.Tensor) -> dict:
    result = binary_metrics(probability, target)
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in result.items()
    }


def _prediction_splits(
    probability: torch.Tensor,
    data: ForkData,
    trajectory_action_by_group: list[int],
) -> dict:
    chosen = torch.tensor(trajectory_action_by_group)[data.group]
    trajectory = data.action == chosen
    alternatives = ~trajectory
    return {
        "all_actions": _binary(probability, data.target),
        "trajectory_action": _binary(probability[trajectory], data.target[trajectory]),
        "other_16_actions": _binary(probability[alternatives], data.target[alternatives]),
        "counts": {
            "trajectory_examples": int(trajectory.sum()),
            "trajectory_deaths": int(data.target[trajectory].sum()),
            "counterfactual_examples": int(alternatives.sum()),
            "counterfactual_deaths": int(data.target[alternatives].sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument(
        "--world",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    worlds = [(name, Path(path)) for name, path in args.world]
    names = [name for name, _ in worlds]
    if len(names) != len(set(names)):
        parser.error("world names must be unique")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("world names may contain lowercase letters, digits, _ and -")

    config = Config(transition="direct", time_mixer="attention")
    base = Config()
    inputs = {
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "worlds": {name: file_digest(path) for name, path in worlds},
    }
    contract = {
        "version": "direct-phase1b-world-probe-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/localize_matched_counterfactual.py"),
            Path("artifacts/localize_direct_transition_stages.py"),
            Path("artifacts/localize_counterfactual.py"),
            Path("artifacts/localize_counterfactual_interaction.py"),
        ),
        "evaluation": "same fixed terminal-opportunity DEV states and all 17 actions",
        "action_split": (
            "fixed trajectory policy action executed at each state versus the other 16; "
            "this is not an offline TRAIN replay action"
        ),
        "probe": "action-centered leave-one-pre-action-state-out linear",
        "seeds": args.seeds,
        "linear_steps": args.linear_steps,
        "permutations": args.permutations,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for child in ("features", "probes"):
        (args.out / child).mkdir(exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Direct world-probe contract changed")
    else:
        atomic_json(contract_path, contract)

    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    saved = torch.load(args.forks, weights_only=False)
    extracted: dict[str, tuple[ForkData, dict]] = {}
    reference_data = reference_replay = None
    for name, path in worlds:
        feature_path = args.out / "features" / f"{name}.pt"
        feature_contract = {
            "version": contract["version"],
            "inputs": inputs,
            "world": name,
            "checkpoint": inputs["worlds"][name],
            "implementation": contract["implementation"],
        }
        if feature_path.exists():
            payload = torch.load(feature_path, weights_only=False)
            if payload["contract"] != feature_contract:
                raise ValueError(f"feature contract changed: {feature_path}")
            data = ForkData(**payload["data"])
            replay = payload["replay"]
        else:
            print(f"extracting matched forks: {name}", flush=True)
            world, dummy_heads = _load_world(path, config)
            data, replay = extract_matched_forks(
                saved,
                encoder,
                trajectory_world,
                trajectory_heads,
                world,
                dummy_heads,
                config,
                config,
            )
            _atomic_torch(
                feature_path,
                {"contract": feature_contract, "data": vars(data), "replay": replay},
            )
            world.cpu()
            dummy_heads.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if reference_data is None:
            reference_data, reference_replay = data, replay
        else:
            _same_examples(reference_data, data, reference_replay, replay)
        extracted[name] = (data, replay)

    encoder.cpu()
    trajectory_world.cpu()
    trajectory_heads.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seeds = [config.seed + 4000 + index for index in range(args.seeds)]
    results = {}
    for index, (name, path) in enumerate(worlds):
        data, replay = extracted[name]
        probe_contract = {
            "version": contract["version"],
            "stage": name,
            "stage_sha256": inputs["worlds"][name],
            "feature": "generated_latent",
            "seeds": seeds,
            "steps": args.linear_steps,
            "lr": 3e-3,
            "weight_decay": 1e-3,
        }
        probability = _resumable_linear_probe(
            data.generated_latent,
            data.target,
            data.action,
            data.group,
            config,
            seeds=seeds,
            steps=args.linear_steps,
            checkpoint=args.out / "probes" / f"{name}.generated_latent.pt",
            contract=probe_contract,
        )
        chosen = torch.tensor(replay["trajectory_action_by_group"])[data.group]
        trajectory_mask = data.action == chosen
        results[name] = {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["worlds"][name],
            "replay": replay,
            "latent_prediction_error": latent_error(data),
            "latent_prediction_error_split": {
                "trajectory_action": _mse_split(data, trajectory_mask),
                "other_16_actions": _mse_split(data, ~trajectory_mask),
            },
            "generated_latent_probe": {
                "binary_split": _prediction_splits(
                    probability, data, replay["trajectory_action_by_group"]
                ),
                "same_action": report_score(
                    probability,
                    data.target,
                    data.action,
                    permutations=args.permutations,
                    seed=config.seed + 7000 + index,
                ),
            },
        }

    report = {"contract": contract, "worlds": results}
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
