"""Evaluate attribution worlds on fixed policy counterfactual forks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from artifacts.evaluate_fatality_direction_delta import delta_metrics, fork_start_latents
from artifacts.localize_counterfactual import ForkData, binary_metrics, load_models
from artifacts.localize_flow_phase1b import FlowForkData, extract_flow_forks
from artifacts.localize_matched_counterfactual import extract_matched_forks
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch, finite_json
from artifacts.predictor_flow_attribution_common import load_world
from artifacts.probe_paired_trajectory_forks import episode_oof_probe
from d4mj.agent import Heads
from d4mj.config import Config


def _dummy_heads(config: Config) -> Heads:
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(config.device).eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    return heads


def _latent_error(predicted: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> dict:
    error = (predicted - observed).pow(2).flatten(1).mean(1)
    target = mask.bool()
    return {
        "all": float(error.mean()),
        "fatal": float(error[target].mean()),
        "safe": float(error[~target].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--fork-starts", type=Path, required=True)
    parser.add_argument(
        "--world",
        nargs=4,
        action="append",
        metavar=("NAME", "TRANSITION", "PREDICTOR", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flow-samples", type=int, default=4)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--probe-seeds", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=600)
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

    base = Config()
    trajectory_config = Config(transition="direct", time_mixer="attention")
    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    saved = torch.load(args.forks, weights_only=False, map_location="cpu")
    starts = fork_start_latents(
        args.phase1a, args.trajectory_phase2, args.forks, args.fork_starts
    )
    inputs = {
        "prepared": file_digest(args.prepared),
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "worlds": {name: file_digest(path) for name, _, _, path in worlds},
    }
    contract = {
        "version": "predictor-flow-policy-fork-evaluation-v1",
        "inputs": inputs,
        "specs": {
            name: {"transition": transition, "predictor": predictor}
            for name, transition, predictor, _ in worlds
        },
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/predictor_flow_attribution_common.py"),
            Path("artifacts/localize_matched_counterfactual.py"),
            Path("artifacts/localize_flow_phase1b.py"),
            Path("artifacts/evaluate_fatality_direction_delta.py"),
        ),
        "flow_samples": args.flow_samples,
        "flow_randomness": "common sample seeds for every Flow world",
        "probe": "action-centered, whole trajectory seed held out",
        "folds": args.folds,
        "probe_seeds": args.probe_seeds,
        "probe_steps": args.probe_steps,
        "bootstraps": args.bootstraps,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("attribution fork contract changed")
    atomic_json(contract_path, contract)

    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a,
        args.trajectory_phase2,
        base,
        trajectory_config,
    )
    probe_seeds = [base.seed + 15_000 + index for index in range(args.probe_seeds)]
    results = {}
    reference = None
    for world_index, (name, transition, predictor, path) in enumerate(worlds):
        config = Config(transition=transition, time_mixer="attention")
        feature_path = args.out / "features" / f"{name}.pt"
        feature_contract = {
            "contract": contract,
            "world": name,
            "checkpoint": inputs["worlds"][name],
        }
        if feature_path.exists():
            payload = torch.load(feature_path, weights_only=False, map_location="cpu")
            if payload["contract"] != feature_contract:
                raise ValueError(f"fork feature contract changed: {feature_path}")
            variants = payload["variants"]
            replay = payload["replay"]
        else:
            print(f"policy-fork extraction: {name}", flush=True)
            world = load_world(path, config, predictor)
            heads = _dummy_heads(config)
            if transition == "direct":
                direct, replay = extract_matched_forks(
                    saved,
                    encoder,
                    trajectory_world,
                    trajectory_heads,
                    world,
                    heads,
                    trajectory_config,
                    config,
                )
                variants = {
                    "generated": {
                        "observed": direct.observed_latent,
                        "predicted": direct.generated_latent,
                        "target": direct.target,
                        "action": direct.action,
                        "group": direct.group,
                    }
                }
            else:
                flow, replay = extract_flow_forks(
                    saved,
                    encoder,
                    trajectory_world,
                    trajectory_heads,
                    world,
                    heads,
                    trajectory_config,
                    config,
                    samples=args.flow_samples,
                    signal_levels=(config.tau_ctx_index,),
                    cache=args.out / "flow_seed_cache" / name,
                    contract=feature_contract,
                )
                variants = {
                    "generated_first": {
                        "observed": flow.observed_latent,
                        "predicted": flow.generated_latent_first,
                        "target": flow.target,
                        "action": flow.action,
                        "group": flow.group,
                    },
                    "generated_mean": {
                        "observed": flow.observed_latent,
                        "predicted": flow.generated_latent_mean,
                        "target": flow.target,
                        "action": flow.action,
                        "group": flow.group,
                    },
                }
            atomic_torch(
                feature_path,
                {"contract": feature_contract, "variants": variants, "replay": replay},
            )
            world.cpu()
            heads.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if replay["trajectory_action_by_group"] != saved["trajectory_action"].tolist():
            raise AssertionError("fixed trajectory actions changed")
        variant_results = {}
        for variant_index, (variant, data) in enumerate(variants.items()):
            identity = (data["observed"], data["target"], data["action"], data["group"])
            if reference is None:
                reference = identity
            else:
                for first, second in zip(reference, identity):
                    if first.dtype in (torch.bool, torch.long):
                        equal = torch.equal(first, second)
                    else:
                        equal = bool(torch.allclose(first, second, atol=1e-6, rtol=0.0))
                    if not equal:
                        raise AssertionError(f"{name}/{variant} changed fork examples")

            probe_contract = {
                "evaluation": contract,
                "world": name,
                "variant": variant,
                "checkpoint": inputs["worlds"][name],
            }
            probability = episode_oof_probe(
                data["predicted"],
                data["target"],
                data["action"],
                data["group"],
                saved["seed"],
                config,
                folds=args.folds,
                seeds=probe_seeds,
                steps=args.probe_steps,
                checkpoint=args.out / "probes" / f"{name}_{variant}.pt",
                contract=probe_contract,
            )
            chosen = saved["trajectory_action"][data["group"]]
            trajectory = data["action"] == chosen
            splits = {
                "all_actions": torch.ones_like(trajectory),
                "trajectory_action": trajectory,
                "other_16_actions": ~trajectory,
            }
            metrics = {}
            for split_index, (split, mask) in enumerate(splits.items()):
                metrics[split] = {
                    "probe": binary_metrics(probability[mask], data["target"][mask]),
                    "latent_mse": _latent_error(
                        data["predicted"][mask],
                        data["observed"][mask],
                        data["target"][mask],
                    ),
                    "delta": delta_metrics(
                        starts[data["group"]][mask],
                        data["observed"][mask],
                        data["predicted"][mask],
                        data["target"][mask],
                        data["action"][mask],
                        saved["seed"][data["group"]][mask],
                        prepared["direction"],
                        prepared["action_means"],
                        bootstraps=args.bootstraps,
                        seed=config.seed
                        + 15_500
                        + world_index * 100
                        + variant_index * 10
                        + split_index,
                        conditional_group=data["group"][mask],
                    ),
                }
            variant_results[variant] = metrics
        results[name] = {
            "transition": transition,
            "predictor": predictor,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["worlds"][name],
            "replay": replay,
            "variants": variant_results,
        }

    for module in (encoder, trajectory_world, trajectory_heads):
        module.cpu()
    report = finite_json({"contract": contract, "worlds": results})
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
