"""Compare Direct trajectory-executed and alternative actions on paired DEV states."""

from __future__ import annotations

import argparse
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
from artifacts.localize_counterfactual_interaction import action_means, report_score
from artifacts.localize_matched_counterfactual import extract_matched_forks
from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from artifacts.probe_direct_phase1b_worlds import _load_world, _mse_split
from d4mj.config import Config


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def episode_oof_probe(
    feature: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
    state_group: torch.Tensor,
    state_seed: torch.Tensor,
    config: Config,
    *,
    folds: int,
    seeds: list[int],
    steps: int,
    checkpoint: Path,
    contract: dict,
) -> torch.Tensor:
    """Action-centered probe with whole trajectories held out in ten folds."""
    feature = flatten(feature).cpu()
    target, action = target.cpu(), action.cpu()
    example_seed = state_seed[state_group].cpu()
    unique = torch.tensor(sorted(set(example_seed.tolist())))
    if len(unique) < folds:
        raise ValueError("fewer trajectory seeds than requested folds")
    order = unique[
        torch.randperm(len(unique), generator=torch.Generator().manual_seed(config.seed + 9300))
    ]
    fold_by_seed = {
        int(seed): index % folds for index, seed in enumerate(order.tolist())
    }
    example_fold = torch.tensor([fold_by_seed[int(seed)] for seed in example_seed])

    prediction = torch.zeros_like(target)
    complete: set[int] = set()
    if checkpoint.exists():
        stored = torch.load(checkpoint, weights_only=False)
        if stored["contract"] != contract:
            raise ValueError(f"paired-action probe contract changed: {checkpoint}")
        prediction.copy_(stored["prediction"])
        complete = set(stored["complete"])

    for test_fold in range(folds):
        if test_fold in complete:
            continue
        validation_fold = (test_fold + 1) % folds
        test_mask = example_fold == test_fold
        validation_mask = example_fold == validation_fold
        fit_mask = ~(test_mask | validation_mask)
        means = action_means(feature, action, fit_mask, config.n_actions)
        centered = feature - means[action]
        fold_predictions = []
        for seed in seeds:
            fold_predictions.append(
                fit_linear_once(
                    centered[fit_mask],
                    target[fit_mask],
                    centered[validation_mask],
                    target[validation_mask],
                    centered[test_mask],
                    seed=seed + test_fold * 101,
                    device=config.device,
                    steps=steps,
                    lr=3e-3,
                    weight_decay=1e-3,
                )
            )
        prediction[test_mask] = torch.stack(fold_predictions).mean(0)
        complete.add(test_fold)
        _atomic_torch(
            checkpoint,
            {
                "contract": contract,
                "prediction": prediction,
                "complete": sorted(complete),
                "fold_by_seed": fold_by_seed,
            },
        )
        print(f"{contract['world']}: {len(complete)}/{folds} folds", flush=True)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument(
        "--world", nargs=2, action="append", metavar=("NAME", "CHECKPOINT"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    worlds = [(name, Path(path)) for name, path in args.world]
    names = [name for name, _ in worlds]
    if len(names) != len(set(names)):
        parser.error("world names must be unique")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("world names may contain lowercase letters, digits, _ and -")

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    inputs = {
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "worlds": {name: file_digest(path) for name, path in worlds},
    }
    contract = {
        "version": "paired-trajectory-action-probe-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/localize_matched_counterfactual.py"),
            Path("artifacts/localize_counterfactual.py"),
            Path("artifacts/localize_counterfactual_interaction.py"),
        ),
        "evaluation": (
            "paired safe/fatal trajectory-executed actions and all 16 alternatives "
            "on the same fixed DEV opportunity states"
        ),
        "split": "whole trajectory seed held out",
        "folds": args.folds,
        "probe_seeds": args.seeds,
        "linear_steps": args.linear_steps,
        "permutations": args.permutations,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("paired trajectory-action experiment contract changed")
    else:
        atomic_json(contract_path, contract)

    saved = torch.load(args.forks, weights_only=False)
    if not torch.equal(
        saved["trajectory_death"],
        saved["true_death"].gather(1, saved["trajectory_action"][:, None]).squeeze(1),
    ):
        raise AssertionError("trajectory outcome is not aligned to its action")
    opportunity = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    if not bool(opportunity.all()):
        raise AssertionError("paired fork set contains a non-opportunity state")
    if int(saved["trajectory_death"].sum()) * 2 != len(saved["trajectory_death"]):
        raise AssertionError("paired fork set is not balanced on trajectory outcome")

    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    extracted: dict[str, tuple[ForkData, dict]] = {}
    for name, path in worlds:
        feature_path = args.out / "features" / f"{name}.pt"
        feature_contract = {
            "version": contract["version"],
            "inputs": inputs,
            "world": name,
            "implementation": contract["implementation"],
        }
        if feature_path.exists():
            payload = torch.load(feature_path, weights_only=False)
            if payload["contract"] != feature_contract:
                raise ValueError(f"paired feature contract changed: {feature_path}")
            data, replay = ForkData(**payload["data"]), payload["replay"]
        else:
            print(f"extracting paired actions: {name}", flush=True)
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
            if replay["trajectory_action_by_group"] != saved["trajectory_action"].tolist():
                raise AssertionError("fixed trajectory actions did not replay exactly")
            _atomic_torch(
                feature_path,
                {"contract": feature_contract, "data": vars(data), "replay": replay},
            )
            world.cpu()
            dummy_heads.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        extracted[name] = data, replay

    encoder.cpu()
    trajectory_world.cpu()
    trajectory_heads.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_seeds = [config.seed + 9400 + index for index in range(args.seeds)]
    results = {}
    for index, (name, path) in enumerate(worlds):
        data, replay = extracted[name]
        probe_contract = {
            "version": contract["version"],
            "world": name,
            "checkpoint": inputs["worlds"][name],
            "forks": inputs["forks"],
            "folds": args.folds,
            "seeds": probe_seeds,
            "steps": args.linear_steps,
        }
        probability = episode_oof_probe(
            data.generated_latent,
            data.target,
            data.action,
            data.group,
            saved["seed"],
            config,
            folds=args.folds,
            seeds=probe_seeds,
            steps=args.linear_steps,
            checkpoint=args.out / "probes" / f"{name}.pt",
            contract=probe_contract,
        )
        chosen = saved["trajectory_action"][data.group]
        trajectory = data.action == chosen
        if not torch.equal(data.target[trajectory].bool(), saved["trajectory_death"]):
            raise AssertionError("trajectory-action examples changed order or outcome")
        alternatives = ~trajectory
        results[name] = {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": inputs["worlds"][name],
            "replay": replay,
            "latent_prediction_error": latent_error(data),
            "latent_prediction_error_split": {
                "trajectory_action": _mse_split(data, trajectory),
                "other_16_actions": _mse_split(data, alternatives),
            },
            "generated_latent_probe": {
                "trajectory_action": binary_metrics(
                    probability[trajectory], data.target[trajectory]
                ),
                "other_16_actions": binary_metrics(
                    probability[alternatives], data.target[alternatives]
                ),
                "all_actions": binary_metrics(probability, data.target),
                "same_action": report_score(
                    probability,
                    data.target,
                    data.action,
                    permutations=args.permutations,
                    seed=config.seed + 9500 + index,
                ),
            },
        }

    report = {"contract": contract, "fork_summary": {
        "states": len(saved["seed"]),
        "trajectory_deaths": int(saved["trajectory_death"].sum()),
        "trajectory_safe": int((~saved["trajectory_death"]).sum()),
        "trajectory_seeds": len(set(saved["seed"].tolist())),
    }, "worlds": results}
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

