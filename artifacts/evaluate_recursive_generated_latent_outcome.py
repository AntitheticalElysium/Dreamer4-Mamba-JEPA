"""Evaluate S83 on the recursive path its outcome objective trained."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from artifacts.evaluate_fatality_direction_delta import (
    archive_current,
    delta_metrics,
)
from artifacts.evaluate_generated_latent_outcome_heads import score as head_score
from artifacts.localize_counterfactual import load_models
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch, finite_json
from artifacts.predictor_flow_attribution_common import load_world
from artifacts.summarize_predictor_flow_attribution import (
    contrast_vector,
    fork_contrast_vector,
    paired_difference,
)
from artifacts.train_generated_latent_outcome_shaping import LatentContinuationHead
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.state import WorldState
from d4mj.transition import advance, commit_inputs, observe


def _led_to(actions: torch.Tensor, start: int, end: int, config: Config) -> torch.Tensor:
    positions = torch.arange(start, end)
    incoming = positions - 1
    return torch.where(
        incoming >= 0,
        actions[incoming.clamp(min=0)],
        torch.tensor(config.n_actions),
    ).long()


@torch.no_grad()
def recursive_archive(
    world,
    records: list[dict],
    config: Config,
    *,
    context: int,
    label: str,
) -> dict[str, torch.Tensor | list[str]]:
    predicted, target, current = [], [], []
    labels, actions, groups, pools = [], [], [], []
    for index, record in enumerate(records):
        transition = int(record["transitions"][0])
        if transition < 1:
            raise ValueError("recursive archive evaluation needs a predecessor action")
        latents = record["latents"]
        taken = record["actions_taken"]
        start = max(0, transition - context)
        block = latents[start:transition][None].to(config.device)
        incoming = _led_to(taken, start, transition, config)[None].to(config.device)
        rng = torch.Generator(device=config.device).manual_seed(
            config.seed + 19_000 + index
        )
        committed, conditioning = commit_inputs(block, rng, config)
        features, _, memory = world(None, incoming, committed, conditioning)
        state = WorldState(block[:, -1:], memory, block.shape[1], features[:, -1:])
        generated_current, _ = advance(
            world,
            state,
            taken[transition - 1].view(1, 1).to(config.device),
            rng,
            config,
        )
        successor, _ = advance(
            world,
            generated_current,
            taken[transition].view(1, 1).to(config.device),
            rng,
            config,
        )
        predicted.append(successor.latent[0, 0].cpu())
        current.append(latents[transition])
        target.append(latents[transition + 1])
        labels.append(record["labels"][0])
        actions.append(taken[transition])
        groups.append(torch.tensor(record.get("group", index)))
        pools.append(record["pool"])
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"{label}: {index + 1}/{len(records)} records", flush=True)
    return {
        "predicted": torch.stack(predicted),
        "target": torch.stack(target),
        "current": torch.stack(current),
        "label": torch.stack(labels),
        "action": torch.stack(actions),
        "group": torch.stack(groups),
        "pool": pools,
    }


@torch.no_grad()
def recursive_policy_forks(
    saved: dict,
    encoder,
    trajectory_world,
    trajectory_heads,
    world,
    trajectory_config: Config,
    evaluation_config: Config,
) -> tuple[dict[str, torch.Tensor], dict]:
    if trajectory_config.device != evaluation_config.device:
        raise ValueError("trajectory and evaluation devices differ")
    varies = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    selected = varies.nonzero().flatten().tolist()
    key_to_row = {
        (int(saved["seed"][row]), int(saved["step"][row])): row
        for row in selected
    }
    if len(key_to_row) != len(selected):
        raise ValueError("fixed forks contain duplicate opportunity states")
    row_to_group = {row: group for group, row in enumerate(selected)}
    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    predicted, observed, labels, actions, groups, starts = [], [], [], [], [], []
    reproduced = set()
    trajectory_actions: dict[int, int] = {}
    device = evaluation_config.device
    for seed in sorted(by_seed):
        wanted = by_seed[seed]
        last = max(wanted)
        observation, env_state = reset(seed)
        trajectory_state = evaluation_state = None
        incoming = torch.full(
            (1, 1), evaluation_config.n_actions, dtype=torch.long, device=device
        )
        trajectory_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        evaluation_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)

        for index in range(last + 1):
            previous_evaluation_state = evaluation_state
            patches = patchify(observation[None, None], evaluation_config.patch).to(device)
            trajectory_state, trajectory_agent = observe(
                trajectory_world,
                encoder,
                trajectory_state,
                incoming,
                patches,
                trajectory_rng,
                trajectory_config,
            )
            evaluation_state, _ = observe(
                world,
                encoder,
                evaluation_state,
                incoming,
                patches,
                evaluation_rng,
                evaluation_config,
            )

            key = (seed, index)
            if key in key_to_row:
                if previous_evaluation_state is None:
                    raise ValueError("recursive fork has no preceding world state")
                row = key_to_row[key]
                group = row_to_group[row]
                first_rng = torch.Generator(device=device).manual_seed(
                    evaluation_config.seed + 2**25 + seed * 4099 + index * 17
                )
                generated_current, _ = advance(
                    world,
                    previous_evaluation_state.world,
                    incoming,
                    first_rng,
                    evaluation_config,
                )
                starts.append(evaluation_state.world.latent[0, -1].cpu())
                for action in range(evaluation_config.n_actions):
                    successor_observation, _, _, terminated, _ = env_step(
                        env_state, action, seed + index + 1
                    )
                    if bool(terminated) != bool(saved["true_death"][row, action]):
                        raise AssertionError(
                            f"truth mismatch seed={seed} step={index} action={action}"
                        )
                    chosen = torch.tensor([[action]], device=device)
                    generated_rng = torch.Generator(device=device).manual_seed(
                        evaluation_config.seed
                        + 2**26
                        + seed * 4099
                        + index * 17
                        + action
                    )
                    generated_successor, _ = advance(
                        world,
                        generated_current,
                        chosen,
                        generated_rng,
                        evaluation_config,
                    )
                    observed_rng = torch.Generator(device=device).manual_seed(
                        evaluation_config.seed
                        + 2**27
                        + seed * 4099
                        + index * 17
                        + action
                    )
                    successor_patches = patchify(
                        successor_observation[None, None], evaluation_config.patch
                    ).to(device)
                    observed_successor, _ = observe(
                        world,
                        encoder,
                        evaluation_state,
                        chosen,
                        successor_patches,
                        observed_rng,
                        evaluation_config,
                    )
                    predicted.append(generated_successor.latent[0, -1].cpu())
                    observed.append(observed_successor.world.latent[0, -1].cpu())
                    labels.append(float(terminated))
                    actions.append(action)
                    groups.append(group)
                reproduced.add(key)

            logits = trajectory_heads(trajectory_agent)["policy"][:, -1, 0]
            action = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            if key in key_to_row:
                group = row_to_group[key_to_row[key]]
                trajectory_actions[group] = action
                if action != int(saved["trajectory_action"][key_to_row[key]]):
                    raise AssertionError("fixed trajectory action did not replay")
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1
            )
            incoming.fill_(action)
            if terminated or truncated:
                if index < last:
                    raise RuntimeError("trajectory ended before a fixed fork")
                break

    missing = sorted(set(key_to_row) - reproduced)
    if missing:
        raise RuntimeError(f"failed to replay fixed forks: {missing}")
    if len(starts) != len(selected):
        raise AssertionError("fork current-state count changed")
    data = {
        "current": torch.stack(starts).repeat_interleave(
            evaluation_config.n_actions, dim=0
        ),
        "predicted": torch.stack(predicted),
        "observed": torch.stack(observed),
        "target": torch.tensor(labels),
        "action": torch.tensor(actions),
        "group": torch.tensor(groups),
    }
    replay = {
        "terminal_opportunity_states": len(selected),
        "examples": len(labels),
        "truth_replayed_exactly": True,
        "trajectory_action_by_group": [
            trajectory_actions[group] for group in range(len(selected))
        ],
        "path": (
            "observed predecessor -> generated fork state under the executed "
            "incoming action -> generated successor under each fork action"
        ),
    }
    return data, replay


def _mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted - target).pow(2).flatten(1).mean(1).mean())


def _cell(
    archive: dict,
    forks: dict,
    head: LatentContinuationHead,
    prepared: dict,
    *,
    bootstraps: int,
    seed: int,
) -> dict:
    archive_delta = delta_metrics(
        archive["current"],
        archive["target"],
        archive["predicted"],
        archive["label"],
        archive["action"],
        archive["group"],
        prepared["direction"],
        prepared["action_means"],
        bootstraps=bootstraps,
        seed=seed,
    )
    fork_delta = delta_metrics(
        forks["current"],
        forks["observed"],
        forks["predicted"],
        forks["target"],
        forks["action"],
        forks["group"],
        prepared["direction"],
        prepared["action_means"],
        bootstraps=bootstraps,
        seed=seed + 100,
    )
    return {
        "archive_support": {
            "mse": _mse(archive["predicted"], archive["target"]),
            "delta": archive_delta,
            "trained_head": head_score(
                head,
                archive["predicted"],
                archive["label"],
                archive["group"],
                bootstraps=bootstraps,
                seed=seed + 200,
            ),
        },
        "policy_forks": {
            "mse": _mse(forks["predicted"], forks["observed"]),
            "delta": fork_delta,
            "trained_head": head_score(
                head,
                forks["predicted"],
                forks["target"],
                forks["group"],
                bootstraps=bootstraps,
                seed=seed + 300,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--allowed-world", type=Path, required=True)
    parser.add_argument("--allowed-model", type=Path, required=True)
    parser.add_argument("--stopped-world", type=Path, required=True)
    parser.add_argument("--stopped-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    base = Config()
    worlds = {
        "allowed_020k": (args.allowed_world, args.allowed_model),
        "stopped_020k": (args.stopped_world, args.stopped_model),
    }
    inputs = {
        "prepared": file_digest(args.prepared),
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "original_report": file_digest(args.original_report),
        "worlds": {
            name: {"world": file_digest(world), "model": file_digest(model)}
            for name, (world, model) in worlds.items()
        },
    }
    contract = {
        "version": "recursive-generated-latent-outcome-evaluation-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/train_generated_latent_outcome_shaping.py"),
            Path("artifacts/localize_matched_counterfactual.py"),
            Path("artifacts/evaluate_fatality_direction_delta.py"),
        ),
        "status": (
            "post-hoc path-alignment correction; archive path was inspected before "
            "this contract, recursive policy-fork results were not"
        ),
        "path": (
            "two recursive Direct advances from an observed predecessor; the first "
            "uses the logged incoming action and the second uses the evaluated action"
        ),
        "endpoint": 20_000,
        "primary": (
            "allowed-minus-stopped fatal-minus-safe recursive successor movement "
            "along the fixed TRAIN fatality direction"
        ),
        "uncertainty": "paired whole DEV episode or whole policy-fork-state bootstrap",
        "minimum_effect": "5% of the corresponding true held-out contrast",
        "bootstraps": args.bootstraps,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("recursive outcome evaluation contract changed")
    atomic_json(contract_path, contract)

    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    saved_forks = torch.load(args.forks, weights_only=False, map_location="cpu")
    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    results, archive_features, fork_features = {}, {}, {}
    replay_reference = None
    for index, (name, (world_path, model_path)) in enumerate(worlds.items()):
        world = load_world(world_path, config)
        head = LatentContinuationHead(config).to(config.device).eval()
        load(model_path, config, part1=head)
        archive_path = args.out / "features" / f"archive_{name}.pt"
        if archive_path.exists():
            payload = torch.load(archive_path, weights_only=False, map_location="cpu")
            if payload["contract"] != contract:
                raise ValueError(f"recursive archive cache changed: {archive_path}")
            archive = payload["data"]
            print(f"cached recursive archive: {name}", flush=True)
        else:
            print(f"recursive archive: {name}", flush=True)
            archive = recursive_archive(
                world,
                prepared["records"],
                config,
                context=16,
                label=name,
            )
            atomic_torch(archive_path, {"contract": contract, "data": archive})

        fork_path = args.out / "features" / f"forks_{name}.pt"
        if fork_path.exists():
            payload = torch.load(fork_path, weights_only=False, map_location="cpu")
            if payload["contract"] != contract:
                raise ValueError(f"recursive fork cache changed: {fork_path}")
            fork, replay = payload["data"], payload["replay"]
            print(f"cached recursive policy forks: {name}", flush=True)
        else:
            print(f"recursive policy forks: {name}", flush=True)
            fork, replay = recursive_policy_forks(
                saved_forks,
                encoder,
                trajectory_world,
                trajectory_heads,
                world,
                config,
                config,
            )
            atomic_torch(
                fork_path,
                {"contract": contract, "data": fork, "replay": replay},
            )
        if replay_reference is None:
            replay_reference = replay
        elif replay != replay_reference:
            raise AssertionError("recursive fork replay changed between worlds")
        results[name] = _cell(
            archive,
            fork,
            head,
            prepared,
            bootstraps=args.bootstraps,
            seed=config.seed + 19_200 + index * 1000,
        )
        archive_features[name] = archive
        fork_features[name] = fork
        world.cpu()
        head.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    archive_start = archive_current(prepared["records"])
    true_archive = results["stopped_020k"]["archive_support"]["delta"][
        "conditional_consequence"
    ]["true_fatal_minus_safe"]
    true_forks = results["stopped_020k"]["policy_forks"]["delta"][
        "conditional_consequence"
    ]["true_fatal_minus_safe"]
    comparisons = {
        "archive_support": paired_difference(
            contrast_vector(
                archive_features["allowed_020k"],
                archive_start,
                prepared["direction"],
            ),
            contrast_vector(
                archive_features["stopped_020k"],
                archive_start,
                prepared["direction"],
            ),
            minimum_effect=0.05 * abs(true_archive),
            seed=config.seed + 19_700,
            samples=args.bootstraps,
        ),
        "policy_forks": paired_difference(
            fork_contrast_vector(
                fork_features["allowed_020k"], prepared["direction"]
            ),
            fork_contrast_vector(
                fork_features["stopped_020k"], prepared["direction"]
            ),
            minimum_effect=0.05 * abs(true_forks),
            seed=config.seed + 19_800,
            samples=args.bootstraps,
        ),
    }
    both_rescue = all(row["material_improvement"] for row in comparisons.values())
    both_equivalent = all(row["practically_equivalent"] for row in comparisons.values())
    verdict = (
        "recursive_path_rescue"
        if both_rescue
        else "no_material_recursive_path_effect"
        if both_equivalent
        else "recursive_path_mixed_or_distribution_specific"
    )
    report = finite_json(
        {
            "contract": contract,
            "replay": replay_reference,
            "cells": results,
            "comparisons": comparisons,
            "original_one_step_verdict": json.loads(args.original_report.read_text())[
                "verdict"
            ],
            "verdict": verdict,
        }
    )
    atomic_json(args.out / "report.json", report)
    print(f"verdict: {verdict}", flush=True)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
