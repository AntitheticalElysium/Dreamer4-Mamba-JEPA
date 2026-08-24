"""Measure whether Direct's one-step DEV prediction uses the realized action."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from artifacts.run_stage_a import corpus
from d4mj.config import Config
from d4mj.data import sample_batch
from d4mj.train import _to, cache_latents
from d4mj.transition import commit_inputs


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--shuffles", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/action_shuffle_dev.json"),
    )
    args = parser.parse_args()

    base = Config()
    config = replace(base, transition="direct", time_mixer="attention")
    encoder, world, _ = load_models(args.phase1a, args.phase2, base, config)
    _, dev = corpus(base, args.expert, print)
    cached = cache_latents(encoder, dev, base)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sampler = torch.Generator().manual_seed(config.seed + 6000)
    model_rng = torch.Generator(device=config.device).manual_seed(config.seed + 6001)
    shuffle_rng = torch.Generator().manual_seed(config.seed + 6002)

    correct_sum = shuffled_sum = sensitivity_sum = 0.0
    correct_changed_sum = shuffled_changed_sum = 0.0
    transitions = shuffled_transitions = changed = 0
    action_count = torch.zeros(config.n_actions, dtype=torch.long)
    correct_by_action = torch.zeros(config.n_actions, dtype=torch.float64)
    shuffled_by_action = torch.zeros(config.n_actions, dtype=torch.float64)
    shuffled_count_by_action = torch.zeros(config.n_actions, dtype=torch.long)

    world.eval()
    with torch.no_grad():
        for step_index in range(args.batches):
            batch = _to(
                sample_batch(
                    cached,
                    sampler,
                    config,
                    step_index,
                    args.batches,
                    mixture=False,
                ),
                config.device,
            )
            committed, conditioning = commit_inputs(batch.latents, model_rng, config)
            features, _, _ = world(
                None,
                batch.led_to_action,
                committed,
                conditioning,
            )
            action = batch.led_to_action[:, 1:]
            target = batch.latents[:, 1:]
            prediction = world.predict(features[:, :-1], action)
            correct_error = (prediction - target).pow(2).mean(dim=(2, 3))

            flat_action = action.flatten()
            flat_correct = correct_error.flatten()
            transitions += flat_action.numel()
            correct_sum += float(flat_correct.sum())
            action_count += torch.bincount(
                flat_action.cpu(), minlength=config.n_actions
            )
            correct_by_action.scatter_add_(
                0, flat_action.cpu(), flat_correct.double().cpu()
            )

            for _ in range(args.shuffles):
                order = torch.randperm(flat_action.numel(), generator=shuffle_rng)
                shuffled_action = flat_action.cpu()[order].to(config.device).view_as(action)
                shuffled_prediction = world.predict(
                    features[:, :-1], shuffled_action
                )
                shuffled_error = (
                    shuffled_prediction - target
                ).pow(2).mean(dim=(2, 3))
                changed_mask = shuffled_action != action

                shuffled_sum += float(shuffled_error.sum())
                sensitivity_sum += float(
                    (shuffled_prediction - prediction).pow(2).mean(dim=(2, 3)).sum()
                )
                shuffled_transitions += flat_action.numel()
                changed += int(changed_mask.sum())
                correct_changed_sum += float(correct_error[changed_mask].sum())
                shuffled_changed_sum += float(shuffled_error[changed_mask].sum())

                shuffled_flat = shuffled_error.flatten().double().cpu()
                shuffled_source = flat_action.cpu()
                shuffled_by_action.scatter_add_(
                    0, shuffled_source, shuffled_flat
                )
                shuffled_count_by_action += torch.bincount(
                    shuffled_source, minlength=config.n_actions
                )

    correct_mse = correct_sum / transitions
    shuffled_mse = shuffled_sum / shuffled_transitions
    report = {
        "contract": {
            "arm": "direct-attention",
            "split": "ordinary DEV windows",
            "history_and_target_fixed": True,
            "only_outgoing_action_shuffled": True,
            "shuffle_preserves_empirical_action_marginal": True,
            "batches": args.batches,
            "shuffles_per_batch": args.shuffles,
        },
        "checkpoint": {
            "phase1a": str(args.phase1a.resolve()),
            "phase2": str(args.phase2.resolve()),
            "phase1a_sha256": _digest(args.phase1a),
            "phase2_sha256": _digest(args.phase2),
        },
        "transitions": transitions,
        "shuffled_transitions": shuffled_transitions,
        "changed_fraction": changed / shuffled_transitions,
        "correct_mse": correct_mse,
        "shuffled_mse": shuffled_mse,
        "shuffled_over_correct": shuffled_mse / max(correct_mse, 1e-12),
        "shuffled_excess_mse": shuffled_mse - correct_mse,
        "prediction_sensitivity_mse": sensitivity_sum / shuffled_transitions,
        "changed_only": {
            "examples": changed,
            "correct_mse": correct_changed_sum / max(changed, 1),
            "shuffled_mse": shuffled_changed_sum / max(changed, 1),
            "shuffled_over_correct": shuffled_changed_sum
            / max(correct_changed_sum, 1e-12),
        },
        "per_realized_action": {
            str(action_index): {
                "examples": int(action_count[action_index]),
                "correct_mse": float(
                    correct_by_action[action_index]
                    / action_count[action_index].clamp(min=1)
                ),
                "shuffled_mse": float(
                    shuffled_by_action[action_index]
                    / shuffled_count_by_action[action_index].clamp(min=1)
                ),
            }
            for action_index in range(config.n_actions)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
