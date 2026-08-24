"""Action-controlled localization on the exact counterfactual gate forks.

Run after artifacts/localize_counterfactual.py exists.

The previous probe showed that action identity alone predicts death well across the
seven opportunity states. This diagnostic removes that shortcut in two ways:

1. Evaluation uses CONDITIONAL AUC: a death score is compared only against safe
   scores for the SAME action across different pre-action states. Any static
   "action 7 is dangerous" score therefore contributes exactly chance.
2. Linear probes are additionally centered per action using TRAIN states only
   before fitting, removing each action's mean feature vector.

Probe predictions are out-of-fold by pre-action state. For every test state:
  - one different whole state is validation;
  - all remaining states are fit;
  - no example from the test state is used for fitting or checkpoint selection.

Outputs answer:
  * Does the generated latent carry state-specific fatality beyond action identity?
  * Does the generated world readout carry it?
  * Can a fresh production-shaped continuation MLP exploit it?
  * Does the actually trained Phase-2 continuation head exploit it?
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from artifacts.localize_counterfactual import (
    extract_exact_forks,
    fit_linear_once,
    fit_mlp_once,
    flatten,
    load_models,
)
from d4mj.config import Config


def conditional_auc(
    score: Tensor,
    target: Tensor,
    action: Tensor,
) -> dict:
    """AUC computed only among examples sharing the exact same action."""
    score = score.detach().float().cpu().flatten()
    target = target.detach().bool().cpu().flatten()
    action = action.detach().long().cpu().flatten()

    per_action: dict[str, dict] = {}
    pair_scores = []
    macro = []

    for a in sorted(set(action.tolist())):
        mask = action == a
        y = target[mask]
        s = score[mask]

        positives = s[y]
        negatives = s[~y]
        if not len(positives) or not len(negatives):
            continue

        delta = positives[:, None] - negatives[None]
        pair = delta.gt(0).float() + 0.5 * delta.eq(0).float()
        value = float(pair.mean())

        per_action[str(a)] = {
            "auc": value,
            "dead_states": int(y.sum()),
            "safe_states": int((~y).sum()),
            "pairs": int(pair.numel()),
            "mean_dead_score": float(positives.mean()),
            "mean_safe_score": float(negatives.mean()),
        }
        macro.append(value)
        pair_scores.append(pair.flatten())

    if not macro:
        raise RuntimeError("No action has both fatal and safe outcomes")

    all_pairs = torch.cat(pair_scores)
    return {
        "varying_actions": len(macro),
        "same_action_pairs": int(len(all_pairs)),
        "macro_auc": float(torch.tensor(macro).mean()),
        "pooled_pair_auc": float(all_pairs.mean()),
        "min_action_auc": min(macro),
        "max_action_auc": max(macro),
        "per_action": per_action,
    }


def permutation_p_value(
    score: Tensor,
    target: Tensor,
    action: Tensor,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Shuffle outcomes within action, preserving each action's death count."""
    observed = conditional_auc(score, target, action)["pooled_pair_auc"]
    rng = torch.Generator().manual_seed(seed)

    target = target.detach().bool().cpu()
    action = action.detach().long().cpu()
    score = score.detach().float().cpu()

    indices_by_action = [
        (action == a).nonzero().flatten()
        for a in sorted(set(action.tolist()))
    ]

    null = []
    for _ in range(permutations):
        shuffled = target.clone()
        for indices in indices_by_action:
            if len(indices) <= 1:
                continue
            order = torch.randperm(len(indices), generator=rng)
            shuffled[indices] = target[indices[order]]
        null.append(
            conditional_auc(score, shuffled, action)["pooled_pair_auc"]
        )

    null_t = torch.tensor(null)
    return {
        "observed": observed,
        "null_mean": float(null_t.mean()),
        "null_std": float(null_t.std(unbiased=False)),
        "one_sided_p": float(
            ((null_t >= observed).sum() + 1) / (permutations + 1)
        ),
    }


def action_means(
    x: Tensor,
    action: Tensor,
    fit_mask: Tensor,
    n_actions: int,
) -> Tensor:
    """Mean feature for each action using fit states only."""
    width = x.shape[1]
    means = torch.zeros(n_actions, width, dtype=x.dtype)

    global_mean = x[fit_mask].mean(0)
    for a in range(n_actions):
        rows = fit_mask & (action == a)
        means[a] = x[rows].mean(0) if bool(rows.any()) else global_mean
    return means


def centered_oof_linear(
    x: Tensor,
    target: Tensor,
    action: Tensor,
    group: Tensor,
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    lr: float,
    weight_decay: float,
) -> Tensor:
    """LO-state-out predictions after train-only per-action centering."""
    x = flatten(x).cpu()
    target = target.cpu()
    action = action.cpu()
    group = group.cpu()

    groups = sorted(set(group.tolist()))
    prediction = torch.zeros_like(target)

    for test_group in groups:
        test_mask = group == test_group
        remaining = [g for g in groups if g != test_group]
        seed_predictions = []

        for seed_index, seed in enumerate(seeds):
            val_group = remaining[seed_index % len(remaining)]
            fit_mask = (group != test_group) & (group != val_group)
            val_mask = group == val_group

            means = action_means(
                x, action, fit_mask, config.n_actions
            )
            centered = x - means[action]

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
                    lr=lr,
                    weight_decay=weight_decay,
                )
            )

        prediction[test_mask] = torch.stack(seed_predictions).mean(0)

    return prediction


def oof_mlp(
    agent: Tensor,
    target: Tensor,
    group: Tensor,
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    lr: float,
    weight_decay: float,
) -> Tensor:
    """LO-state-out fresh production-shaped continuation head predictions."""
    agent = agent.cpu()
    target = target.cpu()
    group = group.cpu()

    groups = sorted(set(group.tolist()))
    prediction = torch.zeros_like(target)

    for test_group in groups:
        test_mask = group == test_group
        remaining = [g for g in groups if g != test_group]
        seed_predictions = []

        for seed_index, seed in enumerate(seeds):
            val_group = remaining[seed_index % len(remaining)]
            fit_mask = (group != test_group) & (group != val_group)
            val_mask = group == val_group

            seed_predictions.append(
                fit_mlp_once(
                    agent[fit_mask],
                    target[fit_mask],
                    agent[val_mask],
                    target[val_mask],
                    agent[test_mask],
                    config,
                    seed=seed + test_group * 101,
                    steps=steps,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            )

        prediction[test_mask] = torch.stack(seed_predictions).mean(0)

    return prediction


def report_score(
    score: Tensor,
    target: Tensor,
    action: Tensor,
    *,
    permutations: int,
    seed: int,
) -> dict:
    return {
        "conditional": conditional_auc(score, target, action),
        "permutation": permutation_p_value(
            score,
            target,
            action,
            permutations=permutations,
            seed=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--linear-lr", type=float, default=3e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-3)
    parser.add_argument("--mlp-steps", type=int, default=800)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/counterfactual_interaction_localization.json"
        ),
    )
    args = parser.parse_args()

    base = Config()
    config = replace(
        base,
        transition="direct",
        time_mixer="attention",
    )

    saved = torch.load(args.forks, weights_only=False)
    encoder, world, heads = load_models(
        args.phase1a, args.phase2, base, config
    )
    data, replay = extract_exact_forks(
        saved, encoder, world, heads, config
    )

    print(
        f"exact replay: {replay['terminal_opportunity_states']} states, "
        f"{replay['examples']} examples",
        flush=True,
    )

    # Sanity: action identity itself must be chance under same-action AUC.
    action_identity = data.action.float()

    seeds = [config.seed + 4000 + i for i in range(args.seeds)]

    predictions = {
        "production_generated": data.production_generated,
        "production_observed": data.production_observed,
        "action_identity_only": action_identity,
    }

    for name, feature in (
        ("observed_latent", data.observed_latent),
        ("observed_readout", data.observed_readout),
        ("generated_latent", data.generated_latent),
        ("generated_readout", data.generated_readout),
    ):
        print(f"action-centered OOF linear: {name}", flush=True)
        predictions[f"linear_{name}"] = centered_oof_linear(
            feature,
            data.target,
            data.action,
            data.group,
            config,
            seeds=seeds,
            steps=args.linear_steps,
            lr=args.linear_lr,
            weight_decay=args.linear_weight_decay,
        )

    print("OOF production-shaped MLP: observed_readout", flush=True)
    predictions["mlp_observed_readout"] = oof_mlp(
        data.observed_readout,
        data.target,
        data.group,
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    print("OOF production-shaped MLP: generated_readout", flush=True)
    predictions["mlp_generated_readout"] = oof_mlp(
        data.generated_readout,
        data.target,
        data.group,
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    scores = {}
    for index, (name, score) in enumerate(predictions.items()):
        print(f"conditional metric: {name}", flush=True)
        scores[name] = report_score(
            score,
            data.target,
            data.action,
            permutations=args.permutations,
            seed=config.seed + 5000 + index,
        )

    varying = scores["action_identity_only"]["conditional"][
        "varying_actions"
    ]

    report = {
        "contract": {
            "arm": "direct-attention",
            "uses_exact_saved_gate_states": True,
            "all_17_actions_replayed": True,
            "evaluation": (
                "dead-vs-safe comparisons only within the same action"
            ),
            "linear_features": (
                "per-action centered using fit states only"
            ),
            "test_split": "leave-one-pre-action-state-out",
            "validation_split": "one different whole pre-action state",
            "probe_seeds": args.seeds,
            "permutations": args.permutations,
        },
        "replay": replay,
        "varying_actions": varying,
        "scores": scores,
        "interpretation_key": {
            "linear_observed_readout_high": (
                "real successor world state contains state-specific fatality"
            ),
            "linear_generated_latent_low": (
                "Direct prediction loses state-by-action fatality"
            ),
            "linear_generated_latent_high_generated_readout_low": (
                "world readout loses information preserved in generated latent"
            ),
            "mlp_generated_readout_high_production_generated_low": (
                "generated representation is usable but Phase-2 continuation "
                "training fails to exploit it"
            ),
            "action_identity_only": (
                "must be 0.5 conditional AUC by construction"
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
