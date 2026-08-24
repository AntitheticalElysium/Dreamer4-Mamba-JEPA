"""Shared mechanics for the logged-transition geometry experiment."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from d4mj.config import Config
from d4mj.state import WorldState
from d4mj.transition import World, advance, commit_inputs


LATENT_WIDTH = Config().n_spatial * Config().d_spatial


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def auc(score: Tensor, target: Tensor) -> float:
    score = score.detach().float().cpu().flatten()
    target = target.detach().bool().cpu().flatten()
    positive, negative = score[target], score[~target]
    if not len(positive) or not len(negative):
        return float("nan")
    delta = positive[:, None] - negative[None]
    return float((delta.gt(0).float() + 0.5 * delta.eq(0).float()).mean())


def _safe_before(episode, terminal: int) -> tuple[int, bool]:
    candidates = torch.arange(terminal)
    candidates = candidates[
        ~episode.terminated[:terminal] & ~episode.truncated[:terminal]
    ]
    if not len(candidates):
        raise ValueError("terminal episode has no preceding safe transition")
    same = candidates[
        episode.actions_taken[candidates] == episode.actions_taken[terminal]
    ]
    return (int(same[-1]), True) if len(same) else (int(candidates[-1]), False)


def terminal_pair_rows(episodes: list, pool: str) -> tuple[dict[str, Tensor], list[dict]]:
    """One terminal and one preceding safe logged transition per terminal episode."""
    feature, target, action, group = [], [], [], []
    records = []
    action_matched = 0
    for episode_index, episode in enumerate(episodes):
        terminals = episode.terminated.nonzero().flatten().tolist()
        for terminal in terminals:
            safe, matched = _safe_before(episode, terminal)
            action_matched += int(matched)
            current_group = len(records)
            transitions = torch.tensor([safe, terminal], dtype=torch.long)
            labels = torch.tensor([0.0, 1.0])
            actions = episode.actions_taken[transitions].long()
            successors = episode.latents[transitions + 1].float()
            feature.extend(successors)
            target.extend(labels)
            action.extend(actions)
            group.extend(torch.full((2,), current_group, dtype=torch.long))
            records.append(
                {
                    "pool": pool,
                    "episode_index": episode_index,
                    "latents": episode.latents.float(),
                    "actions_taken": episode.actions_taken.long(),
                    "terminated": episode.terminated.bool(),
                    "truncated": episode.truncated.bool(),
                    "transitions": transitions,
                    "labels": labels,
                    "same_action_safe_match": matched,
                }
            )
    if not records:
        raise ValueError(f"{pool} has no terminal episodes")
    return {
        "feature": torch.stack(feature),
        "target": torch.stack(target),
        "action": torch.stack(action),
        "group": torch.stack(group),
        "same_action_safe_pairs": torch.tensor(action_matched),
    }, records


def compact_records(records: list[dict], context: int) -> list[dict]:
    """Store each selected transition with only its reset-window prefix."""
    compact = []
    for group, record in enumerate(records):
        for local, transition in enumerate(record["transitions"].tolist()):
            start = max(0, transition + 2 - context)
            compact.append(
                {
                    "pool": record["pool"],
                    "episode_index": record["episode_index"],
                    "group": group,
                    "latents": record["latents"][start : transition + 2].clone(),
                    "actions_taken": record["actions_taken"][start : transition + 1].clone(),
                    "terminated": record["terminated"][start : transition + 1].clone(),
                    "truncated": record["truncated"][start : transition + 1].clone(),
                    "transitions": torch.tensor([transition - start]),
                    "labels": record["labels"][local : local + 1].clone(),
                    "same_action_safe_match": record["same_action_safe_match"],
                }
            )
    return compact


def action_means(
    feature: Tensor, action: Tensor, mask: Tensor, n_actions: int
) -> Tensor:
    means = torch.empty(n_actions, feature.shape[1], dtype=feature.dtype)
    fallback = feature[mask].mean(0)
    for value in range(n_actions):
        rows = mask & (action == value)
        means[value] = feature[rows].mean(0) if bool(rows.any()) else fallback
    return means


def _weighted_bce(logits: Tensor, target: Tensor) -> Tensor:
    positive = target.sum().clamp(min=1.0)
    negative = (1.0 - target).sum().clamp(min=1.0)
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=(negative / positive).detach()
    )


def fit_fatal_direction(
    rows: dict[str, Tensor],
    config: Config,
    *,
    seeds: list[int],
    steps: int,
) -> tuple[Tensor, Tensor, dict]:
    """Fit on real TRAIN successors and return one unit direction in raw Z* space."""
    feature = rows["feature"].flatten(1).float().cpu()
    target = rows["target"].float().cpu()
    action = rows["action"].long().cpu()
    group = rows["group"].long().cpu()
    unique = torch.tensor(sorted(set(group.tolist())))
    order = unique[
        torch.randperm(
            len(unique), generator=torch.Generator().manual_seed(config.seed + 8100)
        )
    ]
    validation_count = max(1, round(0.2 * len(order)))
    validation_groups = set(order[-validation_count:].tolist())
    validation = torch.tensor([int(value) in validation_groups for value in group])
    fit = ~validation
    means = action_means(feature, action, fit, config.n_actions)
    centered = feature - means[action]
    mean = centered[fit].mean(0, keepdim=True)
    std = centered[fit].std(0, unbiased=False, keepdim=True).clamp(min=1e-5)
    standardized = (centered - mean) / std

    directions, validation_scores = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = nn.Linear(feature.shape[1], 1).to(config.device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=3e-3, weight_decay=1e-3
        )
        best_key = (-1.0, float("-inf"))
        best = None
        fit_x, fit_y = standardized[fit].to(config.device), target[fit].to(config.device)
        validation_x = standardized[validation].to(config.device)
        for step in range(steps):
            logits = model(fit_x)[:, 0]
            loss = _weighted_bce(logits, fit_y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step % 10 == 0 or step + 1 == steps:
                with torch.no_grad():
                    probability = model(validation_x)[:, 0].sigmoid().cpu()
                value = auc(probability, target[validation])
                bce = float(
                    F.binary_cross_entropy(
                        probability.clamp(1e-7, 1 - 1e-7), target[validation]
                    )
                )
                key = (value, -bce)
                if key > best_key:
                    best_key, best = key, copy.deepcopy(model.state_dict())
        if best is None:
            raise AssertionError("fatality probe never produced a checkpoint")
        model.load_state_dict(best)
        raw = (model.weight.detach().cpu()[0] / std[0]).float()
        raw = raw / raw.norm().clamp(min=1e-12)
        projected = centered @ raw
        if projected[target.bool()].mean() < projected[~target.bool()].mean():
            raw = -raw
        directions.append(raw)
        validation_scores.append(best_key[0])

    direction = torch.stack(directions).mean(0)
    direction = direction / direction.norm().clamp(min=1e-12)
    validation_projection = centered[validation] @ direction
    report = {
        "examples": len(target),
        "terminal_examples": int(target.sum()),
        "groups": len(unique),
        "fit_groups": len(unique) - validation_count,
        "validation_groups": validation_count,
        "seed_validation_auc": validation_scores,
        "ensemble_validation_auc": auc(
            validation_projection, target[validation]
        ),
        "same_action_safe_pairs": int(rows["same_action_safe_pairs"]),
        "direction_norm": float(direction.norm()),
    }
    return direction, means, report


def precision_from_covariance(
    covariance: Tensor, shrinkage: float
) -> tuple[Tensor, dict]:
    """Fixed shrinkage precision with unit mean eigenweight."""
    if not 0.0 < shrinkage < 1.0:
        raise ValueError("shrinkage must be in (0, 1)")
    covariance = covariance.double().cpu()
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not torch.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -1e-8:
        raise ValueError("covariance is not positive semidefinite")
    eigenvalues = eigenvalues.clamp(min=0)
    mean_variance = eigenvalues.mean()
    regularized = (1.0 - shrinkage) * eigenvalues + shrinkage * mean_variance
    inverse = regularized.reciprocal()
    inverse = inverse / inverse.mean()
    precision = (eigenvectors * inverse[None]) @ eigenvectors.T
    probability = eigenvalues.clamp(min=0)
    probability = probability / probability.sum().clamp(min=1e-30)
    effective_rank = float(torch.exp(-(probability * probability.clamp(min=1e-30).log()).sum()))
    report = {
        "width": covariance.shape[0],
        "shrinkage": shrinkage,
        "trace": float(eigenvalues.sum()),
        "mean_variance": float(mean_variance),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "regularized_condition": float(regularized.max() / regularized.min()),
        "effective_rank": effective_rank,
        "mean_precision_eigenvalue": float(inverse.mean()),
    }
    return precision.float(), report | {"covariance": covariance.float()}


def regularized_precision(samples: Tensor, shrinkage: float) -> tuple[Tensor, dict]:
    """Estimate covariance from samples, then construct its fixed precision."""
    x = samples.flatten(1).double().cpu()
    x = x - x.mean(0, keepdim=True)
    covariance = x.T @ x / max(1, len(x) - 1)
    precision, report = precision_from_covariance(covariance, shrinkage)
    return precision, report | {"samples": len(x)}


def quadratic_error(predicted: Tensor, target: Tensor, precision: Tensor) -> Tensor:
    error = (predicted - target).flatten(2)
    metric = precision.to(error.device, error.dtype)
    return torch.einsum("bti,ij,btj->bt", error, metric, error) / error.shape[-1]


def direct_metric_loss(
    world: World,
    batch,
    rng: torch.Generator,
    config: Config,
    precision: Tensor,
) -> Tensor:
    """Production Direct objective with only its Euclidean metric replaced."""
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, _, memory = world(
        None, batch.led_to_action, committed, conditioning
    )
    predicted = world.predict(features[:, :-1], batch.led_to_action[:, 1:])
    teacher = quadratic_error(predicted, batch.latents[:, 1:], precision).mean(1)

    length = batch.latents.shape[1]
    if length >= 3:
        prefix, _, memory = world(
            None,
            batch.led_to_action[:, :-2],
            committed[:, :-2],
            conditioning[:, :-2],
        )
        state = WorldState(
            batch.latents[:, -3:-2], memory, length - 2, prefix[:, -1:]
        )
        first, _ = advance(
            world, state, batch.led_to_action[:, -2:-1], rng, config
        )
        second, _ = advance(
            world, first, batch.led_to_action[:, -1:], rng, config
        )
        first_error = quadratic_error(
            first.latent, batch.latents[:, -2:-1], precision
        )[:, 0]
        second_error = quadratic_error(
            second.latent, batch.latents[:, -1:], precision
        )[:, 0]
        teacher = teacher + (first_error + second_error) / 2

    mask = batch.rows("dynamics").to(teacher.device).float()
    return (teacher * mask).sum() / mask.sum().clamp(min=1.0)


def _led_to(actions: Tensor, start: int, end: int, config: Config) -> Tensor:
    positions = torch.arange(start, end)
    incoming = positions - 1
    return torch.where(
        incoming >= 0,
        actions[incoming.clamp(min=0)],
        torch.tensor(config.n_actions),
    ).long()


@torch.no_grad()
def predict_record(
    world: World, record: dict, path: str, config: Config
) -> Tensor:
    """Predict selected logged successors under a reset window or recurrent scan."""
    latents = record["latents"]
    actions = record["actions_taken"]
    transitions = record["transitions"].tolist()
    predictions: dict[int, Tensor] = {}
    rng = torch.Generator(device=config.device).manual_seed(
        config.seed + 8200 + int(record["episode_index"])
    )

    if path.startswith("reset"):
        length = int(path.removeprefix("reset"))
        for transition in transitions:
            start = max(0, transition + 2 - length)
            block = latents[start : transition + 1][None].to(config.device)
            led_to = _led_to(actions, start, transition + 1, config)[None].to(
                config.device
            )
            committed, conditioning = commit_inputs(block, rng, config)
            features, _, _ = world(None, led_to, committed, conditioning)
            action = actions[transition].view(1, 1).to(config.device)
            predictions[transition] = world.predict(features[:, -1:], action)[0, 0].cpu()
    elif path == "recurrent":
        memory = None
        wanted = set(transitions)
        final = max(wanted) + 1
        for start in range(0, final, config.sequence_long):
            end = min(final, start + config.sequence_long)
            block = latents[start:end][None].to(config.device)
            led_to = _led_to(actions, start, end, config)[None].to(config.device)
            committed, conditioning = commit_inputs(block, rng, config)
            features, _, memory = world(
                memory, led_to, committed, conditioning, offset=start
            )
            for transition in sorted(wanted.intersection(range(start, end))):
                local = transition - start
                action = actions[transition].view(1, 1).to(config.device)
                predictions[transition] = world.predict(
                    features[:, local : local + 1], action
                )[0, 0].cpu()
    else:
        raise ValueError(f"unknown prediction path: {path}")
    return torch.stack([predictions[value] for value in transitions])


def predict_records(
    world: World, records: list[dict], path: str, config: Config
) -> dict[str, Tensor | list[str]]:
    predicted, target, label, action, group, pool = [], [], [], [], [], []
    for index, record in enumerate(records):
        predicted.append(predict_record(world, record, path, config))
        transitions = record["transitions"]
        target.append(record["latents"][transitions + 1])
        label.append(record["labels"])
        action.append(record["actions_taken"][transitions])
        group.append(
            torch.full(
                (len(transitions),), record.get("group", index), dtype=torch.long
            )
        )
        pool.extend([record["pool"]] * len(transitions))
    return {
        "predicted": torch.cat(predicted),
        "target": torch.cat(target),
        "label": torch.cat(label),
        "action": torch.cat(action),
        "group": torch.cat(group),
        "pool": pool,
    }


def _separation(score: Tensor, label: Tensor) -> dict[str, float]:
    score, label = score.float().cpu(), label.bool().cpu()
    dead, alive = score[label], score[~label]
    pooled = torch.cat(
        [dead - dead.mean(), alive - alive.mean()]
    ).pow(2).mean().sqrt().clamp(min=1e-12)
    return {
        "auc": auc(score, label),
        "dead_mean": float(dead.mean()),
        "alive_mean": float(alive.mean()),
        "mean_gap": float(dead.mean() - alive.mean()),
        "standardized_gap": float((dead.mean() - alive.mean()) / pooled),
    }


def cluster_auc_interval(
    score: Tensor,
    label: Tensor,
    group: Tensor,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Episode-cluster bootstrap interval for a paired fatality AUC."""
    score, label, group = score.cpu(), label.cpu(), group.cpu()
    unique = torch.tensor(sorted(set(group.tolist())))
    rng = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(samples):
        chosen = unique[
            torch.randint(len(unique), (len(unique),), generator=rng)
        ]
        indices = torch.cat([(group == value).nonzero().flatten() for value in chosen])
        values.append(auc(score[indices], label[indices]))
    distribution = torch.tensor(values)
    return [
        float(distribution.quantile(0.025)),
        float(distribution.quantile(0.975)),
    ]


def projection_metrics(
    feature: Tensor,
    label: Tensor,
    action: Tensor,
    group: Tensor,
    direction: Tensor,
    means: Tensor,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    score = (feature.flatten(1).float() - means[action.long()]) @ direction.float()
    result = _separation(score, label)
    result["auc_ci95"] = cluster_auc_interval(
        score,
        label,
        group,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return result


def _correlation(first: Tensor, second: Tensor) -> float:
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    scale = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / scale.clamp(min=1e-12))


def geometry_metrics(
    data: dict,
    direction: Tensor,
    means: Tensor,
    covariance: Tensor,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict:
    predicted = data["predicted"].flatten(1).float()
    target = data["target"].flatten(1).float()
    label = data["label"].bool()
    action = data["action"].long()
    group = data["group"].long()
    direction = direction.float()
    target_projection = (target - means[action]) @ direction
    predicted_projection = (predicted - means[action]) @ direction
    error = predicted - target
    directional_error = error @ direction
    squared_norm = error.square().sum(1)
    direction_mse = directional_error.square().mean()
    target_variance = target_projection.var(unbiased=False)
    total_mse = squared_norm.mean() / target.shape[1]
    orthogonal = (squared_norm - directional_error.square()).clamp(min=0)
    direction_variance_train = float(direction @ covariance @ direction)
    trace = float(torch.trace(covariance))
    target_separation = _separation(target_projection, label)
    predicted_separation = _separation(predicted_projection, label)
    target_separation["auc_ci95"] = cluster_auc_interval(
        target_projection,
        label,
        group,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    predicted_separation["auc_ci95"] = cluster_auc_interval(
        predicted_projection,
        label,
        group,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    return {
        "examples": len(label),
        "terminal_examples": int(label.sum()),
        "total_mse": float(total_mse),
        "direction_mse": float(direction_mse),
        "direction_mse_over_target_variance": float(
            direction_mse / target_variance.clamp(min=1e-12)
        ),
        "orthogonal_mse_per_dimension": float(
            orthogonal.mean() / max(1, target.shape[1] - 1)
        ),
        "direction_error_share": float(
            direction_mse / squared_norm.mean().clamp(min=1e-12)
        ),
        "target_projection_variance": float(target_variance),
        "train_direction_variance": direction_variance_train,
        "train_direction_variance_share": direction_variance_train / trace,
        "train_direction_variance_relative_to_isotropic": (
            direction_variance_train * target.shape[1] / trace
        ),
        "target_prediction_projection_correlation": _correlation(
            target_projection, predicted_projection
        ),
        "target_separation": target_separation,
        "predicted_separation": predicted_separation,
    }


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
