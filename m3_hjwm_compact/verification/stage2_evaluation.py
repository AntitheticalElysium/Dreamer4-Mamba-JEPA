"""Raw, paired evaluation utilities for Stage-2 full-world controls."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from model import WorldState, cosine_distance
from phase_e_continuation_depth import evaluate_world as evaluate_continuation
from phase_e_same_target import (
    HORIZONS,
    HISTORY,
    WINDOW_OBS,
    evaluate_world as evaluate_reward,
    suffix_partition,
)
from phase_e_taskheads import clone_world_state, ranking_metrics
from stage1b_equal_update_analysis import (
    continuation_metrics,
    mean_metrics,
    ranking_arrays,
    reward_bootstrap_metrics,
    reward_metrics,
    zero_suffix_arrays,
)


LATENT_HORIZONS = (1, 2, 4, 8)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def latent_prediction_errors(
    world,
    arrays: dict,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, list[float]]:
    """Last-predictor cosine error to the same frozen target at each depth.

    This measures the predicted target-like latent before it is fed through the
    temporal core. Every depth ends at the identical final observation.
    """
    errors = {f"k{depth}": [] for depth in LATENT_HORIZONS}
    total = len(arrays["obs"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        obs = torch.from_numpy(arrays["obs"][start:stop]).to(device)
        actions = torch.from_numpy(arrays["actions"][start:stop]).to(device)
        previous = torch.from_numpy(
            arrays["previous_actions"][start:stop]
        ).to(device)
        batch = stop - start

        with _autocast(device):
            encoded = [
                world.online_encoder(obs[:, time])
                for time in range(WINDOW_OBS)
            ]
            target = world.target_encoder(obs[:, -1]).float()

            state = world.initial_state(batch, device)
            for time in range(HISTORY):
                index = world._previous_action_indices(previous[:, time])
                value = encoded[time] + world.action_input(index)[:, None]
                output, temporal = world.temporal.step(
                    value, state.temporal
                )
                state = WorldState(temporal, output, state.revision)
        base = state

        for depth in LATENT_HORIZONS:
            state = clone_world_state(base)
            real_times, imagined_actions = suffix_partition(depth)
            prediction = None
            with _autocast(device):
                for time in real_times:
                    index = world._previous_action_indices(
                        previous[:, time]
                    )
                    value = encoded[time] + world.action_input(index)[:, None]
                    output, temporal = world.temporal.step(
                        value, state.temporal
                    )
                    state = WorldState(temporal, output, state.revision)
                for action_index in imagined_actions:
                    state, _, _, prediction = world.imagine_step(
                        state,
                        actions[:, action_index],
                        deterministic_mode=True,
                    )
            if prediction is None:
                raise RuntimeError(f"depth {depth} produced no latent prediction")
            value = cosine_distance(
                prediction.selected.float(), target
            ).mean(-1)
            errors[f"k{depth}"].extend(value.cpu().tolist())
    return errors


@torch.no_grad()
def evaluate_arm(
    world,
    natural_arrays: dict,
    terminal_arrays: dict,
    actual_continue: np.ndarray,
    anchors: list[dict],
    device: torch.device,
) -> tuple[dict, dict]:
    """Return per-arm point metrics and sufficient raw rows for paired audit."""
    reward = evaluate_reward(world, natural_arrays, device)
    continuation = evaluate_continuation(
        world, terminal_arrays, actual_continue, device
    )
    ranking = ranking_metrics(world, anchors, device)
    ranking_rows = ranking.pop("rows")
    latent = latent_prediction_errors(world, natural_arrays, device)

    point = {
        "reward_depth": reward["metrics"],
        "continuation_depth": continuation["metrics"],
        "latent_depth": {
            key: {"cosine_error": float(np.mean(values))}
            for key, values in latent.items()
        },
        "ranking": ranking,
    }
    raw = {
        "reward_predictions": reward["predictions"],
        "continuation_predictions": continuation["predictions"],
        "latent_errors": latent,
        "ranking_rows": ranking_rows,
    }
    return point, raw


def gated_zero_suffix_arrays(
    rows: list[dict],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Raw and continuation-gated predictions for truly zero-return suffixes."""
    raw, clusters = zero_suffix_arrays(rows)
    gated, gated_clusters = [], []
    for row in rows:
        for name, actual in row["actual"].items():
            if abs(actual) <= 1e-9:
                gated.append(float(row["j_gated"][name]))
                gated_clusters.append(row["env_seed"])
    gated_clusters = np.asarray(gated_clusters)
    if not np.array_equal(clusters, gated_clusters):
        raise RuntimeError("raw/gated zero-suffix cluster alignment drift")
    raw["zero_suffix_gated_predicted_sum"] = np.asarray(gated)
    raw["zero_suffix_abs_gated_predicted_sum"] = np.abs(
        np.asarray(gated)
    )
    return raw, clusters


def _contrast(points: dict[str, dict], coefficients: dict[str, float]) -> dict:
    names = next(iter(points.values())).keys()
    return {
        name: float(sum(
            coefficient * points[arm][name]
            for arm, coefficient in coefficients.items()
        ))
        for name in names
        if all(points[arm].get(name) is not None for arm in coefficients)
    }


def _cluster_indices(clusters: np.ndarray, *, seed: int,
                     draws: int) -> list[np.ndarray]:
    unique = np.unique(clusters)
    members = {
        cluster: np.flatnonzero(clusters == cluster)
        for cluster in unique
    }
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([
            members[cluster]
            for cluster in rng.choice(unique, len(unique), replace=True)
        ])
        for _ in range(draws)
    ]


def _bootstrap_metric_contrasts(
    arm_values: dict[str, np.ndarray],
    labels: np.ndarray,
    clusters: np.ndarray,
    metric_fn,
    contrasts: dict[str, dict[str, float]],
    *,
    seed: int,
    draws: int,
) -> dict:
    point = {
        arm: metric_fn(values, labels)
        for arm, values in arm_values.items()
    }
    samples = {
        contrast_name: {
            metric: []
            for metric in _contrast(point, coefficients)
        }
        for contrast_name, coefficients in contrasts.items()
    }
    for indices in _cluster_indices(clusters, seed=seed, draws=draws):
        sampled = {
            arm: metric_fn(values[indices], labels[indices])
            for arm, values in arm_values.items()
        }
        for contrast_name, coefficients in contrasts.items():
            for metric, value in _contrast(sampled, coefficients).items():
                if np.isfinite(value):
                    samples[contrast_name][metric].append(value)
    output = {}
    for contrast_name, coefficients in contrasts.items():
        output[contrast_name] = {}
        for metric, value in _contrast(point, coefficients).items():
            values = samples[contrast_name][metric]
            output[contrast_name][metric] = {
                "delta": value,
                "ci95": [
                    float(x) for x in np.percentile(values, (2.5, 97.5))
                ],
            }
    return output


def _bootstrap_array_contrasts(
    arm_arrays: dict[str, dict[str, np.ndarray]],
    clusters: np.ndarray,
    contrasts: dict[str, dict[str, float]],
    *,
    seed: int,
    draws: int,
) -> dict:
    point = {
        arm: mean_metrics(arrays)
        for arm, arrays in arm_arrays.items()
    }
    samples = {
        contrast_name: {
            metric: []
            for metric in _contrast(point, coefficients)
        }
        for contrast_name, coefficients in contrasts.items()
    }
    for indices in _cluster_indices(clusters, seed=seed, draws=draws):
        sampled = {
            arm: mean_metrics({
                metric: values[indices]
                for metric, values in arrays.items()
            })
            for arm, arrays in arm_arrays.items()
        }
        for contrast_name, coefficients in contrasts.items():
            for metric, value in _contrast(sampled, coefficients).items():
                samples[contrast_name][metric].append(value)
    output = {}
    for contrast_name, coefficients in contrasts.items():
        output[contrast_name] = {}
        for metric, value in _contrast(point, coefficients).items():
            output[contrast_name][metric] = {
                "delta": value,
                "ci95": [
                    float(x) for x in np.percentile(
                        samples[contrast_name][metric], (2.5, 97.5)
                    )
                ],
            }
    return output


def paired_analysis(
    raw_arms: dict[str, dict],
    *,
    reward_actual: np.ndarray,
    reward_clusters: np.ndarray,
    continue_actual: np.ndarray,
    continue_clusters: np.ndarray,
    latent_clusters: np.ndarray,
    contrasts: dict[str, dict[str, float]],
    draws: int = 2_000,
) -> dict:
    """Compute paired episode/env-cluster contrasts for every safety domain."""
    reward = {}
    continuation = {}
    latent = {}
    for depth in HORIZONS:
        key = f"k{depth}"
        reward_values = {
            arm: np.asarray(block["reward_predictions"][key])
            for arm, block in raw_arms.items()
        }
        reward[key] = {
            "points": {
                arm: reward_metrics(values, reward_actual)
                for arm, values in reward_values.items()
            },
            "contrasts": _bootstrap_metric_contrasts(
                reward_values,
                reward_actual,
                reward_clusters,
                reward_bootstrap_metrics,
                contrasts,
                seed=28_000 + depth,
                draws=draws,
            ),
        }

        continue_values = {
            arm: np.asarray(block["continuation_predictions"][key])
            for arm, block in raw_arms.items()
        }
        continuation[key] = {
            "points": {
                arm: continuation_metrics(values, continue_actual)
                for arm, values in continue_values.items()
            },
            "contrasts": _bootstrap_metric_contrasts(
                continue_values,
                continue_actual,
                continue_clusters,
                continuation_metrics,
                contrasts,
                seed=28_100 + depth,
                draws=draws,
            ),
        }

        if depth:
            latent_values = {
                arm: {"cosine_error": np.asarray(
                    block["latent_errors"][key]
                )}
                for arm, block in raw_arms.items()
            }
            latent[key] = {
                "points": {
                    arm: mean_metrics(values)
                    for arm, values in latent_values.items()
                },
                "contrasts": _bootstrap_array_contrasts(
                    latent_values,
                    latent_clusters,
                    contrasts,
                    seed=28_200 + depth,
                    draws=draws,
                ),
            }

    ranking_values, zero_values = {}, {}
    ranking_clusters = zero_clusters = None
    for arm, block in raw_arms.items():
        ranking_values[arm], clusters = ranking_arrays(
            block["ranking_rows"]
        )
        if ranking_clusters is None:
            ranking_clusters = clusters
        elif not np.array_equal(ranking_clusters, clusters):
            raise RuntimeError("ranking cluster alignment differs by arm")
        zero_values[arm], clusters = gated_zero_suffix_arrays(
            block["ranking_rows"]
        )
        if zero_clusters is None:
            zero_clusters = clusters
        elif not np.array_equal(zero_clusters, clusters):
            raise RuntimeError("zero-suffix cluster alignment differs by arm")

    ranking = {
        "points": {
            arm: mean_metrics(values)
            for arm, values in ranking_values.items()
        },
        "contrasts": _bootstrap_array_contrasts(
            ranking_values,
            ranking_clusters,
            contrasts,
            seed=28_300,
            draws=draws,
        ),
    }
    zero_suffix = {
        "points": {
            arm: mean_metrics(values)
            for arm, values in zero_values.items()
        },
        "contrasts": _bootstrap_array_contrasts(
            zero_values,
            zero_clusters,
            contrasts,
            seed=28_400,
            draws=draws,
        ),
    }

    return {
        "reward": reward,
        "continuation": continuation,
        "latent": latent,
        "ranking": ranking,
        "zero_suffix": zero_suffix,
        "bootstrap_draws": draws,
        "bootstrap_units": {
            "reward_continuation_latent": "episode",
            "ranking_zero_suffix": "environment_seed",
        },
    }
