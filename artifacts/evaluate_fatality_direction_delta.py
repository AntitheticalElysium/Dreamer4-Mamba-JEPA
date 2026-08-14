"""Decompose fatal successor prediction into status and directional change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import (
    atomic_torch,
    auc,
    cluster_auc_interval,
    finite_json,
)
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    scale = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / scale.clamp(min=1e-12))


def vector_stats(value: torch.Tensor, label: torch.Tensor) -> dict:
    value, label = value.float().cpu(), label.bool().cpu()

    def one(rows: torch.Tensor) -> dict:
        selected = value[rows]
        return {
            "count": len(selected),
            "mean": float(selected.mean()),
            "median": float(selected.median()),
            "std": float(selected.std(unbiased=False)),
            "q10": float(selected.quantile(0.1)),
            "q90": float(selected.quantile(0.9)),
        }

    return {
        "all": one(torch.ones_like(label)),
        "fatal": one(label),
        "safe": one(~label),
    }


def cluster_mean_interval(
    value: torch.Tensor,
    label: torch.Tensor,
    group: torch.Tensor,
    *,
    fatal: bool,
    samples: int,
    seed: int,
) -> list[float]:
    value, label, group = value.cpu(), label.bool().cpu(), group.long().cpu()
    unique = torch.tensor(sorted(set(group.tolist())))
    rng = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(samples):
        chosen = unique[
            torch.randint(len(unique), (len(unique),), generator=rng)
        ]
        indices = torch.cat([(group == item).nonzero().flatten() for item in chosen])
        selected = indices[label[indices] == fatal]
        estimates.append(float(value[selected].mean()))
    distribution = torch.tensor(estimates)
    return [
        float(distribution.quantile(0.025)),
        float(distribution.quantile(0.975)),
    ]


def classify_fatal_delta(
    true_mean: float,
    predicted_mean: float,
    true_ci: list[float],
    predicted_ci: list[float],
) -> str:
    true_resolved = true_ci[0] > 0 or true_ci[1] < 0
    predicted_resolved = predicted_ci[0] > 0 or predicted_ci[1] < 0
    if not true_resolved:
        return "true_fatal_delta_not_resolved"
    if predicted_resolved and true_mean * predicted_mean < 0:
        return "wrong_direction"
    if not predicted_resolved:
        return "predicted_change_not_resolved_from_status_quo"
    return "tracks_true_direction"


def conditional_metrics(
    true_delta: torch.Tensor,
    predicted_delta: torch.Tensor,
    label: torch.Tensor,
    group: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> dict | None:
    """Within-state/pair consequence contrast and slope, with group bootstrap."""
    rows = []
    for value in group.unique():
        selected = group == value
        target = label[selected]
        if not bool(target.any()) or not bool((~target).any()):
            continue
        truth, estimate = true_delta[selected], predicted_delta[selected]
        x = truth - truth.mean()
        y = estimate - estimate.mean()
        rows.append(
            (
                truth[target].mean() - truth[~target].mean(),
                estimate[target].mean() - estimate[~target].mean(),
                (x * y).sum(),
                x.square().sum(),
            )
        )
    if not rows:
        return None
    values = torch.tensor(rows, dtype=torch.float)
    true_contrast = float(values[:, 0].mean())
    predicted_contrast = float(values[:, 1].mean())
    slope = float(values[:, 2].sum() / values[:, 3].sum().clamp(min=1e-12))
    rng = torch.Generator().manual_seed(seed)
    contrasts, slopes = [], []
    for _ in range(samples):
        chosen = torch.randint(len(values), (len(values),), generator=rng)
        sample = values[chosen]
        contrasts.append(sample[:, 1].mean())
        slopes.append(sample[:, 2].sum() / sample[:, 3].sum().clamp(min=1e-12))
    contrasts, slopes = torch.stack(contrasts), torch.stack(slopes)
    return {
        "groups": len(values),
        "true_fatal_minus_safe": true_contrast,
        "predicted_fatal_minus_safe": predicted_contrast,
        "recovered_fraction": predicted_contrast / max(abs(true_contrast), 1e-12),
        "within_group_predicted_vs_true_slope": slope,
        "predicted_contrast_ci95": [
            float(contrasts.quantile(0.025)),
            float(contrasts.quantile(0.975)),
        ],
        "slope_ci95": [
            float(slopes.quantile(0.025)),
            float(slopes.quantile(0.975)),
        ],
    }


def delta_metrics(
    current: torch.Tensor,
    target: torch.Tensor,
    predicted: torch.Tensor,
    label: torch.Tensor,
    action: torch.Tensor,
    group: torch.Tensor,
    direction: torch.Tensor,
    means: torch.Tensor,
    *,
    bootstraps: int,
    seed: int,
    conditional_group: torch.Tensor | None = None,
) -> dict:
    current = current.flatten(1).float().cpu()
    target = target.flatten(1).float().cpu()
    predicted = predicted.flatten(1).float().cpu()
    label, action, group = label.bool().cpu(), action.long().cpu(), group.long().cpu()
    conditional_group = (
        group if conditional_group is None else conditional_group.long().cpu()
    )
    direction, means = direction.float().cpu(), means.float().cpu()
    center = means[action]
    start = (current - center) @ direction
    true = (target - center) @ direction
    estimate = (predicted - center) @ direction
    true_delta = true - start
    predicted_delta = estimate - start
    error = predicted_delta - true_delta
    centered_true = true_delta - true_delta.mean()
    slope = float(
        (centered_true * (predicted_delta - predicted_delta.mean())).sum()
        / centered_true.square().sum().clamp(min=1e-12)
    )
    true_fatal_ci = cluster_mean_interval(
        true_delta,
        label,
        group,
        fatal=True,
        samples=bootstraps,
        seed=seed,
    )
    predicted_fatal_ci = cluster_mean_interval(
        predicted_delta,
        label,
        group,
        fatal=True,
        samples=bootstraps,
        seed=seed + 1,
    )
    true_sep = vector_stats(true_delta, label)
    predicted_sep = vector_stats(predicted_delta, label)
    true_auc = auc(true_delta, label)
    predicted_auc = auc(predicted_delta, label)
    return {
        "examples": len(label),
        "fatal_examples": int(label.sum()),
        "status": vector_stats(start, label),
        "true_successor": vector_stats(true, label),
        "predicted_successor": vector_stats(estimate, label),
        "true_delta": true_sep,
        "predicted_delta": predicted_sep,
        "directional_error": vector_stats(error, label),
        "true_delta_auc": true_auc,
        "true_delta_auc_ci95": cluster_auc_interval(
            true_delta,
            label,
            group,
            samples=bootstraps,
            seed=seed + 2,
        ),
        "predicted_delta_auc": predicted_auc,
        "predicted_delta_auc_ci95": cluster_auc_interval(
            predicted_delta,
            label,
            group,
            samples=bootstraps,
            seed=seed + 3,
        ),
        "predicted_vs_true": {
            "correlation": _correlation(true_delta, predicted_delta),
            "slope": slope,
            "mae": float(error.abs().mean()),
            "rmse": float(error.square().mean().sqrt()),
            "sign_agreement": float(
                (true_delta.sign() == predicted_delta.sign()).float().mean()
            ),
            "absolute_change_ratio": float(
                predicted_delta.abs().mean()
                / true_delta.abs().mean().clamp(min=1e-12)
            ),
        },
        "fatal_mean_ci95": {
            "true": true_fatal_ci,
            "predicted": predicted_fatal_ci,
        },
        "fatal_failure_mode": classify_fatal_delta(
            true_sep["fatal"]["mean"],
            predicted_sep["fatal"]["mean"],
            true_fatal_ci,
            predicted_fatal_ci,
        ),
        "conditional_consequence": conditional_metrics(
            true_delta,
            predicted_delta,
            label,
            conditional_group,
            samples=bootstraps,
            seed=seed + 4,
        ),
    }


def archive_current(records: list[dict]) -> torch.Tensor:
    return torch.cat(
        [record["latents"][record["transitions"]] for record in records]
    )


@torch.no_grad()
def fork_start_latents(
    phase1a: Path,
    trajectory_phase2: Path,
    forks: Path,
    cache: Path,
) -> torch.Tensor:
    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    contract = {
        "version": "fatality-delta-fork-starts-v1",
        "phase1a": file_digest(phase1a),
        "trajectory_phase2": file_digest(trajectory_phase2),
        "forks": file_digest(forks),
        "implementation": implementation_digests(Path(__file__)),
    }
    if cache.exists():
        saved = torch.load(cache, weights_only=False, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("fork-start cache contract changed")
        return saved["latents"]

    wanted = torch.load(forks, weights_only=False, map_location="cpu")
    encoder, world, heads = load_models(
        phase1a, trajectory_phase2, base, config
    )
    by_seed: dict[int, dict[int, int]] = {}
    for row, (seed, step) in enumerate(zip(wanted["seed"], wanted["step"])):
        by_seed.setdefault(int(seed), {})[int(step)] = row
    latents: list[torch.Tensor | None] = [None] * len(wanted["seed"])

    for seed, steps in sorted(by_seed.items()):
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full(
            (1, 1), config.n_actions, dtype=torch.long, device=config.device
        )
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
        for index in range(max(steps) + 1):
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(
                world, encoder, state, incoming, patches, world_rng, config
            )
            logits = heads(agent)["policy"][:, -1, 0]
            action = int(
                torch.multinomial(logits.softmax(-1), 1, generator=policy_rng)
            )
            if index in steps:
                row = steps[index]
                if action != int(wanted["trajectory_action"][row]):
                    raise AssertionError("fork trajectory action did not replay")
                latents[row] = state.world.latent[0, -1].cpu()
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1
            )
            if index in steps:
                row = steps[index]
                if bool(terminated) != bool(wanted["trajectory_death"][row]):
                    raise AssertionError("fork trajectory outcome did not replay")
            incoming.fill_(action)
            if terminated or truncated:
                if index < max(steps):
                    raise RuntimeError("trajectory ended before a saved fork")
                break

    if any(value is None for value in latents):
        raise RuntimeError("failed to reconstruct every fork start latent")
    stacked = torch.stack([value for value in latents if value is not None])
    atomic_torch(cache, {"contract": contract, "latents": stacked})
    return stacked


def plot_delta(
    path: Path,
    current: torch.Tensor,
    target: torch.Tensor,
    predicted: torch.Tensor,
    label: torch.Tensor,
    action: torch.Tensor,
    direction: torch.Tensor,
    means: torch.Tensor,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current, target, predicted = (
        value.flatten(1).float().cpu() for value in (current, target, predicted)
    )
    label, action = label.bool().cpu(), action.long().cpu()
    center = means.float().cpu()[action]
    start = (current - center) @ direction.float().cpu()
    true_delta = (target - center) @ direction.float().cpu() - start
    predicted_delta = (predicted - center) @ direction.float().cpu() - start
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for values, name, color, style in (
        (true_delta[label], "true fatal", "tab:red", "-"),
        (predicted_delta[label], "pred fatal", "tab:red", "--"),
        (true_delta[~label], "true safe", "tab:blue", "-"),
        (predicted_delta[~label], "pred safe", "tab:blue", "--"),
    ):
        axes[0].hist(
            values.numpy(), bins=25, density=True, histtype="step", label=name,
            color=color, linestyle=style,
        )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("fatality-direction delta")
    axes[0].legend(fontsize=8)
    axes[1].scatter(
        true_delta[~label], predicted_delta[~label], s=9, alpha=0.45,
        color="tab:blue", label="safe",
    )
    axes[1].scatter(
        true_delta[label], predicted_delta[label], s=9, alpha=0.45,
        color="tab:red", label="fatal",
    )
    bound = float(torch.stack([true_delta.abs().max(), predicted_delta.abs().max()]).max())
    axes[1].plot([-bound, bound], [-bound, bound], color="black", linewidth=0.8)
    axes[1].axhline(0, color="grey", linewidth=0.6)
    axes[1].axvline(0, color="grey", linewidth=0.6)
    axes[1].set_xlabel("true delta")
    axes[1].set_ylabel("predicted delta")
    axes[1].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive-features", type=Path, required=True)
    parser.add_argument("--policy-features", type=Path, required=True)
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--fork-starts", type=Path, required=True)
    parser.add_argument("--archive-world", action="append", required=True)
    parser.add_argument("--policy-world", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive-path", default="reset16")
    parser.add_argument("--pools", nargs="+", default=("combined", "support"))
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    forks = torch.load(args.forks, weights_only=False, map_location="cpu")
    starts = fork_start_latents(
        args.phase1a, args.trajectory_phase2, args.forks, args.fork_starts
    )
    archive_start = archive_current(prepared["records"])
    inputs = {
        "prepared": file_digest(args.prepared),
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "archive_features": {
            name: file_digest(args.archive_features / f"{name}.pt")
            for name in args.archive_world
        },
        "policy_features": {
            name: file_digest(args.policy_features / f"{name}.pt")
            for name in args.policy_world
        },
    }
    contract = {
        "version": "fatality-direction-delta-v1",
        "inputs": inputs,
        "implementation": implementation_digests(Path(__file__)),
        "archive_worlds": args.archive_world,
        "policy_worlds": args.policy_world,
        "archive_path": args.archive_path,
        "pools": args.pools,
        "bootstraps": args.bootstraps,
        "direction": "fixed TRAIN observed-successor fatality direction; never refit",
        "centering": "fixed TRAIN action mean applied to start and both successors; cancels in deltas",
        "uncertainty": "whole archive episode or whole policy trajectory seed bootstrap",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("fatality-delta contract changed")
    atomic_json(contract_path, contract)

    direction, means = prepared["direction"], prepared["action_means"]
    report = {"contract": contract, "worlds": {}}
    worlds = list(dict.fromkeys(args.archive_world + args.policy_world))
    for world_index, name in enumerate(worlds):
        world_report = {"archive": {}, "policy_forks": {}}
        if name in args.archive_world:
            archive_payload = torch.load(
                args.archive_features / f"{name}.pt",
                weights_only=False,
                map_location="cpu",
            )
            expected_target = torch.cat([
                record["latents"][record["transitions"] + 1]
                for record in prepared["records"]
            ])
            for path_index, (prediction_path, data) in enumerate(
                archive_payload["paths"].items()
            ):
                if prediction_path != args.archive_path:
                    continue
                if not torch.equal(data["target"], expected_target):
                    raise AssertionError("archive targets changed order")
                path_report = {}
                for pool_index, pool in enumerate(args.pools):
                    if pool not in ("combined", "expert", "support"):
                        raise ValueError(f"unknown archive pool: {pool}")
                    mask = torch.ones(len(data["label"]), dtype=torch.bool)
                    if pool != "combined":
                        mask = torch.tensor([value == pool for value in data["pool"]])
                    path_report[pool] = delta_metrics(
                        archive_start[mask], data["target"][mask], data["predicted"][mask],
                        data["label"][mask], data["action"][mask], data["group"][mask],
                        direction, means, bootstraps=args.bootstraps,
                        seed=Config().seed + 9600 + world_index * 100 + path_index * 10 + pool_index,
                    )
                world_report["archive"][prediction_path] = path_report
            support = torch.tensor([
                value == "support"
                for value in archive_payload["paths"][args.archive_path]["pool"]
            ])
            plotted = archive_payload["paths"][args.archive_path]
            plot_delta(
                args.out / "plots" / f"archive_support_{name}.png",
                archive_start[support], plotted["target"][support],
                plotted["predicted"][support], plotted["label"][support],
                plotted["action"][support],
                direction, means, f"Archive support: {name}",
            )

        if name in args.policy_world:
            policy_payload = torch.load(
                args.policy_features / f"{name}.pt",
                weights_only=False,
                map_location="cpu",
            )["data"]
            group = policy_payload["group"].long()
            current = starts[group]
            trajectory = (
                policy_payload["action"].long()
                == forks["trajectory_action"][group]
            )
            policy_group = forks["seed"][group].long()
            for split_index, (split, mask) in enumerate((
                ("all_actions", torch.ones_like(trajectory)),
                ("trajectory_action", trajectory),
                ("other_16_actions", ~trajectory),
            )):
                world_report["policy_forks"][split] = delta_metrics(
                    current[mask], policy_payload["observed_latent"][mask],
                    policy_payload["generated_latent"][mask], policy_payload["target"][mask],
                    policy_payload["action"][mask], policy_group[mask], direction, means,
                    bootstraps=args.bootstraps,
                    seed=Config().seed + 9800 + world_index * 10 + split_index,
                    conditional_group=group[mask],
                )
            plot_delta(
                args.out / "plots" / f"policy_forks_{name}.png",
                current, policy_payload["observed_latent"], policy_payload["generated_latent"],
                policy_payload["target"], policy_payload["action"], direction, means,
                f"Policy forks: {name}",
            )
        report["worlds"][name] = world_report
        print(f"fatality delta complete: {name}", flush=True)

    atomic_json(args.out / "report.json", finite_json(report))
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
