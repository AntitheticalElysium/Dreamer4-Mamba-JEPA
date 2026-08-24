"""Measure fixed-(state, action) successor modes on saved terminal-critical forks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.representation import Encoder, pack
from d4mj.state import WorldState
from d4mj.transition import World, advance, observe

INFERENCE_BATCH = 8


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_world(path: Path, config: Config) -> World:
    world = World(config).to(config.device)
    load(path, config, part0=world)
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return world


def _repeat_memory(memory, copies: int):
    if memory is None:
        return None
    return tuple(
        tuple(tensor.repeat((copies,) + (1,) * (tensor.ndim - 1)) for tensor in pair)
        for pair in memory
    )


def _repeat_state(state: WorldState, copies: int) -> WorldState:
    return WorldState(
        latent=state.latent.repeat((copies,) + (1,) * (state.latent.ndim - 1)),
        memory=_repeat_memory(state.memory, copies),
        step=state.step,
        features=(
            None
            if state.features is None
            else state.features.repeat((copies,) + (1,) * (state.features.ndim - 1))
        ),
    )


@torch.no_grad()
def _encode_modes(
    encoder: Encoder,
    encoder_memory,
    step_index: int,
    frames: list[torch.Tensor],
    deaths: torch.Tensor,
    config: Config,
):
    stacked = torch.stack(frames)
    flat = stacked.flatten(1)
    unique, inverse, counts = torch.unique(
        flat, dim=0, return_inverse=True, return_counts=True
    )
    unique_frames = unique.view(-1, *stacked.shape[1:])
    encoded = []
    for start in range(0, len(unique_frames), INFERENCE_BATCH):
        frames_batch = unique_frames[start : start + INFERENCE_BATCH]
        patches = patchify(frames_batch[:, None], config.patch).to(config.device)
        memory = _repeat_memory(encoder_memory, len(frames_batch))
        z, _, _ = encoder(patches, memory, offset=step_index)
        encoded.append(pack(z, config)[:, 0])
    centers = torch.cat(encoded)
    weights = counts.to(config.device).float() / len(frames)

    inverse_device = inverse.to(config.device)
    death_sum = torch.zeros(len(unique_frames), device=config.device)
    death_sum.scatter_add_(0, inverse_device, deaths.to(config.device).float())
    death_rate = death_sum / counts.to(config.device).float()
    return centers, weights, death_rate


def _mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).pow(2).flatten(-2).mean(-1)


def geometry_metrics(
    direct_latent: torch.Tensor,
    flow_latents: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    death_rate: torch.Tensor,
) -> dict:
    """Compare generated latents with the empirical fixed-pair successor modes."""
    mean = (centers * weights[:, None, None]).sum(0)
    direct_mode_distance = _mse(centers, direct_latent)
    mean_mode_distance = _mse(centers, mean)
    flow_to_modes = _mse(flow_latents[:, None], centers[None])

    distinct_death_modes = bool(
        (death_rate < 0.5).any() and (death_rate >= 0.5).any()
    )
    fatal = death_rate >= 0.5
    if distinct_death_modes:
        fatal_safe = _mse(centers[fatal, None], centers[~fatal][None])
        fatal_safe_separation = float(fatal_safe.min())
    else:
        fatal_safe_separation = None

    pairwise = _mse(centers[:, None], centers[None])
    diagonal = torch.eye(len(centers), dtype=torch.bool, device=centers.device)
    nonzero_pairwise = pairwise[~diagonal]
    death_probability = float((weights * death_rate).sum())
    direct_to_mean = _mse(direct_latent, mean)
    direct_to_nearest = direct_mode_distance.min()
    return {
        "mode_count": int(len(centers)),
        "mode_weights": [float(value) for value in weights.cpu()],
        "mode_death_rates": [float(value) for value in death_rate.cpu()],
        "death_probability": death_probability,
        "death_stochastic": 0.0 < death_probability < 1.0,
        "death_varies_between_observed_modes": distinct_death_modes,
        "fatal_safe_mode_separation_mse": fatal_safe_separation,
        "mode_variance_mse": float((weights * mean_mode_distance).sum()),
        "mean_to_nearest_mode_mse": float(mean_mode_distance.min()),
        "mean_pairwise_mode_mse": (
            float(nonzero_pairwise.mean()) if len(nonzero_pairwise) else 0.0
        ),
        "direct_to_mean_mse": float(direct_to_mean),
        "direct_to_nearest_mode_mse": float(direct_to_nearest),
        "direct_mean_closer_than_any_mode": bool(
            direct_to_mean < direct_to_nearest
        ),
        "direct_mean_advantage_mse": float(direct_to_nearest - direct_to_mean),
        "flow_precision_mse": float(flow_to_modes.min(1).values.mean()),
        "flow_coverage_mse": float(
            (weights * flow_to_modes.min(0).values).sum()
        ),
        "flow_worst_mode_coverage_mse": float(
            flow_to_modes.min(0).values.max()
        ),
    }


@torch.no_grad()
def _pair_metrics(
    direct: World,
    flow: World,
    direct_state: WorldState,
    flow_state: WorldState,
    action: int,
    centers: torch.Tensor,
    weights: torch.Tensor,
    death_rate: torch.Tensor,
    flow_samples: int,
    seed: int,
    direct_config: Config,
    flow_config: Config,
) -> dict:
    chosen = torch.tensor([[action]], device=direct_config.device)
    direct_rng = torch.Generator(device=direct_config.device).manual_seed(seed)
    predicted, _ = advance(direct, direct_state, chosen, direct_rng, direct_config)
    direct_latent = predicted.latent[0, 0]

    sampled = []
    for start in range(0, flow_samples, INFERENCE_BATCH):
        count = min(INFERENCE_BATCH, flow_samples - start)
        flow_state_batch = _repeat_state(flow_state, count)
        flow_actions = torch.full(
            (count, 1), action, dtype=torch.long, device=flow_config.device
        )
        flow_rng = torch.Generator(device=flow_config.device).manual_seed(
            seed + 2**18 + start
        )
        flow_prediction, _ = advance(
            flow, flow_state_batch, flow_actions, flow_rng, flow_config
        )
        sampled.append(flow_prediction.latent[:, 0])
    flow_latents = torch.cat(sampled)
    return geometry_metrics(
        direct_latent, flow_latents, centers, weights, death_rate
    )


def _median(values: list[float]) -> float | None:
    return float(torch.tensor(values).median()) if values else None


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"pairs": 0}

    numeric = (
        "mode_count",
        "death_probability",
        "mode_variance_mse",
        "mean_to_nearest_mode_mse",
        "mean_pairwise_mode_mse",
        "direct_to_mean_mse",
        "direct_to_nearest_mode_mse",
        "direct_mean_advantage_mse",
        "flow_precision_mse",
        "flow_coverage_mse",
        "flow_worst_mode_coverage_mse",
    )
    report = {"pairs": len(rows)}
    for key in numeric:
        values = [float(row[key]) for row in rows]
        report[f"mean_{key}"] = sum(values) / len(values)
        report[f"median_{key}"] = _median(values)
    report["direct_mean_closer_fraction"] = sum(
        row["direct_mean_closer_than_any_mode"] for row in rows
    ) / len(rows)
    return report


@torch.no_grad()
def diagnose(
    saved,
    encoder,
    trajectory_world,
    trajectory_heads,
    direct,
    flow,
    trajectory_config,
    direct_config,
    flow_config,
    draws: int,
    flow_samples: int,
):
    opportunity = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    selected = opportunity.nonzero().flatten().tolist()
    if not selected:
        raise RuntimeError("saved forks contain no terminal-opportunity states")
    key_to_row = {
        (int(saved["seed"][row]), int(saved["step"][row])): row for row in selected
    }
    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    rows, reproduced = [], set()
    device = direct_config.device
    for episode_seed in sorted(by_seed):
        wanted = by_seed[episode_seed]
        last = max(wanted)
        observation, env_state = reset(episode_seed)
        trajectory_state = direct_state = flow_state = None
        incoming = torch.full(
            (1, 1), direct_config.n_actions, dtype=torch.long, device=device
        )
        trajectory_rng = torch.Generator(device=device).manual_seed(episode_seed + 2**21)
        direct_rng = torch.Generator(device=device).manual_seed(episode_seed + 2**25)
        flow_rng = torch.Generator(device=device).manual_seed(episode_seed + 2**26)
        policy_rng = torch.Generator(device=device).manual_seed(episode_seed + 2**20)

        for index in range(last + 1):
            patches = patchify(observation[None, None], direct_config.patch).to(device)
            trajectory_state, trajectory_agent = observe(
                trajectory_world,
                encoder,
                trajectory_state,
                incoming,
                patches,
                trajectory_rng,
                trajectory_config,
            )
            direct_state, _ = observe(
                direct,
                encoder,
                direct_state,
                incoming,
                patches,
                direct_rng,
                direct_config,
            )
            flow_state, _ = observe(
                flow,
                encoder,
                flow_state,
                incoming,
                patches,
                flow_rng,
                flow_config,
            )

            key = (episode_seed, index)
            if key in key_to_row:
                saved_row = key_to_row[key]
                for action in range(direct_config.n_actions):
                    frames, deaths = [], []
                    for draw in range(draws):
                        stochastic_seed = (
                            episode_seed + index + 1
                            if draw == 0
                            else (
                                1_610_612_741
                                + episode_seed * 1_000_003
                                + index * 9_176
                                + action * 131
                                + draw * 65_537
                            )
                            % (2**31 - 1)
                        )
                        successor, _, _, terminated, _ = env_step(
                            env_state, action, stochastic_seed
                        )
                        if draw == 0 and bool(terminated) != bool(
                            saved["true_death"][saved_row, action]
                        ):
                            raise AssertionError(
                                f"saved truth mismatch seed={episode_seed} "
                                f"step={index} action={action}"
                            )
                        frames.append(successor)
                        deaths.append(terminated)

                    centers, weights, death_rate = _encode_modes(
                        encoder,
                        direct_state.encoder_memory,
                        direct_state.world.step,
                        frames,
                        torch.tensor(deaths),
                        direct_config,
                    )
                    row = _pair_metrics(
                        direct,
                        flow,
                        direct_state.world,
                        flow_state.world,
                        action,
                        centers,
                        weights,
                        death_rate,
                        flow_samples,
                        direct_config.seed
                        + 2**27
                        + episode_seed * 4099
                        + index * 17
                        + action,
                        direct_config,
                        flow_config,
                    )
                    row.update(
                        seed=episode_seed,
                        step=index,
                        action=action,
                        reference_draw_terminal=bool(deaths[0]),
                    )
                    rows.append(row)
                reproduced.add(key)

            logits = trajectory_heads(trajectory_agent)["policy"][:, -1, 0]
            action = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, episode_seed + index + 1
            )
            incoming.fill_(action)
            if terminated or truncated:
                if index < last:
                    missing = sorted(step for step in wanted if step > index)
                    raise RuntimeError(
                        f"baseline trajectory ended before saved steps: "
                        f"seed={episode_seed} missing={missing}"
                    )
                break

    missing = sorted(set(key_to_row) - reproduced)
    if missing:
        raise RuntimeError(f"failed to replay saved fork states: {missing}")

    branching = [row for row in rows if row["mode_count"] > 1]
    reference_fatal = [row for row in rows if row["reference_draw_terminal"]]
    reference_safe = [row for row in rows if not row["reference_draw_terminal"]]
    death_stochastic = [row for row in rows if row["death_stochastic"]]
    death_observable = [
        row for row in death_stochastic if row["death_varies_between_observed_modes"]
    ]
    return rows, {
        "all_pairs": _summary(rows),
        "branching_pairs": _summary(branching),
        "reference_fatal_pairs": _summary(reference_fatal),
        "reference_safe_pairs": _summary(reference_safe),
        "death_stochastic_pairs": _summary(death_stochastic),
        "death_stochastic_observable_pairs": _summary(death_observable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--direct-world", type=Path, required=True)
    parser.add_argument("--flow-world", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--flow-samples", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 2 or args.flow_samples < 1:
        parser.error("draws must be >= 2 and flow-samples must be >= 1")

    base = Config()
    trajectory_config = replace(base, transition="direct", time_mixer="attention")
    direct_config = replace(base, transition="direct", time_mixer="attention")
    flow_config = replace(base, transition="flow", time_mixer="attention")
    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, trajectory_config
    )
    direct = _load_world(args.direct_world, direct_config)
    flow = _load_world(args.flow_world, flow_config)
    saved = torch.load(args.forks, weights_only=False)

    rows, summaries = diagnose(
        saved,
        encoder,
        trajectory_world,
        trajectory_heads,
        direct,
        flow,
        trajectory_config,
        direct_config,
        flow_config,
        args.draws,
        args.flow_samples,
    )
    report = {
        "contract": {
            "fixed_pre_action_state_and_action": True,
            "only_environment_rng_varies_across_true_successors": True,
            "first_draw_replays_saved_gate_truth": True,
            "mode_definition": "exact rendered successor observation under fixed encoder history",
            "selection": "saved states with action-dependent death under the gate draw",
            "draws_per_state_action": args.draws,
            "flow_samples_per_state_action": args.flow_samples,
            "phase1a": str(args.phase1a.resolve()),
            "trajectory_phase2": str(args.trajectory_phase2.resolve()),
            "direct_world": str(args.direct_world.resolve()),
            "flow_world": str(args.flow_world.resolve()),
            "forks": str(args.forks.resolve()),
            "phase1a_sha256": _digest(args.phase1a),
            "trajectory_phase2_sha256": _digest(args.trajectory_phase2),
            "direct_world_sha256": _digest(args.direct_world),
            "flow_world_sha256": _digest(args.flow_world),
            "forks_sha256": _digest(args.forks),
            "no_automatic_mop_decision": True,
        },
        "summary": summaries,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"contract": report["contract"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
