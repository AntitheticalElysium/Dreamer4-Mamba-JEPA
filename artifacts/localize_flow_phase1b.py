"""Localize Flow fatality information across encoding, conditioning, and sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from artifacts.localize_counterfactual import binary_metrics, load_models
from artifacts.localize_counterfactual_interaction import report_score
from artifacts.localize_direct_transition_stages import _resumable_linear_probe
from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.representation import pack
from d4mj.transition import World, advance, observe


@dataclass
class FlowForkData:
    observed_latent: Tensor
    generated_latent_first: Tensor
    generated_latent_mean: Tensor
    generated_latent_variance: Tensor
    generated_readout_first: Tensor
    generated_readout_mean: Tensor
    generated_readout_variance: Tensor
    conditioned_readout_first: dict[int, Tensor]
    conditioned_readout_mean: dict[int, Tensor]
    conditioned_readout_variance: dict[int, Tensor]
    generated_death_first: Tensor
    generated_death_mean: Tensor
    conditioned_death_first: dict[int, Tensor]
    conditioned_death_mean: dict[int, Tensor]
    target: Tensor
    action: Tensor
    group: Tensor


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _summary(samples: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    stacked = torch.stack(samples)
    mean = stacked.mean(0)
    variance = (stacked - mean).pow(2).mean()
    return stacked[0].cpu(), mean.cpu(), variance.cpu()


def _conditioned_agent(
    world: World,
    state,
    action: Tensor,
    latent: Tensor,
    signal_index: int,
    rng: torch.Generator,
    config: Config,
) -> Tensor:
    signal = signal_index / config.k_max
    noise = torch.randn(
        latent.shape,
        generator=rng,
        device=latent.device,
        dtype=latent.dtype,
    )
    committed = signal * latent + (1.0 - signal) * noise
    conditioning = torch.tensor(
        [[[signal_index, config.step_index]]],
        dtype=torch.long,
        device=latent.device,
    )
    _, agent, _ = world(
        state.memory,
        action,
        committed,
        conditioning,
        state.step,
    )
    return agent


def _death(heads: Heads, agent: Tensor) -> Tensor:
    return 1.0 - heads(agent)["continuation"][:, -1, 0].sigmoid()[0]


@torch.no_grad()
def _extract_seed(
    seed: int,
    wanted: set[int],
    key_to_row: dict,
    row_to_group: dict,
    saved,
    encoder,
    trajectory_world,
    trajectory_heads,
    world,
    heads,
    trajectory_config: Config,
    config: Config,
    *,
    samples: int,
    signal_levels: tuple[int, ...],
) -> tuple[dict, dict[int, int]]:
    device = config.device
    last = max(wanted)
    observation, env_state = reset(seed)
    trajectory_state = evaluation_state = None
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=device)
    trajectory_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
    evaluation_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)

    fields = {
        "observed_latent": [],
        "generated_latent_first": [],
        "generated_latent_mean": [],
        "generated_latent_variance": [],
        "generated_readout_first": [],
        "generated_readout_mean": [],
        "generated_readout_variance": [],
        "generated_death_first": [],
        "generated_death_mean": [],
        "target": [],
        "action": [],
        "group": [],
    }
    conditioned = {
        key: {level: [] for level in signal_levels}
        for key in (
            "conditioned_readout_first",
            "conditioned_readout_mean",
            "conditioned_readout_variance",
            "conditioned_death_first",
            "conditioned_death_mean",
        )
    }
    chosen_by_group: dict[int, int] = {}

    for index in range(last + 1):
        patches = patchify(observation[None, None], config.patch).to(device)
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
            config,
        )

        key = (seed, index)
        if key in key_to_row:
            row = key_to_row[key]
            group = row_to_group[row]
            for action_index in range(config.n_actions):
                successor, _, _, terminated, _ = env_step(
                    env_state, action_index, seed + index + 1
                )
                if bool(terminated) != bool(saved["true_death"][row, action_index]):
                    raise AssertionError(
                        f"truth mismatch seed={seed} step={index} action={action_index}"
                    )
                action = torch.tensor([[action_index]], dtype=torch.long, device=device)
                successor_patches = patchify(successor[None, None], config.patch).to(device)
                encoded, _, _ = encoder(
                    successor_patches,
                    evaluation_state.encoder_memory,
                    offset=evaluation_state.world.step,
                )
                clean_latent = pack(encoded, config)

                generated_latents, generated_readouts, generated_deaths = [], [], []
                level_readouts = {level: [] for level in signal_levels}
                level_deaths = {level: [] for level in signal_levels}
                for sample in range(samples):
                    generated_rng = torch.Generator(device=device).manual_seed(
                        config.seed + 2**23 + seed * 4099 + index * 17 + sample
                    )
                    generated_state, generated_agent = advance(
                        world,
                        evaluation_state.world,
                        action,
                        generated_rng,
                        config,
                    )
                    generated_latents.append(generated_state.latent[0, -1])
                    generated_readouts.append(generated_agent[0, -1])
                    generated_deaths.append(_death(heads, generated_agent))

                    noise_seed = config.seed + 2**24 + seed * 4099 + index * 17 + sample
                    for level in signal_levels:
                        conditioned_rng = torch.Generator(device=device).manual_seed(noise_seed)
                        conditioned_agent = _conditioned_agent(
                            world,
                            evaluation_state.world,
                            action,
                            clean_latent,
                            level,
                            conditioned_rng,
                            config,
                        )
                        if (
                            sample == 0
                            and action_index == 0
                            and level == config.tau_ctx_index
                        ):
                            observed_rng = torch.Generator(device=device).manual_seed(noise_seed)
                            observed_state, observed_agent = observe(
                                world,
                                encoder,
                                evaluation_state,
                                action,
                                successor_patches,
                                observed_rng,
                                config,
                            )
                            latent_error = float(
                                (observed_state.world.latent - clean_latent).abs().max()
                            )
                            readout_error = float(
                                (observed_agent - conditioned_agent).abs().max()
                            )
                            if latent_error > 1e-6 or readout_error > 1e-6:
                                raise AssertionError(
                                    "manual Flow conditioning differs from observe: "
                                    f"latent={latent_error:.3e} readout={readout_error:.3e}"
                                )
                        level_readouts[level].append(conditioned_agent[0, -1])
                        level_deaths[level].append(_death(heads, conditioned_agent))

                first, mean, variance = _summary(generated_latents)
                fields["generated_latent_first"].append(first)
                fields["generated_latent_mean"].append(mean)
                fields["generated_latent_variance"].append(variance)
                first, mean, variance = _summary(generated_readouts)
                fields["generated_readout_first"].append(first)
                fields["generated_readout_mean"].append(mean)
                fields["generated_readout_variance"].append(variance)
                death = torch.stack(generated_deaths)
                fields["generated_death_first"].append(death[0].cpu())
                fields["generated_death_mean"].append(death.mean().cpu())

                for level in signal_levels:
                    first, mean, variance = _summary(level_readouts[level])
                    conditioned["conditioned_readout_first"][level].append(first)
                    conditioned["conditioned_readout_mean"][level].append(mean)
                    conditioned["conditioned_readout_variance"][level].append(variance)
                    death = torch.stack(level_deaths[level])
                    conditioned["conditioned_death_first"][level].append(death[0].cpu())
                    conditioned["conditioned_death_mean"][level].append(death.mean().cpu())

                fields["observed_latent"].append(clean_latent[0, -1].cpu())
                fields["target"].append(float(terminated))
                fields["action"].append(action_index)
                fields["group"].append(group)

        logits = trajectory_heads(trajectory_agent)["policy"][:, -1, 0]
        trajectory_action = int(
            torch.multinomial(logits.softmax(-1), 1, generator=policy_rng)
        )
        if key in key_to_row:
            chosen_by_group[row_to_group[key_to_row[key]]] = trajectory_action
        observation, env_state, _, terminated, truncated = env_step(
            env_state, trajectory_action, seed + index + 1
        )
        incoming.fill_(trajectory_action)
        if terminated or truncated:
            if index < last:
                missing = sorted(step for step in wanted if step > index)
                raise RuntimeError(
                    f"baseline trajectory ended before saved steps: {seed=} {missing=}"
                )
            break

    packed = {
        key: torch.stack(value) if key not in ("action", "group", "target") else torch.tensor(value)
        for key, value in fields.items()
    }
    packed.update(
        {
            key: {level: torch.stack(values) for level, values in levels.items()}
            for key, levels in conditioned.items()
        }
    )
    return packed, chosen_by_group


@torch.no_grad()
def extract_flow_forks(
    saved,
    encoder,
    trajectory_world,
    trajectory_heads,
    world,
    heads,
    trajectory_config: Config,
    config: Config,
    *,
    samples: int,
    signal_levels: tuple[int, ...],
    cache: Path,
    contract: dict,
) -> tuple[FlowForkData, dict]:
    if config.transition != "flow":
        raise ValueError("Flow localization requires a Flow evaluation world")
    opportunity = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    selected = opportunity.nonzero().flatten().tolist()
    if not selected:
        raise RuntimeError("saved forks contain no terminal-opportunity states")
    key_to_row = {
        (int(saved["seed"][row]), int(saved["step"][row])): row for row in selected
    }
    if len(key_to_row) != len(selected):
        raise RuntimeError("saved forks contain duplicate terminal-opportunity states")
    row_to_group = {row: group for group, row in enumerate(selected)}
    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    cache.mkdir(parents=True, exist_ok=True)
    pieces, actions = [], {}
    for seed in sorted(by_seed):
        path = cache / f"seed_{seed}.pt"
        seed_contract = contract | {"seed": seed, "steps": sorted(by_seed[seed])}
        if path.exists():
            payload = torch.load(path, weights_only=False)
            if payload["contract"] != seed_contract:
                raise ValueError(f"Flow feature cache contract changed: {path}")
            piece, chosen = payload["data"], payload["trajectory_actions"]
        else:
            print(f"extracting Flow forks for seed {seed}", flush=True)
            piece, chosen = _extract_seed(
                seed,
                by_seed[seed],
                key_to_row,
                row_to_group,
                saved,
                encoder,
                trajectory_world,
                trajectory_heads,
                world,
                heads,
                trajectory_config,
                config,
                samples=samples,
                signal_levels=signal_levels,
            )
            _atomic_torch(
                path,
                {"contract": seed_contract, "data": piece, "trajectory_actions": chosen},
            )
        pieces.append(piece)
        actions.update(chosen)

    simple = (
        "observed_latent",
        "generated_latent_first",
        "generated_latent_mean",
        "generated_latent_variance",
        "generated_readout_first",
        "generated_readout_mean",
        "generated_readout_variance",
        "generated_death_first",
        "generated_death_mean",
        "target",
        "action",
        "group",
    )
    nested = (
        "conditioned_readout_first",
        "conditioned_readout_mean",
        "conditioned_readout_variance",
        "conditioned_death_first",
        "conditioned_death_mean",
    )
    combined = {key: torch.cat([piece[key] for piece in pieces]) for key in simple}
    combined.update(
        {
            key: {
                level: torch.cat([piece[key][level] for piece in pieces])
                for level in signal_levels
            }
            for key in nested
        }
    )
    if set(actions) != set(range(len(selected))):
        raise RuntimeError("failed to reproduce every fixed trajectory action")
    if len(combined["target"]) != len(selected) * config.n_actions:
        raise RuntimeError("Flow extraction did not cover all state-action forks")
    return FlowForkData(**combined), {
        "terminal_opportunity_states": len(selected),
        "examples": len(combined["target"]),
        "truth_replayed_exactly": True,
        "samples_per_state_action": samples,
        "signal_levels": list(signal_levels),
        "production_observe_path_reproduced": "sample 0, action 0, every state",
        "trajectory_action_by_group": [actions[group] for group in range(len(selected))],
    }


def _latent_error(predicted: Tensor, observed: Tensor, target: Tensor) -> dict[str, float]:
    error = (predicted - observed).pow(2).flatten(1).mean(1)
    dead = target.bool()
    return {
        "all": float(error.mean()),
        "fatal": float(error[dead].mean()),
        "safe": float(error[~dead].mean()),
        "fatal_over_safe": float(error[dead].mean() / error[~dead].mean().clamp(min=1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--flow-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--signal-levels", type=int, nargs="+", default=(0, 4, 7))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    signal_levels = tuple(sorted(set(args.signal_levels)))
    base = Config()
    trajectory_config = Config(transition="direct", time_mixer="attention")
    config = Config(transition="flow", time_mixer="attention")
    if any(level < 0 or level >= config.k_max for level in signal_levels):
        parser.error(f"signal levels must be in [0, {config.k_max - 1}]")
    if config.tau_ctx_index not in signal_levels:
        parser.error("signal levels must include the production tau_ctx_index")
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    inputs = {
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "flow_phase2": file_digest(args.flow_phase2),
        "forks": file_digest(args.forks),
    }
    contract = {
        "version": "flow-phase1b-localization-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/localize_direct_transition_stages.py"),
            Path("artifacts/localize_counterfactual.py"),
            Path("artifacts/localize_counterfactual_interaction.py"),
        ),
        "evaluation": "same fixed terminal-opportunity DEV states and all 17 actions",
        "samples": args.samples,
        "common_random_numbers_across_actions": True,
        "signal_levels": list(signal_levels),
        "production_signal_level": config.tau_ctx_index,
        "probe": "action-centered leave-one-pre-action-state-out linear",
        "seeds": args.seeds,
        "linear_steps": args.linear_steps,
        "permutations": args.permutations,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Flow localization contract changed")
    else:
        atomic_json(contract_path, contract)

    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, trajectory_config
    )
    world = World(config).to(config.device)
    heads = Heads(config).to(config.device)
    load(args.flow_phase2, config, part0=world, part1=heads)
    for module in (world, heads):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    saved = torch.load(args.forks, weights_only=False)
    data, replay = extract_flow_forks(
        saved,
        encoder,
        trajectory_world,
        trajectory_heads,
        world,
        heads,
        trajectory_config,
        config,
        samples=args.samples,
        signal_levels=signal_levels,
        cache=args.out / "feature_cache",
        contract=contract,
    )
    _atomic_torch(args.out / "features.pt", {"contract": contract, "data": vars(data), "replay": replay})
    for module in (encoder, trajectory_world, trajectory_heads, world, heads):
        module.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    features = {
        "observed_clean_latent": data.observed_latent,
        "generated_latent_first": data.generated_latent_first,
        "generated_latent_mean": data.generated_latent_mean,
        "generated_readout_first": data.generated_readout_first,
        "generated_readout_mean": data.generated_readout_mean,
        "observed_readout_first": data.conditioned_readout_first[config.tau_ctx_index],
        "observed_readout_mean": data.conditioned_readout_mean[config.tau_ctx_index],
    }
    for level in signal_levels:
        if level != config.tau_ctx_index:
            features[f"conditioned_readout_mean_tau{level}"] = data.conditioned_readout_mean[level]

    seeds = [config.seed + 4000 + index for index in range(args.seeds)]
    probes = {}
    for index, (name, feature) in enumerate(features.items()):
        probe_contract = {
            "version": contract["version"],
            "stage": "equalized_flow",
            "feature": name,
            "inputs": inputs,
            "samples": args.samples,
            "signal_levels": signal_levels,
            "seeds": seeds,
            "steps": args.linear_steps,
            "lr": 3e-3,
            "weight_decay": 1e-3,
        }
        probability = _resumable_linear_probe(
            feature,
            data.target,
            data.action,
            data.group,
            config,
            seeds=seeds,
            steps=args.linear_steps,
            checkpoint=args.out / "probes" / f"{name}.pt",
            contract=probe_contract,
        )
        probes[name] = {
            "binary": binary_metrics(probability, data.target),
            "same_action": report_score(
                probability,
                data.target,
                data.action,
                permutations=args.permutations,
                seed=config.seed + 7000 + index,
            ),
        }

    def production_score(probability: Tensor, seed: int) -> dict:
        return {
            "binary": binary_metrics(probability, data.target),
            "same_action": report_score(
                probability,
                data.target,
                data.action,
                permutations=args.permutations,
                seed=seed,
            ),
        }

    production = {
        "generated_first": production_score(
            data.generated_death_first, config.seed + 8000
        ),
        "generated_mean": production_score(
            data.generated_death_mean, config.seed + 8001
        ),
        "conditioned_first": {},
        "conditioned_mean": {},
    }
    for level in signal_levels:
        production["conditioned_first"][str(level)] = production_score(
            data.conditioned_death_first[level], config.seed + 8100 + level
        )
        production["conditioned_mean"][str(level)] = production_score(
            data.conditioned_death_mean[level], config.seed + 8200 + level
        )

    report = {
        "contract": contract,
        "replay": replay,
        "latent_prediction_error": {
            "first_sample": _latent_error(
                data.generated_latent_first, data.observed_latent, data.target
            ),
            "sample_mean": _latent_error(
                data.generated_latent_mean, data.observed_latent, data.target
            ),
        },
        "sample_variance": {
            "generated_latent": float(data.generated_latent_variance.mean()),
            "generated_readout": float(data.generated_readout_variance.mean()),
            "conditioned_readout": {
                str(level): float(data.conditioned_readout_variance[level].mean())
                for level in signal_levels
            },
        },
        "production_head": production,
        "fresh_probes": probes,
    }
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
