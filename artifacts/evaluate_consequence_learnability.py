"""Score the consequence benchmark on held-out all-action fork states.

Replays each fork trajectory frame by frame with the frozen BC policy, exactly as
`evaluate_recursive_generated_latent_outcome` does, so the benchmark is given the
same causal history the Direct world model receives. At each opportunity state the
simulator executes all 17 actions; the true target is the gate's own quantity
`d^T(z_{t+1} - z_t)` computed from real encoded successors.

Two readouts are recorded per action. `teacher` predicts from the real observed
fork state, the condition matching what the benchmark was trained on, and is the
primary. `generated` predicts from a generated fork state, which is what the
recursive gate hands the world model; it is a transfer diagnostic.

The held-out DEV control scores the same trained model on the same matched
terminal pairs the archive evaluation uses, so a weak fork result can be told
apart from a representation that never carried the consequence at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.benchmark_consequence_learnability import (
    ConsequenceHead,
    predict_scalar,
    scalar_contrast,
)
from artifacts.evaluate_predictor_flow_archive import _led_to
from artifacts.localize_counterfactual import load_models
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset
from d4mj.env import step as env_step
from d4mj.transition import World, advance, commit_inputs, observe

VERSION = "consequence-learnability-evaluation-v1"


@torch.no_grad()
def evaluate_dev_transitions(
    records: list[dict],
    world: World,
    head: ConsequenceHead,
    direction: torch.Tensor,
    config: Config,
    *,
    context: int,
) -> dict[str, torch.Tensor]:
    """The same trained model on held-out DEV matched terminal pairs.

    Teacher forced, exactly as the benchmark trained: features of the real block
    `t`, and the action taken at `t`. No refitting and no replay.
    """
    flat = direction.to(config.device).flatten()
    predicted, true, labels, groups = [], [], [], []
    for index, record in enumerate(records):
        transition = int(record["transitions"][0])
        latents, taken = record["latents"], record["actions_taken"]
        if transition < 1 or transition + 1 >= len(latents):
            continue
        start = max(0, transition + 1 - context)
        block = latents[start : transition + 1][None].to(config.device)
        incoming = _led_to(taken, start, transition + 1, config)[None].to(config.device)
        rng = torch.Generator(device=config.device).manual_seed(
            config.seed + 23_000 + index
        )
        committed, conditioning = commit_inputs(block, rng, config)
        features, _, _ = world(None, incoming, committed, conditioning)
        action = taken[transition].view(1, 1).to(config.device)
        predicted.append(float(predict_scalar(world, head, features[:, -1:], action)[0, -1]))
        current = latents[transition].flatten().to(config.device)
        successor = latents[transition + 1].flatten().to(config.device)
        true.append(float((successor - current) @ flat.to(current.dtype)))
        labels.append(bool(record["labels"][0]))
        groups.append(int(record.get("group", index)))
    return {
        "predicted": torch.tensor(predicted),
        "true": torch.tensor(true),
        "fatal": torch.tensor(labels),
        "group": torch.tensor(groups),
    }


def regression_metrics(predicted: torch.Tensor, true: torch.Tensor) -> dict:
    """Secondary: does the scalar track its target at all, beyond the contrast."""
    error = predicted - true
    centred_p = predicted - predicted.mean()
    centred_t = true - true.mean()
    denominator = (centred_p.norm() * centred_t.norm()).clamp(min=1e-12)
    return {
        "examples": len(true),
        "mse": float(error.pow(2).mean()),
        "correlation": float((centred_p * centred_t).sum() / denominator),
        "true_std": float(true.std()),
        "predicted_std": float(predicted.std()),
    }


@torch.no_grad()
def evaluate_forks(
    saved: dict,
    encoder,
    trajectory_world,
    trajectory_heads,
    world: World,
    head: ConsequenceHead,
    direction: torch.Tensor,
    config: Config,
) -> dict[str, torch.Tensor]:
    varies = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    selected = varies.nonzero().flatten().tolist()
    key_to_row = {
        (int(saved["seed"][row]), int(saved["step"][row])): row for row in selected
    }
    if len(key_to_row) != len(selected):
        raise ValueError("fork set contains duplicate opportunity states")
    row_to_group = {row: group for group, row in enumerate(selected)}
    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    device = config.device
    flat_direction = direction.to(device).flatten()
    true, teacher, generated, labels, groups = [], [], [], [], []
    reproduced = set()

    for seed in sorted(by_seed):
        wanted = by_seed[seed]
        last = max(wanted)
        observation, env_state = reset(seed)
        trajectory_state = benchmark_state = None
        incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=device)
        trajectory_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        benchmark_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)

        for index in range(last + 1):
            previous_state = benchmark_state
            patches = patchify(observation[None, None], config.patch).to(device)
            trajectory_state, trajectory_agent = observe(
                trajectory_world, encoder, trajectory_state, incoming,
                patches, trajectory_rng, config,
            )
            benchmark_state, _ = observe(
                world, encoder, benchmark_state, incoming, patches, benchmark_rng, config,
            )

            key = (seed, index)
            if key in key_to_row:
                if previous_state is None:
                    raise ValueError("fork has no preceding world state")
                row = key_to_row[key]
                group = row_to_group[row]
                current = benchmark_state.world.latent[0, -1].flatten()
                start = float(current @ flat_direction.to(current.dtype))
                generated_rng = torch.Generator(device=device).manual_seed(
                    config.seed + 2**25 + seed * 4099 + index * 17
                )
                generated_current, _ = advance(
                    world, previous_state.world, incoming, generated_rng, config
                )
                for action in range(config.n_actions):
                    successor_observation, _, _, terminated, _ = env_step(
                        env_state, action, seed + index + 1
                    )
                    if bool(terminated) != bool(saved["true_death"][row, action]):
                        raise AssertionError(
                            f"truth mismatch seed={seed} step={index} action={action}"
                        )
                    chosen = torch.tensor([[action]], device=device)
                    observed_rng = torch.Generator(device=device).manual_seed(
                        config.seed + 2**27 + seed * 4099 + index * 17 + action
                    )
                    successor_patches = patchify(
                        successor_observation[None, None], config.patch
                    ).to(device)
                    observed_successor, _ = observe(
                        world, encoder, benchmark_state, chosen,
                        successor_patches, observed_rng, config,
                    )
                    successor = observed_successor.world.latent[0, -1].flatten()
                    true.append(
                        float(successor @ flat_direction.to(successor.dtype)) - start
                    )
                    teacher.append(
                        float(predict_scalar(
                            world, head, benchmark_state.world.features, chosen
                        )[0, -1])
                    )
                    generated.append(
                        float(predict_scalar(
                            world, head, generated_current.features, chosen
                        )[0, -1])
                    )
                    labels.append(bool(terminated))
                    groups.append(group)
                reproduced.add(key)

            logits = trajectory_heads(trajectory_agent)["policy"][:, -1, 0]
            action = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            if key in key_to_row and action != int(saved["trajectory_action"][key_to_row[key]]):
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
        raise RuntimeError(f"failed to replay forks: {missing}")
    return {
        "true": torch.tensor(true),
        "teacher": torch.tensor(teacher),
        "generated": torch.tensor(generated),
        "fatal": torch.tensor(labels),
        "group": torch.tensor(groups),
        "states": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--forks", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--context", type=int, default=Config().dynamics_context)
    args = parser.parse_args()
    if len(args.forks) != len(args.names):
        parser.error("each fork set needs a name")

    config = Config(transition="direct", time_mixer="attention")
    prepared = torch.load(args.prepared, weights_only=False)
    direction = prepared["direction"].float()
    if abs(float(direction.norm()) - 1.0) > 1e-4:
        raise ValueError("fatality direction is not unit norm")

    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, Config(), config
    )
    world = World(config).to(config.device).eval()
    head = ConsequenceHead(config).to(config.device).eval()
    load(args.model, config, part0=world, part1=head)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": {
            "version": VERSION,
            "phase1a": file_digest(args.phase1a),
            "trajectory_phase2": file_digest(args.trajectory_phase2),
            "prepared": file_digest(args.prepared),
            "model": file_digest(args.model),
            "implementation": implementation_digests(Path(__file__)),
            "target": "d^T(z_{t+1} - z_t) from real encoded successors",
            "primary": "teacher -- benchmark predicts from the real observed fork state",
            "secondary": "generated -- transfer to the recursive gate's own condition",
        },
        "sets": {},
    }
    prepared_records = prepared["records"]
    print(f"DEV control: {len(prepared_records)} held-out matched records", flush=True)
    dev = evaluate_dev_transitions(
        prepared_records, world, head, direction, config, context=args.context
    )
    report["dev_transitions"] = {
        "contrast": scalar_contrast(
            dev["predicted"], dev["true"], dev["fatal"], dev["group"],
            samples=args.bootstraps, seed=config.seed + 91,
        ),
        "regression": regression_metrics(dev["predicted"], dev["true"]),
    }
    torch.save(dev, args.out / "scores_dev_transitions.pt")
    print(f"  DEV: {json.dumps(report['dev_transitions'])}", flush=True)

    for name, path in zip(args.names, args.forks):
        print(f"fork set: {name}", flush=True)
        saved = torch.load(path, weights_only=False, map_location="cpu")
        data = evaluate_forks(
            saved, encoder, trajectory_world, trajectory_heads,
            world, head, direction, config,
        )
        entry = {"states": data["states"], "examples": len(data["true"]), "forks": file_digest(path)}
        for readout in ("teacher", "generated"):
            entry[readout] = scalar_contrast(
                data[readout], data["true"], data["fatal"], data["group"],
                samples=args.bootstraps, seed=config.seed + 91,
            )
            entry[f"{readout}_regression"] = regression_metrics(data[readout], data["true"])
        report["sets"][name] = entry
        torch.save(data, args.out / f"scores_{name}.pt")
        print(f"  {name}: {json.dumps(entry['teacher'])}", flush=True)

    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
