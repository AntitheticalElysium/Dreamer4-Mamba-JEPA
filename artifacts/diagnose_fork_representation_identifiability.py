"""Locate action-conditioned death information across the frozen visual stack."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from artifacts.localize_counterfactual import auc, load_models
from artifacts.localize_counterfactual_interaction import conditional_auc
from artifacts.phase1b_geometry_common import atomic_torch
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

VERSION = "fork-representation-identifiability-v1"
REPRESENTATIONS = ("simulator_state", "raw_observation", "pre_bottleneck", "frozen_z")


def simulator_vector(state) -> torch.Tensor:
    """Agent-centred categorical state, avoiding an absolute-coordinate probe."""
    from craftax.craftax_classic.constants import BlockType

    position = np.asarray(state.player_position).astype(np.int64)
    radius = 4
    block = np.asarray(state.map).astype(np.int64)
    mob = np.asarray(state.mob_map).astype(np.float32)
    block = np.pad(block, radius, constant_values=BlockType.OUT_OF_BOUNDS.value)
    mob = np.pad(mob, radius, constant_values=0)
    x, y = position
    local_block = torch.from_numpy(block[x : x + 2 * radius + 1, y : y + 2 * radius + 1])
    local_mob = torch.from_numpy(mob[x : x + 2 * radius + 1, y : y + 2 * radius + 1])
    values = [
        F.one_hot(local_block, len(BlockType)).float().flatten(),
        local_mob.float().flatten(),
        F.one_hot(torch.tensor(int(state.player_direction) - 1), 4).float(),
    ]
    scalar_names = (
        "player_health", "player_food", "player_drink", "player_energy",
        "is_sleeping", "player_recover", "player_hunger", "player_thirst",
        "player_fatigue", "light_level", "timestep",
    )
    values.append(torch.tensor([float(getattr(state, name)) for name in scalar_names]))
    values.append(torch.tensor([float(value) for value in jax.tree_util.tree_leaves(state.inventory)]))
    for name in ("zombies", "cows", "skeletons", "arrows"):
        mobs = getattr(state, name)
        relative = np.asarray(mobs.position).astype(np.float32) - position[None]
        values.extend([
            torch.from_numpy(relative / (2 * radius + 1)).flatten(),
            torch.from_numpy(np.asarray(mobs.health).astype(np.float32)).flatten(),
            torch.from_numpy(np.asarray(mobs.mask).astype(np.float32)).flatten(),
            torch.from_numpy(np.asarray(mobs.attack_cooldown).astype(np.float32)).flatten(),
        ])
    values.append(torch.from_numpy(np.asarray(state.arrow_directions).astype(np.float32)).flatten())
    plant_position = np.asarray(state.growing_plants_positions).astype(np.float32) - position[None]
    values.extend([
        torch.from_numpy(plant_position / (2 * radius + 1)).flatten(),
        torch.from_numpy(np.asarray(state.growing_plants_age).astype(np.float32)).flatten(),
        torch.from_numpy(np.asarray(state.growing_plants_mask).astype(np.float32)).flatten(),
        torch.from_numpy(np.asarray(state.achievements).astype(np.float32)).flatten(),
    ])
    return torch.cat(values).float()


@torch.no_grad()
def extract_features(saved, encoder, world, heads, config: Config, fixed_z: torch.Tensor):
    keys = {
        (int(seed), int(step)): row
        for row, (seed, step) in enumerate(zip(saved["seed"], saved["step"]))
    }
    if len(keys) != len(saved["seed"]):
        raise ValueError("fork file contains duplicate states")
    by_seed: dict[int, set[int]] = {}
    for seed, step in keys:
        by_seed.setdefault(seed, set()).add(step)

    rows: dict[str, list[torch.Tensor | None]] = {
        name: [None] * len(keys) for name in REPRESENTATIONS
    }
    reproduced = set()
    captured: list[torch.Tensor] = []

    def capture(_module, arguments):
        captured.append(arguments[0].detach().cpu())

    handle = encoder.bottleneck.register_forward_pre_hook(capture)
    try:
        for seed in sorted(by_seed):
            wanted = by_seed[seed]
            last = max(wanted)
            observation, env_state = reset(seed)
            state = None
            incoming = torch.full(
                (1, 1), config.n_actions, dtype=torch.long, device=config.device
            )
            world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
            policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
            for index in range(last + 1):
                captured.clear()
                patches = patchify(observation[None, None], config.patch).to(config.device)
                state, agent = observe(
                    world, encoder, state, incoming, patches, world_rng, config
                )
                if len(captured) != 1:
                    raise AssertionError("encoder bottleneck hook did not fire exactly once")
                key = (seed, index)
                if key in keys:
                    row = keys[key]
                    truth = []
                    for action in range(config.n_actions):
                        _, _, _, terminated, _ = env_step(
                            env_state, action, seed + index + 1
                        )
                        truth.append(bool(terminated))
                    if not torch.equal(torch.tensor(truth), saved["true_death"][row].bool()):
                        raise AssertionError(f"fork truth changed at {key}")
                    rows["simulator_state"][row] = simulator_vector(env_state)
                    rows["raw_observation"][row] = observation.flatten().float() / 255.0
                    rows["pre_bottleneck"][row] = captured[0][0, -1].flatten()
                    rows["frozen_z"][row] = state.world.latent[0, -1].detach().cpu().flatten()
                    reproduced.add(key)

                logits = heads(agent)["policy"][:, -1, 0]
                action = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
                observation, env_state, _, terminated, truncated = env_step(
                    env_state, action, seed + index + 1
                )
                incoming.fill_(action)
                if terminated or truncated:
                    if index < last:
                        raise RuntimeError(f"trajectory ended before saved fork for seed {seed}")
                    break
    finally:
        handle.remove()

    if reproduced != set(keys):
        raise RuntimeError(f"failed to replay fork states: {sorted(set(keys) - reproduced)}")
    stacked = {
        name: torch.stack([value for value in values if value is not None])
        for name, values in rows.items()
    }
    z_error = float((stacked["frozen_z"] - fixed_z.flatten(1)).abs().max())
    if z_error > 1e-5:
        raise AssertionError(f"frozen-z replay differs from fixed fork features: {z_error:.3e}")
    return stacked, {"states": len(keys), "truth_replayed_exactly": True, "frozen_z_max_abs_error": z_error}


class AllActionProbe(nn.Module):
    def __init__(self, width: int, actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def within_state_auc(score: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for row in range(len(score)):
        values.append(auc(score[row], target[row]))
    return float(torch.tensor(values).mean())


def score_report(score: torch.Tensor, target: torch.Tensor) -> dict:
    actions = torch.arange(target.shape[1]).expand_as(target)
    groups = torch.arange(len(target))[:, None].expand_as(target)
    centered = score - score.mean(1, keepdim=True)
    return {
        "global_auc": auc(score, target),
        "within_state_auc": within_state_auc(score, target),
        "within_state_centered_auc": auc(centered, target),
        "same_action_auc": conditional_auc(
            score.flatten(), target.flatten(), actions.flatten()
        )["pooled_pair_auc"],
        "examples": target.numel(),
        "states": len(target),
        "groups": int(groups.max()) + 1,
    }


def fold_assignment(pair: torch.Tensor, folds: int, seed: int) -> torch.Tensor:
    unique = pair.unique(sorted=True)
    order = unique[torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))]
    assignment = torch.empty(int(unique.max()) + 1, dtype=torch.long)
    for position, value in enumerate(order.tolist()):
        assignment[value] = position % folds
    return assignment[pair]


def train_fold(
    feature,
    target,
    fit,
    validation,
    test,
    config,
    *,
    seed,
    steps,
):
    mean = feature[fit].mean(0, keepdim=True)
    scale = feature[fit].std(0, unbiased=False, keepdim=True).clamp(min=1e-4)
    feature = (feature - mean) / scale
    torch.manual_seed(seed)
    model = AllActionProbe(feature.shape[1], target.shape[1]).to(config.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x, y = feature[fit].to(config.device), target[fit].float().to(config.device)
    positive = y.sum(0)
    negative = len(y) - positive
    pos_weight = torch.where(positive > 0, negative / positive.clamp(min=1), torch.ones_like(positive))
    best, best_key = None, (-1.0, float("-inf"))
    for step in range(steps):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if (step + 1) % 20 == 0 or step + 1 == steps:
            with torch.no_grad():
                probability = model(feature[validation].to(config.device)).sigmoid().cpu()
            value = within_state_auc(probability, target[validation])
            bce = float(F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), target[validation].float()))
            key = (value, -bce)
            if key > best_key:
                best_key, best = key, copy.deepcopy(model.state_dict())
    if best is None:
        raise AssertionError("probe never produced a checkpoint")
    model.load_state_dict(best)
    with torch.no_grad():
        return model(feature[test].to(config.device)).sigmoid().cpu()


def cross_validated_probe(feature, target, pair, config, *, folds, seeds, steps):
    assignment = fold_assignment(pair, folds, config.seed + 11_000)
    predictions = []
    for seed_index in range(seeds):
        prediction = torch.empty_like(target, dtype=torch.float)
        for test_fold in range(folds):
            validation_fold = (test_fold + 1 + seed_index) % folds
            test = assignment == test_fold
            validation = assignment == validation_fold
            fit = ~(test | validation)
            prediction[test] = train_fold(
                feature,
                target,
                fit,
                validation,
                test,
                config,
                seed=config.seed + 11_100 + seed_index * 100 + test_fold,
                steps=steps,
            )
        predictions.append(prediction)
    ensemble = torch.stack(predictions).mean(0)
    return score_report(ensemble, target) | {
        "seed_reports": [score_report(value, target) for value in predictions]
    }


def cross_validated_action_prior(target, pair, *, folds: int, seed: int):
    assignment = fold_assignment(pair, folds, seed)
    prediction = torch.empty_like(target, dtype=torch.float)
    for test_fold in range(folds):
        test = assignment == test_fold
        prediction[test] = target[~test].float().mean(0)
    return score_report(prediction, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--fixed-z", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    feature_path = args.out / "features.pt"
    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "forks": file_digest(args.forks),
        "fixed_z": file_digest(args.fixed_z),
        "folds": args.folds,
        "seeds": args.seeds,
        "steps": args.steps,
        "split": "outer folds hold out complete safe/fatal trajectory pairs; every fork state's 17 actions remain together",
        "action_conditioning": "one output logit per action; the correct action selects its outcome",
        "simulator_encoding": "agent-centred 9x9 categorical block/mob crop plus structured relative mobs, inventory and player state",
        "implementation": implementation_digests(Path(__file__)),
    }
    saved = torch.load(args.forks, weights_only=False, map_location="cpu")
    fixed = torch.load(args.fixed_z, weights_only=False, map_location="cpu")["latents"]
    if feature_path.exists():
        payload = torch.load(feature_path, weights_only=False, map_location="cpu")
        if payload["contract"] != contract:
            raise ValueError("fork representation feature contract changed")
        features, replay = payload["features"], payload["replay"]
    else:
        encoder, world, heads = load_models(args.phase1a, args.trajectory_phase2, base, config)
        features, replay = extract_features(saved, encoder, world, heads, config, fixed)
        atomic_torch(feature_path, {"contract": contract, "features": features, "replay": replay})

    target = saved["true_death"].bool()
    pair = saved["pair"].long()
    report = {"contract": contract, "replay": replay, "representations": {}}
    report["action_only_control"] = cross_validated_action_prior(
        target, pair, folds=args.folds, seed=config.seed + 11_000
    )
    for name in REPRESENTATIONS:
        print(f"probing {name} ({features[name].shape[1]} features)", flush=True)
        report["representations"][name] = cross_validated_probe(
            features[name].float(), target, pair, config,
            folds=args.folds, seeds=args.seeds, steps=args.steps,
        )
    atomic_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
