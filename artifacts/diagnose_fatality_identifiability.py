"""Can a supervised probe identify one-step death from the frozen state and action?"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from artifacts.phase1b_geometry_common import atomic_torch, auc, finite_json
from artifacts.prepare_phase1b_archive_geometry import split_pools
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.train import _cache_digest, _cache_episode

VERSION = "fatality-identifiability-v1"
VARIANTS = ("state_action", "state_only", "action_only", "state_shuffled_action")


class Probe(nn.Module):
    def __init__(self, variant: str, width: int, actions: int):
        super().__init__()
        self.variant = variant
        self.actions = actions
        state_width = 0 if variant == "action_only" else width
        action_width = 0 if variant == "state_only" else actions
        self.net = nn.Sequential(
            nn.Linear(state_width + action_width, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = []
        if self.variant != "action_only":
            values.append(state)
        if self.variant != "state_only":
            values.append(torch.nn.functional.one_hot(action, self.actions).float())
        return self.net(torch.cat(values, dim=-1)).squeeze(-1)


def stratified_fit_tune(entries: list[tuple[str, int, object]], seed: int):
    groups = defaultdict(list)
    for row in entries:
        pool, _, episode = row
        groups[(pool, bool(episode.terminated.any()))].append(row)
    rng = torch.Generator().manual_seed(seed)
    fit, tune = [], []
    for rows in groups.values():
        order = torch.randperm(len(rows), generator=rng).tolist()
        cut = max(1, int(0.8 * len(rows))) if len(rows) > 1 else len(rows)
        fit.extend(rows[index] for index in order[:cut])
        tune.extend(rows[index] for index in order[cut:])
    return fit, tune


def selected_negative_steps(entries, ratio: int, seed: int) -> dict[int, set[int]]:
    positive = sum(int(episode.terminated.sum()) for _, _, episode in entries)
    counts = [len(episode) - int(episode.terminated.sum()) for _, _, episode in entries]
    total = sum(counts)
    wanted = min(total, ratio * positive)
    if positive == 0 or wanted == 0:
        raise ValueError("identifiability split has no usable positive/negative examples")
    choice = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:wanted].sort().values
    cumulative = torch.tensor(counts).cumsum(0)
    episode_positions = torch.searchsorted(cumulative, choice, right=True)
    previous = torch.cat([torch.zeros(1, dtype=torch.long), cumulative[:-1]])
    local_ranks = choice - previous[episode_positions]
    selected = defaultdict(set)
    for position in episode_positions.unique().tolist():
        _, _, episode = entries[position]
        safe = (~episode.terminated.bool()).nonzero().flatten()
        ranks = local_ranks[episode_positions == position]
        selected[position].update(int(value) for value in safe[ranks])
    return selected


@torch.no_grad()
def extract_features(
    encoder: Encoder,
    entries,
    config: Config,
    *,
    negative_ratio: int | None,
    seed: int,
):
    selected = (
        selected_negative_steps(entries, negative_ratio, seed)
        if negative_ratio is not None
        else None
    )
    digest = _cache_digest(encoder, config)
    state, action, label, group, pool = [], [], [], [], []
    for position, (pool_name, episode_index, episode) in enumerate(entries):
        cached = _cache_episode(encoder, episode, config, digest)
        if selected is None:
            steps = torch.arange(len(episode))
        else:
            safe = sorted(selected.get(position, set()))
            terminal = episode.terminated.nonzero().flatten().tolist()
            steps = torch.tensor(sorted(safe + terminal), dtype=torch.long)
        state.append(cached.latents[steps])
        action.append(episode.actions_taken[steps])
        label.append(episode.terminated[steps])
        group.append(torch.full((len(steps),), position, dtype=torch.long))
        pool.extend([pool_name] * len(steps))
        if (position + 1) % 32 == 0 or position + 1 == len(entries):
            print(f"features: {position + 1}/{len(entries)} episodes", flush=True)
    return {
        "state": torch.cat(state).flatten(1).float(),
        "action": torch.cat(action).long(),
        "label": torch.cat(label).bool(),
        "group": torch.cat(group).long(),
        "pool": pool,
    }


def record_features(records: list[dict]) -> dict:
    states, actions, labels, groups, pools = [], [], [], [], []
    for group, record in enumerate(records):
        steps = record["transitions"].long()
        states.append(record["latents"][steps])
        actions.append(record["actions_taken"][steps])
        labels.append(record["labels"].bool())
        groups.append(torch.full((len(steps),), group, dtype=torch.long))
        pools.extend([record["pool"]] * len(steps))
    return {
        "state": torch.cat(states).flatten(1).float(),
        "action": torch.cat(actions).long(),
        "label": torch.cat(labels).bool(),
        "group": torch.cat(groups),
        "pool": pools,
    }


def policy_features(features: Path, starts: Path, forks: Path) -> dict:
    payload = torch.load(features, weights_only=False, map_location="cpu")["data"]
    start = torch.load(starts, weights_only=False, map_location="cpu")["latents"]
    fork = torch.load(forks, weights_only=False, map_location="cpu")
    group = payload["group"].long()
    return {
        "state": start[group].flatten(1).float(),
        "action": payload["action"].long(),
        "label": payload["target"].bool(),
        "group": group,
        "trajectory": payload["action"].long() == fork["trajectory_action"][group],
    }


def normalized(features: dict, mean: torch.Tensor, scale: torch.Tensor, device: str):
    return (
        ((features["state"] - mean) / scale).to(device),
        features["action"].to(device),
        features["label"].float().to(device),
    )


@torch.no_grad()
def metrics(model, features, mean, scale, device, mask=None):
    if mask is None:
        mask = torch.ones(len(features["label"]), dtype=torch.bool)
    state, action, label = normalized(
        {key: value[mask] if isinstance(value, torch.Tensor) and len(value) == len(mask) else value
         for key, value in features.items()},
        mean,
        scale,
        device,
    )
    logits = model(state, action).cpu()
    labels = label.bool().cpu()
    groups = features["group"][mask].long()
    contrasts = []
    centered = torch.empty_like(logits)
    for value in groups.unique():
        rows = groups == value
        centered[rows] = logits[rows] - logits[rows].mean()
        if labels[rows].any() and (~labels[rows]).any():
            contrasts.append(logits[rows][labels[rows]].mean() - logits[rows][~labels[rows]].mean())
    return {
        "examples": len(labels),
        "fatal_examples": int(labels.sum()),
        "auc": float(auc(logits, labels)),
        "within_group_auc": float(auc(centered, labels)),
        "conditional_logit_contrast": float(torch.stack(contrasts).mean()) if contrasts else None,
        "groups_with_both_outcomes": len(contrasts),
    }


def train_probe(
    variant, fit, tune, mean, scale, config, seed, steps, *, selection="auc"
):
    device = config.device
    torch.manual_seed(seed)
    model = Probe(variant, fit["state"].shape[1], config.n_actions).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    state, action, label = normalized(fit, mean, scale, device)
    positive = label.bool().nonzero().flatten()
    negative = (~label.bool()).nonzero().flatten()
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    shuffled = action.clone()
    if variant == "state_shuffled_action":
        cpu_rng = torch.Generator().manual_seed(seed + 2)
        for rows in (positive.cpu(), negative.cpu()):
            shuffled[rows.to(device)] = action[rows[torch.randperm(len(rows), generator=cpu_rng)].to(device)]
    best, best_auc = None, float("-inf")
    for step in range(steps):
        half = 128
        rows = torch.cat([
            positive[torch.randint(len(positive), (half,), generator=rng, device=device)],
            negative[torch.randint(len(negative), (half,), generator=rng, device=device)],
        ])
        logits = model(state[rows], shuffled[rows])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label[rows])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            score = metrics(model, tune, mean, scale, device)[selection]
            if score > best_auc:
                best_auc, best = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, best_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--policy-features", type=Path, required=True)
    parser.add_argument("--fork-starts", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negative-ratio", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    feature_path = args.out / "features.pt"
    config = Config()
    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "prepared": file_digest(args.prepared),
        "policy_features": file_digest(args.policy_features),
        "fork_starts": file_digest(args.fork_starts),
        "forks": file_digest(args.forks),
        "negative_ratio": args.negative_ratio,
        "steps": args.steps,
        "seeds": args.seeds,
        "split": "production episode-level TRAIN/DEV; TRAIN split again by whole episode for fit/tune",
        "shuffle": "within outcome class, preserving the action-label marginal while breaking state-action pairing",
        "implementation": implementation_digests(Path(__file__)),
    }
    if feature_path.exists():
        saved = torch.load(feature_path, weights_only=False, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("identifiability feature contract changed")
        features = saved["features"]
    else:
        train_pools, dev_pools = split_pools(config, 320)
        train_entries = [
            (pool, index, episode)
            for pool, episodes in train_pools.items()
            for index, episode in enumerate(episodes)
        ]
        dev_entries = [
            (pool, index, episode)
            for pool, episodes in dev_pools.items()
            for index, episode in enumerate(episodes)
        ]
        fit_entries, tune_entries = stratified_fit_tune(train_entries, config.seed + 9900)
        encoder = Encoder(config).to(config.device)
        load(args.phase1a, config, part0=encoder)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
        if _cache_digest(encoder, config) != prepared["cache_digest"]:
            raise ValueError("identifiability encoder differs from prepared states")
        features = {
            "fit": extract_features(encoder, fit_entries, config, negative_ratio=args.negative_ratio, seed=config.seed + 9901),
            "tune": extract_features(encoder, tune_entries, config, negative_ratio=args.negative_ratio, seed=config.seed + 9902),
            "archive_dev_natural": extract_features(encoder, dev_entries, config, negative_ratio=None, seed=config.seed + 9903),
            "archive_dev_matched": record_features(prepared["records"]),
            "policy": policy_features(args.policy_features, args.fork_starts, args.forks),
        }
        atomic_torch(feature_path, {"contract": contract, "features": features})

    mean = features["fit"]["state"].mean(0)
    scale = features["fit"]["state"].std(0).clamp(min=1e-4)
    report = {"contract": contract, "variants": {}}
    for variant_index, variant in enumerate(VARIANTS):
        runs = []
        for seed_index in range(args.seeds):
            seed = config.seed + 9950 + variant_index * 10 + seed_index
            model, tune_auc = train_probe(
                variant, features["fit"], features["tune"], mean, scale, config, seed, args.steps
            )
            policy = features["policy"]
            runs.append({
                "seed": seed,
                "tune_auc": tune_auc,
                "archive_dev_natural": metrics(model, features["archive_dev_natural"], mean, scale, config.device),
                "archive_dev_matched": metrics(model, features["archive_dev_matched"], mean, scale, config.device),
                "policy_forks": metrics(model, policy, mean, scale, config.device),
                "policy_executed": metrics(model, policy, mean, scale, config.device, policy["trajectory"]),
                "policy_counterfactual": metrics(model, policy, mean, scale, config.device, ~policy["trajectory"]),
            })
        report["variants"][variant] = runs
        print(f"identifiability complete: {variant}", flush=True)
    atomic_json(args.out / "report.json", finite_json(report))
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
