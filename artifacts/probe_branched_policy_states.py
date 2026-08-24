"""Test whether competent-policy branches close the fixed-fork support gap."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import auc
from d4mj.config import Config

VERSION = "branched-policy-identifiability-v3"
CELLS = (
    ("state_action_small", "state_action", "small"),
    ("state_action_linear", "state_action", "linear"),
    ("state_only", "state_only", "small"),
    ("action_only", "action_only", "none"),
    ("state_shuffled_action_small", "state_shuffled_action", "small"),
    ("state_shuffled_action_linear", "state_shuffled_action", "linear"),
)


class Probe(nn.Module):
    def __init__(self, variant: str, architecture: str, width: int, actions: int):
        super().__init__()
        self.variant = variant
        self.actions = actions
        if variant == "action_only":
            self.bias = nn.Parameter(torch.zeros(actions))
        else:
            outputs = 1 if variant == "state_only" else actions
            if architecture == "linear":
                self.net = nn.Linear(width, outputs)
            elif architecture == "small":
                self.net = nn.Sequential(
                    nn.Linear(width, 64), nn.GELU(), nn.Linear(64, outputs)
                )
            else:
                raise ValueError(f"unknown probe architecture {architecture!r}")

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.variant == "action_only":
            return self.bias[action]
        logits = self.net(state)
        if self.variant == "state_only":
            return logits[:, 0]
        return logits.gather(1, action[:, None]).squeeze(1)


def split_seed(seed: int) -> str:
    draw = int.from_bytes(hashlib.sha256(f"branch-probe:{seed}".encode()).digest()[:8], "little")
    return "tune" if draw % 5 == 0 else "fit"


def load_collection(path: Path) -> tuple[dict, dict]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("branched-state collection is incomplete")
    rows = []
    for record in manifest["shards"]:
        shard = path / record["file"]
        if file_digest(shard) != record["sha256"]:
            raise ValueError(f"branched-state shard digest mismatch: {shard}")
        payload = torch.load(shard, weights_only=False, map_location="cpu")
        if int(payload["seed"]) != int(record["seed"]):
            raise ValueError(f"branched-state shard seed mismatch: {shard}")
        rows.append(payload)
    return manifest, {
        "state": torch.cat([row["latent"] for row in rows]).flatten(1).float(),
        "target": torch.cat([row["true_death"] for row in rows]).bool(),
        "seed": torch.cat([
            torch.full((len(row["latent"]),), int(row["seed"]), dtype=torch.long)
            for row in rows
        ]),
    }


def fixed_forks(latents_path: Path, forks_path: Path) -> tuple[dict, dict]:
    stored = torch.load(latents_path, weights_only=False, map_location="cpu")
    latents = stored["latents"]
    forks = torch.load(forks_path, weights_only=False, map_location="cpu")
    target = forks["true_death"].bool()
    if len(latents) != len(target):
        raise ValueError("fixed fork latent and truth counts differ")
    opportunity = target.any(1) & (~target).any(1)
    if not bool(opportunity.all()):
        raise ValueError("fixed fork endpoint contains a non-opportunity state")
    return {
        "state": latents.flatten(1).float(),
        "target": target,
        "pair": forks["pair"].long(),
    }, stored["contract"]


def flatten(features: dict) -> dict:
    states, actions = features["state"], features["target"].shape[1]
    return {
        "state": states[:, None].expand(-1, actions, -1).reshape(-1, states.shape[1]),
        "action": torch.arange(actions).repeat(len(states)),
        "label": features["target"].reshape(-1),
        "group": torch.arange(len(states))[:, None].expand(-1, actions).reshape(-1),
    }


def opportunity_subset(features: dict, mask: torch.Tensor) -> dict:
    target = features["target"]
    keep = mask & target.any(1) & (~target).any(1)
    if int(keep.sum()) < 2:
        raise ValueError("probe split has fewer than two terminal-opportunity states")
    return {"state": features["state"][keep], "target": target[keep]}


def normalized(features: dict, mean: torch.Tensor, scale: torch.Tensor, device: str):
    return (
        ((features["state"] - mean) / scale).to(device),
        features["action"].to(device),
        features["label"].float().to(device),
    )


def state_auc_rows(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    values = []
    for group in groups.unique(sorted=True):
        rows = groups == group
        if labels[rows].any() and (~labels[rows]).any():
            values.append(torch.tensor(auc(logits[rows], labels[rows])))
    if not values:
        raise ValueError("endpoint has no state with both outcomes")
    return torch.stack(values)


@torch.no_grad()
def metrics(model, features, mean, scale, config: Config) -> tuple[dict, torch.Tensor]:
    state, action, label = normalized(features, mean, scale, config.device)
    logits = model(state, action).cpu()
    labels = label.bool().cpu()
    groups = features["group"].long()
    centered = torch.empty_like(logits)
    contrasts = []
    for group in groups.unique(sorted=True):
        rows = groups == group
        centered[rows] = logits[rows] - logits[rows].mean()
        if labels[rows].any() and (~labels[rows]).any():
            contrasts.append(logits[rows][labels[rows]].mean() - logits[rows][~labels[rows]].mean())
    per_state = state_auc_rows(logits, labels, groups)
    return {
        "examples": len(labels),
        "fatal_examples": int(labels.sum()),
        "states": len(groups.unique()),
        "global_auc": float(auc(logits, labels)),
        "within_state_auc": float(per_state.mean()),
        "within_state_centered_auc": float(auc(centered, labels)),
        "conditional_logit_contrast": float(torch.stack(contrasts).mean()),
    }, per_state


def shuffled_actions(action: torch.Tensor, label: torch.Tensor, seed: int) -> torch.Tensor:
    shuffled = action.clone()
    rng = torch.Generator().manual_seed(seed)
    for outcome in (False, True):
        rows = (label == outcome).nonzero().flatten()
        shuffled[rows] = action[rows[torch.randperm(len(rows), generator=rng)]]
    return shuffled


def train_probe(
    variant: str,
    architecture: str,
    fit: dict,
    tune: dict,
    mean,
    scale,
    config: Config,
    seed: int,
    steps: int,
):
    torch.manual_seed(seed)
    model = Probe(
        variant, architecture, fit["state"].shape[1], config.n_actions
    ).to(config.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    state, action, label = normalized(fit, mean, scale, config.device)
    if variant == "state_shuffled_action":
        action = shuffled_actions(action.cpu(), label.bool().cpu(), seed + 1).to(config.device)
    positive = label.bool().nonzero().flatten()
    negative = (~label.bool()).nonzero().flatten()
    if not len(positive) or not len(negative):
        raise ValueError("fit split does not contain both outcomes")
    rng = torch.Generator(device=config.device).manual_seed(seed + 2)
    best, best_score = None, float("-inf")
    for step in range(steps):
        half = 256
        rows = torch.cat([
            positive[torch.randint(len(positive), (half,), generator=rng, device=config.device)],
            negative[torch.randint(len(negative), (half,), generator=rng, device=config.device)],
        ])
        loss = F.binary_cross_entropy_with_logits(
            model(state[rows], action[rows]), label[rows]
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            score, _ = metrics(model, tune, mean, scale, config)
            if score["within_state_auc"] > best_score:
                best_score = score["within_state_auc"]
                best = copy.deepcopy(model.state_dict())
    if best is None:
        raise AssertionError("probe produced no selected checkpoint")
    model.load_state_dict(best)
    return model, best_score


def interval(values: torch.Tensor) -> list[float]:
    ordered = values.sort().values
    return [float(ordered[int(0.025 * len(ordered))]), float(ordered[int(0.975 * len(ordered))])]


def cluster_means(values: torch.Tensor, cluster: torch.Tensor) -> torch.Tensor:
    return torch.stack([values[cluster == group].mean() for group in cluster.unique(sorted=True)])


def paired_interval(
    left: torch.Tensor,
    right: torch.Tensor,
    cluster: torch.Tensor,
    seed: int,
    draws: int = 5000,
) -> dict:
    if left.shape != right.shape:
        raise ValueError("paired endpoint rows differ")
    values = cluster_means(left - right, cluster)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(values), (draws, len(values)), generator=rng)
    samples = values[indices].mean(1)
    return {"difference": float(values.mean()), "ci95": interval(samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--fork-starts", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 100 or args.seeds < 1:
        parser.error("steps must be at least 100 and seeds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    config = Config(transition="direct", time_mixer="attention")
    manifest, collected = load_collection(args.collection)
    fixed, fixed_contract = fixed_forks(args.fork_starts, args.forks)
    fork_digest = file_digest(args.forks)
    if fixed_contract.get("phase1a") != manifest["contract"]["phase1a"]:
        raise ValueError("collection and fixed forks use different encoders")
    if fixed_contract.get("forks") != fork_digest:
        raise ValueError("fixed fork latents were built for a different truth table")
    reference = json.loads(args.reference.read_text())
    reference_contract = reference.get("contract", {})
    if reference_contract.get("fork_starts") != file_digest(args.fork_starts):
        raise ValueError("support-v2 reference uses different fixed fork latents")
    if reference_contract.get("forks") != fork_digest:
        raise ValueError("support-v2 reference uses a different fork truth table")
    reference_features = torch.load(
        args.reference_features, weights_only=False, map_location="cpu"
    )
    if reference_features.get("contract", {}).get("phase1a") != manifest["contract"]["phase1a"]:
        raise ValueError("support-v2 features use a different encoder")
    if reference_features.get("contract", {}).get("forks") != fork_digest:
        raise ValueError("support-v2 features use a different fork truth table")
    contract = {
        "version": VERSION,
        "collection_manifest": file_digest(args.collection / "manifest.json"),
        "fork_starts": file_digest(args.fork_starts),
        "forks": file_digest(args.forks),
        "reference": file_digest(args.reference),
        "reference_features": file_digest(args.reference_features),
        "steps": args.steps,
        "seeds": args.seeds,
        "fit_tune_split": "whole collection seed, sha256 80/20; fixed DEV forks untouched",
        "fit_selection": "terminal-opportunity roots only, as an oracle-filtered positive-control ceiling",
        "probe_architectures": "linear and the existing fork probe's 512->64->17 MLP; chosen on branch tune only",
        "checkpoint_selection": "best 100-step checkpoint on the cell's held-out whole-seed tune split",
        "action_path": "state trunk with one output per action; correct action selects its logit",
        "shuffle": "actions permuted within outcome class, preserving state labels and action-label marginals",
        "uncertainty": "paired bootstrap over the 52 whole trajectory pairs",
        "implementation": implementation_digests(Path(__file__)),
    }
    fit_seed = torch.tensor([split_seed(int(seed)) == "fit" for seed in collected["seed"]])
    tune_seed = ~fit_seed
    fit_roots = opportunity_subset(collected, fit_seed)
    tune_roots = opportunity_subset(collected, tune_seed)
    fit, tune, endpoint = flatten(fit_roots), flatten(tune_roots), flatten(fixed)
    mean = fit_roots["state"].mean(0)
    scale = fit_roots["state"].std(0).clamp(min=1e-4)
    report = {
        "contract": contract,
        "collection": {
            "states": manifest["states"],
            "opportunity_states": manifest["opportunity_states"],
            "fit_opportunity_states": len(fit_roots["state"]),
            "tune_opportunity_states": len(tune_roots["state"]),
            "fixed_dev_states": len(fixed["state"]),
        },
        "cells": {},
    }
    endpoint_rows = {}
    for cell_index, (cell, variant, architecture) in enumerate(CELLS):
        runs, rows = [], []
        for seed_index in range(args.seeds):
            seed = config.seed + 14_500 + cell_index * 100 + seed_index
            model, tune_score = train_probe(
                variant,
                architecture,
                fit,
                tune,
                mean,
                scale,
                config,
                seed,
                args.steps,
            )
            tune_metrics, _ = metrics(model, tune, mean, scale, config)
            fixed_metrics, fixed_rows = metrics(model, endpoint, mean, scale, config)
            runs.append({
                "seed": seed,
                "selected_tune_within_state_auc": tune_score,
                "tune": tune_metrics,
                "fixed_policy_forks": fixed_metrics,
            })
            rows.append(fixed_rows)
        ensemble_rows = torch.stack(rows).mean(0)
        endpoint_rows[cell] = ensemble_rows
        report["cells"][cell] = {
            "variant": variant,
            "architecture": architecture,
            "runs": runs,
            "fixed_policy_forks": {
                "within_state_auc_mean": float(ensemble_rows.mean()),
                "seed_minimum": min(run["fixed_policy_forks"]["within_state_auc"] for run in runs),
                "seed_maximum": max(run["fixed_policy_forks"]["within_state_auc"] for run in runs),
            },
        }
        print(f"complete: {cell}", flush=True)

    architecture_scores = {
        architecture: sum(
            run["selected_tune_within_state_auc"]
            for run in report["cells"][f"state_action_{architecture}"]["runs"]
        )
        / args.seeds
        for architecture in ("small", "linear")
    }
    selected_architecture = max(
        architecture_scores, key=lambda name: (architecture_scores[name], name)
    )
    report["architecture_selection"] = {
        "criterion": "highest mean held-out branch-tune within-state AUC; fixed policy forks not inspected",
        "scores": architecture_scores,
        "selected": selected_architecture,
    }

    logged = reference_features["features"]
    logged_mean = logged["fit"]["state"].mean(0)
    logged_scale = logged["fit"]["state"].std(0).clamp(min=1e-4)
    logged_runs, logged_rows = [], []
    for seed_index in range(args.seeds):
        seed = config.seed + 14_800 + seed_index
        model, tune_score = train_probe(
            "state_action",
            selected_architecture,
            logged["fit"],
            logged["tune"],
            logged_mean,
            logged_scale,
            config,
            seed,
            args.steps,
        )
        tune_metrics, _ = metrics(
            model, logged["tune"], logged_mean, logged_scale, config
        )
        fixed_metrics, fixed_rows = metrics(
            model, endpoint, logged_mean, logged_scale, config
        )
        logged_runs.append({
            "seed": seed,
            "selected_tune_within_group_auc": tune_score,
            "tune": tune_metrics,
            "fixed_policy_forks": fixed_metrics,
        })
        logged_rows.append(fixed_rows)
    logged_ensemble = torch.stack(logged_rows).mean(0)
    report["logged_support_control"] = {
        "training_examples": len(logged["fit"]["label"]),
        "training_groups": len(logged["fit"]["group"].unique()),
        "runs": logged_runs,
        "fixed_policy_forks": {
            "within_state_auc_mean": float(logged_ensemble.mean()),
            "seed_minimum": min(run["fixed_policy_forks"]["within_state_auc"] for run in logged_runs),
            "seed_maximum": max(run["fixed_policy_forks"]["within_state_auc"] for run in logged_runs),
        },
    }
    correct = endpoint_rows[f"state_action_{selected_architecture}"]
    shuffled = endpoint_rows[f"state_shuffled_action_{selected_architecture}"]
    pair = fixed["pair"]
    report["comparisons"] = {
        "state_action_minus_shuffled": paired_interval(
            correct,
            shuffled,
            pair,
            config.seed + 14_900,
        ),
        "state_action_minus_state_only": paired_interval(
            correct, endpoint_rows["state_only"], pair, config.seed + 14_901
        ),
        "state_action_minus_action_only": paired_interval(
            correct, endpoint_rows["action_only"], pair, config.seed + 14_902
        ),
        "state_action_minus_matched_logged_support": paired_interval(
            correct, logged_ensemble, pair, config.seed + 14_903
        ),
        "historical_reference_not_used": {
            "reason": "historical within_group_auc is pooled after centering; this report uses mean per-state AUC",
            "reported_full_support_v2_within_group_auc": reference["cells"]["k7244_r0_state_action"]["summary"]["policy_forks"]["within_group_auc"],
        },
    }
    branch_vs_shuffle = report["comparisons"]["state_action_minus_shuffled"]["ci95"][0] > 0
    branch_vs_action = report["comparisons"]["state_action_minus_action_only"]["ci95"][0] > 0
    branch_vs_reference = report["comparisons"]["state_action_minus_matched_logged_support"]["ci95"][0] > 0
    report["decision"] = {
        "positive_gate": bool(branch_vs_shuffle and branch_vs_action and branch_vs_reference),
        "rule": "lower paired trajectory-bootstrap bound above shuffled-action, action-only, and the matched logged-support probe",
        "phase1b_branch_training_authorized": bool(branch_vs_shuffle and branch_vs_action and branch_vs_reference),
    }
    atomic_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
