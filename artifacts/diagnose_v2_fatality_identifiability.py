"""Scale supervised z_t/action identifiability over support-v2 diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from artifacts.diagnose_fatality_identifiability import (
    VARIANTS,
    metrics,
    policy_features,
    record_features,
    train_probe,
)
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch
from artifacts.train_terminal_diversity_scaling import stratified_terminal_ranking
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import EpisodeCorpus, load_episodes
from d4mj.representation import Encoder
from d4mj.train import _cache_digest, cache_latents_to_store

VERSION = "fatality-identifiability-v2-scaling-v1"


def stable_tune(episode_id: str) -> bool:
    return int.from_bytes(hashlib.sha256(episode_id.encode()).digest()[:8], "little") % 10 == 0


def example_features(episodes: EpisodeCorpus, negative_ratio: int, seed: int) -> dict:
    state, action, label, group = [], [], [], []
    rng = torch.Generator().manual_seed(seed)
    for index, episode in enumerate(episodes):
        terminal = episode.terminated.nonzero().flatten()
        if len(terminal) != 1:
            raise ValueError("v2 identifiability expects one terminal per selected episode")
        safe = (~episode.terminated.bool() & ~episode.truncated.bool()).nonzero().flatten()
        count = min(negative_ratio, len(safe))
        chosen = safe[torch.randperm(len(safe), generator=rng)[:count]]
        steps = torch.cat([terminal, chosen])
        state.append(episode.latents[steps].flatten(1).float())
        action.append(episode.actions_taken[steps].long())
        label.append(episode.terminated[steps].bool())
        group.append(torch.full((len(steps),), index, dtype=torch.long))
    return {
        "state": torch.cat(state),
        "action": torch.cat(action),
        "label": torch.cat(label),
        "group": torch.cat(group),
    }


def matched_features(episodes: EpisodeCorpus) -> dict:
    state, action, label, group, same_action = [], [], [], [], 0
    for index, episode in enumerate(episodes):
        terminal = int(episode.terminated.nonzero().flatten()[0])
        candidates = torch.arange(terminal)
        candidates = candidates[~episode.terminated[:terminal] & ~episode.truncated[:terminal]]
        if not len(candidates):
            continue
        same = candidates[episode.actions_taken[candidates] == episode.actions_taken[terminal]]
        safe = int(same[-1]) if len(same) else int(candidates[-1])
        same_action += int(bool(len(same)))
        steps = torch.tensor([safe, terminal])
        state.append(episode.latents[steps].flatten(1).float())
        action.append(episode.actions_taken[steps].long())
        label.append(torch.tensor([False, True]))
        group.append(torch.full((2,), index, dtype=torch.long))
    return {
        "state": torch.cat(state),
        "action": torch.cat(action),
        "label": torch.cat(label),
        "group": torch.cat(group),
        "same_action_pairs": same_action,
    }


def metadata(episodes: EpisodeCorpus) -> dict[int, dict]:
    rows = {}
    for index, episode in enumerate(episodes):
        terminal = int(episode.terminated.nonzero().flatten()[0])
        rows[index] = {
            "pool": "support_v2",
            "epsilon": episode.epsilon,
            "fatal_action": int(episode.actions_taken[terminal]),
        }
    return rows


def remap_groups(features: dict, selected: list[int]) -> dict:
    mapping = torch.full((int(features["group"].max()) + 1,), -1, dtype=torch.long)
    mapping[torch.tensor(selected)] = torch.arange(len(selected))
    keep = mapping[features["group"]] >= 0
    return {
        "state": features["state"][keep],
        "action": features["action"][keep],
        "label": features["label"][keep],
        "group": mapping[features["group"][keep]],
    }


def summarize_runs(runs: list[dict]) -> dict:
    endpoints = ("tune_within_group_auc", "v2_dev_matched", "legacy_dev_matched", "policy_forks", "policy_executed", "policy_counterfactual")
    summary = {"runs": len(runs)}
    for endpoint in endpoints:
        if endpoint == "tune_within_group_auc":
            values = [row[endpoint] for row in runs]
            summary[endpoint] = {"mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values)}
            continue
        keys = ("auc", "within_group_auc", "conditional_logit_contrast")
        summary[endpoint] = {}
        for key in keys:
            values = [row[endpoint][key] for row in runs if row[endpoint][key] is not None]
            summary[endpoint][key] = None if not values else {
                "mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values)
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--policy-features", type=Path, required=True)
    parser.add_argument("--fork-starts", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negative-ratio", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--rungs", type=int, nargs="+", default=(32, 96, 300, 900, 1700, 3800))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = Config()
    support_manifest = args.support / "manifest.json"
    manifest = json.loads(support_manifest.read_text())
    if not manifest.get("complete"):
        raise ValueError("support-v2 collection is incomplete")
    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "support_manifest": file_digest(support_manifest),
        "prepared": file_digest(args.prepared),
        "policy_features": file_digest(args.policy_features),
        "fork_starts": file_digest(args.fork_starts),
        "forks": file_digest(args.forks),
        "negative_ratio": args.negative_ratio,
        "steps": args.steps,
        "seeds": args.seeds,
        "replicates": args.replicates,
        "requested_rungs": args.rungs,
        "split": "support-v2 declared TRAIN only; fixed hash 10% probe-tune; declared DEV evaluation; FINAL untouched",
        "exposure": "fixed optimizer steps and balanced class draws at every diversity rung",
        "implementation": implementation_digests(Path(__file__)),
    }
    raw = load_episodes(args.support)
    raw_fit = EpisodeCorpus(
        episode for episode in raw
        if episode.split == "train" and bool(episode.terminated.any()) and not stable_tune(episode.episode_id)
    )
    raw_tune = EpisodeCorpus(
        episode for episode in raw
        if episode.split == "train" and bool(episode.terminated.any()) and stable_tune(episode.episode_id)
    )
    raw_dev = EpisodeCorpus(
        episode for episode in raw if episode.split == "dev" and bool(episode.terminated.any())
    )
    encoder = Encoder(config).to(config.device)
    load(args.phase1a, config, part0=encoder)
    encoder.eval()
    digest = _cache_digest(encoder, config)
    caches = {}
    for name, episodes in (("fit", raw_fit), ("tune", raw_tune), ("dev", raw_dev)):
        caches[name] = cache_latents_to_store(
            encoder, episodes, config, args.out / f"{name}_latent_cache",
            source_contract={"support_manifest": contract["support_manifest"], "selection": name},
        )
    if any(episode.latent_digest != digest for corpus in caches.values() for episode in corpus):
        raise AssertionError("v2 probe cache digest mismatch")
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feature_path = args.out / "features.pt"
    feature_contract = contract | {"cache_digest": digest}
    if feature_path.exists():
        payload = torch.load(feature_path, weights_only=False, map_location="cpu")
        if payload["contract"] != feature_contract:
            raise ValueError("v2 identifiability feature contract changed")
        features = payload["features"]
    else:
        prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
        features = {
            "fit": example_features(caches["fit"], args.negative_ratio, config.seed + 12_000),
            "tune": example_features(caches["tune"], args.negative_ratio, config.seed + 12_001),
            "v2_dev_matched": matched_features(caches["dev"]),
            "legacy_dev_matched": record_features(prepared["records"]),
            "policy": policy_features(args.policy_features, args.fork_starts, args.forks),
        }
        atomic_torch(feature_path, {"contract": feature_contract, "features": features})

    meta = metadata(caches["fit"])
    requested = sorted(set(args.rungs))
    if requested[0] < 2 or requested[-1] > len(meta):
        parser.error(f"rungs must lie inside the fit universe {len(meta)}")
    if requested[-1] != len(meta):
        requested.append(len(meta))
    report_path = args.out / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["contract"] != contract:
            raise ValueError("v2 identifiability report contract changed")
    else:
        report = {
            "contract": contract,
            "cache_digest": digest,
            "corpus": {
                "fit_terminal_episodes": len(caches["fit"]),
                "tune_terminal_episodes": len(caches["tune"]),
                "dev_terminal_episodes": len(caches["dev"]),
                "dev_same_action_pairs": features["v2_dev_matched"]["same_action_pairs"],
                "final_terminal_episodes_untouched": manifest["split_terminal_counts"]["final"],
            },
            "cells": {},
        }
    for replicate in range(args.replicates):
        ranking = stratified_terminal_ranking(meta, config.seed + 12_100 + replicate)
        for size in requested:
            if size == len(meta) and replicate > 0:
                continue
            selected = ranking[:size]
            fit = remap_groups(features["fit"], selected)
            mean = fit["state"].mean(0)
            scale = fit["state"].std(0).clamp(min=1e-4)
            for variant_index, variant in enumerate(VARIANTS):
                name = f"k{size:04d}_r{replicate}_{variant}"
                if name in report["cells"]:
                    print(f"already complete {name}", flush=True)
                    continue
                runs = []
                for seed_index in range(args.seeds):
                    seed = config.seed + 12_200 + replicate * 1000 + variant_index * 10 + seed_index
                    model, tune_score = train_probe(
                        variant, fit, features["tune"], mean, scale, config,
                        seed, args.steps, selection="within_group_auc"
                    )
                    policy = features["policy"]
                    runs.append({
                        "seed": seed,
                        "tune_within_group_auc": tune_score,
                        "v2_dev_matched": metrics(model, features["v2_dev_matched"], mean, scale, config.device),
                        "legacy_dev_matched": metrics(model, features["legacy_dev_matched"], mean, scale, config.device),
                        "policy_forks": metrics(model, policy, mean, scale, config.device),
                        "policy_executed": metrics(model, policy, mean, scale, config.device, policy["trajectory"]),
                        "policy_counterfactual": metrics(model, policy, mean, scale, config.device, ~policy["trajectory"]),
                    })
                report["cells"][name] = {
                    "unique_fit_terminal_episodes": size,
                    "replicate": replicate,
                    "variant": variant,
                    "summary": summarize_runs(runs),
                    "runs": runs,
                }
                atomic_json(report_path, report)
                print(f"complete {name}", flush=True)
    atomic_json(report_path, report)
    print(f"complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
