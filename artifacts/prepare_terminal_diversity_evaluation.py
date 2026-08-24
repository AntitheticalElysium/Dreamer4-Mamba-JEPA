"""Prepare the large support-v2 DEV terminal set under the fixed Phase-1A encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    data_digests,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch, compact_records, terminal_pair_rows
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import EpisodeCorpus, load_episodes
from d4mj.representation import Encoder
from d4mj.train import _cache_digest, cache_latents_to_store

VERSION = "terminal-diversity-evaluation-v2"
SELF_PATH = "artifacts/prepare_terminal_diversity_evaluation.py"
BROKEN_RESUME_DIGEST = "7654bb867cb04bd266d766e7b645fd4a78fd1afc5e47ad0e38f2555c071fce8d"


def _resume_compatible(saved: dict, current: dict) -> bool:
    if saved == current:
        return True
    saved = dict(saved)
    current = dict(current)
    saved_implementation = dict(saved.pop("implementation", {}))
    current_implementation = dict(current.pop("implementation", {}))
    saved_self = saved_implementation.pop(SELF_PATH, None)
    current_implementation.pop(SELF_PATH, None)
    return (
        saved == current
        and saved_implementation == current_implementation
        and saved_self == BROKEN_RESUME_DIGEST
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--base-prepared", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "prepared.pt"
    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "base_prepared": file_digest(args.base_prepared),
        "data": data_digests(args.support),
        "split": "support-v2 declared DEV episodes only; FINAL untouched",
        "implementation": implementation_digests(Path(__file__)),
    }
    if output_path.exists():
        saved = torch.load(output_path, weights_only=False, map_location="cpu")
        if not _resume_compatible(saved.get("evaluation_contract", {}), contract):
            raise ValueError("terminal-diversity evaluation contract changed")
        print(f"already complete: {output_path}")
        return

    config = Config()
    base = torch.load(args.base_prepared, weights_only=False, map_location="cpu")
    raw = load_episodes(args.support)
    dev = EpisodeCorpus(
        episode
        for episode in raw
        if episode.split == "dev" and bool(episode.terminated.any())
    )
    if not dev:
        raise ValueError("support-v2 has no terminal DEV episodes")
    encoder = Encoder(config).to(config.device)
    load(args.phase1a, config, part0=encoder)
    encoder.eval()
    if _cache_digest(encoder, config) != base["cache_digest"]:
        raise ValueError("Phase-1A cache digest differs from the fixed fatality direction")
    cached = cache_latents_to_store(
        encoder,
        dev,
        config,
        args.cache,
        source_contract={
            "support": contract["data"]["support_manifest"],
            "split": "dev-terminal",
        },
    )

    records, action_matched = [], 0
    for episode_index, episode in enumerate(cached):
        rows, full = terminal_pair_rows([episode], "support")
        compact = compact_records(full, config.sequence_long)
        group = episode_index
        for record in compact:
            record["group"] = group
            record["episode_index"] = episode_index
        records.extend(compact)
        action_matched += int(rows["same_action_safe_pairs"])
    report = {
        "contract": contract,
        "cache_digest": base["cache_digest"],
        "train": base["report"]["train"],
        "dev": {
            "support_v2": {
                "episodes": len(dev),
                "terminal_episodes": len(dev),
                "examples": len(records),
                "same_action_safe_pairs": action_matched,
            }
        },
        "evaluation": {
            "support_v2": {
                "pairs": len(dev),
                "examples": len(records),
                "same_action_safe_pairs": action_matched,
                "group_offset": 0,
            }
        },
        "training_geometry": base["report"]["training_geometry"],
    }
    prepared = {
        "contract": base["contract"],
        "evaluation_contract": contract,
        "cache_digest": base["cache_digest"],
        "direction": base["direction"],
        "action_means": base["action_means"],
        "covariance": base["covariance"],
        "precision": base["precision"],
        "train_records": base["train_records"],
        "records": records,
        "report": report,
    }
    atomic_torch(output_path, prepared)
    atomic_json(args.out / "preparation_report.json", report)
    print(f"complete: {output_path}", flush=True)


if __name__ == "__main__":
    main()
