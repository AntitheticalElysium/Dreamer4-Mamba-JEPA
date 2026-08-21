"""Evaluate trained latent heads without using them as the rescue endpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import auc, cluster_auc_interval, finite_json
from artifacts.train_generated_latent_outcome_shaping import LatentContinuationHead
from d4mj.checkpoint import load
from d4mj.config import Config


@torch.no_grad()
def score(
    head: LatentContinuationHead,
    latent: torch.Tensor,
    fatal: torch.Tensor,
    group: torch.Tensor,
    *,
    bootstraps: int,
    seed: int,
) -> dict:
    logits = []
    for start in range(0, len(latent), 512):
        logits.append(head(latent[start : start + 512].to(head.net[0].weight.device)).cpu())
    continuation_logit = torch.cat(logits).flatten()
    fatal = fatal.bool().cpu().flatten()
    group = group.long().cpu().flatten()
    death_score = -continuation_logit
    continuation = (~fatal).float()
    probability = continuation_logit.sigmoid()
    return {
        "examples": len(fatal),
        "fatal_examples": int(fatal.sum()),
        "death_auc": auc(death_score, fatal),
        "death_auc_ci95": cluster_auc_interval(
            death_score,
            fatal,
            group,
            samples=bootstraps,
            seed=seed,
        ),
        "bce": float(F.binary_cross_entropy_with_logits(continuation_logit, continuation)),
        "accuracy": float(((probability < 0.5) == fatal).float().mean()),
        "continuation_on_fatal": float(probability[fatal].mean()),
        "continuation_on_safe": float(probability[~fatal].mean()),
    }


def load_head(path: Path, config: Config) -> LatentContinuationHead:
    head = LatentContinuationHead(config).to(config.device).eval()
    load(path, config, part1=head)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-features", type=Path, required=True)
    parser.add_argument("--fork-features", type=Path, required=True)
    parser.add_argument(
        "--model", nargs=2, action="append", metavar=("NAME", "CHECKPOINT"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    models = [(name, Path(path)) for name, path in args.model]
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    if any(re.fullmatch(r"[a-z0-9_-]+", name) is None for name in names):
        parser.error("model names may contain lowercase letters, digits, _ and -")
    config = Config(transition="direct", time_mixer="attention")
    inputs = {
        "models": {name: file_digest(path) for name, path in models},
        "archive_features": {
            name: file_digest(args.archive_features / f"{name}.pt") for name in names
        },
        "fork_features": {
            name: file_digest(args.fork_features / f"{name}.pt") for name in names
        },
    }
    contract = {
        "version": "generated-latent-outcome-head-evaluation-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/train_generated_latent_outcome_shaping.py"),
        ),
        "role": (
            "validity check only; trained-head performance cannot establish that "
            "the generated representation improved"
        ),
        "bootstraps": args.bootstraps,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("outcome-head evaluation contract changed")
    atomic_json(contract_path, contract)

    results = {}
    for index, (name, path) in enumerate(models):
        head = load_head(path, config)
        archive = torch.load(
            args.archive_features / f"{name}.pt",
            weights_only=False,
            map_location="cpu",
        )["paths"]["reset16"]
        fork = torch.load(
            args.fork_features / f"{name}.pt",
            weights_only=False,
            map_location="cpu",
        )["variants"]["generated"]
        support = torch.tensor([pool == "support" for pool in archive["pool"]])
        chosen = torch.ones(len(fork["target"]), dtype=torch.bool)
        results[name] = {
            "archive": {
                "combined": {
                    representation: score(
                        head,
                        archive[representation],
                        archive["label"],
                        archive["group"],
                        bootstraps=args.bootstraps,
                        seed=config.seed + 18_500 + index * 100 + offset,
                    )
                    for offset, representation in enumerate(("target", "predicted"))
                },
                "support": {
                    representation: score(
                        head,
                        archive[representation][support],
                        archive["label"][support],
                        archive["group"][support],
                        bootstraps=args.bootstraps,
                        seed=config.seed + 18_520 + index * 100 + offset,
                    )
                    for offset, representation in enumerate(("target", "predicted"))
                },
            },
            "policy_forks": {
                representation: score(
                    head,
                    fork[representation][chosen],
                    fork["target"][chosen],
                    fork["group"][chosen],
                    bootstraps=args.bootstraps,
                    seed=config.seed + 18_550 + index * 100 + offset,
                )
                for offset, representation in enumerate(("observed", "predicted"))
            },
        }
        head.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        args.out / "report.json",
        finite_json({"contract": contract, "models": results}),
    )
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
