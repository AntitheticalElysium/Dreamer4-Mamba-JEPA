"""Does the tokenizer under-represent the death transition?

Compares pixel-space and latent-space displacement for realized death transitions
against action-matched ordinary transitions drawn from the same episodes. The
statistic is the ratio of ratios: latent(death/ordinary) over pixel(death/ordinary).
A value far below 1 means the encoder compresses the death event relative to its
pixel magnitude, which would place the information loss upstream of the dynamics
objective. A value near 1 clears the encoder and leaves the loss downstream.

Action matching is required, not cosmetic: movement actions scroll the whole
player-centred map, so an unmatched comparison measures action mix rather than
death. Evaluation only -- nothing is trained and no world model is loaded.
"""

from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    data_digests,
    file_digest,
    implementation_digests,
)
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import Episode
from d4mj.representation import Encoder
from d4mj.train import _cache_digest, cache_latents

VERSION = "encoder-fatality-fidelity-v1"
MINIMUM_ORDINARY = 20


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@torch.no_grad()
def measure(encoder: Encoder, entries: list[dict], direction: torch.Tensor, config: Config):
    """Per-action death and ordinary displacement, in pixels and in latents."""
    episodes = [Episode(**entry) for entry in entries]
    cached = cache_latents(encoder, episodes, config)

    death = collections.defaultdict(lambda: {"pixel": [], "latent": [], "along": []})
    ordinary = collections.defaultdict(lambda: {"pixel": [], "latent": [], "along": []})

    for episode, entry in zip(cached, entries):
        frames = entry["observations"].float()
        latents = episode.latents.flatten(1).float().to(direction.device)
        actions = entry["actions_taken"]
        last = len(actions) - 1

        for step in range(len(actions)):
            delta = latents[step + 1] - latents[step]
            target = death if step == last else ordinary
            bucket = target[int(actions[step])]
            bucket["pixel"].append(float((frames[step + 1] - frames[step]).abs().mean()))
            bucket["latent"].append(float(delta.pow(2).mean().sqrt()))
            bucket["along"].append(float((delta @ direction).abs()))

    return death, ordinary


def summarize(death, ordinary) -> dict:
    per_action, weights = {}, []
    for action in sorted(death):
        if len(ordinary.get(action, {}).get("pixel", [])) < MINIMUM_ORDINARY:
            continue
        pixel = _mean(death[action]["pixel"]) / _mean(ordinary[action]["pixel"])
        latent = _mean(death[action]["latent"]) / _mean(ordinary[action]["latent"])
        along = _mean(death[action]["along"]) / _mean(ordinary[action]["along"])
        per_action[str(action)] = {
            "death_examples": len(death[action]["pixel"]),
            "ordinary_examples": len(ordinary[action]["pixel"]),
            "pixel_ratio": pixel,
            "latent_ratio": latent,
            "direction_ratio": along,
            "compression": latent / pixel,
        }
        weights.append(len(death[action]["pixel"]))

    if not per_action:
        raise RuntimeError("no action had both death and sufficient ordinary transitions")

    total = sum(weights)
    rows = list(per_action.values())
    weighted = {
        key: sum(row[key] * row["death_examples"] for row in rows) / total
        for key in ("pixel_ratio", "latent_ratio", "direction_ratio")
    }
    compressions = [row["compression"] for row in rows]
    return {
        "actions_compared": len(per_action),
        "death_examples": total,
        "weighted_pixel_ratio": weighted["pixel_ratio"],
        "weighted_latent_ratio": weighted["latent_ratio"],
        "weighted_direction_ratio": weighted["direction_ratio"],
        "weighted_compression": weighted["latent_ratio"] / weighted["pixel_ratio"],
        "macro_compression": statistics.mean(compressions),
        "compression_min": min(compressions),
        "compression_max": max(compressions),
        "per_action": per_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = Config()
    encoder = Encoder(config).to(config.device)
    load(args.phase1a, config, part0=encoder)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    prepared = torch.load(args.prepared, weights_only=False)
    digest = _cache_digest(encoder, config)
    if digest != prepared["cache_digest"]:
        raise ValueError(
            f"encoder produces cache {digest}; the fitted direction was measured "
            f"under {prepared['cache_digest']}"
        )
    direction = prepared["direction"].to(config.device).flatten()

    payload = torch.load(args.support, weights_only=False)
    terminal = [e for e in payload["episodes"] if bool(e["terminated"].any())]
    entries = terminal[: args.episodes]
    if not entries:
        raise RuntimeError("support corpus contains no terminal episodes")

    death, ordinary = measure(encoder, entries, direction, config)
    report = {
        "contract": {
            "version": VERSION,
            "evaluation": "realized death transitions against action-matched ordinary transitions",
            "statistic": "latent(death/ordinary) divided by pixel(death/ordinary)",
            "world_model_used": False,
            "minimum_ordinary_per_action": MINIMUM_ORDINARY,
            "requested_episodes": args.episodes,
            "terminal_episodes_available": len(terminal),
            "terminal_episodes_used": len(entries),
            "cache_digest": digest,
            "direction_source": str(args.prepared.resolve()),
            "inputs": {
                "phase1a": file_digest(args.phase1a),
                "support": file_digest(args.support),
                "prepared": file_digest(args.prepared),
            },
            "data": data_digests(),
            "implementation": implementation_digests(Path(__file__).resolve()),
        },
        "summary": summarize(death, ordinary),
    }
    atomic_json(args.out, report)
    summary = report["summary"]
    print(
        f"pixel {summary['weighted_pixel_ratio']:.2f}x | "
        f"latent {summary['weighted_latent_ratio']:.2f}x | "
        f"direction {summary['weighted_direction_ratio']:.2f}x | "
        f"compression {summary['weighted_compression']:.3f} "
        f"(macro {summary['macro_compression']:.3f})"
    )
    print(f"complete: {args.out}")


if __name__ == "__main__":
    main()
