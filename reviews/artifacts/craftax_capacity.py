"""Capacity ablation: is the JEPA latent's loss of task state a bottleneck size?

The oracle showed food/wood/stone/sapling readable from raw pixels at R^2 ~1.0
(they are drawn as HUD icons) and near-absent from the latent. Two candidate
causes: the latent is too small to carry them, or the objective never asks it
to. This moves ONE axis -- `d_bottleneck` -- and re-runs the same oracle.

d_bottleneck 16 (the trained baseline) / 32 / 64 gives per-frame latents of
256 / 512 / 1024 dims against 12,288 pixels. World phase only: BC and
imagination cannot answer this question and would only add confounds.

Everything else is held at the T-JEPA baseline, including the 20,000-update
budget, batch 8, terminal fraction, jumps, EMA schedule and seed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.craftax_run import SPLIT_SEED, _dev_cosine, _fixed_dev_batches
from d4_mamba_jepa.craftax_runners import craftax_jepa_config, train_craftax_jepa_world
from d4_mamba_jepa.data import (
    load_episode_replay,
    subset_replay,
    whole_episode_splits,
)

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
REPLAY_SHA = "7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b"
OUT = REPO_ROOT / "outputs/d4_mamba_jepa/craftax_capacity"
SEED = 20260727  # identical to the baseline run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bottlenecks", default="")
    p.add_argument(
        "--grid", default="",
        help="comma-separated n_latents:d_bottleneck pairs, e.g. 64:16,256:16",
    )
    p.add_argument("--world-steps", type=int, default=20_000)
    p.add_argument("--backend", default="transformer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    replay = load_episode_replay(REPLAY, expected_sha256=REPLAY_SHA)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])

    if args.grid:
        rungs = [tuple(int(x) for x in item.split(":"))
                 for item in args.grid.split(",") if item.strip()]
    else:
        rungs = [(16, int(x.strip())) for x in args.bottlenecks.split(",") if x.strip()]

    results = {}
    for n_latents, d in rungs:
        cfg = replace(craftax_jepa_config(args.backend),
                      n_latents=n_latents, d_bottleneck=d)
        label = (f"d_bottleneck_{d}" if n_latents == 16
                 else f"n_latents_{n_latents}_d_bottleneck_{d}")
        arm_dir = OUT / label
        print(f"=== n_latents={n_latents} d_bottleneck={d} "
              f"(n_spatial={cfg.n_spatial}, latent {cfg.n_spatial * cfg.d_spatial} dims) ===",
              flush=True)
        dev_batches = _fixed_dev_batches(
            dev_replay, cfg=cfg, count=16, batch_size=8, seed=SPLIT_SEED + 1
        )
        world, _, history = train_craftax_jepa_world(
            replay=train_replay, cfg=cfg, world_steps=args.world_steps,
            batch_size=8, learning_rate=1e-4, seed=SEED, device=device,
            output_dir=arm_dir,
        )
        cosine = _dev_cosine(world, dev_batches, device)
        results[label] = {
            "n_latents": n_latents,
            "d_bottleneck": d,
            "n_spatial": cfg.n_spatial,
            "latent_dims": cfg.n_spatial * cfg.d_spatial,
            "dev_cosine": cosine,
            "final_jepa_loss": history[-1]["jepa"],
            "online_std": history[-1]["online_std"],
            "output_dir": str(arm_dir),
        }
        print(f"  dev_cosine={cosine:.4f} jepa={history[-1]['jepa']:.4f} "
              f"onstd={history[-1]['online_std']:.4f}", flush=True)
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (OUT / "capacity_report.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
