"""T-BASE encoder control: does RECONSTRUCTION retain what JEPA discards?

DIAGNOSTIC CONTROL ONLY. The JEPA line is deliberately reconstruction-free and
the end goal is SIGReg; this run exists to identify what a reconstruction-free
objective must supply by other means, and is NOT a route to adopting a decoder.

The oracle found food/wood/stone/sapling readable from raw pixels at R^2 ~1.0
(HUD icons) and near-absent from the JEPA latent, with health -- the only target
entering the loss, via reward -- the sole exception. An MAE reconstruction
objective must retain everything visible in order to redraw it, so it is the
sharpest available test of "the latent keeps only what a loss term demands".

Only the tokenizer/encoder phase is trained: the oracle probes the ENCODER, so
dynamics, heads, BC and imagination cannot inform this question. The MAE budget
matches the JEPA world budget (20,000 x batch 8) so capacity and data exposure
are comparable. The saved world carries the trained encoder/decoder with the
tokenizer frozen, in the format the oracle driver loads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.cartpole_baseline import sample_cartpole_sequences
from d4_mamba_jepa.checkpoint import save_checkpoint, save_tokenizer_checkpoint
from d4_mamba_jepa.config import D4LiteConfig
from d4_mamba_jepa.craftax_run import SPLIT_SEED, _fixed_dev_batches
from d4_mamba_jepa.data import load_episode_replay, subset_replay, whole_episode_splits
from d4_mamba_jepa.model import D4LiteWorld, build_tokenizer
from d4_mamba_jepa.training import (
    WorldLossNormalizer,
    tokenizer_full_reconstruction_mse,
    tokenizer_reconstruction_loss,
)

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
REPLAY_SHA = "7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b"
OUT = REPO_ROOT / "outputs/d4_mamba_jepa/craftax_tbase"
SEED = 20260727


def craftax_base_config() -> D4LiteConfig:
    """T-BASE: identical to the JEPA arm except the representation objective."""
    return D4LiteConfig(
        representation_objective="base", n_actions=17, image_size=64,
        temporal_backend="transformer",
    )


@torch.no_grad()
def _dev_recon_mse(tokenizer, batches, *, patch_size, device) -> float:
    values = [
        float(tokenizer_full_reconstruction_mse(
            tokenizer, batch.observations.to(device), patch_size=patch_size))
        for batch in batches
    ]
    return float(np.mean(values))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    arm_dir = OUT / "t_base"
    arm_dir.mkdir(parents=True, exist_ok=True)

    cfg = craftax_base_config()
    replay = load_episode_replay(REPLAY, expected_sha256=REPLAY_SHA)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])
    dev_batches = _fixed_dev_batches(
        dev_replay, cfg=cfg, count=16, batch_size=8, seed=SPLIT_SEED + 1
    )

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    tokenizer = build_tokenizer(cfg, training_mask=True).to(device).train()
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=args.learning_rate, weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    rng = np.random.default_rng(SEED + 2)
    before = _dev_recon_mse(tokenizer, dev_batches[:4],
                            patch_size=cfg.patch_size, device=device)
    print(f"dev full-reconstruction MSE before: {before:.6f}", flush=True)

    history: list[float] = []
    started = time.perf_counter()
    for step in range(args.tokenizer_steps):
        batch = sample_cartpole_sequences(
            train_replay, batch_size=args.batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=cfg.jepa_terminal_fraction, device=device, rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, _ = tokenizer_reconstruction_loss(
                tokenizer, batch.observations, patch_size=cfg.patch_size)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite tokenizer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        if step < 250:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * float(step + 1) / 250.0
        optimizer.step()
        history.append(float(loss.detach().item()))
        if (step + 1) % 500 == 0:
            print(f"tokenizer {step + 1}/{args.tokenizer_steps}: "
                  f"mae={np.mean(history[-500:]):.6f} "
                  f"({(time.perf_counter() - started) / (step + 1):.3f}s/update)",
                  flush=True)
    seconds = time.perf_counter() - started
    after = _dev_recon_mse(tokenizer, dev_batches[:4],
                           patch_size=cfg.patch_size, device=device)
    print(f"dev full-reconstruction MSE after: {after:.6f}", flush=True)

    tokenizer_sha = save_tokenizer_checkpoint(
        arm_dir / "tokenizer.pt", tokenizer=tokenizer, config=cfg,
        step=args.tokenizer_steps,
        extra={"format": "craftax_tbase_tokenizer_v1", "seed": SEED},
    )
    # Wrap the trained encoder/decoder in a world so the oracle can load it.
    torch.manual_seed(SEED)
    world = D4LiteWorld(cfg)
    world.encoder.load_state_dict(tokenizer.encoder.state_dict(), strict=True)
    world.decoder.load_state_dict(tokenizer.decoder.state_dict(), strict=True)
    world.freeze_tokenizer()
    world = world.to(device)
    world_sha = save_checkpoint(
        arm_dir / "world.pt", world=world, normalizer=WorldLossNormalizer().to(device),
        optimizer=optimizer, step=args.tokenizer_steps,
        extra={"format": "craftax_tbase_world_v1", "seed": SEED,
               "tokenizer_sha256": tokenizer_sha,
               "note": "tokenizer-only training; dynamics untrained. "
                       "Encoder diagnostic control for the oracle."},
    )
    report = {
        "claim_boundary": "reconstruction ENCODER control for the oracle; "
                          "dynamics are untrained and no policy exists",
        "config": cfg.to_dict(),
        "tokenizer_steps": args.tokenizer_steps,
        "batch_size": args.batch_size,
        "seconds": seconds,
        "dev_recon_mse_before": before,
        "dev_recon_mse_after": after,
        "final_mae_loss": float(np.mean(history[-500:])),
        "tokenizer_sha256": tokenizer_sha,
        "world_sha256": world_sha,
    }
    (arm_dir / "tbase_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
