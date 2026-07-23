"""Is the continuation head calibrated on real states?

The imagination actor's whole learning signal is reward x survival, and survival
is the product of this head's outputs. If the head is miscalibrated the actor
optimises a truncated (or unbounded) horizon regardless of how good the latent
is.

Reports, on held-out real states: mean predicted P(continue), the empirical
continuation rate, reliability bins, and the implied imagined episode length
against the true one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d4_mamba_jepa.cartpole_baseline import (
    _clean_agent_tokens,
    load_cartpole_replay,
    sample_cartpole_sequences,
)
from d4_mamba_jepa.checkpoint import file_sha256, load_checkpoint


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=931000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, _, _ = load_checkpoint(
        args.world, device=device, expected_sha256=file_sha256(args.world),
        strict_implementation=False,
    )
    world.eval()
    replay, _ = load_cartpole_replay(args.replay)
    rng = np.random.default_rng(args.seed)

    probs, truth = [], []
    for _ in range(args.batches):
        batch = sample_cartpole_sequences(
            replay, batch_size=args.batch_size,
            sequence_length=world.cfg.sequence_length,
            terminal_fraction=0.0, rng=rng, device=device,
        )
        tokens = _clean_agent_tokens(world, batch)
        heads = world.forward_task_heads(tokens)
        p = heads["continue_logits"][..., 0].float().sigmoid()
        valid = batch.outcome_valid > 0
        probs.append(p[valid].cpu().numpy())
        truth.append(batch.led_to_continues[valid].cpu().numpy())

    p = np.concatenate(probs)
    y = np.concatenate(truth)
    print(f"[{args.label}] arm={world.cfg.arm_id}  n={len(p)}")
    print(f"  mean predicted P(continue) : {p.mean():.4f}")
    print(f"  empirical continuation rate: {y.mean():.4f}")
    print(f"  calibration error (pred-emp): {p.mean() - y.mean():+.4f}")
    print(f"  implied imagined ep length : {1.0 / max(1e-9, 1.0 - p.mean()):.1f} steps")
    print(f"  true mean ep length        : {1.0 / max(1e-9, 1.0 - y.mean()):.1f} steps")
    print("  reliability (pred bin -> empirical):")
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"    [{lo:.1f},{hi:.1f}) n={m.sum():>6}  pred={p[m].mean():.3f}  emp={y[m].mean():.3f}")
    out = {
        "label": args.label, "arm_id": world.cfg.arm_id,
        "mean_predicted_continue": float(p.mean()),
        "empirical_continue_rate": float(y.mean()),
        "calibration_error": float(p.mean() - y.mean()),
        "n": int(len(p)),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
