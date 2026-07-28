"""When does the encoder discard task state -- at init, early, or gradually?

The random-encoder floor showed an UNTRAINED encoder preserves food/wood/sapling
far better (0.63/0.63/0.66) than the trained one (0.17/0.10/0.04) at identical
geometry, while health moves the other way (0.47 -> 0.87). So training removes
this information rather than failing to acquire it.

This trains ONE world at the baseline geometry and runs the identical oracle at
a ladder of checkpoints. Shape of the curve distinguishes:

  * gone within a few hundred steps -> an early optimization transient, and the
    thing to inspect is initialization/normalization, not the objective's
    long-run pressure
  * monotone decay over 20k -> the objective grinding it away, and the loss is
    the thing to change
  * flat-then-drop -> something schedule-linked (EMA tau ramp, warmup end)

No architectural change, no new data; one training run plus read-only probes.
Every parameter is the baseline's.
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
from d4_mamba_jepa.craftax_oracle import load_probe_data, representation_oracle
from d4_mamba_jepa.craftax_run import SPLIT_SEED, _dev_cosine, _fixed_dev_batches
from d4_mamba_jepa.craftax_runners import craftax_jepa_config
from d4_mamba_jepa.data import load_episode_replay, subset_replay, whole_episode_splits
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.training import WorldLossNormalizer, world_loss

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
REPLAY_SHA = "7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b"
PROBE = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt"
SEED = 20260727
TRACKED = ["health", "food", "drink", "energy", "wood", "stone", "sapling",
           "wood_sword", "iron_sword", "diamond"]


def _summary(report: dict) -> dict:
    out = {}
    for group in ("vitals", "inventory"):
        for name, entry in report[group]["per_target"].items():
            out[name] = {"verdict": entry["verdict"],
                         "latent_linear_r2": entry["latent_linear_r2"],
                         "latent_nonlinear_r2": entry["latent_nonlinear_r2"]}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-steps", type=int, default=20_000)
    p.add_argument("--ladder", default="0,250,500,1000,2500,5000,10000,20000")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "reviews/artifacts/craftax_timecourse.json")
    p.add_argument("--jepa-weight", type=float, default=None,
                   help="diagnostic: 0.0 removes the self-prediction term so the "
                        "encoder is trained ONLY by the reward/continuation heads")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ladder = sorted({int(x) for x in args.ladder.split(",")})

    probe = load_probe_data(PROBE)
    replay = load_episode_replay(REPLAY, expected_sha256=REPLAY_SHA)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])

    cfg = craftax_jepa_config("transformer")
    if args.jepa_weight is not None:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, jepa_weight=args.jepa_weight)
        print(f"DIAGNOSTIC: jepa_weight={cfg.jepa_weight}", flush=True)
    dev_batches = _fixed_dev_batches(dev_replay, cfg=cfg, count=16, batch_size=8,
                                     seed=SPLIT_SEED + 1)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    world = D4LiteWorld(cfg).to(device).train()
    normalizer = WorldLossNormalizer().to(device)
    trainable = [q for q in world.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    rng = np.random.default_rng(SEED + 3)

    curve = {}

    def probe_now(step: int) -> None:
        report = representation_oracle(world, probe)
        cosine = _dev_cosine(world, dev_batches, device)
        curve[step] = {"dev_cosine": cosine, "targets": _summary(report),
                       "audit_pass": report["audit"]["pass"]}
        tracked = " ".join(
            f"{n}={curve[step]['targets'][n]['latent_linear_r2']:+.2f}"
            for n in TRACKED if n in curve[step]["targets"])
        print(f"[step {step:>6}] dev_cos={cosine:+.4f} {tracked}", flush=True)
        args.output.write_text(json.dumps(curve, indent=2, sort_keys=True,
                                          default=float) + "\n")

    started = time.perf_counter()
    if 0 in ladder:
        probe_now(0)
    for step in range(args.world_steps):
        batch = sample_cartpole_sequences(
            train_replay, batch_size=args.batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=cfg.jepa_terminal_fraction, device=device, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, _ = world_loss(world, batch, normalizer=normalizer)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if step < 1_000:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * float(step + 1) / 1_000.0
        optimizer.step()
        if cfg.jepa_anticollapse == "ema":
            frac = step / max(1, args.world_steps - 1)
            tau = cfg.jepa_ema_tau + (cfg.jepa_ema_tau_final - cfg.jepa_ema_tau) * frac
            world.update_jepa_target(tau)
        if (step + 1) in ladder:
            world.eval()
            probe_now(step + 1)
            world.train()
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
