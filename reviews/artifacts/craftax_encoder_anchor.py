"""Encoder-anchor ablation: is unanchored full-LR encoder training the cause?

Every pinned reference anchors the encoder in some way while the predictor /
dynamics learn:
  * V-JEPA 2-AC   -- pretrained encoder, absent from the loss graph
  * Dreamer 4     -- pretrained tokenizer, frozen during dynamics training
  * Dreamer-CDP   -- enc_lr 6e-6 vs dyn_lr 4e-4, a 66.7x timescale separation
  * SPR           -- SPR loss is auxiliary to Q-learning and reward losses
Our arm moves a randomly initialized encoder at the full 1e-4 alongside
everything else. This ablates exactly that axis.

Fixes two defects in the earlier 2,500-update ablations:

  1. EMA SCHEDULE CONFOUND. Those runs ramped tau 0.99 -> 0.999 over
     `world_steps`, so a 2,500-step run reached tau=0.999 by its end while the
     real 20,000-step baseline was only at ~0.9911 at update 2,500. They were
     therefore not the first 2,500 updates of the run that failed. `--ema-steps`
     pins the ramp denominator (default 20,000) independently of the budget.
  2. REPORTING LOSS. The time-course kept only per-target point estimates.
     This writes the FULL oracle report (all 12 inventory + 4 vitals targets,
     linear and nonlinear, bootstrap CIs, pixel ceilings, floors, achievements,
     verdicts) at every probe, plus a final checkpoint, so paired uncertainty
     can be reconstructed later.

Also pins the probe payload by hash, which the time-course runner did not.
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
from d4_mamba_jepa.checkpoint import file_sha256, save_checkpoint
from d4_mamba_jepa.craftax_oracle import (
    INVENTORY_NAMES, VITAL_NAMES, load_probe_data, representation_oracle,
)
from d4_mamba_jepa.craftax_run import SPLIT_SEED, _dev_cosine, _fixed_dev_batches
from d4_mamba_jepa.craftax_runners import craftax_jepa_config
from d4_mamba_jepa.data import load_episode_replay, subset_replay, whole_episode_splits
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.training import LossWeights, WorldLossNormalizer, world_loss

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
REPLAY_SHA = "7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b"
PROBE = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt"
PROBE_SHA = "bb5c7c703c0125131dcdb56cb24660ad22febf18c236cd6cf5336b8f748d1fdb"


def _means(report: dict) -> dict:
    """All 12 inventory and 4 vitals targets, linear AND nonlinear."""
    out = {}
    for group, names in (("vitals", VITAL_NAMES), ("inventory", INVENTORY_NAMES)):
        per = report[group]["per_target"]
        for kind in ("linear", "nonlinear"):
            key = f"latent_{kind}_r2"
            out[f"{group}_{kind}_mean"] = float(
                np.mean([per[n][key] for n in names]))
    verdicts: dict[str, int] = {}
    for group in ("vitals", "inventory"):
        for entry in report[group]["per_target"].values():
            verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    out["verdicts"] = verdicts
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-lr", type=float, default=1e-4,
                   help="0.0 freezes the encoder entirely")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--world-steps", type=int, default=2500)
    p.add_argument("--ema-steps", type=int, default=20_000,
                   help="EMA tau ramp denominator, decoupled from the budget")
    p.add_argument("--ladder", default="0,1000,2500")
    p.add_argument("--terminal-fraction", type=float, default=0.0)
    p.add_argument("--jepa-weight", type=float, default=1.0)
    p.add_argument("--reward-weight", type=float, default=0.0)
    p.add_argument("--continuation-weight", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "reviews/artifacts/encoder_anchor")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ladder = sorted({int(x) for x in args.ladder.split(",")})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual_probe_sha = file_sha256(PROBE)
    if actual_probe_sha != PROBE_SHA:
        raise RuntimeError(f"probe digest drift: {actual_probe_sha} != {PROBE_SHA}")
    probe = load_probe_data(PROBE)

    replay = load_episode_replay(REPLAY, expected_sha256=REPLAY_SHA)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])

    cfg = craftax_jepa_config("transformer")
    dev_batches = _fixed_dev_batches(dev_replay, cfg=cfg, count=16, batch_size=8,
                                     seed=SPLIT_SEED + 1)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    world = D4LiteWorld(cfg).to(device).train()
    normalizer = WorldLossNormalizer().to(device)

    frozen = args.encoder_lr == 0.0
    encoder_params = list(world.encoder.parameters())
    encoder_ids = {id(q) for q in encoder_params}
    if frozen:
        for q in encoder_params:
            q.requires_grad_(False)
    other = [q for q in world.parameters()
             if q.requires_grad and id(q) not in encoder_ids]
    groups = [{"params": other, "lr": args.learning_rate,
               "base_lr": args.learning_rate}]
    if not frozen:
        groups.append({"params": encoder_params, "lr": args.encoder_lr,
                       "base_lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=1e-2)
    trainable = [q for g in groups for q in g["params"]]
    weights = LossWeights(jepa=args.jepa_weight, reward=args.reward_weight,
                          continuation=args.continuation_weight)
    print(f"tag={args.tag} seed={args.seed} encoder_lr="
          f"{'FROZEN' if frozen else args.encoder_lr} lr={args.learning_rate} "
          f"ema_steps={args.ema_steps} tf={args.terminal_fraction} "
          f"weights=({weights.jepa},{weights.reward},{weights.continuation})",
          flush=True)

    rng = np.random.default_rng(args.seed + 3)
    curve: dict[str, dict] = {}
    out_json = args.output_dir / f"{args.tag}.json"

    def probe_now(step: int) -> None:
        was = world.training
        world.eval()
        report = representation_oracle(world, probe)
        cosine = _dev_cosine(world, dev_batches, device)
        summary = _means(report)
        curve[str(step)] = {"dev_cosine": cosine, "summary": summary,
                            "full_report": report}
        print(f"[{args.tag}] step {step:>5} dev_cos={cosine:+.4f} "
              f"inv_lin={summary['inventory_linear_mean']:.3f} "
              f"inv_non={summary['inventory_nonlinear_mean']:.3f} "
              f"vit_lin={summary['vitals_linear_mean']:.3f} "
              f"{summary['verdicts']}", flush=True)
        out_json.write_text(json.dumps(
            {"config": vars(args) | {"probe_sha256": PROBE_SHA,
                                     "replay_sha256": REPLAY_SHA},
             "curve": curve}, indent=2, sort_keys=True, default=str) + "\n")
        world.train(was)

    started = time.perf_counter()
    if 0 in ladder:
        probe_now(0)
    for step in range(args.world_steps):
        batch = sample_cartpole_sequences(
            train_replay, batch_size=args.batch_size,
            sequence_length=cfg.sequence_length,
            terminal_fraction=args.terminal_fraction, device=device, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, _ = world_loss(world, batch, normalizer=normalizer,
                                 weights=weights)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if step < 1_000:
            scale = float(step + 1) / 1_000.0
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * scale
        else:
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"]
        optimizer.step()
        if cfg.jepa_anticollapse == "ema":
            # Ramp pinned to `ema_steps`, NOT the budget, so a short run is a
            # genuine prefix of the long one.
            frac = min(1.0, step / max(1, args.ema_steps - 1))
            tau = cfg.jepa_ema_tau + (cfg.jepa_ema_tau_final - cfg.jepa_ema_tau) * frac
            world.update_jepa_target(tau)
        if (step + 1) in ladder:
            probe_now(step + 1)
    save_checkpoint(
        args.output_dir / f"{args.tag}.pt", world=world, normalizer=normalizer,
        optimizer=optimizer, numpy_rng=rng, step=args.world_steps,
        extra={"format": "craftax_encoder_anchor_v1", "tag": args.tag,
               "seed": args.seed, "encoder_lr": args.encoder_lr,
               "ema_steps": args.ema_steps},
    )
    print(f"[{args.tag}] done in {(time.perf_counter() - started)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
