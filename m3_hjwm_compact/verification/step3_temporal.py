"""Matrix step 3: frozen step-1 encoder; temporal predictor with rollout 0 vs 1.

Protocol and pre-registered gates S3-A/B/C: reviews/2026-07-13-step3-protocol.md
(committed before this file). Evaluation reuses the audited `openloop_eval`
from reviews/artifacts/phase_d_backend.py verbatim.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))
sys.path.insert(0, str(REPO_ROOT / "reviews" / "artifacts"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from ssl_step1 import load_shared_data  # noqa: E402
from phase_d_backend import openloop_eval  # noqa: E402  (audited instrument)

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
ENCODER_CKPT = ARTIFACTS / "ssl_step1_lejepa_global_g1000.pt"
K_PRIMARY = 8
SEEDS = (101, 202, 303)


def build_frozen_world(device) -> M3HJWM:
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    state = torch.load(ENCODER_CKPT, weights_only=False)
    pretrainer.load_state_dict(state["pretrainer"], strict=True)

    world = M3HJWM(cfg).to(device)
    # Step-1 winner's EMA/target weights into BOTH world encoders, then freeze.
    world.online_encoder.load_state_dict(pretrainer.target_encoder.model.state_dict())
    world.target_encoder.model.load_state_dict(pretrainer.target_encoder.model.state_dict())
    for parameter in world.online_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in world.target_encoder.parameters():
        parameter.requires_grad_(False)
    return world


def run_arm(seed: int, rollout_weight: float, data, device, steps: int):
    replay = EpisodeReplay()
    for ep in data["train_episodes"]:
        replay.add(Episode(**ep))
    torch.manual_seed(seed)
    world = build_frozen_world(device)
    weights = dataclasses.replace(
        LossConfig(), variance=0.0, covariance=0.0, rollout=rollout_weight
    )
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    replay_rng = np.random.default_rng(seed)  # identical schedule across the pair
    metrics_hist = {"jepa": [], "reward": [], "rollout": []}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        batch = replay.sample(batch=4, observations=16, device=device, rng=replay_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = world(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        # encoders frozen: no EMA update
        for key in metrics_hist:
            if key in output.metrics:
                metrics_hist[key].append(float(output.metrics[key]))
    minutes = round((time.perf_counter() - started) / 60, 2)

    per_k, extras = openloop_eval(world, data["heldout_episodes"], device)
    result = {
        "seed": seed,
        "rollout_weight": rollout_weight,
        "train_minutes": minutes,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "metrics_first_last_100": {
            k: [float(np.mean(v[:100])), float(np.mean(v[-100:]))]
            for k, v in metrics_hist.items() if v
        },
        "openloop": per_k,
        **({"eval_extras": extras} if isinstance(extras, dict) else {}),
    }
    del world
    torch.cuda.empty_cache()
    return result


def gate_summary(results: list[dict]) -> dict:
    by = {(r["seed"], r["rollout_weight"]): r for r in results}

    def at_k(r, k):
        return next(p for p in r["openloop"] if p["k"] == k)

    s3a_votes, s3b_votes, s3c_votes = [], [], []
    for seed in SEEDS:
        roll = at_k(by[(seed, 1.0)], K_PRIMARY)
        base = at_k(by[(seed, 0.0)], K_PRIMARY)
        lower = roll["paired_window_margin_bootstrap_95"][0]
        s3a_votes.append(bool(lower > 0 and roll["relative_window_margin"] >= 0.05))
        s3b_votes.append(bool(
            roll["paired_window_margin_mean"] > base["paired_window_margin_mean"]
        ))
        one_roll = at_k(by[(seed, 1.0)], 1)["pred_cosine_changed"]
        one_base = at_k(by[(seed, 0.0)], 1)["pred_cosine_changed"]
        s3c_votes.append(bool(one_roll <= 1.2 * one_base))
    return {
        "S3A_votes": s3a_votes, "S3A_pass": sum(s3a_votes) >= 2,
        "S3B_votes": s3b_votes, "S3B_pass": sum(s3b_votes) >= 2,
        "S3C_votes": s3c_votes, "S3C_pass": sum(s3c_votes) >= 2,
        "k_primary": K_PRIMARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tag", default="step3")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    device = torch.device("cuda")
    data = load_shared_data()

    results = []
    for seed in args.seeds:
        for rollout_weight in (0.0, 1.0):
            print(f"[step3] seed {seed} rollout {rollout_weight}", flush=True)
            results.append(run_arm(seed, rollout_weight, data, device, args.steps))
    report = {
        "protocol": "reviews/2026-07-13-step3-protocol.md",
        "encoder": str(ENCODER_CKPT.name),
        "steps": args.steps,
        "gates": gate_summary(results) if len(args.seeds) == len(SEEDS) else "partial",
        "results": results,
    }
    out = ARTIFACTS / f"step3_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
