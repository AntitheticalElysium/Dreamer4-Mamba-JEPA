"""Causal-action probe v2, Stage B (pre-registered training arms + ladder).

Arms (reviews/2026-07-13-microtest-protocol.md, Stage B registration):
  B1 current topology (66 independent streams, GRU), horizon-matched
     rollout_steps=8;
  B2 global shared memory (GlobalGRUTemporal), horizon-matched;
  B3 shuffled-action control on B1 (actions rolled across the batch during
     training) — must stay at chance or the metric is broken.
Ladder 4k -> 8k -> 16k: an arm continues only if 4-way retrieval improves
>= +0.02 or matched separation doubles rung-to-rung.
"""
from __future__ import annotations

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

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import load_scaled_data  # noqa: E402
from openloop_v2 import openloop_eval_v2  # noqa: E402
from causal_stage_a import stage_a_model  # noqa: E402
from fork_oracle_v2 import BUNDLE, ENCODER_CKPT, sha256_file  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
RUNGS = (4000, 4000, 8000)      # cumulative 4k, 8k, 16k
K_PRIMARY = 8


def build_world(backend: str, device) -> M3HJWM:
    cfg = ModelConfig(temporal_backend=backend, predictor="deterministic",
                      mask_ratio=0.0, rollout_steps=8)
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    state = torch.load(ENCODER_CKPT, weights_only=False)
    pretrainer.load_state_dict(state["pretrainer"], strict=True)
    world = M3HJWM(cfg).to(device)
    world.online_encoder.load_state_dict(pretrainer.target_encoder.model.state_dict())
    world.target_encoder.model.load_state_dict(pretrainer.target_encoder.model.state_dict())
    for p in world.online_encoder.parameters():
        p.requires_grad_(False)
    for p in world.target_encoder.parameters():
        p.requires_grad_(False)
    return world


def run_arm(name, backend, seed, shuffle_actions, data, heldout, encoder,
            anchors, device):
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in data:
        replay.add(Episode(**ep))
    torch.manual_seed(seed)
    world = build_world(backend, device)
    weights = dataclasses.replace(
        LossConfig(), variance=0.0, covariance=0.0, rollout=1.0)
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(seed)

    rung_reports, total_steps = [], 0
    prev_retrieval, prev_sep = None, None
    for rung_steps in RUNGS:
        started = time.perf_counter()
        for _ in range(rung_steps):
            batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
            if shuffle_actions:
                batch["actions"] = batch["actions"].roll(1, 0)
                batch["previous_actions"] = batch["previous_actions"].roll(1, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = world(batch, weights)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 100.0)
            optimizer.step()
            world.mark_parameters_updated()
        total_steps += rung_steps
        minutes = round((time.perf_counter() - started) / 60, 1)

        causal, _ = stage_a_model(world, encoder, anchors, device)
        per_k, latency, _ = openloop_eval_v2(world, heldout, device)
        k8 = next(p for p in per_k if p["k"] == K_PRIMARY)
        rung = {
            "cumulative_steps": total_steps,
            "train_minutes": minutes,
            "retrieval_4way": causal["retrieval_4way_mean"],
            "retrieval_ci": causal["retrieval_4way_ci"],
            "matched_separation": causal["matched_separation_mean"],
            "separation_ci": causal["matched_separation_ci"],
            "noop_minus_true": causal["noop_minus_true_mean"],
            "alignment_by_k_median": causal["alignment_by_k_median"],
            "k8_copy_margin": k8["margin_mean"],
            "k8_margin_ci": k8["margin_cluster_bootstrap_95"],
        }
        rung_reports.append(rung)
        print(f"[{name}] {total_steps} steps: retrieval "
              f"{rung['retrieval_4way']:.3f} sep {rung['matched_separation']:+.5f} "
              f"margin {rung['k8_copy_margin']:+.4f}", flush=True)

        improved = (
            prev_retrieval is None
            or rung["retrieval_4way"] >= prev_retrieval + 0.02
            or (prev_sep is not None and prev_sep > 0
                and rung["matched_separation"] >= 2 * prev_sep)
        )
        prev_retrieval, prev_sep = rung["retrieval_4way"], rung["matched_separation"]
        if not improved and total_steps < sum(RUNGS):
            rung_reports.append({"ladder_stopped_after": total_steps})
            break

    torch.save(
        {"trainable": {n: p.detach().cpu() for n, p in world.named_parameters()
                       if p.requires_grad},
         "optimizer": optimizer.state_dict(),
         "numpy_rng": rng.bit_generator.state,
         "torch_rng_cpu": torch.get_rng_state(),
         "backend": backend, "seed": seed, "shuffled": shuffle_actions},
        ARTIFACTS / f"causal_b_{name}.pt")
    del world
    torch.cuda.empty_cache()
    return rung_reports


def main():
    device = torch.device("cuda")
    train, heldout = load_scaled_data()
    anchors = torch.load(BUNDLE, weights_only=False)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    arms = [
        ("B1_gru_s101", "gru", 101, False),
        ("B1_gru_s202", "gru", 202, False),
        ("B2_global_s101", "global_gru", 101, False),
        ("B2_global_s202", "global_gru", 202, False),
        ("B3_shuffled_control", "gru", 101, True),
    ]
    report = {
        "protocol": "causal probe v2, Stage B",
        "bundle_sha256": sha256_file(BUNDLE),
        "rungs_cumulative": [4000, 8000, 16000],
        "arms": {},
    }
    for name, backend, seed, shuffled in arms:
        report["arms"][name] = run_arm(
            name, backend, seed, shuffled, train, heldout, encoder, anchors, device)
        (ARTIFACTS / "causal_stage_b.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["arms"], indent=2)[:3000])
    print("saved", ARTIFACTS / "causal_stage_b.json")


if __name__ == "__main__":
    main()
