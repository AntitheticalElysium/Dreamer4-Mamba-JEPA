"""Causal-action probe v2, Stage A (free measurements on existing artifacts).

Protocol: reviews/2026-07-13-microtest-protocol.md, amendment "causal-action
probe v2". Discriminates H1 (injection), H2 (topology/transport), H3 (horizon)
using the archived fork bundle and the saved S3-v2 / microtest checkpoints.
No training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import ModelConfig, cosine_distance  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import build_frozen_world  # noqa: E402
from microtest import openloop_anchor  # noqa: E402
from fork_oracle_v2 import (  # noqa: E402
    BUNDLE, ENCODER_CKPT, MOVE_DELTA, PREFIX, SUFFIX, encode, sha256_file,
)

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")
NOOP_SUFFIX = [0] * SUFFIX


def cluster_ci(values, clusters, seed=17, draws=2000):
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    means = [
        np.concatenate([values[clusters == c]
                        for c in rng.choice(unique, len(unique), True)]).mean()
        for _ in range(draws)
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def load_checkpoint_world(path, device):
    world = build_frozen_world(device)
    ckpt = torch.load(path, weights_only=False)
    state = dict(world.state_dict())
    state.update(ckpt["trainable"])
    world.load_state_dict(state)
    world.eval()
    return world


@torch.no_grad()
def suffix_targets(encoder, anchor, device):
    """Branch-mean unit-normalized target latents per suffix per k. [4, K, S, D]"""
    per_suffix = []
    for name in SUFFIX_NAMES:
        frames = anchor["branches"][name]["frames"]          # [B, K, C, H, W]
        toks = encode(encoder, frames, device)               # [B, K, S, D]
        per_suffix.append(F.normalize(toks.float(), dim=-1).mean(0))
    return torch.stack(per_suffix)


@torch.no_grad()
def stage_a_model(world, encoder, anchors, device):
    rows = []
    for i, anchor in enumerate(anchors):
        targets = suffix_targets(encoder, anchor, device)     # [4, K, S, D]
        preds = torch.stack([
            openloop_anchor(world, anchor, anchor["suffixes"][name], device)
            for name in SUFFIX_NAMES
        ])                                                    # [4, K, S, D]
        noop_pred = openloop_anchor(world, anchor, NOOP_SUFFIX, device)

        k = SUFFIX - 1
        # 4-way retrieval: pred under suffix s vs all 4 suffix targets
        dist = torch.stack([
            torch.stack([
                cosine_distance(preds[s, k], targets[t, k]).mean()
                for t in range(4)
            ]) for s in range(4)
        ])                                                    # [s, t]
        retrieval = float((dist.argmin(1) == torch.arange(4)).float().mean())
        # matched separation: wrong-suffix preds vs true target, minus matched
        matched = float(dist[0, 0])
        wrong = float(dist[1:, 0].mean())
        # no-action control at true target
        noop_err = float(cosine_distance(noop_pred[k], targets[0, k]).mean())

        # layerwise divergence transmission (suffix pairs at each k).
        # Stage-B correction: pairs with near-zero TRUE divergence (ineffective
        # first actions) explode the ratio; report None for those, aggregate
        # with medians over valid pairs.
        transmission, alignment = [], []
        for kk in range(SUFFIX):
            pd = preds[0, kk] - preds[1, kk]
            td = targets[0, kk] - targets[1, kk]
            denom = float(td.norm())
            if denom < 1e-3:
                transmission.append(None)
                alignment.append(None)
                continue
            transmission.append(float(pd.norm() / denom))
            alignment.append(float(F.cosine_similarity(
                pd.flatten(), td.flatten(), dim=0)))
        rows.append({
            "env_seed": anchor["env_seed"], "anchor": i,
            "retrieval_4way": retrieval,
            "matched_separation": wrong - matched,
            "noop_minus_true": noop_err - matched,
            "transmission_k": transmission,
            "alignment_k": alignment,
        })
    seeds = [r["env_seed"] for r in rows]
    summary = {
        "retrieval_4way_mean": float(np.mean([r["retrieval_4way"] for r in rows])),
        "retrieval_4way_ci": cluster_ci([r["retrieval_4way"] for r in rows], seeds),
        "matched_separation_mean": float(np.mean([r["matched_separation"] for r in rows])),
        "matched_separation_ci": cluster_ci([r["matched_separation"] for r in rows], seeds),
        "noop_minus_true_mean": float(np.mean([r["noop_minus_true"] for r in rows])),
        "transmission_by_k_median": [
            (lambda vals: float(np.median(vals)) if vals else None)(
                [r["transmission_k"][kk] for r in rows
                 if r["transmission_k"][kk] is not None])
            for kk in range(SUFFIX)
        ],
        "alignment_by_k_median": [
            (lambda vals: float(np.median(vals)) if vals else None)(
                [r["alignment_k"][kk] for r in rows
                 if r["alignment_k"][kk] is not None])
            for kk in range(SUFFIX)
        ],
    }
    return summary, rows


@torch.no_grad()
def transport_diagnostic(encoder, anchors, device):
    """Frozen-representation property: content transport under movement at k=1."""
    grid = 8
    pos = encoder.model.pos_embed[0].cpu()                    # [64, D]
    same_pos, transported = [], []
    for anchor in anchors:
        action = anchor["suffixes"]["true"][0]
        if action not in MOVE_DELTA:
            continue
        dx, dy = MOVE_DELTA[action]
        true = anchor["branches"]["true"]
        moved = (true["positions"][:, 0] != anchor["player_pos"][None]).any(-1)
        if not moved.all():
            continue
        anchor_tok = encode(encoder, anchor["obs_hist"][-1][None], device)[0].cpu()
        next_tok = encode(encoder, true["frames"][:, 0], device).mean(0).cpu()
        regs = 2
        anchor_content = anchor_tok[regs:] - pos
        next_content = next_tok[regs:] - pos
        for y in range(grid):
            for x in range(grid):
                sx, sy = x + dx, y + dy                       # source cell
                if not (0 <= sx < grid and 0 <= sy < grid):
                    continue
                dst = y * grid + x
                src = sy * grid + sx
                transported.append(float(F.cosine_similarity(
                    next_content[dst], anchor_content[src], dim=0)))
                same_pos.append(float(F.cosine_similarity(
                    next_content[dst], anchor_content[dst], dim=0)))
    return {
        "n_tokens": len(same_pos),
        "same_position_cos": float(np.mean(same_pos)) if same_pos else None,
        "transported_cos": float(np.mean(transported)) if transported else None,
    }


def main():
    device = torch.device("cuda")
    anchors = torch.load(BUNDLE, weights_only=False)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    report = {
        "protocol": "causal-action probe v2, Stage A",
        "bundle_sha256": sha256_file(BUNDLE),
        "definitions": {
            "held_out_action_use":
                "4-way retrieval cluster-CI lower bound > 0.25 AND matched separation CI > 0",
        },
        "models": {},
    }
    checkpoints = {
        "s3_40k_seed101": ARTIFACTS / "step3_v2_40k_s101_r1.pt",
        "s3_40k_seed202": ARTIFACTS / "step3_v2_40k_s202_r1.pt",
        "s3_40k_seed303": ARTIFACTS / "step3_v2_40k_s303_r1.pt",
        "microtest": ARTIFACTS / "microtest_v1_model.pt",
    }
    for name, path in checkpoints.items():
        world = load_checkpoint_world(path, device)
        summary, rows = stage_a_model(world, encoder, anchors, device)
        report["models"][name] = summary
        (ARTIFACTS / f"causal_stage_a_rows_{name}.json").write_text(json.dumps(rows))
        print(f"[stage A] {name}: retrieval {summary['retrieval_4way_mean']:.3f} "
              f"CI {summary['retrieval_4way_ci']} sep {summary['matched_separation_mean']:+.5f}",
              flush=True)
        del world
        torch.cuda.empty_cache()

    report["transport_diagnostic"] = transport_diagnostic(encoder, anchors, device)
    out = ARTIFACTS / "causal_stage_a.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
