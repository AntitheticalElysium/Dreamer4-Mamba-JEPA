"""Phase B long run: full compact objective at a real budget, one mask setting.

V2 FAIL-CLOSED SCREENING CRITERIA (defined during the 2026-07-13 re-audit):
  P1  observation-sensitive stream rank, pooled rank, fixed-stream variance,
      observation-variance fraction, and same-stream unrelated distance remain
      above explicit relative untrained thresholds (flat rank is diagnostic);
  P2  improvement_over_copy on changed tokens > 0 at the final evaluation;
  P3  semantic probe sane (converged probe >= majority) and trained accuracy
      >= untrained accuracy - 0.02;
  P4  trained inventory R2 (varying keys) >= untrained inventory R2 - 0.02.

Protocol: 4000 updates, batch 4, T=16, GRU backend, deterministic predictor,
LossConfig defaults (VICReg terms on). Train data: random-policy episodes,
seeds 0+1; held-out: seed 2. Shared data cache so both arms see identical data.
Saves model checkpoint for Phase D reuse.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "m3_hjwm_compact"))
sys.path.insert(0, str(ROOT / "m3_hjwm_compact" / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig, cosine_distance  # noqa: E402
from representation_control import (  # noqa: E402
    changed_patch_mask, collect, inventory_probe, patch_change_scores,
    semantic_probe, target_statistics,
)

SCRATCH = Path(__file__).parent
ARTIFACTS = Path(__file__).resolve().parent


def chw(obs):
    return np.ascontiguousarray(obs.transpose(2, 0, 1))


def collect_episodes(seed: int, episodes: int, max_len: int = 200):
    import crafter

    env = crafter.Env(seed=seed, length=max_len)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        obs = env.reset()
        frames, actions, rewards, continues = [chw(obs)], [], [], []
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, info = env.step(action)
            frames.append(chw(obs))
            actions.append(action)
            rewards.append(float(reward))
            continues.append(float(info.get("discount", float(not done))))
        out.append(dict(
            obs=np.stack(frames).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            continues=np.asarray(continues, dtype=np.float32),
        ))
    return out


def load_shared_data():
    cache = SCRATCH / "phase_data.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    data = {
        "train_episodes": collect_episodes(0, 24) + collect_episodes(1, 24),
        "heldout_episodes": collect_episodes(2, 10),
        "heldout_probe": collect(2, 400),
    }
    torch.save(data, cache)
    return data


@torch.no_grad()
def encode_target(model, frames, device, batch=64):
    toks = []
    for start in range(0, len(frames), batch):
        obs = torch.from_numpy(frames[start:start + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            toks.append(model.target_encoder(obs).float().cpu())
    return torch.cat(toks)


@torch.no_grad()
def heldout_prediction_eval(
    model, replay, device, batches=16, rng_seed=303, mask_seed=404
):
    """One-step prediction through the model's own sequence path on held-out
    windows: predictor output vs target tokens, against the copy baseline."""
    preds, tgts, raw_motion = [], [], []
    rng = np.random.default_rng(rng_seed)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    # Replay windows and random masks are both matched across evaluations.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(mask_seed)
        for _ in range(batches):
            batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            b, t = batch["obs"].shape[:2]
            s, d = model.streams, model.cfg.token_dim
            preds.append(
                output.prediction.selected.float().reshape(b, t - 1, s, d).cpu()
            )
            tgts.append(output.targets.float().cpu())
            raw_motion.append(
                patch_change_scores(
                    batch["obs"][:, :-1].cpu(),
                    batch["obs"][:, 1:].cpu(),
                    model.cfg.patch_size,
                )
            )
    pred = torch.cat(preds)                      # [N, T-1, S, D]
    target = torch.cat(tgts)                     # [N, T, S, D]
    d_pred = cosine_distance(pred, target[:, 1:])
    d_copy = cosine_distance(target[:, :-1], target[:, 1:])
    ruler = float(cosine_distance(target[:, :-1], target[:, 1:].roll(3, 0)).mean())
    local_pred = d_pred[..., model.cfg.registers :]
    local_copy = d_copy[..., model.cfg.registers :]
    changed = changed_patch_mask(torch.cat(raw_motion))
    return {
        "pred_cosine": float(d_pred.mean()),
        "copy_cosine": float(d_copy.mean()),
        "unrelated_pair_cosine": ruler,
        "pred_cosine_changed": float(local_pred[changed].mean()),
        "copy_cosine_changed": float(local_copy[changed].mean()),
        "improvement_over_copy_changed": float(
            local_copy[changed].mean() - local_pred[changed].mean()
        ),
        "changed_patch_fraction": float(changed.float().mean()),
        "changed_patch_source": "raw_rgb_mean_absolute_change_top_positive_quartile",
    }


def probes_block(model, probe_data, cfg, device):
    tokens = encode_target(model, probe_data.obs, device)
    grid = cfg.image_size // cfg.patch_size
    return {
        **target_statistics(tokens, cfg.registers),
        **semantic_probe(tokens, probe_data.semantic, probe_data.player_pos,
                         cfg.registers, grid, device),
        **inventory_probe(tokens, probe_data.inventory, cfg.registers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    args = parser.parse_args()
    device = torch.device("cuda")

    data = load_shared_data()
    replay = EpisodeReplay()
    for ep in data["train_episodes"]:
        replay.add(Episode(**ep))
    heldout_replay = EpisodeReplay()
    for ep in data["heldout_episodes"]:
        heldout_replay.add(Episode(**ep))
    probe_data = data["heldout_probe"]

    torch.manual_seed(101)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic",
                      mask_ratio=args.mask_ratio)
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weights = LossConfig(
        variance=args.variance_weight,
        covariance=args.covariance_weight,
        rollout=args.rollout_weight,
    )

    untrained = {
        "probes": probes_block(model, probe_data, cfg, device),
        "prediction": heldout_prediction_eval(model, heldout_replay, device),
    }
    baseline = target_statistics(
        encode_target(model, probe_data.obs[:300], device), cfg.registers
    )
    baseline.pop("covariance_eigenvalues_desc", None)
    untrained["curve_subset_statistics"] = baseline

    rank_curve, losses = [], {
        "jepa": [], "variance": [], "reward": [], "rollout": []
    }
    train_rng = np.random.default_rng(101)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        batch = replay.sample(
            batch=4, observations=16, device=device, rng=train_rng
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        model.mark_parameters_updated()
        model.update_target()
        for key in losses:
            losses[key].append(float(output.metrics[key]))
        if (step + 1) % 250 == 0 or step == 0 or step + 1 == args.steps:
            tokens = encode_target(model, probe_data.obs[:300], device)
            stats = target_statistics(tokens, cfg.registers)
            point = {
                "step": step + 1,
                "flat_rank": stats["target_flat_effective_rank_covariance"],
                "stream_rank": stats["target_stream_effective_rank_mean"],
                "patch_pool_rank": stats["target_patch_pool_covariance_rank"],
                "fixed_stream_variance": stats["target_fixed_stream_variance"],
                "observation_variance_fraction": stats[
                    "target_observation_variance_fraction"
                ],
                "same_stream_unrelated_cosine": stats[
                    "target_same_stream_unrelated_cosine"
                ],
            }
            rank_curve.append(point)
            print(f"[{args.tag}] step {step+1} flat-rank {point['flat_rank']:.2f} "
                  f"stream-rank {point['stream_rank']:.2f} "
                  f"jepa {np.mean(losses['jepa'][-100:]):.4f}", flush=True)
    elapsed = time.perf_counter() - started

    final = {
        "probes": probes_block(model, probe_data, cfg, device),
        "prediction": heldout_prediction_eval(model, heldout_replay, device),
    }
    for block in (untrained["probes"], final["probes"]):
        block.pop("covariance_eigenvalues_desc", None)

    warmup_checks = [
        point["stream_rank"] >= 0.5 * baseline["target_stream_effective_rank_mean"]
        and point["patch_pool_rank"] >= 0.5 * baseline["target_patch_pool_covariance_rank"]
        and point["fixed_stream_variance"] >= 0.5 * baseline["target_fixed_stream_variance"]
        and point["observation_variance_fraction"]
        >= 0.5 * baseline["target_observation_variance_fraction"]
        and point["same_stream_unrelated_cosine"]
        >= 0.5 * baseline["target_same_stream_unrelated_cosine"]
        for point in rank_curve
    ]
    final_point = rank_curve[-1]
    final_checks = (
        final_point["stream_rank"]
        >= baseline["target_stream_effective_rank_mean"] - 0.5
        and final_point["patch_pool_rank"]
        >= baseline["target_patch_pool_covariance_rank"] - 0.5
        and final_point["fixed_stream_variance"]
        >= 0.8 * baseline["target_fixed_stream_variance"]
        and final_point["observation_variance_fraction"]
        >= 0.8 * baseline["target_observation_variance_fraction"]
        and final_point["same_stream_unrelated_cosine"]
        >= 0.8 * baseline["target_same_stream_unrelated_cosine"]
    )
    criteria = {
        "P1_no_observation_collapse": bool(all(warmup_checks) and final_checks),
        "P2_positive_improvement_over_copy_changed": bool(
            final["prediction"]["improvement_over_copy_changed"] > 0
        ),
        "P3_semantic_probe_sane_and_not_degraded": bool(
            final["probes"]["semantic_probe_sane"]
            and final["probes"]["semantic_token_accuracy"]
            >= untrained["probes"]["semantic_token_accuracy"] - 0.02
        ),
        "P4_inventory_r2_not_degraded": bool(
            (final["probes"]["inventory_r2_mean_varying"] or -1)
            >= (untrained["probes"]["inventory_r2_mean_varying"] or -1) - 0.02
        ),
    }

    ckpt = SCRATCH / f"phase_b_v2_{args.tag}.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, ckpt)
    report = {
        "tag": args.tag,
        "mask_ratio": args.mask_ratio,
        "rollout_weight": args.rollout_weight,
        "variance_weight": args.variance_weight,
        "covariance_weight": args.covariance_weight,
        "protocol_version": 2,
        "train_rng_seed": 101,
        "evaluation_rng_seed": 303,
        "evaluation_mask_seed": 404,
        "steps": args.steps,
        "train_transitions": replay.steps,
        "elapsed_minutes": round(elapsed / 60, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "loss_first_last_100": {
            key: [float(np.mean(vals[:100])), float(np.mean(vals[-100:]))]
            for key, vals in losses.items()
        },
        "rank_curve": rank_curve,
        "untrained": untrained,
        "final": final,
        "criteria": criteria,
        "checkpoint": str(ckpt),
    }
    out = ARTIFACTS / f"phase_b_v2_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"criteria": criteria, "final_prediction": final["prediction"]}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
