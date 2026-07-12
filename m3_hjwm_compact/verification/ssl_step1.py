"""Matrix step 1: faithful same-frame I-JEPA vs incumbent hybrid vs untrained.

Protocol and pre-registered gates G1-G5: reviews/2026-07-13-step1-protocol.md
(committed before the first run). Artifacts: reviews/artifacts/ssl_step1_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from representation_control import (  # noqa: E402
    collect, inventory_probe, semantic_probe, target_statistics,
)

ARTIFACTS = Path(__file__).resolve().parents[2] / "reviews" / "artifacts"
# Durable, gitignored cache; scratchpad/tmp caches proved non-reproducible
# (2026-07-13 re-audit, moderate finding 13).
DATA_CACHE = Path(__file__).resolve().parents[2] / "data" / "shared_random_policy_v1.pt"
ABORT_OBS_FRACTION = 0.30


def collect_episodes(seed: int, episodes: int, max_len: int = 200):
    import crafter

    def to_chw(image):
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    env = crafter.Env(seed=seed, length=max_len)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        obs = env.reset()
        frames, actions, rewards, continues = [to_chw(obs)], [], [], []
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, info = env.step(action)
            frames.append(to_chw(obs))
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
    if DATA_CACHE.exists():
        return torch.load(DATA_CACHE, weights_only=False)
    data = {
        "train_episodes": collect_episodes(0, 24) + collect_episodes(1, 24),
        "heldout_episodes": collect_episodes(2, 10),
        "heldout_probe": collect(2, 400),
    }
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, DATA_CACHE)
    return data


@torch.no_grad()
def encode(encoder, frames: np.ndarray, device, batch=64):
    out = []
    for start in range(0, len(frames), batch):
        obs = torch.from_numpy(frames[start:start + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(obs).float().cpu())
    return torch.cat(out)


def probes_block(target_encoder, probe, cfg, device):
    tokens = encode(target_encoder, probe.obs, device)
    grid = cfg.image_size // cfg.patch_size
    stats = target_statistics(tokens, cfg.registers)
    stats.pop("covariance_eigenvalues_desc", None)
    return {
        **stats,
        **semantic_probe(tokens, probe.semantic, probe.player_pos, cfg.registers, grid, device),
        **inventory_probe(tokens, probe.inventory, cfg.registers),
    }


def curve_point(target_encoder, probe, cfg, device, step, loss_mean):
    tokens = encode(target_encoder, probe.obs[:200], device)
    stats = target_statistics(tokens, cfg.registers)
    return {
        "step": step,
        "loss_mean_100": loss_mean,
        "observation_variance_fraction": stats["target_observation_variance_fraction"],
        "stream_rank_mean": stats["target_stream_effective_rank_mean"],
    }


def gates(final: dict, untrained: dict, losses: list[float]) -> dict:
    first = float(np.mean(losses[:100])) if losses else None
    last = float(np.mean(losses[-100:])) if losses else None
    return {
        "G1a_observation_variance_fraction": bool(
            final["target_observation_variance_fraction"]
            >= untrained["target_observation_variance_fraction"] - 0.05
        ),
        "G1b_same_stream_unrelated": bool(
            final["target_same_stream_unrelated_cosine"]
            >= 0.80 * untrained["target_same_stream_unrelated_cosine"]
        ),
        "G2a_stream_rank_mean": bool(
            final["target_stream_effective_rank_mean"]
            >= 0.90 * untrained["target_stream_effective_rank_mean"]
        ),
        "G2b_patch_pool_rank": bool(
            final["target_patch_pool_covariance_rank"]
            >= 0.90 * untrained["target_patch_pool_covariance_rank"]
        ),
        "G3_semantic": bool(
            final.get("semantic_probe_sane", False)
            and final["semantic_token_accuracy"]
            >= untrained["semantic_token_accuracy"] - 0.02
        ),
        "G4_inventory": bool(
            (final.get("inventory_r2_mean_varying") or -1.0)
            >= (untrained.get("inventory_r2_mean_varying") or -1.0) - 0.02
        ),
        "G5_loss_decreased_30pct": bool(
            losses and last is not None and last <= 0.70 * first
        ),
        "loss_first_last_100": [first, last],
    }


def train_ijepa(cfg, frames, probe, steps, device, batch=64):
    torch.manual_seed(101)
    model = IJEPAPretrainer(cfg).to(device)
    params = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    frame_rng = np.random.default_rng(2027)
    mask_generator = torch.Generator().manual_seed(2027)
    losses, curve, aborted = [], [], False
    started = time.perf_counter()
    for step in range(steps):
        idx = frame_rng.integers(0, len(frames), size=batch)
        obs = torch.from_numpy(frames[idx]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(obs, mask_generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        model.update_target()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            point = curve_point(
                model.target_encoder, probe, cfg, device, step + 1,
                float(np.mean(losses[-100:])),
            )
            curve.append(point)
            print(f"[ijepa] step {step+1} loss {point['loss_mean_100']:.4f} "
                  f"obs_frac {point['observation_variance_fraction']:.3f} "
                  f"stream_rank {point['stream_rank_mean']:.2f}", flush=True)
            if point["observation_variance_fraction"] < ABORT_OBS_FRACTION:
                aborted = True
                break
    minutes = round((time.perf_counter() - started) / 60, 2)
    return model, losses, curve, aborted, minutes


def train_hybrid(cfg, episodes, steps, device):
    replay = EpisodeReplay()
    for ep in episodes:
        replay.add(Episode(**ep))
    torch.manual_seed(101)
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(2027)
    losses = []
    for _ in range(steps):
        batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, LossConfig())
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.mark_parameters_updated()
        model.update_target()
        losses.append(float(output.metrics["jepa"]))
    return model, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tag", default="step1")
    parser.add_argument("--arms", nargs="+", default=["ijepa", "hybrid"])
    args = parser.parse_args()
    device = torch.device("cuda")

    data = load_shared_data()
    frames = np.concatenate([ep["obs"] for ep in data["train_episodes"]])
    probe = data["heldout_probe"]

    torch.manual_seed(101)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    untrained_model = IJEPAPretrainer(cfg).to(device)
    untrained = probes_block(untrained_model.target_encoder, probe, cfg, device)
    del untrained_model
    torch.cuda.empty_cache()

    report = {
        "protocol": "reviews/2026-07-13-step1-protocol.md",
        "steps": args.steps,
        "train_frames": len(frames),
        "untrained": untrained,
        "arms": {},
    }
    if "ijepa" in args.arms:
        model, losses, curve, aborted, minutes = train_ijepa(cfg, frames, probe, args.steps, device)
        final = probes_block(model.target_encoder, probe, cfg, device)
        report["arms"]["ijepa"] = {
            "minutes": minutes, "aborted": aborted, "curve": curve,
            "final": final, "gates": gates(final, untrained, losses),
        }
        torch.save(
            {"online": model.online_encoder.state_dict(),
             "target": model.target_encoder.state_dict(),
             "steps": args.steps},
            ARTIFACTS / f"ssl_step1_ijepa_{args.tag}.pt",
        )
        del model
        torch.cuda.empty_cache()
    if "hybrid" in args.arms:
        model, losses = train_hybrid(cfg, data["train_episodes"], args.steps, device)
        final = probes_block(model.target_encoder, probe, cfg, device)
        report["arms"]["hybrid"] = {
            "final": final, "gates": gates(final, untrained, losses),
        }
        del model
        torch.cuda.empty_cache()

    out = ARTIFACTS / f"ssl_step1_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    summary = {
        arm: {k: v for k, v in body["gates"].items()}
        for arm, body in report["arms"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
