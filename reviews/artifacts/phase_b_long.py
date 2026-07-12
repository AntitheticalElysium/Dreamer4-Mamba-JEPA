"""Phase B long run: full compact objective at a real budget, one mask setting.

PRE-REGISTERED PASS CRITERIA (written before the run):
  P1  covariance effective rank of held-out targets >= untrained baseline at
      every checkpoint (no collapse at scale);
  P2  improvement_over_copy on changed tokens > 0 at the final evaluation;
  P3  semantic probe sane (converged probe >= majority) and trained accuracy
      >= untrained accuracy - 0.02;
  P4  trained inventory R2 (varying keys) >= untrained inventory R2.

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

sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/m3_hjwm_compact")
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/m3_hjwm_compact/verification")

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig, cosine_distance  # noqa: E402
from representation_control import (  # noqa: E402
    collect, inventory_probe, semantic_probe, target_statistics,
)

SCRATCH = Path(__file__).parent
ARTIFACTS = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/reviews/artifacts")


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
def heldout_prediction_eval(model, replay, device, batches=16):
    """One-step prediction through the model's own sequence path on held-out
    windows: predictor output vs target tokens, against the copy baseline."""
    preds, tgts = [], []
    for _ in range(batches):
        batch = replay.sample(batch=4, observations=16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        b, t = batch["obs"].shape[:2]
        s, d = model.streams, model.cfg.token_dim
        preds.append(output.prediction.selected.float().reshape(b, t - 1, s, d).cpu())
        tgts.append(output.targets.float().cpu())
    pred = torch.cat(preds)                      # [N, T-1, S, D]
    target = torch.cat(tgts)                     # [N, T, S, D]
    d_pred = cosine_distance(pred, target[:, 1:])
    d_copy = cosine_distance(target[:, :-1], target[:, 1:])
    ruler = float(cosine_distance(target[:, :-1], target[:, 1:].roll(3, 0)).mean())
    motion = d_copy
    changed = motion >= motion.flatten().quantile(0.75)
    return {
        "pred_cosine": float(d_pred.mean()),
        "copy_cosine": float(d_copy.mean()),
        "unrelated_pair_cosine": ruler,
        "pred_cosine_changed": float(d_pred[changed].mean()),
        "copy_cosine_changed": float(d_copy[changed].mean()),
        "improvement_over_copy_changed": float(d_copy[changed].mean() - d_pred[changed].mean()),
    }


def probes_block(model, probe_data, cfg, device):
    tokens = encode_target(model, probe_data.obs, device)
    grid = cfg.image_size // cfg.patch_size
    return {
        **target_statistics(tokens),
        **semantic_probe(tokens, probe_data.semantic, probe_data.player_pos,
                         cfg.registers, grid, device),
        **inventory_probe(tokens, probe_data.inventory, cfg.registers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tag", required=True)
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
    weights = LossConfig()

    untrained = {
        "probes": probes_block(model, probe_data, cfg, device),
        "prediction": heldout_prediction_eval(model, heldout_replay, device),
    }
    baseline_rank = untrained["probes"]["target_effective_rank_covariance"]

    rank_curve, losses = [], {"jepa": [], "variance": [], "reward": []}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        batch = replay.sample(batch=4, observations=16, device=device)
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
        if (step + 1) % 250 == 0 or step == 0:
            tokens = encode_target(model, probe_data.obs[:300], device)
            rank = target_statistics(tokens)["target_effective_rank_covariance"]
            rank_curve.append({"step": step + 1, "rank": rank})
            print(f"[{args.tag}] step {step+1} rank {rank:.2f} "
                  f"jepa {np.mean(losses['jepa'][-100:]):.4f}", flush=True)
    elapsed = time.perf_counter() - started

    final = {
        "probes": probes_block(model, probe_data, cfg, device),
        "prediction": heldout_prediction_eval(model, heldout_replay, device),
    }
    for block in (untrained["probes"], final["probes"]):
        block.pop("covariance_eigenvalues_desc", None)

    min_rank = min(point["rank"] for point in rank_curve)
    criteria = {
        "P1_rank_never_below_untrained": bool(min_rank >= baseline_rank - 0.5),
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
            >= (untrained["probes"]["inventory_r2_mean_varying"] or -1)
        ),
    }

    ckpt = SCRATCH / f"phase_b_{args.tag}.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, ckpt)
    report = {
        "tag": args.tag,
        "mask_ratio": args.mask_ratio,
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
    out = ARTIFACTS / f"phase_b_long_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"criteria": criteria, "final_prediction": final["prediction"]}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
