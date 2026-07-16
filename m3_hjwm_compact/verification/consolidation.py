"""Consolidation round (reviews/2026-07-14-consolidation-protocol.md).

Arms: C1 independent-stream GRU x3 seeds, C2 parameter-matched global-64 x3
seeds, C3 topology-matched shuffled controls (one per topology). All
rollout_steps=2, 16k updates, 40k replay, frozen step-1 encoder. Seeds 21-24
bundle is monitoring only; the FINAL evaluation runs once per arm on the fresh
seeds-63-78 bundle (equal branch counts, common RNG across suffixes).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig, cosine_distance  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import load_scaled_data  # noqa: E402
from microtest import openloop_anchor  # noqa: E402
from causal_stage_a import stage_a_model  # noqa: E402
from representation_control import changed_patch_mask, patch_change_scores  # noqa: E402
from fork_oracle_v2 import (  # noqa: E402
    BUNDLE, ENCODER_CKPT, PREFIX, SUFFIX, chw, encode, run_branches,
    sha256_file, _task_signature,
)

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
FINAL_BUNDLE = REPO_ROOT / "data" / "final_bundle_63_78.pt"
FINAL_SEEDS = tuple(range(63, 79))
DAY_QUOTA, NIGHT_QUOTA = 8, 4
BRANCHES = 3
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")
STEPS_TOTAL, CKPT_AT = 16_000, (8_000, 16_000)
TRAIN_SEEDS = (101, 202, 303)


def collect_final_bundle():
    """Equal branch counts and COMMON simulator RNG seeds across all four
    suffixes (removes both original-bundle confounds; companion spec)."""
    import crafter

    anchors = []
    for env_seed in FINAL_SEEDS:
        env = crafter.Env(seed=env_seed, length=100_000)
        rng = np.random.default_rng(env_seed)
        obs = env.reset()
        obs_hist, act_hist = [chw(obs)], []
        day_left, night_left = DAY_QUOTA, NIGHT_QUOTA
        done, since = False, 0
        while day_left or night_left:
            if done:
                obs = env.reset()
                obs_hist, act_hist, done, since = [chw(obs)], [], False, 0
            daylight = float(env._world.daylight)
            is_night = daylight < 0.5
            ready = len(obs_hist) >= PREFIX and len(act_hist) >= PREFIX and since >= 10
            wanted = (is_night and night_left) or ((not is_night) and day_left)
            if ready and wanted:
                snapshot = copy.deepcopy(env)
                suffixes = {
                    name: [int(rng.integers(env.action_space.n)) for _ in range(SUFFIX)]
                    for name in SUFFIX_NAMES
                }
                base = 300_000 + 977 * len(anchors)   # SAME base for all suffixes
                anchor = {
                    "env_seed": env_seed, "daylight": daylight, "night": is_night,
                    "player_pos": np.asarray(env._player.pos, dtype=np.int64),
                    "obs_hist": np.stack(obs_hist[-PREFIX:]).astype(np.uint8),
                    "act_hist": np.asarray(act_hist[-PREFIX:], dtype=np.int64),
                    "suffixes": suffixes, "branches": {},
                }
                for name, suf in suffixes.items():
                    fr, oc, pos = run_branches(snapshot, suf, base, BRANCHES)
                    anchor["branches"][name] = {
                        "frames": fr, "outcomes": oc, "positions": pos}
                live_done = False
                for a in suffixes["true"]:
                    obs, _, live_done, info = env.step(a)
                    obs_hist.append(chw(obs))
                    act_hist.append(a)
                    if live_done:
                        break
                anchors.append(anchor)
                done = live_done
                if is_night:
                    night_left -= 1
                else:
                    day_left -= 1
                since = 0
                del snapshot
                continue
            a = int(rng.integers(env.action_space.n))
            obs, _, done, _ = env.step(a)
            obs_hist.append(chw(obs))
            act_hist.append(a)
            obs_hist = obs_hist[-(PREFIX + 1):]
            act_hist = act_hist[-(PREFIX + 1):]
            since += 1
        del env
        print(f"[final bundle] seed {env_seed} done ({len(anchors)} anchors)", flush=True)
    return anchors


def build_world(backend: str, hidden: int, device) -> M3HJWM:
    cfg = ModelConfig(temporal_backend=backend, predictor="deterministic",
                      mask_ratio=0.0, rollout_steps=2, global_hidden=hidden)
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    world = M3HJWM(cfg).to(device)
    world.online_encoder.load_state_dict(pretrainer.target_encoder.model.state_dict())
    world.target_encoder.model.load_state_dict(pretrainer.target_encoder.model.state_dict())
    for p in world.online_encoder.parameters():
        p.requires_grad_(False)
    for p in world.target_encoder.parameters():
        p.requires_grad_(False)
    return world


@torch.no_grad()
def symmetric_eval(world, encoder, anchors, device):
    """Raw 4x4 suffix distance matrices per anchor (all-token / patch-only /
    changed-patch); symmetric retrieval and separation. Protocol metrics."""
    regs = world.cfg.registers
    rows = []
    for i, anchor in enumerate(anchors):
        targets, masks = [], []
        for name in SUFFIX_NAMES:
            frames = anchor["branches"][name]["frames"]
            toks = encode(encoder, frames, device)                # [B, K, S, D]
            targets.append(F.normalize(toks[:, SUFFIX - 1].float(), dim=-1).mean(0))
            change = patch_change_scores(
                np.repeat(anchor["obs_hist"][-1][None], len(frames), 0),
                frames[:, SUFFIX - 1], world.cfg.patch_size)
            masks.append(changed_patch_mask(change).any(0))       # union of 3 branches
        preds = [openloop_anchor(world, anchor, anchor["suffixes"][n], device)[SUFFIX - 1]
                 for n in SUFFIX_NAMES]

        # 2026-07-14 companion correction (finding 1): candidate-specific
        # masks make argmin columns incomparable and can leak mask structure.
        # PRIMARY changed metric uses ONE common mask per anchor (union over
        # all suffixes and branches); the per-target variant is retained as a
        # diagnostic only and never used for retrieval.
        common_mask = np.stack([m.numpy() if hasattr(m, "numpy") else np.asarray(m)
                                for m in masks]).any(0)
        common_mask = torch.from_numpy(common_mask)
        d_all = np.zeros((4, 4)); d_patch = np.zeros((4, 4)); d_changed = np.full((4, 4), np.nan)
        d_changed_targetmask = np.full((4, 4), np.nan)
        for s in range(4):
            for t in range(4):
                d = cosine_distance(preds[s], targets[t])          # [S]
                d_all[s, t] = float(d.mean())
                d_patch[s, t] = float(d[regs:].mean())
                if bool(common_mask.any()):
                    d_changed[s, t] = float(d[regs:][common_mask].mean())
                if masks[t].any():
                    d_changed_targetmask[s, t] = float(d[regs:][masks[t]].mean())
        def retrieval(m):
            valid = ~np.isnan(m).any(1)
            if not valid.any():
                return None
            return float(np.mean([np.nanargmin(m[s]) == s for s in range(4) if valid[s]]))
        def separation(m):
            diag = np.nanmean(np.diag(m))
            off = np.nanmean(m[~np.eye(4, dtype=bool)])
            return float(off - diag)
        rows.append({
            "env_seed": anchor["env_seed"], "anchor": i, "night": anchor["night"],
            "d_all": d_all.tolist(), "d_patch": d_patch.tolist(),
            "d_changed": d_changed.tolist(),
            "d_changed_targetmask_diagnostic": d_changed_targetmask.tolist(),
            "retrieval_all": retrieval(d_all),
            "retrieval_patch": retrieval(d_patch),
            "retrieval_changed": retrieval(d_changed),
            "separation_all": separation(d_all),
            "separation_patch": separation(d_patch),
        })
    return rows


def seed_level_summary(rows, key):
    by_seed = {}
    for r in rows:
        if r[key] is not None:
            by_seed.setdefault(r["env_seed"], []).append(r[key])
    means = np.array([np.mean(v) for v in by_seed.values()])
    n = len(means)
    mean = float(means.mean())
    se = float(means.std(ddof=1) / np.sqrt(n))
    tcrit = 2.131 if n == 16 else 2.0   # t(15, .975) = 2.131
    return {"mean": mean, "ci95": [mean - tcrit * se, mean + tcrit * se], "n_seeds": n}


def run_arm(name, backend, hidden, seed, shuffle_actions, train_data, monitor_anchors,
            encoder, device):
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in train_data:
        replay.add(Episode(**ep))
    torch.manual_seed(seed)
    world = build_world(backend, hidden, device)
    n_params = sum(p.numel() for p in world.parameters() if p.requires_grad)
    weights = dataclasses.replace(LossConfig(), variance=0.0, covariance=0.0, rollout=1.0)
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(seed)
    losses, monitor = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, STEPS_TOTAL + 1):
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
        losses.append(float(out.metrics["jepa"]))
        if step in CKPT_AT:
            torch.save(
                {"trainable": {n: p.detach().cpu() for n, p in world.named_parameters()
                               if p.requires_grad},
                 "optimizer": optimizer.state_dict(),
                 "numpy_rng": rng.bit_generator.state,
                 "torch_rng_cpu": torch.get_rng_state(),
                 "config": {"backend": backend, "global_hidden": hidden,
                            "seed": seed, "shuffled": shuffle_actions,
                            "rollout_steps": 2, "trainable_params": n_params},
                 "loss_history": losses},
                ARTIFACTS / f"consol_{name}_{step}.pt")
            causal, _ = stage_a_model(world, encoder, monitor_anchors, device)
            monitor.append({"step": step,
                            "monitor_retrieval": causal["retrieval_4way_mean"],
                            "monitor_separation": causal["matched_separation_mean"]})
            print(f"[{name}] {step}: monitor retrieval "
                  f"{causal['retrieval_4way_mean']:.3f}", flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)
    info = {"trainable_params": n_params, "train_minutes": minutes,
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "monitor": monitor,
            "loss_first_last_100": [float(np.mean(losses[:100])), float(np.mean(losses[-100:]))]}
    return world, info


def main():
    device = torch.device("cuda")
    train, _ = load_scaled_data()
    monitor_anchors = torch.load(BUNDLE, weights_only=False)
    if FINAL_BUNDLE.exists():
        final_anchors = torch.load(FINAL_BUNDLE, weights_only=False)
    else:
        final_anchors = collect_final_bundle()
        torch.save(final_anchors, FINAL_BUNDLE)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO_ROOT).stdout.strip()
    arms = (
        [(f"C1_gru_s{s}", "gru", 192, s, False) for s in TRAIN_SEEDS]
        + [(f"C2_glob64_s{s}", "global_gru", 64, s, False) for s in TRAIN_SEEDS]
        + [("C3_gru_shuf", "gru", 192, 101, True),
           ("C3_glob64_shuf", "global_gru", 64, 101, True)]
    )
    report_path = ARTIFACTS / "consolidation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"[resume] {len(report['arms'])} arms already complete:",
              list(report["arms"]), flush=True)
        report["head_commit_resume"] = head
    else:
        report = None
    report = report or {
        "protocol": "reviews/2026-07-14-consolidation-protocol.md",
        "head_commit": head,
        "hashes": {"encoder": sha256_file(ENCODER_CKPT),
                   "monitor_bundle": sha256_file(BUNDLE),
                   "final_bundle": sha256_file(FINAL_BUNDLE)},
        "final_bundle_stats": {
            "anchors": len(final_anchors),
            "night": int(sum(a["night"] for a in final_anchors)),
            "seeds": len(FINAL_SEEDS),
        },
        "arms": {},
    }
    for name, backend, hidden, seed, shuffled in arms:
        if name in report["arms"]:
            continue
        world, info = run_arm(name, backend, hidden, seed, shuffled,
                              train, monitor_anchors, encoder, device)
        rows = symmetric_eval(world, encoder, final_anchors, device)
        (ARTIFACTS / f"consol_rows_{name}.json").write_text(json.dumps(rows))
        summary = {variant: seed_level_summary(rows, f"retrieval_{variant}")
                   for variant in ("all", "patch", "changed")}
        summary["separation_all"] = seed_level_summary(rows, "separation_all")
        summary["separation_patch"] = seed_level_summary(rows, "separation_patch")
        report["arms"][name] = {**info, "final": summary}
        (ARTIFACTS / "consolidation.json").write_text(json.dumps(report, indent=2))
        print(f"[{name}] FINAL retrieval_all {summary['all']['mean']:.4f} "
              f"CI {[round(x,4) for x in summary['all']['ci95']]}", flush=True)
        del world
        torch.cuda.empty_cache()

    print("saved", ARTIFACTS / "consolidation.json")


if __name__ == "__main__":
    main()
