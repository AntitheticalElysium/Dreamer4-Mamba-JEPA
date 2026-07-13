"""Corrected open-loop evaluation (amendment S3-v2).

Fixes the 2026-07-13 companion findings against the archived evaluator:
- predeclared, deterministic, NON-overlapping window manifest tagged with
  (episode_id, start) — no duplicates, both arms share the manifest;
- episode-cluster bootstrap (windows within an episode are not independent);
- raw per-window margin rows returned for provenance and paired statistics.

The transition math (prefix observation, action indexing, copy anchor,
fixed raw-RGB changed patches) is unchanged from the audited
reviews/artifacts/phase_d_backend.py evaluator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import cosine_distance  # noqa: E402
from representation_control import changed_patch_mask, patch_change_scores  # noqa: E402

PREFIX, HORIZON = 8, 16


def window_manifest(episodes) -> list[dict]:
    """Deterministic non-overlapping windows: stride = PREFIX + HORIZON."""
    span = PREFIX + HORIZON
    manifest = []
    for episode_id, ep in enumerate(episodes):
        limit = len(ep["obs"]) - (span + 1)
        start = 1  # start > 0 so the previous action is known
        while start <= limit:
            manifest.append({"episode": episode_id, "start": start})
            start += span
    if not manifest:
        raise RuntimeError("held-out episodes too short for one window")
    return manifest


def cluster_bootstrap(values: np.ndarray, episode_ids: np.ndarray,
                      seed: int, draws: int = 2000):
    """Resample EPISODES with replacement; mean over their windows per draw."""
    unique = np.unique(episode_ids)
    groups = {e: values[episode_ids == e] for e in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for d in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([groups[e] for e in chosen])
        means[d] = pooled.mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


@torch.no_grad()
def openloop_eval_v2(model, episodes, device):
    """Returns (per_k, latency, manifest). per_k rows carry raw per-window
    margins + episode ids so paired cluster statistics can be computed."""
    manifest = window_manifest(episodes)
    span = PREFIX + HORIZON
    obs = torch.from_numpy(np.stack([
        episodes[w["episode"]]["obs"][w["start"]:w["start"] + span + 1]
        for w in manifest
    ])).to(device)
    actions = torch.from_numpy(np.stack([
        episodes[w["episode"]]["actions"][w["start"]:w["start"] + span]
        for w in manifest
    ])).to(device)
    prev0 = torch.from_numpy(np.asarray([
        episodes[w["episode"]]["actions"][w["start"] - 1] for w in manifest
    ])).to(device)
    episode_ids = np.asarray([w["episode"] for w in manifest])
    n = obs.shape[0]

    model.eval()
    state = model.initial_state(n, device)
    for t in range(PREFIX):
        prev = prev0 if t == 0 else actions[:, t - 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = model.observe_step(obs[:, t], prev, state)

    anchor = model.target_encoder(obs[:, PREFIX - 1]).float()
    per_k = []
    for k in range(HORIZON):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state, _, _, pred = model.imagine_step(
                state, actions[:, PREFIX - 1 + k], deterministic_mode=True
            )
        real = model.target_encoder(obs[:, PREFIX + k]).float()
        d_pred = cosine_distance(pred.selected.float(), real)[:, model.cfg.registers:]
        d_copy = cosine_distance(anchor, real)[:, model.cfg.registers:]
        raw_change = patch_change_scores(
            obs[:, PREFIX - 1].cpu(), obs[:, PREFIX + k].cpu(), model.cfg.patch_size
        )
        changed = changed_patch_mask(raw_change).to(d_pred.device)
        counts = changed.sum(1).clamp_min(1)
        window_pred = (d_pred * changed).sum(1) / counts
        window_copy = (d_copy * changed).sum(1) / counts
        valid = changed.sum(1) > 0
        margins = (window_copy - window_pred).float().cpu().numpy()
        margins_valid = margins[valid.cpu().numpy()]
        ids_valid = episode_ids[valid.cpu().numpy()]
        interval = cluster_bootstrap(margins_valid, ids_valid, seed=500 + k)
        copy_mean = float(window_copy[valid].mean())
        per_k.append({
            "k": k + 1,
            "pred_cosine_changed": float(window_pred[valid].mean()),
            "copy_cosine_changed": copy_mean,
            "margin_mean": float(margins_valid.mean()),
            "margin_cluster_bootstrap_95": interval,
            "relative_margin": float(margins_valid.mean() / max(copy_mean, 1e-9)),
            "fraction_windows_beating_copy": float((margins_valid > 0).mean()),
            "valid_windows": int(valid.sum()),
            "window_margins": [float(x) for x in margins],
            "window_valid": [bool(x) for x in valid.cpu().numpy()],
        })

    timing_action = actions[:, PREFIX - 1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(20):
            state, _, _, _ = model.imagine_step(state, timing_action, deterministic_mode=True)
    torch.cuda.synchronize()
    timings = []
    for _ in range(8):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(50):
                state, _, _, _ = model.imagine_step(state, timing_action, deterministic_mode=True)
        end_event.record()
        end_event.synchronize()
        timings.append(start_event.elapsed_time(end_event) / 50)
    latency = {"median_ms": float(np.median(timings)), "min_ms": float(np.min(timings))}
    return per_k, latency, manifest


def paired_difference_gate(per_k_roll, per_k_base, manifest, k: int, seed: int = 909):
    """S3-B': cluster-bootstrap CI of per-window margin difference at k."""
    row_r = next(p for p in per_k_roll if p["k"] == k)
    row_b = next(p for p in per_k_base if p["k"] == k)
    valid = np.array(row_r["window_valid"]) & np.array(row_b["window_valid"])
    diff = (np.array(row_r["window_margins"]) - np.array(row_b["window_margins"]))[valid]
    ids = np.asarray([w["episode"] for w in manifest])[valid]
    interval = cluster_bootstrap(diff, ids, seed=seed)
    return {
        "paired_diff_mean": float(diff.mean()),
        "paired_diff_cluster_95": interval,
        "pass": bool(interval[0] > 0),
    }
