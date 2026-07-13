"""Counterfactual action-fit microtest (reviews/2026-07-13-microtest-protocol.md, part 2).

Trains the CURRENT frozen-encoder GRU+predictor stack on fork-bundle windows
(identical prefixes paired with 4 distinct action suffixes x 3 RNG branches)
and reads out, on train and held-out anchors:
  - teacher-forced fit (overfit capability),
  - open-loop k=1/k=8 copy margins on per-branch changed patches,
  - correct-vs-shuffled suffix separation.
Pre-registered interpretation table lives in the protocol. Reward/continuation
losses are weighted 0 here (no per-step rewards in the bundle; the question is
prediction only).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import LossConfig, ModelConfig, cosine_distance  # noqa: E402
from representation_control import changed_patch_mask, patch_change_scores  # noqa: E402
from step3_temporal import build_frozen_world  # noqa: E402
from fork_oracle_v2 import BUNDLE, PREFIX, SUFFIX, sha256_file  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")
TRAIN_PER_SEED, HELD_PER_SEED = 6, 2
BRANCHES_USED = 3
STEPS = 2000


def build_windows(anchors, indices):
    windows = []
    for ai in indices:
        anchor = anchors[ai]
        actions_prefix = anchor["act_hist"][1:]                    # 7
        for name in SUFFIX_NAMES:
            branch = anchor["branches"][name]
            for b in range(BRANCHES_USED):
                obs = np.concatenate([anchor["obs_hist"], branch["frames"][b]])
                acts = np.concatenate([actions_prefix, anchor["suffixes"][name]])
                prev = np.concatenate([anchor["act_hist"][:1], acts])[:16]
                windows.append({
                    "anchor": ai, "suffix": name, "branch": b,
                    "obs": obs.astype(np.uint8),                    # [16,C,H,W]
                    "actions": acts.astype(np.int64),               # [15]
                    "previous_actions": prev.astype(np.int64),      # [16]
                })
    return windows


def batchify(windows, idx, device):
    sel = [windows[i] for i in idx]
    return {
        "obs": torch.from_numpy(np.stack([w["obs"] for w in sel])).to(device),
        "actions": torch.from_numpy(np.stack([w["actions"] for w in sel])).to(device),
        "previous_actions": torch.from_numpy(
            np.stack([w["previous_actions"] for w in sel])).to(device),
        "rewards": torch.zeros(len(sel), 15, device=device),
        "continues": torch.ones(len(sel), 15, device=device),
    }


@torch.no_grad()
def teacher_forced_loss(world, windows, weights, device, batch=8):
    losses = []
    for start in range(0, len(windows), batch):
        b = batchify(windows, range(start, min(start + batch, len(windows))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(b, weights)
        losses.append(float(out.metrics["jepa"]))
    return float(np.mean(losses))


@torch.no_grad()
def openloop_anchor(world, anchor, suffix_actions, device):
    """Observe the 8-frame prefix, imagine 8 steps under suffix_actions.
    Returns predicted latents [8, S, D]."""
    obs = torch.from_numpy(anchor["obs_hist"][None]).to(device)
    prev0 = int(anchor["act_hist"][0])
    state = world.initial_state(1, device)
    for t in range(PREFIX):
        prev = torch.tensor([prev0 if t == 0 else int(anchor["act_hist"][t])],
                            device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = world.observe_step(obs[:, t], prev, state)
    preds = []
    for k in range(SUFFIX):
        action = torch.tensor([int(suffix_actions[k])], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state, _, _, pred = world.imagine_step(state, action, deterministic_mode=True)
        preds.append(pred.selected[0].float().cpu())
    return torch.stack(preds)


@torch.no_grad()
def evaluate_split(world, anchors, indices, encoder, device, shuffle_map):
    regs = world.cfg.registers
    rows = []
    for ai in indices:
        anchor = anchors[ai]
        anchor_obs = anchor["obs_hist"][-1]
        true = anchor["branches"]["true"]
        pred_true = openloop_anchor(world, anchor, anchor["suffixes"]["true"], device)
        pred_shuf = openloop_anchor(
            world, anchor, anchors[shuffle_map[ai]]["suffixes"]["true"], device)

        from fork_oracle_v2 import encode
        row = {"anchor": ai}
        for k in (0, SUFFIX - 1):
            targets = encode(encoder, true["frames"][:BRANCHES_USED, k], device)
            change = patch_change_scores(
                np.repeat(anchor_obs[None], BRANCHES_USED, 0),
                true["frames"][:BRANCHES_USED, k], world.cfg.patch_size)
            masks = changed_patch_mask(change)
            anchor_tok = encode(encoder, anchor_obs[None], device)[0]
            copy_vals, pred_vals, shuf_vals = [], [], []
            for b in range(BRANCHES_USED):
                if not masks[b].any():
                    continue
                sel = masks[b]
                copy_vals.append(float(
                    cosine_distance(anchor_tok, targets[b])[regs:][sel].mean()))
                pred_vals.append(float(
                    cosine_distance(pred_true[k], targets[b])[regs:][sel].mean()))
                shuf_vals.append(float(
                    cosine_distance(pred_shuf[k], targets[b])[regs:][sel].mean()))
            if copy_vals:
                row[f"k{k + 1}_copy"] = float(np.mean(copy_vals))
                row[f"k{k + 1}_pred"] = float(np.mean(pred_vals))
                row[f"k{k + 1}_pred_shuffled"] = float(np.mean(shuf_vals))
        rows.append(row)

    def agg(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else None

    out = {"rows": rows}
    for k in (1, SUFFIX):
        out[f"k{k}"] = {
            "copy": agg(f"k{k}_copy"), "pred": agg(f"k{k}_pred"),
            "pred_shuffled": agg(f"k{k}_pred_shuffled"),
            "copy_margin": (agg(f"k{k}_copy") - agg(f"k{k}_pred"))
            if agg(f"k{k}_copy") else None,
            "action_separation": (agg(f"k{k}_pred_shuffled") - agg(f"k{k}_pred"))
            if agg(f"k{k}_pred") else None,
        }
    return out


def main():
    device = torch.device("cuda")
    anchors = torch.load(BUNDLE, weights_only=False)
    by_seed = {}
    for i, a in enumerate(anchors):
        by_seed.setdefault(a["env_seed"], []).append(i)
    train_idx, held_idx = [], []
    for seed, idxs in sorted(by_seed.items()):
        train_idx += idxs[:TRAIN_PER_SEED]
        held_idx += idxs[TRAIN_PER_SEED:TRAIN_PER_SEED + HELD_PER_SEED]
    shuffle_map = {i: j for i, j in zip(
        train_idx + held_idx, np.roll(train_idx + held_idx, 5))}

    windows = build_windows(anchors, train_idx)
    held_windows = build_windows(anchors, held_idx)

    torch.manual_seed(404)
    world = build_frozen_world(device)
    weights = dataclasses.replace(
        LossConfig(), variance=0.0, covariance=0.0, reward=0.0,
        continuation=0.0, rollout=1.0)
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(404)
    losses = []
    for step in range(STEPS):
        idx = rng.integers(0, len(windows), size=4)
        batch = batchify(windows, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        losses.append(float(out.metrics["jepa"]))
        if (step + 1) % 500 == 0:
            print(f"[microtest] step {step + 1} jepa {np.mean(losses[-100:]):.4f}",
                  flush=True)

    from model import ModelConfig
    from ssl_ijepa import IJEPAPretrainer
    from fork_oracle_v2 import ENCODER_CKPT
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    report = {
        "protocol": "reviews/2026-07-13-microtest-protocol.md (part 2)",
        "bundle_sha256": sha256_file(BUNDLE),
        "train_anchors": len(train_idx), "held_anchors": len(held_idx),
        "train_windows": len(windows), "steps": STEPS,
        "jepa_first_last_100": [float(np.mean(losses[:100])), float(np.mean(losses[-100:]))],
        "teacher_forced": {
            "train": teacher_forced_loss(world, windows, weights, device),
            "held": teacher_forced_loss(world, held_windows, weights, device),
        },
        "openloop_train": evaluate_split(world, anchors, train_idx, encoder, device, shuffle_map),
        "openloop_held": evaluate_split(world, anchors, held_idx, encoder, device, shuffle_map),
    }
    torch.save(
        {"trainable": {n: p.detach().cpu() for n, p in world.named_parameters()
                       if p.requires_grad},
         "optimizer": optimizer.state_dict()},
        ARTIFACTS / "microtest_v1_model.pt")
    out_path = ARTIFACTS / "microtest_v1.json"
    slim = {k: v for k, v in report.items()}
    for split in ("openloop_train", "openloop_held"):
        slim[split] = {k: v for k, v in report[split].items() if k != "rows"}
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(slim, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
