"""Stage-1: head-only generated-state task adaptation (H0/H1/H2/H3).

Protocol: reviews/2026-07-18-stage1-head-adaptation-protocol.md
(pre-registered; fresh evaluation bundles pinned at c557166 BEFORE fitting).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import (M3HJWM, WorldState, assert_encoder_frozen,  # noqa: E402
                   enforce_frozen_encoder)
from fork_oracle_v2 import sha256_file  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from consolidation import ARTIFACTS  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402
from exploratory_topology import build_exploratory_world  # noqa: E402
from phase_e_taskheads import ranking_metrics  # noqa: E402
import phase_e_same_target as same_target  # noqa: E402
import phase_e_continuation_depth as cont_depth  # noqa: E402

NATURAL = REPO_ROOT / "data" / "stage1_natural_940_955.pt"
TERMINAL = REPO_ROOT / "data" / "stage1_terminal_916_931.pt"
BUNDLE = REPO_ROOT / "data" / "stage1_bundle_135_142.pt"
MANIFEST = ARTIFACTS / "stage1_eval_bundles.manifest.json"
REPORT_PATH = ARTIFACTS / "stage1_report.json"

CHECKPOINTS = ([("X-FLM", s) for s in (505, 606, 707)]
               + [("X-FLG", s) for s in (505, 606, 707)])
ARMS = ("H1", "H2", "H3")
UPDATES = 3_000
BATCH = 8
WINDOW = 10          # 8-real-obs prefix + 2 transitions
PREFIX = 8
LR = 1e-3


def window_index(train):
    """(episode, start) pairs for T=10 windows; event index for H2."""
    uniform, event = [], []
    for e, ep in enumerate(train):
        rewards = np.asarray(ep["rewards"])
        n = len(ep["obs"])
        for start in range(0, n - WINDOW + 1):
            uniform.append((e, start))
            if np.abs(rewards[start + PREFIX - 1:start + PREFIX + 1]).max() > 1e-6:
                event.append((e, start))
    return uniform, event


def make_batch(train, picks, device):
    obs, actions, rewards, continues, previous = [], [], [], [], []
    for e, start in picks:
        ep = train[e]
        obs.append(ep["obs"][start:start + WINDOW])
        actions.append(ep["actions"][start:start + WINDOW - 1])
        rewards.append(ep["rewards"][start:start + WINDOW - 1])
        continues.append(ep["continues"][start:start + WINDOW - 1])
        prev = np.full(WINDOW, -1, dtype=np.int64)
        if start:
            prev[0] = ep["actions"][start - 1]
        prev[1:] = ep["actions"][start:start + WINDOW - 1]
        previous.append(prev)
    to = lambda x, dt: torch.from_numpy(np.stack(x)).to(device=device, dtype=dt)
    return {"obs": to(obs, torch.uint8), "actions": to(actions, torch.int64),
            "rewards": to(rewards, torch.float32),
            "continues": to(continues, torch.float32),
            "previous_actions": to(previous, torch.int64)}


class AuxHeads(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.event = nn.Linear(dim, 1)
        self.sign = nn.Linear(dim, 1)


def freeze_world_except_heads(world: M3HJWM):
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    for module in (world.reward, world.continuation):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    enforce_frozen_encoder(world)
    return world


def train_heads(world: M3HJWM, arm: str, ckpt_seed: int, train, device):
    freeze_world_except_heads(world)
    aux = AuxHeads(world.cfg.token_dim).to(device) if arm == "H3" else None
    trainable = [p for p in world.parameters() if p.requires_grad]
    if aux is not None:
        trainable += list(aux.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    uniform, event = window_index(train)
    rng = np.random.default_rng(10_000 + ckpt_seed)   # H1/H3 pairing; H2 differs
    losses = []
    world.train()
    for _ in range(UPDATES):
        if arm == "H2":
            half = BATCH // 2
            picks = ([uniform[int(rng.integers(len(uniform)))] for _ in range(half)]
                     + [event[int(rng.integers(len(event)))] for _ in range(half)])
        else:
            picks = [uniform[int(rng.integers(len(uniform)))] for _ in range(BATCH)]
        batch = make_batch(train, picks, device)
        state = world.initial_state(BATCH, device)
        pooled, targets_r, targets_c = [], [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(PREFIX):
                state = world.observe_step(
                    batch["obs"][:, t], batch["previous_actions"][:, t], state)
                if t >= 1:   # context after obs t predicts reward r_{t-1}
                    pooled.append(world.pool(state.tokens))
                    targets_r.append(batch["rewards"][:, t - 1])
                    targets_c.append(batch["continues"][:, t - 1])
            for k in range(2):   # generated steps supervise the SAME heads
                action = batch["actions"][:, PREFIX - 1 + k]
                state, reward_logits, continue_logits, _ = world.imagine_step(
                    state, action, deterministic_mode=True)
                pooled.append(world.pool(state.tokens))
                targets_r.append(batch["rewards"][:, PREFIX - 1 + k])
                targets_c.append(batch["continues"][:, PREFIX - 1 + k])
            pooled = torch.stack(pooled, 1)
            targets_r = torch.stack(targets_r, 1)
            targets_c = torch.stack(targets_c, 1)
            reward_logits = world.reward(pooled)
            continue_logits = world.continuation(pooled)
            loss = (world.reward.loss(reward_logits, targets_r).mean()
                    + F.binary_cross_entropy_with_logits(continue_logits.squeeze(-1)
                                                         if continue_logits.dim() > 2
                                                         else continue_logits,
                                                         targets_c))
            if aux is not None:
                event_logit = aux.event(pooled).squeeze(-1)
                event_label = (targets_r.abs() > 1e-6).float()
                loss = loss + F.binary_cross_entropy_with_logits(
                    event_logit, event_label)
                if bool(event_label.any()):
                    sign_logit = aux.sign(pooled).squeeze(-1)[event_label > 0]
                    sign_label = (targets_r[event_label > 0] > 0).float()
                    loss = loss + F.binary_cross_entropy_with_logits(
                        sign_logit, sign_label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert_encoder_frozen(world, optimizer)
    world.eval()
    return {"loss_first_last": [float(np.mean(losses[:100])),
                                float(np.mean(losses[-100:]))]}


def load_base(arm_kind: str, seed: int, device):
    ckpt = torch.load(ARTIFACTS / f"xtopo_{arm_kind}_s{seed}_16000.pt",
                      weights_only=False)
    world = build_exploratory_world(arm_kind, seed, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    return world.eval()


def evaluate(world, natural_arrays, cont_arrays, actual_continue, anchors,
             device):
    reward_block = same_target.evaluate_world(world, natural_arrays, device)
    del reward_block["predictions"]
    cont_block = cont_depth.evaluate_world(world, cont_arrays, actual_continue,
                                           device)
    del cont_block["predictions"]
    ranking = ranking_metrics(world, anchors, device)
    ranking.pop("rows", None)
    return {"reward_depth": reward_block["metrics"],
            "continuation_depth": cont_block["metrics"], "ranking": ranking}


def main():
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    for key, path in (("natural", NATURAL), ("terminal", TERMINAL),
                      ("bundle", BUNDLE)):
        assert sha256_file(path) == manifest[key]["sha256"], f"{key} drift"
    natural_eps = torch.load(NATURAL, weights_only=False)
    terminal_eps = torch.load(TERMINAL, weights_only=False)
    anchors = torch.load(BUNDLE, weights_only=False)
    natural_rows = same_target.target_rows(natural_eps)
    natural_arrays = same_target.window_arrays(natural_eps, natural_rows)
    cont_rows = same_target.target_rows(terminal_eps)
    cont_arrays = same_target.window_arrays(terminal_eps, cont_rows)
    actual_continue = cont_depth.continuation_targets(terminal_eps, cont_rows)
    train, _ = load_scaled_data()

    report = (json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists()
              else {"protocol": "reviews/2026-07-18-stage1-head-adaptation-protocol.md",
                    "results": {}})
    report["head"] = git_head()
    report["source_digest"] = source_digest()
    report["versions"] = software_versions()
    report["hashes"] = {k: manifest[k]["sha256"] for k in ("natural", "terminal", "bundle")}
    report["hashes"]["replay"] = sha256_file(TRAIN_40K_CACHE)

    for arm_kind, seed in CHECKPOINTS:
        base_tag = f"{arm_kind}_s{seed}"
        for arm in ("H0",) + ARMS:
            tag = f"{base_tag}_{arm}"
            if tag in report["results"]:
                continue
            world = load_base(arm_kind, seed, device)
            info = {}
            if arm != "H0":
                info = train_heads(world, arm, seed, train, device)
                torch.save({"reward": world.reward.state_dict(),
                            "continuation": world.continuation.state_dict(),
                            "arm": arm, "base": base_tag,
                            "source_digest": source_digest()},
                           ARTIFACTS / f"stage1_heads_{tag}.pt")
            block = evaluate(world, natural_arrays, cont_arrays,
                             actual_continue, anchors, device)
            block.update(info)
            report["results"][tag] = block
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
            k8 = block["reward_depth"]["k8"]
            print(f"[{tag}] k8 event_auroc {k8.get('event_auroc')} "
                  f"pearson {k8.get('reward_pearson')} "
                  f"abs_event {k8.get('decoded_abs_event_mean')} "
                  f"rank_adv {block['ranking']['chosen_minus_random_mean']}",
                  flush=True)
            del world
            torch.cuda.empty_cache()
    print("stage1 complete")


if __name__ == "__main__":
    main()
