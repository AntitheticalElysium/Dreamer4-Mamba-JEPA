"""Stage-2 A/B: matched full-world per-step training (GRU s505 discriminator).

Protocol: reviews/2026-07-18-stage2-ab-protocol.md (registered; bundles
pinned at 3c9d6e1 before fitting). Arm A = current frozen_dynamics objective,
fresh, equal updates. Arm B = identical PLUS SPR/V-JEPA-2-SHAPED per-step
K=1..2 generated-state supervision (latent + reward + continuation,
post-terminal masked). Identical init, schedule, optimizer, budget.
"""
from __future__ import annotations

import hashlib
import json
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

from checkpoint import save_world_checkpoint, sprint_candidate_config  # noqa: E402
from model import (M3HJWM, ModelConfig, assert_encoder_frozen,  # noqa: E402
                   cosine_distance, enforce_frozen_encoder,
                   frozen_dynamics_recipe)
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from consolidation import ARTIFACTS  # noqa: E402
from step4_runner import git_head, software_versions, source_digest, tracked_dirty  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from phase_e_taskheads import ranking_metrics  # noqa: E402
import phase_e_continuation_depth as cont_depth  # noqa: E402
import phase_e_same_target as same_target  # noqa: E402

MANIFEST = ARTIFACTS / "stage2_eval_bundles.manifest.json"
REPORT_PATH = ARTIFACTS / "stage2_ab_report.json"
SEED = 505
UPDATES = 16_000
BATCH = 4
WINDOW = 16
PREFIX = 8
K_GEN = 2
TERMINAL_MIX = 0.10   # fraction of schedule windows with a terminal at gen depth


def build_fresh_world(device):
    torch.manual_seed(SEED)
    world = M3HJWM(sprint_candidate_config("gru")).to(device)
    pre = IJEPAPretrainer(ModelConfig(temporal_backend="gru",
                                      predictor="deterministic", mask_ratio=0.0))
    pre.load_state_dict(torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"],
                        strict=True)
    world.online_encoder.load_state_dict(pre.target_encoder.model.state_dict())
    world.target_encoder.model.load_state_dict(pre.target_encoder.model.state_dict())
    return enforce_frozen_encoder(world)


def build_schedule(train):
    uniform, terminal_aligned = [], []
    for e, ep in enumerate(train):
        continues = np.asarray(ep["continues"])
        for start in range(len(ep["obs"]) - WINDOW + 1):
            uniform.append((e, start))
            if (continues[start + PREFIX - 1:start + PREFIX + K_GEN - 1] < 0.5).any():
                terminal_aligned.append((e, start))
    rng = np.random.default_rng(40_000 + SEED)
    n_term = int(UPDATES * BATCH * TERMINAL_MIX)
    picks = ([uniform[int(rng.integers(len(uniform)))]
              for _ in range(UPDATES * BATCH - n_term)]
             + [terminal_aligned[int(rng.integers(len(terminal_aligned)))]
                for _ in range(n_term)])
    order = rng.permutation(len(picks))
    schedule = [picks[i] for i in order]
    digest = hashlib.sha256(
        np.asarray(schedule, dtype=np.int64).tobytes()).hexdigest()
    return schedule, digest, len(terminal_aligned)


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


def per_step_generated_loss(world, batch, device):
    """SPR/V-JEPA-2-shaped: observe an 8-real-step prefix, imagine K_GEN
    steps with the real actions; supervise EVERY generated step's latent
    (vs frozen target-encoder tokens) + reward + continuation, with
    post-terminal masking."""
    b = batch["obs"].shape[0]
    state = world.initial_state(b, device)
    for t in range(PREFIX):
        state = world.observe_step(batch["obs"][:, t],
                                   batch["previous_actions"][:, t], state)
    alive = torch.ones(b, device=device)
    total = torch.zeros((), device=device)
    for k in range(K_GEN):
        a_idx = PREFIX - 1 + k
        state, reward_logits, continue_logits, pred = world.imagine_step(
            state, batch["actions"][:, a_idx], deterministic_mode=True)
        with torch.no_grad():
            target = world.target_encoder(batch["obs"][:, PREFIX + k]).float()
        latent = cosine_distance(pred.selected.float(),
                                 target).mean(-1)                 # [B]
        r_nll = world.reward.loss(reward_logits,
                                  batch["rewards"][:, a_idx])     # [B]
        c_bce = F.binary_cross_entropy_with_logits(
            continue_logits, batch["continues"][:, a_idx], reduction="none")
        total = total + (alive * (latent + r_nll + c_bce)).mean()
        alive = alive * (batch["continues"][:, a_idx] > 0.5).float()
    return total / K_GEN


def train_arm(arm, schedule, train, device):
    world = build_fresh_world(device)
    init_digest = state_digest(world, exclude_heads=False)
    weights = frozen_dynamics_recipe()
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng_note = f"schedule-driven, no sampling rng ({arm})"
    losses, extras = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for u in range(UPDATES):
        picks = schedule[u * BATCH:(u + 1) * BATCH]
        batch = make_batch(train, picks, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch, weights)
            loss = out.loss
            if arm == "B":
                extra = per_step_generated_loss(world, batch, device)
                loss = loss + extra
                extras.append(float(extra.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        losses.append(float(out.metrics["jepa"]))
        if u % 4000 == 3999:
            print(f"[{arm}] {u+1}: jepa {np.mean(losses[-500:]):.4f}"
                  + (f" gen {np.mean(extras[-500:]):.4f}" if extras else ""),
                  flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)
    assert_encoder_frozen(world, optimizer)
    world.eval()
    path = ARTIFACTS / f"stage2_arm{arm}_s{SEED}.pt"
    digest = save_world_checkpoint(
        path, world, weights,
        loss_histories={"jepa": losses, "generated_extra": extras},
        extra={"arm": arm, "seed": SEED, "updates": UPDATES,
               "init_digest": init_digest, "rng": rng_note})
    return world, {"init_digest": init_digest, "train_minutes": minutes,
                   "checkpoint_sha256": digest,
                   "jepa_last500": float(np.mean(losses[-500:])),
                   "peak_vram_reserved_mib":
                       round(torch.cuda.max_memory_reserved() / 2**20, 1)}


def main():
    device = torch.device("cuda")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("commit first:\n" + "\n".join(dirty))
    manifest = json.loads(MANIFEST.read_text())
    dev = manifest["dev"]
    for key in ("natural", "terminal", "bundle"):
        assert sha256_file(Path(dev[key]["path"])) == dev[key]["sha256"], key
    natural_eps = torch.load(dev["natural"]["path"], weights_only=False)
    terminal_eps = torch.load(dev["terminal"]["path"], weights_only=False)
    anchors = torch.load(dev["bundle"]["path"], weights_only=False)
    nrows = same_target.target_rows(natural_eps)
    narrays = same_target.window_arrays(natural_eps, nrows)
    crows = same_target.target_rows(terminal_eps)
    carrays = same_target.window_arrays(terminal_eps, crows)
    actual_c = cont_depth.continuation_targets(terminal_eps, crows)
    train, _ = load_scaled_data()
    schedule, sched_digest, n_term_pool = build_schedule(train)

    report = {"protocol": "reviews/2026-07-18-stage2-ab-protocol.md",
              "head": git_head(), "source_digest": source_digest(),
              "versions": software_versions(),
              "schedule_sha256": sched_digest,
              "terminal_pool": n_term_pool, "terminal_mix": TERMINAL_MIX,
              "hashes": {"replay": sha256_file(TRAIN_40K_CACHE),
                         **{k: dev[k]["sha256"] for k in dev}},
              "arms": {}, "evaluation": {}}
    worlds = {}
    for arm in ("A", "B"):
        world, info = train_arm(arm, schedule, train, device)
        report["arms"][arm] = info
        worlds[arm] = world
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    assert report["arms"]["A"]["init_digest"] == \
        report["arms"]["B"]["init_digest"], "arms did not branch from same init"

    for arm, world in worlds.items():
        reward_block = same_target.evaluate_world(world, narrays, device)
        del reward_block["predictions"]
        cont_block = cont_depth.evaluate_world(world, carrays, actual_c, device)
        del cont_block["predictions"]
        ranking = ranking_metrics(world, anchors, device)
        rows = ranking.pop("rows")
        (ARTIFACTS / f"stage2_ranking_rows_{arm}.json").write_text(
            json.dumps(rows))
        report["evaluation"][arm] = {
            "reward_depth": reward_block["metrics"],
            "continuation_depth": cont_block["metrics"], "ranking": ranking}
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        k8 = reward_block["metrics"]["k8"]
        print(f"[eval {arm}] k1_auroc "
              f"{reward_block['metrics']['k1']['event_auroc']:.3f} "
              f"k8_auroc {k8['event_auroc']:.3f} "
              f"k8_absev {k8['decoded_abs_event_mean']} "
              f"cont_k1 {cont_block['metrics']['k1']['terminal_auroc']:.3f} "
              f"cont_k8 {cont_block['metrics']['k8']['terminal_auroc']:.3f} "
              f"rank_adv {ranking['chosen_minus_random_mean']}", flush=True)
        del world
        torch.cuda.empty_cache()
    print("stage2 A/B complete")


if __name__ == "__main__":
    main()
