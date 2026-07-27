"""Control: is the Craftax BC head under-trained, or information-limited?

Both arms of the first Craftax run reached a held-out action accuracy within
0.01 of the majority-class floor (T-JEPA 0.1594, M-JEPA 0.1557, floor 0.1492
for a constant ``do``) after the INHERITED 3,000-update budget. The BC loss was
still falling when it stopped (2.833 -> 2.410), so under-training is the
simplest explanation and has to be eliminated before any claim about the
encoder or the representation objective.

This re-runs BC ONLY, on the frozen world checkpoints already trained, at 10x
the budget, evaluating held-out accuracy on a fixed dev batch set at a ladder of
checkpoints so the SHAPE of the curve is visible rather than one endpoint. If
accuracy is still at the floor with a flat curve, under-training is refuted and
the oracle's verdict about the latent becomes the interpretable next reading.

Nothing about the world, the recipe, or the architecture moves. Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.cartpole_baseline import CartPoleBCPolicy, _clean_agent_tokens
from d4_mamba_jepa.checkpoint import load_checkpoint
from d4_mamba_jepa.craftax_run import SPLIT_SEED, _dev_bc_accuracy, _fixed_dev_batches
from d4_mamba_jepa.data import (
    load_episode_replay,
    replay_sample_to_sequence,
    subset_replay,
    whole_episode_splits,
)
from d4_mamba_jepa.imagination_actor_critic import freeze_module

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
REPLAY_SHA = "7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b"
RUN_DIR = REPO_ROOT / "outputs/d4_mamba_jepa/craftax_expert_v1"
# Inherited BC settings held fixed; only the update count moves.
BATCH = 16
LEARNING_RATE = 1e-4
WARMUP = 250


def majority_floor(replay) -> tuple[float, int]:
    counts = np.zeros(17, dtype=np.int64)
    for episode in replay.episodes:
        counts += np.bincount(np.asarray(episode.actions), minlength=17)
    return float(counts.max() / counts.sum()), int(counts.argmax())


def run_arm(arm_dir: Path, *, train_replay, dev_replay, device, steps, ladder, seed):
    world, _, _ = load_checkpoint(arm_dir / "world.pt", device=device)
    world = world.to(device)
    freeze_module(world)
    world.eval()
    cfg = world.cfg
    dev_batches = _fixed_dev_batches(
        dev_replay, cfg=cfg, count=16, batch_size=8, seed=SPLIT_SEED + 1
    )
    policy = CartPoleBCPolicy(
        d_model=cfg.dynamics_d_model, n_actions=cfg.n_actions
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )
    rng = np.random.default_rng(seed)
    curve, losses = [], []
    started = time.perf_counter()
    for step in range(steps):
        sample = train_replay.sample(BATCH, cfg.sequence_length, device, rng=rng)
        batch = replay_sample_to_sequence(sample)
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda"):
            agent = _clean_agent_tokens(world, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = policy(agent.detach())
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, cfg.n_actions),
                batch.led_to_actions[:, 1:].reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if step < WARMUP:
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * float(step + 1) / WARMUP
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (step + 1) in ladder:
            accuracy = _dev_bc_accuracy(world, policy, dev_batches, device)
            recent = float(np.mean(losses[-500:]))
            curve.append({
                "updates": step + 1,
                "dev_action_accuracy": accuracy,
                "train_ce_last_500": recent,
            })
            print(f"  [{cfg.arm_id}] {step + 1:>6}: dev_acc={accuracy:.4f} "
                  f"train_ce={recent:.4f} ({time.perf_counter() - started:.0f}s)",
                  flush=True)
    return {"arm_id": cfg.arm_id, "curve": curve,
            "seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reviews/artifacts/craftax_bc_budget.json")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    ladder = {500, 1_000, 3_000, 6_000, 10_000, 15_000, 20_000, 25_000, args.steps}

    replay = load_episode_replay(REPLAY, expected_sha256=REPLAY_SHA)
    splits = whole_episode_splits(len(replay.episodes), seed=SPLIT_SEED)
    train_replay = subset_replay(replay, splits["train"])
    dev_replay = subset_replay(replay, splits["dev"])
    floor, action = majority_floor(dev_replay)
    print(f"dev majority-class floor {floor:.4f} (action index {action}); "
          f"chance {1/17:.4f}", flush=True)

    arms = {}
    for arm in ("t_jepa", "m_jepa"):
        print(f"=== {arm} ===", flush=True)
        result = run_arm(
            RUN_DIR / arm, train_replay=train_replay, dev_replay=dev_replay,
            device=device, steps=args.steps, ladder=ladder, seed=args.seed,
        )
        arms[result["arm_id"]] = result

    payload = {
        "question": "is the Craftax BC head under-trained or information-limited?",
        "inherited_budget": 3_000,
        "control_budget": args.steps,
        "dev_majority_floor": floor,
        "chance": 1.0 / 17.0,
        "batch_size": BATCH,
        "learning_rate": LEARNING_RATE,
        "arms": arms,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["arms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
