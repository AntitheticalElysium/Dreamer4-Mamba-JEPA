"""What distribution does the terminal sampler actually train on?

Claim under test: `terminal_fraction=0.5` plus MTP horizon repetition plus
`terminal_weight=8` turns the effective objective into a death/health objective,
because the replay holds only 58 terminal transitions in 553,145.

Measured by running the REAL sampler (`sample_sequences`) over episode
structure reconstructed exactly from the replay -- true lengths, actions,
rewards and continues -- with 1x1 dummy pixels, since which windows get drawn
depends only on episode length and `continues`. That makes this exact for the
question asked and costs no RAM against a running training job.

Reports, for terminal_fraction 0.5 and 0.0:
  * fraction of sampled transitions whose continue target is 0
  * the same after MTP horizon expansion (what the heads actually see)
  * continuation BCE coefficient mass on terminals under `terminal_weight`
  * reward label statistics: mean, and negative/positive frequencies
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.common import sample_sequences
from d4_mamba_jepa.craftax_runners import craftax_jepa_config
from d4_mamba_jepa.data import Episode, EpisodeReplay, whole_episode_splits

REPLAY = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
SPLIT_SEED = 20260727


def build_structural_replay():
    """Real actions/rewards/continues, 1x1 dummy pixels (mmap: obs never paged)."""
    records = torch.load(REPLAY, map_location="cpu", weights_only=False, mmap=True)
    splits = whole_episode_splits(len(records), seed=SPLIT_SEED)
    replay = EpisodeReplay(capacity_steps=10 ** 9)
    for i in splits["train"]:
        actions = records[i]["actions"].numpy()
        rewards = records[i]["rewards"].numpy()
        continues = records[i]["continues"].numpy()
        n = len(actions)
        replay.add(Episode(
            obs=np.zeros((n + 1, 3, 1, 1), dtype=np.uint8),
            actions=actions, rewards=rewards, continues=continues,
        ))
    return replay, splits


def audit(replay, *, terminal_fraction, batches, batch_size, sequence_length,
          horizon, terminal_weight, seed, jumps):
    rng = np.random.default_rng(seed)
    cont, rew = [], []
    hcont, hrew = [], []   # the HEAD window only
    for _ in range(batches):
        b = sample_sequences(
            replay, batch_size=batch_size, sequence_length=sequence_length,
            terminal_fraction=terminal_fraction, device=torch.device("cpu"), rng=rng)
        valid = b.outcome_valid.numpy()
        cont.append(b.led_to_continues.numpy()[valid])
        rew.append(b.led_to_rewards.numpy()[valid])
        # _jepa_world_loss slices heads to [context, context+K); a terminal
        # window puts its terminal at the final position, inside this slice.
        ctx = sequence_length - jumps
        hcont.append(b.led_to_continues.numpy()[:, ctx:ctx + jumps].reshape(-1))
        hrew.append(b.led_to_rewards.numpy()[:, ctx:ctx + jumps].reshape(-1))
    cont = np.concatenate(cont)
    rew = np.concatenate(rew)
    hcont = np.concatenate(hcont)
    hrew = np.concatenate(hrew)

    # MTP horizon expansion: head h at position t reads target t+h, so a terminal
    # at position t is a target for the `horizon` positions preceding it.
    per_seq = batch_size * (sequence_length - 1)
    n_seq = len(cont) // per_seq if per_seq else 0
    expanded_terminal = 0
    expanded_total = 0
    for s in range(n_seq):
        row = cont[s * per_seq:(s + 1) * per_seq]
        T = len(row)
        for t in range(T):
            for h in range(horizon):
                if t + h < T:
                    expanded_total += 1
                    expanded_terminal += int(row[t + h] == 0.0)

    terminal_frac = float((cont == 0.0).mean())
    exp_frac = expanded_terminal / max(1, expanded_total)
    # BCE coefficient mass: terminals weighted `terminal_weight`, others 1.
    mass_term = exp_frac * terminal_weight
    mass_cont = (1.0 - exp_frac)
    return {
        "terminal_fraction_setting": terminal_fraction,
        "sampled_transitions": int(len(cont)),
        "terminal_target_fraction": terminal_frac,
        "terminal_target_fraction_after_mtp": exp_frac,
        "continuation_bce_mass_on_terminals": mass_term / (mass_term + mass_cont),
        "reward_mean": float(rew.mean()),
        "reward_negative_fraction": float((rew < 0).mean()),
        "reward_positive_fraction": float((rew > 0).mean()),
        "HEAD_terminal_fraction": float((hcont == 0.0).mean()),
        "HEAD_continuation_bce_mass_on_terminals": float(
            (hcont == 0.0).mean() * terminal_weight
            / ((hcont == 0.0).mean() * terminal_weight + (hcont != 0.0).mean())),
        "HEAD_reward_mean": float(hrew.mean()),
        "HEAD_reward_negative_fraction": float((hrew < 0).mean()),
        "HEAD_reward_positive_fraction": float((hrew > 0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "reviews/artifacts/craftax_sampler_audit.json")
    args = p.parse_args()

    cfg = craftax_jepa_config("transformer")
    replay, splits = build_structural_replay()
    all_cont = np.concatenate([e.continues for e in replay.episodes])
    all_rew = np.concatenate([e.rewards for e in replay.episodes])
    baseline = {
        "train_episodes": len(replay.episodes),
        "train_transitions": int(replay.steps),
        "replay_terminal_transition_fraction": float((all_cont == 0.0).mean()),
        "replay_terminal_count": int((all_cont == 0.0).sum()),
        "replay_reward_mean": float(all_rew.mean()),
        "replay_reward_negative_fraction": float((all_rew < 0).mean()),
        "replay_reward_positive_fraction": float((all_rew > 0).mean()),
    }
    print(json.dumps(baseline, indent=2), flush=True)

    out = {"replay": baseline, "sampled": {}}
    for tf in (0.5, 0.0):
        r = audit(replay, terminal_fraction=tf, batches=args.batches,
                  batch_size=8, sequence_length=cfg.sequence_length,
                  horizon=cfg.continuation_horizon,
                  terminal_weight=cfg.jepa_terminal_weight, seed=args.seed,
                  jumps=cfg.jepa_jumps)
        out["sampled"][str(tf)] = r
        print(json.dumps(r, indent=2), flush=True)

    a, b = out["sampled"]["0.5"], baseline
    out["amplification"] = {
        "terminal_x": (a["terminal_target_fraction"]
                       / max(1e-12, b["replay_terminal_transition_fraction"])),
        "terminal_after_mtp_x": (a["terminal_target_fraction_after_mtp"]
                                 / max(1e-12, b["replay_terminal_transition_fraction"])),
        "negative_reward_x": (a["reward_negative_fraction"]
                              / max(1e-12, b["replay_reward_negative_fraction"])),
        "positive_reward_x": (a["reward_positive_fraction"]
                              / max(1e-12, b["replay_reward_positive_fraction"])),
        "reward_mean_replay": b["replay_reward_mean"],
        "reward_mean_sampled": a["reward_mean"],
    }
    print(json.dumps(out["amplification"], indent=2), flush=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
