"""Fork-oracle probe (reviews/2026-07-13-fork-oracle-protocol.md).

Measurement only. Determines (R-A) whether ANY deterministic predictor can
clear the S3-A copy-margin bar on this distribution, (R-B) whether recorded
random actions carry environment-level signal, and (R-C) task-relevant branch
divergence. Uses deep-copied simulator snapshots and in-place world-RNG
reseeding (creatures hold `world.random` by reference).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import ModelConfig, cosine_distance  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from representation_control import changed_patch_mask, patch_change_scores  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
ENCODER_CKPT = ARTIFACTS / "ssl_step1_lejepa_global_g1000.pt"
ENV_SEEDS = (21, 22, 23, 24)
ANCHORS_PER_SEED = 12
BRANCHES = 12
SUFFIX = 8


def chw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def collect_anchors():
    import crafter

    anchors = []
    for env_seed in ENV_SEEDS:
        env = crafter.Env(seed=env_seed, length=10_000)
        rng = np.random.default_rng(env_seed)
        obs = env.reset()
        done, step, taken = False, 0, 0
        while taken < ANCHORS_PER_SEED:
            if done:
                obs = env.reset()
                done, step = False, 0
            if step >= 10 and step % 10 == 0:
                snapshot = copy.deepcopy(env)
                suffix = [int(rng.integers(env.action_space.n)) for _ in range(SUFFIX)]
                anchor = {
                    "env_seed": env_seed,
                    "anchor_obs": chw(obs),
                    "daylight": float(env._world.daylight),
                    "suffix": suffix,
                    "snapshot": snapshot,
                }
                # live rollout continues with the SAME suffix (true actions)
                for a in suffix:
                    obs, _, done, _ = env.step(a)
                    if done:
                        break
                anchors.append(anchor)
                taken += 1
                step += SUFFIX
                continue
            obs, _, done, _ = env.step(int(rng.integers(env.action_space.n)))
            step += 1
        del env
    return anchors


def run_branches(snapshot, suffix, base_seed):
    """B reseeded continuations of the same suffix. Returns frames [B, K, C,H,W]
    and task outcomes."""
    frames, outcomes = [], []
    for b in range(BRANCHES):
        fork = copy.deepcopy(snapshot)
        fork._world.random.seed(base_seed + b)
        obs_seq, reward_sum, terminated = [], 0.0, False
        info = {}
        for a in suffix:
            obs, r, done, info = fork.step(a)
            obs_seq.append(chw(obs))
            reward_sum += float(r)
            if done:
                terminated = True
                while len(obs_seq) < SUFFIX:
                    obs_seq.append(obs_seq[-1])
                break
        frames.append(np.stack(obs_seq))
        inventory = info.get("inventory", {})
        outcomes.append({
            "reward_sum": reward_sum,
            "terminated": terminated,
            "health": float(inventory.get("health", 0)),
            "inventory_nonhealth": tuple(sorted(
                (k, v) for k, v in inventory.items() if k != "health"
            )),
            "achievements": tuple(sorted(
                k for k, v in info.get("achievements", {}).items() if v
            )),
        })
        del fork
    return np.stack(frames), outcomes


@torch.no_grad()
def encode(encoder, frames, device):
    out = []
    flat = frames.reshape(-1, *frames.shape[-3:])
    for start in range(0, len(flat), 64):
        obs = torch.from_numpy(flat[start:start + 64]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(obs).float().cpu())
    tokens = torch.cat(out)
    return tokens.reshape(*frames.shape[:-3], *tokens.shape[-2:])


def oracle_error(branch_tokens):
    """Per token: 1 - ||mean of normalized branch latents|| — the minimizer of
    expected cosine distance over branches."""
    unit = torch.nn.functional.normalize(branch_tokens.float(), dim=-1)
    return 1.0 - unit.mean(0).norm(dim=-1)


def main():
    device = torch.device("cuda")
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True
    )
    encoder = pretrainer.target_encoder.to(device).eval()
    regs = cfg.registers

    anchors = collect_anchors()
    shuffle_map = np.roll(np.arange(len(anchors)), 7)  # fixed derangement

    rows = []
    for i, anchor in enumerate(anchors):
        true_frames, outcomes = run_branches(
            anchor["snapshot"], anchor["suffix"], base_seed=70_000 + 131 * i)
        alt_suffix = anchors[shuffle_map[i]]["suffix"]
        alt_frames, _ = run_branches(
            anchor["snapshot"], alt_suffix, base_seed=70_000 + 131 * i)

        anchor_tok = encode(encoder, anchor["anchor_obs"][None], device)[0]
        true_tok = encode(encoder, true_frames, device)      # [B, K, S, D]
        alt_tok = encode(encoder, alt_frames, device)

        k = SUFFIX - 1  # k=8 (index 7)
        branch_k = true_tok[:, k]                            # [B, S, D]
        copy_err = cosine_distance(
            anchor_tok[None].expand_as(branch_k), branch_k
        ).mean(0)                                            # [S]
        oracle = oracle_error(branch_k)                      # [S]

        change = patch_change_scores(
            np.repeat(anchor["anchor_obs"][None], BRANCHES, 0),
            true_frames[:, k], cfg.patch_size,
        )
        changed_union = changed_patch_mask(change).any(0)    # [64]

        local_copy = copy_err[regs:]
        local_oracle = oracle[regs:]
        sel = changed_union
        mean_dir_drift = float(cosine_distance(
            anchor_tok[regs:][None],
            torch.nn.functional.normalize(branch_k[:, regs:].float(), dim=-1).mean(0)[None],
        ).mean())
        within_var = float(
            (branch_k[:, regs:] - branch_k[:, regs:].mean(0, keepdim=True))
            .pow(2).mean()
        )
        # action signal: true vs shuffled branch means vs within-branch dispersion
        true_mean = branch_k.float().mean(0)
        alt_mean = alt_tok[:, k].float().mean(0)
        effect = float(cosine_distance(true_mean, alt_mean).mean())
        dispersion = float(torch.stack([
            cosine_distance(true_tok[b, k].float(), true_mean).mean()
            for b in range(BRANCHES)
        ]).mean())

        base = outcomes[0]
        rows.append({
            "env_seed": anchor["env_seed"], "anchor": i,
            "daylight": anchor["daylight"], "night": anchor["daylight"] < 0.5,
            "copy_changed": float(local_copy[sel].mean()) if sel.any() else None,
            "oracle_changed": float(local_oracle[sel].mean()) if sel.any() else None,
            "copy_registers": float(copy_err[:regs].mean()),
            "oracle_registers": float(oracle[:regs].mean()),
            "copy_pooled": float(copy_err.mean()),
            "oracle_pooled": float(oracle.mean()),
            "mean_direction_drift": mean_dir_drift,
            "within_branch_variance": within_var,
            "action_effect": effect,
            "action_dispersion": dispersion,
            "action_effective": bool(effect > dispersion),
            "reward_divergence": float(np.mean(
                [o["reward_sum"] != base["reward_sum"] for o in outcomes[1:]])),
            "termination_divergence": float(np.mean(
                [o["terminated"] != base["terminated"] for o in outcomes[1:]])),
            "health_divergence": float(np.mean(
                [o["health"] != base["health"] for o in outcomes[1:]])),
            "inventory_divergence": float(np.mean(
                [o["inventory_nonhealth"] != base["inventory_nonhealth"] for o in outcomes[1:]])),
            "achievement_divergence": float(np.mean(
                [o["achievements"] != base["achievements"] for o in outcomes[1:]])),
        })
        print(f"anchor {i}/{len(anchors)} done", flush=True)

    valid = [r for r in rows if r["copy_changed"] is not None]
    seeds = np.array([r["env_seed"] for r in valid])
    imp = np.array([
        (r["copy_changed"] - r["oracle_changed"]) / max(r["copy_changed"], 1e-9)
        for r in valid
    ])
    rng = np.random.default_rng(11)
    unique = np.unique(seeds)
    boot = []
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        boot.append(np.concatenate([imp[seeds == s] for s in chosen]).mean())
    boot = np.array(boot)

    report = {
        "protocol": "reviews/2026-07-13-fork-oracle-protocol.md",
        "anchors": len(rows), "branches": BRANCHES, "suffix": SUFFIX,
        "R_A_oracle_relative_improvement_over_copy_changed_k8": {
            "mean": float(imp.mean()),
            "cluster_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "upper_bound_below_5pct": bool(np.quantile(boot, 0.975) < 0.05),
        },
        "R_B_action_signal": {
            "fraction_action_effective": float(np.mean([r["action_effective"] for r in rows])),
            "median_effect_over_dispersion": float(np.median(
                [r["action_effect"] / max(r["action_dispersion"], 1e-9) for r in rows])),
        },
        "R_C_task_divergence_means": {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("reward_divergence", "termination_divergence",
                        "health_divergence", "inventory_divergence",
                        "achievement_divergence")
        },
        "register_copy_vs_oracle": [
            float(np.mean([r["copy_registers"] for r in rows])),
            float(np.mean([r["oracle_registers"] for r in rows])),
        ],
        "rows": rows,
    }
    out = ARTIFACTS / "fork_oracle_v1.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
