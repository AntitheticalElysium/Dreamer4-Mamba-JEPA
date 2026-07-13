"""Repaired fork oracle + shift-copy baseline (reviews/2026-07-13-microtest-protocol.md).

Companion-spec corrections over v1: raw bundle saved and hashed before
aggregation; day/night-stratified and interaction-tagged anchors; per-branch
S3-style changed masks (union secondary); leave-one-branch-out oracle; both
estimands; all-pairs task divergence + per-anchor incidence. The collector also
stores 8-step observation/action prefixes and extra suffixes so the
counterfactual action-fit microtest trains on the SAME archived data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import ModelConfig, cosine_distance  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from representation_control import changed_patch_mask, patch_change_scores  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
BUNDLE = REPO_ROOT / "data" / "fork_bundle_v2.pt"
ENCODER_CKPT = ARTIFACTS / "ssl_step1_lejepa_global_g1000.pt"
ENV_SEEDS = (21, 22, 23, 24)
DAY_QUOTA, NIGHT_QUOTA = 8, 4          # per env seed
BRANCH_MAIN, BRANCH_EXTRA = 12, 3       # true/alt0 vs alt1/alt2 (microtest)
SUFFIX, PREFIX = 8, 8
MOVE_DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # world (dx, dy)


def chw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_signature(info):
    inventory = info.get("inventory", {})
    return {
        "health": float(inventory.get("health", 0)),
        "inventory_nonhealth": tuple(sorted(
            (k, v) for k, v in inventory.items() if k != "health")),
        "achievements": tuple(sorted(
            k for k, v in info.get("achievements", {}).items() if v)),
    }


def run_branches(snapshot, suffix, base_seed, branches):
    frames, outcomes, positions = [], [], []
    for b in range(branches):
        fork = copy.deepcopy(snapshot)
        fork._world.random.seed(base_seed + b)
        obs_seq, pos_seq, reward_sum, terminated = [], [], 0.0, False
        info = {}
        for a in suffix:
            obs, r, done, info = fork.step(a)
            obs_seq.append(chw(obs))
            pos_seq.append(np.asarray(info["player_pos"], dtype=np.int64))
            reward_sum += float(r)
            if done:
                terminated = True
                while len(obs_seq) < SUFFIX:
                    obs_seq.append(obs_seq[-1])
                    pos_seq.append(pos_seq[-1])
                break
        frames.append(np.stack(obs_seq))
        positions.append(np.stack(pos_seq))
        outcomes.append({
            "reward_sum": reward_sum, "terminated": terminated,
            **_task_signature(info),
        })
        del fork
    return np.stack(frames).astype(np.uint8), outcomes, np.stack(positions)


def collect_bundle():
    import crafter

    anchors = []
    for env_seed in ENV_SEEDS:
        env = crafter.Env(seed=env_seed, length=100_000)
        rng = np.random.default_rng(env_seed)
        obs = env.reset()
        obs_hist = [chw(obs)]
        act_hist = []
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
                pre_sig = _task_signature(
                    {"inventory": dict(env._player.inventory),
                     "achievements": dict(env._player.achievements)})
                suffixes = {
                    name: [int(rng.integers(env.action_space.n)) for _ in range(SUFFIX)]
                    for name in ("true", "alt0", "alt1", "alt2")
                }
                anchor = {
                    "env_seed": env_seed, "daylight": daylight, "night": is_night,
                    "player_pos": np.asarray(env._player.pos, dtype=np.int64),
                    "obs_hist": np.stack(obs_hist[-PREFIX:]).astype(np.uint8),
                    "act_hist": np.asarray(act_hist[-PREFIX:], dtype=np.int64),
                    "suffixes": suffixes, "branches": {},
                }
                base = 70_000 + 977 * len(anchors)
                for j, (name, suf) in enumerate(suffixes.items()):
                    b = BRANCH_MAIN if name in ("true", "alt0") else BRANCH_EXTRA
                    fr, oc, pos = run_branches(snapshot, suf, base + 100 * j, b)
                    anchor["branches"][name] = {
                        "frames": fr, "outcomes": oc, "positions": pos}
                # live env continues with the TRUE suffix; interaction tag from it
                live_done = False
                for a in suffixes["true"]:
                    obs, _, live_done, info = env.step(a)
                    obs_hist.append(chw(obs))
                    act_hist.append(a)
                    if live_done:
                        break
                post_sig = _task_signature(info) if not live_done else None
                anchor["interaction"] = bool(
                    live_done or post_sig != {**pre_sig}
                )
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
    return anchors


def loo_oracle_error(branch_tokens):
    """Leave-one-branch-out: predict branch b with the mean direction of the
    others; error = mean_b (1 - p_b . u_b). branch_tokens [B, S, D]."""
    unit = F.normalize(branch_tokens.float(), dim=-1)
    total = unit.sum(0, keepdim=True)
    loo_mean = (total - unit) / (unit.shape[0] - 1)
    p = F.normalize(loo_mean, dim=-1)
    return (1.0 - (p * unit).sum(-1)).mean(0)          # [S]


def all_pairs_divergence(outcomes, key):
    n = len(outcomes)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([outcomes[i][key] != outcomes[j][key] for i, j in pairs]))


def shift_copy_frame(anchor_obs: np.ndarray, action: int) -> np.ndarray:
    """Non-privileged baseline: scroll the world-view region (rows < 49) by one
    7-px tile according to the movement action; replicate the leading edge;
    HUD rows unchanged. Non-move actions return the frame unchanged."""
    if action not in MOVE_DELTA:
        return anchor_obs.copy()
    dx, dy = MOVE_DELTA[action]
    out = anchor_obs.copy()
    view = out[:, :49, :63]
    # player moves +dx (world x = image cols): content shifts by -7*dx cols
    view = np.roll(view, shift=(-7 * dy, -7 * dx), axis=(1, 2))
    if dx == 1:
        view[:, :, -7:] = view[:, :, -14:-7]
    elif dx == -1:
        view[:, :, :7] = view[:, :, 7:14]
    if dy == 1:
        view[:, -7:, :] = view[:, -14:-7, :]
    elif dy == -1:
        view[:, :7, :] = view[:, 7:14, :]
    out[:, :49, :63] = view
    return out


@torch.no_grad()
def encode(encoder, frames: np.ndarray, device):
    flat = frames.reshape(-1, *frames.shape[-3:])
    out = []
    for start in range(0, len(flat), 64):
        obs = torch.from_numpy(flat[start:start + 64]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(obs).float().cpu())
    tokens = torch.cat(out)
    return tokens.reshape(*frames.shape[:-3], *tokens.shape[-2:])


def cluster_ci(values, clusters, seed=11, draws=2000):
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    means = [
        np.concatenate([values[clusters == c]
                        for c in rng.choice(unique, len(unique), True)]).mean()
        for _ in range(draws)
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    device = torch.device("cuda")
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    pretrainer = IJEPAPretrainer(cfg)
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()
    regs = cfg.registers

    if BUNDLE.exists():
        anchors = torch.load(BUNDLE, weights_only=False)
    else:
        anchors = collect_bundle()
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(anchors, BUNDLE)
    bundle_hash = sha256_file(BUNDLE)

    rows = []
    k = SUFFIX - 1
    for i, anchor in enumerate(anchors):
        anchor_obs = anchor["obs_hist"][-1]
        anchor_tok = encode(encoder, anchor_obs[None], device)[0]
        true = anchor["branches"]["true"]
        branch_k = encode(encoder, true["frames"][:, k], device)     # [B, S, D]

        copy_err_tok = cosine_distance(
            anchor_tok[None].expand_as(branch_k), branch_k)          # [B, S]
        loo_err_tok = loo_oracle_error(branch_k)                     # [S]

        # per-branch S3-style masks (primary) + union (secondary)
        change = patch_change_scores(
            np.repeat(anchor_obs[None], len(branch_k), 0),
            true["frames"][:, k], cfg.patch_size)
        per_branch_mask = changed_patch_mask(change)                 # [B, 64]
        copy_perbranch, loo_perbranch = [], []
        for b in range(len(branch_k)):
            sel = per_branch_mask[b]
            if sel.any():
                copy_perbranch.append(float(copy_err_tok[b, regs:][sel].mean()))
                loo_perbranch.append(float(loo_err_tok[regs:][sel].mean()))
        union = per_branch_mask.any(0)

        # shift-copy baseline at k=1
        k1_tok = encode(encoder, true["frames"][:, 0], device)
        shifted = shift_copy_frame(anchor_obs, anchor["suffixes"]["true"][0])
        shift_tok = encode(encoder, shifted[None], device)[0]
        change1 = patch_change_scores(
            np.repeat(anchor_obs[None], len(k1_tok), 0),
            true["frames"][:, 0], cfg.patch_size)
        mask1 = changed_patch_mask(change1)
        copy1, shift1 = [], []
        for b in range(len(k1_tok)):
            if mask1[b].any():
                sel = mask1[b]
                copy1.append(float(cosine_distance(anchor_tok, k1_tok[b])[regs:][sel].mean()))
                shift1.append(float(cosine_distance(shift_tok, k1_tok[b])[regs:][sel].mean()))
        moved = bool(
            (true["positions"][:, 0] != anchor["player_pos"][None]).any(-1).all()
        )

        outcomes = true["outcomes"]
        rows.append({
            "env_seed": anchor["env_seed"], "night": anchor["night"],
            "interaction": anchor["interaction"], "anchor": i,
            "copy_changed_perbranch": float(np.mean(copy_perbranch)) if copy_perbranch else None,
            "loo_oracle_changed_perbranch": float(np.mean(loo_perbranch)) if loo_perbranch else None,
            "copy_changed_union": float(copy_err_tok[:, regs:][:, union].mean()) if union.any() else None,
            "loo_oracle_changed_union": float(loo_err_tok[regs:][union].mean()) if union.any() else None,
            "copy_registers": float(copy_err_tok[:, :regs].mean()),
            "loo_oracle_registers": float(loo_err_tok[:regs].mean()),
            "k1_copy_changed": float(np.mean(copy1)) if copy1 else None,
            "k1_shiftcopy_changed": float(np.mean(shift1)) if shift1 else None,
            "k1_action_is_move": anchor["suffixes"]["true"][0] in MOVE_DELTA,
            "k1_all_branches_moved": moved,
            "task_all_pairs": {
                key: all_pairs_divergence(outcomes, key)
                for key in ("reward_sum", "terminated", "health",
                            "inventory_nonhealth", "achievements")
            },
        })
        print(f"anchor {i + 1}/{len(anchors)}", flush=True)

    valid = [r for r in rows if r["copy_changed_perbranch"]]
    seeds = [r["env_seed"] for r in valid]
    ratios = [
        (r["copy_changed_perbranch"] - r["loo_oracle_changed_perbranch"])
        / max(r["copy_changed_perbranch"], 1e-9) for r in valid
    ]
    copy_means = [r["copy_changed_perbranch"] for r in valid]
    loo_means = [r["loo_oracle_changed_perbranch"] for r in valid]

    def block(subset_rows):
        v = [r for r in subset_rows if r["copy_changed_perbranch"]]
        if not v:
            return None
        rr = [(r["copy_changed_perbranch"] - r["loo_oracle_changed_perbranch"])
              / max(r["copy_changed_perbranch"], 1e-9) for r in v]
        return {
            "n": len(v),
            "mean_of_ratios": float(np.mean(rr)),
            "ratio_of_means": float(
                1 - np.mean([r["loo_oracle_changed_perbranch"] for r in v])
                / np.mean([r["copy_changed_perbranch"] for r in v])),
        }

    shift_rows = [r for r in rows if r["k1_shiftcopy_changed"] is not None
                  and r["k1_action_is_move"]]
    report = {
        "protocol": "reviews/2026-07-13-microtest-protocol.md (part 1 + 3)",
        "bundle_sha256": bundle_hash,
        "encoder_sha256": sha256_file(ENCODER_CKPT),
        "anchors": len(rows),
        "night_anchors": sum(r["night"] for r in rows),
        "interaction_anchors": sum(r["interaction"] for r in rows),
        "R_A_perbranch_LOO": {
            "mean_of_ratios": float(np.mean(ratios)),
            "mean_of_ratios_cluster95_screening": cluster_ci(ratios, seeds),
            "ratio_of_means": float(1 - np.mean(loo_means) / np.mean(copy_means)),
            "copy_mean": float(np.mean(copy_means)),
            "loo_oracle_mean": float(np.mean(loo_means)),
        },
        "strata": {
            "day": block([r for r in rows if not r["night"]]),
            "night": block([r for r in rows if r["night"]]),
            "interaction": block([r for r in rows if r["interaction"]]),
        },
        "registers_copy_vs_loo": [
            float(np.mean([r["copy_registers"] for r in rows])),
            float(np.mean([r["loo_oracle_registers"] for r in rows])),
        ],
        "task_all_pairs_means": {
            key: float(np.mean([r["task_all_pairs"][key] for r in rows]))
            for key in rows[0]["task_all_pairs"]
        },
        "task_incidence": {
            key: float(np.mean([r["task_all_pairs"][key] > 0 for r in rows]))
            for key in rows[0]["task_all_pairs"]
        },
        "shift_copy_k1_move_anchors": {
            "n": len(shift_rows),
            "copy_mean": float(np.mean([r["k1_copy_changed"] for r in shift_rows])) if shift_rows else None,
            "shift_mean": float(np.mean([r["k1_shiftcopy_changed"] for r in shift_rows])) if shift_rows else None,
            "relative_improvement": float(
                1 - np.mean([r["k1_shiftcopy_changed"] for r in shift_rows])
                / np.mean([r["k1_copy_changed"] for r in shift_rows])) if shift_rows else None,
            "n_all_branches_moved": sum(r["k1_all_branches_moved"] for r in shift_rows),
        },
        "rows": rows,
    }
    out = ARTIFACTS / "fork_oracle_v2.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
