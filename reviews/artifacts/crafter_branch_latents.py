"""Quantify Crafter's successor-mode separation in MoP-JEPA's own currency.

MoP-JEPA (2607.05238) operates on pooled L2-normalized latents z in R^d with
cosine distance; its propositions assume "well-separated modes" (OGBench
teleport mazes: successor modes are distant maze cells, so mode separation is on
the order of the distance between unrelated states). This probe measures, with
the TRAINED Phase B (unmasked) target encoder:

  - branch dispersion: max pairwise cosine distance between the successor
    latents of 8 same-action, reseeded-RNG branches (pooled and dense-token max);
  - reference scales: one-step copy distance (how far the world moves in one
    step anyway) and unrelated-pair distance (the "different state" scale).

If branch dispersion << copy distance << unrelated distance, Crafter is out of
MoP-JEPA's separation regime by orders of magnitude, and hard mixtures cannot
find assignment signal above noise.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/m3_hjwm_compact")

from model import M3HJWM, ModelConfig, cosine_distance  # noqa: E402

SCRATCH = Path(__file__).parent
N_BRANCHES, N_PROBES, PROBE_EVERY = 8, 60, 8


def chw(obs):
    return np.ascontiguousarray(obs.transpose(2, 0, 1))


def main():
    import crafter

    device = torch.device("cuda")
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    model = M3HJWM(cfg).to(device).eval()
    state = torch.load(SCRATCH / "phase_b_unmasked.pt", weights_only=False)["model"]
    model.load_state_dict(state)
    encoder = model.target_encoder

    env = crafter.Env(seed=7, length=10_000)
    rng = np.random.default_rng(7)
    obs = env.reset()
    done = False

    branch_sets, current_frames, next_frames = [], [], []
    steps = 0
    while len(branch_sets) < N_PROBES:
        if done:
            obs = env.reset(); done = False
        action = int(rng.integers(env.action_space.n))
        steps += 1
        if steps % PROBE_EVERY == 0:
            branches = []
            for b in range(N_BRANCHES):
                fork = copy.deepcopy(env)
                fork._world.random.seed(90_000 + 31 * len(branch_sets) + b)
                b_obs, _, _, _ = fork.step(action)
                branches.append(chw(b_obs))
            branch_sets.append(np.stack(branches))
            current_frames.append(chw(obs))
        obs, _, done, _ = env.step(action)
        if steps % PROBE_EVERY == 0:
            next_frames.append(chw(obs))

    with torch.no_grad():
        def enc(arr):
            out = []
            for start in range(0, len(arr), 64):
                x = torch.from_numpy(np.asarray(arr[start:start + 64])).to(device)
                out.append(encoder(x).float().cpu())
            return torch.cat(out)

        cur = enc(current_frames)                       # [P, S, D]
        nxt = enc(next_frames)
        flat_branches = enc(np.concatenate(branch_sets))  # [P*B, S, D]
        br = flat_branches.reshape(N_PROBES, N_BRANCHES, *flat_branches.shape[1:])

    regs = cfg.registers
    pool = lambda t: t[..., :regs, :].mean(-2)

    # branch dispersion: max pairwise distance among branches, per probe state
    disp_pooled, disp_token = [], []
    for p in range(N_PROBES):
        z = pool(br[p])                                  # [B, D]
        d = cosine_distance(z[:, None], z[None])         # [B, B]
        disp_pooled.append(float(d.max()))
        dt = cosine_distance(br[p][:, None], br[p][None])  # [B, B, S]
        disp_token.append(float(dt.max()))

    copy_pooled = cosine_distance(pool(cur), pool(nxt))
    unrelated_pooled = cosine_distance(pool(cur), pool(nxt).roll(5, 0))
    report = {
        "probes": N_PROBES, "branches": N_BRANCHES,
        "branch_dispersion_pooled_mean": float(np.mean(disp_pooled)),
        "branch_dispersion_pooled_p90": float(np.quantile(disp_pooled, 0.9)),
        "branch_dispersion_pooled_max": float(np.max(disp_pooled)),
        "branch_dispersion_worst_token_mean": float(np.mean(disp_token)),
        "one_step_copy_pooled_mean": float(copy_pooled.mean()),
        "unrelated_pair_pooled_mean": float(unrelated_pooled.mean()),
        "ratio_dispersion_to_copy": float(np.mean(disp_pooled) / max(1e-9, float(copy_pooled.mean()))),
        "ratio_dispersion_to_unrelated": float(np.mean(disp_pooled) / max(1e-9, float(unrelated_pooled.mean()))),
    }
    print(json.dumps(report, indent=2))
    (SCRATCH / "crafter_branch_latents_results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
