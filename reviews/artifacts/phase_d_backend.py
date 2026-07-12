"""Phase D: temporal-backend comparison (GRU vs Mamba-2) + copy-fidelity bar.

PRE-REGISTERED:
  - Mask setting: the Phase B arm with higher improvement_over_copy_changed
    (tie-break: higher final rank). The GRU arm REUSES that Phase B checkpoint
    (identical data/seed/budget); only Mamba-2 is trained fresh here.
  - D1 (copy-fidelity bar, gates ANY policy training): open-loop imagination
    from an 8-step real prefix, replaying the REAL recorded actions for
    k=1..16, must beat the frozen-latent copy baseline on changed tokens at
    k<=8 for at least one backend. Report per-k curves either way.
  - D2: Mamba-2 vs GRU compared on the same multi-step error, recurrent step
    latency, training throughput, and peak VRAM. "Mamba is beneficial" is only
    claimed if D2 shows lower error at matched budget.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/m3_hjwm_compact")
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/m3_hjwm_compact/verification")

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig, cosine_distance  # noqa: E402
from phase_b_long import load_shared_data  # noqa: E402

SCRATCH = Path(__file__).parent
ARTIFACTS = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/reviews/artifacts")
PREFIX, HORIZON, WINDOWS = 8, 16, 48


def pick_mask_setting():
    arms = {}
    for tag in ("masked", "unmasked"):
        path = ARTIFACTS / f"phase_b_long_{tag}.json"
        arms[tag] = json.loads(path.read_text())
    key = lambda a: (a["final"]["prediction"]["improvement_over_copy_changed"],
                     a["rank_curve"][-1]["rank"])
    winner = max(arms, key=lambda t: key(arms[t]))
    return winner, arms[winner]


def train_backend(backend, mask_ratio, data, steps, device):
    replay = EpisodeReplay()
    for ep in data["train_episodes"]:
        replay.add(Episode(**ep))
    torch.manual_seed(101)
    cfg = ModelConfig(temporal_backend=backend, predictor="deterministic",
                      mask_ratio=mask_ratio)
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weights = LossConfig()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(steps):
        batch = replay.sample(batch=4, observations=16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        model.mark_parameters_updated()
        model.update_target()
        if (step + 1) % 500 == 0:
            print(f"[phase_d {backend}] step {step+1} "
                  f"jepa {float(output.metrics['jepa']):.4f}", flush=True)
    return model, {
        "train_minutes": round((time.perf_counter() - started) / 60, 1),
        "train_peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def sample_windows(episodes, count, length, rng):
    windows = []
    while len(windows) < count:
        ep = episodes[rng.integers(len(episodes))]
        span = length + 1
        if len(ep["obs"]) < span + 1:
            continue
        start = int(rng.integers(1, len(ep["obs"]) - span))  # start>0: prev action known
        windows.append({
            "obs": ep["obs"][start:start + span],
            "actions": ep["actions"][start:start + span - 1],
            "prev_action": ep["actions"][start - 1],
        })
    return windows


@torch.no_grad()
def openloop_eval(model, episodes, device):
    """Observe PREFIX real steps, then imagine HORIZON steps replaying the real
    actions; compare generated tokens vs target-encoder tokens of real frames."""
    rng = np.random.default_rng(202)
    windows = sample_windows(episodes, WINDOWS, PREFIX + HORIZON, rng)
    obs = torch.from_numpy(np.stack([w["obs"] for w in windows])).to(device)      # [N,P+H+1,C,H,W]
    actions = torch.from_numpy(np.stack([w["actions"] for w in windows])).to(device)
    prev0 = torch.from_numpy(np.asarray([w["prev_action"] for w in windows])).to(device)
    n = obs.shape[0]

    state = model.initial_state(n, device)
    for t in range(PREFIX):
        prev = prev0 if t == 0 else actions[:, t - 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = model.observe_step(obs[:, t], prev, state)

    # Last observed frame is obs[PREFIX-1]; imagination step k0 applies the real
    # action a[PREFIX-1+k0] (transition obs[PREFIX-1+k0] -> obs[PREFIX+k0]) and
    # its prediction is compared to the real frame obs[PREFIX+k0]. The copy
    # baseline freezes the latent of the last observed frame.
    anchor = model.target_encoder(obs[:, PREFIX - 1]).float()
    per_k = []
    latencies = []
    for k in range(HORIZON):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state, _, _, pred = model.imagine_step(state, actions[:, PREFIX - 1 + k])
        torch.cuda.synchronize(); latencies.append(time.perf_counter() - t0)
        real = model.target_encoder(obs[:, PREFIX + k]).float()
        d_pred = cosine_distance(pred.selected.float(), real)
        d_copy = cosine_distance(anchor, real)
        changed = d_copy >= d_copy.flatten().quantile(0.75)
        per_k.append({
            "k": k + 1,
            "pred_cosine": float(d_pred.mean()),
            "copy_cosine": float(d_copy.mean()),
            "pred_cosine_changed": float(d_pred[changed].mean()),
            "copy_cosine_changed": float(d_copy[changed].mean()),
            "beats_copy_changed": bool(d_pred[changed].mean() < d_copy[changed].mean()),
        })
    return per_k, float(np.mean(latencies) * 1e3)


def main():
    device = torch.device("cuda")
    data = load_shared_data()
    winner, winner_report = pick_mask_setting()
    mask_ratio = winner_report["mask_ratio"]
    steps = winner_report["steps"]
    print(f"[phase_d] mask setting from Phase B: {winner} (ratio {mask_ratio})", flush=True)

    results = {}
    # GRU arm: reuse the Phase B winner checkpoint (identical protocol).
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=mask_ratio)
    gru = M3HJWM(cfg).to(device)
    gru.load_state_dict(torch.load(winner_report["checkpoint"], weights_only=False)["model"])
    per_k, step_ms = openloop_eval(gru, data["heldout_episodes"], device)
    results["gru"] = {"reused_phase_b_checkpoint": True, "openloop": per_k,
                      "imagine_step_ms": step_ms}
    del gru; torch.cuda.empty_cache()

    mamba, stats = train_backend("mamba2", mask_ratio, data, steps, device)
    per_k, step_ms = openloop_eval(mamba, data["heldout_episodes"], device)
    results["mamba2"] = {**stats, "openloop": per_k, "imagine_step_ms": step_ms}
    torch.save({"model": mamba.state_dict()}, SCRATCH / "phase_d_mamba2.pt")
    del mamba; torch.cuda.empty_cache()

    d1 = {
        backend: any(point["beats_copy_changed"] for point in results[backend]["openloop"][:8])
        for backend in results
    }
    report = {
        "mask_setting": winner,
        "criteria": {
            "D1_any_backend_beats_copy_changed_k_le_8": any(d1.values()),
            "D1_per_backend": d1,
        },
        "results": results,
    }
    out = ARTIFACTS / "phase_d_backend.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["criteria"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
