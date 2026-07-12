"""Option-3 test: V-JEPA-2-AC rollout loss (Eq. 3-4, arXiv:2506.09985) added to
the world objective. Scratchpad-only; m3_hjwm_compact is not modified.

L_total = L_forward (unchanged compact objective incl. VICReg)
        + L_rollout, where L_rollout autoregressively composes
          predictor -> temporal-step -> predictor for T_ROLL steps from a
          teacher-forced prefix and penalizes cosine distance of the FINAL
          prediction to the real EMA target (paper: final-state loss, T=2,
          unweighted sum).

Implementation note: the autoregressive continuation re-runs the parallel
`temporal.sequence` on the extended input sequence instead of `step()`, so the
gradient path is the one already covered by the Mamba adapter tests.

PRE-REGISTERED (before the run), unmasked arm, 4000 updates, same data/seed:
  R1: closed-loop changed-token error at k=4 and k=8 improves >= 2x vs the
      attention-predictor run without rollout loss
      (GRU 0.203/0.269; Mamba-2 0.520/0.740).
  R2: report the D1 bar (beats copy at any k<=8) honestly either way.
  R3: one-step heldout prediction does not regress > 20% vs phase_b unmasked
      (pred_changed 0.0319).
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
from model import LossConfig, M3HJWM, ModelConfig, cosine_distance, multi_block_mask  # noqa: E402
from phase_b_long import heldout_prediction_eval, load_shared_data  # noqa: E402
from phase_d_backend import openloop_eval  # noqa: E402

SCRATCH = Path(__file__).parent
ARTIFACTS = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/reviews/artifacts")
T_ROLL = 2


def rollout_loss(model: M3HJWM, batch: dict) -> torch.Tensor:
    obs, actions = batch["obs"], batch["actions"].long()
    b, t = obs.shape[:2]
    prefix = t - T_ROLL                     # obs 0..prefix-1 teacher-forced
    cfg = model.cfg
    grid = cfg.image_size // cfg.patch_size

    flat = obs[:, :prefix].reshape(b * prefix, *obs.shape[2:])
    if cfg.mask_ratio > 0:
        mask = multi_block_mask(b * prefix, grid, cfg.mask_ratio, cfg.target_blocks, obs.device)
    else:
        mask = None
    tokens = model.online_encoder(flat, mask).reshape(b, prefix, model.streams, cfg.token_dim)

    previous = batch["previous_actions"][:, :prefix].long()
    previous = model._previous_action_indices(previous)
    inputs = tokens + model.action_input(previous)[:, :, None]

    with torch.no_grad():
        final_target = model.target_encoder(obs[:, prefix + T_ROLL - 1])

    horizon = torch.ones(b, dtype=torch.long, device=obs.device)
    prediction = None
    for j in range(T_ROLL):
        context, _ = model.temporal.sequence(inputs)
        step_action = actions[:, prefix - 1 + j]
        modes, logits = model.future.all_predictions(context[:, -1], step_action, horizon)
        idx = logits.argmax(-1)
        prediction = modes[torch.arange(b, device=obs.device), idx]
        if j < T_ROLL - 1:
            nxt = prediction + model.action_input(step_action)[:, None]
            inputs = torch.cat([inputs, nxt[:, None]], dim=1)
    return cosine_distance(prediction, final_target).mean()


def run(backend: str, data, device, steps=4000):
    replay = EpisodeReplay()
    for ep in data["train_episodes"]:
        replay.add(Episode(**ep))
    torch.manual_seed(101)
    cfg = ModelConfig(temporal_backend=backend, predictor="deterministic", mask_ratio=0.0)
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weights = LossConfig()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    roll_history = []
    for step in range(steps):
        batch = replay.sample(batch=4, observations=16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, weights)
            roll = rollout_loss(model, batch)
            loss = output.loss + roll        # Eq. 4: unweighted sum
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        model.mark_parameters_updated()
        model.update_target()
        roll_history.append(float(roll.detach()))
        if (step + 1) % 500 == 0:
            print(f"[rollout {backend}] step {step+1} jepa {float(output.metrics['jepa']):.4f} "
                  f"roll {np.mean(roll_history[-100:]):.4f}", flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)

    heldout_replay = EpisodeReplay()
    for ep in data["heldout_episodes"]:
        heldout_replay.add(Episode(**ep))
    one_step = heldout_prediction_eval(model, heldout_replay, device)
    per_k, step_ms = openloop_eval(model, data["heldout_episodes"], device)
    torch.save({"model": model.state_dict()}, SCRATCH / f"rollout_{backend}.pt")
    del model
    torch.cuda.empty_cache()
    return {
        "backend": backend,
        "train_minutes": minutes,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "rollout_loss_first_last_100": [float(np.mean(roll_history[:100])), float(np.mean(roll_history[-100:]))],
        "one_step": one_step,
        "openloop": per_k,
        "imagine_step_ms": step_ms,
    }


def main():
    device = torch.device("cuda")
    data = load_shared_data()
    baseline = {  # attention predictor, no rollout loss (commit d64d586 artifacts)
        "gru": {4: 0.2026, 8: 0.2685}, "mamba2": {4: 0.5200, 8: 0.7403},
    }
    results = [run(b, data, device) for b in ("gru", "mamba2")]
    criteria = {}
    for res in results:
        by_k = {p["k"]: p for p in res["openloop"]}
        criteria[res["backend"]] = {
            "R1_2x_improvement_k4_k8": bool(
                by_k[4]["pred_cosine_changed"] <= baseline[res["backend"]][4] / 2
                and by_k[8]["pred_cosine_changed"] <= baseline[res["backend"]][8] / 2
            ),
            "R2_beats_copy_any_k_le_8": bool(
                any(by_k[k]["beats_copy_changed"] for k in range(1, 9))
            ),
            "R3_one_step_not_regressed": bool(
                res["one_step"]["pred_cosine_changed"] <= 0.0319 * 1.2
            ),
        }
    report = {"t_roll": T_ROLL, "criteria": criteria, "results": results}
    (ARTIFACTS / "rollout_loss_experiment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
