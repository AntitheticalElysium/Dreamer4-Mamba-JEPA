"""Archived rollout experiment plus the corrected core-loss runner.

AUDIT STATUS (2026-07-13): the rollout loss is now implemented and tested in
``m3_hjwm_compact``. This experiment's old verdict is not reproducible evidence:
the representation gate was a false pass, replay sampling was unseeded, changed
tokens came from each model's own latent, and backend latent spaces differed.
``main`` is disabled until a Phase-B-passing shared representation exists.

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "m3_hjwm_compact"))
sys.path.insert(0, str(ROOT / "m3_hjwm_compact" / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig  # noqa: E402
from phase_b_long import heldout_prediction_eval, load_shared_data  # noqa: E402
from phase_d_backend import openloop_eval  # noqa: E402

SCRATCH = Path(__file__).parent
ARTIFACTS = Path(__file__).resolve().parent
T_ROLL = 2

def run(backend: str, data, device, steps=4000):
    replay = EpisodeReplay()
    for ep in data["train_episodes"]:
        replay.add(Episode(**ep))
    torch.manual_seed(101)
    cfg = ModelConfig(
        temporal_backend=backend,
        predictor="deterministic",
        mask_ratio=0.0,
        rollout_steps=T_ROLL,
    )
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weights = LossConfig(rollout=1.0)
    train_rng = np.random.default_rng(101)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    roll_history = []
    for step in range(steps):
        batch = replay.sample(
            batch=4, observations=16, device=device, rng=train_rng
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, weights)
            roll = output.metrics["rollout"]
            loss = output.loss
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
    per_k, latency = openloop_eval(model, data["heldout_episodes"], device)
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
        "imagine_step_latency": latency,
    }


def main():
    raise RuntimeError(
        "rollout efficacy is gated by Phase B and a fixed shared representation; "
        "the archived single-seed protocol must not be rerun as validation"
    )
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
