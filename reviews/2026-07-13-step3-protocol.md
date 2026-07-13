# Step 3 protocol (pre-registered): frozen-encoder temporal predictor, rollout 0 vs 1

Matrix step 3 (2026-07-13 consensus re-audit): "Freeze the one passing
encoder/EMA target. Train deterministic temporal predictor with rollout weight
0 vs 1, identical replay index schedule, three seeds, GRU only. Paired
window-bootstrap copy margin lower CI > 0 and ≥5% at a predeclared k ≤ 8;
one-step/semantic/inventory not regressed." Committed before implementation.

## Fixed components

- Encoder: `lejepa_global` @ 1000 updates (step-1 winner,
  `reviews/artifacts/ssl_step1_lejepa_global_g1000.pt`, strict-load verified by
  both agents). Its EMA/target weights are loaded into BOTH the world model's
  online and target encoders; all encoder parameters frozen
  (`requires_grad=False`); no EMA updates. Semantic/inventory retention is
  therefore fixed by construction and not re-gated.
- World model: compact M3HJWM, GRU backend only (backend attribution is step
  4), deterministic predictor, `mask_ratio=0`, anti-collapse weights
  variance=covariance=0 (nothing trainable to protect; SSL already gated).
- Trainables: temporal core, future predictor, action embedding, reward and
  continuation heads (reliability weights remain 0/shadow).
- Budget: 4000 updates, batch 4 × T16, AdamW lr 1e-4, grad clip 100, bf16
  autocast — the established world-update recipe.

## Arms (6 runs)

rollout weight ∈ {0.0, 1.0} × model seeds {101, 202, 303}. Within each seed,
the two rollout arms consume IDENTICAL replay window indices (explicit
`np.random.default_rng(seed)` handed to the corrected `EpisodeReplay.sample`).
Rollout arm uses the merged core implementation (`LossConfig.rollout=1.0`,
`rollout_steps=2`, prefix encoder detached, gradients through
predictor → temporal → predictor).

## Evaluation (single audited instrument)

`openloop_eval` from `reviews/artifacts/phase_d_backend.py` (the 2026-07-13
re-audit's corrected evaluator): 48 held-out windows, 8-step observed prefix,
16 imagined steps replaying real actions; fixed raw-RGB changed patches
(registers excluded); paired window-level bootstrap (2000 draws).

## Pre-registered gates

- **S3-A (copy-fidelity, the binding D1 successor), predeclared k = 8:** the
  rollout=1 arm must show paired window copy-margin bootstrap 95% lower bound
  > 0 AND relative improvement ≥ 5%, in at least 2 of 3 seeds. Full k = 1..16
  curves reported for all arms either way.
- **S3-B (bridge efficacy):** at k = 8, the rollout=1 arm's paired window
  margin exceeds the rollout=0 arm's in at least 2 of 3 seeds (same windows,
  same replay schedule — paired by construction).
- **S3-C (one-step non-regression):** rollout=1 one-step changed-patch error
  ≤ 1.2 × rollout=0, in at least 2 of 3 seeds.
- Diagnostics (reported, not gated): per-k curves, warm imagine-step latency,
  peak VRAM, reward/continuation training losses.

Decision rule: all three gates pass → step 4 (identical protocol, GRU vs
Mamba-2, same frozen encoder, same replay indices, same instrument). Any gate
fails → stop, report all six runs, consensus before any change.
