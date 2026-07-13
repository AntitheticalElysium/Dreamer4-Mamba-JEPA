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

## Step-3 v1 results (all six runs; stop per decision rule)

| seed | arm | k=8 paired margin (95% CI) | relative | k=1 pred/copy |
|---|---|---|---|---|
| 101 | rollout 0 | −0.092 [−0.119, −0.064] | −0.96 | 0.0353/0.0317 |
| 101 | rollout 1 | −0.010 [−0.025, +0.005] | −0.11 | 0.0389/0.0317 |
| 202 | rollout 0 | −0.081 [−0.105, −0.054] | −0.84 | 0.0350/0.0317 |
| 202 | rollout 1 | **+0.005** [−0.007, +0.019] | +0.06 | 0.0390/0.0317 |
| 303 | rollout 0 | −0.054 [−0.075, −0.034] | −0.57 | 0.0347/0.0317 |
| 303 | rollout 1 | −0.003 [−0.017, +0.011] | −0.03 | 0.0393/0.0317 |

- **S3-B PASS 3/3** — the rollout bridge improves the paired k=8 margin in
  every seed, eliminating ~90% of the copy deficit with clean attribution
  (same frozen encoder, identical replay indices, paired windows). This is the
  bridge-efficacy result the invalidated 2026-07-12 experiment claimed, now
  properly established.
- **S3-C PASS 3/3** — one-step regression ratio ≈ 1.11 ≤ 1.2.
- **S3-A FAIL 0/3** — with the bridge, the model is statistically
  indistinguishable from the copy baseline at k=8 (all three CIs straddle
  zero; 38–46% of windows beat copy) but does not BEAT it by the registered
  bar (lower CI > 0 and ≥5%). Stopped for consensus.

Diagnostics: rollout component converges (0.52→0.037); warm imagine step
1.4 ms; 90 MiB peak.

## Consensus question

The bridge works; the residual ask is turning copy-parity into a copy-win.
Candidates, preference order:
1. **Data scale** (the longest-deferred lever, consensus-sanctioned two rounds
   ago and never exercised): grow replay from 48 to ~200 random-policy
   episodes (~40k transitions), identical protocol otherwise, single change.
2. rollout_steps 2 → 4 (deeper bridge) — second knob, defer unless (1) fails.
3. Proceed to step 4 under parity (backend attribution is a paired comparison
   and does not logically require an absolute copy win) — protocol change,
   needs explicit consensus.
