# Stage-1b protocol: equal-update real-state control

Status: **registered before implementation or execution on 2026-07-18**.
This is a mechanism audit of Stage 1 on already-spent Stage-1 evaluation
artifacts. It cannot grant planner GO or select an architecture.

## Why this control is required

Stage 1 compares untouched H0 heads with H1/H2 heads that receive 3,000 new
updates. Each H1/H2 update supervises seven teacher-forced real contexts and
only two generated contexts. Consequently:

- H1 minus H0 confounds generated-state exposure with additional head
  optimization;
- H2 minus H1 tests the event-focused sampling intervention, but not whether
  generated states were necessary;
- an absolute H2 ranking CI excludes zero does not show that H2 improved
  ranking relative to H0.

The missing control is an equal-update head adaptation in which the final two
contexts are teacher-forced rather than generated. This completes the
smallest 2x2 design.

## Fixed contract

- Base worlds: the same six committed X-FLM/X-FLG checkpoints at training
  seeds 505/606/707.
- Trainable parameters: reward and continuation heads only.
- Training replay, batch size 8, 3,000 updates, AdamW learning rate 1e-3,
  bf16, window length 10, and all other optimizer defaults exactly match
  committed Stage 1.
- The natural schedule is exactly reconstructed from seed `10000 + model
  seed`. The event-focused schedule uses the exact committed H2 construction.
- Nine task targets are used per window in every arm. The only real/generated
  difference is whether contexts for the final two targets come from their
  real observations or autoregressive prediction.
- All non-head parameters and buffers must remain bit-identical. The
  executable frozen-encoder assertion remains mandatory.

## Arms

| sampling | all-real equal-update control | generated-suffix Stage-1 arm |
|---|---|---|
| natural | R1: nine teacher-forced contexts | H1: seven real + K1/K2 generated |
| event-focused | R2: nine teacher-forced contexts | H2: seven real + K1/K2 generated |

H0 remains the untouched operational reference, not the causal
training-budget control. Committed H1/H2 checkpoints and results are reused;
they are not refit.

## Evaluation

Reuse the hash-pinned Stage-1 natural, terminal, and ranking bundles. They are
spent for this paired mechanism analysis and cannot support a new planner
gate. Retain raw per-target predictions and per-anchor ranking rows.

Report per checkpoint and family:

- every registered Stage-1 reward and continuation metric at K=0/1/2/4/8;
- overall and zero-reward NLL/MAE, positive/negative decoded means, reward-sign
  accuracy/AUROC, and cumulative predicted reward on zero-reward suffixes;
- absolute ranking metrics and paired arm-minus-control differences with
  environment-cluster bootstrap intervals;
- exact trainable names, base/non-head digests, schedule digests, checkpoint
  hashes, wall time, and peak allocated/reserved VRAM.

## Outcome-independent interpretation

1. Consistent H1 > R1 and H2 > R2 at generated depth supports generated-state
   supervision as a causal mechanism.
2. R1 approximately matching H1 means extra head optimization, not generated
   exposure, explains the apparent H1 recovery.
3. R2 approximately matching H2 means event-focused real-state training is
   sufficient; it does not license a generated-state mechanism claim.
4. Event sampling is selected only if its signed reward/ranking gains survive
   paired comparison without an unacceptable increase in false reward on
   zero-reward transitions or degradation of shallow-horizon calibration.
5. Mixed results narrow the claim and keep both sampling arms in Stage 2.

No full-world Stage-2 run should begin until this control and the existing
Stage-1 artifact/statistical audit are resolved.
