# Stage-1c protocol: head-only generated-depth ceiling

Status: **registered before implementation or execution on 2026-07-18**.
This is a smallest-discriminating diagnostic on already-spent Stage-1
evaluation artifacts. It cannot select an architecture, grant planner GO, or
turn the spent bundle into fresh evidence.

## Question

Stage 1 trained the shared reward and continuation heads only at generated
depths K=1 and K=2, then proposed full-world retraining to repair the remaining
K8 magnitude and calibration gaps. That proposal assumes, without testing,
that deeper head-only exposure is insufficient. The fixed-context probes show
that K8 task information remains present, so the head-only depth ceiling is
the cheaper causal question.

## Fixed design

- Backends: X-FLM/Mamba-2 and X-FLG/GRU.
- Training seed: 505 only, selected as the lowest registered seed before this
  run. A positive result is exploratory until replicated; a negative result
  cannot reject an effect across training seeds.
- Base: original committed H0 world and task-head states.
- Trainable parameters: reward and continuation heads only. All other
  parameters and buffers must remain bit-identical.
- Replay: uniform natural sampling from the pinned 40K replay; only windows
  eligible for the common 8-real + 8-future contract are used.
- Optimization: batch 8, 3,000 updates, AdamW lr 1e-3 with recorded defaults,
  bf16, identical initial heads and identical replay-window schedule across
  arms.
- Real prefix supervision: seven teacher-forced contexts and their aligned
  task targets in both arms.

## Arms

- D2: task targets at generated K=1,2, matching Stage 1 H1's generated depth.
- D8: task targets at every generated K=1,...,8.

The loss is the mean over all supervised contexts. D8 therefore changes both
depth coverage and the real/generated target proportion (9 targets in D2
versus 15 in D8). This is the intended source-shaped per-step coverage
intervention, not a pure horizon-only operator comparison, and conclusions
must retain that qualification.

No event-focused sampling is allowed: this isolates depth coverage from H2's
reward/terminal sampling and false-reward tradeoff. Reward and continuation
consume the same natural schedule.

## Evaluation and routing

Reuse the hash-pinned Stage-1 natural, terminal, and ranking artifacts, retain
raw paired rows, and report every K=0/1/2/4/8 metric plus training provenance,
state/checkpoint hashes, wall time, and peak VRAM.

Primary D8-minus-D2 readouts:

- K8 reward event AUROC, signed Pearson, event magnitude/MAE, zero-reward MAE,
  sign accuracy/AUROC;
- K8 terminal AUROC, Brier skill, and terminal probability;
- suffix ranking advantage/regret and cumulative absolute predicted reward on
  truly zero-reward suffixes;
- K1 regressions as the shallow-horizon safety check.

Outcome-independent routing:

1. D8 improves K8 task calibration/ranking without a material shallow or
   zero-reward regression: head depth was still binding; do not claim
   full-world retraining is necessary.
2. D8 fails to improve K8 despite the probe headroom: this strengthens the
   case for Stage 2 dynamics/predictor retraining.
3. D8 improves discrimination or magnitude while increasing false reward:
   depth helps but calibration remains binding; test natural-distribution
   recalibration before a full-world run.
4. Any Stage-2 latent/dynamics loss must remain on uniform replay. Event-
   focused reward supervision, if retained, is a separate factorial loss;
   continuation sampling is decoupled from reward-event sampling.
