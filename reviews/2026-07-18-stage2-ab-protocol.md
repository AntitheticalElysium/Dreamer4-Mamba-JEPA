# Stage-2 A/B protocol: matched full-world per-step training

Status: pre-registered 2026-07-18 under USER + COMPANION consensus (user
directive: "matched A/B test, not open-ended full retrain"; companion
consensus route items 1-9). Registered BEFORE bundle collection and any
fitting. Labels: the Stage-2 objective is SPR/V-JEPA-2-SHAPED per-step world
training (SPR algos.py:269-303 supervises every jump's latent+reward with
terminal masking; V-JEPA-2 train.py:425-447 scores the autoregressive
sequence) — NOT a Dreamer-4 reproduction; the C-matrix negative does not
constrain true MTP.

## The single controlled contrast (the anti-jepa/ discipline)

Two arms branch from the IDENTICAL initial state, replay schedule, optimizer
config, and update budget (16k updates, B=4, T=16 windows, frozen encoder
under the executable contract, AdamW 1e-4, clip 100, bf16):

- ARM A (equal-update baseline): the current validated frozen_dynamics
  objective, trained fresh — the equal-update control Stage-1b proved is
  mandatory (any Stage-2 gain must beat MORE TRAINING, not H0).
- ARM B (Stage 2): identical PLUS, at K=1..2 generated steps per window:
  per-step latent targets (cosine, as the bridge) AND per-step reward +
  continuation supervision on the generated contexts, with post-terminal
  masking of every loss. Terminal-aligned continuation curriculum is
  INCLUDED and recorded as part of the combined intervention (user item 4).
  Original registration intended terminal-aligned windows to enter both
  schedules. The pre-outcome structural amendment below supersedes that
  intent: the realizable terminal pool was added only to Arm B, so the final
  contrast is explicitly a combined intervention rather than a pure
  per-step-supervision contrast.
- Event oversampling OFF in both arms (companion item 5). Shared heads
  (item 2) — the failed depth-indexed continuation is not carried.
- Backend: GRU seed 505 first as the cheap discriminator (user item 3);
  Mamba + additional seeds ONLY if the discriminator passes (item 7).

## Fresh data (hash-pinned before fitting; canonical collectors)

- DEV bundles (monitoring + this A/B readout): natural episodes seeds
  960-975, terminal set 932-947, fork bundle 143-150.
- FINAL bundles (untouched until a planner-gate evaluation): natural
  976-991, terminal 948-963, fork 151-158. Collected and pinned NOW,
  opened only for the eventual planner gate.

## Acceptance (B vs A, paired on identical eval; benefit AND harm)

- Deep reward: K8 event AUROC/AP + signed Pearson up, event decoded
  magnitude up, with K1 NOT significantly worse and zero-suffix false
  reward <= A + .02.
- Continuation: terminal AUROC at K=1/2/4/8 up with Brier skill not worse;
  the depth-1/2 slots must be above .5 (the curriculum's job).
- Ranking: paired chosen-minus-random advantage and regret not worse, with
  env-clustered CIs reported.
- Any pass -> replicate (Mamba + seeds 606/707) before claims; then the
  EVALUATION-ONLY planner harness (built during this stage, user item 8)
  runs planner-vs-random Crafter episodes against the FINAL bundles under
  the revised gate. Actor/critic remains NO-GO throughout.

## Amendment (2026-07-18, pre-outcome, structural): unpadded 16-obs windows
## CANNOT place a terminal at generated depths 1-2 (episodes end at their
## terminal — companion HIGH-5, re-encountered at implementation). The
## terminal-aligned curriculum therefore uses a separate 10-obs episode-end
## pool (terminal at generated depth 2) consumed only by Arm B's per-step
## path every 10th update — recorded as part of the COMBINED intervention
## (user directive item 4). Main 16-obs schedules remain identical across
## arms.

## CORRECTED OUTCOMES (independent audit accepted by user, 2026-07-18)

Arm B **fails the full registered acceptance contract**. The implementation,
checkpoints, indexing, frozen encoder, main replay schedule, and dev readouts
are reproducible. The earlier phrases "deep reward repaired", "central
hypothesis confirmed", and "causal per-step-supervision effect" exceeded the
evidence and are withdrawn.

- NARROW POSITIVE: K8 reward-event AUROC rises `.671 -> .730`; the paired
  episode-cluster delta is `+.059`, CI `[+.008,+.111]`. Decoded absolute event
  magnitude rises `.0057 -> .0624`. This licenses "the combined Arm-B
  objective improves K8 reward-event discrimination and amplitude on this
  dev set."
- REWARD ACCEPTANCE FAILS: K8 signed Pearson is flat/slightly lower
  (`.1615 -> .1605`), AP changes only `.119 -> .128`, and MAE worsens
  `.0245 -> .0416`. K0 AUROC significantly falls by `.075`. K8 zero-target
  absolute prediction grows about 25x (`.00082 -> .02038`). Excluding the 16
  terminal-reward rows makes the paired K8 AUROC CI cross zero
  (`[-.004,+.116]`).
- THE CONTRAST IS COMBINED: Arm B adds a natural generated batch every update
  and an additional terminal-pool generated batch every tenth update. It has
  the same optimizer-update count but more examples and compute
  (`18.5 -> 28.8` minutes). The terminal pool is also a reward-event
  intervention: K2 is 100% terminal and 100% nonzero reward with mean reward
  about `-.25`. Therefore realized "event oversampling OFF" is false in
  distributional effect.
- LATENT DYNAMICS WORSEN: paired frozen-target cosine-error deltas for B-A are
  `+.0079/+.0074/+.0124/+.0314` at K1/K2/K4/K8, all CIs excluding zero.
  Main JEPA loss also ends worse (`.02181 -> .02976`). The world model was not
  repaired; generated task fitting traded against latent accuracy.
- CONTINUATION CALIBRATION FAILS: K1 terminal AUROC falls
  `.941 -> .787`. K2/K4 AUROC is similar, but calling those depths "flat" hid
  severe calibration collapse: Arm-B Brier skill is `-2.745/-12.030/-10.499`
  at K2/K4/K8, with nonterminal false-terminal rates
  `2.64%/9.77%/8.78%`.
- PLANNER-SAFETY FAILS: ranking advantage changes `.277 -> .234`
  (paired CI spans zero), while cumulative prediction on truly zero-return
  suffixes grows by `+.1125`, CI `[+.0631,+.1854]`, far beyond the registered
  `+.02` budget. This required metric was absent from the original outcome
  block.
- MECHANISM: an initialization-batch diagnostic found shared-dynamics
  gradient norms about `2.52/23.54/14.52` for the equally weighted generated
  latent/reward/continuation terms. Task gradients dominate and mildly oppose
  the latent gradient. This is evidence for objective interference, not a
  transition-indexing or cache bug.

No replication, Mamba transfer, final-tier evaluation, planner execution, or
actor/critic training is licensed. The registered follow-up is a uniform-data
factorial separating generated latent supervision from a gradient-balanced
generated reward term, with generated continuation and the terminal pool both
absent. See `reviews/2026-07-18-stage2c-decoupled-protocol.md`.
