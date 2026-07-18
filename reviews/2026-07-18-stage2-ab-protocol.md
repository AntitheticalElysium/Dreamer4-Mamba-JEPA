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
  INCLUDED and recorded as part of the combined intervention (user item 4):
  a fraction of windows are aligned so terminal transitions can appear at
  generated depths 1-2 (fixing the C-matrix position shortcut), applied
  IDENTICALLY in both arms' schedules so the contrast stays pure.
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

## OUTCOMES (appended 2026-07-18 after the GRU-505 discriminator; companion
## verification pending — no replication launched)

SPLIT VERDICT — Arm B FAILS full acceptance but confirms the central
hypothesis on its own target:
- DEEP REWARD REPAIRED (the Stage-2 hypothesis): K8 event AUROC .671 -> .730
  (first trained configuration ever past the .70 headroom bar), K2/K4 also
  up (.759->.779, .747->.768), K1 intact (.806 -> .804), decoded K8 event
  magnitude 11x (.0057 -> .0624). Equal updates, identical init+schedule —
  this is the causal per-step-supervision effect at full-world scale.
- CONTINUATION K1 REGRESSED: .941 -> .787 (K2/K4 flat, K8 -.016). Suspect:
  the combined per-step term couples reward and continuation supervision on
  the same generated steps + the depth-2-only terminal pool — exactly the
  coupling the companion warned against (Stage-1 HIGH 3 analog).
- RANKING: point regression (adv .277 -> .234, regret .129 -> .171), paired
  env-clustered CI [-.181, +.146] — unresolved at one seed, not a pass.
- ACCEPTANCE: NOT MET (continuation-K1 and ranking criteria). No
  replication or planner license. Consensus route: decouple continuation
  from the per-step reward term (separate sampling/loss arms per the
  companion's Stage-2 contract 7.3) and rerun the single-factor variant, OR
  accept reward-only per-step supervision (latent+reward, continuation
  teacher-forced-only) as the next A/B arm. Decision needs companion+user.
