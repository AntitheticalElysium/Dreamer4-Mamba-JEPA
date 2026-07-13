# Fork-oracle protocol (pre-registered): is the S3-A bar reachable, and do actions carry signal?

Consensus basis: companion review of S3-v2 (all its findings adopted; the two
strong claims in my S3-v2 write-up are RETRACTED as stated — "data scale
refuted" overreached a three-confound comparison, and "deterministic predictors
can only match copy on stochastic patches" is false under cosine loss, where
the optimal deterministic direction follows the conditional mean of normalized
future latents). User directive on record: once step 3 is validated, the next
goal is the step-4 Mamba backend test.

Verified before registration:
- Crafter same-seed nondeterminism reproduced firsthand (identical seed and
  action stream diverge at step ~90 within one process; episode lengths 118 vs
  127). Nominal seeds do NOT determine trajectories; provenance must rely on
  cached data hashes, never on re-collection.
- Companion's non-binding diagnostics (action-shuffle insensitivity at k=8;
  predicted-register drift −0.024..−0.051 vs copy) accepted as recorded, with
  its own caveat that these are dataset-conditional.

## Design (companion's six points, implemented)

1. Fresh immutable evaluation set: anchors sampled from live random-policy
   rollouts across ≥4 top-level env seeds (~12 anchors each, ≥10 steps apart,
   skipping the first 10 steps); at each anchor a deep-copied simulator
   snapshot is taken; the TRUE 8-action suffix is the one actually executed
   next in the live rollout. Anchor daylight recorded (night render noise is
   RNG-driven; day/night reported separately).
2. From every snapshot: B=12 independent RNG continuations (in-place world-RNG
   reseed) replaying the SAME suffix; observations recorded at k=1..8 plus
   reward sum, termination, health/inventory, achievements at k=8.
3. In the frozen step-1 target space, per anchor and k: conditional-mean drift;
   within-branch variance; copy error; the EMPIRICAL ORACLE deterministic
   cosine error (per token: 1 − ‖mean of branch-normalized latents‖ — the
   minimizer of expected cosine distance); and the oracle's improvement over
   copy.
4. Reported separately: raw-RGB changed patches (per-branch masks and their
   union), registers, pooled tokens, and task-relevant branch divergence
   (reward/termination/inventory/health/achievements).
5. Action control: each snapshot also branches under a SHUFFLED suffix (true
   suffix of another anchor, fixed derangement). Environment-level action
   effect = distance between true-suffix and shuffled-suffix branch-mean
   latents, compared against within-branch dispersion; the action-effective
   anchor subset is where effect > dispersion.
6. The S3-A all-changed-patch result stays on record; any decomposition is
   reported alongside it, never in place of it.

## Pre-registered readouts (no training; measurement only)

- **R-A (reachability):** oracle relative improvement over copy on changed
  patches at k=8, cluster-aggregated over anchors (clusters = env seeds). If
  its 95% upper bound < 5%, the registered S3-A bar is unreachable by ANY
  deterministic predictor on this distribution → S3-A must be redesigned
  around task-relevant quantities (dual consensus), not retrained against.
- **R-B (action signal):** fraction of anchors that are action-effective, and
  the effect/dispersion ratio distribution. If ≈ none are action-effective
  under random-policy suffixes, the evaluation distribution cannot
  discriminate action-conditioned dynamics and a policy-relevant/scripted
  anchor set is required before any further dynamics gate.
- **R-C (task-relevant stochasticity):** branch divergence rates for reward,
  termination, inventory, health, achievements at k=8 — the consequential
  counterpart of the earlier view-cell probe.

Decision rule: results reported with all three readouts regardless of outcome;
next-step selection (S3-A redesign vs anchor-set redesign vs rollout_steps
pilot) is a dual-consensus decision informed by R-A/R-B; step 4 (Mamba)
launches once step 3 is validated under whatever gate the consensus lands on.

Follow-ups also adopted from the companion's verification record: small-replay
hash recorded; per-arm checkpoint hash manifest committed;
m3_hjwm/ARCHITECTURE_SPEC.md status reconciliation delegated to the
implementation agent (its own suggestion) after this consensus round.
