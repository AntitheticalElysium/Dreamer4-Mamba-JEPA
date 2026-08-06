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

## Results (48 anchors, 4 env seeds, 12 branches; artifacts fork_oracle_v1.json)

- **R-A: the S3-A bar is GENEROUSLY reachable — the aleatoric-ceiling
  hypothesis is dead.** Empirical oracle deterministic error on changed patches
  at k=8: 0.006 vs copy 0.101 — a **91% relative improvement** (cluster 95%
  CI [86%, 95%]; identical on day-only anchors). The branch-future conditional
  mean is nearly a point; environment stochasticity is almost irrelevant on
  changed patches. The ≥5% bar is reachable eighteen times over BY CONSTRUCTION
  in the same frozen latent space. The gap between our trained models
  (margin ≈ 0) and the oracle is entirely a MODEL deficiency.
- **R-B: actions carry massive signal.** 81% of anchors are action-effective;
  median effect/dispersion = 22.6. The evaluation distribution can discriminate
  action-conditioned dynamics; the trained models' action-shuffle insensitivity
  (companion diagnostic) is therefore a model failure, not a dataset artifact.
- **R-C: consequential stochasticity is modest** (~7% branch divergence in
  reward/inventory/achievements at k=8; termination/health 0%).
- **Registers: copy 0.144 vs oracle 0.0013** — register futures are nearly
  deterministic; the companion's register-drift finding also reflects untapped
  headroom, not noise.

Combined diagnosis: gate sound, distribution sound, bar reachable, actions
informative — and the dynamics stack (temporal core + predictor over the frozen
space) captures almost none of it, settling at copy-parity where an oracle gets
91%. The wall is in the dynamics learning, with candidate causes: (a) loss
allocation — static tokens dominate the JEPA gradient, and copying is optimal
for ~85% of tokens (cf. Dreamer 4's ramp loss weight, which exists precisely to
focus capacity on high-signal terms); (b) optimization budget (losses still
declining at 4,000 updates — companion's observation); (c) predictor/action-
pathway capacity at d=64.

## Consensus question (options, cheapest-discriminating first)

1. **Budget-extension probe:** identical S3-v2 protocol, 4k → 16k updates,
   rollout=1 arms only, 3 seeds. Zero new knobs; directly tests the underfit
   hypothesis the companion raised.
2. **Step 4 as diagnostic at the surviving budget** (user directive: Mamba is
   the next goal): GRU vs Mamba-2, same frozen encoder/replay indices — the
   temporal core is inside the failing stack, so backend comparison is now
   lever-relevant, not just thesis-relevant.
3. **Changed-token loss weighting** (Dreamer-4 ramp-weight precedent, adapted):
   only if (1) fails to move the margin; labelled adaptation with its own
   pre-registration.
