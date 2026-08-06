# Step-4 protocol (pre-registered): GRU vs Mamba-2 on the shared-global topology

Consensus basis: companion consolidation audit adopted in full. On record:
- Global-64 is the OPERATIONAL step-4 topology, not a proven architectural
  optimum (paired topology difference +1.09 pts, 95% CI [−0.22, +2.39],
  p=0.07 across 3 training seeds; init pairing was imperfect because module
  construction order consumes RNG).
- "Controls exactly at chance" corrected to "statistically consistent with
  chance". Filed G-d (candidate-specific masks) RETRACTED; the repaired
  common-union-mask G-d still selects global 3/3 (its recomputation), and the
  post-hoc canonically-sorted bundle re-evaluation also selects global (its
  sensitivity check).
- "Common simulator RNG" claim for the 63-78 bundle REFUTED (identity-set
  chunk iteration); the archived bundle remains a valid fixed evaluation set
  (identical for every arm) but not a clean common-RNG construction.
- Faithfulness ruling for the record (user question): GlobalGRUTemporal is a
  deliberately minimal shared-memory ABLATION — labelled divergence, no cited
  system mean-pools dense JEPA tokens and broadcasts state back. Its selection
  is an empirically validated engineering choice, not a literature
  reproduction. GlobalMambaTemporal is likewise "SOURCE-INSPIRED" (DRAMA
  supports only the single-global-vector precedent: flattened categorical
  latent + action through a stem; mixer_seq_simple.py:188). A source-faithful
  global arm (CDP-style flattened-latent state) remains an open, separately
  runnable follow-up if architectural claims are ever needed.

## Repairs landed before this protocol (all regression-tested; 69/69)

1. Changed-patch metric: common union mask per anchor is the primary
   definition; per-target masks demoted to diagnostic. Synthetic regression
   proves candidate-specific masks can flip an argmin and the common mask
   cannot.
2. Crafter fork determinism: `crafter_canonical.py` replaces identity-hashed
   chunk sets with position-sorted iteration + defensive RNG rebinding;
   `run_branches_canonical(verify_repeat=True)` enforces bit-exact repeats.
   Regression covers night states (the worst case).
3. `GlobalMambaTemporal` implemented and contract-tested (sequence/step
   equivalence under official kernels, reset isolation with proper cache
   cloning, drop-in world forward). Figures at B=48 deployment shape:
   global-GRU-64 239,747 params / 29,248 temporal / 0.012 MiB cache;
   global-Mamba2 245,083 / 34,584 / 0.891 MiB cache (+2.2% params — reported,
   near-matched).
4. Repo pytest scoped to the compact suite (vendored-tree collection fixed).

## Arms (16k updates, 40k replay, frozen encoder, rollout_steps=2)

- M1 global-GRU-64 × training seeds {101, 202, 303}.
- M2 global-Mamba2 × training seeds {101, 202, 303}.
- M3 shuffled-action controls, one per backend (seed 101).
Paired initialization: for each training seed, ALL shared non-temporal
parameters (encoder frozen anyway; predictor, action embedding, heads,
projector-free) are copied from a single reference initialization so arms
differ ONLY in the temporal core. Identical replay RNG schedules per seed.

## Evaluation

- Final set: fresh seeds 79-94, collected with the CANONICAL collector
  (bit-exact regression must pass in the same session), equal branches (3),
  true common RNG across suffixes, 8 day / 4 night per seed, hashed before
  training starts.
- BLIND: all arms train to completion before any final-set evaluation.
- Metrics: symmetric 4x4 matrices (all-token / patch-only / common-mask
  changed); env-seed-clustered paired per-anchor backend differences;
  inference across training seeds reported separately; day/night and
  action-effective strata mandatory.
- Full manifest before interpretation: HEAD, hashes (encoder, replay, bundle,
  every checkpoint), complete ModelConfig serialization, total+component loss
  histories, CUDA+CPU+NumPy RNG states, VRAM, parameter counts.

## Gates

- Per family: G-a retrieval ≥27% & seed-level LB >25.5%; G-b symmetric
  separation LB >0; G-c ≥ backend-matched shuffled control +1.5 pts; G-d
  common-mask changed retrieval LB >25.5% (majority of 3 seeds each).
- Backend verdict (the thesis readout): paired per-anchor
  Mamba-minus-GRU difference, env-seed-clustered CI, reported per training
  seed AND pooled; a backend "wins" only with a positive pooled lower bound;
  otherwise the result is parity and the choice falls to engineering figures
  (latency, VRAM, cache) — reported warm, both directions.

## Amendment (2026-07-15, adopted from companion NO-GO audit — supersedes
## conflicting clauses above)

1. COLLECTOR REPAIRED: the live env is canonicalized after every reset
   (companion's critical finding: branch forks were canonical but anchor
   DISCOVERY was not, so `canonical_collector: true` was overstated). New
   regression: two full seed-79 collections must be digest-identical
   (test_collector_is_end_to_end_repeatable_one_seed). Bundle REGENERATED:
   sha256 ebabcb2c1c31e82b0aae4d0d9ebc0be63f785ee1718b26a086eaf18e633be674,
   192 anchors / 64 night, manifest records live_env_canonical=true. The
   branch verifier now also compares positions.
2. SHUFFLED CONTROLS x3 SEEDS per backend (M3, seeds 101/202/303) — G-c
   becomes a formal family gate instead of a single-seed screening
   diagnostic. Total arms: 12.
3. HIERARCHICAL DECISION RULE (pre-declared; replaces "positive pooled lower
   bound" which risked treating 3x16 clusters as 48 independent units):
   - Primary metric: all-token symmetric retrieval, paired per anchor.
   - Per training seed: env-seed-clustered bootstrap CI of the mean paired
     Mamba-minus-GRU difference; all three reported.
   - Pooled: TWO-LEVEL bootstrap (resample training seeds, then env seeds
     within each), plus a t interval over the 3 seed means (acknowledged low
     power).
   - Mamba wins ONLY IF pooled two-level LB > 0 AND all three per-seed
     differences are positive; symmetric rule for GRU; otherwise PARITY and
     the choice falls to engineering figures (warm latency, VRAM, cache size,
     both directions).
   - retrieval_changed (common mask) is reported as a secondary verdict, not
     decision-bearing.
4. GRU-72 CONDITIONAL CAPACITY CONTROL pre-registered: if Mamba wins, run
   M4 global-GRU-72 x3 seeds (245,123 params vs Mamba 245,083) under the
   identical contract before attributing the win to the backend rather than
   +2.2% capacity. If GRU-72 matches Mamba, the claim downgrades to
   "capacity, not backend".
5. RUNNER (verification/step4_runner.py) implements five executable checks:
   paired-init shared-parameter digests asserted equal across arms per seed;
   replay-stream sha256 asserted equal across arms per seed (hashed before
   action shuffling); final bundle opened only after all 12 checkpoints
   exist and its sha256 matches the manifest; evaluation under .eval();
   full per-checkpoint provenance (HEAD, versions, full ModelConfig +
   LossConfig, encoder hash, total+component loss histories, NumPy/CPU/CUDA
   RNG states, VRAM, param counts). Smoke mode (--smoke) exercises every
   check on the monitor bundle only.
6. CONSOLIDATION CORRECTION PERSISTED: the eight consol_rows_*.json files
   were re-evaluated from the committed 16k checkpoints with the
   common-union-mask metric (companion's corrected table reproduced exactly:
   per-position GRU 27.73/27.47/27.21, global-64 29.17/27.87/29.56, shuffled
   controls 25.00/25.52 with CIs containing chance). Corrected gate outcome:
   per-position 2/3 seeds, global-64 3/3 — global-64 remains the operational
   selection with a narrower margin than first reported.
7. LossConfig defaults now encode the validated recipe (variance=0,
   covariance=0, rollout=1.0); the runner consumes plain LossConfig().

## Second amendment (2026-07-16, companion runner audit — all 8 findings
## adopted)

1. FAMILY GATES now implement the registered per-training-seed majority rule
   (G-a..G-d decided per seed with env-seed clustering inside each model,
   G-c against the SAME-SEED shuffled control, then 2/3 majority). Synthetic
   regression pins the 2x30%/1x20% case the pooled rule failed.
2. STRICT RESUME + STRICT LOAD: an arm resumes only when its checkpoint's
   source digest, arm config, step count, encoder hash and replay-file hash
   all match; evaluation asserts exact state_dict key/shape equality and
   loads strict=True from FULL state_dict checkpoints (buffers included).
3. DIGESTS: shared-init digest covers the full non-temporal state_dict
   (parameters AND buffers, names/shapes/dtypes); replay digest covers EVERY
   sampled tensor; the replay file's own sha256 is pinned. Regression tests
   cover buffer mutations and per-tensor batch coverage.
4. PROVENANCE: source-file digest over all imported run modules; clean
   tracked tree REQUIRED for real runs (refused otherwise; smoke records
   dirt); Python/torch/NumPy/mamba_ssm/crafter/GPU versions; peak allocated
   AND reserved VRAM; checkpoint sha256s in the report.
5. STRATA implemented and pre-registered: day/night; action-effective with
   PIXEL-effective (any alt suffix changes any final-frame pixel vs true)
   PRIMARY and task/outcome-effective SECONDARY; stratified backend verdicts
   reported for all four strata.
6. GRU-72 is EXECUTABLE via --gru72 (M4 x3 seeds + paired M4-vs-M2 verdict),
   not a printed reminder.
7. ENGINEERING FIGURES measured by the runner itself (warm step ms @B=48,
   warm sequence ms @B=4,T=16, cache MiB @B=48, temporal params) for both
   backends — the pre-declared parity tie-breaker evidence.
8. PHASE RECIPES: `frozen_dynamics_recipe()` (= plain LossConfig, used by the
   runner) vs `online_hybrid_recipe()` (anti-collapse ON, rollout OFF) now
   used explicitly by train.py/ssl_step1.py; ARCHITECTURE_SPEC.md
   reconciliation banner updated (operative two-stage system, global pooled
   topology, phase recipes, reward/continuation already train the temporal
   core). SPR/TACO/DBC sources pinned in SOURCES.lock. The stale 2026-07-15
   smoke report was DELETED; a fresh smoke must run from a clean committed
   tree and its checkpoints must be retained.

Prior-round correction on record: the 2026-07-15 smoke was executed from an
uncommitted source state and its checkpoints were deleted — non-reproducible,
hence inadmissible as evidence. The claim "all NO-GO conditions addressed"
in that round's report was premature.

## Third amendment (2026-07-16, final pre-launch — companion conditional GO
## + user launch authorization)

1. VALIDITY HIERARCHY (pre-registered; the backend comparison is licensed
   only inside it):
   a. Neither family passes its majority gates -> `no_valid_family`: NO
      backend winner is licensed (the paired difference is reported as a
      relative diagnostic only).
   b. Exactly one family passes -> that family is retained OPERATIONALLY
      (it met the validity contract); no general backend-superiority claim.
   c. Both pass -> the paired two-level backend verdict applies.
   d. Both pass at statistical parity -> PRE-REGISTERED TIE-BREAK: choose
      GRU-64, because the intended online imagination path repeatedly calls
      step() (GRU ~4.7x faster warm step, ~76x smaller cache, fewer params,
      simpler dependency); Mamba must show a predictive win or an
      end-to-end imagination-throughput advantage to displace it. Sequence-
      training throughput (where Mamba is ~2.3x faster) is explicitly NOT
      the primary engineering criterion.
   e. Mamba wins and is valid -> verdict is `mamba_wins_pending_gru72` until
      the M4 capacity control has run; only a Mamba win over GRU-72 licenses
      backend (rather than capacity) attribution.
2. RESUME INTEGRITY: an arm resumes only if the checkpoint's CURRENT sha256
   equals the hash recorded in the report when it was written, AND its full
   ModelConfig + LossConfig equal the current run's constructions, AND the
   python/torch/mamba_ssm fingerprint matches (environment drift = retrain).
   Evaluation additionally validates dtypes and finiteness of every loaded
   tensor. Regression: mutating one saved weight while preserving all
   metadata must cause resume rejection.
3. PROVENANCE: state digests hash raw bytes in original dtype (int64
   2^24 / 2^24+1 collision regression); the source digest is derived from
   git-tracked python files under m3_hjwm_compact plus the installed
   mamba2/ssd kernel sources; NVIDIA driver and crafter versions recorded;
   smoke asserts the monitor bundle hash against a pinned constant.
4. STRATA discrimination regression added (synthetic anchors where pixel-
   and task-effectiveness disagree).
5. RECORD CLEANUP: retracted statements rewritten in place in the literature
   notes; ledger "at chance" wording fixed; ModelConfig rollout comment
   fixed; this status section updated.
6. ARTIFACT RETENTION: full-run 16k checkpoints are force-added (they are
   otherwise gitignored) together with the report; 8k rung checkpoints stay
   local with hashes pinned in the report.
7. POST-STEP-4 SEED HYGIENE (companion directive on record): seeds 79-94 are
   SPENT by this run for any future arm selection; BYOL-AC-motivated,
   depth, TACO-inspired, or topology selection requires a newly reserved
   untouched bundle.

## Status

2026-07-16: fresh smoke evidence exists and was independently reproduced by

2026-07-16 RUN COMPLETE (2.4h wall): ALL 24 gate decisions pass (both
families 3/3 seeds, G-a..G-d, blind final set 79-94). Backend verdict:
PARITY (pooled +0.35 pts, two-level CI [-0.69, +1.43]; per-seed +1.30/
+0.26/-0.52 — sign not consistent; all four strata parity). Pre-registered
tie-break applied: OPERATIONAL BACKEND = GRU-64. Note for the record: Mamba
trains FASTER wall-clock (9.7 vs 13.8 min/arm, sequence path) but remains
4.4x slower per deployment step with 76x larger cache — the registered
deployment-centric criterion stands. Controls 24.7-25.5% on the final set.
the companion (12/12 resume-valid, hashes match, parity at 30 steps). The
two launch blockers (validity hierarchy, resume integrity) are repaired in
this amendment's commit. USER AUTHORIZED LAUNCH; companion pre-approved
pending exactly these bounded repairs + a passing resume smoke. The 12-arm
16k run proceeds after this commit's smoke pair passes.
