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

## Status

All companion NO-GO conditions addressed and regression-tested (70/70 incl.
two slow environment regressions). Training still does NOT start until the
companion verifies these repairs.
