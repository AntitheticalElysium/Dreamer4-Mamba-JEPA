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

## Status

Repairs complete and committed; per companion NO-GO, training does NOT start
until it verifies these repairs. Fresh-bundle generation (allowed once the
determinism regression passes — it does) may proceed in parallel.
