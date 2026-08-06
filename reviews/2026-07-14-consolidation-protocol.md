# Consolidation round (pre-registered): three-seed topology selection under clean final-set evaluation

Consensus basis: companion Stage-B audit, all findings adopted after senior
verification (artifact hash + every rung metric reproduced; parameter counts
verified to the digit: B1 235,587 / B2 371,843 (+57.8%, temporal 6.4×) /
global-64 239,747 (+1.8%)). Concessions on record:
- Seeds 21-24 were validation, not held-out test (used for ladder decisions AND
  evaluation) — its 8k "pass" that evaporated on untouched seeds is the
  textbook demonstration.
- Stage B moved four levers at once (budget, horizon, topology, capacity); its
  controls show budget is the proven lever, 8-step bridge unnecessary (seed
  202), and capacity confounded my B1/B2 comparison.
- My separation metric was asymmetric (privileged `true` suffix; per-suffix
  label bias demonstrated).
- B2 naming corrected: a local shared-history ablation, NOT a
  Dreamer-CDP/LeWM-shaped reproduction.
- Registered-but-missing Stage-B analyses acknowledged; this round's harness
  saves all of them.
Headline accepted: **weak but real held-out counterfactual action learning
exists in this stack** (first time), in both topologies; shuffled-action
training does not produce it (and B3's positive copy margin at causal chance
definitively retires copy margin as an action gate).

## Arms (all rollout_steps=2, 16k updates, 40k replay, frozen step-1 encoder)

- C1: B1 topology (independent-stream GRU), seeds {101, 202, 303}.
- C2: parameter-matched global (GlobalGRUTemporal, hidden 64), seeds
  {101, 202, 303}.
- C3: topology-matched shuffled-action controls, one per topology (seed 101).

Design decisions registered: fixed 16k budget (proven lever; no ladder → no
rung-selection leakage; 8k checkpoints still saved for provenance). Both arms
share rollout_steps=2 (its sufficiency was shown for B1; symmetric config
isolates topology; the 2-step global arm is technically untested — accepted
risk, noted). Seeds 21-24 bundle: monitoring/diagnostics only.

## Final evaluation set (generated and hashed BEFORE training)

Env seeds 63-78 (16 clusters), 12 anchors each (8 day / 4 night), 4 suffixes
per anchor with EQUAL branch counts (3) and COMMON simulator RNG seeds across
suffixes (removes both original-bundle confounds); action-effectiveness
recorded per anchor. Evaluated ONCE per arm, after all training completes.

## Metrics (symmetric, from raw distance matrices — all saved)

Per anchor: full 4×4 suffix distance matrix (all tokens, patch-only,
changed-patch). Retrieval = diagonal argmin rate over ALL suffixes as correct
target (not just `true`). Symmetric separation = mean(off-diagonal) −
mean(diagonal). Seed-level Student-t intervals over 16 env-seed means.

## Gates (pre-registered; select topology for step-4 Mamba attribution;
control-sufficiency for policy remains a SEPARATE later gate)

Per arm family, majority of 3 seeds on the final set:
- G-a retrieval: mean ≥ 27% AND seed-level 95% lower bound > 25.5%.
- G-b symmetric separation: 95% lower bound > 0.
- G-c topology-matched shuffled delta: retrieval ≥ control + 1.5 pts.
- G-d changed-patch retrieval lower bound > 25.5% (direction confirmation).

Decision rule: one family passes → it proceeds to step 4 (GRU vs Mamba-2 on
that topology; if global wins, the Mamba comparator requires an explicit
shared-state design — the existing adapter is per-stream and is NOT a valid
global comparator). Both pass → the more reliable family (more seeds passing,
then higher lower bound) proceeds; both fail → stop for consensus.

## Provenance (per arm)

8k + 16k checkpoints with optimizer/RNG state; raw distance matrices; training
loss histories; peak VRAM; full config + parameter counts; sha256 manifest of
encoder, replay, both bundles, and all checkpoints; HEAD commit recorded.

## Results (consolidation.json; final bundle 192 anchors / 16 seeds / 64 night, hashed)

Retrieval (all-token) on the fresh final set, seed-level t-intervals:

| arm | s101 | s202 | s303 | matched shuffled control |
|---|---|---|---|---|
| C1 GRU | 28.3% [26.8, 29.7] | 27.2% [25.4, 29.0] | 27.5% [26.2, 28.8] | 25.7% [24.8, 26.5], sep CI ∋ 0 |
| C2 global-64 | 28.8% [27.6, 30.0] | 28.4% [26.8, 30.0] | 29.0% [27.6, 30.5] | 24.7% [24.1, 25.4], sep CI ∋ 0 |

Symmetric separation: every trained arm's 95% lower bound > 0
(+0.0029..+0.0035); both shuffled controls' CIs contain zero. Changed-patch
retrieval confirms direction in 5/6 trained arms (C1-s303 LB 25.4% marginal).

### Pre-registered gate evaluation (per arm family, majority of 3 seeds)

| gate | C1 GRU | C2 global-64 |
|---|---|---|
| G-a retrieval ≥27% & LB>25.5% | PASS 2/3 (s202 LB 25.43% marginal miss) | **PASS 3/3** |
| G-b symmetric separation LB>0 | PASS 3/3 | **PASS 3/3** |
| G-c ≥ control +1.5pts | PASS 3/3 | **PASS 3/3** |
| G-d changed-patch LB>25.5% | PASS 2/3 | **PASS 3/3** |
| seeds passing ALL gates | 1/3 | **3/3** |

**Decision (per registered tie-break — more seeds passing, then higher lower
bounds): C2, the parameter-matched shared-global-memory topology, is selected.**
Both families exhibit real held-out counterfactual action use; the global
topology is more reliable across seeds and uniformly higher. The topology-
matched shuffled controls sitting exactly at chance with zero separation
certify that the trained deltas are causal, not metric-structural.

### Step-4 handoff (consensus items)

1. Step 4 compares GRU vs Mamba-2 ON THE SELECTED GLOBAL TOPOLOGY. The
   existing Mamba adapter is per-stream and is NOT a valid comparator
   (companion finding, adopted). Proposed design for dual sign-off:
   `GlobalMambaTemporal` mirroring GlobalGRUTemporal exactly — pooled token
   input sequence through Mamba-2 block(s) (official kernels, cache semantics
   already test-covered), per-token context = input + proj(state) — which is
   also the SOURCE-ALIGNED shape: DRAMA runs its Mamba over a single global
   flattened latent per step.
2. Seeds 63-78 are now consumed. Step 4's final evaluation set: fresh seeds
   79-94, same construction, generated and hashed before training.
3. Gates for step 4: same G-a..G-d family plus paired per-anchor
   backend-difference CI (same seeds, same replay schedules, same init where
   shapes permit).
