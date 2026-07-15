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
