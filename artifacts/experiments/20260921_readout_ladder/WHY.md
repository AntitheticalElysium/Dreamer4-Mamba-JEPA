# Why the imagined successor does not show the agent dying — the causal chain, as measured

Draft, 2026-09-26. Each link says what was measured and where. "Sealed" means judged once on seeds
collected after the rules were committed; "post hoc" means read after the block had been opened.

## The decision

A zombie stands next to the player. Staying put costs 2 health and moving away costs nothing; at low
health that is death. The consequence of the action is a change in **one tile**: the health counter,
HUD row 7, column 0. An imagined successor that serves a safe policy must show that change for the
fatal action and not for the others.

## Link 1 — the encoder's compact latent does not hold the zombie (why: JEPA's predictability bias)

- **CLS / projected z read at the action prior on zombie roots.** Measured on two sealed blocks
  (BOUNDARY.md, 51k and 53k) and again on 54k (`h2_bound.py`). The same encoder's patch tokens beat
  DOWN there (+0.075 to +0.126).
- **Visible-hazard decoding (reviewer, `20260925_h2_bound_challenge`).** Zombie adjacency from z has
  linear AUC 0.60, against 0.95 for lava; patch tokens decode both at about 1.00.
- **Why:** Littwin et al. 2024 (*How JEPA Avoids Noisy Features*, arXiv 2407.03475). JEPA's learning
  is greedy not only in feature variance λ but in predictability ρ (the regression coefficient across
  views). A zombie's position is driven by the game's random number generator, so it is a low-ρ
  feature; static lava is high-ρ. A SIGReg-JEPA CLS token keeps the predictable scene and drops the
  mob.
- **Consequence:** any world that consumes only z inherits this. On zombie roots, no head on H2's input
  beats DOWN (`h2_bound.py`: +0.022 [−0.034, +0.083]). That is weak headroom, not a proven ceiling.

## Link 2 — the transition learns the scroll and not the consequence (why: regression is greedy in variance)

**Given a patch-derived state (U), the imagined successor carries action-specific safety
information.** Sealed on 55k; replicated on fresh 56k at 2 seeds:

- U's generated state beats Z's on zombie roots (+0.07 to +0.14);
- shuffling successors within a root costs a probe 0.31–0.35.

**It still loses ~5 points of the zombie consequence against its own root** (56k, both seeds, resolved).

**Where the consequence lives** (`pooled_cells.py`, `transition_diag.py`):

- 66% of U's fatal direction lies in ONE pooled cell, (3,0), the one holding the health counter.
- That cell carries 3% of the action-effect energy; the map cells scroll and carry 88%.
- In U's PCA coordinates, 94% of the fatal direction lies in variance ranks 60–192, which carry 5% of
  the effect energy.

**What the transition gets right and wrong:**

- It predicts the top-10 components with effect R² 0.83 and ranks 100–192 with 0.23.
- Along the fatal direction it moves at near-real magnitude but correlates only ~0.2 with reality:
  **misaligned, not shrunk**. Generated within-root AUC along the direction is 0.61–0.64 (zombie
  roots 0.77–0.78), against 0.9997 on real successors.

**Why:** gradient-descent regression learns directions greedily in their variance λ (Littwin et al.
2024; the classic eigen-direction dynamics of linear regression). U spans a 2,314× variance range; the
var^−½ loss weights (48×) only partly equalize it. The rare, near-static health cell is the tail, and
it is learned last.

**Tests of this link, running:**
- `longer.py` (3× bridge training, sealed 57k): does the tail catch up?
- `whiten.py` (isotropic U, sealed 58k): does equalizing λ let it be learned at the rate of the scroll?

## Link 3 — decision heads must be trained where they are used

- **Canonical H2's continuation head is aliased to recursion depth** (`h2_alias.py`; the reviewer
  corrected the gate position). It reads 0.73 on the second generated step, which is always labelled
  dead in training, and 0.05 on a generated death at the evaluator position. Death versus 10 frames
  earlier: AUC 0.51.
- **A head fitted on real states and read on generated ones loses 0.09–0.11** of within-root choice on
  U (`genhead.py`). This is the training-distribution gap MuZero, EfficientZero and TD-MPC2 avoid by
  fitting every decision head on rolled-out latents.
- **With the alias removed and generated-suffix training, U's own head is as good as a head fitted
  purely on generated factual states** (0.656 vs 0.658). **Even a counterfactually supervised probe
  reaches only 0.672.** For U the head is not the binding constraint any more; Link 2 is.

## What is established vs open

| link | established | open |
|---|---|---|
| 1 encoder | CLS/z weakly encodes mobs, on sealed and exploratory blocks; patch tokens hold them | a JEPA encoder objective that keeps low-ρ, decision-relevant features (not attempted: the encoder is frozen) |
| 2 transition | the loss is spectral: the consequence is a rare change in a near-static HUD cell | does longer training or whitening close it? (running) |
| 3 heads | depth alias in H2; training-distribution gap | with the alias removed, U's head is near its state's ceiling |

## Prior art this maps to

- Littwin et al. 2024, *How JEPA Avoids Noisy Features* (arXiv 2407.03475): links 1 and 2.
- Klindt, LeCun & Balestriero 2026 (arXiv 2605.26379): LeJEPA's isotropic Gaussian as the identifiable
  optimum; the case for an isotropic (whitened) patch state.
- Voelcker et al. 2022, *VaGraM* (arXiv 2204.01464): weight model errors by the decision's gradient,
  because MSE spends capacity by L2 norm. The candidate after link 2's tests, and inside the factual
  contract.
- Schrittwieser et al. 2020 (MuZero), Ye et al. 2021 (EfficientZero), Hansen et al. 2024 (TD-MPC2):
  decision heads on rolled-out latents (link 3).
- Dedieu et al. 2025 (arXiv 2502.01591): per-tile categorical targets on Craftax-Classic. A categorical
  loss is blind to a tile's variance, which is a discrete answer to link 2 (its targets are pixel-patch
  codes, outside our reconstruction-free rule).
- Kim et al. 2026, *Identifiable Token Correspondence* (arXiv 2605.16457): copy-or-generate decoding
  for token persistence on Craftax-Classic. Related to the per-tile world's copied health tile
  (SPATIAL.md), not yet tested here.
