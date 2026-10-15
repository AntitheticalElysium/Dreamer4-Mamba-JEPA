# Does a patch-trained JEPA encoder beat the frozen CLS-trained Raw encoder for a predictive world state? Preregistration

2026-10-16. Written before any code for this study exists. **Stage 1** (screening) is fully specified here. **Stage 2**
(the main comparison) is specified here in its design and readings; its arm is selected by Stage 1's rule below.

## The assumption under test

Every world we built since 09-25 (D, CorrT, IC, PI) uses the frozen Raw encoder's last-layer patch tokens as state and
targets. That encoder (`artifacts/lewm_m4_canonical/raw/joint/step-010000.pt`, recipe `d4mj/recipes/lewm_mamba_raw.json`) was
trained only through projector(CLS):
- next-z MSE + SIGReg (`d4mj/lewm.py` `joint_loss`);
- the patch tokens receive no gradient ("never regularized by SIGReg and never predicted", `lewm.py:77-79`).

**Measured costs of that choice (NOTEBOOK R55):**
- the token target is ~64% nuisance drift, ~27% unseeable terrain, ~9% consequences;
- a HUD value change is smaller than lighting drift on an unchanged tile;
- discrete codes flip on 25-46% of persisting tiles per step;
- the L2 optimum at branching mob cells is a ghost (67% vs 40% mob reading).

The question is whether an encoder whose **patch tokens are themselves trained for action-conditioned prediction** gives a
better predictive world state for imagination and decisions.

## Design principles (each tied to a known failure and its evidence)

| known failure | design choice | evidence |
|---|---|---|
| patch tokens untrained; global CLS loses spatial detail | train the patch tokens directly with an action-conditioned spatial predictor; keep Raw's CLS objective unchanged, so the new encoder's objective is a strict superset of Raw's and the comparison isolates "add patch-level training" | DINO-WM patch vs CLS (PushT 0.90 vs 0.44); PLDM's spatial latent; our CLS world lost mobs (09-26) |
| L2 to a branching future = conditional mean (ghosts) | per-patch **hard-assigned K-head predictor + router** (winner-take-all), not a single regression head | MoP-JEPA proof and results; our R55 ghost measurement |
| collapse (global and per patch) | anti-collapse on the **patch** embeddings; SIGReg-on-patches vs VICReg variance + covariance on patches is screened (Stage 1) | LeWM / LeJEPA (SIGReg); PLDM ablation (variance / covariance removal → 11-29% success) |
| SIGReg's isotropic Gaussian vs nearly categorical patch content (6 tile classes = 97% of map patches; mobs 1.4%): a Gaussian marginal may only be reachable by encoding nuisance | screen SIGReg against VICReg, which a set of separated content clusters can satisfy | measured content shares (10-16); Klindt et al. (Gaussian-only identifiability); AC-MTM (2608.17542): prescribed geometry a bottleneck on multi-object scenes (80.0 vs 58.0) |
| lighting / sleep / night static dominate the target | **photometric multi-view invariance on map patches**: each frame is also encoded under a random per-pixel colour map (R49's generic family) plus, with probability 0.5, additive Gaussian pixel noise (σ ~ U(0, 0.05)) on the map rows. L_inv pulls the augmented view's map-patch embeddings to the clean view's; prediction targets stay clean. The HUD rows are never augmented (the game never relights them) | R2 / R42 / R43 (rendering is the copy harm and is not re-renderable); R47 (appearance is a per-pixel colour map in the source); LeVJEPA / LeJEPA multi-view invariance |
| action-caused consequences are rare in the loss (inventory, tile edits) | **inverse dynamics** (17-way CE on the action from pooled patch transitions [z_t, z_t+1]): dynamics-native pressure to keep what actions change. 40% of transitions show no visible action effect, so this is an auxiliary term, not the anti-collapse | AC-MTM; PLDM IDM ablation (98.7 → 75.5); measured action visibility (10-16) |
| environment-caused rare consequences (zombie damage) | not specially weighted (generic doses failed in E14). The HUD is predicted per patch with the same per-patch weight as the map; whether its digits stay encoded is a primary *reading*, not an assumption | E14; R48 |

**Architecture (matched to Raw where it matters):**
- **Encoder:** ViT-12, width 192, patch 7, 63×63 input, 3 heads (Raw's exact config); fresh initialization.
- **CLS path:** Raw's projector + 6-layer Mamba world + next-z MSE + SIGReg (0.09), unchanged.
- **Patch path:**
  - the patch state is a per-patch projector of the last-layer patch tokens: Raw's projector class, i.e. Linear → BatchNorm
    → GELU → Linear, 192. LeWM states the final LN prevents anti-collapse from acting, which is why its CLS uses a projector;
  - a spatial predictor, a 4-layer pre-LN transformer, width 192, over [z_{t-1} (81), z_t (81), action token] with learned
    frame / position embeddings. It outputs K = 4 candidates per patch for z_{t+1} plus per-patch router logits;
  - predictions are made for the 2 transitions of each 4-frame window that have a previous frame;
  - the inverse-dynamics head concatenates [z_t, z_{t+1}] per patch, a linear layer to 64, mean-pools over the 81 patches,
    then an MLP to 17 logits, for the 3 transitions of each window.
- **Data and schedule (Raw's):** same corpus (`craftax_expert_store_v1` + `craftax_support_v2`, train split), 4-frame
  windows, AdamW, warmup 500, cosine to 10% of the learning rate, grad clip 1.0, bf16.
  - Batch 64 windows (Raw: 128). The patch path needs the memory; the 6 GB budget is measured in a smoke test before
    launch.

**Loss:**

L = L_CLS (Raw's) + L_patch + λ_ac · AC(patches) + λ_inv · L_inv + λ_idm · L_idm

- **L_patch:**
  - mean over patches and predicted steps of min_k ||ĝ_k − sg?(z_{t+1})||²;
  - the target is the same encoder, no stop-gradient (LeWM's choice, kept for comparability);
  - plus λ_r · CE(router, argmin k).
- **L_inv:** per-map-patch MSE between the encoder outputs of two independently augmented views of the same frame.
- **AC:** SIGReg over all patch embeddings of the batch (samples = B·T·81), or VICReg variance (hinge at 1) + covariance
  over the same samples.

Coefficients are fixed a priori: λ_r 0.1, λ_inv 1, λ_idm 0.1, λ_ac as in their sources (SIGReg 0.09 as LeWM; VICReg
variance 25 / covariance 1 as VICReg's defaults relative to an invariance weight of 25, here scaled so the variance term
weight is 1 and covariance 0.04). No tuning is done on evaluation numbers.

## Stage 1 — screening (2,000 updates each, seed 1; Raw's own step-2000 checkpoint is the matched reference)

| arm | L_patch | anti-collapse on patches | L_inv | L_idm |
|---|---|---|---|---|
| S-naive | K = 1 (plain per-patch MSE) | SIGReg | no | no |
| S-mop-sig | K = 4 + router | SIGReg | no | no |
| S-mop-vic | K = 4 + router | VICReg | no | no |
| S-full | K = 4 + router | the better of SIGReg / VICReg by the rule below | yes | yes |

S-full is run after the first three, using their result for its anti-collapse choice (rule below).

**Screening readings** (frozen encoder at step 2000; label-based linear probes are fitted for evaluation only on
diagnosis-store frames, seeds disjoint between fit and test; identical protocol for every encoder including Raw@2000):
- **A1 content:** per-patch tile accuracy, zombie / cow AUC, HUD stats R² (health, food, drink, energy) from the 18 HUD patches.
- **A2 nuisance:**
  - persisting-content drift / between-content distance, by light regime (day steady, day changing, night);
  - the k-means (K = 1024, fitted per encoder) code flip rate on persisting content;
  - HUD salience = code change rate when a displayed value changed minus when none did.
- **A3 collapse:** effective rank of patch embeddings (exp entropy of normalized singular values); per-dimension std.

**Selection rule (declared now):**
- An arm is eligible if it does not collapse (effective rank ≥ 20 and tile accuracy ≥ Raw@2000 − 0.05).
- Among SIGReg and VICReg (S-mop-sig vs S-mop-vic), choose the one with the lower day-changing drift ratio, unless its tile
  accuracy is more than 0.02 below the other's.
- **Stage 2 arm:** S-full, if it is eligible and its drift ratio and HUD R² are not worse than the chosen MoP arm's (drift
  within +0.02, HUD R² within −0.02). Otherwise the chosen MoP arm.

## Stage 2 — main comparison (specified now; run after Stage 1)

- **Arms:**
  - **Spatial**: the selected arm, trained 10,000 updates at 2 seeds;
  - **Raw**: the frozen canonical encoder, the existing checkpoint;
  - **Raw-predictor**: the *same* patch predictor (K = 4 + router, same loss minus the encoder terms) trained on frozen Raw
    tokens for 10,000 updates at 2 seeds.
- The world compared is "encoder + patch predictor".
- **Tier A:** the Stage-1 readings at 10k.
- **Tier B (decisions; readers are the same linear probes, fitted per encoder on real tokens):**
  - **B1 one-step consequence:** on the diagnosis store (4 keys × 17 actions), predicted vs real health change at real
    attacks. Real attack = displayed health drops; recall of a predicted drop ≥ 0.5, magnitude ratio.
  - **B2 branching:** at mob cells whose layer differs across the 4 keys, the mob-reading rate of a router-sampled head vs
    the real rate (no ghosts = close to real), and the router's calibration against the 4-key frequencies.
  - **B3 escape choice:** on DV16's 200 roots, the within-root AUC of the predicted first-step health loss over noop + 4
    moves against the real P16 order (R53's quantity; real one-future 0.71, D 0.711).
  - **B4 recursion:** 16-step rollouts (router sampling), the zombie / cow census and adjacency kept vs real (R53's
    quantities).
- **Primary reading:**
  - **"The patch-trained encoder gives a better predictive world state"** if Spatial beats Raw-predictor on B1 recall and on
    B3 at both seeds (paired bootstrap over roots / events, intervals excluding 0), without being worse on B2 / B4 by more
    than their noise.
  - **"Not supported"** if it is no better on B1 and B3.
  - Anything else is reported as is.

## Limits and honesty clauses

- Stage 1 is one seed and short; it selects, it does not conclude.
- The predictor has two frames of context: no long memory. That is equal for both encoders; long-memory dynamics are a
  later question.
- Linear probes are fitted with simulator labels for evaluation only. No label enters any training.
- Fresh confirmation data stays sealed; the judge block is not read.

## Amendment 1 (2026-10-16, before any result; memory)

- SIGReg over all B·81 = 10,368 patch samples per frame index needs about 2.7 GB for its projections, and fails on the 6 GB GPU.
- It uses **1,024 random patches per frame index** instead.
- Its statistic is the Epps-Pulley test statistic multiplied by the sample count (`d4mj/lewm.py:132`). The weight is therefore
  **0.09 · 128 / 1,024**, which keeps LeWM's per-discrepancy strength at its batch of 128. Using all patches at 0.09 would
  have made it about 81× stronger.
- The batch is Raw's 128 windows (the recipe validator requires it). The earlier "batch 64" is withdrawn.

## Amendment 2 (before any result)

- 1,024 patches still exceed the GPU when combined with the full arm's doubled encoder pass (backward at 4.9 GB).
- Every SIGReg arm uses **512 random patches per frame index, weight 0.09 · 128 / 512**.
