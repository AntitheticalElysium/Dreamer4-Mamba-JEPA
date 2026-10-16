# Frozen-state predictor control — PREREGISTRATION (2026-10-17; user-approved; written before any run of this control)

Approved by the user on 10-17, with these instructions:
- Frozen projected-state predictors first, both seeds; the frozen backbone only if needed to distinguish the remaining
  explanations.
- Matched training exposure; the existing corrected evaluations.
- Distinguish retaining input mobs from predicting their correct future locations.
- In parallel, a read-only check of whether Raw's predicted states contain attack-related health changes.
- No detached-target variant.

Design rationale: `FROZEN_STATE_CONTROL_PROPOSAL.md`. Evidence: NOTEBOOK R63.

**State of knowledge at registration.**
- No frozen-state predictor exists.
- The future-location decomposition and the consequence check below have not been computed for any world.
- Known (R63): retention trajectories of the joint (J) and Raw-predictor worlds.
- Counted before writing the rules, with no scores computed: event counts in the diagnosis store, given in the next section.

**Explicit non-claims** (user):
- Raw's retention is not successful dynamics: the Raw-predictor is a degraded copy, and its future-location metrics are measured
  below.
- Nothing here exonerates the predictive objective.

## Worlds

| world | encoder (frozen at 10k, eval mode) | predictor | status |
|---|---|---|---|
| **FD_s** (stage A) | full arm s ∈ {1, 2}, **deployed** tokens (projector output, BN running statistics) | fresh PatchPredictor, 10k | new |
| **FB_s** (stage B, conditional) | full arm s, **backbone_ln** tokens (LN of last-layer patch tokens) | fresh PatchPredictor, 10k | new, only if the stage-A rule says so |
| J_s | full arm s, deployed | its jointly trained predictor | exists (R62) |
| Raw_s | Raw@10000, LN tokens | Raw-predictor s | exists (R60) |

**Matched training exposure.** `raw_predictor.py` unchanged in substance; it gains `--encoder spatial:<ckpt>` and
`--stage deployed|backbone_ln`, with the defaults reproducing the Raw-predictor.
- PatchPredictor(K = 4); min-over-K MSE + 0.1 router CE; AdamW (d4mj grouping); Raw's 10,000-update schedule; clip 1.0; bf16 for
  the predictor; checkpoints every 1k.
- `torch.manual_seed(seed)` before construction, so the predictor init is identical to Raw_s.
  - **Correction (before any run):** J_s's predictor was constructed after its encoder and projector in the same seeded stream, so
    its init differs from FD's and Raw's. The data stream is identical for all three.
- JointSampler seeded with seed + 1: the **same windows in the same order** as J_s and Raw_s; seed = 20260731 (s1) / 20260732
  (s2).
- One known difference, inherent to freezing: J's predictor saw train-mode-BN deployed tokens (bf16 autocast), while FD's sees
  eval-mode tokens (fp32). Check C-BN below measures whether that alone explains J's erasure.

## Evaluations (existing corrected protocol, plus two registered additions)

**E1. Retention (primary for erasure).**
- `erasure_diagnosis.py` as in R63: depth-1 B4 setting; readers refit with λ 1e-2 on FIT units, scored on TEST units of
  `split_v2.json`.
- Retention = refit AP(expected) / refit AP(copy), zombie (primary) and cow (secondary).
- Also computed by 5-fold episode-grouped cross-fitting over **all** diagnosis roots (144 episodes, ~1,000 roots; each unit scored
  by readers fitted on the other folds) for power; reported beside it.
- Intervals: paired episode-cluster bootstrap (4,000), worlds paired on identical units. Trajectories at 1k / 2k / 4k / 10k.

**E2. Future locations (registered now, before any value is seen; co-primary, descriptive).** Retention says the input's mob
information survives; it does not say the mob is predicted where it will be.
- Per unit (root, alive future), the exact camera shift (dr, dc) (`objective_audit.shift_of` on exact tiles) aligns each next-frame
  cell with its world source cell in the root (`sources`).
- Next-frame cells are classed per mob type:
  - **stay**: the mob is at the same world cell before and after (static camera vs shifted camera reported apart);
  - **arrive**: a mob now occupies a world cell that had none (mob motion or spawn);
  - **enter**: the cell came into view;
  - **vacated**: the source held the mob, the next frame does not;
  - **empty**: everything else.
- Counts before any score, on the 393 split roots (all roots not yet counted, about 2.5× more): zombie stay 191, arrive 244,
  vacated 219; cow stay 448, arrive 377, enter 46, vacated 377.
- Metrics (cross-fitted refit scores on the expected state; transfer scores reported too):
  - **FL-arrive:** AP of arrive cells against all mob-free cells;
  - **FL-vacate:** AUC separating stay from vacated cells (does the world drop the mobs that leave?);
  - **FL-stay:** AP of stay cells, static and shifted camera separately (does the world carry retained mobs to their
    camera-transformed place?).
- Baselines on the same units:
  - **copy**: refit on the root state;
  - **world-aligned persistence**: the real-state reader on the root tokens, remapped by the exact camera shift, i.e. perfect
    persistence with no mob motion;
  - **LOO-frequency oracle**: for each cell, the mean exact presence over the root's *other* alive futures under the same
    action. This is the aleatoric reference an expected state can at most approach.
- **Reading, pre-declared:** a world "predicts mob motion" only if its FL-arrive AP exceeds world-aligned persistence with a paired
  interval above 0, at both seeds, for that mob type. Retention never counts as dynamics.

**E3. Existing corrected evaluations, unchanged.**
- `tierb_eval_v2.py` + `stage2_analysis_v2.py`:
  - B1 attack recall / Pearson;
  - B3 escape AUC;
  - B2 calibration;
  - B4 census through d16 vs decoded-real.
- `census_localization.py`.
- Comparisons: FD vs J (`--prefix-sp fd_s --prefix-raw full_s`) and FD vs Raw. Descriptive; the script's mechanical verdict field
  does not apply.

**C-BN (read-only check on J).** J_s's predictor evaluated on dev-split windows (the training distribution; pseudo-labels =
Raw's readers on the real next frame) under:
- the deployed projector in eval mode (as everywhere);
- the projector in **train-mode BN over training-shaped batches** (128 windows × 4 frames; no buffer updates kept).

If retention is restored under train mode, the J erasure is (partly) a BN-mode artefact, not co-adaptation. Reported before
stage A is read.

## Staging rule (stage A → B), pre-declared

On E1 zombie retention at 10k, cross-fitted over all roots (primary), with the split version as a check:
- **"Frozen projected states work":** at both seeds, FD − J > 0 (interval lower bound > 0) **and** FD − Raw does not lie below 0
  (upper bound ≥ 0).
  → No FB. Next: investigate **harmful joint adaptation**, read-only first (C-BN, and what changes between the J checkpoints at
  1k and 4k). No detached-target run until the user decides.
- **"Frozen projected states also erase":** at both seeds, FD − Raw < 0 (upper bound < 0) and FD − J ≤ 0 (lower bound ≤ 0).
  → Stage B: FB_1, FB_2, which isolates the **representation interface**:
  - **FB − FD > 0 and FB not below Raw:** the projected interface (projector + BN + compression) causes erasure.
  - **FB ≈ FD < Raw:** the patch-trained backbone state itself.
  - Anything else: mixed, reported as is.
- **Anything else (partial):** stage B also runs, because the interface question remains open.

## Read-only consequence check (in parallel; Raw primary, joint worlds as context)

The question: do Raw's predicted states contain attack-related health changes?
- **Units:** all diagnosis roots × 17 actions (alive keys), the B1 panel. 5-fold episode-grouped cross-fitting.
- **Target:** Δh = mean over alive keys of the next shown health − the root's shown health.
- **Attack unit:** at least one key loses ≥ 1 shown health. Counts before any score, split roots only: 964 events, 248 units, 21
  episodes; all roots give more.
- **Readers** (ridge λ 1e-2, cross-fitted), fitted separately on each state:
  - (i) the **HUD tokens** (18): does the predicted health display change?
  - (ii) HUD + the **3×3 map tokens around the player** (27): is attack-related evidence present near the player?
- **States:** predicted expected (Raw-pred s1 / s2 @ 10k; J full / MoP s1 / s2 @ 10k as context); copy (root); real next (the
  ceiling; per-key reads averaged).
- **Metrics:**
  - Pearson(predicted Δh, real Δh);
  - attack AUC (units ranked by predicted Δh);
  - mean predicted Δh at attack vs non-attack units;
  - the B1 transfer reading for reference.
- **Pre-declared readings:**
  - **"The predicted health display carries consequences":** (i) on the predicted state beats (i) on copy (attack-AUC paired
    interval > 0) at both seeds.
  - **"The input already announces them":** (ii) on copy, attack AUC interval > 0.5.
  - **"Carried but not rendered":** (ii) on predicted ≈ (ii) on copy, while (i) on predicted ≈ (i) on copy.
  - **"Destroyed":** (ii) on predicted < (ii) on copy.

## Limits
- One encoder family (the full arm), 2 seeds.
- The frozen arms learn on eval-mode tokens; C-BN addresses the train-mode difference only partially.
- Depth-1 measurements; recursion appears only through E3's B4.
- Pseudo-labels on dev windows.
- Sleep and night are not stratified here (Amendment 6 covers the readers).
