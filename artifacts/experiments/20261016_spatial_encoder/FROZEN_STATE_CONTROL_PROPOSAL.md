# Frozen-state predictor control — PROPOSAL (2026-10-17; not registered, not run; awaiting the user's decision)

## Why this experiment, now

R63 established, from existing checkpoints:
- **F1, erasure.** Every jointly trained Spatial world (MoP and full, 2 seeds each) keeps next-step mob information at copy level
  for its first ~1k updates (retention 0.80-1.02), then destroys it between ~1k and ~4k updates.
  - Zombie retention at 10k: 0.07-0.26.
  - The same predictor (architecture, loss, budget, data stream, seeds) trained on **frozen** Raw tokens never does
    (0.77-1.05 throughout).
- **F2, no consequences.** The Raw-predictor retains mobs yet predicts no attacks (recall ≤ 2%) and ranks escapes at chance.

In every joint run, the state space and the predictor change together. So R63 cannot say whether F1 comes from:
- **(a) co-adaptation:** the joint training dynamics. The patch targets are not detached, as in LeWM; online targets lose rank in
  Ni et al. 2024.
- **(b) the geometry of the projected (deployed) space** that joint training produced: projector + BN + 2048→192 compression.
- **(c) the patch-trained backbone itself.**

A predictor trained from scratch on a **frozen** Spatial encoder separates them at a fraction of the cost of encoder training:
- predictor-only training, ~75 min per run on the 6 GB GPU (Raw-predictor: 73 min per 10k);
- no encoder training at all.

**Is it still the most informative next experiment?** For F1, yes: it is the smallest design that separates (a), (b) and (c).

| Alternative | Why not first |
|---|---|
| Re-training the encoder with detached targets | Tests (a) directly, but costs ~3 h per seed and changes the objective. It is worth running only if this control implicates (a) — Stage B below. |
| An objective change for F2 | Is where the decisions are. But F2 is not diagnosed yet (see the end), and the procedure requires diagnosis before intervention. |

## Design

Encoders: the full arm's final checkpoints (s1, s2), frozen, in eval mode (BN running statistics, as deployed). They are the arm the
corrected Stage 1 selected.

| arm | input / target tokens | trained | exists? |
|---|---|---|---|
| **FD** | the frozen encoder's **deployed** tokens (projector output) | fresh predictor, 10k | new |
| **FB** | the frozen encoder's **backbone_ln** tokens (LN of last-layer patch tokens = Raw's interface) | fresh predictor, 10k | new |
| J | the jointly trained predictor of the same checkpoint | — | yes (R62) |
| Raw | the Raw-predictor of the same seed (= FB of the Raw encoder) | — | yes (R60) |

**Identical to the Raw-predictor protocol (`raw_predictor.py`):**
- PatchPredictor K = 4 + router; min-over-K MSE + 0.1 router CE;
- JointSampler seeded with seed + 1; `torch.manual_seed(seed)` before construction;
- batch 128; AdamW (d4mj grouping); Raw's 10k schedule; clip 1.0; bf16; checkpoints every 1k;
- seed = the encoder's seed: 20260731 for s1, 20260732 for s2. Data stream and predictor init therefore match J and Raw at the
  same seed.

**Code changes** (minimal, and none to running scripts):
1. `raw_predictor.py --encoder spatial:<ckpt> --stage deployed|backbone_ln` (backbone_ln = the same PatchState with the projector
   removed; ≈ 5 lines). The default stays Raw.
2. A stage-aware encoder spec for evaluation (`spatial-bb:<ckpt>`) in `spatial_eval.PatchState`, and backbone-stage readers saved
   by `spatial_eval_v2.py` (the a1 procedure, already computed, not yet saved).
3. `erasure_diagnosis.py` additionally saves per-unit test scores, so retention gets an episode-cluster bootstrap.

**Budget.** 4 runs × ~75 min ≈ 5 h GPU, sequential. Evaluation: about 10 min per world on GPU, with checkpoints at 1k, 2k, 4k, 10k.

**Staging** (cheapest decisive first):
- **A1:** FD s1, FD s2. Decides (a) vs (b + c).
- **A2:** FB s1, FB s2. Decides (b) vs (c), and level-1 parity with Raw at the world level.

A2 runs regardless of A1's outcome: if FD retains, FB tests whether the backbone state is also Raw-like for decisions.

## Readouts (to be registered before any run)

**Primary.** d1 zombie retention on the diagnosis test units (113 roots, 40 episodes; the R63 definition). Each comparison is a
paired episode-cluster bootstrap (4,000) of the retention difference, per seed:
- **FD − J:** does freezing the space, with a fresh predictor, remove the erasure?
- **FB − FD:** backbone vs projected state, same encoder, fresh predictors.
- **FB − Raw:** patch-trained backbone vs Raw, fresh predictors.

**Secondary (reported, no verdict):**
- cow retention;
- transfer AP of expected states (reader compatibility);
- dev-split retention (pseudo-labels);
- background (tile accuracy, camera-changed cells);
- trajectories at 1k / 2k / 4k / 10k;
- Tier B v2: B1 attack recall, B3 escape AUC, B4 census and localization through d16.

The central objective is decisions. B1 / B3 are reported for every arm, but this control is **not** expected to fix F2. Raw-like
retention is necessary, not sufficient.

**Pre-declared readings** (both seeds required; "≈" = the paired interval contains 0):

| finding | reading |
|---|---|
| FD − J > 0 and FD ≈ Raw | (a) **co-adaptation**: the frozen projected space is predictable without erasure; joint training learns to erase. Next: Stage B. |
| FD ≈ J (erases) and FB − FD > 0 | (b) **projected geometry**: the projector's space itself invites erasure; the backbone state does not. |
| FD and FB both erase (FB − Raw < 0) | (c) **the patch-trained backbone** differs from Raw in a way that invites erasure. |
| anything else | mixed, reported as is |

**Stage B (only if (a)):** retrain the full arm (2 seeds, identical everything) with **detached patch targets** (stop-gradient on
`tgt`), and with the CLS path unchanged, as LeWM. It is registered separately before running.
- It tests whether the gradient through the target is what teaches erasure.
- Ni et al. 2024 predict that detaching preserves rank; MuDreamer reports instability with a pure stop-gradient on representations
  (fixed by a small regularization), so both outcomes are informative.

## What it cannot establish
- Why the backbone, projected or joint state fails to carry **consequences** (F2): Raw-like worlds already fail decisions.
- Generality beyond the full arm. MoP FD replications (2 more runs) are optional if A1 is clear.
- Anything about longer context, rollout losses or other objectives.

## The next discriminating question after this (F2), for the whiteboard objective
The question: given a world that retains the mob (the Raw-predictor), why does it not predict the **consequence**, i.e. the health
loss from an attack or the escape ordering?

A cheap pre-check on existing checkpoints, before any intervention: refit a health-change reader on the Raw-predictor's predicted
HUD tokens (fit roots) and score attacks on the B1 units (test roots). Two outcomes:
- Information present but unread, i.e. off-manifold (like early joint mobs): a readout / compatibility problem.
- Absent: the objective never asked for it. Under uniform token MSE, consequence cells are rare (R55: ~9% of target change;
  E14: rarity under a uniform loss).
