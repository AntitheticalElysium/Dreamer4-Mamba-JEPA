# Matched D-class encoder comparison, MoP-s1 vs Raw-s1 (backbone-LN) — PREREGISTRATION

2026-10-17. Written and committed **before** any identity check, pool construction or training for this comparison. User-approved
design (`MATCHED_ENCODER_TEST_REVIEW.md`):
1. MoP-s1, not full-s1;
2. six-frame training windows, as the historical D run;
3. the existing Raw D checkpoints are reused, provided the identity and reproduction checks pass;
4. stages at 6k / 12k, with a conditional extension to 36k for mob motion.

## Objective (WHITEBOARD §19, unchanged)

*Does an encoder whose patch representations are trained for action-conditioned prediction give a better predictive world state
than frozen Raw?*

Operationally: with one capable downstream learner held fixed and only the frozen encoder changed, does the patch-trained encoder
improve spatial dynamics and decision-relevant consequences, separately from any change in static representation quality?

## Arms (only the encoder differs)

| | Raw arm (existing, reused) | MoP arm (new) |
|---|---|---|
| encoder | canonical Raw-s1 (`lewm_m4_canonical/raw/bridge/step-002000.pt` encoder; R0a checks it against `raw/joint/step-010000.pt`) | MoP + SIGReg seed 1 (`stage2_mop_sig_s1/step-010000.pt`), the same initial encoder tensors as Raw-s1 (Codex) |
| features | LN(last-layer 81 patch tokens) via `spatial.encode` | the same function on the MoP encoder (backbone-LN; never the projector) |
| pool | `spatial_pool_v1` (32,647 six-frame windows: 24,576 main + 8,071 terminal-ending) | `spatial_pool_mop_s1_v1`: the **same windows** (the stored `ids`), with actions / rewards / alive / dh / terminal copied; only `tokens` and `z` re-encoded |
| learner | `tworld.py` (sha256 2c31cce7…), `--head corrt --pool raw --loss teacher --seed 7 --updates 36000 --backbone fmamba --snapshots --state-every 2000 --recon deltairis` | the same file, unmodified (a launcher only registers the pool path), the same arguments with `--pool mop_s1` |
| exposure | snapshots at 6k, 12k, 18k, 24k, 30k, 36k | the same snapshots |

**Identical by construction:**
- predictor init (seed 7);
- held-out rows (seed 1);
- batch order (generator seed 11 over the same row count);
- optimizer (phase AdamW, 1e-4, 1k warmup then constant), clip, bf16, the D loss.

The learning rate is constant after warmup, so a 36k run stopped at 12k equals its first 12k.

**Known differences, not removable:**
- The encoders.
- The Raw arm ran on 2026-10-09; the MoP arm runs now, on the same machine and code.
- GPU nondeterminism (bounded by R0b).
- One downstream seed per arm.

## Checks before MoP training (abort, or report and stop, on failure)
- **R0a-i, encoder identity:** the encoder tensors of `raw/bridge/step-002000.pt` equal those of `raw/joint/step-010000.pt`
  bitwise. If not, the Raw arm's encoder is reported as the bridge encoder, and the comparison proceeds only if the MoP initial
  digest still matches that lineage.
- **R0a-ii, pool identity:** for 512 seeded windows of the stored pool, the corpus frames at the stored (episode, start):
  - re-encoded with the Raw arm's encoder, reproduce the stored tokens (max |Δ| ≤ 2e-3, fp16 rounding);
  - the stored actions equal the corpus actions exactly.
- **R0b, recipe reproduction:** the Raw recipe rerun for 1,000 updates (launcher, `--pool raw --updates 1000`) logs objectives within
  **5% relative** of the historical log (0.16536 @ 500, 0.08425 @ 1000).
- **R0c, MoP arm health** (at 6k, mechanical): finite objective; MoP-D one-step map error below copy on the evaluation units.

## Evaluation (identical for both arms at each budget)

**Common setup:**
- Units: all diagnosis roots (143 episodes); depth 1; 4 context frames (`location_dynamics.py` v2: 5,010 units;
  `consequence_check.py`: 17 actions per root).
- Readers: each encoder's own backbone-LN readers. Tier A (`spatial_eval_v2` a1 at the backbone stage) for transfer reads;
  5-fold episode-grouped cross-fitted ridge refits for everything else.
- AP: tie-grouped (`metrics_tied`).
- Intervals: paired episode-cluster bootstrap, 4,000 draws, the same draws for every endpoint at a budget.
- Code: `matched_encoder_analysis.py`, committed with this file.

**Representation (descriptive; never a decision endpoint):**
- R1: Tier A backbone-stage tile accuracy, zombie / cow AP, HUD R², for both encoders.
- R2: decoded-real ceilings on the evaluation units: refit AP of the real next frame's mob cells, by category; the within-root
  attack AUC of the real next frame.
- R3: copy (root-state) refit AP.

**Dynamics:**
- D1: camera-aligned persistence: shifted-camera stay AP of the expected state, zombie and cow.
- D2: static-camera stay AP; stay-vs-vacated AUC.
- D3: motion: zombie arrival AP and stay-vs-vacated AUC.

**Decisions:**
- C1: within-root attack AUC of the predicted state (HUD + 3×3-near-player refit reader; pair-weighted over roots with both
  attacking and non-attacking actions; units with ≥ 2 alive keys); within-root Pearson of predicted vs real Δh.

**Separating representation from dynamics.** Every dynamics endpoint is reported in two forms:
- **absolute**: AP of the expected state;
- **ceiling-normalized**: AP(expected) / AP(real next frame) on the same cell category, with the same encoder's readers.

Interpretation:

| absolute | normalized | reading |
|---|---|---|
| better | better | a dynamics gain beyond representation |
| better | not better | a representation (ceiling) gain |
| not better | better | a dynamics gain masked by a lower ceiling |

For C1 the real-next ceiling is reported (1.00 so far for Raw).

## Primary endpoints, gates and readings

| endpoint | quantity (MoP − Raw, paired) | primary budget | Raw gate (must pass at that budget, else no reading) |
|---|---|---|---|
| **E-C decisions** | C1 within-root attack AUC | **12k** | Raw C1 − action prior (COPY + action) > 0, lower bound |
| **E-P persistence** | D1 shifted-stay AP, mean over zombie and cow (absolute; the normalized form reported beside it) | **12k** | Raw D1 − reader-matched world persistence > 0, lower bound, for the mean |
| **E-M motion** | D3 zombie arrival AP (absolute; normalized beside it) | **36k** (conditional) | Raw arrival AP − screen copy > 0, lower bound |

Readings per endpoint (only when its gate passes):
- **"MoP better":** interval lower bound > 0.
- **"MoP worse":** upper bound < 0.
- **"equivalent within m":** the interval lies inside [−m, m], with m = 0.05 for AUC (E-C) and m = 20% of Raw's endpoint value
  for AP (E-P, E-M).
- **"no difference established":** otherwise.

Further reporting:
- 6k is reported for E-C and E-P as an early, secondary read.
- 6k and 12k are reported for E-M, descriptively; Raw does not pass motion there (calibration).
- All other quantities are descriptive.

## Continuation, decided at 12k (pre-declared)
- **Continue to 36k** (the motion endpoint E-M) unless at 12k MoP is **decisively worse on both** E-C and E-P: upper bound
  < −m on each, with Raw passing both gates.
- Also stop if the MoP arm is unhealthy (non-finite objective, or R0c failing at 12k).
- The run trains toward 36k with state saved every 2,000 updates. "Stop" means terminating it after the 12k snapshot; "continue"
  means letting it finish.

## Replication and scope
- **A "MoP better" reading on any primary endpoint** requires a second downstream seed for **both** arms before any robust claim.
  No D-loss seed-8 Raw run exists, so this would be a new pair, proposed separately.
- **Not tested:**
  - coherent stochastic samples (D is deterministic);
  - recursion beyond depth 1;
  - 16-frame context;
  - other encoder seeds (one paired encoder seed: a component result about this recipe, not about patch training in general).
- **Data:** the evaluation store was collected by the legacy BC policy, while the training pool comes from the PPO-expert archive
  (R59). Both arms share this.
