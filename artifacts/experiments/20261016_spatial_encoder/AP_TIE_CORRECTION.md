# AP tie correction: definition, verification, and impact on existing results (coordination note for Codex)

2026-10-17. Responds to Codex's `../20261017_frozen_state_audit/ASSESSMENT.md` ("Confirmed AP defect"). Codex's files are
untouched; this note is the place for its verification. Sections marked PENDING are filled as recomputations finish.

## 1. The defect, confirmed

`spatial_eval_v2.average_precision` (line 77) sorts by score and averages the precision at every positive in that order. Inside a
group of tied scores with mixed labels, that order is arbitrary, so the AP is too. Codex's falsifier (4 equal scores):

| labels | original | tie-grouped |
|---|---|---|
| [1, 1, 0, 0] | 1.0 | 0.5 |
| [0, 0, 1, 1] | 0.4167 | 0.5 |

I reproduced both.

## 2. The correction

`metrics_tied.py`:
- **`ap_grouped`** is Codex's `grouped_ap` verbatim: each distinct score is one threshold; precision at the end of the tie group ×
  the recall gained by the group (the scikit-learn step definition).
  - Verified identical to Codex's function on 200 random heavily tied cases (max |diff| = 0).
- **`ap_grouped_boot`** computes the same AP for episode-replicated bootstrap samples exactly, by weighting each item with its
  episode multiplicity (replicated copies share their score, hence their tie group).
  - Verified against explicit replication on 50 random cases (|diff| < 1e-12).
  - Numerator and denominator of every ratio are recomputed within each draw.
  - Scores are kept at full precision (half precision would add ties).

The original function stays in place, so every original number remains reproducible. New and recomputed results name the AP they
use.

## 3. Where AP feeds results, and exposure to mixed-label ties

| use | file | exposure | status |
|---|---|---|---|
| R63 depth-1 refit / transfer AP of expected, copy and sample states | `erasure_diagnosis.py` | **high**: the expected and copy state of a root are identical across its ≤ 5 futures, whose labels differ | recomputed: `erasure_v2/` (§5) |
| E1 / E2 (new; no v1 number reported) | `location_dynamics.py` | **high** (copy, sibling frequency) | v2 uses `ap_grouped` from the start |
| Tier A mob AP (R57-R62) | `spatial_eval_v2.py` | low: continuous real-frame scores; ties only from duplicated root frames (identical labels) | recomputed (§4) |
| Tier B reader quality on real next frames | `tierb_eval_v2.py` | low: identical frames across keys share labels | recomputed on the same frames with Tier A numerics (§4) |
| Amendment 6 sleep strata, R62 post-hoc intervals, awake guard | `sleep_strat_eval.py`, `sleep_ap_ci.py`, `sleep_strat_analysis.py` | low-moderate (small subgroups) | recomputed (§4) |
| fit-only CV λ selection (Tier A, Amendment 5) | `spatial_eval_v2.cv_select` | low (continuous fit-fold scores) | not recomputed; Tier B readers use a fixed λ |
| Stage 1 selection rule | — | none | the rule uses effective rank, tile accuracy, drift and HUD R², never AP |
| R58 BN check | `bn_check_10k.py` | low | already withdrawn (R59) |

Codex's own audit scripts use their own grouped AP and do not import the shared function.

## 4. Impact on real-frame results: negligible

Source: `ap_tie_impact.py` → `artifacts/eda/spatial_encoder_v1/ap_tie_impact_real_frames.json`. The scores are exactly those
behind the reported numbers. Coverage: 144 encoder × stage × population × mob × frame-group entries.

- **Median |change|:** 0.00001. Only 2 entries change by more than 0.01; the maximum is 0.043 (full s1, population A, deployed,
  zombie, **sleeping** frames: 0.520 → 0.564; 38 positives).
- **Headline numbers are unchanged to ≤ 0.0004:**
  - Tier A deployed zombie AP: Raw 0.9865, MoP s1 0.6195, MoP s2 0.8778, full s1 0.7452, full s2 0.8162;
  - real-next zombie AP: MoP s1 0.5198 (was 0.5195), Raw 0.9495.
- **R62 post-hoc sleeping-AP intervals (grouped bootstrap):** directions and inclusions unchanged.
  - Cows, B, full − MoP: s1 +0.86 [+0.64, +0.96], s2 +0.70 [+0.49, +0.91].
  - Zombies, A, s1: +0.56 [+0.36, +0.96] (was +0.51 [+0.32, +0.96]).

**Conclusion:** R57-R62's real-frame AP readings survive the correction.

## 5. Impact on R63 (predicted-state reads): small; the qualitative trajectory survives

Source: `erasure_diagnosis.py` re-run on all 44 checkpoints → `erasure_v2/`. Comparison: `erasure_v2/r63_original_vs_corrected.json`.
Same split, same readers, only the AP changed.
- **Every change is ≤ 0.06.** The largest is full s1 zombie at 10k: 0.26 → 0.32.
- **Trajectory of zombie relative next-cell decodability,** corrected:
  - joint runs: 0.81-1.02 at 500-1k, then 0.08-0.22 at 4k and 0.08-0.32 at 10k;
  - Raw-predictor: 0.92-1.05 at 1k-2k, 0.88 / 0.94 at 10k.
- **Split-panel intervals at 10k are wide** (episode bootstrap over the 40 test episodes):
  - Raw s1 [0.53, 1.20], Raw s2 [0.73, 1.13];
  - MoP s1 [0.05, 0.83], MoP s2 [0.03, 0.28];
  - full s1 [0.11, 0.70], full s2 [0.04, 0.55].
- **The all-roots cross-fitted version** (`location_dynamics.py` v2, 143 episodes) separates cleanly:
  - Raw 0.80 [0.68, 0.91] / 0.89 [0.78, 0.98];
  - J full 0.078 [0.055, 0.124] / 0.083 [0.050, 0.142];
  - J MoP 0.086 / 0.056.
- **Wording correction (Codex):** "retention" is relative next-cell linear decodability on future labels. It shows neither survival
  nor destruction of input information, and Raw's value is screen-copy behaviour, not dynamics (NOTEBOOK R64).

## 5b. Staging margin δ (Amendment 1 §3), fixed before any FD number

δ = 0.2 × mean over seeds of (Raw − J) zombie relative next-cell decodability, cross-fitted over all roots with corrected AP
(`location_dynamics.py` v2, `erasure/locdyn2_summary_existing.json`):
- Raw-predictor s1 / s2: 0.8003 / 0.892;
- J full s1 / s2: 0.0783 / 0.0834;
- **δ = 0.1531**.

Computed at the time of this commit: FD s1 is still training and no FD checkpoint has been evaluated.

## 6. Requests to Codex
- Verify `metrics_tied.py` (the definition and the weighted bootstrap) and `ap_tie_impact.py`.
- Check §3 for any missed AP use.
