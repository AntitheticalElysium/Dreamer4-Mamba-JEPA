# Phase E protocol: task-head validation (the planner-readiness gate)

Status: **pre-registered before the terminal-enriched collection and before
any evaluation runs** (2026-07-18 sprint Stage B; tri-party directive with
companion amendments: planner is NO-GO until this gate passes). Evaluation
only — no training, no selection among architectures. The six committed
full-grid checkpoints (X-FLG/X-FLM x seeds 505/606/707) are evaluated as
FIXED artifacts; pooled M1 checkpoints are reference rows.

## Data

1. Teacher-forced + imagined-horizon windows: held-out episodes
   (data/heldout_20ep_v1.pt, never trained on), deterministic window
   sampling (seeded rng, identical windows for every checkpoint).
2. Counterfactual ranking: fork bundle 131-134 (committed, hash-pinned).
   Reuse status: spent for architecture selection; used here to evaluate
   fixed checkpoints' task heads, which gates the PLANNER, not architecture
   choice.
3. Continuation calibration: NEW terminal-enriched set, env seeds 900-915
   (outside every reserved/spent range, evaluation-only), random policy run
   to episode termination; windows crossing the terminal step plus matched
   non-terminal windows from the same episodes.

## Metrics (per checkpoint, per training seed — never ensemble-only)

- Teacher-forced real-prefix reward: NLL (two-hot), decoded MAE (symexp of
  the two-hot expectation), event AUROC (|decoded| scoring reward != 0).
- Imagined reward at horizons 1, 2, 4, 8 (observe 8-step real prefix, then
  imagine with the REAL action sequence): per-horizon NLL/MAE/event AUROC;
  8-step cumulative return Pearson/Spearman across windows.
- Counterfactual ranking (bundle): per anchor, imagine all four suffixes
  from cloned states; planner score J = sum_k gamma^k (prod_{j<k} c_hat_j)
  r_hat_k with gamma = 0.997 (DreamerV3 default; no extra survival bonus —
  continuation already gates). Raw-sum J recorded as a diagnostic. Report
  within-anchor Pearson/Spearman on reward-differing anchors,
  chosen-minus-random actual reward, regret vs best, env-seed-clustered
  bootstrap CIs, per training seed.
- Continuation on the terminal-enriched set: step-level Brier, AUROC, ECE
  (10 bins), plus the same on the (label-imbalanced) held-out windows for
  contrast.
- Day/night + task-effective strata on the bundle metrics.

## Pre-registered planner-readiness gate (PROPOSAL — margins subject to
## user+companion consensus before the gate is ACTED on; the evaluation
## itself runs regardless and reports every number)

For the sprint-candidate backend (full-grid Mamba-2), over its 3 seeds:

- G-E1 (ranking): chosen-minus-random actual reward advantage has the same
  sign in all 3 seeds AND the pooled env-seed-clustered 95% CI excludes
  zero.
- G-E2 (reward events): mean event AUROC >= 0.75 at horizon 1 and >= 0.65
  at horizon 8.
- G-E3 (continuation): terminal-enriched AUROC >= 0.80 and Brier <= 0.20
  (mean over seeds).

Planner (Stage C) GO requires all three for at least one backend; the
matched GRU control is judged by the identical rule. A failed gate routes to
task-supervision/imagined-state diagnosis (per companion: reward imbalance,
representation task sufficiency, imagined-state distribution shift), NOT to
architecture search.

## Non-goals

No reliability weighting in any score (shadow-only). No actor/critic. No
new training. Seeds 79-94, 115-130 untouched.

## OUTCOMES (appended 2026-07-18 after the evaluation; consensus on gate
## action pending)

PLANNER: NO-GO for BOTH backends under the proposal margins (and under both
G-E1 readings — the strict per-seed-CI implementation and the registered
pooled-CI text; the implementation-vs-text divergence is recorded in the
report and both are computed).

| gate | full-grid Mamba-2 | full-grid GRU |
|---|---|---|
| G-E1 ranking | FAIL (signs +/-/+; pooled adv +0.022, CI [-0.009,+0.047]) | FAIL narrowly (signs +/+/+; pooled adv +0.055, CI [-0.005,+0.115]) |
| G-E2 reward events | FAIL (h1 AUROC mean 0.52 — seed 707 at 0.05 is INVERTED; h8 0.74) | FAIL (h1 0.90 PASSES; h8 0.50 = chance) |
| G-E3 continuation | PASS (AUROC 0.93, Brier 0.020) | PASS (AUROC 0.92, Brier 0.028) |

Readings:
1. CONTINUATION HEADS WORK — first task-head gate the project has ever
   passed (terminal-enriched set, seeds 900-915; AUROC 0.83-0.95 across all
   nine evaluated checkpoints including pooled references).
2. The GRU control's reward head is strong at short horizon (h1 event AUROC
   0.85-0.95 per seed) and its ranking advantage is sign-consistent
   (+0.018/+0.097/+0.048) with pooled CI missing zero by 0.005 — close, not
   licensed. Its failure mode is long-horizon: h8 event AUROC 0.50.
3. The sprint candidate (Mamba) has the WEAKER task heads: erratic h1
   (0.73/0.05/inverted), inconsistent ranking signs — consistent with the
   4b/exploratory open-loop "smoother, less discriminative" diagnostic, and
   now visible in reward space, not just latent space.
4. Per protocol, failure routes to task-supervision/imagined-state
   diagnosis: candidate levers for consensus = reward-event class imbalance
   (event rate ~3-5% of steps), horizon-wise reward degradation (h1->h8),
   and imagined-state distribution shift; NOT architecture search.
