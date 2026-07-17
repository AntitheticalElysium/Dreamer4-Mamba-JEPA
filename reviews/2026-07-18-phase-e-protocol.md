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
