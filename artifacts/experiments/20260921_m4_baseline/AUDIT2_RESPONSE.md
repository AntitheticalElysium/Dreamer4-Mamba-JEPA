# Response to the no-go audit

**No finding denied.** Both release-blockers reproduced exactly as described.

## Reproduced before fixing

**Split leakage.** `_gate_traces` received the whole latent cache and sampled it unfiltered, while
training subsets to `split == "train"` (train.py:906, 1040). The gate scored the model partly on
its own training data and opened FINAL.

**Wrong bound.** `interval[1]` is the *upper* bound. With `[-0.20, +0.01]` and `[-0.10, +0.02]` the
rule returned `pass, noninferior=True` — intervals permitting a 0.20 AUC loss. My commit said
noninferiority was now "positively established". **That was false and is retracted.**

## Fixed

| # | finding | fix |
|---|---|---|
| 1 | TRAIN/FINAL sampled | DEV-only subset built in `_gate_traces`; membership asserted; FINAL raises; exact episode/start ledger persisted and sealed into the report |
| 2 | upper-bound noninferiority | now every probe family's **lower** bound must exceed `-auc_margin` |
| 3 | generated semantics ungated | `_head_readouts` now evaluates reward, continuation and policy on **generated** states at the same positions; `outcome_calibration` and `observed_bc` gate on the generated path, with observed reported beside it and the observed→generated transfer interval recorded |
| 4 | derangement as "marginal"; BC never fails; one interval sufficed | true action-blind control = conditional mean over all actions at every step (derangement kept, clearly labelled, as sensitivity only); `observed_bc` binding on both observed and generated vs most-frequent; `paired_uncertainty` requires **every** decision-bearing contrast to resolve |
| 5 | circular critic screen | critic now correlated against the **real recorded return** on the DEV trajectory, which it never saw; the λ-return correlation is retained and explicitly labelled self-consistent. Action collapse now also fails below `n_actions // 4` distinct actions |
| 6 | status strings trusted | every component declares a `criterion` (quantity, value, threshold, direction) and `_sealed_phase_report` **recomputes** status from it; a flipped status with failing numbers is refused even after re-digesting. Evaluation recipe, batch count, seed and sample ledger sealed; a reused evidence directory raises |
| 7 | no enforced G4 verdict | `evaluation.json` carries a `verdict`: positive achievement lower bound **and** no score decrease **and** no terminal collapse; the driver exits 3 when neither arm passes |

## NOT fixed — the gate measures less than G3/G4 specify

These are coverage gaps, not invalid-green defects, but they mean a pass is weaker than the spec's:

- **Retention still compares projected `z` against its own CLS only.** No preselected reference,
  and the labels remain reward-positive / reward-negative / event / termination. Health,
  inventory/resources, local tiles and action prerequisites are absent (§G2).
- **No held-out all-action simulator forks.** `action_effects` has no within-root selection,
  opportunity strata, effect-equivalence classes or consequent action regret (§G3).
- **No true-successor / oracle substitutions and no BC-relative safety strata** in the actor screen
  (§G4). The critic check is now external but still coarse.
- **Evidence files hold aggregate metrics, not raw rows** sufficient to recompute them.
- The recursive action-blind reference averages over actions at **every** step. That is a defensible
  recursive definition, but it is one I chose; the spec does not fix an exact H16 marginal over
  action *sequences*.

## Status

Suite **312 passed, 4 skipped, 0 failed**. Closure re-sealed. Still not relaunched.
