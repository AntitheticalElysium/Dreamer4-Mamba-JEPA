# The three gates — current contract

> Superseded twice by audit. This reflects `d4mj/lewm_diagnostics.py` as of the gate-contract
> repairs, not the original design. **The gate is not yet approved for launch** — see "Still
> missing" below.

Code: `d4mj/lewm_diagnostics.py` (`bridge_gate`, `actor_gate`). Boundary: `d4mj/gates.py`.
Invoke: `python -m d4mj gate --run <arm> --stage h2|h16|actor --dataset <stores>`.
The evaluation recipe is **pinned in code** (`d4mj/recipes/joint_screen.json`); the CLI no longer
accepts a custom one, and the boundary refuses a report measured with anything else.

## Sampling

DEV only. `_gate_traces` builds a DEV subset, asserts every drawn episode is in it, raises if a
FINAL episode appears, and seals the exact episode/start ledger into the report. Cluster identities
are SHA-derived, not `hash()`, so bootstrap groups are stable across processes.

## Bridge gate — 7 components

| component | passes iff |
|---|---|
| `source_contract` | frozen encoder, predictor BN buffers in eval, cache/parent bound |
| `semantic_retention` | every probe family resolves an interval **and** every **lower** bound exceeds `-0.03` |
| `recursive_dynamics` | beats persistence **and** the action-blind mean at **every** evaluated depth, intervals excluding zero |
| `action_effects` | `R² > 0` **and** beats the action-blind mean **and** re-labelling the action costs error |
| `outcome_calibration` | **on generated states**: reward beats zero and marginal; balanced terminal BCE `< log(2)` classwise and aggregate |
| `observed_bc` | observed **and** generated top-1 both beat the most-frequent action |
| `paired_uncertainty` | **every** decision-bearing contrast resolves an interval |

Depths: H2 `(1,2)`, H16 `(1,2,4,8,16)`. `validated_recursive_depth` = largest depth clearing both
baselines.

## Actor gate — 4 components

`model_validity` (still beats persistence, world frozen), `critic_direction` (start value vs that
row's **own recorded** discounted return — the λ-return correlation is retained but labelled
self-consistent), `action_distribution` (no action above 95% share **and** at least `n_actions // 4`
distinct actions used), `paired_uncertainty`.

## Forgery resistance

The boundary computes each verdict from `metrics["failed_checks"]` with the threshold **in code**.
A report-supplied `criterion` is descriptive only. Editing status, or status and criterion
together, is refused. Tests cover: flipped status, forged criterion, missing measurements, custom
evaluation recipe, re-digested tampering, moved evidence bytes, and shallow sealed depth.

## Still missing — why this is not yet approved

- **No all-action DEV fork gate.** Global trivial baselines permit state-independent action
  knowledge: observed agreement 80%, generated 10%, marginal 5% would pass both. Needs
  within-root regret, opportunity strata, equivalence classes, and BC-relative policy-weighted
  death. `d4mj/counterfactual.py` has this machinery for the legacy world types.
- **No critical retention panel.** Still projected `z` vs its own CLS on four labels; no
  preselected reference, no health/inventory/tiles/prerequisites. Addresses exist in
  `artifacts/experiments/20260918_m03_probe_coverage_audit/`.
- **Evidence holds aggregates, not raw rows.**
- **Terminal-safety in the G4 verdict is an aggregate rate**, which is neutral when both policies
  eventually die. Needs the policy-weighted death comparison above.
- The recursive action-blind reference averages recurrent state at each step. It is a named
  diagnostic, **not** the exact conditional mean over action sequences.
