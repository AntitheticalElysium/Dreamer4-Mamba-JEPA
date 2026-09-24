# G2/G3 bridge-gate evaluator — signature plan

Written before code, as the `d4mj` contract requires. Placement: `d4mj/lewm_diagnostics.py`, beside
`screen_joint_pair`, because this is the M4 analogue of G1 and belongs in the package rather than
under `artifacts/` — only `d4mj/*.py` is pinned into a checkpoint's source manifest.

## What the gate boundary demands

`gates.require_bridge_gate` accepts only a `d4mj_lewm_bridge_gate_v1` whose `report_id` is
`contract_digest(body)`, whose `checkpoint_sha256` / `recipe_id` / `cache_id` / `stage` / `decision`
match the live model, whose `validated_recursive_depth` is an int ≥ the stage minimum, and whose
**seven** components each carry a non-empty `metrics` dict and at least one `evidence` entry whose
file still hashes to the recorded digest.

## Functions

| function | in → out |
|---|---|
| `bridge_gate(run, settings, output, *, stage)` | orchestrates; writes evidence; seals the report |
| `_gate_traces(bundle, episodes, settings, depth)` | held-out DEV traces encoded once: latents, actions, labels, episode clusters |
| `_recursive_dynamics(bundle, traces, depths)` | rollout error per depth vs **persistence** and **marginal-action** baselines |
| `_action_effects(bundle, traces)` | predicted effect vs real change; action sensitivity by derangement; no-op equivalence |
| `_outcome_calibration(heads, traces)` | reward error vs zero/marginal; continuation Brier/BCE vs the `log(2)` balanced reference, split live/dead |
| `_observed_bc(heads, bundle, traces)` | observed-path BC agreement on the relevant half, with its own recurrent state |

`semantic_retention` reuses the existing `screen_retention` (projected vs CLS, paired AUC interval,
`auc_margin = .03` — exactly the spec's margin). `source_contract` reuses the identity checks the
loader already performs, recorded as measurements rather than re-derived.
`paired_uncertainty` is the episode-clustered bootstrap over the above contrasts, via the existing
`paired_auc_interval` and a matching interval for the error contrasts.

## Pass rules, from `EVALUATION.md` §G3

> Permit the H16 bridge only when H2 meets source contracts, retention, valid action-effect
> improvement over persistence/marginal controls, and outcome calibration better than the declared
> trivial baselines with paired uncertainty.

- `recursive_dynamics` / `action_effects` — pass iff the rollout beats **both** persistence and the
  marginal-action baseline, with the paired interval excluding zero.
- `outcome_calibration` — pass iff reward error beats zero/marginal and balanced terminal BCE beats
  `log(2) = 0.6931`, classwise **and** aggregate. An always-continue head must fail.
- `semantic_retention` — pass iff projected is noninferior to CLS within `.03` macro-AUC.
- `observed_bc` — recorded at H2; the `.5`-achievement noninferiority comparison belongs to G4,
  so at this stage BC is reported and gated only on being measurable, not on a margin.

**Failing closed.** A component with insufficient support reports `insufficient_coverage` and does
**not** pass. Silence, absent labels or an empty stratum never authorize H16.

## Declared scope — what this gate does NOT measure

`EVALUATION.md` §G3 also asks for stochastic successor mode fidelity over repeated simulator seeds,
persistent-memory utility under varied earlier context, and decoded-tile comparisons. Those need a
simulator-fork harness and a renderer, are not part of the seven gate components, and are **not**
implemented here. The report records them as `not_evaluated` so nothing claims coverage it lacks.
