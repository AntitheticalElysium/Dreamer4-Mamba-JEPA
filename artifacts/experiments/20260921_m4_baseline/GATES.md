# The three gates — frozen contract

Code: `d4mj/lewm_diagnostics.py`. Boundary: `d4mj/gates.py`.
Invoke: `python -m d4mj gate --run <arm> --stage h2|h16|actor --dataset <stores>`.

## Sampling and identity

- **DEV only.** `_gate_traces` builds a DEV subset, asserts membership, raises on FINAL, and seals
  the exact episode/start ledger into the report.
- **Stable clusters.** SHA-derived, not `hash()`, so bootstrap groups do not move between processes.
- **Pinned evaluation recipe.** `d4mj/recipes/joint_screen.json` is fixed in code; the CLI takes no
  `--screen-recipe` and the boundary refuses a report measured with anything else.
- **Derived verdicts.** The boundary computes each component from `metrics["failed_checks"]` with
  the threshold in code. A report-supplied `criterion` is descriptive only. Editing status, or
  status and criterion together, is refused.
- **Immutable evidence.** A directory already holding evidence raises; every component writes raw
  per-row inputs (`*.rows.pt`) beside its aggregates so results can be recomputed.

## Bridge gate — 7 components

| component | passes iff |
|---|---|
| `source_contract` | frozen encoder, predictor BN buffers in eval, cache/parent bound |
| `semantic_retention` | **critical panel**: 129 verified addresses, 25 `STATIC_BINARY` predicates (front tiles, near table/furnace, mobs, and every action prerequisite), 21 supported. Projected `z` noninferior to **CLS and the preselected old export**, every probe family's **lower** 95% bound above `-0.03` |
| `recursive_dynamics` | beats persistence **and** the action-blind mean at **every** depth, intervals excluding zero |
| `action_effects` | latent effects positive, action-sensitive, beats the action-blind mean — **and** the all-action fork gate below |
| `outcome_calibration` | **on generated states**: reward beats zero and marginal; balanced terminal BCE `< log(2)` classwise and aggregate |
| `observed_bc` | observed **and** generated top-1 both beat the most-frequent action |
| `paired_uncertainty` | **every** decision-bearing contrast resolves an interval |

### The all-action fork gate (inside `action_effects`)

Held-out roots from `broad_forks_v2`, seeds 15000–16504 — **disjoint from the expert archive
(0–319), support-v2 (20270731+) and the sealed M03 seeds (13000–14511)**, verified. 512 roots,
all 17 actions from each, with real outcomes. Every contrast is **within-root**, so a model that
has only learned which actions are good on average cannot pass — that is exactly the
action-marginal control it is measured against.

- **reward regret**: within-root argmax choice vs the action-marginal choice, paired interval
- **terminal safe choice** and death AUC on opportunity roots
- **true-successor substitution** throughout, separating transition error from outcome-head error
- **policy-weighted true death**, actor vs its own immutable BC
- **effect-equivalence**: actions whose real successors coincide must not be driven apart
- opportunity floor of 24 roots per stratum; below it, `insufficient_coverage` and **no pass**

## Actor gate — 4 components

`model_validity`, `critic_direction` (start value vs **that row's own recorded** discounted return
— the λ-return correlation is retained but labelled self-consistent), `action_distribution`
(no action above 95% **and** at least `n_actions // 4` used, **and** no policy-weighted death
regression against BC), `paired_uncertainty`.

## Measured feasibility (CUDA, production geometry)

| stage | peak | time |
|---|---:|---:|
| action-blind H16, batch 16 × 128 frames | 1,647 MiB | 5.8 s |
| all-action forks, 512 roots | 335 MiB | 8.2 s |
| critical retention panel | 82 MiB | 34.4 s |

Worst peak **1,647 MiB of 6,144**. Evidence: `gate_smoke.json`.

## Declared scope — measured elsewhere or not at all

Deferred for this **one-seed baseline**, and recorded as `not_evaluated`: decoded tiles / renderer,
stochastic multimodal fidelity, persistent-memory utility, and 3–5 seed replication. The result is
a one-seed M4 baseline, **not** a raw-versus-TC architecture verdict; promoting TC or claiming
training robustness requires replication.

The recursive action-blind reference averages recurrent state at each step. It is a named
diagnostic, **not** the exact conditional mean over action sequences — the all-action fork gate is
what carries the causal claim, which is why it is binding.
