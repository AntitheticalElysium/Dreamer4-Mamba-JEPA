# The three gates, for validation before relaunch

Code: `d4mj/lewm_diagnostics.py` (`bridge_gate`, `actor_gate`). Boundary: `d4mj/gates.py`
(`require_bridge_gate`, `require_actor_gate`) — it owns identity and the stop decision and refuses
to compute evidence, which is why the measurements live separately.

Invoke: `python -m d4mj gate --run <arm> --stage h2|h16|actor --dataset <stores>`

## Bridge gate (H2→H16, and H16→actor) — 7 components

| component | measures | **passes iff** |
|---|---|---|
| `source_contract` | frozen encoder, predictor BN buffers in eval, cache/parent binding | all three hold |
| `semantic_retention` | projected `z` vs its own CLS, paired AUC, both probe families | every family resolves an interval **and** none is confidently inferior beyond `.03` |
| `recursive_dynamics` | rollout MSE at each depth vs persistence and marginal-action | beats **both**, each interval excluding zero, at **every** evaluated depth |
| `action_effects` | predicted change vs real change; derangement cost | `effect_R² > 0` **and** re-labelling the action costs error with the interval excluding zero |
| `outcome_calibration` | reward vs zero/marginal; continuation BCE/Brier | reward beats **both**; balanced terminal BCE `< log(2)` **classwise and aggregate** |
| `observed_bc` | top-1 agreement on the relevant half vs most-frequent action | **reported, not margin-gated** — see below |
| `paired_uncertainty` | every decision-bearing contrast with its interval | at least one contrast resolves |

Depths: H2 evaluates `(1, 2)`; H16 evaluates `(1, 2, 4, 8, 16)`.
`validated_recursive_depth` = the largest depth beating both baselines. The H16 gate therefore
needs depth **16** to clear both, since `require_bridge_gate` is called with
`minimum_depth=horizon`.

## Actor gate (screen→budget) — 4 components

| component | passes iff |
|---|---|
| `model_validity` | the frozen world still beats persistence at the horizon, interval excluding zero |
| `critic_direction` | value–return correlation `> 0` |
| `action_distribution` | no single action takes `> 95%` of choices |
| `paired_uncertainty` | at least one contrast resolves |

## Four judgment calls you should check

**1. The marginal-action baseline is a fixed derangement, not an average over actions.** I roll out
with each row's actions re-labelled by a derangement, which doubles as the sensitivity test. But a
true marginal is `E_a[f(z,a)]`, the conditional mean, which is the *best* action-blind predictor —
whereas one specific wrong action is noisier and therefore **easier to beat**. So this gate is
**more lenient than the spec's "marginal-action baseline"**. Fixing it costs 17× rollouts at gate
time only (not training). I'd take the fix; flagging it rather than quietly shipping the weaker one.

**2. `critic_direction` passes on correlation > 0.** That is very weak — it only asks that the
critic is not anti-correlated with its own returns. G4's real concern is exploitation, which this
does not detect. A stronger rule would require a margin, or compare against true-successor
substitutions as §G4 describes.

**3. `action_distribution` fails only above 95% single-action share.** Near-total collapse is
caught; a policy that degenerates to two or three actions passes. Entropy and KL-to-prior are
recorded but not gated.

**4. `observed_bc` returns `pass` whenever it is measurable.** I read §G2's `.5`-achievement
noninferiority as belonging to G4, against a real-game BC, not to the bridge. If you want it
binding here it needs a declared margin and a reference.

## Not evaluated, and recorded as such

§G3 also asks for stochastic successor mode fidelity over repeated simulator seeds,
persistent-memory utility under varied earlier context, and decoded-tile comparisons. Those need a
fork harness and a renderer. The report carries them under `not_evaluated` so nothing claims
coverage it lacks.

## A fail-open I found while writing this up

`semantic_retention` first used `not projection_stop`. That expression is
`all(interval is not None and ...)`, so with **no label support** every interval is `None`, the
`all` is `False`, and absent measurement read as a **pass**. Noninferiority now has to be positively
established: every probe family must resolve an interval. Test:
`test_retention_without_coverage_does_not_pass`.

## Test evidence

`d4mj/tests/test_bridge_gate.py`, 8 tests: an untrained world does **not** pass; the boundary
accepts a genuinely passing report; and a report doctored after sealing, sealed with too shallow a
depth, or whose evidence bytes moved is refused in all three cases.
