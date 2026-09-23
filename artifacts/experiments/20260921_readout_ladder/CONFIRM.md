# Confirmation run — the trained head, the exact refit, and `generated_z`

Run 2026-09-23 on the sealed Raw H2 bridge checkpoint. Every head below is scored on **one**
population: 4,047 roots from the **405 fork seeds the sealed partition left unallocated**, which
no head here was fitted or selected on. 670 of them have termination varying across actions.
Fresh heads were fitted on the 7,085 already-retired fit roots, with selection inside a 15%
by-root slice of those. Evidence: `evidence/confirm.json`.

This removes the two confounds that left the ladder's fourth claim provisional: the fresh head now
has **no adapter** and reproduces the deployed `Heads.model_body + reward/continuation` exactly,
and the trained head is scored on the **same roots** as the fresh ones — read through the source
closure's own `_fork_readouts`, not a reimplementation.

Provenance: checkpoint `ffb852c1f650a106`, partition `aab9f3e483bf6d2c`, fork-file bytes
`bb8699c6b0432f4d`, judgement seed list `d05f2e2651318bf5`, materialized rows `a06d7caa484e5fad`
(judge) / `bbfe6bf3cfcf6cdb` (fit). The fork store has no manifest of its own, so the bytes and the
exact `(seed, step)` rows are bound here instead.

## Safe-choice on the judgement roots

Action-marginal control **0.6090**; `context_action` (root + action, no successor) **0.5448**.

| rung | safe-choice | vs marginal | vs context | vs trained head |
|---|---|---|---|---|
| successor_cls (real) | **0.8866** | +0.278 [+0.222,+0.334] | +0.342 [+0.265,+0.404] | +0.187 [+0.123,+0.247] |
| **exact:real_features** | **0.8478** | +0.239 [+0.187,+0.290] | +0.303 [+0.238,+0.360] | **+0.148 [+0.093,+0.203]** |
| adapter:real_features | 0.8284 | +0.219 [+0.162,+0.275] | +0.284 [+0.214,+0.345] | +0.128 [+0.067,+0.189] |
| **trained bridge head (real)** | **0.7000** | **+0.091 [+0.030,+0.158]** | — | reference |
| successor_z (real) | 0.6537 | +0.045 [−0.021,+0.108] | +0.109 [+0.041,+0.174] | −0.046 [−0.101,+0.012] |
| *action marginal* | *0.6090* | — | — | — |
| *context_action* | *0.5448* | −0.064 [−0.146,+0.011] | reference | −0.155 [−0.220,−0.089] |
| exact:generated_features | 0.4642 | −0.145 [−0.213,−0.078] | −0.081 [−0.133,−0.028] | +0.036 [+0.003,+0.071] |
| adapter:generated_features | 0.4507 | −0.158 [−0.226,−0.096] | −0.094 [−0.157,−0.033] | +0.022 [−0.022,+0.070] |
| **generated_z** | **0.4343** | −0.175 [−0.239,−0.108] | −0.110 [−0.175,−0.052] | −0.266 [−0.324,−0.205] |
| trained bridge head (generated) | 0.4284 | −0.181 [−0.248,−0.118] | — | — |

Fit-minus-judge gaps are small (reward −0.009 to −0.077, terminal −0.023 to +0.092); selected
steps 300–3,900 of 6,000.

## The four declared branches

**1. "Exact fresh head succeeds, trained head fails" → bridge training data/objective.**
*Partly.* Both **succeed**: the trained head reaches 0.700 against a 0.609 marginal, +0.091
[+0.030,+0.158], resolved. **This overturns the ladder's fourth claim**, which said the trained
head fails to read termination present in its own features. That rested on the H2 gate's 0.559
against a 0.657 marginal — a 512-root, 102-opportunity population — and the sign flips on a larger
matched one. `RESULT.md` §4 is corrected.

What survives is narrower and now directly tested: from identical features on identical roots, the
exact-capacity fresh head reaches **0.848** where the trained head reaches **0.700**, a paired
**+0.148 [+0.093,+0.203]**, and on reward regret **+0.124 [+0.108,+0.142]**. The trained head
extracts materially less than an exact-capacity refit — it does not fail to extract.

**2. "Adapter succeeds, exact fails" → readout capacity/interface. Ruled out.**
The exact deployed capacity is *better* than the adapter head (0.848 vs 0.828 terminal, 0.179 vs
0.195 reward regret). The extra adapter was never doing the work, so the gap in branch 1 is not
about capacity or the input interface. Combined with branch 1, the remaining explanation for the
trained head's shortfall is its training data or objective — not proven here, but the two
competing explanations are now eliminated.

**3. "`generated_z` fails" → the transition's latent is the problem. Confirmed, strongly.**
`generated_z` — `advanced.latent`, read **before** `world.features` — is the worst rung in the
table: 0.434, below the marginal (−0.175), below root+action (−0.110), and below the trained head
(−0.266), all resolved. The termination signal is already gone at the transition's own latent.

**4. "`generated_z` succeeds, generated features fail" → generated agent readout. Ruled out.**

## What the two chains look like

```
real:       CLS 0.887  ->  z 0.654  ->  observe_latent 0.848  ->  trained head 0.700
generated:                 z 0.434  ->  features       0.464  ->  trained head 0.428
                                                     (context 0.545, marginal 0.609)
```

The real chain loses a lot at the projector (0.887 → 0.654) and `observe_latent` then recovers it
by recombining with root state (0.654 → 0.848). The generated chain is **flat and below
`context_action` at every rung**: nothing downstream damages it, because there is nothing left to
damage. Conditioning on the generated latent is worse than not looking at a successor at all.

Heads still fail to transfer: an exact head fitted on real features scores 0.434 on generated ones.

## Reward, unchanged

`context_action` remains the best reward rung (regret 0.1637, +0.103 over the marginal).
`exact:real_features` is 0.1786, and against `context_action` it is **−0.015 [−0.037,+0.008]** —
indistinguishable. The ladder's first finding replicates on a fresh population: the successor adds
nothing to within-root reward ranking. `generated_z` is the worst reward rung too (0.3547).

## What this establishes, and what it does not

Established on one matched, untouched population with three controls and direct paired tests:

- The transition's latent is where the generated pathway's termination signal is lost. Not the
  agent readout, not `world.features`, not head capacity.
- The deployed outcome-head capacity is sufficient; an exact refit beats the adapter.
- The trained bridge head beats the action-marginal on real successors and underperforms an exact
  refit by +0.148 safe-choice on identical roots.
- CLS → z projector loss replicates on a fresh population (+0.342 vs +0.109 over `context_action`).

Not established:

- That the trained head's shortfall is *caused* by its data or objective. Capacity and interface
  are ruled out; the remaining explanation is not thereby proven.
- That repairing the transition latent would move the gate. Two deficits remain separable.
- Anything licensing H16 or actor training. The stop decision stands.
- The exact head fits and reads lead 0 only, because fork roots carry one-step outcomes; the
  deployed head trains across 8 MTP leads. That makes the exact head's job *easier*, which is the
  right direction for a control meant to rule out capacity, but it is not the deployed objective.

## Reproduce

```
./artifacts/experiments/20260921_readout_ladder/run.sh confirm
```
