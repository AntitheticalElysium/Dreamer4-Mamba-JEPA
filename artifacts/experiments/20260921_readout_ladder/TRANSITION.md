# Internal transition ladder — where inside `advance()` does safety go missing?

Run 2026-09-24 on the sealed Raw H2 bridge checkpoint. Frozen; only probe heads were fitted. Same
fit roots (7,085, retired) and judgement roots (4,047 from the 405 unallocated seeds, 670 with
termination varying) as `CONFIRM.md`. Rules were committed before launch (`c9ea9cc4`).
Evidence: `evidence/transition.json`, per-row judgement predictions `evidence/transition_rows.pt`
(sha `2e1d75ec31050134`, bound in the JSON).

## What the code settles before any probe runs

`observe_latent` runs `advance` **unchanged** and swaps only the latent slot, and
`features = agent_readout(cat(latent, u))`. So the Mamba output `u` is identical in the real and
generated paths, every rung below is shared by both, and the 0.848 → 0.464 real/generated feature
gap in `CONFIRM.md` is entirely one input slot: `z_true` versus `ẑ`.

Every rung is a deterministic function of the root state and the action — what the root+action
control (`context_action`) sees. The real successor is **not** a fair ceiling for any of them:
Craftax draws zombie movement (75% chase, otherwise random) and mob spawns from the step RNG, so
`z_true` can carry outcome randomness no function of (state, action) can. Root+action is the
information-matched bar.

## Capture checks — passed

- The `final_norm` hook equals `advanced.history` exactly, and the predictor hook equals
  `advanced.latent` exactly, on every batch.
- The four rungs shared with `CONFIRM.md` reproduce it to **0.0** difference: `generated_z` 0.4343,
  `generated_features` 0.4507 (adapter) and 0.4642 (exact), root+action 0.5448.

## Terminal safe-choice, judgement roots

Action-marginal **0.6090**; root+action **0.5448**. `*` = interval excludes zero.

| rung | probe | safe-choice | vs marginal | vs root+action |
|---|---|---|---|---|
| pair_projection | adapter | 0.527 | −0.082 [−0.157,−0.008]* | −0.018 [−0.068,+0.032] |
| pair_projection | exact | 0.484 | −0.125 [−0.196,−0.060]* | −0.061 [−0.117,−0.004]* |
| block_1 | adapter | 0.428 | −0.181 [−0.245,−0.116]* | −0.116 [−0.175,−0.061]* |
| block_1 | exact | 0.484 | −0.125 [−0.194,−0.058]* | −0.061 [−0.124,+0.000] |
| block_2 | adapter | 0.484 | −0.125 [−0.203,−0.043]* | −0.061 [−0.131,+0.006] |
| block_2 | exact | 0.437 | −0.172 [−0.247,−0.103]* | −0.107 [−0.163,−0.053]* |
| block_3 | adapter | 0.519 | −0.090 [−0.166,−0.012]* | −0.025 [−0.085,+0.031] |
| block_3 | exact | 0.448 | −0.161 [−0.235,−0.094]* | −0.097 [−0.157,−0.038]* |
| block_4 | adapter | 0.415 | −0.194 [−0.263,−0.127]* | −0.130 [−0.183,−0.075]* |
| block_4 | exact | 0.491 | −0.118 [−0.190,−0.050]* | −0.054 [−0.114,+0.005] |
| block_5 | adapter | 0.469 | −0.140 [−0.212,−0.071]* | −0.076 [−0.137,−0.016]* |
| block_5 | exact | 0.481 | −0.128 [−0.205,−0.057]* | −0.064 [−0.106,−0.024]* |
| block_6 | adapter | 0.451 | −0.158 [−0.226,−0.093]* | −0.094 [−0.154,−0.035]* |
| block_6 | exact | 0.499 | −0.110 [−0.186,−0.037]* | −0.046 [−0.101,+0.009] |
| **u** | adapter | **0.518** | −0.091 [−0.170,−0.017]* | **−0.027 [−0.089,+0.031]** |
| **u** | exact | **0.519** | −0.090 [−0.164,−0.017]* | **−0.025 [−0.083,+0.033]** |
| predictor_hidden | adapter | 0.521 | −0.088 [−0.162,−0.015]* | −0.024 [−0.080,+0.032] |
| generated_z | adapter | 0.434 | −0.175 [−0.239,−0.108]* | −0.110 [−0.175,−0.052]* |
| generated_features | adapter | 0.451 | −0.158 [−0.226,−0.096]* | −0.094 [−0.157,−0.033]* |
| generated_features | exact | 0.464 | −0.145 [−0.213,−0.078]* | −0.081 [−0.133,−0.028]* |

Fit-minus-judge terminal gaps +0.03 to +0.16; selected steps 300–1,200.

> **CORRECTED 2026-09-24 — read this before the verdict below.** The declared rule returns
> `dynamics_never_construct_it`, and that label must not be read as its words say. Three reasons,
> all raised by a review and verified against this run's saved rows:
>
> 1. **The probes are trained on the wrong objective for the question.** They fit and select on
>    reward cross-entropy plus continuation BCE, then are judged on *within-root* action choice.
>    This project already measured that mismatch on 2026-09-19
>    (`../20260919_localization_ladder/README.md`): on identical real-z rows, switching only the
>    supervision from BCE to within-root ranking moved Raw from 7.0–17.7/36 to 27.0–29.7/36. A rung
>    failing to "carry" under these probes is not evidence it holds nothing. That audit also
>    **retracted a headline of the same form** — "the consequence is never constructed" — and this
>    label repeats it.
> 2. **The predictor does lose probe-accessible safety.** Post hoc, paired and seed-clustered, `u`
>    beats `generated_z` by **+0.084 [+0.021, +0.146]**; their within-root death associations are
>    0.187 versus 0.034. The sentence below saying there is "nothing for the predictor to drop" is
>    withdrawn, and so is "the ladder is flat": the blocks are within noise, the `u` → `ẑ` step is
>    not. The pair was chosen after the fact, so this is a lead, not a declared result — but it
>    rules out the opposite claim as firmly as it supports this one.
> 3. **Every probe here reads a four-frame root context** (the gate's `window_layout` span); each
>    fork stores 32. The reviewer measured the frozen trained head at 32 frames: 0.419 versus 0.428
>    at four, interval [−0.026, +0.009] — no rescue for that head. A fresh, ranking-trained probe on
>    32 frames is untested.
>
> What survives: no rung, under a BCE-style probe on a four-frame context, beats root+action on
> within-root terminal safe-choice. Nothing more.

## Declared verdict (as returned by the rule): `dynamics_never_construct_it`

Every rung is a clean non-carry — none unresolved, none probe-dependent. **No rung beats
root+action**, and the rung your first outcome hinged on, `u`, is indistinguishable from it under
both probes (−0.027 and −0.025, intervals straddling zero), so the declared
predictor-bottleneck branch does not fire. ~~It is not the case that `u` carries the consequence
and the predictor drops it: there is nothing for the predictor to drop.~~ *Withdrawn — see the
correction above: `u` beats `generated_z` by +0.084 [+0.021, +0.146].*

The ladder had the power to see a signal. Its intervals against root+action are about ±0.06 wide;
the real successor beats root+action by +0.30 (`CONFIRM.md`). A `u` carrying a fifth of that would
have resolved.

Read this with two limits:

- **The blocks are within probe noise; the predictor step is not.** The same features give 0.428
  under one probe and 0.484 under the other at `block_1`, and 0.415 vs 0.491 at `block_4`, so do
  not read a shape into the blocks. But `u` → `generated_z` is a resolved drop (+0.084, see the
  correction above). ~~The ladder is flat, not declining.~~
- **This is an absence claim about these probes.** It says no readout tried here extracts
  within-root safety from any rung beyond what it extracts from root+action. It does not show the
  rungs contain nothing.

## Reward, exploratory

Not part of the declared rule, which reads terminal safe-choice only. Root+action regret 0.1637,
marginal 0.2665.

Reward shows the shape termination does not. The root+action-level signal is present at the input
— `pair_projection` −0.009 [−0.020,+0.001] against root+action — then erodes through the stack:
`u` is −0.065 / −0.032 (adapter / exact), `predictor_hidden` −0.087, and `generated_z` −0.191, which
falls **below the marginal** (−0.088 [−0.119,−0.060]). The largest single loss is at the predictor,
`u` → `ẑ`. Reward information the stack does carry is mostly destroyed on the way into the
generated latent. This was not predeclared and is reported as a lead, not a finding.

## Root-side controls (post hoc)

Added **after** the declared call, committed before running (`c6cd8135`), never read by the
declared verdict. `dynamics_never_construct_it` presumes the stack's *input* holds the information,
and root+action already sits below the marginal. Evidence: `evidence/transition_root.json` — the
same run replicates the declared ladder exactly (identical readings, identical safe-choice in every
declared cell, shared rungs matching `CONFIRM.md` to 0.0), so the switch to `encoder.export` left
`z` unchanged.

| rung (adapter) | safe-choice | vs marginal | vs root+action | fit-root safe-choice |
|---|---|---|---|---|
| root_z_action — what the transition consumes | 0.521 | −0.088 [−0.151,−0.028]* | −0.024 [−0.072,+0.024] | 0.629 |
| root_cls_action — before the projector | 0.531 | −0.078 [−0.154,−0.006]* | −0.013 [−0.060,+0.035] | 0.660 |
| root_cls_patches_action — + pooled patch grid | 0.558 | −0.051 [−0.118,+0.012] | +0.013 [−0.039,+0.062] | 0.710 |

> **CORRECTED 2026-09-24.** This call carries the same objective mismatch as the declared one,
> and reads the same four-frame context. Root CLS + patches also *raises* within-root death
> association over root+action, 0.297 versus 0.184, even though its safe-choice does not resolve —
> the ordering signal is there to be trained for. Read the label below as "these BCE-style probes on
> four frames did not beat root+action", not as a statement about what the root holds.

**Post-hoc call: `not_predictable_from_root`, for this probe family.** No representation of the
root observation — the `z` the transition consumes, the CLS before the projector, or CLS with
TC-LeWM's pooled patch grid — ranks the 17 actions by death better than root+action, and none beats
the action marginal.

So the declared call cannot yet be read as a transition *failure*. The stack may have had nothing
it could construct. Two explanations fit, and these data cannot separate them:

- **(a) Termination given (state, action) is substantially random or driven by hidden state.**
  Zombie movement is drawn from the step RNG and the attack cooldown is not rendered. All 17
  actions at a root are forked with **one shared step key** (`collect_broad_forks.py:176-178`), so
  the real successor shows each action's outcome under a draw nothing at decision time knows. The
  real successor's +0.30 edge would then be realized-outcome information, not a consequence a model
  failed to compute.
- **(b) It is in the root frames and these probes cannot get it out.** Fit-root safe-choice climbs
  as the input gets richer (0.629 → 0.660 → 0.710) while judgement barely moves — a fit/judge gap
  up to +0.15 on roughly 1,260 terminal-opportunity fit roots. That is the signature of a probe
  short of data, not of an input with nothing in it.

This matters for the counterfactual-MSE A/B. Under (a), all-action MSE toward one realized `z_true`
per action asks a deterministic transition to fit outcome noise, and its promotion metric —
generated safety beating the marginal and root+action — may be out of reach for **any**
deterministic one-step model. Under (b), the A/B is well-posed. What separates them is the
achievable ceiling: replay terminal-opportunity roots with K independent step keys per action and
measure P(death | state, action) directly. The collector already replays BC trajectories
deterministically from the seed, so this reuses existing machinery.

## Reproduce

```
./artifacts/experiments/20260921_readout_ladder/run.sh transition          # declared ladder
# root-side controls: the same script with --name transition_root (see git c6cd8135)
```
