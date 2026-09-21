# M4 baseline — result

**The H2 gate refused H16.** `validated_recursive_depth = 1`, below the 2 the stage requires.
Five of seven components failed. This is a measured negative result, not a crash.

Scope: **one arm** (raw). TC could not export a TC-14 compliant cache — see `TC_EXPORT_FINDING.md`.

## Joint phase (both arms, byte-identical initialization)

| arm | init | G1 (2k) | final (10k) |
|---|---:|---:|---:|
| raw | 11.26 | 0.044 | **0.019** |
| tc | 11.26 | 0.938 | **0.327** |

G1 passed both on all six components. TC plateaued near 0.3 from update 4,000 while raw kept
descending — a 17× gap on the shared objective from the same weights, seeds and corpus.

## The H2 gate (raw, 2,000 bridge updates)

| component | verdict |
|---|---|
| `source_contract` | **pass** |
| `paired_uncertainty` | **pass** — 17 contrasts resolved, none unresolved |
| `recursive_dynamics` | fail |
| `action_effects` | fail |
| `outcome_calibration` | fail |
| `observed_bc` | fail |
| `semantic_retention` | fail |

### What the world can do

**Latent prediction at depth 1 works.** Generated 0.0322 against persistence 0.0483 and the
action-blind mean 0.0506 — it beats both with the interval excluding zero.

**Effects are action-conditional.** Effect R² 0.164, cosine 0.317, re-labelling the action costs
error, and it beats the action-blind rollout.

**Reward regression on generated states works.** Generated MSE 0.0395 against a zero predictor's
0.0702 and the marginal's 0.0654 — beats both.

### What it cannot do

**Depth 2 is not established.** Generated 0.0492 is numerically better than persistence 0.0602 and
action-blind 0.0708, but the paired interval does not exclude zero. One step is validated; two are
not.

**Within-root action decisions are worse than action-blind.** On 512 held-out fork roots, all 17
actions from the same state:

| decision | generated | action-marginal | |
|---|---:|---:|---|
| reward regret (lower better) | **0.2781** | 0.2277 | worse |
| terminal safe choice (higher better) | **0.4510** | 0.6569 | worse |

512 reward-opportunity roots and 102 terminal-opportunity roots, both far above the floor of 24.
Choosing by the *global* action average beats consulting this world about *this* state.

**BC barely exceeds its floor.** Observed top-1 0.1771, generated 0.1667, most-frequent-action
0.1562. Neither beats the marginal with a resolved interval.

**Retention is not established.** Macro AUC projected 0.701 vs CLS 0.708 (linear), 0.705 vs 0.729
(MLP), on 21 supported critical labels. At least one family's lower bound falls past the −0.03
margin.

## Why this matters

`outcome_calibration`'s reward component **passed** against zero and marginal baselines, while the
within-root fork decision **failed** against the action-marginal. The same model, on the same
outcome, looks competent globally and incompetent per-state.

That is precisely what the second audit predicted global baselines would hide, and precisely why
the all-action fork gate was made binding. Without it this run would have shown a mixed but
arguable picture; with it, the finding is unambiguous: **this world does not carry
state-conditioned action consequences, while its aggregate latent and reward metrics look
reasonable.**

## Declared limitations

- **One arm.** No raw-versus-TC architecture verdict is available.
- **One seed.** No training-robustness claim.
- **Retention compares against CLS only** — the preselected old-export reference could not be
  loaded, see the gate-runner note.
- `continuation` reported `insufficient_coverage`: too few dead examples in the DEV suffix sample.
- Stochastic multimodal fidelity, persistent-memory utility and decoded tiles remain
  `not_evaluated`.

## Evidence

`artifacts/lewm_m4_canonical/raw/gates/h2/` — the sealed report, per-component evidence files and
`*.rows.pt` raw inputs for every aggregate above.
