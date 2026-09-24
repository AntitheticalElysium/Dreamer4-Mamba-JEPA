# Frozen Raw readout ladder — predeclaration

Written 2026-09-21, before the run's numbers existed. The decision rules below are the ones
encoded in `ladder.py:verdict`; they were fixed in code before the run was launched.

## What this measures, and what it cannot

The H2 gate refused H16 because the system could not rank actions **within a root**. The
true-successor substitution then showed the reward head was no better when handed the REAL
successor (regret 0.2879 real vs 0.2781 generated), which rules out generated transition error as
the main reward explanation. But a real successor still traverses `observe_latent`, the agent
readout, pooling and `model_body`. The failure is somewhere on that shared route, and the gate
cannot say where, because the gate only ever reads the *trained bridge heads*.

This ladder fits **fresh** heads on **frozen** features taken from four points along that route.
Nothing in the world model is updated. A fresh head succeeding where the bridge head failed
localizes the problem to the bridge head's data or objective; a fresh head failing too localizes
it upstream, to whichever rung it first fails at.

It cannot tell us whether a successful rung would survive being trained jointly, and it cannot
license H16. It is a localization instrument, not a rescue.

## Families (the rungs)

| family | features | what its success would mean |
|---|---|---|
| `context_action` | root agent features + one-hot action, **no successor** | the outcome was predictable before any successor was seen; the "within-root" contrast is weaker than assumed |
| `successor_cls` | the encoder's unprojected CLS of the real successor | the representation carries it *upstream of the projector* |
| `successor_z` | the projected `z` the world actually transitions | the representation carries it |
| `real_features` | `observe_latent(root, a, z_true)` | the world's own readout of a REAL successor carries it |
| `generated_features` | `advance(root, a)` | the deployed path carries it |

`successor_cls` was added after the decision rules were written, to separate "the representation
lacks it" from "the projector discarded it". It is therefore **excluded from every decision rule
below** — a rung added later must not be able to move a predeclared decision. It is supplementary
evidence only.

## Declared decision rules

Read on the **DEV** (judgement) roots, on the diagonal cells, where "succeeds" means beating the
action-marginal control on reward regret **or** on safe-choice rate with a seed-clustered
bootstrap interval excluding zero.

1. real **and** generated both succeed → `bridge_head_data_or_objective`. The features carry it;
   the bridge heads' data/objective is the problem. Consistent with only 25% of their supervision
   coming from generated suffixes, all on logged actions.
2. real succeeds, generated does not → `transition_or_generated_feature`. The generated path is
   at fault.
3. neither succeeds, but `successor_z` does → `observe_latent_or_agent_readout_bottleneck`.
4. nothing succeeds, including `context_action` → `representation_or_context_deficiency`.
5. anything else → `mixed`; read the matrix, claim nothing.

**Overriding condition.** A route that succeeds on the fit roots but fails on the judgement roots
is a generalization failure and must not be read as a success, whatever rule it would trigger.
`train_dev_gap` is reported on every cell for exactly this.

## Protocol commitments

- **Root partition is sealed** (`evidence/root_partition.json`, ledger `aab9f3e483bf6d2c`):
  50 seeds reserved for the gate, 700 fit, 350 judgement, 405 unallocated. Fit and judgement
  seeds are disjoint and are RETIRED from future evaluation.
- **Model selection never touches the judgement roots.** Selection runs every 100 updates on 15%
  of the *fit* roots, held out by root so a root's 17 rows never straddle the split. This is not
  optional: a 600-root calibration showed every family, including `context_action` which carries
  no successor information at all, reaching within-root association above 0.9 on its own fit data
  and near zero on held-out roots. Measured off the last update, a DEV failure would have been
  indistinguishable from pure memorization.
- **Identical head, initialization and batch order across families.** The input adapter is
  declared: families differ in dimension and something must reconcile them.
- **The control is the action-marginal**, not a global baseline. The M4 result turned on exactly
  this: `outcome_calibration`'s reward component PASSED against global baselines while the
  within-root fork decision FAILED against the action-marginal.
- Bootstrap intervals are clustered by **episode seed**, not by root.
