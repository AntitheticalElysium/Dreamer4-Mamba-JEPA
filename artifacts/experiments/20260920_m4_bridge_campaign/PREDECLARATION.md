# Predeclared before any fresh training — checklist 3.1 / 3.2

Committed before Stage 4 launches. Nothing below may be revised after a number is seen; if it is,
the revision is a new document and the original stays in history.

## Primary result

**Each arm's imagination actor against its own frozen BC prior, on real Craftax.**

| | |
|---|---|
| seeds | `30000 … 30511` — 512 shared DEV seeds, every policy on the same ones |
| horizon | the native 10,000-step Craftax cap, not the collector's 2,500 |
| sampling | categorical at temperature 1; greedy is a declared secondary, never the headline |
| statistic | mean achievements per episode |
| interval | `execution.evaluate`'s paired episode bootstrap, 2,000 draws, 95% |
| decision | **actor beats its own BC iff the achievement-gap interval excludes zero on the positive side** |
| control | a random policy on the same seeds, reported but not part of the decision |

This is `eda/run_paired_execution.py`'s protocol unchanged, so the result is directly comparable to
the Direct table it is being read against:

| backend | BC achievements | actor achievements | actor − BC (95%) |
|---|---:|---:|---|
| attention | 8.762 | 7.811 / 7.863 | −0.951 [−1.244, −0.639] / −0.898 [−1.188, −0.582] |
| mamba | 9.150 | 7.693 / 8.547 | −1.457 [−1.779, −1.117] / −0.604 [−0.930, −0.254] |

**The BC arm is the frozen prior saved beside the actor** — the same heads the actor was
initialized from and KL-bounded against. Not another model's BC.

## Staged stop points, and what each assigns

Read in order. The first one that fails is the diagnosis; later stages are not evidence about
earlier ones.

| observation | assignment |
|---|---|
| poor observed-path BC | representation / readout / data |
| good BC, poor generated heads | world / recursive bridge |
| good generated heads, actor < BC | critic / policy / imagination — Direct's failure, reproduced |
| actor > BC | genuine architectural signal → replicate across training seeds before any claim |

"Poor" and "good" are relative to the arm's own observed-path numbers and to the random control;
no absolute threshold is declared, because none is defensible in advance on this task. What IS
declared is the ordering above and that the first failure is the diagnosis.

## Declared secondaries

Reported, never substituted for the primary: Craftax official score and its interval; mean reward;
episode length; per-achievement rates; greedy execution; the M03 / fork / semantic probes.

## What would make the result uninterpretable

Declared now so it cannot be rationalized later:

- Either arm failing to complete its declared budget.
- An arm whose paired initialization identity differs from the other's.
- A fork root appearing on a sealed evaluation seed (the loader refuses this outright).
- A DEV or FINAL episode reaching joint training.
- The actor imagining past the depth the bridge trained (`horizon <= trained depth`, checked in code).

## The fork condition, declared

**FLATTENED.** Each `broad_forks_v2` (root, action) counterfactual is one ordinary episode in the
corpus, drawn through the normal `JointSampler` minibatch path at the same batch size. `fork_mass`
is **0.0**: there is no separate branch term and no extra per-update computation. The world
therefore never sees two actions from the same root in one batch.

This is deliberately *not* the grouped alternative (Direct's branch contract: 4 roots x 17 actions
+ surviving second steps at 20% loss mass, ~136 extra differentiable transitions per update). The
two answer different questions, and the one declared here is the purer of them:

> can **native** LeWM work when simply exposed to the same examples?

rather than

> can LeWM work when given **Direct-style counterfactual supervision**?

Consequences, stated before any number:

- Fork exposure is **~12.1% of draws**, a share set by `JointSampler`'s window weighting (S56), not
  tuned. It is a property of the flattening rule; it was not chosen to hit a target.
- The flattened corpus reaches **joint world training only**. Its history transitions carry
  fabricated zero rewards — those rewards were never collected — and a counterfactual branch is not
  logged behaviour, so it is `bc_eligible: False` and is excluded from the bridge's reward,
  continuation and BC heads by construction, not by convention.
- This is still **not paper-minimal LeWM**: counterfactual transitions are in the training
  distribution. It is closer to canonical LeWM than the grouped form, because nothing about the
  objective or the per-update computation changes.
- Whether the *supervision structure* matters beyond the *examples* is the grouped-vs-flattened
  ablation, and this run does not answer it. The 2026-09-19 A' arm does not answer it either: that
  arm trained on the **logged action only**, one per root.

Corpus figures, for the FLAT condition actually declared (the earlier numbers here were the
two-source ones and understated the world's exposure by more than an order of magnitude):

| | episodes | note |
|---|---:|---|
| world sees (TRAIN) | **236,924** | archive 256 + support 8,069 + flattened forks 228,599 |
| bridge/actor see (TRAIN) | **8,325** | forks excluded: fabricated zero rewards, not logged behaviour |
| BC-eligible (TRAIN) | **256** | archive only, 552,998 transitions |

Merged three-source audit: 265,672 episodes, 4,872,668 transitions, TRAIN/DEV/FINAL
236,924 / 27,714 / 1,034. Joint-window draw shares: support 72.01%, expert 15.65%, forks **12.35%**.

The primary comparison is internally consistent — actor versus *its own* BC,
both reading the same heads — but the BC's data support is narrower than the world's, and that is a
property of the corpus, not of either arm.

One training seed per arm. A single seed cannot separate an architectural effect from seed noise,
so "actor beats BC" at one seed authorizes replication, not a claim.
