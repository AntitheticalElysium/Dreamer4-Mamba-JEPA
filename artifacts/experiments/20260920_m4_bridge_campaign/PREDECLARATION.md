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

## Scope, stated before the result

This is **not** paper-minimal LeWM. All-action counterfactual branches contribute to world training
through a separately declared term at Direct's mass. It is our LeWM-Mamba candidate under the
project's corrected Direct data contract. Whether the fork component was necessary is a later
ablation, not a claim this run can make.

Only the archive is BC-eligible: the world sees 8,325 TRAIN episodes, BC sees 256 of them
(552,998 transitions). The primary comparison is internally consistent — actor versus *its own* BC,
both reading the same heads — but the BC's data support is narrower than the world's, and that is a
property of the corpus, not of either arm.

One training seed per arm. A single seed cannot separate an architectural effect from seed noise,
so "actor beats BC" at one seed authorizes replication, not a claim.
