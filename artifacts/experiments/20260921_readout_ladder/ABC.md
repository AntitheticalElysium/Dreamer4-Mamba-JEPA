# A / B / C — terminal exposure, then a reweighted loss, on the u→u world

Run 2026-09-24/25, `abc.py` (`b41989d3`, recipes and every decision rule committed before the pool
was built, any arm trained, or any sealed seed walked). A machine shutdown at 20:47 on 2026-09-24
interrupted C's training and the sealed collection; both were relaunched unchanged (the collector
resumed at seed 52,203, counting the files already on disk toward its stop rule).

## Design

Identical in everything but what each arm changes: the frozen TC-consecutive encoder and its
persisted PCA, the world architecture, init seed 7, batch seed 11, AdamW from the old recipe, 10k
updates of batch 128, a fixed 25,600-window pool, all trained from scratch. The training loop
reproduces A's recorded step-500 loss to 6×10⁻⁸.

| arm | pool | loss | deaths as targets |
|---|---|---|---|
| A | the existing world: old 13-frame layout | MSE | 0 |
| B | 95% uniform 4-frame windows + 5% death-ending windows, each a distinct terminal episode | MSE | 1,365 unique, 67,859 presentations |
| C | B's exact pool and batch order | MSE weighted by TRAIN-variance^(−½), mean one (116.7× range) | the same |

## Stage 1 — exploratory panel (the observability roots)

B − A = +0.002 [−0.033, +0.034] against a root-to-A gap of 0.078: B did not largely close the gap,
so by the declared rule C was trained. A rescore of all three arms later on the same panel moved A
and B by about 0.02 (GPU nondeterminism in the probe fits and the Mamba rollout), about the size of
the differences there; nothing on the exploratory panel resolved.

## Stage 2 — sealed panel, scored once

Seeds 52,000–52,405 (402 with roots), collected after the rules were committed; stop rule met at 800
one-step opportunity roots (1,442 two-step). Generated-fitted heads, three probe seeds, paired
episode-seed-clustered 95% intervals.

| one-step death | expected safe | zombie | lava | night | SLEEP choices | fatal alignment | action-effect error |
|---|---|---|---|---|---|---|---|
| prior (always DOWN) | 0.601 | | | | | | |
| root u | 0.800 | 0.742 | 0.973 | 0.745 | 116 | | |
| real successor u | 0.991 | 0.990 | 1.000 | 0.988 | 321 | | |
| **A** | 0.739 | 0.652 | 0.950 | 0.686 | 240 | 0.492 | 0.051 |
| **B** | **0.697** | **0.583** | 0.983 | 0.655 | **451** | 0.486 | 0.049 |
| **C** | **0.768** | 0.679 | 0.977 | 0.716 | 327 | **0.562** | 0.047 |

| paired contrast | overall | zombie | lava |
|---|---|---|---|
| B − A | **−0.042 [−0.071, −0.013]*** | **−0.069 [−0.104, −0.034]*** | +0.033 [+0.006, +0.067]* |
| C − A | **+0.029 [+0.003, +0.054]*** | +0.027 [−0.012, +0.064] | +0.027 [−0.004, +0.065] |
| **C − B** (same data, same batches: the loss alone) | **+0.070 [+0.040, +0.102]*** | **+0.096 [+0.053, +0.138]*** | |
| root u − A | +0.061 [+0.033, +0.090]* | | |

Two-step death (1,442 roots, prior 0.622): no arm contrast resolves (C − A +0.006, B − A −0.004).

## Declared reading: `stop_loss_tweaking`

| | S1 improves on A | S2 improves on A, zombie roots | S3 retention | succeeds |
|---|---|---|---|---|
| B | no (resolved worse) | no (resolved worse) | yes | **no** |
| C | **yes** | no (+0.027, unresolved) | yes | **no** |

Neither arm meets S1 ∧ S2 ∧ S3, so the rule returns `stop_loss_tweaking`: return to spatial dynamics
or uncertainty modelling rather than a larger LeWM-variant run on this loss.

## What the numbers show beyond the rule

- **Adding deaths under plain MSE made the world worse.** B is resolved below A overall (−0.042) and
  on zombie roots (−0.069), better near lava (+0.033), and chooses SLEEP almost twice as often (451 vs
  240). B changed two things relative to A — terminal access and the window distribution (uniform
  4-frame starts instead of 13-frame-span starts) — so which of them hurt is not separable here.
- **The loss change is large on identical data.** C − B, the clean loss comparison, is +0.070 overall
  and +0.096 on zombie roots, both resolved; C is the only arm whose generated u begins to align with
  the fatal direction (0.562 vs ≈0.49). But part of that gain is C undoing a harm B's data introduced:
  against the original world A, C gains +0.029 overall — about half the root-to-A gap — and +0.027 on
  zombies, unresolved.
- **Movement prediction survives every arm** (action-effect error 0.047–0.051); the bounded weighting
  did not buy its gain by giving up the large components.

## Limits

- One training seed per arm, one fixed pool per arm; run-to-run noise measured at about 0.02 on the
  exploratory panel.
- Training sufficiency is untested: the loss was logged as one batch every 500 steps, too noisy to
  show convergence, and no longer or larger-pool run was made.
- Everything here is the frozen TC-consecutive encoder with a post-hoc patch PCA, not canonical LeWM.
