# Why the u→u world's prediction loses mob consequences — the one-day diagnostic queue

> **Headline, amended 2026-09-24 after review:** the queue **prioritizes a loss intervention, with
> terminal exposure unresolved**. It does not identify the loss as the cause. The first version said
> the declared reading "points at the prediction loss"; that is what the rule returns, not a finding.

Run 2026-09-24. Rules for parts 1–3 committed before either ran (`diagnose.py`, `e86e33e1`); the
post-hoc damage test committed before it ran (`f44e16ca`). All on the observability roots,
**exploratory**. Evidence: `evidence/diagnose_critical.json`, `evidence/exposure.json`,
`evidence/damage_direction.json`.

**Where the loss was known to sit** (U_WORLD.md): root `u` 0.804, generated `u` 0.727. **Why** was not.

## Part 2 — action-critical prediction error: `concentrated_error` (valid)

The fatal-versus-safe direction `w`, fit on within-root-centred real successor `u` from FIT roots:

| | within-root AUC along `w` |
|---|---|
| held-out **real** successors | **0.9975** (the direction is valid) |
| held-out **generated** successors | **0.495** — chance |

| 493 opportunity roots | world | copy the root |
|---|---|---|
| normalized within-root error, all directions | **0.045** | 1.000 |
| normalized within-root error along `w` | **1.137** | 1.000 |
| error ratio, along `w` / all | **25.2** (declared threshold 2) | |
| generated effect along `w`, magnitude / real | 0.39 | |
| correlation of generated and real effect along `w` | **0.02** | |
| share of the within-root action effect lying along `w` | 0.04% | |

The world reproduces 95.5% of the within-root action effect overall, and not the action effect along
this direction: its error there is *worse* than predicting no effect at all.

Two limits, from review. **This is an alignment test, not an information ceiling.** `w` was fit on
*real* successors and transferred unchanged; a head fit on generated `u` itself still chose safely at
0.727, above the 0.628 prior (U_WORLD.md) — so 0.495 means the generated states do not put the
fatal-versus-safe difference where real ones do, not that they hold none. **And the direction may
chiefly recognize a successor that is already dead** — easy to read in hindsight, while the predictor
has to forecast it from the root, and (Part 1) this world was never given a terminal target. The same holds by
zombie (ratio 29.7), lava (16.2), night (33.2) and day (18.2).

## Part 1 — training exposure: not sparse by the declared thresholds, but **no deaths at all**

The old world's exact pool, rebuilt by replaying its sampler (all 25,600 cached action triples
reproduced byte for byte). Labels: health change from the reward (checked **exactly** on 7,021 fork
transitions); zombie adjacency and night from pixel classifiers trained on FIT roots with known
simulator state, validated on fresh roots — adjacency precision / recall **1.000 / 1.000** by day,
**0.999 / 1.000** at night; night accuracy 0.982.

| old u→u pool | unique transitions | declared minimum |
|---|---|---|
| all | 75,547 | |
| beside a zombie | 5,747 (1,904 episodes; 3,354 at night) | |
| … staying put | 2,624 (19.7% then lose 2+ health) | 200 |
| … moving | 3,123 (5.7% then lose 2+ health) | 200 |
| … followed by 2+ health lost | 694 | 50 |
| **deaths** | **0** | not declared |

By its thresholds, exposure to the *situations* and to *damage* is adequate. What the thresholds did
not count: **the pool contains no death at all**, though its corpus's training split holds 8,015. The
cause is structural. `JointSampler` draws windows starting in `[0, len + 1 − span]`; the
TC-consecutive recipe's span is 13 native frames (offsets up to 12) but it predicts only the first
three transitions, so **the last 9 transitions of every episode, every death included, can never be a
prediction target**. The Raw recipe's span of 4 reaches the terminal transition, in each episode's
final window only.

The world's one-step error on the logged pool transitions, normalized by the variance of next `u`:
along `w` it is 0.125 elsewhere and 0.207 on damaging transitions beside a zombie.

## Declared reading: adequate exposure + concentrated error → prioritizes a loss intervention

**Part 3, the averaged-successor test, did not run.** Its declared condition — adequate exposure with
error *not* concentrated, or an invalid direction — was not met. That is a branch of the rule, not a
scientific exclusion of averaging: one-step death outcomes are rarely stochastic on this panel (15 of
500 roots), but random mob imagery can still shape a latent MSE target. It is lower priority than the
known terminal-target exclusion, not ruled out.

## Post hoc — the damage direction: **void** by its declared gate, and descriptively the same failure

Zero deaths in the pool gave a second explanation the declared queue could not separate from the
loss: the fatal direction is the look of a dead successor, which this world never saw. The pool
*does* hold 894 non-fatal damage transitions, so a damage direction was fit the same way
(surviving branches, health −2 or worse versus unharmed).

| 2,268 damage-opportunity roots | real AUC | generated AUC | error ratio | effect corr. |
|---|---|---|---|---|
| all | **0.8968** | **0.500** | 35.2 | 0.087 |
| zombie adjacent (1,210) | 0.883 | 0.452 | 40.1 | 0.057 |
| night (988) | 0.923 | 0.476 | 44.7 | 0.112 |

The declared reading is **void**: the direction reads held-out real successors at 0.8968, under the
0.9 gate fixed before the run. Read only descriptively, the world fails the damage direction exactly
as it fails the fatal one — chance, ratio 35 — on a consequence it *was* shown 894 times. So "never
shown a death" does not look like the whole story. It does not settle it either.

## What this establishes, and what it does not

- **Established (valid, declared):** the world's prediction reproduces the bulk of each action's
  effect and none of the small component that decides death — within-root AUC along it falls from
  0.9975 on real successors to chance on generated ones.
- **Established (structural, verified in the sampler):** a TC-consecutive recipe can never train on
  the last 9 transitions of an episode, deaths included. Any future world on that layout inherits it.
- **Not established:** that the loss *causes* this. Small energy share plus concentrated error is the
  pattern the September 19 report once read as "the loss under-allocates" and had to retract. And the
  world is not zeroing the direction: along damage its effect has 78% of the true RMS magnitude,
  uncorrelated with the truth (0.087). Its error there is misalignment, not silence.
- **The next discriminating step is an intervention, not another probe** — and not one that changes
  two things at once. The first version of this line proposed a new loss *and* a terminal-reaching
  window together; that confounds them. Three arms separate them: the existing world (old window, MSE);
  a terminal-inclusive window with MSE (what repaired exposure buys); and exactly those windows and
  batches with a predeclared weighted MSE (what the loss adds beyond exposure). Judged together on a
  new seed block (52,000+), with generated-fitted readouts as well as real-fitted alignment.

## Where the fatal direction sits in `u`'s spectrum (added before choosing a loss)

`u`'s coordinates are PCA components; their variance on the training pool runs from 794.6 (largest)
through a median of 0.20 to 0.058 (smallest). The fatal direction `w` puts **0.6%** of its squared norm
in the 20 largest-variance components, 12% in the top 50, 54% in the top 100 and **46% in the lowest
92**; its largest weights sit at variance ranks 53–96. Plain MSE on `u` is dominated by a handful of
components this direction barely touches. Weighting each component by its inverse TRAIN variance would
raise the direction's share of the loss about **50×**. That makes a bounded, train-statistics-weighted
MSE the more direct first intervention; a cosine target normalizes the vector as a whole and need not
reach these components at all.
