# Premise readings for the learned fovea: results (10-31)

PLAN.md's declared readings. A and D complete; B running (its section follows when it ends).

## A. What part of failure 3 could fixation reach? (`decompose.py`, commit 03afde3d, unit fovea-decompose, 1 h 31 min; `decompose/results.json`)
Instrument check: world 1 under random gaze reproduces the 10-29 generation diagnosis on its five classes (risky
recorded 0.127, field at sigma 6.67 0.026, production 0.002, Heun 32 0.025; 10-29: 0.127 / 0.025 / 0.0018 / 0.022).
Columns: the recorded rate of the class's consequence f (start -> start + 1 in the counter's token, 1 = always), the
field's conditional mean at sigma 2 / 6.67 (law of total expectation: equals the recorded f for the minimum-MSE
denoiser), production 3-step samples, Heun 32 on the same contexts and draws.

| class (contexts) | recorded | world 1 field / prod / Heun32 | P1-F random gaze | P1-F rule gaze |
|---|---|---|---|---|
| drink (3,000) | 1.00 | 0.30, 0.18 / 0.13 / 0.25 | 0.21, 0.08 / 0.05 / 0.14 | 0.79, 0.78 / 0.75 / 0.63 |
| wood (3,000) | 1.00 | 0.38, 0.24 / 0.15 / 0.20 | 0.16, 0.04 / 0.03 / 0.04 | 0.95, 0.96 / 0.94 / 0.84 |
| stone (3,000; not in the rule) | 1.00 | 0.43, 0.34 / 0.25 / 0.30 | 0.73, 0.65 / 0.62 / 0.52 | same as random gaze |
| coal (1,184; not in the rule) | 1.00 | 0.27, 0.05 / 0.02 / 0.06 | 0.49, 0.09 / 0.03 / 0.13 | same |
| iron (669; not in the rule) | 1.00 | 0.33, 0.03 / 0.01 / 0.06 | 0.54, 0.08 / 0.01 / 0.13 | same |
| sapling (3,000; coin ~5%) | 0.03 / 0.045 | -0.01, -0.02 / -0.02 / 0.02 | 0.02, -0.01 / -0.01 / 0.01 | 0.04, 0.03 / 0.005 / 0.09 |
| risky damage (3,805) | 0.13 | 0.07, 0.03 / 0.002 / 0.03 | 0.11, 0.05 / 0.03 / 0.11 | 0.14, 0.13 / 0.11 / 0.19 |
| adjacent damage (3,000) | 0.18-0.19 | 0.08, 0.09 / 0.06 / 0.10 | 0.07, 0.03 / 0.01 / 0.06 | 0.12, 0.10 / 0.07 / 0.12 |

(sapling's recorded f: world 1's reader 0.03, P1-F's 0.045; the gain rate is 0.051 in both. Bias SEs 0.002-0.011.)

Mobs on random frames (S1's frozen readers; real / production / Heun 32): world 1 zombies 0.45 / 0.25 / 0.59, cows
0.52 / 0.27 / 0.63, skeletons 0.15 / 0.16 / 0.16, arrows 0.08 / 0.15 / 0.05; P1-F (either gaze) zombies 0.45 / 0.26 /
0.50, cows 0.53 / 0.29 / 0.62, skeletons 0.16 / 0.05 / 0.22, arrows 0.08 / 0.14 / 0.14.

Hidden state (damage by the simulator's cooldown readiness, recorded vs field at 6.67): world 1 risky ready 0.39 vs
0.03, not ready 0.004 vs 0.02; adjacent ready 0.33 vs 0.04, not ready 0.03 vs 0.14 (wrong way round, as 10-29). P1-F
rule gaze risky ready 0.39 vs 0.17, not ready 0.01 vs 0.11.

Declared reading, class by class:
- **Field deficits** (bias below -0.05 beyond 3 SE at sigma 2-6.67): drink, wood, stone, coal, iron, risky and adjacent
  damage, in world 1 and in P1-F under random gaze. These are the training-signal deficits a fixation chooser could
  reach. Mining is the largest and was never fixated by the rule: coal and iron are carried at 0.27-0.54 at sigma 2 but
  0.03-0.09 at sigma 6.67, where sampling starts. Under P1-F's own gaze the field carries wood (0.95-0.96) and most of
  drink (0.78-0.79): looking at a consequence repaired the field for the classes the rule covered (gated, Codex's test).
  Stone, not in the rule, is better in P1-F under either gaze (0.34 -> 0.65 at 6.67): a recipe effect, not looking.
- **Sampler deficits** (field near the recorded rate, production below): the mobs. Heun 32 on the same field restores
  zombies and cows (and overshoots: 0.59 / 0.63 vs 0.45 / 0.52); P1-F's missing skeletons (0.05 vs 0.16) come back at
  0.22. So P1-F's skeleton regression is in sampling, not in what the field learned. Sapling under P1-F's own gaze:
  the field carries the coin (0.04 / 0.03 vs 0.045), production suppresses it (0.005), Heun 32 overshoots (0.09).
  Risky damage in world 1 also loses an order of magnitude in production (field 0.026 -> 0.002). Fixation cannot fix
  these; the sampler is the lever (10-29's exact-field result, and its finding that many-step samplers overshoot from
  the production start).
- **Hidden state:** damage's field ignores the cooldown in every condition (world 1 even the wrong way round). Under
  P1-F's own gaze the risky field matches the recorded mean (0.13) without tracking readiness (0.17 vs 0.11 where the
  record is 0.39 vs 0.01): the right average from a label-derived cue, not the mechanism. Memory is the lever.
- **Declared threshold's limit:** sapling's whole rate (0.03-0.05) is below the 0.05 threshold, so "bias below -0.05"
  cannot be met; world 1's field carries none of it (-0.015 vs 0.030 at 6.67, about 10 SE). The threshold was written
  for unit-scale consequences.

What this rules out: expecting a fixation chooser to fix the mobs (sampler), the cooldown (memory), or the sapling
coin once the field has it (sampler). What remains reachable: every deterministic visible consequence, mining first.

## D. Is there a substrate for value of information? (`substrate.py`, commit 03afde3d, unit fovea-substrate, CPU 2 h 43 min; `substrate/`)
P1-F checkpoint 10, the 10-29 probes unchanged (250 dev episodes, S1's seeded random fixations, identical draws to
world 1's run: the same 64,141 day / 30,707 night exposed cells). Different model outputs (TSP NLL 2.29 vs 2.19 nats /
pixel, per-cell correlation 0.80).

| reading | world 1 | P1-F |
|---|---|---|
| F1 one-step retention, normalized-weight IPW gain (nats / pixel), all day | +0.009 (0.018) | +0.009 (0.008) |
| same, all night | -0.007 (0.016) | -0.007 (0.015) |
| same, strict static terrain day / night | -0.013 (0.018) / -0.018 (0.012) | -0.003 (0.008) / -0.007 (0.010) |
| HUD cells, lag-1 retention day / night (glimpsed by the random draws only) | +0.002 (0.010) / -0.007 (0.010) | +0.003 (0.005) / -0.001 (0.005) |
| F2 state dependence R2 (per-frame best vs best fixed fixation), day / night / night with a mob | 0.056 / 0.112 / 0.123 | 0.070 / 0.078 / 0.140 |
| same as % of the next-frame loss | 0.61 / 0.26 / 0.25% | 0.57 / 0.08 / 0.12% |

Declared reading: no stratum's retention beyond 3 SE, and R2 nowhere twice world 1's: no substrate for VoI on P1-F
either, by these instruments. Two qualifications. (1) Under random gaze, off P1-F's training distribution on the
rule's frames (stated in PLAN). (2) The HUD reading now has a ceiling: P1-F's periphery tokens carry the digits at
12-19x the within-value spread (follow-up geometry, 10-31), and TSP reads the periphery, so neither the belief nor a
glimpse has anything left to add at the HUD. The glimpse's evidence role, if any, is in map content at night, where
its one-step value fell (0.26 -> 0.08% of the loss).
