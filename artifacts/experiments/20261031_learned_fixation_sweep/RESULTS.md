# Premise readings for the learned fovea: results (10-31)

PLAN.md's declared readings. A, B and D complete.

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

## B. Progress of what? (`progress.py`, commits 79bdd057 + 205fc694, unit fovea-progress-chain, 39 min; `progress/b1.json`, `progress/summary.json`)
Run after Codex's partial review and my audit (PLAN B, "Review and audit before launch"). The first full launch ran
out of GPU memory at checkpoint 1 (two target graphs held at once; the 2-row GPU smoke could not show it); fixed in
205fc694 (one graph at a time, 8 rows per backward, same group means); peak 4.19 GiB.

### B1. Token drift (150 dev episodes, checkpoints 1-10 of each world)
- Nonlinear re-coding is negligible: the least-squares map from checkpoint k-1's tokens leaves a residual of at most
  8% of a class's own frame-to-frame step (typically 0-1%), at most 3.4% of the token variance (P1-F's glimpse, 1->2),
  ~1e-4 by 9->10.
- The targets' SCALE moves, early: the declared test (|dlog step| >= half |dlog loss| on change classes with |dlog
  loss| >= 0.02) marks 19 of 65 tested class-intervals contaminated in world 1 and 12 of 65 in P1-F, all but three in
  the first half of training (1->5). World 1's 4->5 interval moves together (drink, wood, zombie hit, health, HUD
  change, map moved; health changes' step doubles, dlog +1.0, while their loss rises +0.54). After checkpoint 5 the
  targets are stable, but most classes' losses change by less than 2% per interval ("not tested").
- Reading: token-space progress is uninterpretable for about a third of class-intervals in world 1's first half
  (a fifth in P1-F's), where a target's scale change accounts for at least half its loss change; late in training
  the targets are stable and progress is small. The movement is re-scaling, not re-coding.

### B2. Transfer: does the look teach the gaze-free prediction? (96 contexts per class in 6 episode-disjoint groups; fresh targets 2 x 64)
**Correction (10-31, night): the first version of this section read wrong z-scores.** `summary()` indexed
`delta[:, j, [i0, i1], :, q]`: a list and an integer separated by a slice, so NumPy moved the halves axis to the front
and the "SE over the 6 groups" was the spread of the 2 target halves (n = 2, one degree of freedom). Estimates and
module splits were right; every SE and z was not (hence values like -183 and -424). Fixed in progress.py (groups axis
kept first; per-group values now saved); summary re-run from the saved arrays: all 12,240 estimates and module splits
identical, B1 identical, one SE checked by hand. The declared reading is re-applied below with the group SE and the
paired (per-group) difference from the control, as PLAN states; the first version also left out the faced window,
which PLAN's reading includes. Retracted from the first version: world 1 "holds for wood (1, 7) and stone (4)";
P1-F "wood holds at 1", "NEGATIVE from checkpoint 4 beyond 3 SE (z -2.4 to -183)", "drink negative in both at 9",
"stone negative at 5", "reward negative across classes (z -424)"; "re-coding hides a quarter of P1-F's wood harm".

Declared reading: the window's transfer to its own gaze-free target > 0 beyond 3 SE, above the control window's (3 SE
of the paired difference) and above its transfer to every other class's target, in both Adam geometries (SE over 6
groups, 5 degrees of freedom).

World 1 (trained under uniform gaze):
- **Holds at three points:** the drink counter at checkpoint 3; the faced cell of wood (the tree, which stays) at 4;
  the faced cell of stone (stone -> path, mining's map-side consequence) at 9.
- **Below the bar, consistent in sign.** Drink's counter transfer is beyond 3 SE in both geometries at 7 of 10
  checkpoints (1-4, 7, 9, 10), above the control at 1-3, not specific at 1, 2, 9, 10 (the look helps the other
  counters too). Wood's counter transfer is positive at every checkpoint in both geometries but beyond 3 SE only in
  adam_step (1, 2, 4), never in adam_linear (z 0.8-1.8). Stone's map-side transfer is positive and class-specific at
  4-10 in both geometries, beyond 2.5 SE in both at 5-10 (z 2.5-4.9; against the control 2.2-5.5): the steadiest
  positive result of B2. Stone's counter: adam_step only (1, 2).
- Size (estimates, unaffected by the correction): the counter look adds 2-51% (adam_linear) / 10-74% (adam_step) for
  wood and 5-19% for drink at 1, 2, 4, 5 (72-112% at 3) to the transfer the same frame gives under a uniform look; the
  faced look adds 12-34% for stone's map cell at 5, 6, 8 (ratios above 1 late in training are ratios of small numbers).
- Route: 80-100% of each positive counter or map transfer goes through the denoiser, most of the rest through the
  belief.
- **Sapling: unmeasurable**, as before: its gaze-free target gradient does not reproduce across halves (reliability
  -0.31 to 0.53).
- **The reward readout.** On drink attempts the counter look makes the gaze-free reward readout worse beyond 3 SE in
  both geometries and below the control at checkpoints 2 and 3 (z -10.0 / -6.9, -6.9 / -5.8; against the control
  -5.9 / -4.3, -4.3 / -4.2), through the belief (~80%) and the reward head; not at 1 (-2.9 / -2.9) or later. Sapling at
  1 (-9.2 / -4.9) but not clearly below the control (-2.8 / -3.6); wood and stone no consistent sign. The first-order
  seed of P1-F's reward gate exists under uniform gaze, for drink, early.

P1-F (trained with the rule's gaze):
- **Nothing beyond 3 SE in either direction** on any counter or map target in both geometries. The sign changes:
  wood's counter transfer is negative in both geometries at every checkpoint from 4 to 10 (adam_linear z -2.0 to -2.9,
  adam_step -0.5 to -1.5; the look's added signal is -0.2 to -1.0 times the ordinary look's), where world 1's is
  positive at all ten. The checkpoints share contexts and targets, so the seven are not independent replications.
  Stone's map-side transfer, steady in world 1, is near zero from checkpoint 4, where P1-F's field already carries
  more of stone (A: 0.65 vs 0.34 at sigma 6.67). Drink: no consistent sign.
- Reward: drink worse beyond 3 SE in both geometries at 3, 4, 5 and 8 (below the control at 3-5: z -4.1 / -3.3,
  -3.6 / -3.1, -3.3 / -3.1), adam_linear alone at 2, 6, 7, 9, 10; the other classes negative in sign, not significant.
  At those checkpoints the route is perception (50-70%) and the reward head (20-30%), not the belief as in world 1.

Fixed vs moving targets (counter_full minus counter, adam_linear, paired over groups):
- World 1: |difference| <= 0.09 against transfers up to 0.93 (often consistent across groups, always small): a
  token-space signal reads the look about as the prediction does.
- P1-F wood: +0.00 to +0.75 at 4-10 against fixed transfers of -0.4 to -2.8, within noise (paired z ~1.2): no measured
  masking.
- P1-F drink: the moving target reads the guided look significantly more negatively early (checkpoints 3-4: -0.66,
  -1.52; paired z -4.5, -4.2) and significantly more positively late (8-10: +0.19 to +0.35; z +3.1 to +4.3), where the
  fixed transfer is -0.06 to -0.12 (z -0.7 to -2.0) and the moving one +0.14 to +0.27 (z +1.3 to +2.9). The
  representation route changes the token-space reading of the guided look, in both directions over training.

What B establishes and does not:
- Establishes, to first order, under the checkpoint's Adam state, at 96 contexts per class: in a uniform-gaze world,
  looking at a deterministic consequence adds training signal to its gaze-free prediction with a consistent positive
  sign (drink's counter early, mining's map cell late, wood's counter throughout), meeting the declared bar at three
  points; the counter look also makes the gaze-free reward readout worse for drink early; in P1-F the sign of the
  look's transfer to the gaze-free wood prediction turns negative and stays there, no single checkpoint significant.
- Does not establish: a large or reliable per-look effect (most transfers are within 3 SE of zero at this sample
  size); multi-step effects; what a learned chooser would do; coal or iron (not among B's classes).

## Across A, B and D (descriptive; no design selected; revised with B2's correction)
- A learning-driven fovea has something to reach: deterministic visible consequences are field-limited (A), and in a
  uniform-gaze world looking at them adds gaze-free training signal of a consistent positive sign, small next to the
  noise at this sample size (B2). Mining is the largest field deficit (A), was never fixated, and its map-side
  consequence carries B2's steadiest positive transfer.
- Two levers are outside the fovea: the sampler (mobs; sapling's coin once learned) and memory (cooldown).
- The reward readout's dependence on the glimpse has a first-order seed under uniform gaze (drink, early, through the
  belief); under deterministic placement the look's transfer to the gaze-free wood prediction changes sign. A chooser
  rewarded on the gaze-free prediction would see a negative sign there, one rewarded on the glimpse would not; the
  size of that difference is not established.
- Progress on tokens: B1 (unchanged) finds the targets' scale moving in about a third of world 1's early
  class-intervals; B2 finds the moving-token reading of P1-F's guided drink look diverging from the fixed-target
  reading in both directions. The fixed-target gaze-free prediction remains the safer "of what"; B2's support for it
  rests on one class, not two.
- D: no VoI substrate yet on P1-F; the HUD reading is ceilinged because the periphery carries the digits.
