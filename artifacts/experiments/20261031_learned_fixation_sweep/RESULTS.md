# Premise readings for the learned fovea: results (10-31)

PLAN.md's declared readings. A, B, D and F complete.

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
  0.22. So P1-F's skeleton regression is sampler-sensitive at fixed weights (qualified after Codex's review: the
  field's own mob counts were never calibrated as the counters were, so this does not show the field learned the
  right distribution, nor exclude a training change that made the representation harder for the 3-step sampler;
  Codex's frozen factorial separates step count and churn: P1-F skeletons 0.051 production, 0.107 Euler-8 with
  churn, 0.162 Euler-8 without, real reader 0.133). Sapling under P1-F's own gaze:
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
  the targets are stable and progress is small. Qualified after Codex's review (`../20261031_learning_foveation_review/
  REVIEW.md`): the residual test shows the movement is affine, not that it is harmless. An affine map can rotate the
  token basis (zero residual, zero step change, a stale denoiser's loss rising: Codex's CPU counterexample), and how
  far each fitted map is from a scaled identity was not measured. The scale part is measured; a rotation part is not
  excluded. The contamination flag compares log changes of observables on partly different frame populations: a
  description, not a decomposition of why the loss changed.

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

## F. Can the world point its own eye at where the action will act? (`locus.py`, commits 38ce9b60 + fix, unit fovea-locus, 20 min; `locus/summary.json`)
First 400 dev episodes; 300 contexts per DO class (217 failed stone attempts), 600 random frames. Belief replay for
the recorded action within 0.65-0.86% relative L2 at all four checkpoints (bar 2%). Coverage = P(the arg-max window
covers a consequence cell), episode-bootstrap SE; chance in brackets.

| world | drink (0.12) | wood (0.12) | sapling (0.12) | stone (0.31) | coal (0.27) | iron (0.22) | stone, no pickaxe |
|---|---|---|---|---|---|---|---|
| world 1, ckpt 2 | 0.11 | 0.15 | 0.10 | 0.35 | 0.23 | 0.14 | 0.29 |
| world 1, ckpt 5 | 0.27 | 0.14 | 0.16 | 0.95 | **0.93** | 0.67 | 0.28 |
| world 1, ckpt 10 | 0.38 | 0.25 | 0.22 | 0.95 | **0.93** | 0.58 | 0.37 |
| P1-F, ckpt 10 (uniform query) | 0.21 | 0.04 | 0.13 | **0.99** | **0.98** | **0.90** | 0.09 |
| world 1 ckpt 10, realized change (post-outcome reference) | **0.77** | 0.42 | 0.07 | 0.80 | **0.94** | **0.88** | 0.01 |

(bold: the declared "locates", coverage >= 0.5 and coverage - 3 SE > 3 x chance; SEs 0.01-0.03, 0.05-0.06 for the
failed attempts. Stone misses the bar at ckpts 5 and 10 by its high chance: 0.95 - 0.03 against 3 x 0.31.)

Declared reading:
- **Locates** at world 1 checkpoint 10: coal (stone at the bar, iron 0.58 short of it); also at checkpoint 5; nothing
  at checkpoint 2 (every class near chance). A from-scratch chooser has no such prior early.
- **Where the peak is.** The contingency's maximum CELL is the faced tile for wood 0.68 (world 1, ckpt 10; P1-F
  0.90), coal 0.64, stone 0.46, failed stone attempts 0.49 (P1-F 0.87), almost never the player's own cell (<= 0.03
  at world 1 ckpt 10; drink 0.16 at ckpt 5): the
  world's prediction of the next frame depends on DO most at the tile DO acts on, including trees, which do not
  change, and stone without a pickaxe, where nothing happens. The signal finds the action's locus (the contact
  landmark), not the outcome; it fires on failed attempts (matching negatives).
- **The declared circularity did not occur; its mirror did.** Consequences on the map at the locus (mining:
  block -> path) are located; consequences shown only in the HUD inventory (drink, wood, sapling) are not
  (counter coverage 0.22-0.38 in world 1, 0.04-0.21 in P1-F), although the realized change shows them after the
  fact (drink 0.77, wood 0.42). The HUD counters are what A found the field under-predicts (sigma 6.67: drink 0.18,
  wood 0.24, coal 0.05): a self-referential signal cannot point at a consequence the world does not yet predict.
  Much of the HUD mass is on OTHER HUD cells (drink: arg-max on another HUD cell 0.52): diffuse, not specific.
- **Specificity**: not "always the HUD" (the faced tile dominates for wood, coal, stone).
- **Competing signals.** Predicted change and sample spread locate mining less well (coal, stone 0.71-0.81; iron
  0.46-0.47 at ckpt 10) and the
  HUD classes no better; the coarse region measure on random frames shows no night capture for either (descriptive).
- **P1-F**: under a uniform query its own counter consequences are invisible to the signal (wood 0.04): the gate
  hides gated knowledge from a label-free chooser, as Codex's test predicts.

What F establishes: from mid-training a frozen world supplies, label-free and before the outcome, a prior that puts
the eye on the tile the action acts on, failures included. What it does not: a way to find consequences displayed
away from the locus (the inventory) from the world's own predictions; for those, only the realized outcome carries
the location. Not covered: whether looking at the locus teaches anything (B2), what a trained chooser would do.

## G. Where does the learning a look produces land, over P1? (`landscape.py`, PLAN G; units fovea-landscape3, ~30 min; `landscape/summary.json`)
F's rows (2,617: up to 300 per DO class, 217 failed stone, 600 random), 49 windows each, world 1 and P1-F checkpoints
1-10. Alignment (TSP NLL at the recorded fixation vs S.encode): median 0.03-0.11% at every checkpoint (bar 2%; the
max, 0.3-2.3%, is the half-precision belief, PLAN G amendments). Mean pixel MSE, checkpoint 1 -> 10: world 1 view
0.090 -> 0.041 (flat from checkpoint 4), TSP 0.033 -> 0.034 and reconstruction 0.033 -> 0.034 (its NLL head does not
train MSE: descriptive only, declared); P1-F view 0.043 -> 0.010 (flat from 5), TSP 0.0143 -> 0.0012 and
reconstruction 0.0083 -> 0.0009 (flat from 4-5). Progress is in units of 1e-4 pixel MSE.

Declared readings:
1. **Structured: yes.** Split-half rho of the 49-window progress vector >= 0.5 for nearly every class and interval
   of the view instrument in both worlds (0.5-0.97), and of TSP while TSP still progresses (P1-F intervals 1-5); the
   top 5 windows hold 0.13-0.45 of positive progress where progress is substantial (uniform 0.10).
2. **Not raw error: yes for the view, no for early TSP.** View: rho(progress, error) is often NEGATIVE in world 1 at
   intervals 2-6 (-0.3 to -0.85; error split-half 0.8-0.96) and mixed in P1-F (-0.5 to +0.9). TSP tracks error in
   the first intervals (0.81-0.99), then has almost no progress left. Night static does not attract progress: the
   dark rows' share of positive progress is at or below their share of error (world 1 TSP 0.17-0.26 vs 0.45-0.49;
   P1-F TSP 0.31-0.60 vs 0.45-0.79; view similar or mixed).
3. **Located: not beyond F's contingency, where a clean test exists.** World 1 (random gaze) mining, view progress
   arg-max covering the consequence: stone 0.31-0.60 (chance 0.31), coal 0.07-0.61 (0.27), iron 0.10-0.46 (0.22);
   none meets F's bar. F's contingency on the same rows: stone 0.95, coal 0.93, iron 0.58-0.67 (checkpoints 5, 10).
   The consequence excess exceeds 3 SE in some intervals with BOTH signs (world 1 stone +73 at 1-2, +11 at 3-4;
   coal -12.5 at 4-5). P1-F's drink / wood / sapling (rule-confounded and gated) do show located view progress
   (drink arg-max 0.55-0.79 in four intervals, chance 0.12), but the ERROR arg-max covers the same windows as often
   (0.63-0.86): the gated world's forward view is wrong there under a non-rule query. The remote-consequence case
   (a HUD counter in a world that is neither digit-blind nor gated) has no clean test in these two worlds.
4. **Moves: no.** World 1's consequence progress is front-loaded with everything else (interval 1-2: wood +343,
   sapling +175, stone +73, failed stone +174) and near zero or negative afterwards; random rows: the HUD first
   (+237 world 1, +107 P1-F at 1-2), then small and sign-alternating. No rising consequence share: the user's
   "first movements, then the starved events" is not what progress does under random looks.
5. **Bandit (descriptive).** Separating the progress-best window from the class's median at 2 SE takes 2-350 looks
   per interval (mostly < 100); a 10% floor gives 13-210 per class per interval (coal 21, iron 13): feasible for the
   common classes, marginal for the rare ones, even with labels pooling the contexts.

Post-hoc diagnostics (not in the plan; they qualify reading 1, they select nothing):
- **Late progress is mean-reverting at this horizon.** Consecutive intervals' class-mean progress vectors correlate
  positively early (0.3-0.94) and negatively in the second half (P1-F random rows -0.36 to -0.92 from interval 3-4;
  most classes -0.2 to -0.85 at the last intervals): a window that improved in one interval worsens in the next. The
  checkpoint spacing is equal in frames (2.56 M) but 2,500 updates through checkpoint 5 and 625 afterwards (Codex,
  11-01; corrected from "1.5-2.5 k"), so the reversals sit at 625-update intervals, a third of gamma-progress's
  2,000-update horizon: they bear on progress at that short lag, not at gamma-progress's.
- Cross-world agreement of the interval's progress vector (both worlds saw the same seed-1 stream): weak for the DO
  classes (-0.4 to +0.6), 0.61-0.78 for random rows at intervals 5-8: consistent with data order moving generic
  content, not established.
- Over 4-interval spans some located progress persists: P1-F drink counter +99 (7) at 2-6 and +25 (3) at 6-10 (rule-
  confounded); world 1 stone +15.5 (3.6) and +2.9 (0.9).

What G establishes: under random looks, the look-local predictions' progress is replicable across episodes and is
not pulled by night static, but it does not locate action consequences better than the world's own action
contingency, does not move toward them over training, and in the second half reverses sign between consecutive
intervals 625 updates apart (a shorter lag than gamma-progress's). What it does not: the dynamics under a chooser that follows progress
(positive feedback), the factorised glimpse stage (stood in for by the joint denoiser), the remote counter in a clean
world, more than one seed.

## H. Does choosing which crop is trained on speed up look-local learning? (`selection.py`, PLAN H; unit fovea-selection, 1 h 34 min; `selection/summary.json`, `results.json`)
Fresh TSP heads on frozen P1-F checkpoint-10 inputs (fit: 64 k frames of the even train-400 episodes; IL learner: the
odd ones), identical init / frames / coin flips / budget (6,000 x 512), held out on G's 2,617 dev rows at all 49
windows. Held-out pixel MSE relative to uniform (negative = better), at 1,000 / 2,000 / 3,000 / 6,000 steps, and the
steps each arm needs to reach uniform's loss at 2,000 / 3,000 / 4,500 / 6,000:

| learner, population | learnability (RHO) | error | progress (gamma) |
|---|---|---|---|
| sr, random rows, all windows | -16 / -43 / -34 / -3 %; 1515 / 2332 / 3889 / 5779 | -16 / -43 / -31 / -9 %; 1517 / 2369 / 3983 / 5526 | +37 / +20 / +63 / +61 % |
| sr, DO classes, consequence windows | -28 / -52 / -41 / -5 %; 1420 / 2244 / 3725 / 5687 | -28 / -52 / -36 / -12 % | +16 / +3 / +57 / +59 % |
| sr, random rows at night | -18 / -27 / -10 / -1 % | -18 / -28 / -8 / +2 % | +16 / +1 / +25 / +33 % |
| fwd, random rows, all windows | -6 / -9 / -4 / +2 %; 1622 / 2407 / 4167 / never | +5 / +5 / +12 / +17 % | +15 / +7 / +9 / +13 % |
| fwd, DO classes, consequence windows | -6 / -12 / -6 / 0 % | -4 / -10 / -7 / -11 %; reaches uniform's final at 3,605 | +15 / +5 / +7 / +10 % |
| fwd, DO classes, other windows | -8 / -28 / -18 / +5 % | 0 / -12 / +6 / +25 % | +37 / +13 / +14 / +31 % |

(episode-bootstrap SEs of the paired differences 0.00000-0.00047, i.e. every percentage above 2% is many SEs.)
Selections on the training frames (uniform: HUD 29%, faced tile 18%, faced tile on DO frames 18%): learnability HUD
44-45%, faced 19%, faced on DO 19-20%; error HUD 45%, faced 16-18%; progress HUD 78-80%, faced 5%.

Declared reading:
- **Claim (uniform's FINAL loss in clearly fewer steps): not met.** sr 5,779 steps (4% fewer), fwd never (2% worse at
  6,000). **Competing (a), every arm ~ uniform: not met either.** Mid-budget, learnability needs 14-24% fewer steps
  to match uniform (sr) and 7-20% (fwd), and at 2,000 steps its loss is 43% lower (sr; 52% at the DO consequence
  windows) and 9% lower (fwd). The arms converge by the end: at 6,000 x 512 samples each fit frame has been trained
  ~48 times, about once per window.
- **Regime.** P1 trains on 2.44 M frames for 25.6 M samples (10.5 passes): ~10 crops per frame, H's step ~1,250, where
  learnability is 16-43% ahead (sr) and 6-9% (fwd). A coverage analogy, not a transfer of the numbers to joint training.
- **Competing (b), a trade-off: for error on the forward learner.** Error selection buys the DO consequence windows
  (-11% at the end, uniform's final in 40% fewer steps) at the cost of everything else (+17% overall, +25% on the DO
  frames' other windows). Learnability does not trade off (+0-5% at the end, ahead throughout the middle).
- **Competing (c), noise.** For super-resolution (little irreducible error by day) error ~ learnability; for the
  forward prediction (a partly unpredictable next view) error is worse than uniform overall and at night (+10% at the
  end) while learnability is not: the irreducible-loss reference matters exactly where the target is stochastic,
  RHO-LOSS's argument. Progress in this form (bidirectional, greedy, full feedback, gamma 0.9995) is worse than
  uniform everywhere: it puts ~80% of looks on the HUD.
- **Where the selections go.** Neither learnability nor error moves looks toward the action's locus (faced tile on DO
  frames 19-20% vs 18%); they move them to the HUD (44-45% vs 29%). Locating action loci remains contingency's (F).
What H establishes: in this data, choosing crops by learnability (current loss minus an irreducible-loss reference)
makes both look-local predictors learn faster in the coverage regime P1 operates in, never worse at the end, and is
robust where error-seeking is not; temporal progress in this greedy full-feedback form hurts. What it does not: the
joint system (representation, belief, denoiser), an online predictor of the score, a matched need prior, one seed.

## H2. Does need-weighting change what learnability buys? (`selection.py --need`, PLAN H2; unit fovea-selection-need, 54 min; `selection_need/summary.json`)
Reproduction: the re-run uniform and learnability arms equal H's held-out means to every printed digit (both
learners). Belief replay for the need computation within 1.06% (bar 2%). Need puts 46% of DO-frame looks on the
faced tile (uniform 18%); need x learnability 36-42%; learnability alone 19-20%. Held-out loss relative to uniform at
1,000 / 2,000 / 3,000 / 6,000 steps (paired SEs 0.1-2.5% of uniform's loss):

| learner, target | learnability | need | need x learnability |
|---|---|---|---|
| sr, random rows | -16 / -43 / -34 / -3 | -3 / -2 / -19 / -3 | -22 / -44 / -35 / -5 |
| sr, DO consequence windows | -28 / -52 / -41 / -5 | 0 / +2 / -14 / -2 | -31 / -53 / -41 / -6 |
| sr, need-weighted, all rows | -12 / -42 / -40 / -4 | -6 / -10 / -28 / -9 | -20 / -44 / -41 / -6 |
| sr, night | -18 / -27 / -10 / -1 | +3 / +8 / +1 / +2 | -22 / -30 / -13 / -2 |
| fwd, random rows | -6 / -9 / -4 / +2 | +9 / +11 / +12 / +12 | -7 / -9 / -4 / 0 |
| fwd, DO consequence windows | -6 / -12 / -6 / 0 | +2 / +4 / +4 / 0 | -8 / -11 / -6 / -1 (uniform's final at 5,159) |
| fwd, need-weighted, all rows | -3 / -12 / -8 / +2 | +2 / +2 / +2 / 0 | -7 / -12 / -8 / -1 |

Declared reading:
- **Competing (a), need adds little beyond learnability: mostly yes.** The product is ahead of learnability alone by
  1-6 points (largest early: -22 vs -16 at 1,000 steps on sr random rows) and ends 1-3 points better on the
  need-weighted and consequence targets, the forward learner no longer behind uniform at the end (0 / -1%).
- **Competing (c), need alone suffices: no.** Need alone is far behind learnability on super-resolution and WORSE than
  uniform on the forward learner (+9 to +12% overall, +14% at night): sending 46% of DO-frame looks to the faced tile
  starves the rest.
- **The claim (product faster on the need-weighted target and at the loci than learnability alone): only marginally.**
- Defect (Codex, P1-F2 review section 3, found before this run): need's arg-max takes window 0 when need is flat
  (frames before the 8-frame context, 2.25%, and some NOOP frames); preserved as this policy's result.
What H2 establishes: in this harness learnability carries the selection gain; need does not substitute for it and
adds a small margin on top. What it does not: the realized-crop scores are outcome-dependent (Codex: a curator, not a
pre-outcome chooser), so neither arm is yet the deployable form (TEACHING 13).

## H3. Can learnability come from the system itself, before the crop? (`selection.py --arms=uniform,excess,priority,priority_raw`, PLAN H3; unit fovea-selection-endo, 20:26-21:37, clean exit; `selection_endo/summary.json`, `dissect.json`)
Reproduction: uniform equals H's held-out losses bitwise (both learners). Held-out loss relative to uniform at
1,000 / 2,000 / 3,000 / 6,000 steps (largest paired SE per cell in brackets, % of uniform's loss); H's reference-based
RHO learnability for comparison (H, identical frames / batches / coin flips):

| learner, target | excess (realized) | priority (pre-crop, predicts excess) | priority_raw (pre-crop, predicts raw error) | H: learnability |
|---|---|---|---|---|
| sr, random rows | +55 / +96 / +170 / +68 (3.5) | +5 / +58 / +144 / +75 (2.8) | +13 / -28 / -12 / +6 (1.2) | -16 / -43 / -34 / -3 |
| sr, DO consequence windows | +73 / +141 / +287 / +160 (10.6) | +7 / +81 / +226 / +148 (7.1) | -8 / -38 / -17 / +2 (1.9) | -28 / -52 / -41 / -5 |
| sr, need-weighted, all rows | +48 / +73 / +128 / +63 (3.7) | +12 / +55 / +127 / +70 (2.2) | +26 / -25 / -17 / +1 (1.3) | -12 / -42 / -40 / -4 |
| sr, night | +64 / +127 / +166 / +48 (4.3) | -3 / +62 / +120 / +44 (2.8) | +1 / -20 / 0 / +7 (1.4) | -18 / -27 / -10 / -1 |
| fwd, random rows | +38 / +41 / +42 / +29 (1.4) | +8 / +17 / +21 / +13 (0.9) | +9 / +5 / +9 / +16 (1.6) | -6 / -9 / -4 / +2 |
| fwd, DO consequence windows | +49 / +44 / +38 / +26 (1.5) | +4 / +17 / +24 / +13 (0.6) | +3 / -6 / -2 / -2 (0.7) | -6 / -12 / -6 / 0 |
| fwd, need-weighted, all rows | +42 / +43 / +42 / +30 (1.0) | +10 / +20 / +25 / +13 (0.5) | +13 / -1 / +3 / +8 (0.7) | -3 / -12 / -8 / +2 |

Steps to reach uniform's 2,000-step loss (sr, random rows): learnability 1,515, H's realized error 1,517,
priority_raw 1,731, priority 3,231, excess 3,399. Looks on HUD windows: uniform 28.5%, excess 7.9 / 7.7% (sr / fwd),
priority 13.8 / 13.5%, priority_raw 47.2 / 45.9% (H's learnability and error 44-45%). The priority head tracks its
targets: R^2 = 1 - loss / target variance 0.95-0.99 at the end for excess, 0.46 (sr) / 0.85 (fwd) for raw error;
held-batch rank correlation 0.64 / 0.65 (excess) and 0.41 / 0.52 (raw). On the evaluation rows its ranking agrees
with realized excess (rho 0.64 / 0.62) and is ANTI-correlated with realized error (rho -0.57 / -0.41); priority_raw's
agrees with error (0.38 / 0.56).

Declared reading:
- **The claim (excess recovers RHO's gain without a reference; priority keeps most of it): rejected.** Both arms
  that subtract the floor are far worse than uniform on both learners, everywhere, at every step after 500.
- **Competing (a), the floor too coarse, excess ~ error: no, the opposite.** The floor is not small beside the error;
  it dominates it (all windows, sr, end: pi^2 s^2 / 3 = 0.21 against an error of 0.0020), so excess ~ minus the floor,
  a map of position and lighting (which is why it is predictable before the crop, R^2 0.95-0.99) that points AWAY from
  error. The planned rule-out for (a) ("needs content-aware aleatoric estimation") does not follow: see the
  dissection, the floor fails for a different reason.
- **Competing (b), excess unpredictable before the crop: no** (it is very predictable, for the wrong reason). Raw
  error is partly predictable from the periphery, lighting and belief (R^2 0.46 / 0.85, rank 0.41 / 0.52).
- **Competing (c), priority works for super-resolution only: for raw error, yes.** priority_raw buys about half of
  the realized error arm's mid-budget gain on sr (269 of 483 steps at uniform's 2,000-step loss; 222 of 631 at
  3,000; none by 4,500) and ends +6%; on fwd it hurts (+5 to +16% overall), as the realized error arm did in H
  (+5 to +17%): error-seeking on the forward view chases what cannot be predicted from the previous frame, which is
  exactly what a working floor would have had to remove.

**Dissection** (post hoc additions declared at 468c8f89, after sr excess; every arm's final heads on the evaluation
rows; `dissect.json`). The uniform learner's heads came from a re-run (PLAN item 3). Its first run did NOT reproduce
H3 bitwise: max 2.6e-5 already at step 0, then within 0.4% of the mean. Traced: the encodings are identical between
the current and 60b98881's model module and between repeats in one process, so not the decision-1 code; mamba_ssm's
Triton kernels autotune their block sizes per process unless MAMBA_DETERMINISTIC=1, and with it the step-0 evaluation
equals H3's exactly. Re-run with MAMBA_DETERMINISTIC=1: uniform equals H3 bitwise on both learners (the sr IL
learner, unused by H3 and the dissection, differs by 0.07% of its mean; not chased). The dissection ran in the same
mode. By window type, the uniform learner at the end (error; the decoder's own predicted variance, exactly; pi^2 s^2 / 3;
share of saturated pixels, 0 or 255):

| sr, uniform | error | exact variance | pi^2 s^2 / 3 | saturated | saturated, mean off by > 1 bin |
|---|---|---|---|---|---|
| HUD | 0.00128 | 0.0313 (25x) | 0.49 (385x) | 44% | 12% |
| DO consequence windows | 0.00130 | 0.0126 (10x) | 0.17 | 24% | 7% |
| entity | 0.00228 | 0.0056 (2.5x) | 0.077 | 7% | 2% |
| faced | 0.00122 | 0.00085 (0.7x) | 0.00087 | 0.9% | 0.4% |
| other map | 0.00109 | 0.00082 (0.8x) | 0.00080 | 0.9% | 0.4% |

- **Where no pixel is saturated the endogenous floor works:** faced and other-map windows, exact variance 0.7-0.8x
  the realized error on sr (1.2-1.9x on fwd, whose error contains more that cannot be predicted), and pi^2 s^2 / 3
  equals it there (small interior scales, as Codex said).
- **Where pixels are saturated it fails, and under uniform looks too: the floor's construction, not the selection.**
  99.4% of pi^2 s^2 / 3 comes from saturated pixels; the exact bounded variance is 10-25x the error on HUD and
  DO-consequence windows; with the HUD digits learned as well as the map (0.00128 vs 0.00109), the scale head still
  calls them noisy. Mechanism (pixel_nll's formula, and the measured condition): an edge bin absorbs everything beyond
  it, so for a saturated pixel whose mean sits inside the range by more than half a bin, the NLL falls monotonically
  toward log 2 as s grows, rewarding an unbounded scale; faithful's squared-error mean regresses to exactly 0 / 255
  from inside about half the time, so the position-only scale inflates wherever such pixels are common (12% of HUD
  pixels under uniform; 0.4% on the map). PixelCNN++ trains its mean by the same likelihood, which can push it beyond
  the edge; our faithful mean cannot. pi^2 s^2 / 3 then amplifies it (unbounded, where the bounded variance stays
  below 1: at s = 20 it gives 1,327 against 0.98).
- **Selection then locks it in.** The floor-subtracting arms look at HUD and consequence windows less, so those are
  learned less (sr excess: HUD error 3.5x uniform's), more saturated pixels stay mispredicted (19% vs 12%) and the
  floor there doubles (exact HUD 0.064 vs 0.031); floor and error become co-located (rank rho over windows 0.37
  under uniform, 0.80 under excess). The 20% random looks did not break the loop within 6,000 steps.
- Under raw-error selection (priority_raw) the logged floor explodes (fwd, end: pi^2 s^2 / 3 = 1,037 on HUD, exact
  0.18), without consequence for that arm, which ignores it.
What H3 establishes: the faithful scale head is not a usable noise floor wherever pixels saturate (HUD, sprites'
outlines, consequence windows), under any looking policy, and a floor-subtracting score then steers looks away from
exactly the content still to be learned, a self-confirming loop; where nothing saturates, the position-only floor is
calibrated. A pre-crop predictor of raw error is learnable and buys about half of the realized error's early gain on
super-resolution, but error-seeking still hurts the forward learner. What it does not establish: whether a
well-posed floor (one whose noise estimate stays finite at the range's edges, fit from looks the score did not
choose) recovers RHO's gain; nothing was trained beyond the heads; one seed, frozen P1-F inputs.
