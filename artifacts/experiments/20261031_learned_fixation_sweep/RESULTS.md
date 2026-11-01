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
  checkpoint spacing (1.5-2.5 k updates) is gamma-progress's horizon (2,000 updates).
- Cross-world agreement of the interval's progress vector (both worlds saw the same seed-1 stream): weak for the DO
  classes (-0.4 to +0.6), 0.61-0.78 for random rows at intervals 5-8: consistent with data order moving generic
  content, not established.
- Over 4-interval spans some located progress persists: P1-F drink counter +99 (7) at 2-6 and +25 (3) at 6-10 (rule-
  confounded); world 1 stone +15.5 (3.6) and +2.9 (0.9).

What G establishes: under random looks, the look-local predictions' progress is replicable across episodes and is
not pulled by night static, but it does not locate action consequences better than the world's own action
contingency, does not move toward them over training, and in the second half reverses sign between consecutive
intervals at gamma-progress's horizon. What it does not: the dynamics under a chooser that follows progress
(positive feedback), the factorised glimpse stage (stood in for by the joint denoiser), the remote counter in a clean
world, more than one seed.
