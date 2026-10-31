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
Declared reading: the counter look's transfer to its own gaze-free counter target > 0 beyond 3 SE, above the control
window's (3 SE of the difference) and above its transfer to every other class's target, in both Adam geometries.

World 1 (trained under uniform gaze):
- **Holds** for drink (checkpoint 4), wood (1 and 7) and stone (4). Wood's transfer is positive and class-specific at
  every checkpoint (z 1.3-14 adam_linear, 1.3-37 adam_step); drink's is positive in both geometries at 8 of 10
  checkpoints but not specific at 1, 9, 10; stone's is strongly positive at 2-5 and 8 but helps the other counters as
  much (not specific) except at 4.
- Size: the counter look adds 2-51% (adam_linear) and 9-74% (adam_step) to the transfer the same frame gives under a
  uniform look, larger late in training, when the ordinary transfer is small.
- **Sapling: unmeasurable.** Its gaze-free target gradient does not reproduce across halves (reliability -0.31 to
  0.53); no significant transfer in both geometries at any checkpoint.
- **The reward readout goes the other way.** On drink attempts the counter look makes the gaze-free reward readout
  worse at checkpoints 1-3 in both geometries (z -161 / -3.5, -6.8 / -5.2, -4.5 / -7.6; the control window's
  transfer is positive); sapling at 4 and 8 likewise. The first-order seed of P1-F's reward gate exists in a world
  trained under uniform gaze.

P1-F (trained with the rule's gaze):
- Wood holds only at checkpoint 1. From checkpoint 4 on, its transfer is NEGATIVE in both geometries and below the
  control (z -3.0 to -183 adam_linear, -2.1 to -11 adam_step): the training signal of the guided look pushes the
  gaze-free wood prediction the wrong way. PLAN's competing explanation "transfer only before the gaze cue forms"
  is supported for wood. Drink: the geometries disagree (adam_step positive to checkpoint 6, adam_linear not) until
  checkpoint 9, negative in both. Stone (outside the rule): negative in both at checkpoint 5 (z -3.1 / -3.1).
- Reward: negative across classes and checkpoints, stone included (drink 2: z -21 / -19; sapling 10: -424 / -20).

Fixed vs moving targets (counter_full minus counter, adam_linear): in world 1 small (|difference| <= 0.09 against
transfers up to 0.25): a token-space signal reads the look about as the prediction does. In P1-F's wood it is +0.1 to
+0.75 against fixed transfers of -0.4 to -2.8 (re-coding hides a quarter of the harm, e.g. checkpoint 5: -2.8 fixed,
-2.0 moving); in P1-F's late drink it flips the sign (checkpoint 10: -0.085 fixed, +0.27 moving; the moving target's
half-reliability is low there, 0.21-0.33).

What B establishes and does not:
- Establishes, to first order and under the checkpoint's Adam state: in a world trained under uniform gaze, looking at
  a deterministic consequence's counter adds a specific, positive training signal to its gaze-free prediction (a
  physical-progress chooser would have a reason to look there); the same look also starts teaching the reward
  readout to depend on the glimpse; in a world trained with deterministic placement, the look's signal turns against
  the gaze-free prediction once the cue has formed.
- Does not establish: multi-step effects, what a learned chooser would do, or anything about mining (not among B's
  classes) beyond A's field deficit.

## Across A, B and D (descriptive; no design selected)
- A learning-driven fovea has something to reach: deterministic visible consequences are field-limited (A), and in a
  uniform-gaze world looking at them adds specific gaze-free signal (B2). Mining is the largest field deficit and
  was never fixated.
- Two levers are outside the fovea: the sampler (mobs; sapling's coin once learned) and memory (cooldown).
- The gate is a training-signal phenomenon with a measurable seed: the reward readout's dependence on the glimpse
  appears at first order even under uniform gaze (B2), and once placement is deterministic the look's signal turns
  against the gaze-free prediction (P1-F wood). A chooser's reward measured on the gaze-free prediction would register
  that turn as negative progress; one measured on the glimpse would not.
- Progress measured on tokens is unreliable early (B1, scale changes) and can hide or flip the sign of a gated look
  (B2, P1-F): the gaze-free prediction at fixed targets, or a fixed-target space, is the safer "of what".
- D: no VoI substrate yet on P1-F; the HUD reading is ceilinged because the periphery carries the digits.
