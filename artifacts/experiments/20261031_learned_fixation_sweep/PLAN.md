# Premise readings for the learned fovea (plan, 10-31; user: "run A, B and D now"; Codex reviews post hoc)

SWEEP.md section 6 named three cheap readings that test the premises of P1-F2 before any design is chosen. All three
read frozen checkpoints; nothing is trained. Operational: one GPU job at a time, and Codex is running its own
experiments (grounding-loss review). A starts only when the GPU is idle; D runs on CPU (4 threads, MemoryMax 8G).

## A. What part of failure 3 could fixation reach? (`decompose.py`, GPU)
- **Claim.** P1-F's remaining consequence deficits split by cause: some are the trained field's (its conditional
  mean underweights the consequence: a training-signal deficit, the only kind a fixation chooser can address), some
  the sampler's (the field carries the rate, production samples do not), some hidden state (the field ignores the
  zombie cooldown).
- **Competing.** All of them are field deficits (fixation could reach every one), or all sampler deficits.
- **Instruments** (generation diagnosis 10-29, unchanged): A's law-of-total-expectation calibration on context
  classes (the field's conditional mean f(D(x0 + sigma eps)) vs the recorded f, classes defined from state t - 1 and
  a_{t-1} only) with D's sampler comparison (production 3-step Euler vs Heun 32 on the same contexts and draws).
  Classes: 10-29's RISKY, ADJACENT, DRINK, SAFE, RANDOM, plus WOOD (DO facing a tree), SAPLING (DO facing grass, a
  ~5% coin in dev), STONE and COAL (DO with a wood pickaxe), IRON (stone pickaxe); the functional is each class's HUD
  counter (start value -> start + 1). Diamond has 12 dev contexts: not read. Mobs: S1's frozen readers count
  zombies, cows, skeletons and arrows on RANDOM's real successors, production samples and Heun-32 samples.
- **Controls.** SAFE and RANDOM bias near 0; world 1 under random gaze must reproduce 10-29's numbers on its five
  original classes (instrument check); the same contexts and noise draws for both samplers.
- **Conditions.** World 1 under random gaze (its training gaze); P1-F under random gaze (gaze-free) and under the
  rule's gaze (its training gaze, label-derived).
- **Declared reading, per class.** Field deficit: bias below -0.05 beyond 3 SE at sigma 2-6.67 (as 10-29). Sampler
  deficit: field mean within 0.05 of recorded while production is below it by more than 3 SE, and Heun 32 closer.
  Hidden state: the damage classes' field mean insensitive to cooldown readiness (10-29's split). Mobs: Heun 32
  restores counts toward the recorded -> sampler; it does not -> field. Descriptive beyond that; no gate.
- **Rules out.** Expecting a fixation chooser to fix the classes that read as sampler or hidden state.

## D. Is there a substrate for value of information? (`substrate.py`, CPU)
- **Claim.** P1-F's glimpse now carries the digit (geometry 1.5-2.0x vs 0.25-0.40x), so the world uses glimpse
  evidence: positive one-step retention through the belief (F1) and a larger state-dependent value of looking (F2)
  than world 1's.
- **Competing.** As world 1: no retention (one-step contrast within noise) and a tiny value of looking (0.25-0.6% of
  the next-frame loss), so VoI has nothing to work with yet.
- **Instruments** (fovea probes 10-29, unchanged): measure.py on P1-F checkpoint 10, first 250 dev episodes, S1's
  seeded random fixations (the randomised design the contrasts need), seeds 2029; analyse.py; Codex's corrected F1
  one-step inverse-probability-weighted contrast and F2 sets (check_saved_outputs.py). Control: world 1's existing
  run (same script, episodes, seeds).
- **Caveat.** Random gaze is off P1-F's training distribution on rule frames (~13%) and the histories behind them;
  stated, not corrected.
- **Declared reading.** F1 one-step gain beyond 3 SE (any stratum, HUD or night especially) or F2 R2 at least twice
  world 1's -> a substrate exists; otherwise VoI still has none on this world.

## B. Would a progress signal see consequences? Redesigned, NOT launched (needs the user's go-ahead)
SWEEP.md's version (pixel error at consequence windows across world 1's checkpoints) does not measure what it claims:
TSP predicts the crop from the SAME frame's periphery tokens plus the belief (super-resolution, not dynamics), world
1's decoder was digit-blind (no pixel progress possible at counters), and pixel targets exist only for the crop. The
replacement, which follows from the "progress of what" reasoning in PROGRESS.md, is proposed there.
