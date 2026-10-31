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

## B. Progress-of-what premises (`progress.py`, GPU after A; user 10-31: "B1/B2 are essential, launch them"; Codex
reviews the design)
SWEEP.md's first version (pixel error at consequence windows) did not measure what it claimed (TSP is same-frame
super-resolution, world 1's decoder was digit-blind, pixels exist only for the crop); PROGRESS.md section 7 replaced
it. Both readings use the ten checkpoints of world 1 and of P1-F; consecutive checkpoints are 1,562 updates apart,
about gamma-progress's horizon (gamma 0.9995: 2,000 updates). Reading only; no weight changes.

**B1. Do token targets move as much as the loss? (token drift)**
- Claim: between consecutive checkpoints, the change in a class's G1 loss (training units) reflects learning, not
  the targets changing. Competing: the encoder re-scales or re-codes the class's tokens by as much (an encoder that
  shrinks a consequence's step lowers its loss without any learning; re-coding raises it).
- Instruments, first 150 dev episodes, S1's seeded random fixations (identical across checkpoints and worlds): per
  class (M0.5's class_losses: drink up, wood up, sapling up, table placed, zombie hit, health other, HUD other change,
  HUD static, map moved, map still, glimpse, lighting; all windows, fixed noise) the G1 loss; the class's own
  frame-to-frame token change (copy-previous squared error per element, clean token units; periphery and lighting); and
  the drift from checkpoint k - 1 to k: the per-token least-squares map (per group: map cells, HUD cells, glimpse,
  lighting) fitted on even episodes, its residual per element on odd episodes, overall and per class.
- Reading (descriptive): per class and interval, relative loss change vs relative change of the class's own token
  step, and the unexplained drift residual against that step (clean units both). Token-space progress is read as
  contaminated where |relative step change| >= half the |relative loss change|, or the residual >= the step.
  Rules out token-space progress (PROGRESS candidates 1-3 in tokens) for the classes and stages where it holds.

**B2. Does looking at a consequence teach its gaze-free prediction? (transfer, first order)**
- Claim: the training signal ADDED by fixating a consequence window at the attempt frame (instead of a uniform draw)
  decreases the gaze-free loss of that consequence on fresh frames: positive, specific alignment.
- Competing: no transfer (alignment within noise); transfer not specific (any fixed look adds it: control window);
  in P1-F, transfer only before the gaze cue forms.
- Added signal: the gradient of the exact training objective (train.p1_losses -> harmonised groups + the faithful
  noise term) on chunks of the 383 train-400 episodes in the training split (change targets included): 8 prefix
  frames, the attempt frame t and 7 frames after it (so the glimpse's routes into the context, the belief and later
  losses count), Mamba state and predicted beliefs carried from the episode start (training's truncated-BPTT state),
  history fixations = the world's training gaze (world 1: S1's seeded prior; P1-F: the rule's window on rule-decided
  frames). Only frame t's fixation varies: the counter's window (7, column), the faced cell's window, a control
  window (1, 1) covering neither, or a uniform draw (4 per group); common random numbers across conditions.
  Classes: drink, wood, sapling (coin), stone (deterministic, outside the rule: unguided control); awake DO facing the
  block, counter below 9, tool held. 96 contexts per class in 6 episode-disjoint groups of 16 (at most 4 per episode).
- Targets, on fresh dev frames of the class (first 400 dev episodes, 2 episode-disjoint halves of 64): the denoising
  loss of the frame's counter cell and faced cell (periphery tokens, training units, the frame's query fixation a
  uniform draw, history under the same uniform prior, context at the inference context level, the frame at a
  p_train sigma), and the reward readout's NLL; belief and percepts of the 7 frames before carry gradient (the routes
  training can change). Bulk target: all periphery cells of random dev frames.
- Alignment: <P Delta, g_T>, Delta = the condition's gradient minus the uniform mean; positive = the step lowers the
  target loss. Per module (perception, belief, denoiser, pixels, tsp, change, outcome). Scales: the ordinary look's
  transfer <P g_uniform, g_T> and direct training on the target <P g_T^A, g_T^B> (cross halves). SE across the 6
  groups (halves averaged), the half-to-half spread reported beside it.
- Amendment before launch (CPU smoke, 10-31): P in three geometries, because the first (Adam linear at the
  checkpoint's v_hat, the loss accounting convention) is fragile: its per-parameter step runs from a median of ~5 to
  lr / eps ~1,000 (world 1 checkpoint 10; 9,400 at checkpoint 2), at the ceiling on head parameters training barely
  touches (unused reward-bin rows, the change map's cell table: sqrt(v_hat) < 1e-6), where real Adam saturates (a new
  gradient enters v). Geometries: euclid (P = 1; SGD, LESS's InfSGD); adam_linear (lr / (sqrt(v_hat) + eps)); 
  adam_saturating (LESS's Gamma: the group's own clipped gradient enters v, v_new = beta2 v + (1 - beta2) g^2,
  bias-corrected; bounds each parameter's step). Xia et al. (LESS, ICML 2024, section 3) use the Adam update on the
  training side and the plain gradient on the target side, as here.
- Reading: the premise holds for a class if the counter (or faced) window's transfer to its own target is > 0 beyond
  3 SE and above both the control window's and the off-class targets' at some checkpoint of world 1, in BOTH Adam
  geometries (they bracket the linearization; disagreement = linearization-dependent, reported as such); euclid is
  reported beside them as the optimizer-free reference, not part of the reading: the world trains with Adam, and SGD
  is a different counterfactual. (Revised after the second CPU smoke, whose values are noise at 2 contexts per group
  but showed euclid's sign often opposite to Adam's; the first wording required all three.) P1-F's trajectory
  descriptive. Rules out (if null in world 1 for every class and checkpoint): the fovea's training signal
  reaching the gaze-free prediction through the parameters to first order, i.e. a physical-progress chooser would
  have no first-order reason to look at consequences. Not covered: the representation route (pixel grounding at the
  window changing the encoder, hence the targets: second order), Adam momentum, clipping (a common scale).
