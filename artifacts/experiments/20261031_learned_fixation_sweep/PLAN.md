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
- Reading (descriptive): per class and interval, dlog loss = dlog(loss / step) + dlog step (the share of the loss's
  log-change explained by the class's own token step changing), and the unexplained drift residual against the step
  (clean units both). Token-space progress is read as contaminated where |dlog step| >= half |dlog loss| or the
  residual >= the step; tested only for classes whose target is a change (drink / wood / sapling up, table placed,
  zombie hit, health other, HUD other change, map moved, lighting) and intervals with |dlog loss| >= 0.02 (a declared
  floor; "not tested" otherwise). Static classes (map still, HUD static) have ~no step: residual against token variance
  only. Rules out token-space progress (PROGRESS candidates 1-3 in tokens) for the classes and stages where it holds.
  (Amended 10-31 before launch, own audit: the first test flagged static classes through a ~0 step and tested
  intervals with no loss change.)

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
  training can change), target and context tokens stop-gradient as in training (prediction at fixed targets). Also
  `counter_full`: the counter loss with the tokens moving with the encoder (the loss a token-space progress signal
  sees); full minus fixed = the recoding of the target attributable to the look. Bulk target: all periphery cells of
  random dev frames.
- Alignment: <P Delta, g_T>, Delta = the condition's gradient minus the uniform mean; positive = the step lowers the
  target loss. Per module (perception, belief, denoiser, pixels, tsp, change, outcome). Scales: the ordinary look's
  transfer <P g_uniform, g_T> and direct training on the target <P g_T^A, g_T^B> (cross halves). SE across the 6
  groups (halves averaged), the half-to-half spread reported beside it.
- Amendment before launch (CPU smoke, 10-31): P in three geometries, because the first (Adam linear at the
  checkpoint's v_hat, the loss accounting convention) is fragile: its per-parameter step runs from a median of ~5 to
  lr / eps ~1,000 (world 1 checkpoint 10; 9,400 at checkpoint 2), at the ceiling on head parameters training barely
  touches (unused reward-bin rows, the change map's cell table: sqrt(v_hat) < 1e-6), where real Adam saturates (a new
  gradient enters v). Geometries: euclid (P = 1; SGD, LESS's InfSGD); adam_linear (lr / (sqrt(v_hat) + eps): the frame
  as an infinitesimal part of a batch); adam_step (LESS's Gamma: Adam's full update for one step whose batch is this
  chunk, each condition's own gradient clipped as training clips it entering both moments, m1 = beta1 m + (1 - beta1)
  G, v1 = beta2 v + (1 - beta2) G^2, bias-corrected, minus the momentum-alone update (common to all conditions):
  the chunk as the whole batch, maximally saturating). Xia et al. (LESS, ICML 2024, section 3) use the Adam update on
  the training side and the plain gradient on the target side, as here.
- Review and audit before launch (10-31). Codex: the code labelled "adam_saturating" used ONE denominator built from
  the uniform draws' mean gradient for every condition and no first moment: not LESS's Gamma nor the difference of
  the steps the fixations would produce. Verified in the code (my comment and this plan said "LESS's Gamma":
  wrong); replaced by adam_step above. Own audit (audit_b.py, audit_b.json, CPU fp32, world 1 checkpoints 2 and 10):
  common random numbers exact (gradients equal to 1e-10); adam_step(0) = 0; the per-parameter step reaches Adam's
  cap 3.16 lr at checkpoint 2 (saturation real); module dots sum to the total; drift() returns 0 on an exact affine map
  and 0.0100 with 0.01 noise; P1-F's rule places 12.9% of train-400 frames (training: 13.1%); finite differences
  (theta - eta d, the same noise draws) match the predicted target change to 0.99-1.04 (euclid) and 0.87-0.97
  (adam_linear) at the smaller step for the fixed target with the encoder out of the step, and for the full target with
  every parameter stepped (curvature at the larger steps, shrinking with eta). With the encoder in the step, the FIXED
  target's change was 3-5x its prediction: a step on the encoder moves the tokens the fixed target holds still. So
  the representation route is FIRST order in the measured token loss (this plan said second order: wrong), hence
  `counter_full`. At world 1 checkpoint 10 (4 contexts) it adds ~19% to the adam_linear dot, 1% at checkpoint 2.
- Reading: the premise holds for a class if the counter (or faced) window's transfer to its own target is > 0 beyond
  3 SE and above both the control window's and the off-class targets' at some checkpoint of world 1, in BOTH Adam
  geometries (they bracket the linearization; disagreement = linearization-dependent, reported as such); euclid is
  reported beside them as the optimizer-free reference, not part of the reading: the world trains with Adam, and SGD
  is a different counterfactual. (Revised after the second CPU smoke, whose values are noise at 2 contexts per group
  but showed euclid's sign often opposite to Adam's; the first wording required all three.) P1-F's trajectory
  descriptive. The reading uses the fixed target (prediction); `counter_full` and its difference from the fixed
  target are reported beside it (how much of a token-space progress signal would be target recoding). Rules out (if
  null in world 1 for every class and checkpoint): the fovea's training signal reaching the gaze-free prediction
  through the parameters to first order, i.e. a physical-progress chooser would have no first-order reason to look
  at consequences. Not covered: multi-step effects, a real batch's composition (the two Adam geometries bracket it).

## F. Can the world point its own eye at where the action will act? (`locus.py`, GPU; user 10-31: "are there any tests you want to run? feel free")
Design question it serves: what a learned fovea should be taught to look at in P1, and from what signal. Brain
precedent: in manipulation, gaze goes to the contact landmark before the hand arrives and leaves it when the
contact event occurs; "the salience of gaze targets arises from the functional sensorimotor requirements of the
task" (Johansson et al., J Neurosci 2001); saccades go ahead of an expected event using learned models of the
world's dynamics (cricket bounce point; Hayhoe & Ballard, TiCS 2005). The label-free analogue: the cells where the
world's own prediction of the next percept depends on the action taken.
- **Claim.** The frozen world's action contingency of its predicted next periphery, computed from pre-outcome inputs
  only, puts its maximum window on the cells where the action's consequence will appear (the counter; the faced
  block for mining), for each DO class, without labels.
- **Competing.** (1) Not localized: the maximum falls elsewhere (mobs, HUD noise, the map) as often as chance.
  (2) Circular: localized for the consequences the field already carries (drink, wood) but not for the ones it
  under-predicts (A: coal and iron, field 0.03-0.09 at sigma 6.67), which are the ones a learning fovea is for.
  (3) Not specific: it always points at the HUD on DO, whatever the class.
- **Signal.** D(a) = the denoiser's clean estimate of frame t's periphery at production's first evaluation
  (sigma_hat = sigma_max (1 + min(S_churn / steps, sqrt 2 - 1)) = 6.67) from PURE noise (no information about the
  outcome), 2 noise draws common to all actions; context = the 8 recorded frames before t (context noise common to all
  actions); belief bp(t) recomputed for every action a_{t-1} from the Mamba carry after frame t - 1's percept token
  (replay check: the recorded action reproduces S.encode's bp within 2% relative L2, else abort); query fixation
  and history = S1's seeded uniform draws. Per cell: contingency ||D(a_taken) - D(NOOP)||^2 (sum over the 16
  channels); secondary: against the mean over all 17 actions. Window score = sum over its 9 cells; the chooser is the
  arg-max of the 49 windows.
- **Competing signals (same contexts, same draws).** Predicted change ||D(a_taken) - p_{t-1}||^2; predictive spread =
  variance over 4 production samples (the uncertainty / surprise family; expected to be captured by night static,
  Codex 10-31: night periphery noise variance 6.69x in P1-F). Post-outcome reference (not a chooser input): the
  realized change ||p_t - p_{t-1}||^2.
- **Contexts** (first 400 dev episodes; labels select and score, never enter the signal): DO attempts as A (awake,
  facing the block, counter below 9, tool held): drink, wood, sapling, stone, coal, iron; a failed attempt,
  stone_no_tool (no wood pickaxe); random frames (any action). At most 300 per class (600 random), seeded.
- **Worlds.** World 1 checkpoints 2, 5, 10 (when a from-scratch chooser could first use it); P1-F checkpoint 10 under
  uniform query (descriptive: its counter consequence is gated on the rule's window, Codex).
- **Declared reading** (per world, class, signal). Coverage = P(the arg-max window covers a consequence cell: the
  class's counter; also the faced cell for stone, coal, iron), episode-bootstrap SE; chance = the share of the 49
  windows covering one. "Locates" if coverage >= 0.5 and coverage - 3 SE > 3 x chance. Specificity: the arg-max
  CELL of the contingency equals a consequence cell (chance 1/81 per cell). Circular: drink / wood locate, coal /
  iron do not, at the same checkpoint. Failed attempts: stone_no_tool's maximum window score against stone's
  (descriptive: whether the signal fires where nothing happens). Random frames, bright (light >= 0.7) vs dark
  (<= 0.3): the share of arg-max windows by region (faced-cell windows, HUD row, rest of the map) per signal
  (descriptive: noise capture).
- **Rules out.** Locates at world 1 checkpoint 10 for no class: the world's own predictions cannot point a label-free
  action-locus prior at consequences; a chooser then needs a learned value map trained on realized outcomes (Daucé
  et al., J Vis 2020's "where" network) or progress with exploration. Circular: such a prior cannot bootstrap the
  starved consequences without an exploration floor. Locates broadly including mining and early: a cheap, label-free
  action-locus prior exists. Not covered: whether looking there then teaches anything (B2), what a trained chooser does.

## G. Where does the learning a look produces land, over P1? (`landscape.py`, GPU, frozen checkpoints; user 11-01: decision 2, "read the papers and run the test")
Design question it serves: decision 2 (TEACHING 8-9, 11), what the chooser's learned part is trained on. Under
decision 1 (agreed 11-01) a look at window w trains the predictions of what is seen at w: the glimpse stage (from
history: forward; from the periphery: super-resolution), TSP (the crop from the current periphery and predicted
belief) and the fovea encoder; the gaze-free physical stream trains on every cell wherever the eye is. The candidate
"of what" is therefore the error of those look-local predictions against the crop's PIXELS (fixed targets; B1: token
targets move early), and the candidate learned part a predictor of its progress per (context, window): IAC's
regional progress, Kim et al.'s gamma-progress (EMA reference model; beat error-seeking, RND and disagreement, no
noise fixation, ICML 2020), Kanitscheider et al.'s fast-minus-slow EMA (bidirectional), Ten et al.'s progress +
competence (Nat Commun 2021). G asks whether that signal exists in our data, in the form a chooser starting from
random looks would meet, before anything is built.
- **Rows.** F's exactly (first 400 dev episodes; up to 300 DO attempts per class: drink, wood, sapling, stone, coal,
  iron; 217 failed stone attempts; 600 random frames; F's seed). Every row is read at all 49 windows. Labels select
  rows and label windows; they never enter an instrument.
- **Instruments** (per world, checkpoints 1-10; pixel MSE in [-1, 1] against the real crop, decoded means clamped):
  - `tsp`: TSP's prediction of the crop at w from the real periphery tokens, lighting and predicted belief of frame t
    (the super-resolution part of "what will I see at w"; unchanged by decision 1).
  - `view`: the forward prediction of the crop at w from history: the denoiser's clean estimate of frame t's glimpse
    with query w at production's first evaluation (sigma_hat 6.67) from pure noise, i.e. its conditional mean given
    the 8 recorded context frames, the predicted belief and the recorded action; one noise draw per row, common to its
    49 windows and to every checkpoint; decoded through the checkpoint's reconstruction path with the estimated
    lighting. The joint denoiser's glimpse is the closest existing stand-in for decision 1's glimpse stage in forward
    mode, and is trained only by looks at w.
  - `rec`: the reconstruction from the real glimpse tokens at w (the fovea encoder and decoder's competence; read
    beside `view`, which passes through the same decoder).
  - History fixations: S1's seeded uniform draws, both worlds. Alignment check: TSP's NLL at each row's recorded
    fixation equals S.encode's within 2% (else abort).
  - Progress LP_k = e_k - e_{k+1} on the same held-out rows: fresh samples and fixed targets (Graves et al.'s target
    prediction gain), per interval k -> k + 1 (9 intervals of 2.56 M frames: 2,500 updates through checkpoint 5, 625 after (corrected 11-01), near gamma-progress's
    2,000-update horizon).
- **Worlds.** World 1 (random gaze throughout; its NLL pixel head renders map content, not HUD values: digits
  0.01-0.67, icons 0, so its HUD windows are uninterpretable). P1-F (faithful pixel head renders the HUD, but 13% of
  its training frames were fixated by the rule on drink / wood / sapling / table / zombie-adjacent contexts, and its
  glimpse is gated on the rule's window (Codex): its drink / wood / sapling rows are rule-confounded; stone, coal,
  iron, failed stone and random rows are random-look).
- **Claim** (the premises of a learned progress part). (a) Structured: within a class, progress varies across the
  49 windows replicably (odd vs even episodes). (b) Not noise: it does not follow raw error (renderer static at night,
  new terrain). (c) Located: for DO classes, progress concentrates on the consequence windows (F's definition: the
  counter; the faced tile for mining) in some interval after checkpoint 2, including the HUD counter where F's
  contingency is blind. (d) Moves: its share on consequence windows grows over training against common content (the
  user, 10-29: "at first it focuses on movements, and as those start to become learnt, it fixates on other things
  that have been starved of gradients").
- **Competing.** (1) Uniform: after the first intervals progress is spread over the windows (the shared decoder and
  denoiser improve everywhere) with no replicable ranking, so a progress predictor has nothing to learn beyond a
  uniform draw. (2) Error-tracking: progress ranks windows as raw error does. (3) Contingency-redundant: progress is
  located only where F's contingency already is. (4) Static: the composition does not change over training.
- **Controls.** Split halves by episode (ranking noise floor); the error ranking on the same rows; F's contingency
  coverage on the same rows (world 1 ckpts 2, 5, 10; P1-F ckpt 10: `locus/summary.json`); the share of windows covering
  the consequence (chance); bright (light >= 0.7) vs dark (<= 0.3) random rows (static enters only at night); world 1
  against P1-F for the rule classes.
- **Declared readings** (episode-bootstrap SEs, `gates.estimate`).
  1. Structure: per world, class, interval and instrument, Spearman rho between the two halves' 49-window mean-progress
     vectors; "structured" if rho >= 0.5; with the top-5 windows' share of positive progress (uniform 0.10).
  2. Not error: rho between the progress vector and the error vector at the interval's start, beside the error
     vector's own split-half rho; random rows bright vs dark.
  3. Located: per DO class, the consequence windows' mean progress minus the row's 49-window mean ("excess"),
     "located" if it exceeds 3 SE; the per-row arg-max's coverage of the consequence beside chance, the error arg-max
     and F's contingency (F's bar: >= 0.5 and - 3 SE > 3 x chance).
  4. Moves: the consequence excess by interval (DO classes); random rows: the excess on HUD, faced, entity windows.
  5. Bandit feasibility (descriptive): per class, the looks at the progress-best window needed to tell it from the
     class's median window at 2 SE, against those a uniform 10% floor gives it in a 25.6 M-frame run (class frequency
     in these episodes x 25.6 M x 0.1 / 49).
- **Rules out / supports.** No class structured in either world after interval 2: rules out a learned progress map
  as the location source (decision 2 reduces to contingency + floor; progress at most a global "still learning"
  schedule). Structured but error-tracking: rules out raw pixel progress without an irreducible-noise correction
  (RHO-LOSS's holdout floor, or the night stratum excluded). Structured, not error, located only where F's
  contingency is: progress is the modulation (when to stop), contingency the location. Structured, not error, located
  at the counter where contingency is blind, moving over training: supports the user's mechanism and a learned
  progress part. None of these shows that a chooser would realise it (positive feedback under its own looks is not
  in frozen readings) or that it improves the world (P1-F2 itself).
- **About the data / environment.** Where in Craftax, and when in training, the predictions a look trains improve;
  whether night static yields learnable progress.
- **Limits.** Frozen checkpoints of worlds trained under random / rule looks, not under the proposed chooser; the
  joint denoiser's glimpse stands in for the factorised glimpse stage; one seed each; coarse checkpoint spacing.
- **Cost.** Per checkpoint one S.encode of 400 episodes and ~3,000 denoiser passes over 392 streams; est. 1-1.5 h for
  both worlds. Queued after the beta-TSP readings (one GPU job at a time), smoke first.
- **Smoke (11-01, before the full run; scratchpad, world 1 ckpts 1-2, 42 rows).** Runs end to end, 30 s; TSP's NLL at
  the recorded fixation matches S.encode within 0.08-0.4% (bar 2%). Declared from it, before the full run: world 1's
  pixel head was trained by NLL, not MSE (its TSP / reconstruction MSE rose 0.0315 -> 0.0321, 0.0320 -> 0.0328 from
  checkpoint 1 to 2), so pixel-MSE progress is not its objective's progress: world 1's `tsp` and `rec` are descriptive
  only; P1-F (trained by MSE) is the primary world for them; `view` (the denoiser's forward glimpse) is read in both.
- **Run amendments (11-01, before any reading).** (1) The 8-frame context is computed once per row and its keys /
  values repeated over the 49 windows (identical inputs and draws; F's helper ran it 49 times): the stopped run's
  world 1 checkpoint 1 equals the new one (reconstruction identical; TSP mean |diff| 1.3e-5, view 1.9e-4, i.e. 0.04% /
  0.2%; TSP does not read the cache, so this is run-to-run GPU noise). (2) The alignment check aborted at world 1
  checkpoint 7 on its MAX statistic (2.3% > 2%), growing 0.3 -> 2.3% over checkpoints. Diagnosed: a full-precision
  recompute equals S.encode's stored TSP NLL exactly (0.0); rounding the predicted belief to half precision (how
  S.encode stores it, and what the instrument reads) alone moves a row's NLL by up to 2.5% in the same pass; over all
  rows the median is 0.06%, p99 0.64%. The check now bounds the MEDIAN at 2% (a misalignment shifts it) and records the
  max. Limit: the half-precision belief adds a small per-row noise to every instrument, equal across checkpoints and
  worlds. Rerun from the start.

## H. Does choosing which crop is trained on speed up look-local learning? (`selection.py`, GPU; user 11-01: "I'm not sure we should be dropping learning-progress so easily ... maybe we're missing something")
Why (TEACHING 12): G measured progress between checkpoints under uniform looks, credited to no look. The validated
forms are different: progress credited to the samples that caused it (Graves et al.'s prediction gain, the strategic
student's slope against the samples allocated, Kim et al.'s gamma-progress, Poli et al.'s per-trial information
gain), and learnability against a reference (RHO-LOSS: current loss minus an irreducible-loss model's, 18x fewer
steps on Clothing-1M; JEST, up to 13x; PLR's value loss + staleness, +28% on Procgen). P1 is an offline data-selection
problem (the chooser picks which crop of a stored frame becomes a training target), the setting RHO-LOSS / JEST address.
The premise every such chooser rests on is causal and untested here: selecting crops makes the look-local predictions
learn faster than uniform crops.
- **Learners.** Fresh TSP heads (adapter + belief projection + cell embedding + PixelDecoder), identically seeded,
  P1-F's faithful loss, on frozen P1-F checkpoint-10 inputs (train-400; periphery, lighting, predicted belief): `sr`,
  the crop at w from frame t's periphery (super-resolution, decision 1's glimpse stage near sigma_p = 0); `fwd`, the
  crop at w of frame t from frame t-1's periphery and lighting and frame t's predicted belief (which has read a_{t-1}):
  the next fine view through the action (decision 1's forward mode).
- **Arms** (the same frames, batches and coin flips; 6,000 steps x 512 frames, refit.py's budget; per frame one window
  trained): uniform; learnability (RHO-LOSS approximation 2: current loss minus the IL learner's loss, the IL learner
  trained uniformly for the same budget on the other half of the episodes, never updated); error (current loss:
  Haber et al.'s adversarial signal, the noise-prone control); progress (|EMA-copy loss - current loss|, Kim et al.'s
  gamma 0.9995, bidirectional as Kanitscheider / MAGELLAN / Ten). Non-uniform arms take the arg-max window with
  probability 0.8, a uniform one otherwise (MAGELLAN's final epsilon 0.2). Full feedback (all 49 windows scored per
  frame) is what offline P1 allows; an online agent would need a predictor of the score (not tested here).
- **Readings** (held-out: G's rows, dev first 400 episodes, all 49 windows, steps 0-6000; episode-bootstrap SEs):
  1. Global: mean pixel MSE over all windows of the 600 random rows (the uniform-target distribution), per step; steps
     for each arm to reach uniform's final value.
  2. Consequences: the DO classes' consequence windows (G's definition), same.
  3. Night vs day (random rows): does the arm buy day or night error.
  4. Selection statistics on the training frames: share of trained windows covering the faced tile (all frames; DO
     frames), the HUD, night frames, against the uniform arm.
- **Claim.** Learnability selection reaches uniform's final held-out loss on (1) and (2) in clearly fewer steps.
- **Competing.** (a) No exploitable redundancy for this learner: every arm ~ uniform (Graves et al.'s implicit
  curriculum). (b) Selection buys the selected crops only: (2) improves, (1) worsens (a trade-off). (c) Error selection
  is captured by night static (3, 4) and is slower on (1); progress at this horizon chases fluctuations.
- **Control.** The uniform arm; the IL learner's own held-out loss; day / night strata.
- **Rules out.** Learnability ~ uniform on (1) and (2) for both learners: crop selection has no data-efficiency value
  for look-local learning in this data (the learned part of a chooser buys nothing; contingency / floor suffice).
  Learnability > uniform: the selection premise holds here (supports a learnability chooser; does not show the joint
  system benefits, nor that an online predictor of the score can be learned). Error ~ learnability: no noise problem
  here at this scale.
- **About the data / environment.** How redundant the 49 crops of a Craftax frame are for these predictions, and
  whether the learnable part is where consequences are.
- **Limits.** Frozen representation (P1-F checkpoint 10), the head not the joint system; full feedback; one seed;
  the IL learner is this architecture on half the episodes (RHO used a smaller model: its ablations kept the gains).

## H2. Does need-weighting change what learnability buys? (`selection.py --need`, GPU; user 11-01 evening: "feel free to launch it now")
H (RESULTS H) left one open term of TEACHING 12b's priority, need x learnability: H had no need (contingency) arm, and
learnability alone did not move looks toward the action's locus (faced tile on DO frames 19-20% vs uniform 18%).
- **Need** (declared working form, untuned): per frame t, 1 + C_w / mean_w C, C_w = F's action contingency of frame t
  over window w's cells (||D(a_{t-1}) - D(NOOP)||^2, D the clean estimate at production's first evaluation from pure
  noise, the 8 recorded context frames, the belief recomputed per action, one draw common to both actions), on the
  frozen P1-F checkpoint 10 for the fit (even) episodes; 1 where t < 8. The base 1 keeps every window's need at least
  its average-contingency half (Codex's additive point); no source fixes the base, so it is part of the arm.
- **Arms** (one invocation, same frames / batches / coin flips / budget / learners as H): uniform and learnability
  (re-run: the reproduction check against H, GPU non-determinism expected), need (the arg-max need window), need x
  learnability (need times RHO's learnability); epsilon 0.2 uniform as before.
- **Readings** as H, plus the need-weighted held-out loss (sum_w need_w loss_w / sum_w need_w on G's rows, need from F's
  saved P1-F contingency on the same rows) and the selections' faced-tile share on DO frames.
- **Claim.** need x learnability keeps learnability's speed on the uniform target (H's readings 1-2) and is faster on
  the need-weighted target and at the action loci than learnability alone.
- **Competing.** (a) need adds nothing beyond learnability (the loci are learned through generalisation: H's
  consequence windows already gained most); (b) need costs the uniform target (looks pulled to loci); (c) need alone
  is as good as the product (learnability adds nothing once need directs the looks).
- **Rules out.** (a): the need term is not worth its two denoiser evaluations per frame in P1 (learnability + floor);
  (c): learnability is not worth its reference model (contingency + floor); claim: keep the product.
- **Limits.** As H (frozen P1-F inputs, heads only, full feedback, one seed); the need is computed from the frozen
  world, not the world being trained; the base is untuned.
