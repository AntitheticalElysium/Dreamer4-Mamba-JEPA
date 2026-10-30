# M0.5: guided fixation during joint world training (plan, 2026-10-30, for Codex's review before launch)

## Question
User, 10-30: a fovea pre-trained to look where the world model is starved of learning (learning progress; value of
information later). M0.5 tests the premise with hand-selected placements standing in for that chooser: if fixation
goes to the consequences the world misses, during JOINT world training, does the world model as a whole learn them
better than under random fixation: perception (what the glimpse preserves), memory (what the belief carries into the
prediction) and prediction (what the denoiser generates)? Starting from P1 world 1's checkpoint 10 matches the
second phase of the user's curriculum: movement is learned (generated shift 0.91), the sparse consequences are not.

- **Claim.** Guided fixation on every attempt frame, with P1's own joint update, improves the direct glimpse
  prediction of the looked-at consequences, with perception and memory reliance moving with it.
- **Competing explanations.** (a) Continued joint training at a restored learning rate does the same under random
  fixation. (b) The gain stays in the guided glimpse and does not reach the periphery. (c) Other modules or other
  content regress. (d) The fixation acts as a cue: placement depends on the state at t - 1 only (action, faced
  block, adjacent zombie, sleep), never the outcome, so it carries no outcome information.
- **Control.** Arm R: the identical fine-tune with the random prior (same base, data order, noise, updates, learning
  rate, clipping); only the fixation of attempt frames differs. Run BY HAND, decided after dissecting arm G (user,
  10-30: no automatic gate; "G vs base" cannot hide or establish attribution, it is descriptive).

## What changed from v2 (`../20261030_fovea_placement`), and why
- Every module trains, on P1's full objective through the harmoniser (G1 next percept; G2 change map, TSP, fovea
  reconstruction; G3 reward, continuation), clipped over all parameters as P1 (user and Codex, 10-30: v2
  backpropagated G1 alone, so the pixel grounding that teaches perception what to preserve trained nothing, and the
  Mamba's weights could not learn to retain or use the glimpse; DECISIONS: the encoder trains jointly).
- Placement on every attempt frame (Q = 1): 13.09% of train frames placed (v2: 6.55%, exactly its coin-selected
  half, same windows; checked). Predicted coverage of each target class's own cells (from v2's corrected map
  statistics; realised coverage is logged): drink 1.00, wood 1.00, sapling 0.97, table 0.97, zombie hit 0.78; all
  targets 0.95 (v2: 0.53; random prior 0.11). The half-placement came from the chooser's 10-24 random floor, which
  has no role in a substitution test.
- Readouts: the direct effect (the generated glimpse) is primary; perception and memory reliance are co-primary;
  the periphery is secondary (extent of transfer); drink and wood split into appearance (0 -> 1) and increment
  (v -> v + 1) (the v2 aggregate hid opposing changes); a regression panel over every module.
- Clipping: v2 clipped the denoiser alone, so Adam saw gradients ~20x its second-moment estimate (the denoiser moved
  1.34 in its first 300 updates against 0.57 in P1's last 625). P1's all-parameter clip is restored; whether a
  transient remains is a hypothesis (Codex), tested by the logged per-module update norm. Reference: one P1-regime
  update at 1e-5 moves the denoiser 0.0073 (smoke run); at 5.8e-5 a steady Adam ratio predicts ~0.042.

## Fine-tune (`finetune.py`)
P1 world 1 checkpoint 10 (model, AdamW moments, streams, Mamba carry, RNG); P1's update (train.p1_losses ->
harmoniser, all parameters, clip 1.0 over all); 16 streams x 256 steps in 4 microbatches; learning rate linear from
the checkpoint's 1e-5 to 5.8e-5 (P1's at checkpoint 5) over 100 updates, then constant, each group keeping its
lr_scale; 1,500 updates (6.1 M frames, 24% of P1, ~2.2 h); checkpoints at 300, 750, 1,500 with exact resume state,
each followed by P1's monitor. The prior is drawn for every frame in both arms (v2's GuidedStreams), so everything
but the attempt frames' fixation matches. Logged every 25 updates: losses, harmoniser sigmas, gradient norm and clip,
per-module update norm, guided share, coverage.

## Readouts (`evaluate.py`; per checkpoint, its own encoder; v2's label-defined rows, seeds and generation)
Dev split; up to 1,000 rows per class: test F (drink, wood, table, sapling, move), test A (adjacent, risky: damage),
failed attempts (drink 141 available, table 4,097 -> 1,000). Route `guided`: every attempt frame of dev fixated by
the rule; route `periphery`: the S1-seeded fixations.
1. **Prediction (primary: glimpse).** Guided route, the consequence cell inside the target frame's window: the
   generated glimpse token read by the checkpoint's glimpse reader; success = the recorded end value (counters) or a
   table (faced cell), copy = the start. Secondary: periphery route, the cell outside the window, the generated
   periphery cell. Failed attempts: false change. Every evaluated row is also read on its REAL token by the same
   instrument (instrument check, per row group: it must reproduce the recorded outcome; v2 had this, my first M0.5
   draft dropped it and the smoke run exposed the gap), and on the guided route the generated periphery cell is read
   beside the generated glimpse (coherence of the two outputs).
2. **Perception (co-primary).** The same rows' real dev frames, guided route, eye on: pixel NLL (nats / pixel) of the
   consequence cell under the fovea reconstruction (from the glimpse tokens) and under TSP (predicted belief +
   periphery); the window's 9-cell mean beside it. Pixel space is fixed, so checkpoints compare directly. Caveat:
   these targets are legible in the periphery (HUD values exact even at 2 x 2 pixels, acuity EDA), so TSP here
   measures sharpening, not anticipation, and the glimpse adds a sharper target more than new information.
3. **Memory reliance (co-primary).** Drink, wood, table, guided route, eye on: the generated glimpse with the belief
   recomputed from the stored tokens vs recomputed after a NOOP at t - 1, the denoiser's own action input kept, same
   draws; reliance = success(recomputed) - success(NOOP belief). On the base, the NOOP belief removed the residual
   appearance (increment probe B1 -> B2: 0.038 -> 0.000).
4. **Regression panel.** Dev reward NLL, continuation BCE and TSP NLL over all frames; movement (MOVE rows' generated
   shift); dev G1 by class (own token space, descriptive); reader qualification and periphery value separation per
   transition; at base and at 1,500 the S1 entity counts (S1's reader protocol refitted on the checkpoint's tokens;
   real vs generated per kind); P1's monitor G1 by level (fine-tune log).

## Instruments (`instruments.py`; fitted per checkpoint on its real train-400 tokens, qualified on its real dev tokens)
Periphery values: nearest value mean. Glimpse values: one MLP per (counter cell, window centre) on the cell's own
token (>= 4,000 AdamW steps), qualified per centre; rows at an unqualified centre are not read. Tables: an MLP on the
periphery token + lighting and one on the glimpse token + slot. Bar: accuracy >= 0.99 (tables: recall and
specificity >= 0.99 on faced cells). Support: a generated token is read only within the 99th percentile of the real
tokens' distances to the predicted value's mean; else unreadable (reported, never a success).
Base checkpoint (`checks/instruments_base_per_centre_steps.json`): periphery 1.000 (all four); glimpse drink
>= 0.9995, wood 0.995-0.998, sapling >= 0.997, health 1.000 at every centre; table periphery 1.000 / 0.99994, glimpse
0.991 / 0.9995. Designs rejected on the base checkpoint's real tokens before any M0.5 model existed (the bar never
moved): one MLP per cell pooled over centres (wood 0.963, health 0.979), one on the window's 9 tokens (drink 0.905,
wood 0.885), per-centre MLPs trained 30 epochs (~120 steps; drink 0.72-0.96).

## Declared readings (`readings.py`; classification only, no gate)
Per comparison B - A, paired by rows and draws, episode bootstrap, 95% intervals (evaluation rows only):
- Prediction, per group (drink / wood appearance, table: fixed outputs; drink / wood increment), MEI = 0.25 x (A's
  recorded rate - A's success): POSITIVE (est >= MEI, lower > 0), DETECTED (lower > 0, est < MEI), NEGATIVE (upper
  < 0), BOUNDED NULL (holds 0, upper < MEI), INCONCLUSIVE (holds 0, upper >= MEI). Sapling and damage are stochastic:
  calibration, generated beside recorded rate. Failed attempts: false change MORE / LESS / NO CHANGE.
- Perception: BETTER / WORSE / NO CHANGE by the NLL difference's interval. Memory reliance: MORE / LESS / NO CHANGE.
- Regressions: every panel difference listed; none gates (user, 10-30: a large consequence gain with a regression
  elsewhere is a trade-off to weigh, not a failure by rule).
- guided_1500 - base: descriptive (placement and continued joint training not separated); pilots at 300 and 750
  descriptive. guided_1500 - random_1500 (if R is run): attribution.

## Order
1. Arm G to 1,500. 2. Evaluate base, guided_0300, guided_0750, guided_1500 (~2.5-3 h; entity counts at base and
1,500 only, their reader refit being the costliest step). 3. Dissect with the user. 4. Arm R by hand, if decided:
`finetune random 1500`, then evaluate random_1500 with guided_1500.

## Limits
One world seed, one base checkpoint, a privileged rule (labels), one learning rate and budget. The intervals cover
evaluation rows, not training noise (GPU nondeterminism behaves like a small seed change). The targets are legible
in the periphery (Perception caveat). A positive result says guided fixation with joint training can teach this
model these consequences; not that a label-free chooser would find these places, nor what training from scratch
would do. Carried from v2's review (Codex, 10-30): v2's "all added signal went to the glimpse output" is withdrawn
(the glimpse loss trains the shared transformer), as are "the bulk degraded the periphery" (mechanism unseparated),
"copying costs almost nothing" (no copy-reference loss) and the "midpoint" accounting for half placement.

## Pre-launch checks (done)
- Fine-tune smoke, 3 updates per arm (nothing saved): coverage 0.93-0.97 (guided) vs 0.08-0.11 (random), guided
  share 0.11-0.13, clip factor 0.05-0.10 (P1's regime), per-module update norms logged, 5.1-5.5 s per update, peak
  3.8 GiB. P1's monitor restores the training mode and uses its own generators (no training-state change).
- Placement file: 13.09% of train frames; v2's placements are exactly its coin-selected half, same windows.
- Instruments qualified on the base checkpoint (above).
- Evaluator smoke on 30 dev episodes with base and v2's ft_1500 (a checkpoint without cfg, as the fine-tune saves;
  `checks/eval_smoke`), end to end in 4.6 min: perception and regression readings identical for the two (only v2's
  denoiser differs), as they must be; real-token checks 1.0 on every readable group but periphery damage (risky,
  0.947 on 30 episodes); on the one failed drink (drink at 9) the generated glimpse reads 2 / 2 / 5 / 4 while the
  real glimpse reads 9 (the generation, not the reader); wood at centre 68 and the glimpse table reader do not
  qualify on 30 episodes (unmeasured there; they qualify on the full split). A bug found by the smoke run and fixed:
  the perception readout passed scattered frames to data.gather, which reads contiguous ranges; frames are now read
  from the episode's observations, checked equal to data.gather's.
- Not separately tested: resume of the M0.5 fine-tune (its logic is v2's, which passed a resume test).

## For review (points I am least sure of)
- Memory reliance is a narrow notion of memory use: it measures whether consequence generation relies on the belief
  knowing the action, not whether the belief retains what the glimpse saw (for these targets the periphery shows
  the counter every frame).
- MEI = 25% of the gap to the recorded rate, carried from v2; a doubling from a low base (v2's glimpse drink
  appearance, +0.098) is DETECTED, not POSITIVE.
- The learning rate (5.8e-5, P1's at checkpoint 5) is carried from v2; with the full objective it moves every module.
- The support check (99th percentile radius) decides which generated tokens are readable; unreadable shares are
  reported per group.
