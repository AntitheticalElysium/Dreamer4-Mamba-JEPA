# M0.5: guided fixation during joint world training (plan, 2026-10-30; revised after Codex's review of b90e61dc)

## Question
User, 10-30: a fovea pre-trained to look where the world model is starved of learning (learning progress; value of
information later). M0.5 tests the premise with hand-selected placements standing in for that chooser: if fixation
goes to the consequences the world misses, during JOINT world training, does the world learn to generate them?
Starting from P1 world 1's checkpoint 10 matches the second phase of the user's curriculum: movement is learned
(generated shift 0.91), the sparse consequences are not.

- **Claim.** Guided fixation on every attempt frame, with P1's own joint update, improves the direct glimpse
  prediction of the looked-at consequences. Perception, the belief and the periphery are MONITORED: none needs to
  improve, and a regression beside a consequence gain is a trade-off weighed by hand (user, 10-30).
- **Competing explanations.** (a) Continued joint training at a restored learning rate does the same under random
  fixation. (b) The gain stays in the guided glimpse and does not reach the periphery. (c) Other modules or other
  content regress. (d) The fixation acts as a predictive context cue: placement is decided from the state at t - 1
  (action, faced block, adjacent zombie, sleep), so it cannot select the outcome's random draw, but the rule reads
  simulator state that a learned belief need not retain, so "the eye is on the counter" can itself predict "a
  consequence is likely" (Codex). UNRESOLVED by this design: arm R does not separate it (a route putting the eye on
  the cell through a window other than the rule's would; not included).
- **Control.** Arm R: the identical fine-tune with the random prior (same base, data order, noise, updates, learning
  rate, clipping); only the fixation of attempt frames differs. Run BY HAND, decided after dissecting arm G (user,
  10-30: no automatic gate). guided_1500 - base is descriptive only.

## What changed from v2 (`../20261030_fovea_placement`), and why
- Every module trains, on P1's full objective through the harmoniser (G1 next percept; G2 change map, TSP, fovea
  reconstruction; G3 reward, continuation), clipped over all parameters as P1 (user and Codex, 10-30: v2
  backpropagated G1 alone, so the pixel grounding that teaches perception what to preserve trained nothing, and the
  Mamba's weights could not learn to retain or use the glimpse; DECISIONS: the encoder trains jointly). Codex
  verified finite gradients in every module and the intended routes (G1, TSP, reconstruction) on the base.
- Placement on every attempt frame (Q = 1): 13.09% of train frames placed (v2: 6.55%, exactly its coin-selected
  half, same windows; checked, and by Codex on 16,384 stream records: all 2,020 eligible attempts placed, order and
  RNG matched, resume reproduced). Predicted coverage of each class's own cells (from v2's corrected map; realised
  coverage is logged): drink 1.00, wood 1.00, sapling 0.97, table 0.97, zombie hit 0.78; all targets 0.95 (v2 0.53,
  random prior 0.11). v2's half-placement came from the chooser's 10-24 random floor, which has no role here.
- Readouts: the direct effect (the generated glimpse) is primary; the rest is monitored; drink and wood split into
  appearance (0 -> 1) and increment (v -> v + 1) (the v2 aggregate hid opposing changes).
- Clipping: v2 clipped the denoiser alone, so Adam saw gradients ~20x its second-moment estimate (the denoiser moved
  1.34 in its first 300 updates against 0.57 in P1's last 625). P1's all-parameter clip is restored; whether a
  transient remains is a hypothesis (Codex), checked by the logged per-module update norm. Reference: one P1-regime
  update at 1e-5 moves the denoiser 0.0073 (smoke run); at 5.8e-5 a steady Adam ratio predicts ~0.042.

## Fine-tune (`finetune.py`)
P1 world 1 checkpoint 10 (model, AdamW moments, streams, Mamba carry, RNG); P1's update (train.p1_losses ->
harmoniser, all parameters, clip 1.0 over all); 16 streams x 256 steps in 4 microbatches; learning rate linear from
the checkpoint's 1e-5 to 5.8e-5 (P1's at checkpoint 5) over 100 updates, then constant, each group keeping its
lr_scale; 1,500 updates (6.1 M frames, 24% of P1, ~2.2 h); checkpoints at 300, 750, 1,500 with exact resume state,
each followed by P1's monitor. The prior is drawn for every frame in both arms (v2's GuidedStreams). Logged every 25
updates: losses, harmoniser sigmas, gradient norm and clip, per-module update norm, guided share, coverage. The resume
identity hashes this script, the canonical P1 sources, the placement and target files, and the imported helpers
(v2's finetune.py, placement_map.py, map.json).

## Readouts (`evaluate.py`; per checkpoint, its own encoder; v2's label-defined rows, seeds and generation)
Dev split; up to 1,000 rows per class: test F (drink, wood, table, sapling, move), test A (adjacent, risky: damage),
failed attempts (drink 141 available, table 4,097 -> 1,000). Route `guided`: every attempt frame of dev fixated by
the rule; route `periphery`: the S1-seeded fixations.
1. **Prediction (primary: glimpse; secondary: periphery).** Guided route: rows whose consequence cell is inside the
   target frame's window, the generated glimpse token; periphery route: rows whose cell is outside it, the
   generated periphery cell. The POPULATION is fixed by the inputs alone (Codex, 10-30: v2/b90e61dc averaged over
   generated tokens the reader accepted, so a generator could drop its failures; 10 verified successes among 100
   rows with 90 unreadable read as 100%): the route condition and a qualified instrument (glimpse reader at the
   row's window centre; table readers) whose classes hold the row's start and end. Each generated sample is a
   verified success (readable, read == the event's end value), a verified copy (read == start), another readable
   value, or UNKNOWN (its token outside the reader's real support). Reported over the whole population: verified
   success, unknown share, bounds [verified, verified + unknown]; the rate among readable samples is a diagnostic
   only. The recorded event rate is over the same population. Failed attempts: verified false change.
   Instrument check per row: the same instrument on the row's REAL token against the value actually observed at t
   (Codex: the b90e61dc reference assumed health either took the hypothesised loss or stayed, misstating 30 of the
   first 1,000 rows of each damage class, e.g. 3 -> 4), with the real-token rejection share. Guided route: the
   generated periphery cell read beside the generated glimpse (coherence).
2. **Perception (monitored).** The same rows' real dev frames, guided route, eye on: pixel NLL (nats / pixel) of the
   consequence cell under the fovea reconstruction (glimpse tokens) and under TSP (predicted belief + periphery);
   the window's 9-cell mean beside it. Pixel space is fixed, so checkpoints compare directly. Caveat: these targets
   are legible in the periphery (HUD values exact even at 2 x 2 pixels, acuity EDA), so TSP here measures
   sharpening, not anticipation, and the glimpse adds a sharper target more than new information.
3. **Action routing (diagnostic; NOT memory).** Drink, wood, table, guided route, eye on: the generated glimpse with
   the belief recomputed from the stored tokens vs recomputed after a NOOP at t - 1, the denoiser's own action input
   kept, same draws; routing = verified success(recomputed) - verified success(NOOP belief). Codex checked that only
   the current predicted belief changes (earlier beliefs identical): it measures use of the latest action through
   the belief, which a system using only the current percept and action could show; not retention.
4. **Belief content (monitored; base and 1,500).** S1.2's belief probe (seed 0) on S1-seeded train-400 / dev
   encodings: departed-entity retention by time since departure, HUD values, achievements. Limits: it measures what
   the belief keeps of entities that left view, not what it keeps of glimpses; for these targets glimpse retention
   cannot be read at all (the periphery shows the counter every frame).
5. **Regression panel.** Dev reward NLL, continuation BCE and TSP NLL over all frames; movement (MOVE rows' generated
   shift); dev G1 by class (own token space, descriptive); reader qualification and periphery value separation per
   transition; at base and 1,500 the S1.3-style entity counts with S1.1's reader protocol refitted on the
   checkpoint's own tokens (real vs generated per kind); P1's monitor G1 by level (fine-tune log).
Saved per checkpoint (Codex, provenance): per-row arrays, the fitted readers with supports and qualification
(`LABEL_readers.pt`) and the S1 reader (`LABEL_s1_reader.pt`, S1's own persistence), bound to the checkpoint's hash;
results.json records the hashes of every source and label file the evaluation imports or reads.

## Instruments (`instruments.py`; fitted per checkpoint on its real train-400 tokens, qualified on its real dev tokens)
Periphery values: nearest value mean. Glimpse values: one MLP per (counter cell, window centre) on the cell's own
token (>= 4,000 AdamW steps), qualified per centre; rows at an unqualified centre are outside the population.
Tables: an MLP on the periphery token + lighting and one on the glimpse token + slot. Bar: accuracy >= 0.99 (tables:
recall and specificity >= 0.99 on faced cells). Support: a token is readable only within the 99th percentile of the
real tokens' distances to the predicted value's mean (>= 30 real tokens per value).
Base checkpoint (`checks/instruments_base_per_centre_steps.json`): periphery 1.000 (all four); glimpse drink
>= 0.9995, wood 0.995-0.998, sapling >= 0.997, health 1.000 at every centre; table periphery 1.000 / 0.99994, glimpse
0.991 / 0.9995. Designs rejected on the base checkpoint's real tokens before any M0.5 model existed (the bar never
moved): one MLP per cell pooled over centres (wood 0.963, health 0.979), one on the window's 9 tokens (drink 0.905,
wood 0.885), per-centre MLPs trained 30 epochs (~120 steps; drink 0.72-0.96).

## Declared readings (`readings.py`; classification only, no gate)
Per comparison B - A, paired by rows and draws on rows in both checkpoints' populations, episode bootstrap, 95%
intervals (evaluation rows only):
- Prediction, per group (drink / wood appearance, table: fixed outputs; drink / wood increment), on verified success
  (UNKNOWN is not a success), MEI = 0.25 x (A's recorded rate - A's verified success): POSITIVE (est >= MEI, lower >
  0), DETECTED (lower > 0, est < MEI), NEGATIVE (upper < 0), BOUNDED NULL (holds 0, upper < MEI), INCONCLUSIVE (holds
  0, upper >= MEI). Flag UNKNOWN-HEAVY when either checkpoint's unknown share exceeds 0.10 or the unknown difference's
  interval excludes 0: the verified comparison may then be biased, and the bounds and the among-readable diagnostic
  are read with it. Sapling and damage are stochastic: calibration, generated beside recorded rate. Failed attempts:
  verified false change MORE / LESS / NO CHANGE.
- Perception: BETTER / WORSE / NO CHANGE by the NLL difference's interval. Action routing: MORE / LESS / NO CHANGE.
- Regressions: every panel difference listed; none gates.

## Order
1. Arm G to 1,500. 2. Evaluate base, guided_0300, guided_0750, guided_1500 (~1 h from measured components: per
checkpoint encodings and readers 1.5 min, generation ~8-12 min (v2's evaluation: 49 min for 4 checkpoints with more
classes), action routing ~2 min, perception and class losses ~1-2 min; at base and 1,500 also the S1 encodings, reader
refit, entity counts and belief probe, ~5 min). 3. Dissect with the user. 4. Arm R by hand, if decided:
`finetune random 1500`, then evaluate random_1500 with guided_1500.

## Limits
One world seed, one base checkpoint, a privileged rule (labels), one learning rate and budget. The intervals cover
evaluation rows, not training noise (GPU nondeterminism behaves like a small seed change). The targets are legible
in the periphery (Perception caveat). Fixation as a context cue stays unresolved (competing explanation d). A positive
result says guided fixation with joint training can teach this model these consequences; not that a label-free
chooser would find these places, nor what training from scratch would do. Carried from v2's review (Codex, 10-30):
v2's "all added signal went to the glimpse output", "the bulk degraded the periphery", "copying costs almost
nothing" and the "midpoint" accounting for half placement are withdrawn.

## Review of b90e61dc (Codex, `../20261030_m05_review/REVIEW.md`) and the changes
1. Primary denominator depended on which generated samples the reader accepted (also the recorded rate, failed
   attempts, paired and memory comparisons): the population is now input-defined, UNKNOWN is its own category, bounds
   and the among-readable diagnostic are reported, the recorded rate is over the population, pairs use rows in both
   checkpoints' populations. Verified on Codex's counterexample (`checks/verify_review1.json`): verified success 0.10,
   unknown 0.90, bounds [0.10, 1.00], recorded 0.10, among-readable 1.00 (diagnostic only).
2. "Memory reliance" measured the latest action's route through the belief: relabelled ACTION ROUTING (diagnostic);
   S1.2's belief probe added as the belief-content reading, with its limits.
3. Health real-token reference: now the observed value at t; it equals the labels on Codex's rows, where the old
   reference misstated 30 of 1,000 per damage class (reproduced).
4. Wording: perception and the belief monitored, not co-primary; fixation as a context cue left unresolved.
5. Provenance: the fine-tune identity hashes the imported helpers; readers and the S1 reader are saved per
   checkpoint; results.json records source and label hashes.

## Pre-launch checks (done)
- Fine-tune smoke (3 updates per arm, b90e61dc; 1 update after the identity change): coverage 0.93-0.97 (guided) vs
  0.08-0.11 (random), guided share 0.11-0.13, clip factor 0.05-0.10 (P1's regime), per-module update norms logged,
  5.1-5.5 s per update, peak 3.8 GiB. P1's monitor restores the training mode and uses its own generators.
- Placement file: 13.09% of train frames; v2's placements are exactly its coin-selected half, same windows.
- Instruments qualified on the base checkpoint (above).
- Evaluator smoke on 30 dev episodes with base and v2's ft_1500 (a checkpoint without cfg, as the fine-tune saves;
  `checks/eval_smoke`), corrected version, end to end with readings: perception and regression identical for the two
  (only v2's denoiser differs), as they must be; real-token checks 1.0 wherever read; real-token rejection 0 on
  every drink, wood, table and sapling group, but 21-37% on the damage (health) rows, a calibration-only class, on
  30 episodes (sparse low-health supports; reported per group in the full run); on the one failed drink (drink at
  9) the generated glimpse reads 2 / 2 / 5 / 4 while the real glimpse reads 9 (the generation, not the reader);
  wood at centre 68 and the glimpse table reader do not qualify on 30 episodes (outside the population there; they
  qualify on the full split); several smoke groups are UNKNOWN-HEAVY (periphery unknown 0.08-0.33 on 30 episodes).
  Bugs found by smoke runs and fixed: data.gather misuse in the perception readout; the missing per-row real-token
  control.
- Not separately tested: resume of the M0.5 fine-tune with optimizer steps (its logic is v2's; Codex verified the
  stream resume).

## For review
- MEI = 25% of the gap to the recorded rate, carried from v2; a doubling from a low base (v2's glimpse drink
  appearance, +0.098) is DETECTED, not POSITIVE.
- The learning rate (5.8e-5, P1's at checkpoint 5) is carried from v2; with the full objective it moves every module.
- The support radius (99th percentile) decides UNKNOWN; its real-token rejection is reported per group.
