# Fovea placement fine-tune: results (10-30, seed 1, arm G only)

## Run
Arm G (guided placement, PLAN.md) from P1 world 1 checkpoint 10, 1,500 updates (6.1 M frames, ~2.5 passes over train),
09:38-11:45. Manipulation check passed: 53.3% of the 406,638 target frames had the consequence cell in the glimpse
(prior: 11.2%), 6.6% of frames guided; 217 k covered consequence frames, against ~190 k in all of P1. P1's monitor
G1: 0.01908 (base) -> 0.01904 / 0.01853 / 0.01758 at 300 / 750 / 1,500. Arm R not run (user, 10-30: not on this
result; the method is to be dissected first).

## Stage 1 (declared readings, `readings.py`, committed before the full evaluation was read)
Primary route (periphery, eye elsewhere), G1500 - base, 95% intervals (evaluation rows only):

| Class | MEI_p | Field 6.67 | MEI_r | Reader production | Verdict |
|---|---|---|---|---|---|
| drink | 0.203 | -0.010 [-0.014, -0.006] | 0.239 | -0.005 [-0.012, 0.002] | BOUNDED NULL |
| wood | 0.188 | +0.012 [+0.007, +0.016] | 0.229 | +0.002 [-0.006, 0.010] | BOUNDED NULL |
| table | 0.126 | -0.043 [-0.049, -0.038] | 0.183 | -0.031 [-0.044, -0.018] | BOUNDED NULL |

Failed-attempt gate holds (false change: drink -0.033 [-0.049, -0.016], table -0.003 [-0.006, -0.001]).
**Stage 1: NULL.** Guided placement on half the attempt frames, with the restored learning rate, does not raise the
generation of these consequences by the minimal effect within 1,500 updates. Caveat (PLAN): if fine-tuning alone
degrades them, placement could still protect them; untested (no arm R). Table got worse (field -0.043, reader
-0.031): as the 10-29 accounting predicts for a bulk-dominated update (table's drive reversed, +128 / 1,000 steps at
checkpoint 10), not separated from placement without R. Guided route (eye on) the same: drink +0.002, wood +0.037
(field) / +0.011 (reader), table -0.036.

## Dissection (`dissect.py`, `token_change.py`; user, 10-30: global metrics may mislead)
1. **The failure is value-specific.** Generated consequence by the counter's start value (periphery, eye elsewhere,
   base; G1500 the same within noise):

   | start value | 0 | 1 | 2 | 3 | 4 | 6 | 7 | 8 |
   |---|---|---|---|---|---|---|---|---|
   | drink generated | 0.31 | 0.004 | 0.00 | 0.00 | 0.035 | 0.00 | 0.00 | 0.014 |
   | drink train events | 4,518 | 6,715 | 7,536 | 6,955 | 5,507 | 2,876 | 2,010 | 1,319 |
   | drink token step | 4.94 | 1.94 | 1.74 | 3.59 | 2.21 | 1.38 | 2.51 | 1.20 |
   | wood generated | 0.30 | 0.022 | 0.011 | 0.003 | 0.058 | | | |
   | wood train events | 17,041 | 18,323 | 15,891 | 11,893 | 3,677 | | | |
   | wood token step | 5.65 | 1.67 | 1.59 | 3.45 | 2.11 | | | |

   The renderer (craftax_classic/renderer.py) draws an empty cell at 0 and the icon plus a digit glyph (lower-right
   60%) at v >= 1: 0 -> 1 makes the icon appear; v -> v + 1 changes the digit only. Generation succeeds almost only
   on 0 -> 1. Not frequency: drink 0 -> 1 is rarer than 1 -> 2, 2 -> 3, 3 -> 4; wood 0 -> 1 and 1 -> 2 are matched
   (17.0 k vs 18.3 k) at 0.30 vs 0.02. Not step size alone: 3 -> 4 is a large step (3.6 / 3.5, step / within-value
   spread ~14-15, the same as 0 -> 1) and is generated 0.00 / 0.003. Token step = |mean(v + 1) - mean(v)| of the base
   encoder's real 16-d periphery tokens, 200 dev episodes; every value is read at accuracy 1.0 on real tokens.
   Table (a fixed new object, step 4.21) is generated on 0.24 of placements (recorded 0.97).
2. **What is generated instead:** the unchanged value (drink 0.75, wood 0.81, table 0.76) or, for the counters, some
   other value (drink 0.21, wood 0.10). The same "other" rate appears on failed drinks (false change 0.13): the drink
   cell's samples are noisy, not discriminating. Tables are rarely invented (0.004 of failed placements).
3. **Noise level:** the drink projection is 1.00 / 0.97 / 0.86 / 0.64 / 0.44 / 0.32 / 0.19 at sigma 0.05 / 0.2 /
   0.377 / 0.7 / 1.2 / 2 / 6.67 (wood alike); movement holds 0.91-0.93 at every sigma. G1500's profile is within 0.02
   of the base's at every sigma.
4. **Not churn:** per-context success is stable (correlation base vs G1500 0.80 drink, 0.87 wood, 0.85 table; 93% of
   drink contexts never succeed in either); rows gained and lost: drink 12 / 20, wood 26 / 22, table 30 / 79.
5. **Glimpse output (eye on):** the qualified glimpse reader gives drink 0.043 (base) / 0.053 / 0.045 / 0.038
   (300 / 750 / 1,500): the glimpse output did not learn drink either. Wood: 22 qualified rows only. The ridge glimpse
   projection on GENERATED tokens swings 0.40 / 0.95 / 0.49 / 0.08 across checkpoints while the reader is flat: it
   qualified on real tokens but is not valid on generated ones, and is not read.
6. **Loss vs generation:** dev G1 loss per element, base -> G1500: drink -3%, wood -10%, sapling -8%, zombie -5%,
   table +9%; the bulk -4% to -12% (glimpse -10%, HUD static -12%). Wood's loss fell 10% with no change in its
   generation.

7. **The direct effect, split by transition (post hoc, user 10-30: measure the glimpse, where the manipulation
   acted).** Drink, paired G1500 - base, 95% intervals (evaluation rows only; one run, so training noise is not in
   them):

   | | glimpse output, eye on (qualified reader) | periphery, eye elsewhere |
   |---|---|---|
   | 0 -> 1 | 0.088 -> 0.186, +0.098 [+0.044, +0.152] (204 rows) | 0.31 -> 0.27, -0.033 [-0.089, +0.023] (182) |
   | v -> v + 1 | -0.018 [-0.026, -0.011] (1,582 rows) | -0.001 [-0.005, +0.002] (1,384) |

   The glimpse output learned part of the appearance and none of the increments; the periphery received neither.
   The aggregate glimpse figure (0.043 -> 0.038) hid both. Wood and table: no adequate glimpse instrument (22
   qualified wood rows; none for the table).

## Design limits raised by the user (10-30)
- The primary readout was the transfer (periphery, eye elsewhere), not the direct effect (glimpse output), which
  is measurable only for drink.
- Placement on 6.6% of frames: attempt contexts (13.1% of frames, 95% of target events) times a coin of 1/2. The
  1/2 came from the chooser's random floor (10-24), which has no role in a substitution test, and from keeping the
  periphery readout in distribution. All attempt frames would have covered ~95% of target events instead of 53%.
  Events themselves occur on 6.6% of frames.
- Encoder, belief and heads frozen (targets fixed, readers valid, deficit localised in the denoiser by A / E): this
  tests only the denoiser's share of what looking provides, not perception or memory learning from the glimpses.

## Reading
"Drink" and "wood" were never one problem each: the world generates the appearance of a counter (0 -> 1) on ~30% of
events and its increment (v -> v + 1) almost never, whatever the training frequency, and the guided arm changed none of
it, in the periphery or in the glimpse output. Step size is not the separating variable (3 -> 4 fails at a 0 -> 1-like
step). Candidates, not established: the increment's target depends on the current value (a value-to-value mapping
read from history), the appearance's does not (one fixed output, like the table); or something specific to the
empty-cell state. Limits: one seed, one base checkpoint, privileged placement, no arm R.
