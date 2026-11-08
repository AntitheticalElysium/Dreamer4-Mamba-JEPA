# Whole-screen gaze and E3 belief dissection

2026-11-03, requested by the user after E1/E2/E3. Post-result descriptive
analysis, not a new adoption gate. No world training, head refitting, policy
tuning, or examination of P1-F1.6 results.

1. Whole-screen gaze: all archived heads, with original E1 and longer E2
   endpoints retained separately for both head seeds. Compare the expensive
   native chooser on the identical frames. The natural-frame cohorts (600
   saved / 128 confirmation frames) are primary for allocation, reported
   separately. They contain 404 / 86 frames with visible entities, including
   61 / 9 with arrows; sparse subsets keep their counts and uncertainty.
   Event-balanced cohorts are secondary, never treated as natural frequencies.
   Compare fixed existing mixed, C-only, S-only, arithmetic probability mixture,
   and uniform policies; no policy chosen from these readings.

   Measure (a) expected fraction of the nine crop cells devoted to disjoint
   semantic categories; (b) probability that a crop covers an entity, conditional
   on its presence; (c) coverage of newly entering map strips, changed HUD cells,
   and edited terrain. Compare each with uniform on the same frame to account
   for screen geometry. Entity coverage and HUD-touch coverage overlap and are
   explicitly not a partition of looks. Natural cohorts, lighting, entity kind,
   proximity <=2 cells, and action families are fixed strata. Report both seeds;
   do not select the successful long-fit seed only.

2. Noise: reuse the eight exact physical states with four independent renderer
   RNGs each from the previous fixed-state test. Histories and head forecasts
   remain identical; change only current rendered evidence. Re-encode these
   32 frames with the frozen F1.5 checkpoint. Compare cheap mixed/S-only gaze
   sensitivity with the archived native test, whose forecast randomness was
   also held fixed. Report within-scene policy total variation, argmax changes,
   and map/HUD allocation. This identifies response to fresh visual noise, not
   whether all surprise-driven looks in real trajectories are noise or useless.
   Eight selected night states do not establish a population noise rate.

3. E3: recompute full-precision per-cell errors and C/S for the four saved E3
   endpoints, with no optimization. Check agreement with original float32 MSE
   and C/S archives. Decompose the global rich-minus-coarse prediction change
   into disjoint target categories (each mob kind, empty terrain, player, HUD
   body fields, inventory, blank HUD); contributions must sum to the full mean.
   Retain separate saved and confirmation cohorts and seeds. Paired episode
   bootstrap intervals accompany changes in error and policy allocation.

   Fixed secondary strata: current light <0.5 / >=0.5, episode age 8-32 /33-128
   />128, visible entity count 0/1/2+, near entity <=2 cells, previous action
   NOOP/move/DO/other, sleeping, health loss, achievement unlock, body-stat
   change, and actual counter increment versus no increment. Report exposure
   counts, per-cell effect and contribution to total error reduction; rare
   subsets are exploratory and not a multiple-comparison-adjusted discovery.
   E3 changes the auxiliary representation, not just memory length, so even a
   selective gain would not identify a memory mechanism by itself.

Instruments: normalized probabilities, nine-cell allocation sums, exact metadata
matching, no-op C=0, identical C under fixed-history renderer repeats, exact
paired zero difference controls, full-precision archive agreement, and error
decomposition closure. Bootstrap 2,000 draws by episode; never pool head seeds
as independent world seeds. GPU work inference only, hard cap 10 minutes; check
other jobs first. Discuss the results before any further experiment or repair.
