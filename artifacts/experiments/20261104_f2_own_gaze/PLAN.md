# P1-F2 under its own gaze (user 11-04: run, then report)

**Why.** F2's world was evaluated only under the fixed random gaze and M0.5's label-derived guided rule (P1-F2 review
section 5). Its glimpse stage, outcome heads and belief trained on chooser-selected windows (80% learned, 20% uniform).
Precedent: P1-F's own-gaze re-read (`../20261030_p1_guided/own_gaze.py`, 0920b660) showed its reward / HUD regression
read under random gaze was the evaluation gaze, while its map-dynamics regression was real. F2's regressions (wood,
stone, zombie, wake, crafting slices; S1.3 C2ST .920 -> .968) may likewise be partly off-policy evaluation.

**Gaze.** F2 checkpoint 10's saved online head (`fixation_controller`), applied along every real dev episode exactly as in
training (`../20261103_p1f2/chooser.py`: the existing coarse encoding without a crop, the same eight-frame history with
episode-start padding, actual and NOOP forecasts in float32, C / S window maps, the existing combined-score policy
0.8 softmax(.5 C/mean + .5 S/mean) + .2/49, sampled with an independent seeded generator 612401). Frame 0 of each
episode (no previous action) keeps the seeded prior draw, as in training. The head is frozen; it never sees fine pixels
or labels. Each world is otherwise read through unchanged instruments; the only change is the encoding's fixations.

**Readings.**
1. Actor interface (`../20261030_actor_interface/interface.py`'s classes, futures and summaries, K = 4, its seeds), in ONE
   process for pairing: `p1f16` (random gaze: F1.6's own training gaze), `p1f2` (random), `p1f2_own` (own gaze). Paired
   differences on identical histories and future draws: p1f2_own - p1f16 (primary), p1f2_own - p1f2. Instrument check:
   `p1f16` and `p1f2` reproduce the archived `../20261103_p1f2/readings/interface_random.json` in every numeric field.
2. S1 in full for F2 with its encoding bound to the own gaze (as the precedent): S1.3 C2ST, power controls, reader counts,
   reward calibration primary; S1.1 / S1.2 / S1.4 descriptive. F1.6's and F2-random S1 exist (`../20261102_p1f16/run/s1`,
   `../20261103_p1f2/run/s1`).
3. Gaze descriptives on dev: share of frames whose window differs from the seeded prior, window histogram, crop coverage
   of any visible mob (simulator labels, reading only) under own vs prior gaze.
Not included, deliberately: M0.5's evaluator and Codex's counter panel fix the target crop at the counter by design, and
past glimpses barely reach the belief (uptake readings), so the history gaze would hardly move them.

**Declared reading.** Compare each world under its own training gaze (F2 own vs F1.6 random). Where an F2 regression read
under random gaze shrinks to within its paired interval or reverses under own gaze, it was the evaluation gaze; where it
stays, it is a property of the recipe (acquisition changed what the world learned). Both are reported per class; no class
is dropped. Under own gaze F2 also gets easier fixations for its own heads; the random-gaze reading stays the robustness
reading, not replaced. One world seed each; intervals are evaluation uncertainty, not training-seed uncertainty.
