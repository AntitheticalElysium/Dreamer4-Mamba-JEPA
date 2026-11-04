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

## Amendment (11-04, Codex's review; user: stop, fix, rerun): closed-loop generated window
**Defect.** In the run above, the GENERATED frame t kept the window `own_windows` chose for the RECORDED frame t, whose
surprise term was scored against the recorded coarse frame, and the outcome heads read the generated percept at that
window (`interface.futures`: `take("fix", 0)`). The window could therefore carry the recorded outcome's location into the
generated reading. F2's training is unaffected (the cascade's physical tokens never read the current fixation) and so
are: the history encoding, the real-successor heads, S1's reader counts (periphery stage only), the gaze descriptives.
The first run was stopped after F1.6 (its partial output is discarded); nothing from it is reported.
**Fix.** `interface.futures(..., choose=None)`: optional cascade hook (`model.sample`'s existing `choose(p, v)`, between
the physical and glimpse stages). `closed_loop`: the saved head's history-only forecasts and C map (as for the recorded
frame), S scored against the GENERATED periphery, the training policy, sampled with the conditional run's own uniform
draw per row (common random numbers: identical maps give identical windows). The outcome heads read the generated
percept at the chosen window. The real-successor heads keep the conditional window (the real frame's own gaze).
**Checks done before launch.** Smoke: random and conditional worlds reproduce the pre-amendment smoke output in every
field (max difference 0); conditional vs closed-loop real-successor outputs identical (shared encodings and world
draws). The hook fed the RECORDED coarse frame reproduces all 2,384 conditional windows of six dev episodes exactly.
**Readings now.** Interface worlds: p1f16 random, p1f2 random, p1f2_own (conditional, kept as Codex's conditional
reading), p1f2_closed. Primary: p1f2_closed - p1f16; also p1f2_closed - p1f2, and p1f2_closed - p1f2_own (the leak:
paired on identical physical samples, the window alone differs; with each class's window agreement). S1 is run as
before: its S1.3 C2ST and generated outcomes are CONDITIONAL (a closed-loop C2ST needs the window moved into the
outcome features, a different instrument, not run); its reader counts, probes and calibration are valid.
**Declared reading** unchanged, read on p1f2_closed.
