# Would a gaze forecast price the pixel clocks? (EDA, 2026-11-15)

User 11-15, after the fovea discussion: "a neat idea ... run that EDA and report the results back". The idea: the belief
forecasts, before frame t, where the learned eye will be drawn at frame t (the chooser's 49-window distribution, which
reads frame t). The question for this EDA is only whether that target gives the pixel clocks a real price, against the
~0.4-1.8% they get in every observation currency measured so far (token MSE, discrete codes, horizon targets, G1).
No model is trained; F6's checkpoint 10 and its own chooser (`../20261113_p1f6/run/chooser_last.pt`) are read on all DEV.

- **Claim.** In the gaze currency (the chooser's softmax over windows of surprise and contingency, each normalised by
  the frame's mean) a digit change in the quiet HUD is a whole event, so the clocks carry a substantial share of what
  the target does not get from copying the previous frame's distribution, and a brightness step barely moves it.
- **Competing explanation.** The chooser's surprise is squared token error too: on frames with scroll, mobs or night
  static the digit change competes with everything else and is diluted as in the other currencies; or the normalisation
  is undone by a global brightness step.
- **Separating construction.** For every clock event at frame t, a counterfactual frame: the same frame with the
  digit's cell copied from frame t - 1 (no event), and that frame with +1 / +3 / +6 brightness (labels choose the
  frames; the counterfactual is an evaluation device only, nothing here trains).

## Instrument (`gaze.py`)

Per DEV episode, the chunk as training builds it (`data.gather`, eight invalid frames in front so every frame is a
target), coarse tokens by the chooser's own `coarse`, inputs by its own `inputs`, scores and policy by its own `Head` and
`policy` with F6's final head weights and recipe. Events from `../20261114_learnability/learnable.py`'s `events`
(food drop, drink drop, awake energy drop, asleep energy rise; recovery as the reward-priced reference), digit cells
63-66 (health, food, drink, energy).

## Readings (fixed before results)

1. **Pull:** the probability that the glimpse covers the digit's cell, at event frames, with the real frame and with its
   no-event counterfactual, per clock.
2. **Event price:** KL(real || no-event) of the 49-window distribution at event frames; per frame of all DEV, the
   clock's rate times its mean event KL (what a forecaster that knew the timing would save, to first order).
3. **Share (primary):** the four pixel clocks' summed price over the mean KL(p_t || p_{t-1}) on all frames with a
   previous scored frame (what copying the previous gaze distribution misses; frame 0 of each episode has no previous
   action and is not scored, as in training), with the same ratio split by moved / still
   frames and day / night. Earlier currencies gave the pixel clocks 0.4-1.8%.
4. **Nuisance:** per clock, the share of event frames where the +1 / +3 / +6 brightness foil's KL (foil vs no-event
   counterfactual) is at least the event's KL. The change-map forecast gate failed here (+1/255 won 2 of 32).

Pass for designing a training arm: share at least 10% and the +3/255 foil winning in at most 10% of each clock's events.
Either failure is reported as a failure of this target, not of the idea of a gaze teacher.

## Fair denominator (proposed 11-15 after the user's challenge; not run)

The share bar used KL to the previous frame's gaze distribution, which counts change a forecaster gets without memory
(RESULTS.md, corrected). Proposed correction of the instrument, same bar: fit a small forecaster of the gaze
distribution from exactly the chooser's inputs (the last 8 frames' coarse tokens and outcomes, and the action into the
frame), on 4/5 of DEV episodes, soft cross-entropy; read on the held-out fifth its KL to the real distribution, overall
and by moved / still and day / night. Share = the four pixel clocks' summed price (as measured) over that held-out KL.
Pass at 10%, as before. Its own limit: the forecaster's capacity bounds how much of the predictable change it removes,
so the share it gives is a lower bound on the share a better memoryless forecaster would leave.

## Oracle phase (proposed by Codex 11-15; run on CPU with the user's go for CPU tests)

The fair share's numerator is the flip's effect on the eye (the same frame with the digit reverted, rerun through the
chooser), not what remembering the clock would save a forecaster; a better memoryless forecaster could even raise the
ratio while lowering memory's value (Codex). This measures memory's value directly.

- **Claim.** Knowing the clocks' phase, which a belief with memory could carry, lowers the gaze-forecast loss beyond the
  8-frame forecaster by a large part of the first-order price.
- **Competing explanations.** (a) How far a flip pulls the eye depends on the rest of the frame, so knowing *when*
  buys little of the price; (b) the last 8 frames already carry the predictable part (a flip seen recently means none
  is due); (c) the gain is "look at the digit more often", not timing.
- **Separating control (`oracle.py`).** fair.py's forecaster, same chooser, inputs, capacity, episode split, budget and
  batch order, with one extra input: the simulator's four hidden counters (hunger, thirst, fatigue, recover) at frame
  t - 1, one-hot over their exact values, from the exact replay (`../20261107_f4_followups/readings/counters_dev.npz`,
  checked equal to the labels at every state). Known answer, checked before this plan: the counters at t - 1 decide
  each clock event at t (agreement 99.97% food and drink, 100% both energy clocks). Arm *true* against arm *shuffled*:
  the counter vectors permuted within (split, asleep at t - 1). Per counter: the true arm with only that counter
  shuffled the same way at test time. The counters are an evaluation device only.

### Readings (fixed before results)

1. **Primary: recovered fraction** R = the pixel counters' memory value (held-out KL increase when hunger, thirst or
   fatigue alone is shuffled, summed, per held-out frame) over the four pixel clocks' first-order price per frame
   (`fair.json`, the same held-out frames). R >= 0.5: the price is mostly memory value, and the next question is the
   dose into Mamba (a GPU measurement, after F6.1). R < 0.25: drop this target. In between: reported, for the user.
2. Per clock: the gain at event frames, at eligible frames without an event (a head that only looks at the digit more
   often loses there), at ineligible frames; R per counter (fatigue against both energy clocks' prices).
3. Total: shuffled-arm against true-arm held-out KL (all, moved / still, day / night); the shuffled arm against
   fair.py's 0.314 (expected within a few percent).
4. Memory share: the pixel counters' value over the shuffled arm's KL (the like-for-like replacement for the 9.3%,
   for the user's judgement; no bar). Recover, the reward-priced reference, in the same currency.

Bounded: F6's frozen final chooser; the oracle forecaster's capacity and 4-epoch budget (it bounds what a forecaster
of this size extracts from exact phase, not what Mamba would learn); no gradient into any model.
