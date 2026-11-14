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
   previous frame (what copying the previous gaze distribution misses), with the same ratio split by moved / still
   frames and day / night. Earlier currencies gave the pixel clocks 0.4-1.8%.
4. **Nuisance:** per clock, the share of event frames where the +1 / +3 / +6 brightness foil's KL (foil vs no-event
   counterfactual) is at least the event's KL. The change-map forecast gate failed here (+1/255 won 2 of 32).

Pass for designing a training arm: share at least 10% and the +3/255 foil winning in at most 10% of each clock's events.
Either failure is reported as a failure of this target, not of the idea of a gaze teacher.
