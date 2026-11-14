# Gaze-forecast EDA: results (2026-11-15)

F6 checkpoint 10 and its own final chooser on all DEV (303,889 frames with a previous scored frame), readings as fixed
in `PLAN.md` (`gaze.py`, `result.json`). No model was trained.

## Verdict under the fixed bars: fails the share bar, passes the nuisance bar

- **Share (primary): 3.8%** of what copying the previous gaze distribution misses (bar: at least 10%). Food 1.1%,
  drink 1.2%, awake energy 0.7%, asleep energy rise 0.8%. Recovery, the reward-priced reference: 1.1%.
- **Nuisance: passes for every clock.** The +3/255 brightness foil outweighs the event in 0.1-3.3% of events (bar: at
  most 10%); +1/255 in at most 0.5%; +6/255 in 11-29%. The frame-mean normalisation of the chooser's scores does what it
  was meant to: this is the first target in the project whose clock events beat a +3/255 step this clearly (the
  change-map forecast lost at +1/255 in 2 of 32 scenes; per-channel precision and binned tokens lost to +3/255).

## The numbers

| Event | Events | Glimpse covers the digit: real / no-event | Event KL, mean (median) | Price per DEV frame | +3/255 foil wins |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Food drop | 8,275 | 0.241 / 0.088 | 0.304 (0.019) | 0.0083 | 3.3% |
| Drink drop | 9,543 | 0.259 / 0.097 | 0.285 (0.055) | 0.0090 | 0.8% |
| Energy drop, awake | 6,575 | 0.241 / 0.093 | 0.245 (0.035) | 0.0053 | 0.1% |
| Energy rise, asleep | 4,463 | 0.346 / 0.095 | 0.388 (0.182) | 0.0057 | 2.4% |
| Recovery (reference) | 3,569 | 0.295 / 0.042 | 0.683 (0.110) | 0.0080 | 0.03% |

- **The eye is pulled to the digit 2.6-3.6 times as often when it changes**, consistent with F5's training-stream count
  (21-34% of events caught).
- **The price is concentrated on quiet frames.** Event KL on still frames 0.30-0.41 against 0.17-0.22 when the player
  moved; by day 0.33-0.50 against 0.10-0.15 at night. The median event is cheap (0.02-0.18 nats): most changes happen
  while something else holds the eye.
- **The denominator is large because the gaze target is volatile:** the distribution changes by 0.74 nats per frame on
  average (moved 0.90, still 0.65, day 0.90, night 0.42), mostly from scroll and daytime map events. This is the same
  competitor that dominated every earlier currency (the horizon EDA: daytime map content 81-83% of the residual).

## Reading

The gaze target prices a clock change as an event rather than as a few small numbers, and it is robust to brightness,
but on a busy screen the clocks are still a small part of where the eye is drawn: 3.8% against 0.4-1.8% in the earlier
currencies, short of the 10% fixed for designing a training arm. Recorded as a failure of this target (the reactive
chooser's distribution, forecast one step ahead), not of the idea of a gaze teacher.

Context, read after the bar (not a criterion): per DEV frame, the four pixel clocks together carry 0.028 nats in this
currency; the reward forecast priced the recovery counter at 0.068 nats per eligible frame (about 0.025 per DEV frame)
when it taught it. Each pixel clock here gets roughly a quarter to a third of that.

Limits: F6's final chooser (trained 25.6M frames) stands for a chooser that would co-evolve from scratch; the price is a
first-order bound (a forecaster that knew the timing exactly); the share's denominator counts all of the gaze target's
frame-to-frame change, predictable or not.
