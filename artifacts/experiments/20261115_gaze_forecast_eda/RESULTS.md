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
first-order bound (a forecaster that knew the timing exactly).

**The share's denominator is too harsh, so the verdict is premature (user's challenge, 11-15).** It counts all of the
gaze target's frame-to-frame change, including change any forecaster gets without memory: the contingency term is an
exact function of the last 8 frames and the action (the chooser's head with the taken action against NOOP), and where
scroll reveals new terrain follows the action. The 3.8% is therefore not like-for-like with the horizon EDA's 1.2-1.8%,
whose denominator removed transport. The fair denominator is what a forecaster with the last 8 frames and the action
still misses (the gaze analogue of the transport baseline); not yet measured.

## Fair denominator (`fair.py`, `fair.json`; run 11-15 after the user's go)

A forecaster of the gaze distribution from exactly the chooser's inputs (the chooser head's trunk, a score per cell, a
window's logit the sum of its cells' plus a bias), fitted on 4/5 of DEV episodes for 4 epochs (training cross-entropy
3.67 -> 3.53, still falling), read on the held-out fifth (58,103 frames). CPU, full precision; event prices recomputed on
the same held-out frames (they match the GPU run within a few percent).

- **What a memoryless forecaster still misses: KL 0.314 nats per frame**, against 0.739 for copying the previous
  frame's distribution on the same frames (moved 0.375, still 0.281, day 0.376, night 0.186). It removes 57% of what
  copying misses.
- **Pixel-clock ratio: 9.3%** of that remainder (food 2.7%, drink 2.9%, awake energy 1.7%, asleep energy rise 2.0%;
  recovery 2.5%), against 3.9% with the copy denominator on the same frames.
- The forecaster does not anticipate the events: its KL at event frames is 0.37-0.47, above its 0.31 mean.

Reading: the denominator is now like-for-like, but the numerator is not memory's value (Codex, 11-15). It is how far
the flip moves the eye (the same frame with the digit reverted, rerun through the chooser), not how much forecast error
knowing the clock's phase would remove. So the 9.3% is neither a share of memory-learnable signal nor a lower bound on
one: a better memoryless forecaster that anticipated some flips would raise the ratio while lowering memory's value.
The 9.3%-vs-10% comparison is therefore beside the point; memory's value is measured directly below.

## Oracle phase (`oracle.py`, `oracle.json`; Codex's test, run 11-15 on CPU)

fair.py's forecaster with the simulator's exact hidden counters at t - 1 as an extra input, true against shuffled
within (split, asleep); same chooser, inputs, capacity, split, budget and batch order. Known answer held (the counters
decide each event: 99.97% food and drink, 100% both energy clocks). The shuffled arm reproduces fair.py (held-out KL
0.3128 against 0.3140).

- **Knowing all four counters helps the forecast.** Held-out KL 0.3128 -> 0.2917 nats per frame (gain 0.021, 6.7%;
  still 0.028 against moved 0.008, day 0.028 against night 0.007); the gap was still growing at the last epoch
  (training cross-entropy gap 0.008, 0.015, 0.018, 0.021 over the four epochs). **This is not a measure of long
  memory** (Codex): 6,739 of the 10,153 held-out eligible asleep rows already show sleep onset or an energy rise inside
  the eight-frame history, so part of the gain is phase the small forecaster did not read from its own window. The
  strict cohorts below isolate older history.
- **Per-clock readings from this run are not identified.** Each came from shuffling one counter at test time in the
  arm trained with all four, which makes it confidently wrong (a shuffled "due" predicts a pull that does not come):
  the four single-counter gains sum to 0.045, 2.1 times the retrained true-vs-shuffled gap. The fixed primary reading
  (R = 1.15 against the 0.5 bar) rests on that measure and overstates; rescaling the single-counter gains to the
  joint gap (my R = 0.54 and per-counter values) does not identify each clock's contribution either (Codex), and is
  withdrawn. The same artefact applies to the event-frame shares (gain at events against the flip's first-order KL:
  food 37%, drink 45%, awake energy 43%, asleep rise 57%, recovery 46%), which are upper readings. What does hold
  from this run: gains at eligible frames without an event are positive (0.004-0.023), not the loss a "look at the
  digit more often" head would show.

## Strict cohorts (Codex, CPU, 11-15; `../NOTEBOOK.md`, "Adversarial audit of the gaze-phase oracle" and "Gaze forecast and the awake resource clocks")

Separate matched refits, one clock phase at a time, on held-out frames whose last eight frames show no change of the
relevant digit (so any gain needs older history); phase either exact or reconstructed from rendered history before
the target frame; baselines with the phase shuffled within strata and held at zero; F6's frozen final chooser;
intervals resample held-out episodes (one training seed per arm, so retraining variability is open).

| Cohort | Rows | Phase input | Gaze KL saved per row (nats) |
| :--- | ---: | :--- | ---: |
| Repeat sleep (WAKE_UP already paid, 8 asleep frames, constant energy) | 909 | from visible history | 0.0364 [0.0266, 0.0464] |
| Awake, drink unchanged 8 frames | 3,436 | from visible history | 0.0079 [0.0051, 0.0108] |
| Awake, food unchanged 8 frames | 3,880 | from visible history | 0.0046 [0.0014, 0.0079] |
| Awake, energy unchanged 8 frames | 4,339 | from visible history | 0.0038 [0.0013, 0.0063] |

- Sleep: 0.118 on the 213 rises, 0.011 [0.008, 0.015] on the 696 non-rises; against the zero baseline 0.039. In this
  cohort the last rise was 8-11 steps before the target (the last four steps of the 11-step asleep cycle; event rate
  23%), so the memory needed is just past the
  eight-frame window.
- Non-drops, full KL: drink 0.0046 [0.0028, 0.0064]; food 0.0004 [-0.0015, 0.0022]; energy 0.0018 [-0.0006, 0.0041].
- Awake energy carries its value before the first sleep (0.0110 [0.0059, 0.0166], 1,651 rows), where steps since the
  episode began is an equally good cue; after sleep the full gain is -0.0006 [-0.0030, 0.0017] (2,688 rows), only the
  digit-coverage part survives (0.0017 [0.0007, 0.0027]).
- These gains are in the fitted head's loss. A nat saved here has no known conversion to the gradient that reaches
  Mamba under this loss, or to the reward forecast's (Codex), so comparisons with the reward forecast's counter gains
  (`../20261114_currency_eda/`, section 1) are not made; the dose into Mamba has to be measured.
