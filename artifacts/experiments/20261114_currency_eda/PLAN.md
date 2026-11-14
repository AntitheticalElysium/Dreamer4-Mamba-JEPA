# What could price a timer? Two data readings (2026-11-14 evening)

User, 11-14: the world model should learn everything predictable, but not everything matters equally; would
predicting consequences further into the future help the timers; are there other options for the loss's unit. No
training, no architecture selected: both readings are facts about the data and about F6's frozen tokens.

## 1. How much of the future reward does each timer's phase explain? (`consequence.py`, CPU, DEV labels)

Claim under test: a reward forecast at longer distances (Dreamer 4's multi-token prediction, distances 1-8, or
further) would ask the belief for the pixel-only timers, because their phase changes when damage, recovery or
achievements arrive. Competing explanation: within any usable distance the reward depends on the timers' phase only in
rare states (a need at zero), so such a forecast asks for them almost never.

Reading: the reward class at distance n (n = 1, 2, 4, 8, 16, 32 steps ahead; classes: <= -0.15, -0.1, 0, +0.1,
>= +0.2, episode over) predicted by gradient-boosted trees from the state's observable content (HUD digits,
inventory, sleep, light, achievements, facing, entity counts and adjacency, zombie cooldown) plus the recovery
counter (which F5 and F6 carry), with and without each hidden counter (hunger, thirst, fatigue; the exact counters of
`../20261107_f4_followups/readings/counters_dev.npz`). Episodes split 60 / 20 / 20 (fit / early stopping / test).
Gain = test log-loss without the counter minus with it, in nats per frame, episode-bootstrap interval; also on the
strata where the need is low (food, drink or energy <= 1) and, for fatigue, asleep. Positive control: the recovery
counter's own gain at distance 1 (the clock the one-step reward forecast did teach). A gain near zero at every
distance means no reward forecast, however long, asks for that timer in this corpus; a gain only in the low-need
strata means it asks only there. The behaviour policy is the corpus's (the forecast marginalises over future actions).

## 2. Would a learned discrete code price a digit change without failing the brightness check? (`codes.py`, GPU, frozen F6 checkpoint 10)

Squared error prices an event by its size on screen. Likelihoods on discrete outcomes price it by surprise (the
reward head; DreamerV3's categorical latents; IRIS-style token models). Our two fixed attempts on continuous tokens
(fixed precision, per-number bins) lost to a +3/255 brightness edit. A learned code book over whole tokens is a
different object: a code changes only when the token crosses a boundary between codes. Whether it does better is a
fact about F6's tokens:

- One k-means code book per size K = 64, 256, 1024 over all cells' 16-number tokens (cell-agnostic; fitted on the
  fit episodes' tokens, read on the test episodes).
- **Does the code see the events?** Share of digit changes (health, food, drink, energy) that change their cell's code.
- **What else changes codes?** Code-change rate between consecutive frames by frame class (moved; still awake by
  day, dusk / dawn and night; asleep) for map cells, HUD digit cells without a digit change, other HUD cells.
- **Brightness.** Share of cells whose code changes under a +1, +3, +6 / 255 edit of the frame.
- **Prices on the exact-history pairs (energy 32, hunger 24), as in the earlier gates:** a forecaster that knows each
  cell's own change rate (by cell and by night) but not the timer pays -log(rate) for a change it did not expect.
  Consequence = that cost for the due member's digit cell; foil = the summed cost over the cells whose code the
  brightness edit changes. Reading: foil against consequence per pair, per K. This prices the target only; it does
  not show that a model trained on such codes would learn the timer.
