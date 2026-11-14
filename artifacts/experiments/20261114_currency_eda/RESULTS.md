# What could price a timer? Results (2026-11-14 evening)

Plan and instruments committed before results (PLAN.md). Facts about the DEV data and F6's frozen tokens; no
training, nothing selected.

## 1. Future reward explained by each hidden counter's phase (`consequence.json`)

Boosted trees predicting the reward class n steps ahead from the observable state plus the recovery counter, with
and without each hidden counter; test log-loss gain in nats per frame, episode-bootstrap 95% interval (58,304 test
states, 1/5 of DEV episodes).

| Distance | Recovery counter (positive control) | Hunger | Thirst | Fatigue | Fatigue, asleep frames |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.068 [0.060, 0.076] | 0.0002 | 0.0004 | 0.0003 | 0.0000 |
| 2 | 0.069 | 0.0006 | 0.0006 | 0.0064 [0.0053, 0.0077] | 0.036 [0.029, 0.045] |
| 4 | 0.063 | 0.0017 [0.0007, 0.0027] | 0.0013 | 0.0087 | 0.047 [0.039, 0.058] |
| 8 | 0.056 | -0.0003 | -0.0004 | 0.0074 | 0.045 |
| 16 | 0.044 | -0.0012 | -0.0010 | 0.0022 | 0.009 |
| 32 | 0.023 | 0.0004 | 0.0008 | 0.0021 | 0.011 |

In the low-need strata (food, drink or energy <= 1; 1,217-8,559 test states) no hunger or thirst gain has an
interval excluding 0 at any distance. Asleep frames are 18% of states. The behaviour policy is the corpus's.

## 2. A learned discrete code over F6's periphery tokens (`codes.json`)

k-means code books over all cells' tokens (fit episodes), read on the test episodes.

| K | digit changes that change the code: health / food / drink / energy | map code change: still day / still night / moved | cells changed by +1 / +3 / +6 per 255 | pairs where +3 beats the consequence (energy, hunger) |
| ---: | :--- | :--- | :--- | :--- |
| 64 | 0.92 / 0.54 / 0.89 / 0.44 | 0.055 / 0.38 / 0.44 | 0.03 / 0.08 / 0.16 | 6 / 32, 6 / 24 |
| 256 | 1.00 / 0.60 / 0.80 / 0.32 | 0.11 / 0.53 / 0.60 | 0.10 / 0.28 / 0.45 | 20 / 32, 20 / 24 |
| 1024 | 1.00 / 1.00 / 0.89 / 0.93 | 0.18 / 0.69 / 0.74 | 0.21 / 0.50 / 0.68 | 31 / 32, 23 / 24 |

Consequence on the pairs: 2.6-4.4 nats for a forecaster that knows each cell's change rate but not the timer.

## 3. Share of a code-likelihood loss (`share_codes.json`)

Forecaster: each cell's change rate by frame class (moved, still day / dusk-dawn / night, asleep) and its new-code
distribution; test episodes.

| K | loss per frame | moved | still night | asleep | map cells | food + drink + energy changes | nats per food / drink / energy change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 64 | 97.8 | 0.55 | 0.19 | 0.13 | 0.94 | 0.0041 | 2.7 / 4.3 / 2.3 |
| 256 | 171.3 | 0.51 | 0.18 | 0.12 | 0.93 | 0.0022 | 3.3 / 4.0 / 1.8 |

Without the moved frames (a forecaster that transports perfectly), the K = 64 share would be 0.0041 / 0.45 = 0.9%:
the same as the squared-error terms' 0.9-1.2%.

## Readings

- **A reward forecast further ahead asks for sleep's clock and not for food's or drink's.** Fatigue's phase is worth
  0.036-0.047 nats per asleep frame at 2-8 steps, about half of what the recovery counter is worth at one step (the
  clock the one-step reward forecast taught). Hunger and thirst are worth nothing measurable up to 32 steps, also
  where a need is low: damage from unmet needs is timed by the recovery counter, which the belief already carries.
- **A discrete code changes the price of an event, not its share.** A coarse code prices a digit change at 2-4 nats
  (squared error: 0.004-0.012 per frame) and mostly ignores small brightness edits (K = 64), but every unpredictable
  map change is priced in nats too, so the events' share stays near 1% once transport is handled, and falls with a
  finer code. Under any proper score the share is set by how much of the observation's unpredictability the event
  is, not by the unit.
