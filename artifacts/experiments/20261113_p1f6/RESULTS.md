# P1-F6 results (2026-11-14): change-form token forecast in place of TSP's forecast mode

Run: `saccade-p1f6`, F5's recipe with one change (PLAN.md), 25.6M frames, checkpoints 1-2 as the early test and
3-10 resumed at update 5,000 (exact resume). The monitor loss never rose three times in a row, so checkpoint 10 was
selected (`run/ckpt_10.pt`, SHA256 `a848e5f5...`). Every stage of the queue completed (`readings/STATUS.json`). One
world seed. The dissection is `../20261114_f6_dissection/` (plan and code committed before its results).

## 1. Summary

1. **The primary reading is negative: no pixel-only clock formed, awake or asleep, at any checkpoint.** Food, drink
   and energy drops read from the belief stay at their base rates, the asleep energy rise stays at F5's level, and
   hunger pairs are drawn untimed. The move to 256-step chunks at checkpoint 5 changes nothing.
2. **The clocks the reward prices are kept** (recovery, cooldown, need damage, danger), on F5's schedule.
3. **Replacing the TSP forecast recovered F5's glimpse-prediction cost** (2.28 nats per pixel against 2.57; F4 2.22)
   and gave small gains elsewhere; coal and iron stay collapsed, iron lower than F5.
4. **Why the forecast taught nothing (dissection, section 5):** as trained, its request for a clock was four to five
   orders of magnitude below the reward forecast's, and the term is 1-2% of the belief's gradient, itself 70% scene
   scroll that this head cannot predict. The belief is trained almost entirely by the two reward heads, in F5 and F6
   alike. Whether a larger, cleaner dose would form the clocks is not established.

## 2. Clocks (`readings/belief_probe.json`, `native_pairs.json`, `trajectory/`, `gate3/`)

| Reading, checkpoint 10 | base rate / truth | F5 | F6 |
| :--- | ---: | ---: | ---: |
| Food drop, awake, AP from the predicted belief | 0.037 | 0.042 | 0.038 |
| Drink drop, awake | 0.044 | 0.048 | 0.046 |
| Energy drop, awake | 0.027 | 0.035 | 0.034 |
| Energy rise, asleep | 0.086 | 0.184 | 0.180 |
| Recovery (+1 health), awake | 0.041 | 0.585 | 0.590 |
| Need damage, awake | 0.035 | 0.250 | 0.289 |
| Zombie ready / hit awake | 0.502 / 0.161 | 0.992 / 0.960 | 0.989 / 0.956 |
| Hunger pairs drawn, due / not due | 1 / 0 | 0.069 / 0.068 | 0.160 / 0.159, contrast [-0.004, 0.007] |
| Cooldown pairs drawn, ready / cooling | 1 / 0 | 0.862 / 0.025 | 0.912 / 0.048 |
| Recovery pairs drawn, due / not due | 1 / 0 | 0.508 / 0.000 | 0.466 / 0.000 |
| Energy pairs drawn, due / not due | 1 / 0 | 0.009 / 0.009 | 0.014 / 0.013 |

Checkpoint trajectory (F6 / F5): recovery from the belief 0.05 / 0.05, 0.24 / 0.20, 0.34 / 0.28, 0.46 / 0.43,
0.58 / 0.57, 0.59 / 0.59 at checkpoints 1, 2, 3, 5, 8, 10; the asleep energy rise 0.218, 0.214, 0.214, 0.195, 0.183,
0.180 (F5 0.221 ... 0.184); food 0.038-0.039, drink 0.045-0.048 and awake energy 0.034-0.035 at every checkpoint.

The arm's own forecast head on the exact-history pairs (`gate3`, arm `native`; share of the true due / not-due token
difference at the digit's cell), beside a fresh change-form head fitted on the same frozen belief:

| | own head, ckpt 2 | fresh head, ckpt 2 | own head, ckpt 10 | fresh head, ckpt 10 |
| :--- | ---: | ---: | ---: | ---: |
| Recovery | 0.0002 | 0.128 | 0.027 | 0.403 |
| Cooldown | -0.0008 | 0.112 | 0.067 | 0.640 |
| Energy | 0.000 | 0.000 | 0.000 | 0.000 |
| Hunger | 0.000 | 0.000 | 0.000 | 0.000 |

The trained head lags what its own belief holds by an order of magnitude or more; for energy and hunger neither head
finds anything. Held-out token error at checkpoint 10: own head 0.0975, fresh 0.0937, copying the previous frame 0.1052.

## 3. Sleep (`readings/followups/sleep/`)

Restricted wait ratio against the simulator: energy 9 at 2.56 [2.10, 3.22], awake at 2.44 [2.04, 3.01] (F5 3.24 and
3.06). The rise is still an untimed hazard: 0.069 per asleep step, flat over 48 steps (F5 0.048, F4 0.025; about 0.09
would match the mean), rank correlation with the true due step 0.27 (F5 0.32). Waking at energy 9 in the interface
0.962 (F5 0.906); in the rollout, generated energy 9 is followed by waking in 0.50 of 460 eligible transitions (F5
0.51; biased low as in F5).

## 4. Standard readout against F5 (`readings/comparison.json`, `run/s1/results.json`)

- TSP 2.280 nats per pixel, day 2.034 (F5 2.571 / 2.412; F4 2.218 / 1.950); reactive TSP error on matched windows
  0.00107 (F5 0.00189). Generated-vs-real classifier 0.972 (F5 0.988).
- Danger: woken beside a ready / cooling zombie 0.944 / 0.020 (F5 0.959 / 0.018); risky class P(negative) /
  P(termination) 0.116 / 0.110 (recorded 0.127 / 0.128; F5 0.118 / 0.102).
- Achievements: PLACE_FURNACE 0.906 (F5 0.805), DEFEAT_ZOMBIE 0.549 (0.480), EAT_COW 0.362 (0.292), PLACE_PLANT
  0.932 (0.872); COLLECT_COAL 0.740 (0.815), COLLECT_IRON 0.397 (0.513).
- Increments: coal 0->1 0.476 (F5 0.569; F4 0.986), coal 1->2 .. 4->5 0.02-0.14; iron 0->1 0.187 (F5 0.321); wood
  0.84-0.93 (F5 0.69-0.90); stone 1->2 .. 4->5 0.67-0.82 (F5 0.61-0.74). Mining pairs: coal DO/DO 0.090 (F5 0.055),
  wood 0.938 (0.914).
- Recursion, free-running entities at steps 1 / 4 / 8 / 16: 0.941 / 0.861 / 0.785 / 0.703 (F5 0.951 / 0.852 / 0.729 /
  0.650; truth 1.109 / 1.219 / 1.328 / 1.273).
- One-step entities by day: persisting 0.804 (F5 0.798, F4 0.840); arrows 0.273 (F5 0.167, F4 0.282); cows 0.828
  (F5 0.823, F4 0.874). Movement: 64 / 64 pairs correct.
- Monitor G1 0.00409 (F5 0.00392); periphery effective rank 5.20 (5.22). Digit exact at the belief's input: food
  0.354, drink 0.474, energy 0.508 (F5 0.350, 0.486, 0.474).

## 5. Dissection (`../20261114_f6_dissection/readings/`)

**Dose on the exact-history pairs (`pairs_*.json`).** Separation gradient into the belief, the token forecast's own
head against the reward forecast, both in the per-frame units they share inside G4:

| Checkpoint | Recovery: token / reward | ratio (median) | Cooldown: token / reward | ratio (median) |
| :--- | :--- | ---: | :--- | ---: |
| 1 | 1.0e-4 / 4.7 | 2.4e-5 | 4.0e-4 / 5.8 | 5.0e-5 |
| 2 | 7.6e-4 / 119 | 4.9e-6 | 2.7e-3 / 31 | 7.3e-5 |
| 5 | 6.2e-3 / 230 | 2.2e-5 | 6.0e-2 / 158 | 9.2e-5 |
| 10 | 3.8e-2 / 641 | 5.4e-5 | 0.14 / 266 | 2.2e-4 |

Energy and hunger: 1e-5 to 5e-5 at every checkpoint, with no sign. The term prices an energy or food drop at 0.004-0.012
per frame; the reward forecast prices a missed recovery at 1.3-4 nats.

**Gradient budget into the belief on training chunks (`batch_*.json`, `p1f5_batch_*.json`; medians over 8 streams,
64 new frames, prior fixations).** Raw gradient norm of each term:

| | reward forecast | reward | forecast term | TSP | G1 | cosine of the reward forecast to the weighted total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| F6 ckpt 1 | 1.50 | 1.24 | 0.047 (token) | 0.012 | 0.006 | 0.96 |
| F6 ckpt 2 | 3.05 | 1.72 | 0.048 | 0.016 | 0.008 | 0.93 |
| F6 ckpt 5 | 2.30 | 0.63 | 0.032 | 0.002 | 0.007 | 0.96 |
| F6 ckpt 10 | 2.32 | 0.15 | 0.032 | 0.001 | 0.012 | 1.00 |
| F5 ckpt 1 | 1.59 | 1.65 | 0.16 (TSP forecast) | 0.039 | 0.008 | 0.95 |
| F5 ckpt 2 | 2.20 | 1.14 | 0.25 | 0.027 | 0.008 | 0.89 |
| F5 ckpt 10 | 2.63 | 0.18 | 0.35 | 0.008 | 0.026 | 0.99 |

The two reward heads supply almost all of the belief's gradient in both arms. The token forecast is 1-2% of it and
3-10x smaller than the TSP forecast it replaced; G1 is 0.2-0.6% (F5 0.2-1%).

**What the token forecast's loss pays for (`lossmap_*.json`, all DEV, 304,898 frames).** Frames: player moved 35%,
still and awake 46%, asleep 18%. Share of the term and the head's lift over copying, checkpoint 10:

| Frames / cells | share of the term | lift over copying |
| :--- | ---: | ---: |
| moved / map | 0.698 | 0.028 |
| still awake / map | 0.217 | 0.115 |
| asleep / map | 0.057 | 0.256 |
| HUD digit cells, all frames | 0.015 | 0.02-0.14 |
| other HUD cells, all frames | 0.012 | up to 0.74 asleep |

Share of each digit change the head predicts at checkpoint 10 (checkpoint 2): health down awake 0.35 (0.03), down
asleep 0.53 (0.08), health up awake 0.07 (0.005); food, drink and energy changes 0.000-0.005 (0.00). All food, drink
and energy changes together are 0.9% of the term.

## 6. What this does not establish

- Whether a reward-sized, scroll-free request in the same currency would form a pixel clock. The arm delivered a
  dose four to five orders below the only request that has ever formed one, so its null does not answer that.
- Why the jointly trained head lags its belief (its learning rate against the fresh fit's, non-stationary beliefs
  and capacity are not separated). F5's TSP forecast head showed the same lag.
- Why the generated hazards moved (hunger pairs 0.16 against 0.07; asleep rise 0.069 against 0.048): one seed.
- The cause of the coal and iron collapse, unchanged by this arm.

The frozen gate (`../20261113_token_forecast_gate/`, round 3) reported request cosines and a unit-step update;
both are scale-free. Its files already held the sizes (separation gradient 1.7e-4 for recovery under the fresh
change-form head at F5's checkpoint 1, against the reward forecast's 4.9); they were not read against each other
before the run.
