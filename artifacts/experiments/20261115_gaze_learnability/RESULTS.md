# Gaze learnability: results (2026-11-15)

Run as planned (PLAN.md, 968dc944), launched by hand at 19:05 after F6.1's readouts with the user's go: A0's natural
probe APs reproduced `reference.json` to 1e-7 (the asserted known answer); peak 3.50 GiB after the first update; 0.70 s
per update; GPU step 19:05-19:41, CPU summary 19:41-19:43 (`result.json`, `trained.json`, `references.json`). One seed.

## Verdict under the fixed readings (`result.json`, "verdict")

**Not "drop the target": one clock passes; per the plan the reading is for the user.** At 3,000 updates the asleep
energy rise passes the screen and the time-and-light bar ("event readability beyond time and light"; no exact-pair
bank exists for it). Food, drink and the awake energy drop are "not taught" at 1,500 and 3,000.

| Clock | Natural AP 750 / 1,500 / 3,000 (A0; screen bar) | Beyond time and light, trained - A0, nats per frame, 3,000 [95%] |
| :--- | :--- | :--- |
| Asleep energy rise | 0.189 / 0.223 / **0.374** (0.170; 0.367) | **+0.035 [0.028, 0.046]** (750 +0.004, 1,500 +0.010) |
| Food drop | 0.036 / 0.037 / 0.039 (0.037; 0.080) | -0.0001 [-0.0003, 0.0002] |
| Drink drop | 0.047 / 0.046 / 0.047 (0.047; 0.085) | 0.0000 [-0.0004, 0.0004] |
| Awake energy drop | 0.031 / 0.031 / 0.032 (0.035; 0.189) | -0.0005 [-0.0008, -0.0002] |
| Recovery (no reward request here) | 0.127 / 0.168 / 0.156 (0.541) | -0.042 [-0.045, -0.037] |

- **The asleep phase is learned where no reward asks for it.** On 8 constant asleep frames (steady8) the trained
  belief's gain over A0 beyond time and light is +0.057 [0.040, 0.078]; on the repeat-sleep cohort (after_reset:
  WAKE_UP already earned) +0.026 [0.015, 0.042]. For scale: F6.1's whole reward-forecast run, from scratch, reached
  +0.012 [0.003, 0.025] on that cohort over F6 (`../20261115_f61_dissection/`, check 1). The gain was still
  accelerating at the last reading (1,500 to 3,000: +0.010 to +0.035).
- **No awake clock in 3,000 updates.** The exact-history banks show no separation, trained or A0 (energy and food:
  probe logit due minus delayed within +-0.004), and the trained gaze head forecasts the same digit coverage on both
  branches (energy 0.211 / 0.211, food 0.096 / 0.096 at 3,000), where the chooser itself looks 0.50 / 0.13 and 0.55 /
  0.10. Awake energy lost a little of what F6 had (-0.0005 nats).
- **The order matches the measured memory value** (gaze-forecast EDA, Codex's strict cohorts): repeat sleep 0.036 nats
  per row, learned; drink 0.008, food 0.005, awake energy 0.004, not learned at this budget. Sleep also has the shortest
  span (an 11-step asleep cycle against 21-31 awake steps) and the densest events.
- **Recovery decays without its reward request** (0.54 to 0.16, faster than under G1 alone, A5: 0.28 at 2,000), and the
  gaze head keeps reading the remaining trace (recovery bank: forecast coverage due 0.206 against delayed 0.097).

## Bounds

A fine-tune from F6's checkpoint with the whole optimiser on this loss, perception frozen, prior fixations, F6's fixed
final chooser as teacher, one seed. It says nothing about the dose inside a joint recipe, and nothing about the world
drawing the clock: F6.1's dissection found that link to be the binding one for the clock it taught (the belief expects
the rise at the due step, 0.61, the denoiser draws it 0.05). A null on the awake clocks bounds this continuation at
3,000 updates (12.2M frames), not longer training or training from scratch.
