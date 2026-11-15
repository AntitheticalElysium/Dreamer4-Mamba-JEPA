# P1-F6.1 results (2026-11-15): Dreamer 4's multi-distance reward forecast, no visual forecast

Run: `saccade-p1f61`, F5's recipe with the token forecaster removed and reward forecasts at distances 2-8 added
(PLAN.md), 25.6M frames, one world seed. The monitor loss never rose three times in a row, so checkpoint 10 was
selected (`run/ckpt_10.pt`, SHA256 `c0a3c129...`). The evaluation queue was interrupted by the machine's shutdown at
16:52 during `trajectory` (checkpoint 6) and resumed by hand at 18:11 with the queue's own resumable runner, the
original environment (two CPU threads) and the pinned sources validated (89 files); finished stages were skipped and
`trajectory` resumed at checkpoint 6. Every stage completed except `mining`, whose 0.04 token pre-check failed on one
pair by 0.0008 (numerical; the stage reran in the dissection with the check recorded). Dissection:
`../20261115_f61_dissection/` (checks 1-6, each committed before it was read).

## 1. Summary

1. **The primary reading is mixed under its fixed rule.** On first sleeps (WAKE_UP not yet paid) the world now draws
   the phase: phase advantage 1.31 steps against F6's 0.46, real distance 5.41 against 9.31 (p = 0.0001), early wakes
   0.035 against 0.155, and drawing energy 9 is followed by waking in 224 of 224 transitions (F6 0.50). But late or
   missing wakes rose (0.91 against 0.775), the third condition of "supports the claim".
2. **Repeat sleeps are timed too** (dissection check 2, 48 scenes with WAKE_UP already earned): advantage 0.82 steps
   against F6's 0.34, real distance 7.3 against 15.1. The plan's explanation (b), "repeat sleeps stay untimed", is
   rejected in its strong form, though the forecast heads themselves use the phase only before WAKE_UP has paid.
3. **The lateness is a use failure in the denoiser** (checks 3, 4, 6). The world orders the scenes correctly (rank
   correlation of drawn and true rise steps 0.82, F6 0.28) and almost never rises early, but releases the rise about
   5% of the time at the due step and 11-14% per step after it. Along its own rollouts the belief expects the rise at
   the due step (probability 0.61, 0.05 before it). Doubling the sampler's evaluations lifts the release only by a
   quarter.
4. **The predicted negatives hold:** food, drink and awake-energy drops stay at their base rates from the belief, add
   nothing beyond elapsed time and lighting (check 1), and the hunger and energy pairs are drawn untimed.
5. **The reward-priced clocks improve:** recovery read from the belief 0.71 (F6 0.59), drawn due / not due 0.643 /
   0.000 (F6 0.466 / 0.000).
6. **Costs, beside a growing reward share of the encoder:** coal and iron lower again (coal 0->1 0.35, iron 0.14),
   the generated-vs-real classifier 0.991 (F6 0.972, F5 0.988), the periphery tokens' effective rank 4.88 (F6 5.20,
   F5 5.22, F4 5.96), danger slightly weaker. TSP, G1 and reward calibration slightly better.

## 2. Primary: waking on the quiet sleep bank (`readings/followups/sleep_split.json`)

| Checkpoint 10 | Scenes | Phase advantage (p) | Real / shuffled W1 | Awake by 12 | Wait ratio | Early | Late or none |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| F6.1, not yet paid | 25 | 1.31 (0.0001) | 5.41 / 6.72 | 0.545 | 1.79 [1.60, 2.07] | 0.035 | 0.910 |
| F6, not yet paid (PLAN) | 25 | 0.46 (0.006) | 9.31 / 9.77 | 0.46 | 2.25 | 0.155 | 0.775 |
| F6.1, already paid | 7 | 0.24 (0.30) | 12.14 / 12.38 | 0.25 | 2.34 [1.87, 2.95] | 0.125 | 0.875 |
| F6, already paid (PLAN) | 7 | none (0.95) | | 0.286 | 2.97 | 0.214 | 0.768 |

Against the fixed readings: a phase advantage clearly above F6's with a lower real distance (yes), late wakes falling
without early wakes rising (no: early wakes fell, late wakes rose). Explanation (a), heads using the phase with the bank
unchanged, is rejected: the bank changed. The rest of the sleep stage: restricted wait to energy 9 2.13 (F6 2.56, F5
3.24), to waking 1.94 (F6 2.44, F5 3.06); waking at energy 9 in the interface 0.994 (F6 0.962, F5 0.906).

## 3. The new heads (`readings/ahead_*.json`, Codex's donor readout)

Phase effect (different-phase donor's reward log-loss minus the same-phase donor's, nats per asleep origin):

| Distance | ckpt 1 not paid | ckpt 2 not paid | ckpt 3 not paid | ckpt 10 not paid [95%] | ckpt 10 already paid [95%] |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | -0.002 | 0.002 | 0.004 | 0.008 [-0.036, 0.048] | 0.012 [-0.016, 0.040] |
| 2 | 0.104 | 0.259 | 0.365 | 0.661 [0.618, 0.701] | -0.008 [-0.030, 0.012] |
| 4 | 0.065 | 0.123 | 0.190 | 0.405 [0.375, 0.435] | 0.003 [-0.015, 0.020] |
| 8 | 0.008 | 0.038 | 0.040 | 0.199 [0.184, 0.217] | -0.005 [-0.018, 0.008] |

F6's distance-1 head at checkpoints 1-3: 0.0005 / 0.006 / 0.009 not paid. The heads at distances 2-8 use the phase
strongly, growing with training, and only before WAKE_UP has paid (after it no reward follows waking). Every-frame
own log-loss at checkpoint 10: 0.079 at distance 1, 0.19-0.27 at 2-8.

## 4. Clocks (`readings/trajectory/`, `belief_probe.json`, `native_pairs.json`, `trace_10.json`, `d1/`)

Read from the predicted belief (AP; base rates food 0.037, drink 0.044, awake energy 0.027, asleep rise 0.086,
recovery 0.041), F6.1 / F6:

| Checkpoint | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Asleep energy rise | .205 / .218 | .277 / .214 | .363 / .214 | .405 / .197 | .445 / .195 | .454 / .187 | .463 / .184 | .464 / .183 | .470 / .181 | .471 / .180 |
| Recovery | .056 / .050 | .219 / .235 | .353 / .336 | .468 / .402 | .571 / .461 | .648 / .533 | .676 / .565 | .692 / .584 | .708 / .588 | .710 / .590 |

Food 0.039-0.040, drink 0.047-0.049 and awake energy 0.034-0.037 at every checkpoint (F6 the same). Need damage 0.256
(F6 0.289, F5 0.250); zombie ready / hit 0.991 / 0.960 (F6 0.989 / 0.956).

Drawn on the exact-history pairs at checkpoint 10 (due / not due): recovery 0.643 / 0.000 (F6 0.466 / 0.000; trace at
the due step 0.48 against 0.08), cooldown 0.812 / 0.004 (F6 0.912 / 0.048; contrast 0.81 [0.72, 0.89] against 0.86),
energy 0.007 / 0.007, hunger 0.089 / 0.089 (both untimed, as in F6). Registration of the pixel resets is unchanged
(`d1`): food drops reach Mamba's input at AP 0.064 and its state at 0.036 (F6 0.063 / 0.033), awake energy drops 0.285
/ 0.039 (F6 0.245 / 0.042).

## 5. The share (`readings/budget_*.json`, weighted gradient norms, medians over 8 streams)

| | Mamba: reward fc. d1 / d2-8 / reactive reward / G1 / TSP | Encoder: reward fc. d1 / d2-8 / reactive / TSP / reconstruction |
| :--- | :--- | :--- |
| F6.1 ckpt 3 | 2.64 / 1.02 / 2.28 / 0.011 / 0.012 | 8.31 / 3.28 / 6.18 / 1.12 / 0.37 |
| F6 ckpt 3 | 3.05 (one term) / - / 0.98 / 0.012 / 0.011 | 6.07 / - / 2.80 / 0.92 / 0.14 |
| F6.1 ckpt 10 | 5.40 / 2.17 / 0.90 / 0.043 / 0.004 | 8.99 / 6.79 / 2.00 / 0.40 / 0.03 |

Mamba stays the reward forecast's (cosine to the total 0.92 at distance 1). The reward heads' share of the encoder
against TSP, the periphery tokens' designated owner, rose from about 10x (F6, checkpoint 3) to 16x (F6.1, checkpoint 3)
and 45x (F6.1, checkpoint 10).

## 6. Standard readout and guards (`run/s1/results.json`, `readings/comparison.json`, `dissection/entities.json`)

- TSP 2.242 nats per pixel, day 1.977 (F6 2.280 / 2.034, F5 2.571 / 2.412, F4 2.218 / 1.950). Generated-vs-real
  classifier 0.991 (F6 0.972, F5 0.988). Reward calibration NLL 0.0121 (F6 0.0128, F5 0.0139).
- Monitor G1 0.00383 (F6 0.00409, F5 0.00392, F4 0.0032); logged G1 / TSP / reconstruction over the last 1M frames
  0.00393 / 0.00123 / 0.00087 (F6 0.00414 / 0.00125 / 0.00092). Periphery effective rank 4.88 (F6 5.20, F5 5.22, F4
  5.96).
- Danger: woken beside a ready / cooling zombie 0.901 / 0.028 (F6 0.944 / 0.020, F5 0.959 / 0.018); risky class
  P(negative) / P(termination) 0.117 / 0.105 (recorded 0.127 / 0.128; F6 0.116 / 0.110).
- Achievements: PLACE_FURNACE 0.928 (F6 0.906), DEFEAT_ZOMBIE 0.608 (0.549), EAT_COW 0.362 (0.362), WAKE_UP 0.860
  (0.851), COLLECT_DRINK 0.872 (0.920), COLLECT_COAL 0.628 (0.740), COLLECT_IRON 0.294 (0.397), COLLECT_DIAMOND 0.025
  (0.095).
- Increments: coal 0->1 0.351 (F6 0.476, F5 0.569, F4 0.986), iron 0->1 0.144 (0.187, 0.321, 0.923); wood 0->1 0.947
  (0.900), stone 2->3 0.835 (0.820). Mining pairs (dissection check 5b, the stage's own code): coal DO/DO 0.067 (F6
  0.090, F5 0.055), wood 0.954 (0.938).
- Recursion, free-running entities at steps 1 / 4 / 8 / 16: 0.973 / 0.887 / 0.783 / 0.643 (F6 0.941 / 0.861 / 0.785 /
  0.703; truth 1.109 / 1.219 / 1.328 / 1.273). One-step entities by day: persisting 0.806 (F6 0.804, F4 0.840), arrows
  0.251 (0.273), cows 0.830 (0.828). Movement 64 / 64 pairs.

The cost signals line up across runs with the reward heads' growing hold on the encoder (F4 had no forecast heads):
effective rank 5.96 / 5.22 / 5.20 / 4.88 and coal 0->1 0.99 / 0.57 / 0.48 / 0.35 for F4 / F5 / F6 / F6.1. Four runs
that differ in more than this, one seed each: a consistent trend, not an established cause.

## 7. Dissection (`../20261115_f61_dissection/RESULTS.md`)

1. Check 1: the belief carries the asleep phase strongly on first sleeps (AP 0.82 against F6's 0.31; 0.12 nats per
   frame beyond time and light) and weakly on repeat sleeps (0.23 against 0.12; on 8 constant asleep frames +0.011
   nats, F6 none). Awake pixel clocks: nothing beyond time and light in either world.
2. Check 2: repeat sleeps timed in generation (section 1).
3. Checks 3, 4, 6: a timed clock with a late, smeared release; the belief expects the rise (0.61 at the due step along
   its own rollouts, 0.38 on repeat sleeps) that the denoiser draws at 0.05; the learned conditional, not the sampler.
4. Check 5: the mining pre-check failure is bf16 arithmetic (frames identical on all 402 pairs).

## 8. What this does not establish

- One world seed; the bundle cannot separate the removed token forecaster from the new heads (the classifier and
  coal / iron changes could come from either).
- The use failure is measured for the asleep energy rise only, with a linear DEV probe read on generated beliefs
  (calibrated on DEV at 0.27 for true rises against 0.07 elsewhere).
- The encoder trend is correlational across four recipes.

## 9. For the next overnight run

- **A request built a pixel clock in the belief, and the world drew it in order but late.** This is the first pixel
  clock to form, and only where a loss priced it (the reward forecast before WAKE_UP pays), with a weaker general part
  that also times repeat sleeps.
- **The binding link for that clock is now the denoiser's use of the belief**, not the belief: 0.61 expected against
  0.05 drawn at the due step. Any request that teaches the belief another clock (the gaze forecast included) should be
  expected to hit the same wall for single-cell HUD changes unless the denoiser's conditional is addressed.
- **The encoder's ownership needs a decision before more reward pressure is added:** the reward heads now send 45x
  TSP's gradient into the encoder, alongside falling token rank and coal / iron since F4.
