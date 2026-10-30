# P1-F results (10-31): guided fixation + faithful pixel decoder, from scratch, against P1 world 1

Run: unit saccade-p1f, commit 3b9b33fa, 00:25-09:42 (9 h 17 min), 25.6 M frames, clean exit. Monitor fell at every
checkpoint (0.106 -> 0.0286), so P1's rule selects checkpoint 10 (`run/ckpt_10.pt`, sha256 82b0ecc0...6e6271).
Control: P1 world 1 checkpoint 10 (74eccc43...). Readings as declared in PLAN.md, run by hand in its order. The
comparison is of the combined recipe; nothing below assigns an effect to guidance or to the decoder alone.

## 5. Training balance (logs; read first because it qualifies the rest)
G2 (change map + TSP + reconstruction) shares one harmoniser sigma. Under world 1's NLL the group loss was ~2
(sigma 1.98, weight 0.51); under faithful's squared error it is ~0.025 (sigma 0.36, weight 2.78). Raw squared-error
gradients are also far smaller. Encoder pull, checkpoints 5-8 mean, times the checkpoint-8 weight (one-stream
snapshots): TSP 36 -> 1.2, reconstruction 17 -> 0.09, change map 1.1 -> 2.0. World 1's encoder was trained mainly by
pixels; P1-F's mainly by the change map and reward. Periphery tokens: channel std 0.023 -> 0.013, effective rank
4.0 -> 5.7; lighting RMS 0.047 -> 0.086. Training losses (last 50 logged updates): change 0.042 -> 0.023, reward
0.024 -> 0.0036. The launch check (gradients at initialisation, same order) did not predict this: the NLL's
gradients grow as its scale narrows, squared error's shrink with the error. So P1-F differs from world 1 in three
ways, not two: guidance, decoder loss, and through the latter the encoder's grounding balance.

## 4. Decoder rendering of the trained world (`readings/render_p1f.json`; world 1: decoder diagnosis "trained")
The fix holds in joint training. Every HUD counter, both paths: digit and icon 1.000 (world 1 digits: health 0.014,
food 0.033, drink 0.165, energy 0.025, wood 0.674, stone 0.415; body icons 0). Digit probe on the path features 1.0
(world 1 0.98-1.0). Mean absolute error (0-255), reconstruction path: iron 19.1 -> 4.1, furnace 17.0 -> 4.2, table
9.1 -> 3.6, coal 6.4 -> 2.6, diamond 36 -> 23, zombies 23.8 -> 5.4, skeletons 26.5 -> 13.8, cows 7.2 -> 4.4; grass
1.38 -> 0.90, path 2.8 -> 1.7; blurrier: water 0.88 -> 1.24, stone 0.77 -> 1.18, tree 1.03 -> 1.88, lava 3.6 -> 5.2;
arrows worse 10.4 -> 14.9; night 4.43 -> 4.09. TSP's path the same pattern. The weaker pixel pull did not cost the
tokens HUD content.

## 2. S1 (`run/s1/results.json`; world 1 `artifacts/saccade_m0/p1_world1/s1`)
- S1.1 perception: day equal (~1.0); night slightly better (presence recall r<=4 0.852 -> 0.878; arrows 0.73 ->
  0.78-0.82). The night-static risk did not show in perception.
- S1.2 belief: HUD exact equal or better (wood 0.69 -> 0.80, health 0.94 -> 0.95, food 0.39 -> 0.42; drink 0.48 ->
  0.44); achievements and departed-entity retention equal.
- S1.3 one-step generation: C2ST accuracy 0.761 -> 0.960, with power controls equal in both worlds (shuffled history
  0.97, mismatched actions 0.94, frozen time 0.91): P1-F's generated next percepts are easier to tell from real than
  real outcomes of the wrong action. Broad: attempt frames (~13%) alone could raise it to ~0.79 at most. Generated
  counts (real): skeletons 0.138 -> 0.053 (0.157), cows 0.25 -> 0.28 (0.47), zombies 0.23 -> 0.22 (0.42). Reward
  calibration on real percepts worse: NLL 0.025 -> 0.036 (S0's no-memory pair reference 0.024), ECE 0.0011 -> 0.0043.
- S1.4 TSP NLL 2.18 -> 2.30 nats / pixel (faithful does not train the mean for likelihood: not a quality reading).

## 3. M0.5's evaluator (`readings/eval_m05`, `readings_p1f-base.txt`)
Verified success of the generated cell, world 1 -> P1-F:

| row | glimpse (guided route) | periphery (random route) |
|---|---|---|
| drink appearance | 0.064 -> 0.951 | 0.132 -> 0.000 |
| drink increment | 0.038 -> 0.649 | 0.004 -> 0.000 |
| wood appearance | 0.123 -> 0.974 | 0.164 -> 0.000 |
| wood increment | 0.083 -> 0.918 | 0.011 -> 0.000 |
| table | 0.001 -> 0.457 | 0.242 -> 0.397 |
| damage, risky (recorded 0.205 / 0.139) | 0.000 -> 0.112 | 0.001 -> 0.001 |
| damage, adjacent (0.222 / 0.156) | 0.072 -> 0.063 | 0.002 -> 0.000 |
| false drink change on failed attempts | 0.66 -> 0.07 | 0.115 -> 0.004 |

The increments M0.5 could not draw are drawn when the glimpse is on the counter. The periphery draws no HUD change
(copies, all readable), though the reader reads P1-F's real periphery tokens perfectly (1.0). Several rows are
UNKNOWN-HEAVY (world 1's unknown shares 0.11-0.35). Movement shift (periphery) 0.910 -> 0.858 (paired difference
-0.052 [-0.057, -0.047]): worse. Reward NLL +0.011 worse.

## 1. Actor interface (`readings/interface_random.json`, `interface_guided.json`; corrected evaluator)
Generated rates, world 1 -> P1-F (recorded):

| | random gaze | guided gaze |
|---|---|---|
| death, risky (12.8%) | 0.04% -> 1.2% | 0.03% -> 7.9% |
| damage, risky (12.8%) | 4.1% -> 2.7% | 5.1% -> 11.9% |
| damage, adjacent (15.9%) | 3.9% -> 0.7% | 4.1% -> 5.9% |
| first wood: head on real successor / generated | 0.97 / 0.71 -> 0.10 / 0.04 | 0.98 / 0.72 -> 0.997 / 0.97 |
| first drink: head on real / generated | 0.74 / 0.73 -> 0.23 / 0.24 | 0.72 / 0.71 -> 0.94 / 0.91 |
| place table, generated | 0.89 -> 0.67 | 0.89 -> 0.95 |

Under both policies P1-F is better on mobs and rare unlocks (random: zombie 0.12 -> 0.30, skeleton 0.22 -> 0.49,
cow 0.14 -> 0.22, stone 0.44 -> 0.82, eat plant 0.02 -> 0.30; head on real: diamond 0.52 -> 0.93). Death
discrimination (held-out reliability): random, 95 risky histories flagged >= 20% with recorded death 28-50% (world
1: none); guided, 463 flagged at 33-38% recorded vs 9.6% unflagged, over-confident at the top (0.65 predicted, 0.38
recorded). Per-achievement rows are outcome-selected (recall only).

## What the readings establish
1. The faithful decoder repairs pixel grounding in joint training: every HUD value and icon rendered, rare blocks
   and mobs far better, at a small blur on common textures.
2. With the glimpse on the consequence cell, P1-F generates the HUD consequences (increments included) and stops
   inventing failed-attempt changes. That was M0.5's open failure.
3. Those gains live only in the guided glimpse. The rule is Q = 1 and deterministic in its context, so P1-F never saw
   an attempt frame without the glimpse on the counter. Under random gaze, attempt frames are off its training
   distribution: its reward head misses the first wood on REAL frames (0.10) although its belief (0.80 exact) and
   tokens (decoder 1.000, reader 1.0) carry the count. Its periphery never draws the change.
4. The guided policy's placement is computed from the true state: 123 of the rule's 129 contexts are "zombie
   adjacent", most at the health window (7, 1). The guided death and damage gains are therefore confounded with a
   label-derived cue (PLAN's competing explanation b): policy-specific, not world quality.
5. One-step generation regressed broadly (C2ST 0.761 -> 0.960, movement, skeleton counts, reward calibration). Cause
   not localized: the harmoniser re-weighting (section 5) and guidance both remain candidates.

## Retracted or not supported (mine)
- "The 30x weaker pixel pull may have lost the HUD from the tokens": refuted (decoder 1.000, digit probe 1.0, belief
  HUD equal or better, reader 1.0 on real periphery tokens).
- The launch check's "same order at initialisation" did not carry to training (section 5).

## Limits
One seed per arm; three recipe differences, none isolated; guided-route readings carry the label-derived placement.
