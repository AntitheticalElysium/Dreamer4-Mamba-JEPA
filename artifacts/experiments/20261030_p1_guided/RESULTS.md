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

The increments M0.5 could not draw are drawn when the glimpse is on the counter. On the rows eligible in both worlds
(Codex, 10-31; the standalone rates above use different reader-qualified populations): drink increment 0.038 ->
0.728, wood increment 0.083 -> 0.936. Under the guided route the generated periphery agrees with the correct glimpse
(lower bounds, verified success and coherent): drink appearance 0.897, wood appearance 0.923, drink increment 0.471,
wood increment 0.694, table 0.451. Under the random route the periphery draws no HUD change (copies, all readable),
though the reader reads P1-F's real periphery tokens perfectly (1.0). Several rows are UNKNOWN-HEAVY (world 1's
unknown shares 0.11-0.35). Movement shift (periphery) 0.910 -> 0.858 (paired difference -0.052 [-0.057, -0.047]):
worse. Reward NLL +0.011 worse. Dev G1 loss by class (random-route encoding; both token spaces per-token LayerNorm
then the same clamp): map still 0.022 -> 0.042, map moved 0.034 -> 0.065, drink up 0.16 -> 0.38, wood up 0.11 ->
0.40, zombie hit 0.20 -> 0.30, health other 0.15 -> 0.30, table placed 0.13 -> 0.12, glimpse 0.032 -> 0.024,
lighting 0.0056 -> 0.0034. P1's monitor (held-out G1, random fixations) at checkpoint 10: 0.0191 -> 0.0286, higher at
every noise level.

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
3. Those gains depend on the gaze P1-F was trained with, not on the glimpse alone: under the rule's gaze the generated
   periphery carries them too (section 3); under random gaze neither glimpse nor periphery draws them. The rule is
   Q = 1 and deterministic in its context, so P1-F never saw an attempt frame without the eye on the counter. Under
   random gaze its reward head misses exactly the rule's events on REAL frames (first wood 0.10, drink 0.23, sapling
   0.60, table 0.79; world 1 0.97, 0.74, 0.89, 0.95; non-rule unlocks 0.91-1.00) although its belief (0.80 exact) and
   tokens (decoder 1.000, reader 1.0) carry the count. World 1 reads the same under both policies.
4. The guided policy's placement is computed from the true state: 123 of the rule's 129 contexts are "zombie
   adjacent", most at the health window (7, 1). The guided death and damage gains are therefore confounded with a
   label-derived cue (PLAN's competing explanation b): policy-specific, not world quality.
5. One-step generation regressed broadly under random gaze (C2ST 0.761 -> 0.960, monitor +50%, map G1 ~2x, movement,
   skeleton counts, reward calibration). The own-gaze re-read (follow-up readings) splits it. Map dynamics (map G1
   ~2x, movement, skeleton counts), health changes other than zombie hits (~2x) and the C2ST stay worse under P1-F's
   own gaze: a regression of the recipe,
   guidance in training vs the harmoniser re-weighting (section 5) not separable. Reward calibration and the HUD
   consequences were the evaluation's gaze: under its own gaze P1-F is better than world 1 on both (label-derived
   placement, item 4).

## C2ST localization (`localize_c2st.py`, `readings/c2st_localization.json`; user 10-31, Codex post hoc)
S1.3's path and seeds, one outcome group at a time (all tuples), world 1 -> P1-F: periphery map 0.570 -> 0.706,
periphery HUD 0.800 -> 0.919, glimpse 0.736 -> 0.857, lighting 0.567 -> 0.626, reward + termination 0.511 -> 0.512.
All outcomes on non-attempt tuples (87%): 0.803 -> 0.958. Attempt tuples alone: 0.506 -> 0.529, both at chance, so
underpowered (2,736 test tuples) and uninformative. Measurement spread: P1-F reproduces S1.3 (0.961 vs 0.960), world
1 does not exactly (0.731 vs 0.761), and world 1's non-attempt subset reads above its full set (0.803 vs 0.731), so
fits vary by up to ~0.07; lighting's difference is inside that, the map / HUD / glimpse differences (0.12-0.14)
narrowly outside it. Reading: the regression is not confined to the rule's own frames or to one token group. The
declared reading (non-attempt map regression points at the grounding balance rather than guidance) does NOT hold:
45% of the non-attempt tuples have a rule context in the 8 frames before their deciding state (Codex, 10-31;
reproduced on CPU, 0.451), encoded with random fixations off P1-F's training distribution, and guidance changes the
shared parameters everywhere. Where, not why; the declared split was not clean. (Withdrawn: "tokens half as spread,
so equal error is twice as large": the 0.023 / 0.013 statistic is the raw p_out output BEFORE the per-token
LayerNorm; the denoiser sees normalised tokens in both worlds. Post-normalisation effective rank 4.0 -> 5.7 is a real
difference.)

## Follow-up readings (user 10-31; PLAN "Follow-up readings"; Codex post hoc)
### Own-gaze re-read (`own_gaze.py`, commit 1ac1c723, unit p1f-own-gaze, 1 h 52 min, clean exit; `readings/own_gaze/`)
Guided gaze refixates 16.8% of dev frames (the rule-decided ones), the same frames for both worlds. Periphery tokens
do not depend on the fixation (per-cell MLP; only the crop does), so the G1 targets on periphery cells are identical
under both gazes: only the predictions' inputs change (the predicted frame's fixation and, where an earlier context
was a rule context, the history's glimpses). The random rows reproduce the earlier readings exactly (P1-F's S1 equals
`run/s1` in every S1.3 number; world 1's C2ST 0.7613 equals S1's; class losses and movement as section 3).

| | world 1 random | world 1 guided | P1-F random | P1-F guided |
|---|---|---|---|---|
| C2ST accuracy | 0.761 | 0.848 | 0.960 | 0.942 |
| power: frozen time (shuffled 0.98 / 0.98 / 0.975 / 0.975, mismatched 0.94 all) | 0.907 | 0.857 | 0.914 | 0.898 |
| reward NLL (ECE) | 0.0251 (0.0011) | 0.0250 (0.0011) | 0.0362 (0.0043) | 0.0073 (0.00008) |
| generated skeletons / cows / zombies (real 0.15-0.16 / 0.47 / 0.42) | 0.138 / 0.25 / 0.235 | 0.138 / 0.25 / 0.235 | 0.052 / 0.28 / 0.224 | 0.052 / 0.28 / 0.227 |
| movement shift | 0.910 | 0.910 | 0.858 | 0.858 |
| G1 map still / map moved | 0.0216 / 0.0336 | 0.0216 / 0.0336 | 0.0423 / 0.0646 | 0.0423 / 0.0646 |
| G1 all periphery cells | 0.0224 | 0.0224 | 0.0414 | 0.0412 |
| G1 drink up / wood up / sapling up | 0.158 / 0.106 / 0.443 | 0.161 / 0.108 / 0.438 | 0.378 / 0.398 / 0.517 | 0.087 / 0.031 / 0.230 |
| G1 table placed / zombie hit | 0.132 / 0.201 | 0.131 / 0.203 | 0.122 / 0.303 | 0.095 / 0.213 |
| G1 health other / HUD other change / HUD static | 0.148 / 0.137 / 0.0080 | 0.148 / 0.137 / 0.0080 | 0.303 / 0.154 / 0.0071 | 0.297 / 0.153 / 0.0068 |
| G1 glimpse | 0.032 | 0.035 | 0.024 | 0.024 |
| TSP nats / pixel | 2.18 | 2.09 | 2.30 | 2.21 |

S1.1 perception: no reading moves by more than 0.01 with the gaze in either world. S1.2's belief probe: equal under
both gazes in world 1; in P1-F apart from collect diamond (AP 0.53 -> 0.50) and recently departed zombies (AP at 1-4
steps 0.13 -> 0.15). S1.2's percept-only and last-8-steps controls do move: they read the glimpse directly, and the rule's placement marks zombie
adjacency (last-8 departed-zombie AP at 1-8 steps: world 1 0.06-0.09 -> 0.15-0.17, P1-F 0.08-0.11 -> 0.18-0.22).

Instrument check (world 1, guided vs random): passes for the G1 classes (largest move 0.003), movement, counts,
perception, belief and reward calibration. It FAILS for the C2ST: +0.087, above the ~0.07 fit spread, frozen-time
control -0.05. The C2ST depends on where the eye is, not only on the world, so the declared comparison (P1-F guided
against world 1 random) mixes a world difference with a test difference. Why world 1 reads worse under guided gaze is
not measured; the refixated frames alone could raise it to 0.80 at most (0.832 x 0.761 + 0.168), so the rise reaches
beyond them.

Declared reading applied, instrument by instrument:
- Regression of the recipe: map dynamics, and health changes other than zombie hits. Under P1-F's own gaze map G1 is
  still ~2x world 1's (still 0.042 vs 0.022, moved 0.065 vs 0.034), movement 0.858 vs 0.910, generated skeletons 0.052
  vs 0.138, health other 0.30 vs 0.15; none moves by more than 0.006 with P1-F's gaze. C2ST 0.942 vs 0.761 is clearly worse by the declared comparison and at
  matched gaze too (random +0.20, guided +0.09; P1-F itself moves -0.018, within spread): the gap's size depends on the
  gaze through world 1. The periphery G1 total, which the map dominates, does not move with the gaze (0.0414 /
  0.0412), consistent with the monitor's +50% being the recipe's. Guidance in training vs the harmoniser re-weighting
  remains inseparable by reading (declared).
- The evaluation's gaze: reward calibration and the HUD consequences. Under P1-F's own gaze its reward NLL is 0.0073,
  under a third of world 1's 0.025 (ECE 0.00008 vs 0.0011), and the consequence classes are below world 1's (drink up
  0.087 vs 0.158, wood up 0.031 vs 0.106, sapling up 0.23 vs 0.44, table 0.095 vs 0.13; zombie hit 0.21 vs 0.20,
  equal). These own-gaze numbers carry the label-derived placement (establish 4).
- Mechanism, within what is measured: the targets are the same under both gazes, so P1-F's drink-up loss falling
  from 0.378 to 0.087 comes from the predictions' inputs alone, while the next section shows the digit loud in its
  real periphery tokens. P1-F's prediction of a counter change keys on the gaze the rule computed from the context,
  not on the action and faced block it was computed from: policy confounding (Suau, Spaan, Oliehoek, "Bad Habits",
  RLC 2024), the rule having made the fixation a perfect proxy of the context in training. Whether the key is the
  predicted frame's fixation or the history's glimpses is not separated here.

### Glimpse token geometry (M0.5's `dissect/token_geometry.py` on this run's readers; `readings/token_geometry.json`)
World 1 refit here reproduces M0.5's base to every printed digit (M0.5 at 1500 updates equals its base). Step between
adjacent counter values over the within-value radius, mean over the glimpse windows holding the counter (share of
the token's squared norm):

| | world 1 | P1-F |
|---|---|---|
| glimpse drink increment | 0.37 (0.009) | 1.51 (0.39) |
| glimpse wood increment | 0.25 (0.005) | 1.61 (0.39) |
| glimpse health +-1 / damage | 0.40 (0.016) / 0.58 | 1.99 (0.47) / 2.15 |
| glimpse appearance: drink, wood, sapling (range over windows) | 2.3-3.4, 2.7-3.7, 1.5-2.3 | 2.3-3.0, 2.2-3.2, 1.9-3.0 |
| periphery increment: health, drink, sapling, wood | 4.8, 3.9, 3.5, 4.2 | 12.7, 11.8, 19.3, 13.0 |

Reading (declared): above 1, so the digit is now a prominent direction of the glimpse token (M0.5's dissection:
~0.3x, "glimpse digit ~invisible in token space"); appearance steps unchanged. Not attributable to the decoder or to
guidance. With the re-read: under random gaze the digit is loud in P1-F's real tokens on both paths, so the random-gaze
failures (no change drawn, the reward head's misses on real frames) are not a perception failure.

## Retracted or not supported (mine)
- "The 30x weaker pixel pull may have lost the HUD from the tokens": refuted (decoder 1.000, digit probe 1.0, belief
  HUD equal or better, reader 1.0 on real periphery tokens).
- The launch check's "same order at initialisation" did not carry to training (section 5).
- "Those gains live only in the guided glimpse" / "the periphery never draws the change": wrong as stated (Codex);
  the periphery draws them under the rule's gaze; the dependence is on the gaze policy (section 3).
- The C2ST localization's declared non-attempt reading and the token-scale explanation (that section).
- Omitted from the first write-up: world 1's monitor and the G1 loss by class (section 3, now reported).
- Not run: PLAN reading 3's glimpse token geometry (M0.5 dissection), declared and skipped (run 10-31: follow-up
  readings).
- The follow-up PLAN took world 1's gaze invariance at the actor interface as holding for every instrument: not for
  the C2ST (+0.087 under guided gaze).

## Limits
One seed per arm (P1 world seed 2 not run: no seed spread for any difference); three recipe differences, none
isolated; the monitor was read under random gaze only (S1, the class losses and movement under both: follow-up
readings); the C2ST is not gaze-invariant (world 1 +0.087), so cross-gaze C2ST comparisons mix world and test;
guided-route readings carry the label-derived placement.
