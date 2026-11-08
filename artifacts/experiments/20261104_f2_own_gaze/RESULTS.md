# P1-F2 under its own gaze: the regressions are not the evaluation gaze

2026-11-04 (user: run, then report; Codex's review -> closed-loop amendment, PLAN.md). Actor interface only: the user
skipped S1 tonight to free the GPU. One world seed each; intervals are evaluation uncertainty, not training-seed
uncertainty.

**Answer.** F2's generated consequences barely depend on the gaze it is read with. Under its own closed-loop gaze every
class is within 1.3 pp of the fixed random gaze, and the leak Codex identified (the conditional window chosen on the
recorded frame) moves no class by more than 0.25 pp. So, by the declared reading, F2's regressions against F1.6 (wood,
stone, zombie, wake-up, wood / iron pickaxe, sapling) are NOT the evaluation gaze: they are properties of the trained
world (the acquisition recipe, or training-seed variation, which one seed per world cannot separate). Its gains
(coal, iron, cow, skeleton) also stay. The random-gaze review table stands as F2's reading.

## Instrument checks
- F1.6 and F2 under random gaze reproduce the archived queue output in all 3,440 numeric fields (max difference 0).
- Conditional and closed-loop F2 share encodings and world draws exactly (real-successor outputs identical, 0.0).
- The closed-loop hook fed the RECORDED coarse frame reproduces all 2,384 conditional windows of six dev episodes.
- Own gaze on dev (frames t >= 1): mob crop coverage 49.5% vs 20.6% under the prior (training trace: 48.7 / 20.7%);
  HUD-touching windows 26.9% (natural-frame review: 26.5%). The gaze is the trained policy's.

## Generated event mass (P(reward > 0.5); risky: P(termination); adjacent: P(reward < -0.05)), dev, K = 4

| Class | n | Recorded | F1.6 random | F2 random | F2 own (cond.) | F2 own (closed) | closed - F1.6 [95%] | closed - F2 random [95%] | closed - cond. | window agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 6000 | 3.4 | 3.97 | 4.01 | 3.99 | 4.00 | +0.02 [-0.17, +0.22] | -0.01 [-0.06, +0.04] | +0.00 | .33 |
| sapling context | 2000 | 13.5 | 7.79 | 5.58 | 5.60 | 5.63 | -2.16 [-2.82, -1.51] | +0.05 [-0.10, +0.20] | +0.03 | .39 |
| risky (death) | 3805 | 12.8 | 1.09 | 1.27 | 1.28 | 1.29 | +0.20 [-0.05, +0.45] | +0.02 [-0.04, +0.09] | +0.00 | .31 |
| adjacent (damage) | 3000 | 15.9 | 1.63 | 1.23 | 1.23 | 1.22 | -0.40 [-0.74, -0.07] | -0.01 [-0.06, +0.04] | -0.01 | .33 |
| COLLECT_WOOD | 300 | 100 | 88.45 | 82.33 | 81.93 | 81.83 | -6.62 [-9.63, -3.61] | -0.50 [-1.51, +0.50] | -0.10 | .43 |
| COLLECT_STONE | 300 | 100 | 90.47 | 83.32 | 84.24 | 84.30 | -6.17 [-8.67, -3.68] | +0.97 [-0.00, +1.95] | +0.06 | .43 |
| DEFEAT_ZOMBIE | 300 | 100 | 29.53 | 23.28 | 23.68 | 23.57 | -5.96 [-8.17, -3.76] | +0.29 [-0.21, +0.79] | -0.11 | .60 |
| WAKE_UP | 300 | 88 | 49.53 | 39.69 | 39.85 | 39.87 | -9.67 [-12.29, -7.05] | +0.17 [-0.21, +0.56] | +0.01 | .27 |
| MAKE_WOOD_PICKAXE | 300 | 100 | 91.76 | 87.05 | 87.89 | 87.88 | -3.87 [-6.29, -1.46] | +0.83 [+0.04, +1.61] | -0.01 | .52 |
| MAKE_IRON_PICKAXE | 124 | 100 | 88.37 | 82.46 | 83.26 | 83.30 | -5.07 [-8.67, -1.48] | +0.84 [-0.47, +2.15] | +0.03 | .48 |
| COLLECT_SAPLING | 300 | 100 | 5.22 | 2.67 | 2.97 | 2.93 | -2.29 [-3.69, -0.89] | +0.26 [-0.04, +0.57] | -0.04 | .15 |
| EAT_COW | 300 | 100 | 17.22 | 20.66 | 20.36 | 20.31 | +3.09 [+0.89, +5.28] | -0.35 [-1.05, +0.36] | -0.05 | .36 |
| DEFEAT_SKELETON | 300 | 100 | 52.41 | 56.49 | 57.54 | 57.38 | +4.97 [+2.70, +7.24] | +0.89 [+0.08, +1.70] | -0.17 | .45 |
| COLLECT_COAL | 300 | 100 | 17.49 | 24.37 | 25.43 | 25.67 | +8.18 [+5.33, +11.04] | +1.30 [+0.36, +2.23] | +0.25 | .23 |
| COLLECT_IRON | 287 | 100 | 5.46 | 9.18 | 8.71 | 8.68 | +3.22 [+1.35, +5.10] | -0.50 [-1.23, +0.24] | -0.03 | .17 |

All 26 classes (including tables, furnace, swords, plants, drink, diamond n = 12, eat plant n = 25) are in
`readings/interface.json`; none moves by more than 1.3 pp between random and own gaze. Real-successor heads are
equally gaze-insensitive (e.g. risky 12.77 under every gaze). Window agreement (common random numbers: identical maps
give identical windows) is the share of futures whose closed-loop window equals the conditional one; low agreement
with near-zero outcome change means the heads and generator hardly read the generated glimpse's location.

## What this does and does not establish
- Established: F2's one-step generated outcomes are gaze-insensitive at evaluation (own vs random <= 1.3 pp; the
  leak <= 0.25 pp). The comparison with F1.6 is therefore not an evaluation-gaze artifact.
- Not established: that learned acquisition CAUSED the regressions. F2 and F1.6 are one training seed each; per-class
  differences of 2-10 pp are within what two seeds of one recipe might differ by (uncalibrated).
- Not read tonight: S1 (C2ST, reader counts, belief probes) under own gaze; a closed-loop C2ST needs the window moved
  into the outcome features (a different instrument). Multi-step imagination under own gaze is not read.
- Consistent with the earlier uptake readings: the world's consequences ride on the belief and the periphery, not on
  where the eye is at the generated step.
