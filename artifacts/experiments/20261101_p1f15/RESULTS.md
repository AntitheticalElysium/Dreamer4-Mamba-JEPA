# P1-F1.5: the architectural baseline (factorised denoiser, uniform fixations, faithful decoder): results (11-02)

Run: unit saccade-p1f15, 11-01 22:51 -> 11-02 09:56 (11 h 05 min), 25.6 M frames, clean exit; checkpoint 10 by P1-F's
rule (the monitor fell at every checkpoint; recorded before the readings). Readings 1-6 as fixed in PLAN.md (1-5 before
launch, 6 added 11-02 before any result), run by `queue.sh` (unit p1f15-readings, 09:58-12:53) with P1-F re-run beside
this world on the same rows and draws (`p1f` / `p1f10` / `base` = P1-F checkpoint 10; `p1f15` / `p1f15_1500` = this
run). Summaries: `run/s1/results.json`, `readings/interface_{random,guided}.json`, `readings/eval_m05/readings_p1f15_1500-
base.{json,txt}`, `readings/codex_summary.json` (`readings_summary.py`: beta-TSP's summary, Codex's estimators, its keys
"beta" = this run), `readings/render_p1f15.json`, `readings/consistency/`, `readings/uptake/`, `readings/locus/`.
**Against P1-F this run changes two things (fixations: rule-placed on 13% of frames -> uniform; generator: joint ->
factorised), so no difference below is attributed to either alone.** One seed; descriptive, no gate.

Follow-up 11-02 (Codex, now owning this segment): native weighted gradients, outcome-chain sampler checks,
probe-independent token geometry, a controlled own-history pilot and training-window support are consolidated in
[`saccade/design/P2_READINESS.md`](../../../saccade/design/P2_READINESS.md), with protocol/raw evidence in
`../20261102_p1f15_readiness/`. These do not change this run or retroactively turn its S1 readings into S2 results.

## Instrument checks
- Interface (random gaze): P1-F's re-run equals its 10-31 output exactly (recorded quantities, generated rates, head on
  the real successor; max difference 0).
- Codex's sampler factorial: its production check (Euler-3 / churn 1 in cascade form against M.sample) exact on both
  panels; contexts asserted equal to P1-F's.
- Readings 3-4: the copied glimpse stage equalled M.sample's own glimpse bitwise on all 32 batches.
- Reading 6 (F): belief replay 0.010 (bar 0.02). S1.3's classifier power controls as P1-F's (shuffled history 0.976 /
  0.975, frozen time 0.919 / 0.914, mismatched actions 0.943 / 0.941).
- Token spaces: P1 world 1's decoder (NLL) and P1-F / P1-F1.5's (faithful) give different token spaces (monitor G1 at
  checkpoint 10: 0.0191 / 0.0286 / 0.0271); losses are compared only between P1-F and this run, and against world 1
  only as ratios.

## 1. Physical current-query invariance and reduced dependence on the training gaze
Correction 11-02 (Codex): the physical marginal is independent of the CURRENT query at fixed history by structure.
The random/guided comparisons below do not prove full-world invariance under every gaze policy. Reading 3 finds
remaining fixation-dependent integrated reward expectations, with a largest per-context range of 0.231 reward
units. The effect is not uniformly negligible on rewarding contexts.
Actor interface, this run's generated rates under random vs guided gaze (P1-F's in brackets): first wood 0.842 / 0.838
(0.038 / 0.972), first drink 0.874 / 0.874 (0.238 / 0.913), place table 0.912 / 0.901 (0.672 / 0.953), damage (risky)
0.054 / 0.052 (0.027 / 0.119), death (risky) 0.017 / 0.016 (0.012 / 0.079). Codex's HUD-counter consequences (generated
glimpse decoded): at the rule's crop / another crop containing the counter, wood 0.67 / 0.67 and drink 0.58 / 0.59
(P1-F 0.98 / 0.14 and 0.84 / 0.14: the gate); NOOP false increases 0 (P1-F drink 0.016). P1-F drew consequences where
its training gaze looked; this world draws them wherever the eye looks, at a lower rate than P1-F at its own crop.

## 2. Consequences
- **The periphery stream now draws HUD consequences; P1-F's never did.** M0.5's evaluator, random gaze, success /
  unknown (reader cannot read the generated digit) / copy of the old value: drink appearance 0.53 / 0.29 / 0.17 (P1-F
  0 / 0 / 1.00), drink increment 0.08 / 0.40 / 0.47 (0 / 0.07 / 0.90), wood appearance 0.41 / 0.33 / 0.27 (0 / 0 /
  1.00), wood increment 0.34 / 0.46 / 0.21 (0 / 0.04 / 0.96); among readable reads 0.74 / 0.13 / 0.60 / 0.62. The
  generated HUD digits are often unreadable (29-46%). Next-frame loss on those elements (M0.5's class losses, same token
  family): drink up 0.379 -> 0.163, wood up 0.398 -> 0.072; sapling up 0.518 -> 0.464; zombie hit 0.303 -> 0.277.
- Under the rule's gaze the generated glimpse draws fewer (drink increment 0.65 -> 0.18, wood increment 0.92 -> 0.64,
  drink appearance 0.96 -> 0.77): P1-F trained on exactly that gaze; the cascade's glimpse follows its own periphery.
- Actor interface, random gaze (paired, same draws, * = CI excludes 0; recorded rate first): first wood 1.00: 0.038 ->
  0.842*; first drink 0.238 -> 0.874*; place table 0.672 -> 0.912*; iron pickaxe 0.771 -> 0.866*; sapling 0.022 ->
  0.058*; damage risky (0.127) 0.027 -> 0.054*; damage adjacent (0.159) 0.007 -> 0.021*; death risky (0.128) 0.012 ->
  0.017*; random-history P(unlock) (0.034) 0.044 -> 0.040*. **Worse:** wake up (0.880) 0.783 -> 0.489* (also under
  guided gaze, -0.27*), defeat zombie 0.301 -> 0.200*, collect coal 0.159 -> 0.103*. The reward head on REAL successors
  reads unlocks under random gaze far better (first wood 0.100 -> 0.995, first drink 0.232 -> 0.895, sapling 0.601 ->
  0.918, place table 0.794 -> 0.985). Damage and death stay far below their recorded rates (0.05 / 0.02 vs 0.13).
- Failed attempts: no change in false changes (drink, table; both routes).

## 3. Generation, perception, belief, decoder
| reading | P1-F | P1-F1.5 | |
|---|---|---|---|
| S1.3 C2ST (0.5 = indistinguishable) | 0.960 | 0.946 | better |
| S1.3 generated zombies / cows / skeletons / arrows (real 0.42 / 0.47 / 0.16 / 0.09) | 0.224 / 0.282 / 0.053 / 0.132 | 0.243 / 0.304 / 0.052 / 0.122 | slightly better |
| sampler factorial Euler-3 / churn 1 (real 0.50 / 0.50 / 0.13-0.14 / 0.074) | 0.344 / 0.287 / 0.051 / 0.111 | 0.340 / 0.330 / 0.057 / 0.117 | cows better |
| Euler-8 / churn 0; Euler-32 / churn 1 (secondary) | 0.480 / 0.447 / 0.162 / 0.141; 0.482 / 0.490 / 0.182 / 0.098 | 0.471 / 0.498 / 0.166 / 0.117; 0.525 / 0.504 / 0.170 / 0.115 | |
| S1.3 reward calibration NLL / Brier | 0.0362 / 0.0071 | 0.0108 / 0.0028 | better |
| S1.4 TSP nats / pixel (day / night) | 2.30 (2.05 / 2.80) | 2.26 (2.00 / 2.79) | |
| movement shift, generated (real) (Codex's movement panel) | 0.834 (0.850) | 0.846 (0.852) | paired residual +0.0097 [0.003, 0.017] |
| periphery generation, mean mismatch, day / night (Codex panel; paired) | | +0.018 [0.000, 0.047] / +0.015 [0.003, 0.032] | worse |
| next-frame loss, map moved / still (M0.5 class losses) | 0.0646 / 0.0423 | 0.0656 / 0.0432 | +1.6% / +2.3% |
| next-frame loss, health change other than zombie hit | 0.303 | 0.328 | worse |
| night periphery token variance under renderer static (Codex panel) | | ratio 1.15 [+0.0055, +0.0089] | more static kept |
| S1.1 night presence recall (arrows) | 0.970 (0.873) | 0.965 (0.853) | |
| S1.2 belief: achievements AP / HUD exact (current-percept control 0.924) | 0.916 / 0.785 | 0.919 / 0.787 | unchanged |
The decoder render (trained world): every HUD digit and icon correct on both paths (1.0, as P1-F); HUD pixel error
higher (TSP MAE health 1.98 -> 3.27, food 1.62 -> 2.03; reconstruction health 1.34 -> 2.00), map content equal or
better (TSP iron 7.8 -> 6.2, zombies 7.2 -> 5.9, diamond 27 -> 25, ripe plant 28 -> 23; reconstruction diamond 23 -> 17,
skeletons 13.8 -> 12.1). No rare-content regression (beta-TSP's failure).

## 4. Per-query consistency and the conditioning gap (readings 3-4; 128 uniform + 128 post-DO contexts, K = 8)
- **The reward head's expectation integrated over generated glimpses depends on the fixation, beyond sampling error,
  and can be material on rewarding contexts.** In every context the spread of the 49 per-fixation means is 9-18x their sampling error (median 14 /
  12); in absolute terms it is tiny where no reward is likely, and in the 15 contexts where it is (E[r] > 0.05) the
  spread is 0.000-0.048 against E[r] 0.05-1.0. Continuation is 1.0 everywhere (no terminal contexts): uninformative.
  Across those 15 rewarding contexts the median max-minus-min of the per-fixation means is 0.0566 and the maximum
  0.2308. Example e79,t14: 0.6654-0.8962 (largest reported per-fixation MC SE 0.01194). These are finite-K selected
  extrema, not a measured learned-policy return gain. Full reward-distribution consistency was not measured here.
- Coherence (TSP render vs reconstruction render of the crop): generated 0.0015 vs real 0.0006 pixel MSE, the same on
  HUD and map windows: modestly less coherent than real percepts, no window-type-specific leak.
- Conditioning: given the real periphery (noised to 0.1), the glimpse stage reproduces the real glimpse closely (token
  MSE 0.0013, render pixel MSE 0.0019); given its own generated periphery, 0.0040 / 0.014 (post-DO 0.0034 / 0.0099).
  The second mostly measures the generated world's departure from the recorded one, not the stage's robustness.

## 5. Does a useful past glimpse reach the next physical prediction through the belief? (reading 5; F2's 1,339 frames)
- Barely. Across the 49 counterfactual glimpses of frame t, the next frame's map-row prediction varies by 1.8x (day)
  and 3.9x (night) the two noise sets' disagreement: an effect exists, of 0.06% of the loss in both strata.
- No detectable benefit of the informative glimpses: windows that held a mob the periphery missed minus windows that
  held none, at night -2.1e-5 (SE 2.1e-5, 59 frames; loss 0.116); by day -1.3e-6 (SE 1.9e-6, 28 frames); windows that
  held any mob, and the recorded window against the others, no difference either.
- Against P1 world 1's F2 (same frames and seeds; both routes, context and belief; ratios only, different token
  spaces): the spread was 0.30% (day) / 0.16% (night) of the loss there. The cut (decision 1's bet: past glimpses only
  through the belief) reduced the glimpse's influence on the next prediction by day; at night the belief alone carries
  about the same absolute spread as both routes did (6.9e-5 vs 7.3e-5).
- Decision 1's standing reading (Codex): "informative past glimpses improve the next physical predictions through the
  belief" is not supported at a detectable level in this world.

## 6. Does the world's own expectation locate the consequences? (reading 6: F re-read with surprise; 400 dev episodes)
Coverage of the consequence by the arg-max window (chance in brackets; * = locates by F's rule), this run / P1-F /
P1 world 1:
| attempt | contingency | surprise (new) | realized change (post-outcome) |
|---|---|---|---|
| coal (0.27) | 0.97* / 0.98* / 0.93* | 0.80 | 0.77 / 0.78 / 0.94* |
| stone (0.31) | 0.97* / 0.99* / 0.95 | 0.48 | 0.69 / 0.71 / 0.80 |
| drink (0.12) | **0.77* / 0.21 / 0.38** | 0.44 | 0.61* / 0.63* / 0.77* |
| iron (0.22) | 0.69 / 0.90* / 0.58 | 0.79* | 0.76* / 0.78* / 0.88* |
| wood (0.12) | 0.31 / 0.04 / 0.25 | 0.41 | 0.39 / 0.51* / 0.42 |
| sapling (0.12) | 0.32 / 0.13 / 0.22 | 0.18 | 0.04 / 0.04 / 0.07 |
Top cell of the contingency map is the consequence: drink 0.78 (P1-F 0.19), coal 0.84, stone 0.92, iron 0.56, wood
0.30, sapling 0.19. Correction 11-02: locus.py's saved HUD share counts only windows centred on row 7 (bottom
row), for which uniform is 14.3%: contingency 10%, surprise 4%. Counting ANY window overlapping HUD rows 7-8
(centres 6 or 7, consistent with selection.py/consistency.py), shares are **20.3% / 21.2%**, uniform 28.6%.
Row 6 overlaps the upper HUD row but not the bottom inventory row. Historical raw outputs are retained;
the correction is recomputed from them in `../20261102_p1f15_review/readout.json`.

## What this establishes for decision 2 (TEACHING 15)
- **The physical current-query marginal is invariant by construction**, and the old large random/guided gate is
  substantially reduced. This does not establish full-world consistency under any gaze: the outcome-head test
  above still fails. A sampling floor gives overlap; it does not prove calibration or remove policy confounding.
- **Contingency from the world's own expectation locates most attempt consequences in this world, including the HUD
  drink counter (0.77), which P1-F's world could not (0.21); wood and sapling stay weak (0.31 / 0.32).** Surprise is
  complementary (better than contingency for iron 0.79 and wood 0.41, worse for drink and stone). Both point at the
  bottom inventory strip rarely on random frames; any-HUD coverage uses the corrected shares above.
- **A short recency trace remains the proposed controller choice** (TEACHING 15). Reading 5 does not establish
  its necessity or measure the proposed controller's surprise after a look: it measures next-step whole-map
  denoising loss at noisy-real-target levels, not the pure-noise first-call forecast used by the controller.
  It finds weak evidence uptake on that statistic. Discouraging revisits and learning to use evidence remain
  separate requirements; an explicit trace does not repair memory uptake by itself.
- What uniform gaze leaves unlearned (the baseline's deficits, the chooser's targets): wake up (0.49 vs 0.88
  recorded), damage and death (0.05 / 0.02 vs 0.13), defeat zombie (0.20), coal (0.10), iron (0.03), unreadable
  generated HUD digits (29-46%), HUD pixel error up, map generation slightly worse. Which of these a chooser could
  move is not established here; the P1-F2 comparison is the test.
- Not established: anything about either change alone; anything across seeds; the cost of the controller.
