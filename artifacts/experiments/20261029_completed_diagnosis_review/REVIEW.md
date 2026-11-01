# Review of the completed generation and fovea reports

2026-10-29, reviewed at repository HEAD `68031a42`. Read-only with respect to the model, design and recipe. The only
new computation is the saved-output sensitivity analysis declared in PROTOCOL.md; zero model forward passes or
optimizer steps. Existing notebook changes and other agents' artifacts were preserved.

## What the evidence supports

The learned denoiser's outputs attenuate small consequences on the selected cohorts, including deterministic wood
and drink increments. The consequence catalogue replicates the earlier drink result. At sigma 6.67 the catalogue
records mean projections 0.238 for wood and 0.182 for drink, versus real values 0.994 and 1.001. Production generation
gives 0.148 and 0.140; Heun 32 gives 0.204 and 0.227. Movement is much better preserved (real 0.930, production 0.909).
These projections are not probabilities or physical event counts.

This is useful localization: the problem is visible in learned predictions, not only at the end of full generation.
Sampling choices also change the deployed prediction. No tested recipe restores the measured consequences. The
matched risky subset in D has 12.8% recorded deaths, versus native-head means approximately zero for production and
1.57% for Heun 32. D's JSON also includes A's full-cohort references; those are not the matched D-subset reference.

## Generation claims that remain too strong

1. A high-noise conditional-mean projection is not the probability mass of a branch and is not an upper bound on
   every possible sampler. The exact-mixture test shows that a particular numerical algorithm can distort known
   mixture weights. It does not quantify a second, independent tenfold causal loss in M0. A and B cannot be chained
   by treating A's projection as B's mixture probability.
2. E's 0.45 versus 0.07 is not a matched cooldown comparison. In information_probe.py:100-101, the first group is
   **ready AND adjacent after the move**; its complement includes nonadjacent zombies as well as cooling-down ones.
   The probe demonstrates accessible attack-relevant information, not that cooldown alone is encoded and ignored.
3. Seeing the prerequisites in pixels is not identical to their being exactly measurable from the denoiser's
   compressed, noisy conditioning. The tower-property null is exact under that measurability condition, including
   the coefficients of the context-specific linear functional. Direction decodability supports one prerequisite,
   not exact sufficiency of all conditioning. Drink is strong descriptive evidence; the complete information
   pathway has not been cleared.
4. Low-noise agreement is a useful consistency check involving the learned model, not a model-independent
   instrument certification. TABLE reaches its defined outcome in 96.9% of rows, below catalogue.py's declared
   deterministic >=99% control. That arm needs qualification. The wood/drink findings do not depend on TABLE.
5. The geometry statistic behaves differently across samplers. That does not prove geometry and consequence errors
   have independent underlying causes.

## Fovea: the negative-cost claim is not robust

The original F1 calculation is reproduced: all-day lag-1 gain is -0.071390 nats/pixel. But its comparison group is
**not glimpsed in any of eight frames**, not simply **not glimpsed one frame earlier** (analyse.py:67-78).
Motion, screen coverage, and available episode history affect membership in this derived group. Random fixation
centres do not automatically make it a matched causal comparison after stratifying only by current screen cell
and day/night.

The additional analysis compares lag-1 inclusion versus exclusion, irrespective of earlier fixations, using the
known inclusion probabilities under all 49 uniform crop centres. The probability formula was independently checked
by exhaustive enumeration on 225 screen positions. Standard errors bootstrap 250 physical episodes, 500 resamples.
Positive gain means lower subsequent TSP loss after exposure.

| Population | Original lag-1 versus never-in-8 gain | One-step normalized-IPW gain | Bootstrap SE |
|---|---:|---:|---:|
| All day | -0.07139 | +0.00913 | 0.01796 |
| All night | -0.02138 | -0.00664 | 0.01580 |
| Static camera, same terrain code, player excluded, day | -0.06852 | -0.01246 | 0.01843 |

These are post-hoc sensitivity checks with a different comparator/target population, not a replacement gate. They do
not demonstrate useful memory or refute every possible cost over several fixations. They do show that the stated
immediate causal cost does not survive this simpler one-step comparison. Horvitz-Thompson estimates are also saved
in checks.json; no estimate is selected as a new passing gate.

The original addendum's phrase "unchanged cell" is also too strong: it checks absence of entities, not equality of
terrain codes (analyse.py:89-94). The stricter analysis above checks terrain-code equality and excludes the player;
it still does not assert identical pixels because lighting/noise can change.

F1 measures use through the trained TSP readout of the predicted-belief output. It cannot establish that the entire
belief contains no glimpse information. Nor does the small gap between two trained pixel heads constitute a
ceiling on the additional information in high-resolution input. The denoiser also directly reads recent glimpse
tokens through its eight-frame context, a route F1 does not test.

## Fovea: F2 has a valid narrow result and two interpretation limits

The raw arrays reproduce the reported oracle improvements relative to best fixed fixation: 0.612% day, 0.258%
night, 0.254% for the separate night/mob set. These concern the model's bulk one-step periphery/lighting denoising
loss, not reward, survival, information value for decisions, or long-range memory. The oracle chooses its fixation
using a denoising loss computed with the realized next frame. Independent noise sets reduce noise selection bias;
they do not make that future-informed choice a deployable policy using current observations alone.

The R3 HUD excess statistic is structurally zero. analyse.py:161 subtracts each fixation position's mean value;
:163-168 then averages across the HUD-covering positions, whose membership is fixed by screen position. Each
position's residual already has zero sample mean. Recomputed values are about 1e-10 in magnitude. This cannot
support "HUD looking has no value." The uncentred positional map R1 is a separate, interpretable result.

An unsuccessful relationship with TSP-minus-reconstruction is not a general certification against noisy-TV
behaviour. There is no learned fixation policy yet, and that head-loss gap is not a direct measure of irreducible
sensor-noise entropy. The narrow result is no measured positive correlation in that particular proxy.

## Literature and completion claims

OC-STORM and Segmentation Dreamer support the concern that task-relevant small regions can be neglected by broad
visual objectives. They use pretrained/few-shot or simulator segmentation and change representation/target
allocation; they do not isolate uniform latent loss as M0's cause. Dreamer 4's fast-sampling ablation motivates
checking few-step sampling, not selecting a fix for this model.

Sources read:
- https://arxiv.org/html/2501.16443 (introduction, method and limitations).
- https://arxiv.org/html/2410.09972 (introduction, method and experiments).
- The earlier source-by-source audit is ../20261029_generation_literature/READING.md.

"All six tests finished" needs a scope qualification: generation RESULTS.md explicitly says F's bulk/tail summaries
were not run. C's native-head part is reported; the planned separate external HUD-reader qualification is not
reported in these result artifacts. A-F headings do not certify every planned subtest as complete.

## Appropriate next reading

A matched checkpoint trajectory for movement, wood and drink is worthwhile: same histories/actions, same noise
levels/draws, valid references and positive controls at each checkpoint, and production samples alongside denoising.
Encoders change across checkpoints, so compare against each checkpoint's own real-target scale rather than raw
latent loss alone. Such a trajectory shows whether this run improved, stalled or regressed; it does not prove what
longer training would do or isolate capacity, weighting and coverage.

Seed 2 remains replication, not a causal diagnosis. The present findings do not choose a new acuity, sampler or
training objective, and they do not establish failure of the general fovea hypothesis.
