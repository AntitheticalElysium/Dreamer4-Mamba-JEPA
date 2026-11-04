# P1-F1.5 results and TEACHING 15 review

2026-11-02. Source snapshot 5d84c53b; inputs and independently recomputed outputs in `readout.json`.
No canonical changes, no training or GPU job launched/interrupted, no new toy-harness ablation. The separate
generation-regression repair investigation remains on hold. CPU suite: 53 passed, 5 GPU skips, 20.88 s.

## Overall reading

The architectural baseline is materially better on several formerly gaze-gated consequences. It now supports a
useful controlled comparison of an emergent fixation curriculum. It is not an adequate general imagination world
yet: wake, mining/combat outcomes, danger/death, unreadable generated digits, and useful glimpse uptake remain weak.
This is neither a failed overnight run nor a demonstrated repair of the whole earlier problem.

Uniform fixation and factorized generation changed together. Improvements cannot be assigned to either one alone;
one world seed does not establish a recipe-wide effect. No actor was trained/evaluated by these readings.

## 1. Identity and measurement checks

- Checkpoint 10 hash 95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128 matches S1 and the new readings.
- Completed 25,600,000 frames / 15,625 updates. All ten monitor values decrease, .09856 -> .02706; checkpoint 10
  follows the predeclared rule. Faithful TSP/reconstruction, factorized=True, guided config absent. All four saved
  training-source hashes match the current files.
- S1 now dispatches the factorized world through M.sample and frame-by-frame context commits, with early-history
  grouping instead of padded-frame commits. The added tests cover t=1,6,24. Existing joint path is preserved.
- The new consistency evaluator checks its copied stage 2 against production on each of 32 batches. Saved flag is
  true. The codex sampler adapter checks production and previous-world context identity.
- Actor-interface unlock means are head probability mass on reward >.5, with a real-successor reference. Achievement
  cohorts are outcome-selected recall diagnostics, NOT unconditional event calibration or real actor performance.
  Risky/adjacent cohorts instead come from pre-outcome conditions. K=4 futures does not multiply independent scene
  support; paired/episode uncertainty is conditional on these trained worlds.

## 2. Important improvements and remaining failures

Random-gaze actor interface, mean probability of reward>.5 on the relevant recorded achievement cohort:

| Cohort | Recorded event | P1-F | P1-F1.5 | F1.5 head on real successor |
|---|---:|---:|---:|---:|
| First wood | 1 | .0381 | .8422 | .9952 |
| First drink | 1 | .2382 | .8739 | .8947 |
| Place table | 1 | .6721 | .9119 | .9852 |
| Wake | .88 | .7830 | .4885 | .8419 |
| Defeat zombie | 1 | .3011 | .1997 | .7580 |
| Coal | 1 | .1589 | .1031 | .9935 |
| Iron | 1 | .0359 | .0285 | .9892 |

Many crafting outcomes improve too. Near-.99 real-successor recognition on coal/iron versus .10/.03 with generated
successors keeps generation/its interaction with the outcome readout central; it does not mean a chooser has been
shown to fix these cases. Zombie recognition is imperfect even on real evidence, so its whole deficit cannot be
assigned to the denoiser.

In risky contexts recorded death .1277 versus generated .0175, real-successor head .1277; damage .1275 versus
generated about .054. These are still severe underprediction, despite statistical improvements over P1-F.
Natural random-history reward-event means are .0337 recorded and .0395 generated: the global mean can look close
while dangerous conditional outcomes remain badly wrong.

Periphery digit transitions, read through the frozen reader, are a real improvement over previous near-always-copy
outputs. Keep UNKNOWNs in the denominator: drink appearance/increment about .53/.08 successful; wood .41/.34;
29-46% unknown. Conditional-on-readable accuracy (.74/.13/.60/.62) is not the unconditional ability. A reader that
is accurate on real tokens can fail on generated off-manifold tokens; unknown does not identify one unique cause.

S1 C2ST .946 still strongly separates generated from real percepts. Its .960 -> .946 change is descriptive
improvement, not distribution matching. The perception blocker passes. Renderer digits/icons are retained in real
tokens/decoding, while generated digits are often unreadable: these are different interfaces. Cross-model latent
MSE changes, even in the same normalized token family, are not direct semantic improvement measures.

The current-percept HUD probe is .924, last-eight-step .878, belief .787. This is limited-reader accessibility,
not a proof of information destruction, but it supplies no broad improvement in exact HUD readout through memory.
Achievements AP is .919 for belief versus .924/.921 for those controls. Do not conflate a rich reconstructable
glimpse or a high achievement mean with successful uptake of additional fine evidence.

## 3. Correction: physical current-query invariance is not full-world gaze invariance

The mask guarantees p(p,v | fixed history/action) cannot depend on the current query. Real/random-versus-guided
interface comparisons show the old large gate has mostly gone. These do not establish invariance under EVERY
fixation or every changed gaze history: past evidence legitimately changes belief, and the glimpse/outcome heads
still depend on the query.

The required pre-observation mean consistency is, for every fixed f,

    E_g[ E[r | H,a,p,v,g,f] | H,a,p,v,f ] independent of f.

Claude's new test finds violations. Reanalysis of `consistency.npz`:

- 15/256 contexts have mean expected reward >.05.
- Among those, median spread of the 49 query means is .0130; median max-minus-min is **.0566**, maximum **.2308**.
- Worst saved example: dev episode79, t14, post-DO. Mean across queries .7962; minimum query mean .6654, maximum
  .8962. Largest per-query reported MC SE .01194. Other examples range .6773-.8815 and .5293-.7311.
- These ranges are selected extrema of K=8 estimates, not a held-out learned-policy gain, and MC error must stay
  visible. Nevertheless the conditional variations are much larger than the reported SEs. The small all-context
  median mostly reflects near-zero-reward contexts; it should not dismiss rewarding-context violations.

This is a remaining actor-exploitation opportunity to measure, not evidence an actor already exploits it.
Only mean reward was tested here; matching it alone would not establish equality of the full reward distribution.
Continuation=1 in these sampled contexts gives no informative terminal consistency result.

TSP-vs-reconstruction agreement is another useful diagnostic, but the two learned renderers share a mean decoder.
They can agree while missing a detail; small average pixel MSE is not proof of semantic/per-query coherence.
The real/generated conditioning gap is correctly qualified in RESULTS: a valid different stochastic future can
differ from the recorded one. Its MSE is not by itself an isolated test of conditional robustness.

## 4. Correction: the HUD share uses inconsistent definitions

`locus.py:145` defines HUD as fixation centre row7, only 7 windows. `selection.py` and `consistency.py` use any window
touching HUD rows7-8, which includes centre row6 too: 14 windows. RESULTS compares the first count with the second
uniform reference (28.5%). Recomputed directly from the saved 600 random-frame signal maps:

| Definition | Uniform | Contingency argmax | Surprise argmax |
|---|---:|---:|---:|
| Bottom centre row7 only | 14.29% | 10.00% | 4.00% |
| Any HUD overlap (centre rows6-7) | 28.57% | **20.33%** | **21.17%** |

Episode-bootstrap SEs for the latter are 1.60 and 1.56 percentage points (see readout.json for exact values).
Row6 includes the upper HUD row but not the bottom inventory row. Report those distinctions rather than claim
that only 4% of surprising windows touch the HUD. This correction does not change consequence coverage or show
the eventual softmax controller's allocation; these are frozen-world argmax maps.

I did not overwrite historical outputs. A consistent any-HUD summary would use `(WIN.numpy() >= 63).any(1)`;
alternatively keep the bottom-row definition and use its 14.29% reference explicitly.

## 5. The useful-glimpse result is important, but does not prove inhibition is necessary

Reading5 measures how changing the REAL crop at t affects whole-map denoising loss at t+1, averaged over 8 target
noise levels and 2 common noise sets. In the factorized model the context route to physical prediction is cut,
so any current-glimpse influence reaches it through recomputed belief. The effect is small on this statistic.

The missed-mob-versus-no-mob night comparison has 59 qualified frames: difference -2.12e-5, SE2.06e-5 against mean
map loss .1157. Approximate 95% interval is about [-.053%,+.017%] of the GLOBAL map-loss mean. It does not support
an appreciable global one-step improvement there. By-day/random comparison has only28 qualified frames.

Limits matter for the design conclusion:

- A periphery-reader error plus a simulator mob label does not establish the fine glimpse supplied additional
  reliable information; the glimpse can also be ambiguous/noisy.
- Whole-map one-step denoising MSE can dilute local/rare/entity-state benefits. It does not test all longer memory,
  consequence prediction or decision value.
- This test corrupts the REAL next target at sigma .106-4.225. The proposed controller instead uses the first
  estimate from PURE NOISE at sigma_hat6.667. It does not directly measure the proposed surprise changing after
  a look. Even most of the small night effect is at the largest tested level, not uniformly across levels.
- There is no temporal scanpath/dwell experiment here. Contingency can keep attracting gaze even when a look
  lowers surprise, and true stochastic prediction error persists even with a correct belief.

Therefore an explicit short recency trace is a reasonable design bet. “Reading5 proves IOR must be explicit” is
too strong. Such a trace controls revisits; it does not make the belief use the evidence. Track overlap and camera
motion matter: inhibiting one of 49 overlapping centres does not necessarily inhibit the content just inspected.

## 6. The emergent-fovea proposal is promising, with specific qualifications

The useful idea is a model-driven curriculum over sensory evidence: use one physical forecaster to identify
action-sensitive regions and deviations from prediction, sample views with a coverage floor, let the existing
grounding/belief/dynamics objectives learn. Which places are selected can change as the learned model changes,
without a pretrained reference, simulator classes, or an extra reducible-error estimator. This fits the same-frame
periphery -> fixation -> glimpse architecture and later decision-oriented VoI.

Actual baseline evidence for a substrate is good: contingency's argmax covers drink's consequence .77 (P1-F .21),
coal/stone .97; wood/sapling .31/.32 remain weak. Surprise complements it (iron .79, wood .41), while differing
on other classes. These are localization findings, not evidence that the selected crop supplies useful new
information or improves joint learning. ReadingF covers awake DO/mining/drinking attempts; wake/death/combat
deficits have not been shown reachable by these scores.

### Do not turn the following claims into guarantees

**D is a first denoising estimate, not an exact conditional mean.** `locus.run` uses production's first call on
pure noise with two draws and averages squared maps; it does not estimate the model's entire mean distribution.
The implemented surprise includes any variability of that estimate. This can be a useful inexpensive proxy;
retain the measured definition rather than call its values calibrated causal effects or information gains.

**Within-frame competition does not eliminate noisy-TV effects.** For a fixed predictor D,

    E ||P-D||^2 = ||E[P]-D||^2 + tr Var(P).

With an independent random first-call estimate, averaging also adds its variance. Framewise softmax cancels an
exact common additive offset, not heterogeneous variance, realized noise extremes or content-dependent encoder
sensitivity. Night renderer noise was measured in this very token space and remains. Mobs can be relevant objects
of attention, so this is a risk to measure, not a verdict that the policy must fail. Twenty-percent uniform mixture
guarantees positive support; it still permits one window to receive up to .8+.2/49 = .8041 of looks. It does not
make the preferred part of the curriculum informative or impose a hard dwell bound.

**Full-frame G1 does not make the score's learning gaze-independent.** The periphery sees every frame, but its
encoder also receives TSP gradients only at selected windows (`train.p1_losses`, cells from fix). Its representation
and the belief it conditions on remain coupled to the viewing policy. This avoids H3's particular faulty scale
subtraction; it does not prove there can be no self-reinforcing representational blind spot. The existing open
admission about night content invisible in the periphery is appropriate.

**Contingency is not diminishing learnability.** An accurately learned action can continue to have a large physical
effect, and its contingency remains large. That is appropriate if this term supplies ongoing action relevance.
Only the reducible component of surprise is expected to decline with learning. The priority distribution need
not automatically move on from mastered effects; normalization/temperature/combination are real controller choices.

### Literature boundary

[Barto et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3858647/) supports distinguishing novelty from prediction
violation and discusses error-based as well as Bayesian accounts. [Itti & Baldi](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/)
supports surprise-driven gaze, but its measure is a prior-to-posterior KL, not our squared residual. It explicitly
distinguishes belief change from merely unexpected data. It does not establish this controller's noise robustness.
[NGU](https://arxiv.org/html/2002.06038) supports separate fast/slow exploration mechanisms; it explicitly uses
episodic memory, RND and a controllability representation. It is an analogy for timescales, not validation of this
single-forecaster mechanism. [Klein's review](https://pubmed.ncbi.nlm.nih.gov/10740278/) supplies an IOR precedent,
not a proof about this model. No need to add these papers' machinery merely to obtain a bio-inspired label.

## 7. Next direction

Agree a compact, explicit controller contract: score/noise draws, stable normalization and temperature, weighted
combination, random mixture, short-history/reset/overlap rules and conditioning versions. Benchmark the ACTUAL
sequential full-model path before promising an overnight runtime. Two extra calls per frame plus recurrent/cache
work are not free, even without another network.

The collection pass must start from the same incoming carry as the subsequent differentiable training pass,
without double-advancing it. Rebuild/invalidate denoiser K/V when the model updates; preserve each chosen context
glimpse. Separate controller RNG from episode/training-noise draws so P1-F1.5 remains a useful matched control.
Reconcile clean/noised chooser inputs in BOTH real and imagined steps. Keep the physical marginal query-independent
and read conditional outcome consistency again under the new policy.

Then use the full P1-F1.5 recipe plus this controller, with the fixed existing readings and gaze descriptives,
to test joint learning benefit. Keep warm-up/random exploration and attention distribution on the actual unresolved
wake/combat/HUD cohorts visible; do not infer that DO-localization fixes every failure. A successful curriculum
would improve learning from selected evidence, while memory uptake and eventual P3 decision value remain separate
empirical requirements. Nothing in this review calls for another toy-only H-series campaign or a decoder-loss
change bundled into this comparison.
