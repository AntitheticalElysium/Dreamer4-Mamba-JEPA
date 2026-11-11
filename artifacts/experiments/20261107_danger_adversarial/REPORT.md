# Adversarial audit of Claude's post-F4 synthesis

2026-11-07. User authorized CPU/GPU falsification checks. Reviewed 7ecf451a,
danger_readings.py, source game rules and the attached synthesis. One bounded
GPU diagnostic (plus its mechanical smoke), no training, no changes to Claude's
jobs or canonical code, no follow-up jobs queued. PLAN.md precedes new generation.

## Main result

The timing failure is real and operationally large. It is not established that
the memory never inferred the phase, that first-guess copying commits the
sampler, or that F4's danger/calibration regression disappears because F3 also
failed to time events. Those are separate questions.

An obvious control was missing: distinguish **information available from history**,
**exported/current belief information**, and **native calibrated prediction**.
For the energy timer we can now establish the first without another neural probe.

## 1. Energy phase is exactly recoverable from observed sleep history

The installed Craftax source initializes fatigue to 0. At every transition it
uses the final sleep state: if asleep, fatigue <- min(fatigue-1,0); otherwise
fatigue <- fatigue+1. It resets fatigue to 0 beyond 30 or below -10. No other
code writes fatigue. Sleep state is visually observable (the existing real-frame
reader reports perfect accuracy on the dev panel; our fresh reader also reads
every selected real frame correctly).

Reconstructing this counter uses sleep observations **up to the current state**,
not upcoming energy or the simulator's hidden fatigue. The resulting causal
reference predicts the next change exactly on the declared quiet cohorts:

| Cohort | Transitions | Actual changes | Reference errors |
|---|---:|---:|---:|
| asleep, energy<9, quiet | 33,324 | 2,872 increases | 0 |
| awake, energy>0, quiet, action != SLEEP | 129,311 | 3,526 decreases | 0 |

It also matches true internal fatigue on all **32 independently replayed source
states**, maximum error 0. This is an evaluation-only rule-based control, not a
proposed hard-coded model input or generic learner. Labels supply the observed
sleep flags; readability is separately checked. The observer/source check was
added exploratorily after the native timing run, with no threshold fitting.

R1's 'since last same-direction digit change' is a proxy, not the actual timer:
sleep reverses fatigue, and timers can reset without a visible digit change at
saturation. Food also resets on feeding; recovery changes direction with unmet
needs. R1 does show poor response to its history bins. It does not prove that all
timer phase is unknowable within those bins or that no timing information exists.
F4's energy-up prediction, for example, rises .016 -> .051 between early/due bins:
weak relative to truth, not literally zero discrimination.

The next belief check should use this exact observable-history reference for the
energy target. A positive readout shows availability; a failed linear probe is
still only failure of that reader on that exported representation, not proof that
the recurrent state never inferred a clock.

## 2. Native recursion confirms the sleep delay instead of inferring 3.7x

32 prospectively selected DEV episodes, one energy-8 sleeping source per episode;
health>=7, food/drink>=3, initially no arrows or hostiles within six cells. Exact
replay/render shams pass. NOOP throughout, uniform sensing, K=4 native sampler,
eight futures/source, H=48, full Mamba feedback, sampled reward/termination and
one native context commit per finished frame. The simulator uses fresh keys from
the same source states. No selection on future waking, damage or phase.

| Outcome | Simulator | F4 |
|---|---:|---:|
| reaches energy 9 by step 12 | 100% | 23.05% |
| awake by step 16 | 100% | 28.52% [21.48%,36.72%] |
| awake by step 48 | 100% | 69.14% |
| median first waking | 7.5 steps | 30.5 steps |
| restricted mean waiting, horizon48 | 7.125 | 29.898 [26.757,32.746] |

The ratio of restricted means is **4.196 [3.553,5.112]**. Because 30.86% of model
futures have not woken by 48, this is NOT an unrestricted mean sleep-duration
ratio. The quoted 3.7x was not a measured duration; this run confirms a serious
delay on a stated cohort without the iid/stationary hazard assumption.

Of 165 generated transitions with energy9/asleep followed by an active next
state, **160 wake next step (96.97%)**. The delay is mainly accumulating the last
energy unit, not failure of the already learned energy9->wake relation.

Fresh energy, health and sleep readers are 100% correct on all 9,418 real frames
of the selected episodes; source checks all pass. Whole-prefix corrected-belief
replay RMS error <=.00905 versus the stored encoding (bound .03). Raw sampled
percepts, event timelines, identities and counter states are saved in run.pt;
SUMMARY.json has source-bootstrap intervals. The figure is sleep_timing.png.

Limits: one world seed, 32 quiet high-energy starts, random fixation, no F3 native
duration arm, no actor, and H48 exceeds P3's H16. Generated-domain reader error is
not bounded by real-frame accuracy. This is P1's untrained self-feeding starting
condition, not evidence that P2 cannot repair it. It does not isolate initial
conditional error from additional own-history drift.

## 3. Lack of timer learning does not erase calibration regressions

Timing resolution, probability calibration and return usefulness are different
axes. F3 can lack an accurate clock yet supply a better average hazard. Its
stochastic sampling is part of the deployed predictive package, not a reason to
discard a measurable advantage. F4 is much better on other consequences; no
claim of overall F3 superiority follows.

Claude's 'expected Brier' uses ready/cooling group-average probabilities. This is
a score for those coarsened forecasts, not the full history-conditioned model.
We recomputed **per-history** damage Brier scores from identical saved F3/F4
zombie transition IDs (5,619 histories, four futures each), with the unbiased
finite-ensemble correction:

    k(k-1)/(m(m-1)) - 2(k/m)y + y

where k is damaging futures, m=4 and y the independently recorded damage event.
Paired uncertainty resamples physical episodes, not individual futures.

| Matched panel | F3 | F4 | F4 minus F3, 95% |
|---|---:|---:|---:|
| awake | .0958 | .1293 | +.0335 [.0271,.0403] |
| asleep | .2631 | .4182 | +.1550 [.1356,.1759] |

Lower is better. Reweighting the capped strata to their DEV population sizes
preserves the direction (overall difference +.0371 [.0292,.0448]). The published
F3-better direction survives this stronger adversarial check.

A training-only, two-rate reference (damage conditional on adjacency and
awake/asleep, no cooldown) has sleeping rate .5272 and Brier **.2505**, well below
F4's .4182. It is a cohort/state-label evaluation control, not a trained replacement
model or proof of universal observability under night noise. It establishes why
failing to know exact readiness does not justify arbitrarily low hazard estimates.
F3's asleep score is not significantly better than this simple reference; its
awake forecasts DO improve over it. Do not flatten all F3 behavior into noise.

## 4. The cache is not a memory-free route

Native context_pass receives belief.memory(bp) for every past frame. Memory
tokens are included among physical tokens; their K/V are cached. Other cached
tokens can also have integrated historical memory through earlier layers.

Thus the existing swaps separate **current memory input** from a **hybrid
observation-and-historical-memory cache**. 'Cache carries it' does not establish
that Mamba was uninvolved. A CPU structural test with trained F4 denoiser weights
holds physical inputs and current memory fixed, changes historical memory, and
changes physical outputs (max absolute1.482). This is a synthetic dependency
test, not a scientific benefit/necessity ablation of realistic histories.

The finite raw visual reach remains eight frames. Longer history can enter via
belief summaries both directly now AND through cached past memory. Upcoming
belief probes should not equate 'not linearly readable in current bp' with
'never inferred by the recurrent state'. Internal carry/export/readout differ.

## 5. A copy-like first estimate is not itself a sampler commitment

F4's first Euler move at K4 is:

    z_next = .75 * z_noise + .25 * clean_estimate

Most initial noise remains. This differs from the old three-step EDM near-snap
intuition. R3 probes noised real endpoints, not the exact intermediate states
of each sampled trajectory. It shows weak conditioning/reliance patterns, not
that the first output causally fixes all later samples.

CPU counterexample uses the exact posterior-mean flow for a two-atom distribution
with rare-mode probability .03. EVERY first clean estimate projects to .03
(copy-dominated), but accurate integration produces .0298 rare modes. At 4/8
steps it produces .0146/.02274. At probability .1 the corresponding rates are
.06538/.08407/.09923 at 4/8/128 steps. This is not F4's trained shortcut field
and does not exonerate its sampler; it disproves the inference from a first mean
alone to locked outcomes. Native K4/K8 results still show that doubling F4's
supported sampling steps is not the practical repair.

Attention mass is similarly descriptive. Spatial 'beside now' keys in R2 are
current noisy-generation tokens, not literal raw observed zombies. Evidence may
already have moved into other token values, memories and residual streams.
Different attention maps do not by themselves establish a deficient causal route.

## 6. Confidence is not a reader correctness certificate

Using a fresh F4 health reader with the same protocol, 27.81% of deliberately
channel-permuted real health tokens still receive confidence>.9 (median .747).
These invalid controls have no asserted health label: this is NOT evidence of
actual generated-token error. It establishes the narrower point that high
softmax confidence cannot rule such error out. Native outcome deficits and
cross-checks still make the functional danger problem real.

## What follows, and what does not

The useful next question is not merely 'does a probe decode a timer?'. It is:
what does an exact history reference know, what is accessible from the belief
and its exported/cache representations, and how much predictive skill does the
native world retain relative to a simple calibrated reference?

For energy we now know the history is sufficient, and the model's delayed
progress harms H16 imagination on the tested cohort. For danger, exact readiness
is not required even to improve the current miscalibration. Neither result
selects a new mixer, more data, a new loss or an attention redesign as the repair.
The revised S2 is the appropriate place to measure timing and decision effects.

No further experiments are automatically chained. Code/results: cpu.py, sleep.py,
analyse.py, CPU.json, SUMMARY.json; weights frozen and source hashes recorded.

## New belief-probe results landed during this review (b630fb13)

The positive zombie result is valuable: F4 bp predicts readiness with AP .872
and the actual awake/asleep hit with AP .700/.807, versus frame/action .471/.634
for the hit. This supports available predictive information that the native
generator underuses. It contradicts a blanket statement that no part of the
model learned readiness, while not establishing perfect inference.

The energy-up 'share .96' needs a weaker interpretation. Belief AP .1547 nearly
matches the binned-age reference .1573, but this is a weak reference: our exact
causal fatigue observer predicts every eligible energy-up event. The probe's
early-bin probability is .115 where the event rate is zero, versus .130 in its
due bin where the event rate is .197. This is partial predictive information,
not an accurately represented timer established by a near-one share. AP is not
an additive fraction of retained phase information; do not substitute another
floor-to-ceiling ratio as proof either.

Food and several other timer targets have little linear trace in current bp.
That is useful negative evidence at the exported vector, not proof that the
internal recurrent carry never encoded the phase. Keep the linear-probe caveat
operational when deciding between inference/export/use repairs, rather than
adding it after a categorical 'inference failure' headline.

Finally, 'no loss asks belief to anticipate HUD changes' is literally too strong:
G1 DOES ask for future HUD tokens through the memory conditioning path. There is
no separate belief-only HUD anticipation loss, and G2/G3 can take current-percept
shortcuts; these facts make diluted predictive pressure a plausible hypothesis.
They do not prove absence of predictive supervision or select a new auxiliary
head as the repair. The probe reads bp after the action; it is not by itself a
test of the actor's corrected pre-action belief interface.

Primary references checked:
- Ferro, fair finite-ensemble scores: https://doi.org/10.1002/qj.2270
- Jain & Wallace, attention as explanation limitations:
  https://aclanthology.org/N19-1357/
- DeVries & Taylor, confidence/OOD limitations:
  https://arxiv.org/abs/1802.04865
