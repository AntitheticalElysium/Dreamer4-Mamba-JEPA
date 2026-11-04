# Direct spatial chooser: not qualified

2026-11-02. Both authorized head fits and independent-cohort readings finished.
No world weights, foveal route, corpus or production recipe changed. No automatic
sparse-distillation, longer head fit or P1 run follows this negative result.

## What ran

Protocol fixed before fitting: PLAN.md. Generic 101,584-parameter two-block spatial
attention head, eight previous coarse frames and actions/outcomes, detached targets.
First 400 train episodes, monitor excluded: 107,091 fitting transitions and 16,403
validation transitions. Two seeds, 4,000 updates x 256 = 1.024M sampled transitions
each, natural frequencies, final weights. Fitting took 54.0 / 53.6 seconds.

Saved benchmark: 2,617 contexts from first 400 dev episodes. Fresh confirmation:
978 contexts from dev400:600 (128/class except iron82); no overlap with the old locus
or prior dev250 conditional-readout episode bank. Native reference refresh plus a
256-context old-bank noise repeat: 1,234 contexts from 314 episodes, 49.2 seconds.
Collection/checking time is additional; these are not end-to-end P1 timing estimates.

## Primary component retention: fails

Argmax window includes remote counter, saved benchmark, 300 contexts/class:

| Capability | Native | Head seed 1 | Head seed 2 |
|---|---:|---:|---:|
| Drink contingency | 77.3% | 3.3% | 1.3% |
| Stone contingency | 39.3% | 4.7% | 0.0% |
| Coal surprise | 78.0% | 73.0% | 72.0% |
| Iron surprise | 75.3% | 76.0% | 73.3% |

Drink/stone fail the predeclared five-point noninferiority margin in both seeds.
Coal surprise is inconclusive in both; iron passes in seed1 and is inconclusive in
seed2. These last two rows do not rescue the failed requirements.

Fresh cohort confirms the same pattern: native drink C82.8%, heads2.3%/0.8%; stone
C57.8%, heads4.7%/0%. Coal S78.1%, heads72.7%/73.4%; iron S80.5%, heads81.7%/76.8%.

## The implemented policy loses remote looks too

Exact probability of covering the remote counter under equal normalized C/S,
softmax T1, uniform floor .2:

| Class | Native | Head seed 1 | Head seed 2 |
|---|---:|---:|---:|
| Drink | 55.3% | 30.4% | 29.4% |
| Stone | 32.6% | 19.0% | 16.8% |
| Coal | 38.2% | 25.5% | 20.8% |
| Iron | 36.6% | 25.0% | 23.4% |

All four mass gates fail in both seeds. Paired episode-bootstrap 95% intervals for
candidate-minus-native, in percentage points:

| Class | Seed 1 | Seed 2 |
|---|---:|---:|
| Drink | [-28.0,-22.2] | [-29.1,-23.2] |
| Stone | [-15.7,-11.6] | [-18.2,-13.5] |
| Coal | [-14.6,-10.8] | [-19.7,-15.3] |
| Iron | [-14.1,-9.2] | [-15.7,-10.9] |

Fresh-cohort native versus head1/head2 mass: drink54.9->29.2/29.4%; stone39.1->21.6/19.1%;
coal39.0->24.5/21.0%; iron43.0->26.5/24.8%. All eight fresh mass gates fail too.
Paired 32-draw sampled coverage agrees; exact mass is the primary quantity.

It is not a total loss of remote localization: surprise finds drink counters in
64.7/65.0% (native44.3%) and stone counters in51.3/52.3% (native33.3%). But contingency
mostly selects contact locations instead: C contact coverage stone87.3/95.0%,
coal96.7/100%, iron82.0/88.7%. The unchanged mixed policy still loses the remote looks.
Do not silently retune the mixture after seeing this; that would be a new candidate.

## General forecast learning succeeds; selection does not

Saved random contexts: MSE .03681/.03755 versus copy-previous .12682, a70.4–71.0%
reduction. Fresh random contexts: .03124/.03175 versus .11524, a72.4–72.9% reduction.
The last logged validation subset (first4,096 of the16,403 validation rows) similarly
improves. No checkpoint was selected using that subset or any development result.
At the four consequence classes, factual MSE improves by25–34% versus copying.

Thus low ordinary forecast error is demonstrably insufficient for this chooser's
adoption. These comparisons do not establish that the head predicts better than the
native stochastic world; the reference here is copying the previous frame.

## Checks and limits

* CPU checks pass: no target-frame leakage, reset padding, exact NOOP C=0, exact
  policy match to the previous implementation, head-only gradients. The initial CPU
  check hit an existing module's CUDA allocation at import; its unchanged policy
  function is now compared without importing those unrelated GPU allocations.
* New coarse collection matches previous native encode exactly (max absolute error0);
  action alignment is exact. Native actual-action belief replay worst relative error
  .00925, below the existing .02 tolerance. Frozen world SHA remains
  95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128.
* Comparison instrument passes identical-reference scores and rejects a uniform
  replacement on all eight primary gates.
* Native independent-noise repeat,32/class: drink C75->75%, stone C59.4->59.4%,
  coal S87.5->84.4%, iron S78.1->75%. It does not resemble the heads' catastrophic
  drink/stone C loss. Full noisy-reference intervals remain in results.json.
* Natural fitting data contains1,716 drink,4,040 stone,462 coal and269 iron attempts
  under the evaluation definition. This is not zero example coverage. However,
  epsilon1 episodes contribute19 drink and zero tool-qualified stone/coal/iron
  attempts. Those selected mining counterfactuals have no pure-random support here.
* The head has less memory, data exposure and capacity than the native world and uses
  mean contrasts rather than common-noise denoiser contrasts. This bounded comparison
  does not identify which difference caused failure, or rule out all direct-head designs.
* Early-checkpoint adaptation, full own-history/night/noise checks and production
  pipeline qualification were not extended after clear primary failure. They are
  mandatory before positive adoption, not evidence manufactured for this failed head.
* Sparse endogenous distillation remains untested. The foveal information route remains
  a separate question; this experiment has not changed or diagnosed it.

## Evidence

`results.json`, `reference.json`, `fit_*.json`, head checkpoints, per-context
`predictions_*.npz`, `native_confirmation_and_noise.npz`, `training_coverage.json`,
`checks.json`, `comparison_instrument.json`, `audit.json` and logs preserve the readings.
Native GPU work completed and released the device before the next authorized test.
