# Independent audit and resumed diagnostics, October 4

## What is now supported

The terrain-memory advantage is more than a window-length correlation. The new fixed-length intervention changes only
historical token contents. The current frame, actions, time positions and checkpoint stay identical. 512 same-slot cases
span 168 episodes; 236 moved-slot cases span 35. Donors come from other held-out pool rows in matched source/target groups.
The original history, target-sighting swaps and equally sized unrelated-token swaps are evaluated on the same cases.

| world, 36k | same-slot error, original | target sightings replaced | unrelated history replaced | target − unrelated, episode-clustered 95% interval |
|---|---:|---:|---:|---:|
| attention s7 | 37.133 | 52.965 | 36.962 | +16.003 [10.651, 21.487] |
| Mamba s7 | 16.740 | 125.825 | 16.623 | +109.202 [92.734, 126.301] |
| attention s8 | 50.370 | 50.332 | 50.291 | +0.041 [−0.029, 0.131] |
| Mamba s8 | 20.053 | 101.108 | 19.979 | +81.129 [66.567, 96.026] |

The swapped contents pull Mamba's prediction towards the donor (mean projected displacement 0.278 / 0.204; pool-row
intervals [0.240,0.316] / [0.172,0.237]); unrelated-token swaps give essentially zero displacement. On moved-slot cases,
Mamba's target-minus-unrelated error changes are −0.268 [−0.635,0.011] and −0.121 [−1.015,0.523]. There is no resolved
effect in either seed. Attention s7 has a much smaller moved-slot donor pull, so "no backbone recalls any moved cell" is
too categorical; substantial reliable moved-slot retrieval is not demonstrated.

This establishes use of past cell content, strongest when the content returns to the same screen slot. It does not
measure an actor benefit or provide an architecture capacity
ceiling. Histories with swapped tokens are exploratory interventions and need not be naturally realizable trajectories.
Evidence: `sighting_results.json`, `sighting_rows.pt`, `sighting_episode_intervals.json`. The planned row bootstrap and
the additional, more conservative episode bootstrap agree. Checkpoints, script, plan and pool are hash-bound in the report.

**Storage-path follow-up completed:** 64 same-slot cases (55 episodes), 64 moved-slot cases (30 episodes), both Mamba
seeds, CPU float32 reference scan. Split-versus-unsplit maximum absolute differences are 1.67e−6 / 3.76e−6. Just before
the last input in each of the six layers, reset only the relevant screen slot's convolution carry, SSM carry, or both.
Current inputs, positions and actions are held fixed. Repeat donor swaps with each carry intervention.

| same-slot donor pull | s7 | s8 |
|---|---:|---:|
| carries intact | 0.248871 | 0.256607 |
| target SSM reset | 0.136614 | 0.097742 |
| target convolution reset | 0.074911 | 0.034068 |
| both target carries reset | 0.001349 | 0.001000 |
| both unrelated carries reset | 0.248634 | 0.256864 |

SSM-reset attenuation is +0.112 [0.050,0.176] / +0.159 [0.095,0.226]; convolution-reset attenuation +0.174
[0.107,0.246] / +0.223 [0.152,0.295]; both +0.248 [0.173,0.327] / +0.256 [0.175,0.335]. Unrelated-reset
attenuation intervals include zero. Both carries contribute to this measured history-content pathway; it is not solely
a long SSM or solely a short convolution effect. These nonlinear interventions are not an additive partition of memory
or prediction error. Other slots and spatial routing remain intact. Evidence: `carry_results.json`, `carry_rows.pt`,
`carry_attenuation_intervals.json`; original declarations preserved in `SIGHTING_PLAN.md` and `CARRY_PLAN.md`.

## Delta as an error signal: declared negative result

Completed the interrupted Experiment 0 on both M6 seeds, 1,009,764 alive map-cell predictions per world. Fixed the AUC
implementation to average tied ranks before running; a constant score correctly returns 0.5, and hand-computed pairwise
controls return 0.75 and 1.0. No old completed report was overwritten.

| error detector | s7 AUC | s8 AUC |
|---|---:|---:|
| mean Mamba Delta | 0.564267 | 0.532289 |
| final-layer Delta | 0.585018 | 0.600444 |
| input-token change | 0.471142 | 0.471620 |
| two-world disagreement | 0.641758 | 0.651290 |
| mean Delta, within input-change quintiles | 0.563717 | 0.531722 |

`delta_flags_error`, `delta_beyond_saliency`, and `delta_vs_ensemble` are all FALSE. Mean Delta has weak error discrimination
and fails the declared useful-signal thresholds. This is not a calibrated epistemic uncertainty estimate. These pooled
cell AUCs are descriptive, with strongly dependent cells; they do not establish a deployment threshold for the ensemble.
Evidence: `artifacts/eda/levers_logs/oct04_check_delta.log` (copied to `delta_results.json`).

## Corrections required for a valid long-context comparison

1. Lane73 requested 16 inputs for several evaluations although its L16 teacher loss trains predictions only at positions
   0..14. CPU differentiation of the actual A16 s7 checkpoint confirms zero gradient at row 15, nonzero at all prior rows.
   Changed those commands and comparison tags to 15. `window_proof.py/json` preserves the independent reproduction.
2. Teval lacked the `rawlong` encoder/cache mapping and would raise a KeyError. Rawlong's manifest points to Raw bridge;
   its frozen encoder equals Raw joint in all 209 tensors. The Raw evaluation cache alias reproduces all five tensors
   exactly, including context shape [1002,4,81,192]. Training-pool choice remains recorded in checkpoint args.

Completed checkpoints are unchanged. Forward, loss, sampler and optimizer are unchanged. The restarted E17/E18 jobs now
save a full resume state every 1000 updates, instead of waiting until 6000. The new CLI argument records that cadence;
the previous training source is preserved in `TRAIN_SOURCE_BEFORE_RESUME.py`. CPU comparison verifies identical teacher
loss and gradients (maximum differences 0) and no RNG change from serialization. This avoids losing another logged
1000 updates on interruption. E17 training is resumed from the declared parent where no
full state exists, and completed A16 states at update 6000 are retained. Full E16 seed-7 evaluation now uses the corrected
all-window posterior conditioning; seed-8 training is held because the first seed's ordinary-health bottleneck is already
localized, and a replication alone would not separate layout, loss dose and encoding. E19 remains a pending intervention,
not silently included in another run. Its de-alignment and selective weighting need separate arms for causal attribution.

## What the papers actually say

- [Delta-IRIS](https://arxiv.org/html/2406.19320v1), §2.2 and appendix B: reconstruction uses L1 0.1, L2 1.0 and max-pixel
  0.01 plus commitment. Removing max-pixel slightly improves average L2 (0.000185 → 0.000178) while worsening worst-pixel
  error (0.018 → 0.031). Our E16 uses uniform latent L1. This is a relevant separating loss contrast for sparse local
  consequences, not proof of its causal effect on our health token and not a licence to copy a pixel coefficient into a
  latent loss. The paper also explicitly separates decoder adequacy from stochastic prior prediction.
- [Po et al.](https://arxiv.org/html/2505.20171v1), §5.4: small temporal blocks aid memory but can hurt spatial reasoning.
  In their maze reasoning ablation, block size 1 gives SSIM 0.766 versus 0.855 for their varied-block model; retrieval
  full-context attention reaches 0.914 versus their model's 0.898. Our fmamba is a block-size-1-like variant with separate
  frame attention, not the paper's complete model. The paper supports testing spatial/temporal routing, not a claim that
  Mamba universally wins or that frame-slot state automatically tracks world coordinates.
- [Mamba](https://arxiv.org/html/2312.00752v2), §3.5: Delta generalizes a recurrent retain/update gate. It is not trained
  to estimate prediction confidence. The negative local AUC result measures that distinction directly.
- [VaGraM](https://arxiv.org/abs/2204.01464): weights model error by empirical value gradients, with precautions against
  unstable gradients and model/value dependence. E19's hand-labelled true-next-frame health mask is a targeted diagnostic,
  not VaGraM and not a verified general replacement for an actor-derived objective.
- The checked local DRAMA replay samples starts using count-dependent probabilities, not strictly uniformly. Its
  continuous-stream windows do avoid deterministically putting every death at the last supervised position. That is the
  relevant contrast with our end-aligned terminal sampler.

## Remaining scope

Health drawing has a measured time-row / scan-length artifact; the data concentration makes its training origin plausible,
but a de-alignment-only retrain is needed to isolate that origin. E16 localizes weak hit encoding even when its posterior
sees the true future, so missing stochastic prior samples cannot explain that particular failure. History-feature hit
rates are not Bayes ceilings. Root-aware and per-step trajectory heads improve decisions but remain fork-supervised
diagnostics under fixed continuations. Canonical Raw/TC have not been repaired by these experimental worlds, and no
imagination-trained LeWM actor result exists. The restored E17/E18 comparisons test context use and coordinate alignment;
they do not by themselves answer those actor and sparse-consequence questions.

**Full corrected E16 seed-7 result:** all 1002 roots, 8 prior samples, posterior codes condition every input frame. Health
readout accuracy: damage ≥2, 0/305; −1, 0/32; +1, 0/112; unchanged, 0.9995 on 15387 transitions. Sampled prior hit rate
on fresh-arrival cases 0.0000 versus empirical 0.9728; already-beside/no-recent-hit 0.0003 versus 0.4802;
recent-hit 0.0003 versus 0.0280. Declared health-drawing and hit-calibration tests fail; no-false-hits passes because it
draws virtually no hits. Delta reduces decoder error by 79.2% on HUD, 98.5% on player, 67.5% on entering cells, 92.8%
on other map cells. Those aggregate improvements do not rescue drawn health changes. They also show that the Delta
channel still carries substantial map information in this adaptation, rather than exclusively unpredictable damage.
This is a reconstruction/readout bottleneck even with the true future available, not something more prior samples can
remove. It does not prove absence of all hit information from hidden features. Evidence: `e16_fixed_result.json` and
`artifacts/eda/levers_logs/oct04_e16_fixed_s7.log`. No seed-8 stochastic world has been trained.
