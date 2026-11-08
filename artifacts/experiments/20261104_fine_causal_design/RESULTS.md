# Consequence-specific fine-evidence test — completed 2026-11-04

**Finding:** the tested fine cues change the projected input and full Mamba state,
but do not produce a useful change in the measured next-step consequences.
There is also a severe native generation/readout failure when arrow PRESENCE is
clear in the direct coarse context. Presence is not the complete predictive
state: heading, for example, was not qualified in the learned coarse tokens.
This control therefore does NOT exonerate the fine route or rule out missing
fine detail as a cause. This campaign does **not** uniquely distinguish an unusable
recoding, acquisition/training failure in Mamba, denoiser utilization, or sampling.

Frozen F1.6 checkpoint10, SHA256
`34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045`.
F2 scientific results were not inspected. No model/probe fitting or architecture
change. The GPU is released; the pre-existing F2 CPU evaluation was undisturbed.

## One joined experiment and its controls

```
actual cue frame, fixed real prefix, action, random draws
      │
      ├─ focused versus disjoint genuine look
      │        coarse/light identical; gaze and fine tokens change
      │
      └─ same-gaze fine-content control
               coarse/light/history/gaze fixed; fine cue present versus absent
                      │
             fusion → projection → FULL Mamba carry → exported belief
                      │
             native next physical prediction → destination / health reading

physical cache equality checked: raw fine cache cannot explain physical effects
```

67 source-qualified cases,64 shared random futures per condition,4 conditions,
two fixed sampler recipes (production3/churn1; secondary8/churn0). The extra
same-gaze content control uses12 cases and the same budgets:40,448 one-step
futures altogether. All histories before prediction are real. This is not an
untrained multi-step self-feeding task.

`PLAN.md` records the source-driven fallback and the gaze-mask confound amendment
before inspecting native consequence numbers. The endpoints of the semantic
counterfactuals are valid simulator renders. Heading edits/removals are not
claimed to have an identical feasible past; crossed fine/coarse combinations
are internal interventions, not guaranteed natural sensor samples. The genuine
gaze comparison uses consistent real inputs. No generated outcome selected cases.

## 1. Why the first candidate was insufficient

24 adjacent-arrow heading contrasts,16 dark and8 daylight. Same NOOP and simulator
key; changing toward to away changes next health by exactly2 in all24. Selected
source states re-render bit for bit. However:

- All24 heading contrasts ALSO change the3x3 coarse input. None gives an exact
  coarse-only alias in the learned tokens either.
- The existing reconstruction decoder chooses the correct heading alternative
  only27/48 times. This does not prove the encoder lacks direction, but fails to
  establish a reliable fine-direction acquisition control using this decoder.
- Native generated health barely distinguishes headings: focused expected-health
  contrast0.002–0.006 across light/sampler groups, against the true2.

So a null here cannot localize a missing fine-information route. Preserve the
negative control; do not call it a pure memory test.

## 2. Qualification exposed redundancy and instrument errors

A bounded survey of nearby night threats in recorded damage transitions finds
the coarse reader correct on74/75 arrows and256/256 zombies; both fine readers
are correct on all331 factual source frames. These are selected source counts,
not corpus-wide accuracy estimates. Nearby threat presence is mostly redundant
with coarse perception in this cohort. Of25 selected nearby cases,6 fail the
simulator-removal test: the selected arrow was not causing the recorded damage.
They are excluded and retained in `causal_qualification.json`.

The fallback is a distant night arrow whose next move brings it closer. Both
fine readers must recognize it; the coarse destination reader must recognize
the real successor. Removing the arrow under the SAME action/key must remove
destination occupancy. Select12 coarse-clear and12 coarse-missed episodes.

**The paired source control matters:** of the12 coarse-missed cases, both fine
readers distinguish actual presence from rendered absence in only5. In the
other7 at least one reader still calls the removed arrow present. Those7 cannot
establish semantic fine acquisition with these instruments and remain reported
separately. All12 coarse-clear controls pass both fine readers on both alternatives.
No new probe was fitted to rescue qualification.

The5 stronger cases are episodes8/t216,35/t189,36/t206,40/t218,41/t513. All begin
a new visibility interval. Only3 have no arrow visibility in the preceding8
frames; the other2 are re-entries. `first_visible` in raw metadata means absent
in the immediately preceding frame, not first-ever sighting. See
`prior_visibility.json`; do not overstate independence from prior evidence.

## 3. The stronger fine-cue test does not show useful native generation

Five episodes;320 generated futures per factual gaze/sampler. On matched REAL
successors the frozen physical reader detects the destination arrow in5/5
present cases and0/5 removed cases.

| Same actual source, genuine look | 3-step destination detections | 8-step destination detections |
|---|---:|---:|
| Look directly at the arrow | 0/320 | 1/320 |
| Look at disjoint same-row crop | 0/320 | 1/320 |

The same-gaze control holds coarse inputs/history/coordinates fixed and changes
only fine content, under BOTH coarse contexts. In each sampler there are640
matched comparisons across the5 stronger cases: **zero changed destination
classifications**. There is weak continuous sensitivity: under the factual
coarse context, the8-step reader-score increase is0.0000371,95% episode-bootstrap
interval[-0.00000590,+0.0000819]. Under absent coarse context the estimate is
-0.0000294, interval[-0.000108,+0.0000234]. A useful correct semantic response is
not demonstrated. Reader scores are not calibrated world-event probabilities.

Zero observed classification differences is a finite-sample result. The
degenerate[0,0] empirical bootstrap intervals in raw output are NOT population
proofs of exact zero influence. Likewise a tiny score delta is not a repaired
world-model capability. Full12-case results and all source failures are retained
in `CONTENT_RESULTS.json` and `RESULTS.json`.

## 4. Coarse-presence-readable controls also fail; this does not clear the fine route

In the12 controls where the source arrow IS clear in the periphery, the same
destination reader recognizes all real successors and rejects all removed-arrow
successors. Yet focused native generation gives only8/768 destination detections
(1.04%) with3 steps and26/768(3.39%) with8. Looking elsewhere gives the same8/768
and25/768. Recent coarse evidence enters the denoiser directly, so access to
coarse arrow PRESENCE does not depend solely on fine projection or Mamba
transport. However, the presence reader does not establish that these tokens
provide heading or every fact needed to predict the destination. The earlier
interpretation that this control excludes a fine-route-only cause was too strong.
Real-successor recognition is an instrument check, not a proof that the model's
conditional predictive probability of that exact destination should be100%.

Native damage heads also show a functional deficit beyond this arrow reader:

| Causally verified natural task | Negative-reward probability on real successor | On generated successor,3 steps | On generated successor,8 steps |
|---|---:|---:|---:|
| Arrow causes damage,7 cases | 99.83% | 0.223% | 4.41% |
| Zombie causes damage,12 cases | 99.94% | 9.46% | 21.33% |

All matched real-health readings are correct and within their saved support.
For zombies the generated present-versus-removed expected-health contrast is
0.185/0.366 with3/8 steps, versus true2. Focus changes that contrast by only
0.00260/0.00130, intervals including zero. Arrows give0.00446/0.0536 versus true2.

These are failures at the generated interface. This test cannot fully separate
missing semantic events from off-manifold generated tokens misread by the fixed
reader. The native reward-head deficit means the concern is not solely that
external arrow probe, but does not identify one generator component as the cause.
Eight steps help some readings but are far from fixing them; neither recipe
exhausts possible samplers.

## 5. No literal input or memory-state collision was found

Across all67 semantic pairs, the controlled fine-only input changes the actual
projected embedding, post-action emission, and complete conv/SSM carry. There
are zero exact embedding, full-carry or emitted-state collisions. The physical
cache equality assertions pass for every genuine-look and content comparison.
Maximum full-prefix replay emission RMS against the saved source is0.01153
(predeclared bound0.03). Raw inputs and complete full-route carry tensors are
saved in `activations/`; the content control saves state differences.

In the5 stronger same-gaze cases, mean fine-content-induced embedding RMS is
0.00178 and post-action-emission RMS0.00507. These are DIFFERENT coordinate spaces;
the ratio is not an information-retention or amplification estimate. FP64 linear
accumulation in the main diagnostic uses a native-rounded joint difference,
not a full FP64 re-execution of the nonlinear encoder.

The projection is not mapping these particular cues to one identical vector.
That does not establish accessible semantic retention after projection or prove
Mamba knows which consequence the difference means. **Distinct states are not
the same thing as a usable distinction.** No finite-reader null is relabelled
information-theoretic erasure.

## Disposition

This confirms poor consequence-specific uptake/output for the tested F1.6 cues,
not complete disconnection of the fovea. It also shows why merely looking at a
nearby threat was a weak diagnostic: coarse redundancy, some wrong causal
examples, and imperfect paired reader specificity. The stronger fallback still
has no useful demonstrated benefit on the measured destination, while
coarse-presence-readable controls also fail badly. This does not establish that
fine content is unused everywhere, nor clear the mixer of losing accessible
heading/other semantic detail. Do not uniquely attribute the whole result to
the mixer OR to the denoiser/sampler. Stop for discussion; no followup job or
training change queued.

Interpretation corrected2026-11-04 after the user's follow-up. Raw predictions,
case selection, metrics and experiment code are unchanged.

## Evidence and reproduction

- `qualification.json`, `survey.json`, `survey_expanded.json`,
  `causal_qualification.json`: all source candidates and rejection reasons.
- `native_results.pt`, `activations/`, `stage_differences.json`: full native runs.
- `RESULTS.json`, `CONTENT_RESULTS.json`: per-case readings and episode intervals.
- `SOURCE_PINS_pre_content.json`, `SOURCE_PINS.json`, `SIMULATOR_PINS.json`:
  source/checkpoint/instrument lineage, with protocol amendment retained. Model,
  trainer, data code, reader and checkpoint hashes verified unchanged.
- CPU qualification: `python -m artifacts.experiments.20261104_fine_causal_design.qualify`;
  GPU survey: `...test_native survey`; CPU replay: `...test_native sources`;
  GPU smoke/run: `...test_native run [--smoke]`; content smoke/run:
  `[SACCADE_CONTENT_SMOKE=1] python -m ...content_control`;
  CPU reports: `...analyze`, `...analyze_content`.

Methodological scope follows [Zhang & Nanda,ICLR2024](https://arxiv.org/html/2309.16042v2)
and [Heimersheim & Nanda,2024](https://arxiv.org/html/2404.15255v1): semantic
contrasts, metrics, redundant paths and intervention distribution determine which
mechanistic claim a patching result supports. These papers do not validate Saccade.
