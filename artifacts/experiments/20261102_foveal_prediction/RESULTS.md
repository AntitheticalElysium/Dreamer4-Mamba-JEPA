# F1: fine evidence predicts future fine detail; exported access is weak

2026-11-02. Authorized predictive-value stage complete. No native world parameter,
architecture, route, sampler or corpus changed. No causal routing continuation launched.

## What ran

PLAN.md fixed before fitting. Five supplements x two seeds, with independent coarse
and fine prediction networks (20 predictors total). Each target branch is MLP256-256;
2,400 AdamW updates, batch512, same frame draws/init per seed across arms. No semantic
labels in optimization. Targets: actual next coarse percept, plus actual next fine
tokens at the CURRENT crop position. The latter is a forecast from history, not a
conditional glimpse forecast provided with the next coarse frame.

Base sees full current coarse grid/light, prior belief before the current observation,
eight-step physical history with prior beliefs/actions/outcomes, current arrived
reward/termination, next physical action and fixation/mask. Whole-crop fine supplement,
native fine input projection, post-action emission and memory supplement all use the
same frame and metadata; all padded1024. The memory supplement is the actual four-token
projection seen by the native denoiser, not an invented query into hidden carry.

Train first400 episodes; fit0:360/validate360:400, with world-monitor episodes excluded.
All128,564 train transition targets collected, terminal successors included; fitting
uses the declared subset. Read130,261 dev transitions: first250 episodes diagnosis,
next150 disjoint confirmation. These remain development cohorts, not final evaluation.

Primary stratum: night, current crop contains an entity, no camera scroll on the next
transition. Diagnosis2,628 frames/164 episodes; confirmation1,824/102. True mob motion
remains stochastic. Source cells are CURRENT positions, not oracle future locations.

Fit/evaluation process:281.2s, peak3.96GiB. Collection and CPU analysis additional.
Frozen ck10 SHA95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128.
Mechanical checks passed: full9x64 crop supplied, future targets excluded from inputs,
coarse copy expressible, nonzero fine-input gradients. Source current coarse data exactly
matches the independent native coarse cache. Readouts and errors are archived in float32.

## 1. Raw fine evidence is useful for predicting future fine content

Percentage MSE reduction versus matched base predictor, primary population. Positive
means improvement. Full-crop fine target:

| Supplement | Diagnosis seed1 / seed2 | Confirmation seed1 / seed2 |
|---|---:|---:|
| Whole raw current fine crop | 40.5 / 43.9% | 38.9 / 42.2% |
| Native mixed fine contribution | 6.0 / 6.0% | 4.6 / 5.2% |
| Post-action emission | 0.1 / -3.9% | -0.3 / -4.9% |
| Four projected memory tokens | -0.1 / 2.9% | -0.8 / 1.2% |
| Copy current fine tokens | 61.7 / 62.5% | 61.0 / 61.7% |

Raw-fine improvement has positive episode-bootstrap95% intervals in all four readings;
diagnosis38.0–42.5/41.5–45.8%, confirmation36.5–41.0/39.8–44.3%. Copying current fine
is a diagnostic reference which also has current fine evidence, not a coarse-only arm.
Its strong result shows the bounded learned instrument is not extracting all available
temporal predictability. It is not a proposed world-model sampler.

This advances beyond current-frame classification: the observed fine content is useful
for forecasting real future targets. It is not merely present information with no temporal
predictive value. It does not establish that all this advantage concerns entity identity.

## 2. Local check: not just background inside an entity-containing crop

An additional source mask was declared DURING fitting, before development predictions
were read: current entity cells inside the crop, and their one-ring neighborhoods. It is
secondary, labelled as such, and does not retrospectively replace the primary reading.
It uses current source positions only and identical target cells across arms.

Future fine error at the entity's CURRENT cell:

| Supplement | Diagnosis seed1 / seed2 | Confirmation seed1 / seed2 |
|---|---:|---:|
| Raw fine | 26.3 / 26.9% | 26.4 / 25.9% |
| Mixed fine contribution | 3.4 / 2.8% | 2.4 / 1.8% |
| Post-action emission | 1.6 / -2.7% | 0.6 / -3.2% |
| Projected memory | 0.1 / 0.8% | -0.8 / 0.1% |

Raw-fine intervals: diagnosis[24.4,28.2]/[24.7,29.0]%; confirmation[24.5,28.2]/
[23.2,28.2]%. Raw fine also beats the memory-supplement predictor by25.8–27.0% on
this reading, with positive paired intervals in all four cases.

At the current entity's one-ring neighborhood, raw-fine gains33.7–37.9%, mixed4.1–5.8%,
and projected-memory effects range about0–2.4%. Source-only and neighborhood readings
reduce the background explanation but remain latent prediction errors, not simulator
identity accuracy. A fine token's receptive field can also contain neighboring content.

## 3. The future coarse target gives a different result

Next coarse MSE at the CURRENT crop, same primary population:

| Supplement | Diagnosis seed1 / seed2 | Confirmation seed1 / seed2 |
|---|---:|---:|
| Raw fine | -0.05 / 0.18% | -0.07 / 1.47% |
| Mixed fine contribution | -0.63 / 1.79% | -0.60 / 1.83% |
| Post-action emission | 3.52 / 6.84% | 2.75 / 6.25% |
| Projected memory | 4.64 / 5.45% | 4.10 / 5.59% |

Raw fine has no reproducible coarse-crop advantage across both instrument seeds: seed1
intervals include zero, seed2 confirmation is positive. Current-source coarse-cell gains
are likewise small, -0.16–0.70%, while native emission/memory provide roughly1–3%.
Do not call that exactly zero or claim that the physical target cannot use fine evidence.

The coarse predictors are also only slightly better than copy-current on this no-scroll
primary stratum (one baseline seed is approximately tied). Compression/architecture and
optimization limits of this instrument remain material. The exported supplement contains
coarse/history information too, so its benefit does NOT identify fine-specific uptake.

## 4. Warranted conclusions

* Current fine observations have demonstrated future predictive value, including at
  entity source cells, beyond the declared coarse/history/prior/action baseline.
* That value is substantially easier to exploit directly than through the native mixed
  fine contribution or exported memory in this matched instrument family.
* Native memory is not a universally useless/dead channel: it improves coarse prediction.
* This strengthens an accessibility-gap diagnosis; it does not establish irreversible
  erasure or uniquely blame the input projection, Mamba retention or output readout.
  Feature scale/geometry, capacity and finite training can affect accessibility.
* There is not yet a robust raw-fine positive control on the PHYSICAL coarse target in
  both seeds. This cannot alone qualify a physical-route repair.
* The future fine target here does NOT receive future coarse information, whereas the
  native cascade's conditional fine prediction does. How much additional fine-history
  value remains under that native conditioning is unresolved. Do not import this gain
  as a guaranteed reduction of the native G1 objective.
* MSE concerns conditional means. It does not establish equality/inequality of all
  predictive moments, coherent stochastic sampling, actor benefit or P2 robustness.
* No learned-fixation or privileged-route design follows automatically. Return for
  discussion; the separate chooser campaign also failed its adoption prerequisites.

## Evidence

PLAN.md, predict.py, analyze.py, checks.json, population.json, fit.json, identity.json,
per-cell errors_*.pt, dev_metadata.pt, results.json and logs. Raw future targets preserve
terminal successors and actual screen query positions. All native weights remain frozen.
