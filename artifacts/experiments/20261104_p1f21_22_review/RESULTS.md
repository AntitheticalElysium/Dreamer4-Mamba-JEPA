# Review of the two prepared F2 arms and the train/dev interpretation

2026-11-04. User requested review, not launch or architecture repair. Snapshot
review while Claude builds union targets and runs the arm preflights. No shared
source or arm source was edited; no GPU job launched, stopped or altered by this
review. Source hashes in CPU_CHECKS.json identify the inspected implementation.

## Verdict

The diversity arm is a coherent fixed-compute dataset intervention, with good
source/split preservation. Its target build, full union validation and GPU
preflight were still pending when inspected. The 2x2 arm's real-path preflight
has passed, but its initialisation is not paired with F2, its source seal omits
new shared dependencies, and its scientific readouts need narrower interpretation.
Both workers currently stop at selection; the evaluation queue is not built.
Neither package is an unconditional launch approval from this review.

**Subsequent user decision:** diversity arm approved, 2x2 deferred. The selected
arm's preparation, evaluation queue and provenance fixes are recorded in
`../20261104_p1f21/CHECKS.md`; the initial findings below are preserved as the
pre-fix review. This does not qualify the deferred 2x2 arm's initialization.

## Verified mechanics

- Config comparison: diversity differs from the F2 recipe only in data.corpus
  and p1.out_dir (new targets); acuity differs only in periphery.acuity, 3->2.
- Independent CPU initialisation: diversity has all226 state tensors and the
  post-initialisation Torch RNG identical to the F2 recipe.
- Union manifest: all1,060 shard paths/order/declared hashes match their three
  source manifests, whose hashes also match. Original420 shards remain first;
  old dev/final (shard,slot) meanings are preserved. This check does not replace
  hashing all113GB of shard bytes: the arm's union.verify is responsible for it.
- union.use keeps original monitor episodes and moves new monitor-hash episodes
  to a separate held_out list. The code excludes them from training and keeps
  the original monitor/selection rule. Old train is intended as a prefix;
  targets append only new actual-train episodes in the same order.
- Change-target code uses the frozen original teacher, checks12 original
  episodes bitwise, copies the original block, checks part ranges/shapes, and
  hashes final targets. Runtime completion remains a required gate.
- Reward bin decode means are recomputed from union train rewards. This is a
  declared consequence of changing data, not a new objective. Old dev/final and
  new monitors do not enter this calibration.
- Chooser code, optimizer, budget, phase boundaries and policy are reused, not
  redesigned. Data naturally changes the chooser's training distribution.
- Independently reran CPU suite:57 passed,6 skipped (CUDA disabled deliberately
  to avoid competing with Claude). Log:pytest_cpu.log.
- Claude's completed2x2 checks: exact model/streams/carry/cache/RNG/chooser/gaze
  resume; optimizer difference2.84e-14; real launcher smoke;200 updates at64
  steps plus3 at256; finite monitor; chooser and world coarse paths exactly
  equal on256 frames; peak4.294GiB. This review inspected those artifacts, not
  reran their GPU workload. Sustained early speed1.83s/update and256-step
  smoke6.91s/update are not a whole-run ETA.

## Findings before launch

### 1. Pair the 2x2 shared initialisation, or withdraw the matched claim

train_p1f22.recipe changes the input matrix from256x27 to256x12. Its smaller
initialisation draw shifts all following random draws. CPU_CHECKS.json measures
149 changed equal-shape state tensors and11,106,248 changed values out of
14,453,236 equal-shape values, including the fine encoder and later modules.
Post-initialisation RNG also differs, shifting subsequent training noises.
The plan acknowledges this; it is not hidden. But it is avoidable and materially
weakens a one-world-seed ablation aimed at acuity. Preserve common-shape baseline
weights and the baseline post-initialisation RNG, with the resized input weight
initialised under an isolated declared stream. Check the actual trainer path,
not only a stand-alone model constructor; the old3x3 control should reproduce
F2's nonzero-update behaviour. Rerun affected smokes and seal after the change.
If deliberately left, it must be described as an extra uncalibrated seed source,
not a strictly paired sensing-only result.

### 2. Seal the imported shared arm files

The generated2x2 SOURCE_PINS.json has101 files and ZERO paths from
20261104_p1f21. But2x2 run.py imports that folder's run.py; its checks import
checks.py; and its pinning command imports pin_sources.py. Those files can
change without invalidating the2x2 seal. Add all three as explicit dependency
inputs and re-seal after review/preflight. Avoid copying the implementations:
shared helpers are reasonable; their provenance must be closed.

### 3. Complete and smoke automatic evaluation before overnight launch

Both arm run.py paths train, select, mark phase=trained, and return. Unlike F2's
worker, neither dispatches an evaluation queue. Plans explicitly defer building
it until training. That is not the previously agreed train->evaluation contract.
Prepare the same F2 comparison battery, output isolation, selected-checkpoint
routing and failure status before launch. For diversity, complete and review
new held-out label assembly now if it is part of the scientific question; an
unbuilt future reading cannot qualify the run beforehand. New-source labels
must use their diagnostic sidecars or correct source-specific keys, never the
old replay key construction on concatenated shard indices. Old label references
remain unchanged. 2x2 readers must be fitted to that world's representations,
not reuse the3x3 latent reader as if the coordinate systems matched.

### 4. Tighten what the 2x2 experiment can identify

The acuity EDA measures achieved classifier recall. Skeleton recall.53 and
arrow.42 are not proof the fact is absent from every coarse/history observer;
HUD remains readable. It is a sensing/representation design trial, not an
already qualified fine-required task for every measured class.

The old58-scene uptake panel is useful as an unchanged descriptive reference,
but its prior3x3 reader-miss selection did not establish incremental fine value
over the full coarse/history/prior. For the new model, report separately whether
(a) the fine crop actually reveals the cue, (b) the coarse/history baseline misses
it, (c) the policy acquires it, and (d) later native predictions use it. Keep the
fixed broad panel; do not quietly replace it by output-selected successes.

Own-vs-random whole-history gaze can validly measure the operational acquisition
policy, but greater reward/death sensitivity is not itself proof of fine use:
it mixes past evidence with current query choice. Holding past belief and
physical action fixed, changing the current fixation should not change the
marginal distribution of physical rewards/deaths. A useful REAL past glimpse
can legitimately change the belief and hence later forecasts. Keep those two
contracts distinct; an emergent current-query shortcut must not count as a win.

Predeclare possible failed prerequisites: absent fine cue, policy failing to look,
and lack of native competence even on clear-coarse controls. A null is a failure
of this2x2+chooser recipe at its budget, not proof the foveal route can never
integrate evidence. A positive needs fine-content-dependent later prediction,
not merely more predicted positive events. Do not promote the old projection
probe deficit to a proven irreversible bottleneck.

### 5. Remove the unsupported exclusion in the diversity prior

PLAN.md says a gain on coal/iron/death/damage would contradict the train/dev
reading, and excludes those four from the diversity hypothesis. It would not
contradict any observed result. Current F2 can fail on seen contexts and another
data distribution can help learn the shared relation. No-large-train-advantage
is evidence against ordinary memorisation/generalisation failure, not a proof
that distinct data cannot help. A negative prior about benefit is reasonable;
an impossibility/exclusion claim is not. Report all four without assuming away
success. Raw union remains a legitimate diversity-at-fixed-compute experiment;
it is not increased exposure on every failure and not a fine-use experiment.

## What the train/dev diagnosis actually shows

Verified raw summary on one F2 checkpoint, random-gaze instrument:

| Outcome | Train generated | Dev generated | Recorded train / dev |
|---|---:|---:|---:|
| First coal reward |22.41%|24.37%|100 /100%|
| First iron reward |7.23%|9.18%|100 /100%|
| Risk-class death |1.00%|1.27%|12.02 /12.77%|
| Adjacent-class damage |1.05%|1.23%|14.43 /15.90%|

This is a generation deficit on seen physical histories as well as dev. It
weakens the story that a competent training fit simply fails on new episodes.
It does not locate the failure among predictive state, denoising, sampling and
outcome reading; or establish low/high training-objective fit. The native P1
objective is denoising plus auxiliaries, whereas this reading is event mass
from sampled futures. Real-successor heads doing well is recognition, not proof
of an adequate preceding predictive belief.

Important limits:
- Training histories are re-encoded with random gazes, not the exact saved
  training crops. Dev own-gaze invariance makes a huge effect less plausible,
  but does not prove the same for the TRAIN cohort.
- Three primaries are not exact recollection losses; their targets are
  stochastic/cohort-dependent probabilities. No individual-event100% criterion
  is imposed for risk/damage. Coal/iron cohorts are selected successful events.
- Closure>=1/3 is an operational predeclared large-train-advantage rule, not a
  universal definition of memorisation or proof of architectural incapacity.
- compare() propagates generated train-dev SE but not full joint uncertainty of
  generated and recorded ratios and the dev denominator. A conservative local
  first-order sensitivity check (not a replacement bootstrap interval) keeps
  the four upper values <.063, well below1/3. This statistical simplification
  does not drive the large absence of a train advantage; do not overstate the
  reported closure intervals as a complete ratio-bootstrap analysis.
- Drink/furnace show train advantage and some rare real-successor heads have
  train/dev gaps. Different classes can have different limitations.

Underexposure is also a learning-pressure problem: ten corpus passes do not
make a rare one-cell consequence an equally weighted lesson. Missing/rare
state-action contexts can still matter. The evidence shifts priority toward
learning/generation rather than plain overfitting, but cannot rule out data
benefits or select one architectural repair.

## Literature cross-check

- Hewitt et al.,2021, Conditional probing:
  https://aclanthology.org/2021.emnlp-main.122/
  Measures usable information beyond an explicit baseline; standalone reader
  failure must not be promoted to proof of universal information absence.
- JEDI,2026:
  https://arxiv.org/html/2605.13013v1
  Joint latent learning is driven by a denoising objective; sampling is a
  separate iterative operation (3 Euler steps in its table). This motivates
  keeping denoising fit separate from sampled-semantic quality. It neither
  diagnoses our model nor proves either new arm will succeed.

## Scope and next action

Do not rerun an endless probe ladder. Correct the bounded preparation issues,
finish the already declared gates/evaluation, and discuss which hypothesis the
user wants to spend the overnight budget on. No arm is launched by this review.
