# What can identify the fine-information failure?

2026-11-04. Discussion/research pass after the completed F2 review. No model,
architecture, optimizer, sensing or training change. No new GPU inference,
simulator collection, fitted probe, training experiment or queued follow-up.
One CPU-only reanalysis reads existing forecasts on existing native-eval rows.

## Answer

The four proposed explanations are not exclusive, and are target-dependent.
There is no newly identified cheap frozen-checkpoint test that reliably selects
one of them on the present evidence. Repeating probes with another width,
another fine-content swap, or another global projection analysis would mostly
repeat a completed experiment's identification limits.

The unrun, genuinely different family is **native training under a controlled
sensing condition in which a task-relevant fact must come through fine vision**.
This can establish whether the existing route learns useful evidence when the
task requires it. It is not a retrospective proof of the current recipe's cause,
and a failed bounded fit is not proof of architectural impossibility. No such
training test is silently adopted or launched here.

## Existing evidence by explanation

| Explanation | Verified evidence | What is not established |
|---|---|---|
| 1. Fine is redundant | Native coarse codebook recovers 70,208/70,208 tested HUD values; zero task-relevant coarse-reader misses in 17,518 local resource/blockage frames. Most hard terrain pairs are already distinguished by native pre-glimpse TSP. | Universal redundancy, irrelevance of fine appearance, or no benefit under limited inference/memory. One finite-observer task-relevant natural fine correction survived, with limited noise robustness and no independent confirmation. |
| 2. Fine is lost during integration | Practical conditional future-fine readout gains fall from roughly 9% at raw fine to roughly 4% at the mixed vector; finite readers do not recover the whole local-joint benefit. | Irreversible loss of the particular task fact. Retained recoding recovers some whole-crop benefit, and traced fine contrasts change both projection and full recurrent carry. Distinct activations do not prove semantic retention. |
| 3. Fine survives but is unused | Native useful-look effects are tiny/null on the evaluated panels. Some conditional fine-prediction benefit survives projection. | That the SAME task-relevant incremental distinction remains accessible to the native consumer at the point where it fails. A positive auxiliary reader is not a native-use result. |
| 4. Generator fails despite useful fine | Important generated consequences, damage and death are poor even on real-history one-step tests; real-frame fine/HUD reconstruction can be good. | Inherent generator incapacity, full predictive-state sufficiency of a presence probe, or all relevant fine information reaching the generator in usable form. Denoising, sampling, conditioning and generated-token readout remain possible contributors. |

Sources reread:
- `20261104_task_information/RESULTS.md`
- `20261104_learned_fine_candidates/RESULTS.md`
- `20261104_fine_causal_design/RESULTS.md`
- `20261103_belief_information/RESULTS.md`
- `20261103_fusion_projection/RESULTS.md`
- `20261104_p1f2_review/RESULTS.md`
- current `saccade/design/PERCEPTION.md`, `STEP_SPEC.md`, native belief,
  factorized denoiser and P1 loss implementation.

The hypothesis of insufficient learning pressure / competition overlaps 2 and 3:
when the direct coarse route solves much of the target, joint training may never
develop a useful fine-conditioned memory/consumer code. This is a plausible
mechanism, not a measured unique cause in Saccade. Fine can also help learning
or computation despite redundancy in an unlimited-information sense.

## What would repeat previous work?

| Candidate | Already covered / limit |
|---|---|
| Stronger conditional readers including whole coarse scene/history | Completed; native future-coarse conditioning was also supplied in the later fine-target audit. Predictive-family limitations remain. |
| Follow the actual cue through mixer and full Mamba carry | Completed semantic contrasts; no exact collisions, which does not prove usable semantics. |
| Compare genuine focused and disjoint looks | Completed for broad panels, arrows and learned terrain families. Redundant cues or unqualified native competence limit the null. |
| Hold gaze/coarse fixed and swap only fine content | Completed; contradictory coarse/fine evidence may also be outside natural support. |
| Recode the retained projection subspace | Completed; some whole-crop recovery, no general semantic/native rescue. |
| Increase sampling steps | Existing alternate recipes help some outcomes but do not isolate fine transport or establish an intrinsic limit. Some comparisons also change churn. |
| Add more generated samples | Can narrow Monte Carlo error; cannot repair a failed information or positive-control premise. |
| Inject a label/direction into memory or give the denoiser a new raw-fine input at test time | The native consumer was not trained on this code/route; a negative is not a usable incapacity diagnosis. |

These are methodological limits, not evidence that the architecture is healthy.

## Research read for this discussion

1. [Hewitt et al. 2021, Conditional probing](https://aclanthology.org/2021.emnlp-main.122.pdf).
   Conditions on the baseline and measures information usable by a specified
   prediction family. This supports evaluating fine *in addition to* coarse and
   history. Failure of a finite family does not establish information-theoretic
   absence. Our later conditional campaigns already use this basic construction;
   it is not a new proposal under a new name.
2. [Heimersheim & Nanda 2024, How to use and interpret activation patching](https://arxiv.org/html/2404.15255v1).
   Restoration and disruption have different implications; AND/OR routes,
   contrasts and metrics affect what a patch can identify. This supports caution
   about a small native response, not another unqualified patching ladder.
3. [Canby et al. 2025, How Reliable are Causal Probing Interventions?](https://aclanthology.org/2025.ijcnlp-long.47.pdf).
   Completeness of the intended semantic change and selectivity with respect to
   other properties both matter; interventions trade these off empirically.
   Their LLM results are not a validation of any Saccade intervention.
4. [Huang et al. 2022, Modality Competition](https://proceedings.mlr.press/v162/huang22e/huang22e.pdf).
   Proves a competition mechanism in a simplified late-fusion model and provides
   empirical examples. This is precedent for learning a subset of available
   streams, not a theorem about our Mamba/diffusion architecture.
5. [Neverova et al., ModDrop](https://arxiv.org/pdf/1501.00102).
   Uses modality-specific initialization and random dropping of channels during
   learning. Establishes a training-time method for learning across available
   channel subsets. Does not justify test-only removal of an input from a model
   trained to rely on it, or prove that dropout would fix Saccade.

No paper supplies a reliable retrospective four-way diagnosis without qualified
task and intervention controls. Do not promote an analogy into a measured cause.

## The one small additional check, and its limits

`semantic_control.py` consumes only saved ck10 cheap forecasts and existing M0.5
native futures on exact overlapping physical episode/time pairs. Existing readers,
their qualification and real-support radii are unchanged. Both saved and
confirmation head cohorts; drink/wood; both original evaluation routes; no
output-selected cases. Both original real values and previous counter values
match their labels exactly. Every output remains correct/copy/other/UNKNOWN.

On the independent confirmation overlap for guided query evaluation:

| Counter | Unique contexts | Cheap mean mapped to nearest value: correct | Cheap mean outside real-token support | Native generated fine: verified correct |
|---|---:|---:|---:|---:|
| Drink |31|27/31 (87.1%)|30/31 (96.8%)|37.1%|
| Wood |14|14/14 (100%)|13/14 (92.9%)|78.6%|

The original real-support endpoint is retained. The nearest-value point-accuracy
diagnostic was added after seeing high abstention, with no reader/radius/model
change. It shows operational predictability of those values from clean coarse
history, not valid latent samples or calibrated event probabilities. Cheap means
and native sampled futures have different objectives and conditioning noise;
this is not an apples-to-apples generative-quality benchmark. The small overlap
is not the full class distribution or a new training seed. The scientific claim
is bounded: some counter predictions can be learned without fine input; this does
not clear the native integration route or qualify a full-state generator control.

Outputs: `SEMANTIC_CONTROL.json`, `semantic_control.npz`,
`semantic_control.log`. No more fitting or threshold tuning follows this check.

## A different experiment would change the learning conditions

To test whether the whole route CAN learn to use fine evidence, one could train
the actual native system with a declared coarse blind spot on a task-relevant
cue, leaving the real fine crop and real future targets intact. Cue alternatives
must have different future semantics, and the history/direct-coarse/lighting/
outcome/gaze paths must not disclose the answer. Include a clear-coarse version
of the same predictive task as a positive control. A deterministic, already
learnable terrain/occupancy consequence is preferable to an unlearned arrow skill.

This differs from the completed frozen-path tests: the learner must acquire the
fine-conditioned function under native losses; it is not asked to use a route
that training never developed. Genuine paired cue alternatives then test native
prediction, not merely a probe on a hidden vector.

Interpretation must be predeclared:
- Native fine-conditioned success establishes useful end-to-end use under that
  sensing/training condition and rejects an absolute inability claim there.
- Failure while clear-coarse succeeds establishes a fine-path learning deficit
  at the declared task/budget, not a unique mixer-versus-Mamba attribution.
- Failure of the clear-coarse control leaves the test unqualified for route
  diagnosis. More training is not silently added to obtain a useful answer.

This is a controlled capability task / training experiment, not proof about the
original natural corpus and not automatically a production repair. It has not
been built or launched. If the user does not want to change the sensing problem,
we should acknowledge the remaining uncertainty instead of proposing another
nominally decisive frozen probe.

## What an overnight acuity change would mean

2x2 acuity is an existing proposed design trial, not a newly discovered diagnosis.
It removes more entity detail, especially at night. The original acuity EDA still
reads HUD values exactly at2x2, so it does NOT force reliance on the fovea for the
counter cases. A worse outcome would not distinguish reduced coarse evidence
from poor fine integration. A better world score alone would not establish
online fine-information use; the fine-content-dependent native prediction must
be measured as well. Keep chooser, architecture, losses and data fixed if this
trial is chosen. Do not simultaneously change fusion or Mamba and call it one
tested repair.

Recommendation: stop the frozen component-search ladder. No particular mixer or
Mamba change is justified as a diagnosed cure. Choose explicitly between testing
fine-dependent native learning under a declared sensing constraint and improving
the already measured generation deficit. A night-long design trial can be a
stated hypothesis test, but it must not be presented as certainty about the cause.
