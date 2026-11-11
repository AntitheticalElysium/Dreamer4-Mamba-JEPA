## Candidate and decision rule, before execution

One blind Mamba scan, sampled prefix. Start H=1, then include H=2/4/8/16 while retaining short spans.
At each elapsed step a training-only forecast reads the pre-observation latent. Targets are the existing
full-frame two-comparison soft change map, recorded reward bin and termination. No privileged labels.
For this gate define auxiliary = (mean valid change BCE + mean reward CE + mean termination BCE)/3,
means include time. Keep existing G1 content rendering; no intermediate denoiser calls. No outer weight
or training implementation selected. No bootstrap jump/distillation in this first test.

Oracle information-erasure audit, not a learned reader: compare best separate conditional forecasts vs
best pooled forecast on the pair. Score actual training reductions, with exact Jensen entropy difference.
Hold the observable targets fixed. Test fixed early fork (24) and late fork (31), dense time vs endpoint,
and previously verified achievement novelty pair (non-timer, same RGB). Negative controls: identical
target paths and a prescribed mild rendering exposure step (+1,+3,+6 RGB at elapsed step8) on an otherwise
identical future. The exposure foil is a deliberately predictable observation nuisance, not an estimate
of real render noise or a claim that lighting is always irrelevant.

Require the existing teacher to expose the paired event at the right steps, score a real outcome that
images cannot distinguish, and charge the mild rendering foil less than hunger under the same actual
reductions. If nuisance outranks the phase signal, reject this target/reduction as a launch candidate.
Passing these conditions is only an incentive check. A genuine paired visual-re-entry case, clean recurrent
fork and a learned native-uptake test are still required before claiming a general memory repair.
