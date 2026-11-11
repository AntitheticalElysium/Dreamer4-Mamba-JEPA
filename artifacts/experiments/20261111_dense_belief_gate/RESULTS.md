# Dense teacher-belief MSE gate

Frozen experiment only. No change detector, training or changes to canonical/original arm code.
Replay reproduces the exact-history hunger bank before extending the same NOOP actions and RNG.
23 backgrounds survive through the common 16-step window, episode84 excluded for termination.
The last eight input tokens per world step match at the fork. Food changes at elapsed8/16;
the final images agree. Original MultiStep checkpoints1/5/10 include the trained mask parameter.
Each student uses one 16-step scan. Teacher scans real observations uninterrupted, avoiding
the artificial reset in the original trainer. Compare post-mask to post-percept, as in the old loss.

For the teacher, the oracle gap when forecasts are tied is mean squared difference/4, per step.
For the actual student, average its two forecasts and rescore, holding the teacher fixed. This
arithmetic erasure does not estimate the performance of a trained no-memory model. Check
the exact decomposition into pair-mean and contrast error, and oracle control. Identical
observation/history sham gives zero. Score averaged over both time and latent coordinates.

Checkpoint10, same fork:
| H | Mean across steps, oracle gap | Only endpoint, oracle gap | Mean across steps, actual gap |
|---|---:|---:|---:|
| 1 | .00003825 | .00003825 | -.00000940 |
| 2 | .00003520 | .00003215 | -.00003334 |
| 4 | .00003182 | .00002723 | -.00004745 |
| 8 | .00046117 | .00350053 | .00002240 |
| 16 | .00305938 | .00136338 | .00034592 |

At16, dense oracle gap 2.244x endpoint; actual 1.777x. Endpoint teacher still differs even when
images agree (teacher has seen the different paths). Dense actual student error .184136,
tied .184482: only .188% of the tied loss. Conditional scene-bootstrap95% for actual MSE gap
[.000296,.000392], not retraining uncertainty. Even the full teacher contrast at fixed pair
mean is only1.66% of the tied loss. No claim these fractions transfer to other checkpoints,
cases or to the joint training gradient. H8 endpoint is stronger than the dense16 oracle;
adding earlier steps to a mean can also dilute it. Dense16 oracle gap increases across snapshots
.000416/.001899/.003059, not a causal claim about the original auxiliary.

Equalizing all future teacher inputs (same two histories) drops the dense teacher gap to
.00001931 at ck10. Most of the desired distinction enters when the teacher sees the differing
observations. Useful predictive signal is present in targets; simply matching them cheaply
does not require the student to forecast the right hidden phase.

Brightness foil on the same physical future, same physical history/outcomes:
| Step exposure offset | Mean H16 teacher gap | Worst ratio to matched hunger oracle gap | Outranks hunger |
|---|---:|---:|---:|
| +0 | 0 | 0 | 0/23 |
| +1/255 | .00007026 | .049 | 0/23 |
| +3/255 | .00028765 | .222 | 0/23 |
| +6/255 | .00133798 | .934 | 0/23 |

Dense MSE passes this particular nuisance comparison, unlike the earlier change teacher.
Compare relative penalties within each objective, not MSE to nats. No evidence of broad
invariance or natural nuisance frequency. The non-timer reward alias reproduces identical
RGB/done and reward1/0; first teacher MSE gap .01254, since actual reward is an input. This
establishes outcome sensitivity after observation, not memory retrieval from known earlier
achievement history.

**Decision:** good simple detector-free candidate for a controlled learning comparison; weak
as a guarantee of timer learning. Dense supervision strengthens this fixed-fork long horizon
and removes the known translation-teacher artifact. The hidden distinction is still a small
part of full-state agreement. Compare endpoint-only and dense using identical corrected fork,
initialization, horizons and other objectives; mean the time losses instead of inadvertently
increasing auxiliary weight by summing 16 terms. Keep ordinary prediction grounding.
A native erasure/readout on multiple useful history facts is still the criterion. No run queued.

RAW.json contains all step-level results and input/checkpoint hashes; SUMMARY.json contains
aggregates. No new labels in an objective.
