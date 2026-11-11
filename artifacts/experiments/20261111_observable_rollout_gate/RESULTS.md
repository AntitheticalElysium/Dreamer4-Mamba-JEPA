# Gate: observable supervision along a blind latent rollout

2026-11-11. Completed CPU replay and exact score comparison. No world training, no canonical edits.
The proposed one-scan curriculum is specified in PLAN.md. This is the prerequisite objective test, not
an implementation or training of that curriculum. Existing change-target detector and reward bins are reused.

Compute the optimal forecast with phase available, versus the optimal forecast with histories tied, leaving
all targets unchanged. Bernoulli gap is entropy of the pooled soft target minus mean entropy. Reward/termination
use the analogous exact categorical log-score gap. The auxiliary averages the three group means; change BCE
uses all valid cell/time/plane entries across both members (singly valid entries add no alias cost). No new
outer coefficient is chosen. Exact-duplicate control gives zero, hard binary gap reproduces ln2.

## Timing, reward and nuisance

23/24 original hunger pairs qualify through the long fork (one terminated). Replay reproduces original recorded
targets, rewards and termination byte for byte before extending. Common earlier fork at step24: food decrements
occur at elapsed steps8/16. Later fork at step31: first decrement one step away. Target is the existing
soft change score of **each consecutive observation pair**, not the absolute value at a final frame.

| Fixed earlier fork, H | Erasure cost, mean of all steps | Erasure cost, last transition only |
|---|---:|---:|
| 1 / 2 / 4 | 0 | 0 |
| 8 | .00034296 | .00274685 |
| 16 | .00034683 | .00280359 |

Units are nats under the provisional equally weighted auxiliary. The last transition at16 **does** carry
a difference because it is the time of the second branch's drop. The *final images* agree; the last
**change targets** do not. A path score sees both drop times, but averaging every step dilutes the
penalty versus scoring one informative transition. At the later fork, one-step cost is .00274685.
Most pairs' teacher signal is at8/16; one also has extra change gaps at9/11.

Non-timer: re-replayed the achievement novelty alias, identical current and future RGB and termination, actual
reward1 vs0 on the first action. Phase/history erasure costs ln2 nats in the reward group at one step,
.231049 under the full auxiliary. This verifies reward sensitivity, not learnability from a pair of actual
achievement histories. It is a state counterfactual, not a qualified visual revisit case.

Prescribed brightness step on a copied, otherwise identical future (same physical path/outcomes):
+1/+3/+6 out of255. These are deliberate nuisance foils, not measured prevalence estimates.

| Exposure foil | Mean dense-H16 penalty | Max |
|---|---:|---:|
| +0 | 0 | 0 |
| +1 | .00018726 | .00324345 |
| +3 | .00015008 | .00323045 |
| +6 | .00014905 | .00322694 |

Mean guard passes. **Paired robustness does not**: same scene (episode9) fails for each brightness amplitude.
For +3, nuisance is12.09x that scene's hunger cost. 99.86% of its unnormalised change-entropy gap comes from
the dominant-translation plane. Geometry validity differs in27 cell/time/plane entries (consistent with three
translations of a nine-cell edge). No physical movement was introduced. This localizes the problem to the
translation target path, not to the time schedule or reward bins. No claim that exposure is always irrelevant
or that the foil occurs with this frequency in normal training.

## Decision

Do not launch the proposed auxiliary as-is. The temporal target and ordinary outcomes give real, label-free
incentives, but "all intermediate steps" is not a strength multiplier, and the inherited correspondence
teacher can give much more pressure to an appearance artifact than to hidden dynamics. Qualify a more stable
change target on this exact foil and real motion before making it a forecasting teacher. Preserve the existing
G2 supervision; it was validated for a different role. Keep short queries in a horizon curriculum. No evidence
here that a shortcut head, larger weight, or broad time average alone solves memory.

RAW.json, SUMMARY.json and gate.py reproduce the comparison. SCENE_9.json was an interim localization run
before changing the geometric-mask denominator to pool valid BCE entries; final numbers above use RAW/SUMMARY.
