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

Mean guard passes. Paired robustness does not in this computation: the same scene (episode 9) fails for each
amplitude; for +3 the nuisance is 12.09x that scene's hunger cost, 99.86% of it in the dominant-translation plane,
with geometry validity differing in 27 entries. The cause is how this gate computes the targets, not the foil:
`target` derives the temporal-variability support from the 17-frame branch window, whereas the canonical store
derives it from the whole episode. With so little variation in the window, the PLAIN member picks a wrong camera
shift at 3 of 16 transitions although the player never moves; the foil makes every cell vary and restores the
correct shift in the foil member only. With episode-level support every shift is correct and the foil costs
.0022 / .0004 / .0003 (+1 / +3 / +6) against .0084 for hunger at the informative step
(`../20261111_change_forecast_gate/RESULTS.md`, Stage 0). The dense-time and reward-bin conclusions are unaffected.

## Decision

Do not launch the proposed auxiliary as-is. The temporal target and ordinary outcomes give real, label-free
incentives, but "all intermediate steps" is not a strength multiplier. The one-step forecast was then gated
with episode-level support, a zombie-readiness case and natural noise floors in
`../20261111_change_forecast_gate/` (fails narrowly: pricing 2.7-2.8x G1 at pure noise against a 3x bar, and
a +1/255 step out-prices 2 of 32 zombie consequences). Preserve the existing
G2 supervision; it was validated for a different role. Keep short queries in a horizon curriculum. No evidence
here that a shortcut head, larger weight, or broad time average alone solves memory.

RAW.json, SUMMARY.json and gate.py reproduce the comparison. SCENE_9.json was an interim localization run
before changing the geometric-mask denominator to pool valid BCE entries; final numbers above use RAW/SUMMARY.
