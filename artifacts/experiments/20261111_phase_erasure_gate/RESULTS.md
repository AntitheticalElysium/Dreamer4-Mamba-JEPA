# Phase erasure as an objective check

2026-11-11. CPU simulation/analytic scoring only; no model training or objective selected.

The original hunger pair bank was reproduced bitwise (frames, food, reward and termination) before extending
the trajectories with the same NOOP actions and environment RNG. Same fork at branch step 24, exact same last
eight frames and incoming rewards, older phase evidence differs. 23/24 survived to the 16-step endpoint; one
termination excluded, explicitly recorded. This is a constructed bank, not a prevalence estimate.

| Future horizon | Differing food endpoint | Differing full RGB endpoint | Percept MSE erasure gap |
|---|---:|---:|---:|
| 1, 2, 4 | 0/23 each | 0/23 each | numerical zero |
| 8 | 23/23 | 23/23 | .00177444 |
| 16 | 0/23 | 0/23 | numerical zero |

At h8 exactly one screen cell differs. F4 encoder CPU fp32, common central fixation; compare oracle target
reproduction with best shared mean after erasure. Percept gap is ||y1-y2||²/(4*1880). The largest local
periphery-token gap is .208492, versus .001774 averaged over all percept coordinates. This demonstrates
spatial dilution on these pairs; it is not a measured training gradient, a monitor comparison, or an
estimate of G1's loss on natural frequencies. For byte-identical frames, <1.5e-14 differences are rounding.

First food decrements at offsets 8 and 16 for all qualified pairs. At H16 the endpoint loses the distinction
but the trajectory does not. A simple analytic candidate query uses the first time a screen patch differs
from its starting patch (including a no-change category); it applies the same test to every cell, without
hunger/phase labels. Exactly one query separates the cases. Its optimal phase-aliased forecast assigns
probability .5 to each observed onset, paying ln 2 nats for the distinguishing query, .008557 nats when
averaged over all 81 cells. The endpoint-change query has the same gap at H8, zero at H16. Raw RGB change
is a clean diagnostic on these pairs; it is not a qualified training target for camera moves or night noise.
Compare different objectives at their actual chosen weights and normalizations, not by raw nats vs MSE.

The horizons are measured from a common fork. Earlier blind-student audit selected a different fork for
each H so that all queries ended at the due event, so it answers another question. A two-step forecast
can require old phase at a different fork. Under ideal awake constant actions, uniform phases and period
P=26, for 0<=h<=P the chance of a first drop by h is h/P. The phase discrimination from that single
endpoint goes back to zero at a full period, even though onset is different.

## Gate for a proposed loss

1. Objective: use matched recent inputs/actions, two older histories and unchanged recorded targets.
   Compare the best phase-aware forecast to the optimal pooled forecast once that distinction is erased.
   Price the gap with actual weighting, sampling and reductions, both per query and total. Qualify on
   other memory-dependent facts (occlusion, a consequence depending on old interactions, fine evidence)
   and sham pairs with no relevant future difference. Compare against easy appearance differences under
   the same score. Hidden state may verify the evaluation, not supply a training label.
2. Learned system: after training test native generated effects under the two natural histories. For
   causal erasure hold recent visual inputs fixed and tie the history-derived route, including memory
   contributions to the cached keys/values. Do not just zero a few coordinates or only the current bp.
   A strong oracle loss penalty is necessary incentive on the tested cases, not proof SGD learns to use it.

No training or simulator-label objective. Primary comparisons: DRAMA, DreamerV3, Dedieu et al. on Craftax
and Dreamer4/JEDI; their reported task/video performance does not establish exact hunger-phase prediction.
