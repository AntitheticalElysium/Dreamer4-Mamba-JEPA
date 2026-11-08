# Task-relevant incremental fine information — 2026-11-04

User requests actual investigation of the prerequisite, not a proposed repair.
Frozen F1.6ck10 (34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045).
No world training, architecture/sensor/chooser change, F2 science, or actor claim.

## Questions kept separate
1. Does real fine sensing improve a held-out estimate of a task-relevant fact,
   beyond the full coarse scene, recent history and prior native belief?
2. Does that fact change damage/reward under matched fixed action plans in the
   actual environment? Native generation need not already master that event.
3. Does the native system itself acquire/use the information? NOT the purpose
   of this campaign; no pathway repair follows automatically.

## Facts and controls
Current visible foreground identity (empty/zombie/cow/skeleton/arrow) and arrow
heading (four directions). Distinguish a confidence improvement from a class
flip. Forecasting fine latents or pixels alone does not meet the prerequisite.

Sources: existing first400 train episodes (360 fit,40 calibration), first400 dev
(250 discovery,150 confirmation), all actual map queries inside the original
random genuine crop. Keep all source counts; no future-outcome selection.
Episode boundaries/history index/fixation/target alignments are asserted.

Strong coarse baseline: current full81x16 tokens/light, eight previous full
physical percepts, prior belief outputs, actions/outcomes/fixation/query identity.
Add the actual raw3x3 pooled inputs over the entire current scene AND eight
previous scenes, plus the native pre-glimpse TSP pixel prediction for the queried
cell. This is deliberately generous to coarse vision. It prevents treating a
weak local probe miss as information absent from the available coarse path.
The baseline never reads the current fine crop or corrected native belief.

Compare equal-parameter readers with zero extra slot, actual native fine crop,
or actual full-resolution pixels of that crop. The crop/coordinates are the same
in all arms. Two seeds and two reader parameterizations, fixed1200 updates each;
no negative-result budget extension. Class-balanced diagnostic training,
calibration on natural-frequency held-out TRAIN episodes only. Report NLL,
Brier and confusion/recall by light/range/kind/new-visibility as sample counts
permit. Paired episode-bootstrap intervals on confirmation data. Positive held-
out gains are practical conditional predictive value, not proof of information-
theoretic absence from an unlimited observer of the entire history/full carry.

Retain both a broad-input MLP and an explicitly separated coarse/extra reader
so a single optimization geometry cannot decide the conclusion. A coarse-only
negative result does not prove information absence; a fine gain must replicate
and improve calibrated semantic predictions, not merely training fit.

## Task relevance, once an information gain qualifies
From discovery select at most12 different-episode cases per informative task
type (threat identity, arrow heading), plus up to12 from confirmation using the
UNCHANGED source rule. Fine raises calibrated true-fact probability by>=.15 on
average over the readers, and to>=.65; all individual changes must be positive.
Do not require coarse misclassification. If no case meets this hard example
rule, retain aggregate gains and do not tune the rule after seeing outcomes.

Replay source exactly; pair with removed foreground or changed arrow heading,
holding other simulator state/action/key fixed. Genuine encoded alternative
must confirm that the fine reader is responsive in both directions (not a
persistent false-positive prior). Report prior visibility and native TSP/source
readout; do not pretend the alternatives share an entire feasible past merely
because they share a rendered current background.

Six-step plans: constant NOOP,LEFT,RIGHT,UP,DOWN,DO.32 common random-key sequences
per plan/world. Record cumulative health decrements, environmental reward,
termination, and paired source removal/direction effects. No recorded reactive
future actions or native forecast supplies these labels. Simulator state is
used to evaluate causal relevance only, never as a reader input.

The requirement is task-relevant BELIEF information; a changed optimal action
is not mandatory. Do not manufacture an auxiliary navigation reward to force
an action flip. Any decision-value illustration using a restricted set of plans
must be labeled as such, not as native actor or population policy improvement.

## Stop and report
If no replicated semantic gain exists, report which sensor/encoding/readout
premise failed. If fine helps a fact but no matched consequence changes, report
that limitation. If both hold, the prerequisite is demonstrated in the stated
cohort, not globally or for the native world model's use of the glimpse.
No sampler/mixer repair or overnight launch is part of this task.

Analysis declaration before inspecting any held-out scores: report kind,
threat-vs-empty/cow, and heading. Primary regions are night and night distance>=3;
all-day results also retained. A replicated semantic gain requires positive
confirmation NLL benefit in each of the four reader fits, plus positive lower95%
episode-bootstrap bounds for mean NLL AND Brier benefit. This is a conservative
operational gate, not a theorem about infinite-capacity information. Case rules
remain unchanged. Native-fine candidates take precedence; raw-pixel-only gains
are explicitly a sensor/encoding finding and are never called native fine use.

Independent structural witness search: inspect valid rendered source alternatives
for EXACT equality of all raw3x3 periphery pixels and native lighting/periphery
outputs, despite different task facts and fine crops. An exact witness avoids
relying on negative classifiers. Verify full prior observations using a replayed
alternate history before calling it beyond memory. Then verify a causal action
consequence (movement/blockage, harvest, health or reward) in the real simulator.
A witness proves existence for the stated sensor, not prevalence or native use.

Post-result separating control, before its outputs: the zero extra slot in the
coarse arm leaves its side branch inactive. Test whether a more convenient
coarse representation explains the observed gain. Four additional coarse-only
readers get the FULL native pre-glimpse TSP crop in that same1323-dimensional
extra slot; otherwise identical1200-step fits. Compare to the existing fine arm;
do not extend its training. Choose the stronger coarse variant per family/seed/
task by NLL on the40 held-out TRAIN calibration episodes only. Confirmation data
never chooses the baseline. Retain original results and all24 source examples.
This is a control for access/formatting, not a world-model architecture repair.

Source-driven affordance dissection after the six-step constant-plan results:
confirmation episode307,t161 has the clearest cross-reader confidence separation
(all fine estimates exceed all coarse estimates), but zero injury difference in
all six constant-action plans. Its visible map has a stone barrier DOWN and iron
LEFT; inventory has wood picks but no stone pick. Test exactly DOWN,DO,DOWN,then
six NOOPs under32 paired random-key sequences. This plan is fixed from source
geometry/tools before its outcomes. Check the barrier is actually mined in both
worlds. It is an actual environmental mining consequence, with no added reward,
teleporting or actor claim. Keep the original zero-effect six-step results.

Critical falsification: the existing fixed S1 LOCAL coarse reader recognizes all
24 initially selected threats with near-certainty. Thus uncertainty in the larger
context readers does not qualify those examples as fine-required. Retain their
real causal effects but do NOT count them as proof of incremental information.

Final acquisition search, declared before its outputs: scan all current map
queries in the400 dev episodes for night hostile entities whose S1 true-kind
probability is<.8. At most2000 distinct source frames, prioritizing no visibility
in the preceding8 frames, then corpus order. Take a genuine focused crop at the
query, without changing any coarse pixels/tokens/history or fitting new readers.
Evaluate the existing four fine readers against their train-calibration-selected
coarse/context counterparts AND the S1 local coarse reader. Qualification:
EVERY fine true-kind probability exceeds the BEST of all five coarse estimates
by>.05, and mean fine true-kind probability>=.8. At most12 distinct episodes per
discovery/confirmation stratum. Future outcomes do not select these sources.
Then run the SAME fixed simulator relevance plans and fresh paired-absence
checks, including S1. If this stricter acquisition search fails, report failure;
no new readout fitting or automatic component-repair campaign follows it.

Symmetric existing-instrument check, before values: the wide fine-augmented
readers yield zero qualified focus examples. Since a local coarse instrument
already falsified the wide-coarse examples, check the two EXISTING fixed local
fine instruments too, without fitting anything. On the SAME2000 focused crops,
require all five coarse instruments to misclassify the query and give its true
kind<.5; both local fine instruments must classify it correctly with score>=.9.
These class-balanced local scores are not calibrated environmental probabilities.
This is an example-existence/acquisition check, not a population value estimate.
Counterfactual background checks and simulator consequences remain mandatory.
Do not infer absence of fine information from failure of a wide readout to use
an otherwise readable current fine token. No new raw-feature readout fits follow.

Matched existing-reader safeguard, before these values: the same saved local
instrument bundle also contains p_0, trained with the same class-balanced
protocol as g_0/g_1. Include p_0 among the mandatory coarse controls, rather than
letting different class priors create the apparent gain. Re-evaluate the SAME
2000 focused sources without any refitting; retain the previous five-control
results separately. Qualification now requires all SIX coarse readers to be
wrong, best coarse true-kind score<.5, and both fine scores>=.9. The source caps
and source order are unchanged. Pair checks must also include this matched local
coarse reader. This control can falsify the preliminary 356-query finding.

Final witness validity audit, before outputs: check the FOUR surviving paired
sources only. No replacements or new candidate search. Record the complete
source mask/visibility history, keeping slot reuse distinct from current active
interval. For each fixed factual/removed state, render 64 fresh common night-
static keys and score the EXISTING local coarse and fine readers (no fitting).
Also compare native PRE-glimpse TSP and fine reconstruction on the exact pixel
contrast between factual/removed natural source images. If TSP already separates
both alternatives, do not call this a fine-exclusive fact. Independent-render
scores test instrument reliability, not native forecast or policy improvement.
Keep the original natural-frame classification and all 6 fixed action plans;
report NOOP separately so a selected maximum plan is not sold as population gain.
Then stop and report the bounded campaign, including failed confirmation checks.
