# Gaze qualification and actual P1 coverage

2026-11-04. Completed CPU census and archived-results audit. No GPU experiment,
model/probe fit, architecture change, training launch or queued follow-up.

## 1. Two different gaze questions

**Current-query invariance:** at fixed prior history, action and sampled physical
percept, integrating over the not-yet-seen glimpse should give the same expected
environmental reward for every fixation. A useful gaze can change the posterior
after acquisition; it cannot change the physical reward distribution in advance.
This consistency question does not require a fine-only informative crop.

F2 already repeated the F1.5 all-49-query test in its automatic evaluation. Both
have 256 contexts and eight sampled glimpses per query. Reanalysis using the
original descriptive rewarding-context criterion (mean reward across queries
greater than .05):

| World | Rewarding contexts | Median query max-minus-min expected reward | Largest range |
|---|---:|---:|---:|
| F1.5 |15|.05657|.23076|
| F2 |12|.10124|.27928|

The selected subsets differ between worlds; do not call this a matched causal
regression estimate. Query extrema contain Monte Carlo selection effects. The
largest F2 example is episode70,t56: query means .18371–.46299, largest reported
per-query MC SE .00504. It supplies a substantive remaining consistency concern,
not evidence of an actor exploiting it. F2 continuation is identically1 in this
panel; F1.5 minimum is .99999988. Neither provides a useful terminal-consistency
test. Physical p/v current-query invariance is enforced by the factorized mask;
that guarantee does not extend to the glimpse-dependent outcome heads.

**Past-glimpse uptake:** does acquiring one genuine crop rather than another
improve the next native prediction? This is the fine-information question. A
negative is diagnostic of a deficient fine route only if the crop supplies a
relevant distinction unavailable through the coarse/history route and the task
and readout are qualified.

F1.5 did perform a qualification: among595 fixed dev scenes,58 night scenes in48
episodes had a true entity missed by the native coarse reader and correctly
identified by both local fine readers. It compared genuine informative crops
with entity-free, same-row crops, holding complete earlier history, action,
outcomes, future query and noise fixed. A stricter refitted-coarse-miss subset
contained25 scenes. These were not unfiltered random gazes.

However, this did **not** establish incremental information beyond the FULL
coarse scene and prior belief, nor useful task consequences/native competence.
Later readers recovered about half the initial coarse misses. Later HUD and
terrain campaigns exposed stronger native coarse/TSP shortcuts as well. This
qualification limit can explain some nulls; it does not prove redundancy caused
all nulls, or clear integration/Mamba/generation of a fault.

F2 also already repeated the58-scene native local uptake test, with its own fine
readers and unchanged reference scenes. Local denoising loss changed +.00223%
[95% episode-bootstrap interval -.00503%,+.00975%]. Its34 common-readable scenes
remain inconclusive. This is a limited native-objective uptake measurement,
not proof of equality of entire generated distributions or absence of actor
value. The old reader-miss cohort does not become fine-required because F2 has
a learned chooser. A rerun with the same qualification adds no missing premise.

Sources: F1.5 `20261102_p1f15_readiness/UPTAKE_RESULTS.md`,
`uptake_local.py`; F2 `readings/fovea/summary.json`, `fovea_readings.py`;
the two `readings/consistency/consistency.npz` archives. `COMPARISONS.json`
retains the current-query reanalysis and archived interface comparisons.

## 2. Complete P1 training census

Actual train split:7,672 episodes,2,432,119 transitions;397 monitor episodes
excluded. Actions, rewards and termination labels equal the raw stored arrays
for EVERY selected episode. State/transition length and offset alignment are
asserted; all termination flags equal next-health<=0. Counts use simulator
labels for diagnosis only, never learning. Source/script/manifest hashes are
saved in `COVERAGE.json`.

| Event | Unique transitions | Episodes containing it | Share of transitions | Actual repeated F2 world targets |
|---|---:|---:|---:|---:|
| Drink increases |39,435|4,073|1.621%|413,542|
| Wood increases |64,952|6,671|2.671%|681,902|
| Stone increases |88,495|4,750|3.639%|928,734|
| Coal increases |9,727|3,398|.400%|102,137|
| Iron increases |5,981|2,249|.246%|62,755|
| Health decreases |47,618|7,672|1.958%|499,533|
| Death |7,621|7,621|.313%|79,980|
| Low health plus adjacent zombie |28,891|4,934|1.188%|303,209|
| Death within that risk class |3,701|3,701|.152%|38,817|

The25.6M-row actual gaze trace was counted, not an assumed number of epochs.
It contains25,519,470 prediction-target rows plus80,530 episode starts. Every
counted event was encountered; average exposures are about10.5. Repeats are not
additional independent evidence or a guarantee of effective learning pressure.
P1 streams include terminal observations. There is no P1 exogenous-window filter.
Current P1 loss masks exclude episode starts, not terminal transitions.

Coal/iron are about9.1/14.8 times rarer than stone. Tool-qualified, uncapped,
awake DO mining attempts equal successful increments exactly for wood, stone,
coal and iron in these labels. Thus these mining increments are not visibly
stochastic under that state-defined condition. This says nothing about whether
the eight-frame coarse head reliably infers the condition.

Once-per-episode coal and iron unlocks number3,398 and2,249. A zero-to-positive
increment can recur after crafting consumes inventory; it must not be confused
with the first unlock. Upper-value support is genuinely thin: iron7->8 occurs
4 times, iron8->9 never; coal8->9 occurs30 times. Such holes cannot alone explain
failure to predict first unlocks, which have thousands of source episodes.

Support is strongly behavior-dependent:

| Collector epsilon | Transitions | Coal increments | Iron increments | Deaths in risk class |
|---|---:|---:|---:|---:|
|.10|1,289,865|6,394|4,699|579|
|.25|524,328|2,658|1,214|824|
|.50|345,127|675|68|1,105|
|1.00|272,799|0|0|1,193|

The pure-random episodes lack coal/iron events. This matters especially to the
planned P2 exogenous-only objective. It is not evidence that P1 dropped them:
P1 used all epsilon groups. Nearly all deaths occur in training episodes, while
the one-step death transition itself is rare. Both descriptions are necessary.

The head sampled256 examples with replacement per1,024 new world frames. Expected
coal/iron training draws are about25,613/15,736, accounting exactly for each
logged chunk's eligible population and number of head updates. Exact minibatch
identities were not recovered from CUDA RNG and are not claimed. These are repetitions, not
25,000/15,000 independent coal/iron situations. Coverage alone cannot distinguish
target competition, lack of model capacity, inadequate visual/history support,
optimization, or counterfactual generalization.

## 3. Older danger results really were better under guided gaze

The archived comparison script asserts identical class history counts and ALL
recorded target metrics, including across gaze policies. Risk class has3,805
histories and4 sampled futures each. Primary saved production samplers only;
no alternative-sampler selection. Model outcomes are archived conditional-head
probabilities averaged on those futures, not simulator danger probabilities.

| World | Generated death, random gaze | Generated death, guided gaze | Head on actual next frames, guided gaze |
|---|---:|---:|---:|
| Original world1 |.039%|.027%|12.772%|
| P1-F |1.221%|7.900%|12.773%|
| beta-TSP |1.443%|4.994%|12.773%|
| F1.5 |1.750%|1.556%|12.772%|
| F1.6 |1.091%|1.093%|12.772%|
| F2 |1.266%|1.221%|12.773%|
| Recorded |12.773%|12.773%|—|

For guided damage the parallel comparison is P1-F11.876%, F2 4.143%,
recorded12.746%. The improved old result is real under the same evaluation
condition, but P1-F was strongly gaze-gated. Under random gaze F2 exceeds P1-F
on both damage and death. Both columns matter.

All worlds used the same P1 corpus; F1.5 changed acquisition AND factorized the
generator, F1.6 changed fusion, F2 changed acquisition only versus F1.6. Thus a
changed corpus cannot explain these differences. Learning/generation recipe
differences matter, but this is not a clean architecture-only causal isolation.
Real-next-frame head recognition is excellent; it does not establish that the
prior belief contains sufficient information to forecast that next frame.

F2 coal/iron unlock predictions are actually better than older versions under
common random gaze:24.37%/9.18%, versus F1.6 17.49%/5.46% and P1-F15.89%/3.59%.
The cheap head's continued surprise is a different metric from native reward
prediction; neither means that there were no examples or that all rare skills
regressed together. See `COMPARISONS.json` for archived SEs and source hashes.

Audit note (Claude, 11-04): the guided gaze is M0.5's privileged rule (`20261030_fovea_placement/PLAN.md`, "The
guided rule"): its context comes from simulator state at t-1, including "zombie at Manhattan distance 1", and those
contexts send the eye to the health-counter window. P1-F was trained on that placement and its joint denoiser read
the query fixation, so under guided evaluation the fixation itself tells P1-F's world that a zombie is adjacent: the
shortcut decision 1 removed. P1-F's 7.9% guided death and 11.9% guided damage therefore measure that privileged cue
together with the world, not danger generation from the agent's own inputs; they do not show that this corpus
already suffices for honest danger prediction. Under random gaze (no cue) P1-F gives 1.2%, like every later world.

## 4. What fine vision should contribute

Exact HUD values need not depend on fine sensing. The acuity EDA reads HUD values
exactly at both3x3 and2x2; the later native codebook also recovers all70,208
tested HUD values. These are finite data/readout results, not proof of perfect
native memory/use. At night, reducing3x3 to2x2 instead reduces near-player kind
recall: zombie.96->.87, cow.96->.83, skeleton.90->.53, arrow.77->.42.
The original acuity probe omits arrow heading, so its values do not measure that
potential detail requirement. Low-resolution entity evidence is the relevant
2x2 tradeoff; HUD is not a reason it would force fine reliance.

Two valid roles remain distinct: acquire new state information for belief/
decisions, and selectively provide precise learning targets in P1. Redundancy
of a HUD fact does not invalidate the second role or the whole architecture.
It does invalidate using a null HUD gaze effect alone to diagnose a broken
fine-information channel. F2's extra entity coverage measures acquisition,
not incremental information or action value.

## 5. The training proposal and actual data readiness

The proposed experiment changes LEARNING conditions, not only a frozen input.
Concrete candidate: the same native world predicts the result of DO in matched
tree-present/tree-absent worlds, with the same initial inventory and action.
The source tree cue is concealed from the coarse context but visible in a
genuine focused crop; the next real/paired simulator consequence is still the
target. Use the native objective and end-to-end update, no extra semantic label
head as a substitute for prediction. A clear-coarse version is the positive
control. Balance/hold out source pairs and test the fine alternatives so a
frequency prior cannot pass.

Leakage controls are part of the design, not implementation trivia: prior
coarse frames, earlier fine acquisitions, lighting's pooled statistics, current
reward/termination, gaze identity, and the future-coarse direct path must not
reveal the named cue before its prediction. A blank-fine trained control or
equivalently qualified paired design must establish that fine is necessary.

We have raw training episodes, complete pixel frames, actions/outcomes and exact
replay/render tools. Wood positives are plentiful. We do NOT yet have a qualified,
split-disjoint fine-required paired TRAIN dataset or measured budget for this
native experiment. Existing saved terrain intervention pairs are DEV diagnostic
cases; they cannot silently be reused as a clean training set. Current natural
fine-only evidence includes one weakly replicated task-relevant night witness,
not enough for a generalization-training dataset. New qualified pairs would have
to be produced from train sources, with held-out sources preserved.

Success would show native fine-path capability under the changed sensor/task.
Failure with a successful clear-coarse control would establish a bounded fine-
path learning deficit; it would not uniquely blame mixer or Mamba. Neither result
would alone establish the utility of fine sensing in the original3x3 corpus.
This is an engineered capability test, not a ready production repair or a promise
to settle all four original hypotheses. It remains a proposal for discussion.

Research checked: Huang et al., ICML2022, modality competition
https://proceedings.mlr.press/v162/huang22e.html supports the plausibility of a
stronger route dominating learning in a simplified fusion model, not a Saccade
diagnosis. Neverova et al., ModDrop, https://arxiv.org/abs/1501.00102 uses
training-time channel dropping and gradual fusion; it is precedent for changing
learning conditions, not proof that a frozen-route null is meaningful.
