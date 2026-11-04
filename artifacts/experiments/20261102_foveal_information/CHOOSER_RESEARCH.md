# Cheap endogenous fixation: source pass and computational contract

2026-11-02. Candidates for discussion, not an adopted P1-F2 recipe. Corpus unchanged.
The user's main priority remains diagnosing fine-evidence uptake. These sources address
how to choose cheaply; none establishes that our world will use what it sees.

## First correct the premise about H3

Read `20261031_learned_fixation_sweep/selection.py` (Priority/train) and RESULTS H3.
The excess-error head achieved R² .95–.99 but its target was dominated by an invalid
noise floor. The resulting selection was substantially worse than uniform. The raw-error
head achieved R² .46 (super-resolution) / .85 (forward), with evaluation rank correlation
.38 / .56. Its super-resolution learner reached uniform's 2,000-step loss at 1,731 steps,
but ended 6% worse; forward learning ended 16% worse overall. H3 demonstrates some
pre-crop predictability, not that an accurate head automatically chooses useful evidence.
This was a small frozen-encoder harness, not a native joint P1 comparison.

## Runtime: two different costs, not one

Frozen-native B=16,T=64 benchmark (`20261102_p1f15_readiness/gaze_bench16/`):

* Uniform sequential collection: 2.210 s / 1,024 frames.
* C/S sequential collection: 3.245 s / 1,024 frames.
* Difference: 1.035 s / 1,024 frames.

Extrapolated to 25.6M frames: collection ~22.5 additional GPU-hours; replacing the
forecast calls alone saves ~7.2 hours, leaving ~15.3 hours of the prototype's sequential
perception/Mamba/context work. These are benchmark extrapolations, not a production ETA.
A head consuming the MAIN belief is fast individually but still requires the preceding
chosen glimpse before computing the next main belief. It does not restore parallel P1.

The batched candidate must instead use current coarse tokens and a coarse-only history,
or explicitly adopt stale/periodic decisions. That is an INFORMATION change, not a
transparent optimization. The current real periphery is available before fixation, so
using it is allowed; realized fine crops and future outcomes are not chooser inputs.

## Families and their actual contracts

| Family | Cheap selection | How trained endogenously | Principal cost/risk in Saccade |
|---|---|---|---|
| Online distillation of native C/S priority | One small priority map on current coarse tokens + coarse history + previous action | Occasional detached native DO/NOOP teacher scores on ordinary world-training histories; fit KL to teacher policy or regress separate C/S maps | Teacher has fine history the student lacks; teacher drift; C/S is not learning progress |
| Direct action-conditioned expectation head | Predict coarse expected next tokens for actual action and NOOP, then compare; surprise against arrived coarse frame | Detached recorded coarse targets, head-only loss; coarse history allows batched training | Its competence is not the world's competence; prediction error includes irreducible randomness; action contrasts need coverage |
| Head predicting existing native training loss | One scalar per candidate window | Reuse detached per-cell G1 errors already computed by P1; condition score on noise band or average declared bands | No extra teacher calls, but error is not reducible error; random noise level and moving encoder complicate target |
| Learned acquisition policy / contextual bandit | One policy pass | Prediction-quality or held-out progress reward, with independent evaluation and exploration | Delayed, noisy credit; must specify prediction target; main-belief input still creates sequential dependency |
| Inverse dynamics / contingency attention | Small shared per-cell attention model | Predict previous action from consecutive coarse frames | Finds action-diagnostic regions, can miss uncontrolled threats and lighting uncertainty; not equivalent to native DO/NOOP sensitivity |
| Small predictive ensemble | Several tiny forward heads, disagreement map | Same detached coarse transitions, independent models/bootstrap | Can distinguish some reducible uncertainty from stochastic outcomes, but disagreement is not calibrated epistemic uncertainty automatically |

A conditional mean is defensible as a cheap EXPECTATION used to rank fixation locations.
It is not being installed as the imagined world. The old objection to averaging mutually
exclusive world states does not by itself rule out a mean-based priority estimate.
A shared encoder trained to produce a sharper mean would be a different intervention;
head-only gradients keep that distinction explicit.

## Primary sources read and the limits of transfer

1. **Policy Distillation**, Rusu et al. (2016), sections 3 and 4.5:
   https://arxiv.org/html/1511.06295
   Fits smaller students from teacher action values/policies, including online training
   as the teacher learns. This supports concurrent teacher/student operation, not our
   specific sparse refresh rate or altered student information set. A score teacher can
   be the CURRENT Saccade, so no separately pretrained reference is required.
2. **Deep Adaptive Design**, Foster et al., ICML 2021, sections 2–3:
   https://proceedings.mlr.press/v139/foster21a/foster21a.pdf
   Amortizes sequential experimental design into one history-conditioned policy pass,
   trained over simulated experiments using contrastive information bounds. It assumes
   an experimental likelihood/simulator and prior. Our moving, implicit latent world
   does not inherit its exact information objective for free.
3. **Learning to Acquire Information**, Pu et al., UAI 2017, sections 2–4:
   https://arxiv.org/html/1704.06131
   Predicts observation outcomes directly, avoiding enumeration of hidden hypotheses.
   Maximizing observation entropy equals their information-gain objective only under
   uniform conditional observation entropy. That assumption is unsuitable as a blanket
   justification in day/night static; high entropy alone can reward noise.
4. **Maximizing Information Gain ... via Prediction Rewards**, Satsangi et al., AAMAS 2020,
   sections 3–4: https://arxiv.org/html/2005.04912
   Learns sensing and a prediction model together; derives bounded approximations between
   prediction rewards and convex belief objectives. Their DAN experiments use labelled
   prediction targets (person location, digit class). We need a label-free world-prediction
   target; their result does not equate arbitrary latent MSE with information gain.
5. **Contingency-Aware Exploration**, Choi et al., ICLR 2019, section 3.1:
   https://arxiv.org/html/1811.01483
   Inverse dynamics predicts recorded actions, while spatial attention identifies
   action-predictive regions. Current-frame local features generate the attention;
   consecutive frames train the action prediction. This is a natural cheap contingency
   signal but may ignore exogenous events. Offline action correlations are not an
   identification guarantee of causal influence.
6. **Self-Supervised Exploration via Disagreement**, Pathak et al., ICML 2019:
   https://proceedings.mlr.press/v97/pathak19a/pathak19a.pdf
   Prediction-model disagreement is an alternative to single-model error for stochastic
   environments. It avoids treating all outcome variability as learnable error under
   the model/coverage assumptions; shared bias and poor ensemble diversity remain risks.
7. **Prediction-Oriented Bayesian Active Learning**, Bickford Smith et al., AISTATS 2023:
   https://proceedings.mlr.press/v206/bickfordsmith23a.html
   EPIG values information about predictions on a target input distribution. This supports
   defining what prediction improvement we want; it does not make native C/S equivalent
   to EPIG, or solve the compute problem.
8. **Active Fovea-Based Vision Through Computationally-Effective Model-Based Prediction**,
   Daucé (2018), section 3.2.3: https://pmc.ncbi.nlm.nih.gov/articles/PMC6302111/
   Uses approximations and memory-based scan-path readout to bypass expensive generative
   planning. MNIST classification, not joint stochastic world learning. It reinforces
   that an expensive online generative evaluation is a choice, not a requirement of foveation.

## Narrowest comparisons worth considering after the uptake finding

First compare TWO information contracts before choosing a head: native fine-dependent
belief versus a coarse-only history. Keep the same frozen world, scenes, actions and
teacher noise. For each head use episode-disjoint fitting/reporting and both noisy and
low-noise conditions; report rank AND induced policy TV/allocation, not R² alone.

The least disruptive student is an endogenous, periodically refreshed distillation head:
coarse-only inputs permit all fixations to be chosen in a batch; detached teacher targets
can be calculated sparsely AFTER the ordinary P1 pass, when its actual chosen history is
known. There is one update of lag and missing fine history, both explicit. A direct
expectation head is a useful matched alternative because it avoids teacher calls entirely.
A native-loss head is cheaper still but needs a noise/progress falsification, given H3.

Reuse previous validation questions: independent forecast-noise seeds; exact static
re-renders with physical state fixed; own-history selection coverage/repetition; coverage
of action and exogenous events; actual collector/training runtime; clean pre-crop inputs.
Add checkpoint 1/5/10 transfer/refresh tests to expose teacher drift. These are feasibility
checks; only matched joint world training establishes learning benefit. No new priority
head experiment or overnight run is selected by this research note.

## Sparse teacher is feasible in principle, but its exact work is not free

A native DO/NOOP teacher needs the NOOP action applied to the BELIEF as well as to the
denoiser. An ordinary full-chunk Mamba pass emits all beliefs but returns only the final
carry. It does not expose every pre-action carry for arbitrary teacher frames. The
batched route is a current-parameter prefix replay with the selected preceding action
changed, recomputing BOTH branches together. Ordinary training's noisy denoiser activations
also cannot simply be reused as the teacher's fixed-context, high-noise forecasts.

Thus a concrete implementation must benchmark sparse prefix replays plus native context
forecasts, not count only two cheap last-token calls. One teacher frame per stream/chunk
is a possible frame-budgeted candidate, not a chosen refresh rate. Labels should be
computed under the same weights as their detached coarse inputs, e.g. after backward
frees world activations and before the optimizer changes those weights. Changing a
whole policy or using teacher labels one update later is allowed only if that lag is
explicit. A sampler/student warm-up, refresh age and exploration floor still need tests.

A useful two-arm student test would retain the same network, coarse inputs and budget:
(A) fit native C/S maps from sparse self-teacher labels; (B) fit action-conditioned coarse
expectations and construct C/S without teacher labels. Match teacher noise/seeds during
reporting and distinguish forecast error from ranking/allocation. This tests the estimator;
matched joint P1 training is a later, different claim. No ETA for either student is
asserted before its actual P1 collection/training path is measured.

## A batching alternative: blockwise feedback, rather than dropping all fine history

Coarse-only conditioning is not the only way to remove the per-frame dependency. A
head can use the EXACT native belief at a block's start, plus the current coarse scene
and a coarse-only within-block history, choosing all of that block's fixations in one
batch. Fine observations still enter the main belief during ordinary world training,
and the chooser refreshes its anchor at the next block. This deliberately delays fine
feedback inside the block; it is NOT the original fully adaptive chooser evaluated faster.

The static-versus-adaptive distinction is standard in experimental design (DAD section
2); this particular Saccade block policy is our candidate adaptation. It preserves fine
history preceding the block, unlike a wholly coarse-only head, but block length becomes
a substantive approximation parameter. With 64/256-step P1 chunks, blindly using an
entire chunk as the block is not justified. Shorter blocks trade fresh fine feedback
against parallelism and repeated collection. Validate on own histories, especially
immediately after a useful glimpse and at episode resets. The chosen head inputs must
also be available to the imagined cascade and its conditioning; do not silently add an
unobserved policy state. This is a third computational contract for discussion, not a
new production collector or a selected workaround.

## Timing: a score predictor and a forward predictor have different legal inputs

At arrived frame t, the previous physical action is a[t-1]; the eye selects crop g[t].
A PRIORITY-map student may read p[t], because p[t] exists before that crop. It predicts
teacher scores C[t],S[t] without seeing g[t].

A direct FORWARD-mean head intended to replace the teacher's forecast must instead use
history THROUGH t-1, before seeing p[t], and action a[t-1]:

```
mu_taken = mu(coarse_history_through_t_minus_1, a[t-1])
mu_noop  = mu(coarse_history_through_t_minus_1, NOOP)
C[t] = squared_difference(mu_taken, mu_noop), pooled by crop
S[t] = squared_difference(p[t], mu_taken), pooled by crop
```

Feeding p[t] into that mean head would permit copying its target and collapse the
meaning of surprise. A main-belief anchor used for this contrast must also precede the
action being varied; an anchor already containing the taken action cannot be reused
unchanged for the NOOP branch. These conditions must be tested explicitly, including
C=0 for actual NOOP. A score-regression head can use p[t] legitimately, but should not
be described as a pre-outcome forecast. None of these forms receives the realized crop.

## Concrete small-head compute check (no learning claim)

`head_cost.py/json`: a 46,978-parameter shared per-window MLP, 128/128 hidden units,
current local coarse tokens, lighting, four coarse-summary slots, previous-action one-hot
and coordinates; two score outputs/window. Native real frames, B=16 x T=64 flattened;
zero history/action context in this shape-only timing check. Five warm-up passes,
30 timed repetitions, CUDA synchronization:

* head only: median **1.37 ms / 1,024 frames**;
* extra full native perception pass plus head: **15.44 ms / 1,024 frames**.

The latter even computes an unused dummy glimpse. This demonstrates that batched head
arithmetic can be cheap; it does NOT measure trained policy quality, head backward steps,
NOOP teacher/prefix cost, reset handling or the production pipeline. It is not an ETA.
The decisive runtime issue is the feedback dependency, not the size of this tiny head.
