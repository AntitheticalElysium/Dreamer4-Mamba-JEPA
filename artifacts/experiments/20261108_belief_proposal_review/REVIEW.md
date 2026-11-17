# Review: belief as an approximate information state

2026-11-08, Codex. Reviewed the supplied Claude proposal against the implementation, current configuration, gradient
study and primary papers. No GPU work, production code changes or other-agent job changes. A CPU backward check
uses the actual F4 denoiser weights; two algebraic counterexamples test proposed metric guarantees.

## Judgment

Support the shift toward **explicit predictive supervision of the actor's representation**, with a baseline plan
before selecting a loss. Do not regard the objective as an established cause or contrastive learning as the chosen
repair. The strongest practical finding is a mismatch between the generator's conditioning and the actor's planned
input. That does not mean the cache is inherently wrong or that the actor must reconstruct all visual detail.

## 1. Define the state precisely before applying information-state theory

The spec's opening explicitly says sections1-6 are the larger design and section7 is minimal M0. Section7, line94:
belief = Mamba state PLUS denoiser recent frames. `m0.yaml:202`, however, says actor/BC read corrected_belief, the
Mamba output b_t alone. There is no finished P3 actor yet. The mismatch is a future input contract, not an already
observed failed actor.

There are actually three quantities:

- C_t: internal Mamba recurrent carry (SSM and convolution states per layer);
- b_t: exported 256-D output after the percept, read by the proposed actor;
- K_t: denoiser cache containing earlier visual AND memory tokens.

`Belief.forward` (`model.py:236-242`) returns output and carry separately. The next recurrent update uses carry,
not just b_t. Therefore b_t is not structurally a recursively closed information state. Failure to probe a timer
from b_t does not establish that every internal recurrent state lacks it. What matters for the actor is whether
b_t exports enough decision-relevant information; it need not reproduce every texture or exact counter value.

Subramanian et al. provide reward prediction plus prediction of the next information state as their basic
conditions, and a stronger alternative involving a recursive update plus next-observation prediction (Proposition4).
The paper explicitly gives an example where a valid information state need not predict all observations. The
approximation guarantees also depend on the error metric, value-function class, action/history coverage and horizon;
an empirical mean log-loss or CPC value is not automatically the theorem's epsilon/delta.

Primary source: https://jmlr.org/papers/volume23/20-1165/20-1165.pdf .

## 2. Correct the gradient evidence before using it to choose a repair

`belief_gradient.py` says bootstrap rows are omitted because G1 memory is detached. That is false of the student.
`model.py:565` calls the student using live memory; `:576` detaches memory only for the stopped teacher calls;
`:585-586` trains the student prediction against that target.

Actual F4 weights, CPU, B=2/T=2, one flow and one bootstrap row, backpropagate bootstrap loss ONLY:

- bootstrap loss .00268249;
- bootstrap memory-input gradient norm .00585479;
- unselected flow-row memory gradient norm exactly0.

Thus the study's 7% concerns its measured non-bootstrap rows and is not the whole P1 gradient accounting. Moreover,
it sums some separate token-gradient norms and divides by a combined norm: these are not additive budget shares,
nor a measure of predictive dependence on the belief. This correction does not negate the measured dominance of
reactive objectives on that subset, but its cause/scale must be described accurately.

The quoted sleep7.1x is the TOTAL gradient excess at wake vs recovery, not a change-map-only ratio.

## 3. Existing heads are legitimate conditional outcome readers

The factorisation is p(P_next | history,action) p(r_next | predicted_belief,P_next)
p(c_next | predicted_belief,P_next,r_next). Supplying real P_next to train the latter factors is correct likelihood
training, not erroneous target leakage. They simply do not require anticipating the transition without P_next.

Nor do they only teach the present: reward involves health differences and previous achievement status, so the
belief can be taught to retain past information. The measured strong health-input sensitivity is consistent with
that function. 'No independent requirement to anticipate phase' is more accurate than 'no temporal teaching'.

Support adding a BEFORE-outcome predictive role, without destroying the existing outcome-reader role. These are
different conditional functions; shared weights need an explicit conditioning/missing-input contract if reused.
Future-observation teacher targets must not read the intervening real future observations via teacher-forced belief
updates. Multi-step forecasts also need declared future actions; expert future action sequences can themselves carry
outcome information, so observational prediction is not automatically a causal planning model.

Predictive-head success alone will not certify the denoiser repair. Our health-readout audit already showed useful
predictive clues can coexist with under-produced damage. Read generated consequences as well as auxiliary scores.

## 4. Contrastive loss is a candidate, not automatic protection for rare consequences

CPC fits a density ratio using a negative-sampling task. Its finite-negative MI lower bound is capped by logN.
Static scene identity can solve the task while a rare health change is discarded. Exact counterexample in CHECKS:
128 different scene IDs, correct-ID score30 and other scores0, InfoNCE1.188e-11 despite retaining NO event bit.
This is an analytic possible solution, not a Saccade experiment or claim that all CPC setups fail.

The concern is established in the contrastive literature: Robinson et al. show feature suppression and shortcut
solutions, including at optimum. It is not enough to replace per-element MSE with generic cross-scene InfoNCE and
assert that small predictive events now receive appropriate attention.

Sources:
- CPC: https://arxiv.org/html/1807.03748v2 , sections2.2-2.3.
- https://papers.nips.cc/paper/2021/file/27934a1f19d678a1377c257b9a780e80-Paper.pdf .
- Ni et al. explicitly discuss useful self-predictive learning with stop-gradient/EMA; their paper does not recommend
  observation prediction universally. In distracting tasks their observation-predictive methods degrade faster:
  https://arxiv.org/html/2401.08898v3 , sections4 and5.2.

Specify target, negative distribution, action/time conditioning, encoder gradient routes and anti-shortcut controls
before making a contrastive score a gate or training objective. A learned-variance likelihood is not the only other
choice. Small predictive reward heads are a simpler initial candidate than a wholesale loss replacement.

## 5. Repair the proposed measurement definitions

- **Exact digit vs decision relevance:** 34% exact food readout is a concern, not proof that starvation cannot be
  recognised. Low/zero food could remain separable even when other digit values are confused.
- **Readiness adds little:** .793 current-frame AP to .872 belief AP is a useful gain, and the actual hit probes
  rise .471 -> .700 awake and .634 -> .807 asleep. These are specialised probes, not proof of global native use, but
  do not minimise positive memory evidence.
- **Beyond-current-frame gain:** b+current-frame outperforming current-frame demonstrates extra history information;
  it does NOT show b retains the current frame. That test could miss the very food-retention defect being discussed.
- **Innovation:** for a correct conditional mean, the signed innovation has conditional mean0. Its magnitude or log
  score may remain predictable because uncertainty varies with known state. Example: full state A=0/1, Y|A Gaussian
  mean0, variance1/9. Squared error expectation1/9 is predictable despite sufficient state. Also a biased decoder can
  give predictable signed errors when the representation is perfect. Residual predictability must be compared with
  a correction from b itself before attributing improvement from extra history to missing representation.
- **Decision sufficiency:** S2 real returns remain important, but failure combines representation, world generation,
  planner/actor optimisation, data support and horizon. It is an end-to-end outcome, not unique localisation to b.

## 6. Fovea/brain points

Support not treating absent fine evidence as the explanation for already-readable HUD facts. Redundancy of current
information does not exclude a teaching/grounding benefit from looking, nor establish perfect night-entity vision.

Preserve the grounding role of the existing TSP: LOSSES principle7 assigns it ownership of periphery features.
Removing its current-periphery input changes that gradient route; a new predictive loss should not silently remove
the existing fixed-target grounding mechanism.

Current chooser timing is coarse first, then gaze, then fine crop (TEACHING14). A new belief-only glimpse forecast
needs explicit time indices: is it before the next coarse observation, or after assimilating current coarse vision?
A gaze chosen using the target frame's coarse view can itself reveal that frame. Do not reintroduce the old gaze
cue shortcut by calling that gaze an earlier action. Existing TSP properly conditions on the coarse evidence used
to choose the gaze.

Predictive remapping is evidence of anticipatory changes in retinal spatial reference, not proof of this particular
pixel loss: https://pubmed.ncbi.nlm.nih.gov/1553535/ . Leaky-integrator brain models make SSM counting plausible,
not a proof that capacity, readout, compression and optimisation are irrelevant in this trained model.

## Recommended baseline plan

Proceed to a plan, not a chosen loss or architecture change. Fix the actor-input/state definitions first. For a
single bounded baseline, compare future outcome prediction from b alone, recent observations alone, and both,
with matching action/observation availability and episode-held-out fitting. Include current critical-state retention
so 'extra history beyond the frame' cannot conceal losing the frame itself. Multi-step action support must be
explicit. Keep task-specific timer/danger spot checks and native generation alongside generic scores.

A representation is implicated when extra history improves a suitably matched predictive reader beyond what b
supports. If b supports a good forecast but the native model fails, the fault remains downstream use/training.
If both fail, the result is unresolved without a stronger reference. No metric should pass merely by discriminating
scene identity; a correct heteroscedastic stochastic model must pass any proposed innovation gate.

Approve direct pre-outcome reward/continuation supervision as a serious next candidate. Do not yet approve generic
CPC as the fix, exclusive attribution to the objective, or a claim that improving an auxiliary head guarantees
better generated damage or P3 performance. No new full run was launched.
