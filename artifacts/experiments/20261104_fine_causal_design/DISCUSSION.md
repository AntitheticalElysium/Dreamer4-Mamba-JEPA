# Consequence-specific fine-evidence test: discussion, not an approved run

Historical pre-launch discussion. The user subsequently authorized execution;
see `PLAN.md` and the completed `RESULTS.md` (2026-11-04). The status statements
below describe the earlier discussion, not the present run status.

2026-11-04. User requests one decisive experiment rather than another reader/ablation
ladder. F2 evaluation is explicitly out of scope and has not been inspected.
No GPU job, model inference, fitting or new experiment queue started. One tiny
CPU renderer-input sanity check is retained; its narrow scope is below.

## What the earlier tests did and did not establish

`20261103_belief_information/paired.py:14–40` picks64 no-scroll scenes by day/night
and map/HUD future-query location. It compares current crops overlapping the
future query to row-matched nonoverlapping crops. Native scores are denoising
errors, not a named future consequence. This genuinely varied real observations
and fixed prior/query/target; it did NOT require a cue to distinguish future hit
from no hit, or prove periphery/history lacked that cue. Its positive controls
establish practical next-fine prediction value, not unique consequence knowledge.
The preceding58-scene entity test qualified fine legibility/current-source-local
regions but also did not establish that exact semantic consequence contrast.

This limits the preceding localization claim: useful readout performance worsens
along the route, but we have not identified a fact-specific causal collision or
proved one architectural component is responsible for failed consequences.

## The actual computation and collision claim

`model.py:227–248`: a NONLINEAR local joint transform first combines each observed
fine token with its corresponding coarse token and lighting. Those nine64D joint
tokens are scattered into the fine canvas, then combined linearly with the
coarse grid, lighting and fixation mask into256D. At a FIXED gaze and fixed other
inputs, e_A-e_B=A_f(u_A-u_B), with u the576D JOINT crop, not raw pixels.

All49 trained A_f matrices have rank256 in the previous exact algebra, so their
nullspaces have dimension320. That establishes possible collisions in ambient
joint-token space, not collisions between real informative observations. The
physical scene/token manifold may not occupy arbitrary null directions.

If two valid alternative facts yield identical actual embeddings, with identical
prior carry, actions/outcomes and every other input to this route, that pair
cannot be distinguished through this route. Check actual native precision and
an FP64 reference separately: equality due to rounding and an exact linear
nullspace collision are different explanations. Unequal vectors establish only
that some distinction survives; they do not certify all semantic facts survive
or are usable. At different gazes, the A_f and fixation-mask offset both change;
it is incorrect to use one fixed matrix or attribute every delta to fine content.

`model.py:Denoiser.embed/Block/glimpse_step`: raw fine evidence ALSO enters the
recent fine temporal-cache route. The factorized physical prediction cannot read
those fine keys directly. Mixer equality therefore does not prove all future
fine predictions/rewards/actor behavior ignore the glimpse. It can block this
memory route while a direct recent-fine route still works. At later generated
steps, generated fine observations may feed back into belief as well.

Mamba's emitted256D vector is not its full persistent conv/SSM carry. Equal
emissions alone do not prove storage loss. The full carry and later emissions
must be distinguished. The existing full-rank memory expansion is not a new
exact-arithmetic compression bottleneck.

## Recommended single experiment, conditional on source qualification

One semantic paired-evidence experiment on frozen F1.6, with multiple independently
qualified pairs. No new learned probes, no layer sweep, no architecture update.
The task is to predict ONE declared physical consequence under a fixed action,
from a fact observed at the cue time. Begin with a one-step target so poor P2
self-feeding cannot explain a failure in this P1 model.

An arrow heading toward the player versus away is a plausible candidate, not yet
a qualified case. Verified installed code: renderer.py324–369 makes orientation
visible; game_logic.py1157–1209 moves by arrow_directions and subtracts2 health
on contact. Old presence/kind acuity labels omit direction (PERCEPTION.md106).
However, the CPU sprite sanity check finds ALL four directions remain distinct
in a clean3x3 pooled example. Hence direction is not automatically fine-only.
Night/learned-token conditional ambiguity must be demonstrated, not assumed.

Before paying for the native paired test, the sources must establish:

1. The cue changes the named future event under fixed action/appropriate simulator
   replay controls. A cue merely being near a later event is insufficient.
2. Fine evidence genuinely resolves ambiguity left by the common coarse/prior
   evidence. Being missed by one weak reader is insufficient proof of absence.
3. The cue is visible at high resolution; the target fact is not an invisible
   simulator field such as cooldown/health without a visual basis.
4. Native model/head/instrument can express the consequence on a legitimate clear
   reference. If the causal behavior is absent even there, a routing null cannot
   localize its missing implementation. That is a declared task-capability result.

Prefer paired real/valid rendered alternatives at the SAME gaze, differing in
that cue. This controls gaze-mask/position shortcuts. Do not transplant a clearly
contradictory crop into a confident coarse scene. If only approximate coarse
matches exist, disclose residual confounding; do not claim a pure fine effect.
Do not manufacture a novel masked sensor or fake symbol and call its failure a
native-world failure. If adequate pairs cannot be qualified, STOP before model
ablation: the available task does not yet support the claimed diagnosis.

Replay common history and matched randomness. Score event probability/correct
versus alternative event support, not whether one random sample differs or
whole-frame latent MSE changes. Never supply the actual post-consequence frame,
reward, termination or future coarse evidence before asking for the consequence.
Use the existing fixed physical reader only if its error is independently bounded
on these real cases. Sample count, practical effect threshold and case budget
must be declared after source feasibility/cost and before model outcomes.

Capture the original cue tokens, local joint tokens, complete projected input,
full Mamba carry, emitted belief, exported memory and native consequence output.
Run the genuine full route first. If path attribution is needed, use only the two
architectural routes as a predeclared crossed comparison: belief update versus
fine cache. A direct-pixel/reconstruction read is an acquisition/measurement
control, not evidence the causal prediction mechanism already works.

## Readings fixed before any launch

| Observation | Permitted conclusion |
|---|---|
| No qualified coarse-ambiguous, fine-resolved consequence case | Current test/task lacks the needed premise; do not diagnose memory. |
| Clear reference cannot express the consequence | Native consequence competence is absent for this task; no unique mixer localization. |
| Qualified fine cue changes native event probabilities correctly across both alternatives | Functional fine use for this capability is demonstrated. |
| Relevant alternatives collapse to the same complete embedding, all other route inputs equal | Exact pair-specific acquisition loss on the memory route. |
| Embeddings differ but full post-cue carry and emission collapse, controlled other inputs | This native update cannot carry that pair's distinction forward. |
| Carry differs but emitted/consumer inputs do not yet distinguish the cue | Export is the current access limit; do not call the carry empty. |
| Internal states differ but native consequence does not improve | Fine-specific semantic use was not demonstrated; no claim of unique compression/entanglement/objective cause. |
| Benefit only through recent fine cache | Direct use, not demonstrated Mamba retention. |
| Only raw vector magnitudes change | Sensitivity, not correct consequence knowledge. |

No design can promise a unique diagnosis for every possible null. The aim is to
make failures informative by declaring exactly what they do and do not exclude,
and to stop before an unqualified contrast becomes another architectural patch.

## Research basis and limits

Zhang & Nanda, ICLR2024, Towards Best Practices of Activation Patching, sections3–4:
choice of input corruption and output metric can change localization conclusions;
semantically valid matched alternatives are preferable to arbitrary noise when
that is the intended hypothesis. https://arxiv.org/html/2309.16042v2

Heimersheim & Nanda2024, sections2–4: distinguish necessary/sufficient path tests,
backup routes and the specific behavior a patching contrast actually addresses.
https://arxiv.org/html/2404.15255v1 . These are methodological precedents from
language models, not evidence Saccade will pass this task.

Freeman & Simoncelli2011 use physically different stimuli with matched model
responses (metamers) to test lost perceptual distinctions. This supports the
logic of collisions; constructing arbitrary token-nullspace vectors is not the
same as producing valid Craftax counterfactuals.
https://pmc.ncbi.nlm.nih.gov/articles/PMC3164938/
