# Frozen F1.5/F1.6 information-accessibility audit

2026-11-03. User authorized launch after discussing this design. No world,
encoder, fusion, Mamba, loss, sampler, corpus or chooser update. Learning-pressure
is a candidate explanation, not an assumed cause or an intervention in this audit.

## Primary: future-fine predictive accessibility

Repeat the previous F1 predictive task on BOTH frozen selected checkpoints:
next fine crop at the CURRENT observed fixation. Baseline sees full current coarse
scene/light, prior belief BEFORE this look, eight physical frames and their
prior beliefs/actions/outcomes, arrived reward/termination, next action and gaze.
No future percept/outcome enters inputs; no future coarse conditioning is added.

Six supplements, padded to1024, identical MLP256-256 fine predictors:
zero baseline; raw9x64 fine;9x64 local joint tokens; ACTUAL complete256D mixed
embedding from native BF16 execution on the cached observations; native post-action
emission; actual projected memory. F1.5's local tokens are its raw fine tokens,
so that identity arm aliases raw exactly, without a redundant fit. F1.6's local
tokens use its real nonlinear fusion, not the old raw linear slice or SVD inverse.
Store local/mixed transformations inFP32; native observed/emitted caches remain
the prior studies'FP16. Disclose this numerical convention.

Two seeds611701/611702;2400 AdamW steps,batch512,lr.001,wd.0001,clip1;
same head initialization and frame draws across supplements and worlds. Same
train400/dev400 physical episodes and uniform look sequences. Fit first360,
validate last40; exclude world-monitor episodes. Main cohort: night, entity in
current crop, no next scroll. Report whole next crop, current entity cells and
their one-ring neighborhood; also day/entity, night/entity, night/empty and all.
Episode-bootstrap95% intervals, diagnosis dev0:250 and disjoint confirmation
250:400. Final split untouched. Relative gains are within-world, beside raw-fine
and copy-current references; cross-world token MSE is not a fixed semantic unit.
Check old F1.5 base/raw error arrays for exact numerical reproduction.

## Secondary: fine-specific current information under genuine alternative looks

A supplementary supervised diagnostic reader is required because an improved
future predictor with a memory supplement could merely read coarse information
more easily. The original prediction task ties future query to current fixation;
do NOT feed an alternative crop at incompatible coordinates into that predictor.
Instead use the established conditional current-entity reader, whose queried cell
and observed fixation are distinct inputs and whose training supports inside and
outside-crop queries.

Same full-coarse/history/prior/action/outcome baseline, queried coarse token and
cell identity; compare zero, queried raw fine token, queried local joint token,
mixed input, post-action emission and memory. One query-gated reader family,
two seeds,2400 AdamW updates,batch1280 (equal5 kinds x inside/outside x128),
lr.001,wd.0001,clip1. Labels train disposable diagnostic readers only. Worlds
frozen. One family is a bounded accessibility instrument, not an information bound.

Use the previous fixed58 night scenes/48 episodes and original source cells,
informative-look/entity-free control weights with matched fixation row. For ALL49
real crops, replay the identical full prior and correct the native belief with the
actual crop, fixed action and arrived outcomes. Source targets never use future
prediction gains. Evaluate true-kind log-probability contrast informative-minus-
control, subtracting the same contrast in the gaze-aware baseline reader. Report
both reader seeds and positive raw-fine control; do not interpret a failed raw
control as evidence of retention failure. Keep all scenes, no outcome-based
selection. Selection by old fine readability can favor F1.5; these are secondary
source-defined cases, not population rates. Native replay must match cached
post-action emission within.03 RMS, as in previous qualified instrumentation.

## Mechanics, scope and stopping

Verify cached physical episode/time/fixation parity, strict checkpoint hashes,
actual nonlinear/complete-embedding calculations, target exclusion, stage alignment,
gradient-connected predictors and all world parameters frozen. Smoke BOTH
predictor families and both worlds before launching full fits. No budget ladder
or scientific-success checkpoint selection. Retain raw errors, weights, source
pins and logs. Single GPU job at a time; never terminate another agent's job.

Positive readout gains establish accessible predictive information in that
instrument. Nulls do not prove erasure. The primary forecast does not supply
future coarse tokens, unlike native cascade glimpse generation; its gain is not
a promised native-G1 gain. This does not causally test learning pressure or
learned fixation. It locates which preservation/use hypothesis deserves discussion
before any further training change. No automatic P1-F2/P2/actor launch follows.
