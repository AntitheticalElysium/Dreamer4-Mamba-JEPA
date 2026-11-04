# Bounded fine-evidence / belief / consumer diagnosis

2026-11-03. User authorized launch, explicitly requiring the broad world-model
question rather than an entity-only optimization. No world or chooser training.

## Decision this campaign must resolve

Does a genuine fine observation supply useful prediction information under the
native consumer's conditioning, and is that advantage accessible after the
input projection and native belief update? Entities are one secondary region;
whole predicted percepts, map/HUD query regions and day/night are also reported.
No claim that finite negative readers prove information-theoretic erasure.

## One experiment, three joined readings

1. Train a query-aware predictive reader on frozen F1.6 percepts. The physical
   head sees the existing full coarse/history/prior/action/outcome baseline.
   The fine head additionally sees the older eight real glimpses and their gaze
   coordinates, plus next coarse/light with one fixed independent sigma0.1
   perturbation per frame (the native cascade information set). This same
   perturbation is shared by every reader arm and the paired native fine test.
   Future fine QUERY is independent of the current observed gaze, so genuine
   alternative looks are supported rather than queried out of distribution.
   Train queries: half current gaze, half independent uniform; dev queries use
   the same declared mixture. This is a diagnostic distribution, not a policy.
2. Compare zero, whole raw fine, whole local joint, actual mixed input, retained
   joint recovered from actual BF16 mixing, and post-action belief. Memory is
   a full-column-rank linear expansion of that belief, not another claimed
   information bottleneck. Use train-only per-channel normalization of each
   supplement and separate context/side processing with query-aware gating.
   Every arm uses identical architecture, initialization and training examples.
   The two heads are independent: future coarse never enters physical inputs.
   A zero-initialized physical correction can express current-coarse copying.
   Fine targets are next raw encoder tokens at the independent query.
3. On64 fixed real dev scenes (16 each day/map, night/map, day/HUD, night/HUD;
   query defines map/HUD), replay the complete SAME prior with all49 genuine
   current looks. Hold future query/action/target fixed. Read the same predictors
   and the frozen native denoiser from each consistent belief/context. A future
   query is not changed when the present look changes. Main contrast: present
   crop overlaps future query versus row-matched disjoint crop, no next scroll;
   preserve all scenes and report scroll separately. Subtract the zero-reader
   contrast to remove gaze-only effects. No selection using future loss.

## Fixed scope

Selected frozen F1.6ck10 SHA
34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045.
Existing400 train/dev episodes; first360 fit, last40 validation, monitor excluded.
Dev0:250 diagnosis /250:400 existing disjoint confirmation; final split untouched.
Two seeds612101/612102, six arms,2400 AdamW updates each, batch256,lr.001,
wd.0001,clip1. Width256, separate heads and context/side routes, fixed final
weights. No fitting extension or post-result target/threshold change. Target
channel scales measured on fit data only. Raw/current-copy references retained.
Input standard deviations are floored at0.01; target channel scales at0.05.
Context and side features are normalized separately. Physical/fine parameters
and gradient clipping are independent, so future coarse in the fine head cannot
affect the physical head through shared parameters or clipping. Auxiliary error
is channel-standardized latent MSE; native error is EDM raw-output training MSE.
Compare benefits/signs within each measurement, not their absolute units.
Native checks use eight pre-existing noise levels, two shared draws; physical
and fine heads scored separately. Fine denoising conditions on next coarse/light
at the declared cascade level0.1. Consistent real caches include the current
look; physical attention cannot read the fine cache. No sampler changes.

## Qualification before interpretation

- Raw-fine predictor must improve the SAME target/cohort versus zero, with a
  positive episode-bootstrap95% interval in both seeds. Otherwise downstream
  nulls cannot be called failures to retain useful fine evidence there.
- Genuine-look raw-fine contrast must improve prediction in both seeds before
  downstream look contrasts are interpreted as fine-specific. Aggregate belief
  gains alone may merely reformat coarse information.
- Actual nonlinear fusion/mixed outputs and full-prefix native replay qualified;
  no future fine or future outcomes in inputs; future coarse only in fine head.
- Same-context controls, gradients, strict source/checkpoint pins, normalization
  computed without dev data, train/dev episode disjointness, and unchanged world.

## Predeclared disposition and stop

- Raw fine adds no qualified gain with native conditioning: no demonstrated
  missing predictive value for that target. Do not repair memory to force it.
- Raw fine helps but joint/mixed/recovered do not: input acquisition interface
  remains the first practical accessibility failure; not proof of erasure.
- Mixed/recovered helps but post-action belief does not: native belief's usable
  acquisition/storage/emission is the next failure, conditional on qualification.
- Belief has a qualified fine-specific advantage but native consumer does not:
  consumer utilization/training pressure is implicated for that target.
- Native fine benefits but physical does not: distinguish direct fine-history
  use from durable belief use. Do not call this a memory repair.
- Instruments disagree or positive controls fail: explicitly unidentifiable;
  no architecture/overnight run justified by a null. Preserve this outcome.

One collection,12 reader fits, one paired native/readout battery, one report.
Stop for discussion; no followup ladder, architecture patch or successor job.
The aim is an actionable routing diagnosis, not a promise of a unique cause
regardless of what the evidence permits.
