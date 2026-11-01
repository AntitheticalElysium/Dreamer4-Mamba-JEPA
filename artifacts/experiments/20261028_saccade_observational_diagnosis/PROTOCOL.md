# Observational diagnosis — 2026-10-28

Scope: understand causes/mechanisms and whether the reported symptoms are consequential. No interventions, fixes,
new world training, P2 or actor. Do not change inputs, fixations, tokens, histories, sampler settings, noise schedules,
losses or model weights. Fresh post-hoc readers and mathematical verification are measuring instruments only.
The previous crop-branch intervention is not used in these tests. Final split remains untouched.

## Ordered questions and interpretations (before results)

1. **Density versus displayed median.** Build exact HUD digit templates from the original probe-fit episodes, with
   physical-episode holdout and raw-image agreement checks. On unchanged real S1 crops, score the legal digit images
   under the decoder's COMPLETE predicted discretised logistic distribution. Report relative digit ranking,
   conditional probability given a legal glyph, and total probability assigned to the legal glyph set on the
   coordinates that vary between digits. Report all four body fields by value/light. A constant median glyph is
   not itself a failed distribution. Conversely, good conditional ranking with negligible legal-glyph probability
   does not establish coherent image sampling. Use each checkpoint's own perception/decoder for snapshots 1, 5, 10
   on the fixed 8,192 time-course tuples. No cross-checkpoint latent reads. Validate normalisation and derivatives
   of the implemented likelihood against float64 formulae at captured outputs. No gradient updates.

2. **Meaning of state-readout errors.** Reuse the fixed body readers and their actual real-path predictions. On living
   states, report signed error, zero/nonzero and low-value confusion, false-safe/false-alarm rates, by kind of body
   quantity, depth, behavior mixture, light, recent changes and proximity to threats. Exclude terminal health=0 from
   the living-health danger test; termination is itself an input. Compare full percept, actual projected input and
   corrected state. This assesses accessible distinctions, not whether an actor would fail.

3. **Naturally occurring glimpse exposure.** Use ONLY the recorded random S1 fixations. Compare readout errors when
   a field happened to be inside/outside its actual crop and by time since the last such crop. Standardise exposure
   comparisons by true value, light and behavior mixture; report positivity/coverage. Follow observed errors around
   actual glimpses without synthesising alternate crops or changing the recurrent path. These are natural-exposure
   associations; field inclusion changes the entire crop and is not a field-only causal manipulation.

4. **Generated-reader support.** Build a real-token nearest-neighbour reference using probe-fit episodes only, with
   train-derived scaling. Keep a fixed natural-frequency reference sample (300k map cells maximum). Calibrate
   distances/margins and label error on the original physical-episode holdout before reading generated samples.
   Query fixed samples from the original 30k production-sampler tuples; do not regenerate with different settings.
   Report real/generated reference distance, abstention coverage, and agreement between the frozen neural reader
   and reference labels in reliable/ambiguous/unsupported regions. Held-out reliability on real tokens is not
   a guarantee of generated semantics; unsupported samples remain unclassified, not forced to background.

5. **Training trajectory and incentives.** Inspect already recorded loss/gradient/normalisation/optimizer statistics
   and saved weights. Use unchanged real samples and captured decoder outputs for derivative analysis, partitioning
   existing losses by pixel role/value. Locate where a readable quantity becomes hard to recover along the frozen
   adapter/decoder pathway only if test 1 leaves a semantic-density deficit. Gradient magnitudes/signs describe
   current local incentives; they do not prove a causal explanation of the complete training run. No simulated
   optimizer step, gradient surgery, loss reweighting or hypothetical model repair.

## Stopping and evidence

- Use all available natural dev states for reuse-based summaries; use the already fixed S1 tuple panel for costly
  forward work. Two existing reader seeds, physical-episode bootstrap uncertainty; one world seed only.
- No threshold or hypothesis is silently changed after results. Failed controls remain in the report. Additional
  measurement questions are written here before execution. No automatic continuation into world/actor training.
- Learning curves, proper scores and thresholds are interpreted in their own target spaces. An actor-value verdict
  requires an actor test and is outside this turn's scope; consequence calibration can supply narrower evidence.
- Opening/closing checkpoint/source/reader hashes must agree. All outputs stay in this investigation directory.

## Follow-up A — distinguish danger information from multiclass-reader errors

The reused multiclass readout misses many living low-health states. Before calling that absent information, fit
direct binary readers of health<=2, food=0, drink=0 and energy=0 on unchanged living real states. Compare full percept,
the projected token and corrected state; one field per reader, two seeds, same physical splits/MLP/AdamW/30-epoch cap
as earlier readers, no rebalancing. Primary AP versus prevalence; also threshold .5 and a threshold chosen on held-out
training episodes for 95% recall. Report dev recall/false alarms; do not claim an actor fails from a reader error.

In parallel, read the ALREADY SAVED production-sampler reward and termination samples, with contexts stratified by
the preceding real health/food/drink, threat presence, light and behavior mixture. Compare event rates and mean reward
against the matched recorded successors, with physical-episode paired uncertainty. No new sampling or recipe changes.
These test conditional one-step consequences, not free-running or actor value.

## Reference-support implementation details

Fixed seed 447; maximum 300k natural-frequency reference map cells from the original fit episodes, 100k from the
physical-episode calibration holdout, and 4,096 systematic tuples from the existing 30k production samples. Player
cell excluded. Exact k=8 nearest neighbours in (a) periphery tokens and (b) periphery+lighting, each block scaled by
its reference RMS variation and number of channels. No transform is fed back into the model. Class = nearest
reference label. Support requires distance <= its predicted-class calibration 95th percentile and >=7/8 neighbour
agreement; classes with <50 calibration predictions use the global threshold and are marked low-power. Show real
calibration/dev error and coverage by class before any generated interpretation. These are reference-support and
reader-agreement measurements, not semantic guarantees on generated content.

## Follow-up B — trace the native outcome discrepancy without changing observations

The saved production samples contain far fewer deaths/negative rewards than recorded successors. Recompute the
native outcome probabilities for the same COMPLETE real and generated observations, each with its own reward (real
reward or already saved generated reward) and unchanged preceding belief. No mixed bundles or substituted parts.
Read periphery HUD health with the existing real-calibrated local reader; add a field-specific nearest-real-token
reference with the same fit/holdout policy. Report generated support before interpreting its label. Check whether
health-zero evidence is absent, unsupported, or present but not followed by native termination. Primary context
strata depend only on the preceding real state. Fatal-successor strata are diagnostic positives, not requirements
that a particular stochastic sample reproduce that fatal successor.

## Follow-up C — explain scale geometry, not just its magnitude

For the single discretised logistic with absorbing 0/255 bins, derive its endpoint-mass limit as location/scale
grow together. On captured outputs and snapshots 1,5,10, measure endpoint versus interior mass, per-pixel roles,
location/scale ratio and local radial derivative; no perturbation or update is applied. Verify probability
normalisation at captured parameters. A large underlying logistic scale need not imply broad uncertainty over
all bounded RGB values. Trace the learning path with each snapshot's own perception/decoder on the fixed 8,192
tuples. Only if the semantic density remains weak, fit fresh readers of frozen adapter/decoder activations to
locate the last readable stage. These are information/readout measurements, not evidence that changing a stage
would repair training.

## Follow-up D — observed training trajectory of native predictive risk

Evaluate the ACTUAL saved models at checkpoints 1,5,10, using each model's own real-history encoding and native
unchanged production sampler. Fixed panel: 1,024 systematic original S1 tuples, all original contexts with preceding
health<=2 and an adjacent zombie, plus at most 512 randomly chosen preceding-food-zero contexts (seed 1102). No
successor-based selection. Four independent draws/history by repeating tuples in the native sampler call; the
sampler settings and internal RNG recipe are unchanged. Teacher-forced outcome probabilities on the matched real
successors supply the positive control at each checkpoint. Report population and the context-defined subgroups
separately. This is a longitudinal measurement of actual training snapshots, not a parameter ablation; changes in
both encoder and heads prevent attributing the trajectory to one component alone.

## Frozen activation readout details

The semantic-density deficit persists. Capture actual selected HUD cells from the unchanged real paths at these
stages: glimpse token, reconstruction adapter, pixel-mean hidden layer, location output, clipped-location display,
location+log-scale, endpoint CDF logits, scale-only context control. Fit common post-hoc MLP readers (256-256,
AdamW1e-3/wd1e-4/batch4096, original physical splits, two seeds, <=30 epochs with original patience). Features
are standardised using fit-episode means/stds only (std floor 1e-6), padded to 294 slots; every arm also receives
lighting and cell position. No transform is fed into the world. This asks whether a quantity remains readable,
not whether the world uses it correctly. Glimpse-token positive-control failure invalidates a negative layer-loss
interpretation. Microscopic readable variations are not evidence of a calibrated observation distribution.

## Follow-up E — probability readout versus rounding

The direct binary health reader fails the full-percept positive control. Retain that failure. Extract critical-state
probabilities by summing the appropriate classes of the EXISTING multiclass body readers, without refitting them.
Report all three representations and both reader seeds, AP and fixed .5 plus validation-95%-recall thresholds.
Check argmax identity against their saved predictions. This separates loss of exact rounded values from an
accessible danger-ranking signal; no reader or world-model selection is made on dev performance.

## Follow-up F — observe the native sampler's input distribution

Use a read-only forward hook on the unchanged production denoiser to record noise level, mean-direction projection,
RMS and clean-estimate geometry at its three existing evaluations. No added noise, altered settings or interpreted
intermediate entity counts. Fixed original population/risk panel; verify hook/no-hook identity on 64 prescribed
tuples first. Compare the actual initial Gaussian's mean-direction projection with the distribution of real latent
targets plus the same noise variance analytically. This measures the finite-noise-start approximation; a mismatch
does not by itself prove it caused the risk deficit, and no alternate sampler is run.

## Follow-up G — local shared-decoder gradient accounting

On 1,024 fixed natural frames from the probe-fit episodes (seed 1901), decompose the unchanged TSP+reconstruction
loss gradient into its two routes and into body-value-varying pixel coordinates versus the remaining coordinates.
Use the original per-RGB averaging in every component. Record norms, inner products and cosines on the shared mean
and scale parameters; no loss is removed from training and no step is applied. This can reveal local competing or
weak pressures at the current checkpoint. It cannot identify the actual Adam update or the cause of the whole
training trajectory. Keep the existing checkpoint route-gradient logs as longitudinal context.

## Follow-up H — actual carried training state versus actual fresh-evaluation path

Reconstruct the exact recorded current-episode fixation histories of all 16 streams at checkpoints 1,5,10, requiring
stream-state equality with the saved artifacts. Compare their stored predicted beliefs/carries with fresh encoding
of the SAME real histories/fixations under that snapshot, as evaluation does. No alternate state is fed into a
future prediction and no input is changed. Report depth and reset cases; this is a small mechanical compatibility
sample, not a rare-risk population result. Differences include the intended parameter-history lag of TBPTT and
numerical execution differences, so do not label them a defect or a causal explanation without further evidence.

## Follow-up I — observed threat/consequence co-occurrence

Within the same saved real and production-generated observations, tabulate native negative-reward/death
probabilities by the existing reader's current adjacent-zombie evidence, conditioned on preceding real context.
No factor is swapped or forced. This is an associative joint-consistency check; generated entity labels retain
the reference-support limitations, and no causal fraction of the risk deficit is assigned to missing entities.

## Follow-up J — temporal shape of the existing readout errors

On the unchanged real path, compare current predictions with actual levels at fixed lags 0,1,2,4,8,16,32,64, using
the SAME living depth>=64 cohort for all lags. Also align errors around naturally occurring increases/decreases,
with per-lag exposure counts and no future-outcome filtering; unavailable/terminal follow-up is omitted and reported.
This tests whether the finite readouts resemble lagged/smoothed values. It does not prove the recurrent state itself
is delayed, and no lag is selected as a new model setting.
