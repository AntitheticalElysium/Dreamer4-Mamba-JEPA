# Frozen P1/S1 dissection — 2026-10-28

User scope: understand the numbers and fovea behavior. No model/training/evaluation fixes, P2, actor training, new world
seed or deployment change. Checkpoint: world 1, ckpt_10.pt; saved S1 reader. Canonical sources and checkpoint are hashed
before/after. Post-hoc readers are measurement instruments; their training never changes the world model.

## Questions and separating measurements (declared before diagnostic results)

1. **Where did it look?** Reconstruct all 25.6 M training fixation records from the original Streams/seed/chunk schedule
   without loading images, and require exact stream-state agreement at all ten saved checkpoints. Measure crop-centre
   and cell coverage, HUD-field exposure and coverage in the labelled train-400 subset. Reconstruct actual S1 fixations
   on all dev episodes; measure exposure conditional on entities, night, damage and HUD values against the analytic
   uniform-prior geometry. Source says uniform random: no learned fixation policy exists in P1.
2. **What did the fovea encode and predict?** Score the existing TSP and fovea reconstruction routes on the SAME real
   crop pixels, by cell kind, map/HUD, HUD field, light and value. Both routes share exactly the same decoder scale;
   check this rather than attributing likelihood gains to mean prediction by assumption. Report NLL and clipped-median
   RGB error; inspect preselected raw/reconstructed/predicted examples. Train fresh matched readers on real periphery,
   fovea, joint features, low-resolution pixels and full-resolution pixels (same cells, labels, architecture, splits,
   seeds, budgets). Missing feature slots are absent in BOTH fitting and evaluation, never an inference-only ablation.
3. **Belief vs readout:** compare real full percept, the actual 256-D projected percept+outcome token, and the actual
   256-D corrected Mamba output. Projected-token/Mamba comparisons have identical input dimension and reader capacity.
   Fit joint, body-only, inventory-only and recent-sighting readers, two seeds, same initialization/order, 30-epoch cap.
   Recover the original S1 joint stopping rule as a reproduction check; fixed-budget task-selected snapshots separate
   reader objective/selection from representation. Report exact values, MAE, within-one errors, rare-value/balanced
   recall and absence AP, not just aggregate accuracy dominated by zeros. No failed probe is an information bound.
4. **One-step generation vs its reader:** first reproduce S1's 30k selected tuples and frozen-reader counts. Save the
   samples. Compare real/generated token mean/RMS/range geometry and a geometry-only classifier. Use independent
   fovea/pixel readers only after measuring their performance on known-label real reconstructions; agreement is
   triangulation, not simulator truth for generated latents. No latent renormalization or token replacement as a fix.
5. **Frozen sampler diagnosis:** fixed S1-derived population and past-entity cohorts, not selected on successor labels.
   Compare production churn=1/3 steps with churn=0/3 steps; separately compare churn=0 at 3, 8 and 32 steps using common
   initial noise and identical context noise. These identify a churn effect and deterministic integration effects
   separately; the two-axis production-vs-32 contrast is not assigned a single cause. Use four noise draws per history.
   Denoising known noised successors at fixed sigma is scored beside clean/raw-noisy controls; an intermediate
   conditional mean is NOT required to preserve a sampled world's counts, and reconstruction is not forecasting.
6. **Was glimpse evidence used?** If reader calibration is adequate, branch identical real histories at 64-frame
   boundaries with two valid current fixation choices: HUD target covered vs another crop in the same row (same HUD
   area). Periphery/light/history/actions/outcomes stay identical, new crops are real, and both choices are within the
   training prior. Compare corrected-state readouts immediately and through 16 subsequent real observations. Include
   unaltered-fixation parity; failure of this control makes the branch comparison uninterpretable. This measures a
   passive sensor-placement effect, not learned attention, actor value or closed-loop decisions.

## Data, readers and stopping

- World-train corpus is untouched; train-400 labels fit diagnostic readers, its original episode-based holdout selects
  epochs, all dev labels evaluate. Final-split observations/labels remain untouched. These are exploratory diagnostics
  on the existing dev set, not new confirmatory gates or a checkpoint search.
- Population generation: the exact original 30k S1 history tuples. Sampler panel: 1,024 fixed systematic population
  tuples plus up to 128 per kind/light selected using PREVIOUS-frame entities (no future-outcome selection); four
  draws/condition. Report conditional/cohort results separately from population counts.
- Reader comparisons: same labelled crop instances and physical-episode partitions; two reader seeds; MLP 256-256
  GELU, AdamW 1e-3/wd1e-4/batch4096, at most 30 epochs. Report both seeds and paired physical-episode uncertainty.
  Raw pixels and known-label reconstructions are positive controls. Failed calibration or nonconvergence is recorded.
- One GPU task at a time; CPU metadata work may run concurrently. No result extends model training or changes the
  checkpoint. Additional diagnostics require an explicit written amendment before execution; retain failed controls.
- Unsupported/empty strata are unmeasured. No claimed causal explanation from a loss gradient, visualization, probe,
  or attention map alone. Each report distinguishes observations, supported localization and unresolved explanations.

## Amendment A — prompted by the paired decoder measurements, before further decoding diagnostics

The initial summaries found very large HUD logistic scales and substantial mean-image error despite low NLL.
Inspect actual raw HUD pixel endpoint frequencies, location/scale distributions, and per-coordinate likelihoods.
Compare both learned routes with a zero-location logistic at the SAME predicted scales to determine whether the
reported likelihood is sensitive to the decoded location on these pixels. This is a scoring diagnostic only, not
a proposed loss or model fix. Inspect predefined real examples and known-label real reconstructions before using a
pixel reader on generated reconstructions. Also verify the custom sampler against the unchanged S1 function on a
single prescribed batch with identical random draws before interpreting its controlled step/churn comparisons.
The copy-previous location is also scored with that exact same learned scale on the same targets. This comparison
isolates a change of predicted location from a change of likelihood scale; it is not a newly optimised baseline.
For the fixation branches, use an untouched third branch with the original fixation. Compare it both with the
unbranched chunk and the original full-episode cache. Require >=99.5% agreement of HUD reader decisions and feature
RMSE <0.002 (allow max absolute BF16 rounding error up to 0.02); otherwise retain the measurements but make no sensor
effect claim. Choose the targeted HUD field by episode/boundary index, not by outcome or reader performance.
The geometry audit also checks the analytic encoder constraint: applying inverse tanh to a real output token
recovers a layer-normalised vector (approximately zero channel mean and unit channel variance). This is a
measurement of support mismatch only; it is not applied to generated tokens as a correction.

## Amendment B — frozen checkpoint time course

To locate the reconstruction/scale behavior in training, take 8,192 systematic tuples from the fixed S1 sequence,
independent of labels and errors. Re-encode their actual frames and actual fixations with EACH snapshot's own
perception, fovea adapter and pixel decoder (initialization and all ten saved checkpoints). Measure paired fovea
reconstruction NLL, RGB error and scale by map/HUD and HUD field. No stale-token cross-checkpoint reads, reader reuse,
checkpoint selection or model update. This establishes a descriptive trajectory, not a causal explanation.

## Amendment C — sampler distribution check

The count sweep motivates scoring the complete sampled latent distribution as well as thresholded reader counts.
On the same 1,024 population histories, use the already fixed four independent samples per arm to estimate the
energy score, beta=1, with the unbiased off-diagonal sample-pair term. Report target distance and sample diversity
separately, for p, g, v and their concatenation, with paired episode-bootstrap uncertainty. No sample/arm selection,
model tuning or new interpretation of a latent proper score as semantic truth.

## Amendment D — fovea information may be distributed across its crop

The cell-local fovea reader has poor night-arrow recall, while the fovea encoder has contextual convolutions. Before
claiming lost information, fit crop-wide readers using all nine tokens, with the same physical-episode splits,
two seeds, 30-epoch cap and common MLP width. Compare periphery, fovea, their joint representation and the raw full
crop. Test (i) any-kind presence anywhere in the crop, and separately (ii) localisation through nine kind outputs.
All arms receive the same fixation and lighting inputs. Positive raw-pixel controls and convergence logs are
required; a failed finite reader remains a lower bound on recoverable information, not an absence proof.

## Amendment E — retain failed crop-reader power; use a shared query reader

The whole-crop MLP fails to localise rare arrows even from raw pixels. Its negative fovea result therefore cannot
resolve distributed information. Retain it. A narrower diagnostic restores a shared cell-query readout: put the
queried token first, followed by the other eight crop tokens in cyclic local-index order, plus the local index and
lighting. Compare query-G alone vs all-nine-G, and a raw queried-cell positive control. Pad all to the same 1,323
feature slots, MLP 256-256, same selected cells/splits/seeds/natural frequencies and 30-epoch cap. Missing context is
absent during both reader fitting and reading. Report raw night-arrow recall; below 0.75 near the player makes a
negative context result uninformative. Even above it, reader failure is not an information-theoretic absence proof.

## Amendment F — deterministic glyph check on REAL fovea reconstructions

The raw-pixel neural reader reads real HUD values perfectly but reads the learned reconstructions poorly. Because
decoded RGB itself is outside its training distribution, verify appearance with a simpler instrument: train-field
conditional white-pixel templates on the renderer's number area (last 5x5 pixels of each 7x7 cell), then nearest-template
classification. Read raw and decoded real crops for the same 30k S1 tuples. Use thresholds 210, 240 and 250, fixed
before results, report raw-control accuracy and decoded class histograms. This diagnoses the auxiliary decoder only;
it does not turn decoded images into ground truth for the latent generator or actor.
