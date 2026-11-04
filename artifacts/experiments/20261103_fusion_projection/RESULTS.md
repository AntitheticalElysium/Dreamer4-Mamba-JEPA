# F1.6: some projection information is recoverable; entity-cell rescue failed

2026-11-03. Completed14 bounded diagnostic fits on the frozen F1.6 checkpoint.
No world/encoder/Mamba/denoiser/chooser parameter changed. Protocol: PLAN.md;
source/history audit: PROVENANCE.md. Full readings: RESULTS.json.

## Question and answer

Can the previous mixed-vector null be a reader/representation problem rather
than information destruction? **Partly, demonstrably yes for whole-crop future
prediction. No entity-cell rescue was demonstrated.** Do not generalize the old
F1.5 positive recovery to F1.6 or call the whole gap a probe artifact.

At each fixation the native projection is e=A_f u+c, where u is the ACTUAL
nonlinear local joint crop, not the raw glimpse. Recode the retained component
as A_f^+ A_f u. This has576 coordinates but only256 degrees of freedom, exactly
the same information as the projected fine contribution in exact arithmetic.
It cannot restore the discarded component. The actual-BF16 recovery subtracts
the known quantized coarse/mask/light contribution and applies the corresponding
inverse. Current coarse information and fixation were already available to all
readers; no target/label information builds these maps. Native Mamba sees the
mixed vector, not separate current coarse tokens: this is conditional survival,
not a demonstration of native subtraction or a production repair.

## Held-out confirmation

Night, entity in current crop, no next scroll:1824 frames in102 episodes. Figures
are percent reductions in next-fine MSE versus the SAME coarse/history baseline,
not semantic accuracy or fractions of information. Two independent reader seeds;
one trained world seed. Entity-cell target means next fine latent at the cell
containing the CURRENT entity, not the next entity's tracked destination.

| Supplement | Whole crop, seed1 / seed2 | Current entity cells, seed1 / seed2 |
|---|---:|---:|
| Raw fine (existing reference) |45.79 /45.72|31.11 /30.51|
| Local fused tokens |22.38 /22.06|11.45 /11.71|
| Isolated native projected contribution |2.34 /1.93|0.16 /1.06|
| Recode retained contribution |4.82 /7.24|−4.45 /−1.55|
| Discarded component alone |4.39 /7.32|−2.95 /−0.74|
| Shared random rank256 subspace |−1.18 /0.31|−2.55 /−1.17|
| Actual BF16 mixed vector |0.24 /0.21|−0.79 /0.64|
| Recover retained from actual BF16 vector |4.16 /6.48|−4.29 /−1.68|

Whole-crop retained gains have95% episode-bootstrap intervals[2.93,6.59] and
[5.76,8.75]%. Relative to the native projected reader, recoding reduces error by
2.53%[0.93,4.10] and5.42%[3.76,6.91]. Positive in both reader seeds and diagnosis
and confirmation cohorts: representation/readout geometry explains a real portion.
Actual-BF16 recovery also has positive whole-crop gains in both seeds/cohorts.

The local-joint advantage remains much larger. Recovered/retained entity-cell
scores are poor; confidence intervals and every other predeclared population are
in RESULTS.json. No seed/cell/cohort was dropped. Both discarded and retained
components having whole-crop utility does not imply unique or additive information.
The random subspace does not rescue the task here, unlike the old F1.5 result.

## What was verified

- Same400 train/dev physical episodes and exact targets, fit/validation split,
  monitor exclusion, 2400 updates per reader, minibatches, initialization and
 1024 padded supplement dimensions as the prior audit. No fit-budget extension.
- Local-joint and mixed-vector control error arrays reproduce the preceding
  audit BIT FOR BIT in both seeds (max difference0.0 in all four comparisons).
- All49 fixation slices have rank256; condition14.64–16.70; right-inverse maximum
  error3.44e-15; null projection residual2.53e-15 (double-precision algebra).
- Actual nonlinear joint/native mixed outputs match the unmodified preceding
  transform. Future-target mutation leaves inputs unchanged; backward reaches
  the diagnostic supplement; retained+null equals joint; world weights frozen.
- BF16 mixed error is2.36% of the isolated fine-branch RMS; decoded retained error
  is5.56% of retained RMS. Retained/recovered predictive scores are much closer
  than joint/retained. These results do not identify rounding as the main gap;
  they are not exact numerical-equivalence or recurrent-rounding claims.
-112 source/input pins verified before/after fitting; selected checkpoint SHA
 34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045 unchanged.
- Real-path smoke passed. Fits and numerical/evaluation work took251.14 seconds,
  peak allocated GPU memory2.83GiB. CPU bootstrap report separate. Synthetic
  paired-report checks recover known25% gain and exact zero null.

## Limits and implications for the model

The surviving whole-crop gain is a positive proof of some useful conditional
information, not a universal sufficiency result. The failure to rescue entities
does NOT prove deletion: these finite MLPs may fail to decode the retained
representation, and their whole-crop training objective can leave rare entity
directions poorly learned. This audit changes neither the training objective nor
native memory acquisition and cannot causally settle learning pressure.

The four memory tokens are a1024D linear expansion of the256D Mamba emission;
the trained expansion is full-column-rank256 (condition14.01). This specific
operation does not further reduce emission dimension in exact arithmetic.
Mamba's full internal carry is larger than its emission; weak emission readout
does not prove the carry lacks fine information. Native use by the denoiser
remains distinct from access by an auxiliary reader.

The next design discussion should treat the compact spatial-to-global interface
as an unresolved design bet, not infer that Mamba or the denoiser alone is at
fault. The original future-fine instrument also lacks the future coarse evidence
available to native cascade fine prediction, so its gain is not automatically
the native consumer's potential gain. No replacement architecture or training
run follows this bounded diagnostic; return for discussion.
