# Frozen failure localization — post-hoc, before computation

2026-10-14. The registered paired experiment failed coherent completion. No
new training or criterion change is authorized by this diagnostic. The final
104 exploratory roots,64 draws, batch4, context/action/noise seed rule, checkpoint
and both frozen semantic readers remain fixed. No fresh judgment data is opened.

Questions measured separately:

1. Does source-wise hard routing remove most absent/multiple outcomes, or do
   incorrect contextual edits remain even when every source chooses one route?
2. Does excess variation originate in packets outside the scored35 domain?
3. Are small pre-LN reference magnitudes amplifying edits, or is the large
   variation already present before normalization?

Frozen paired variants: original; source routes replaced by their one-hot
argmax with event/gate/belief fixed; only scored-source gates retained; only
unscored-source gates retained. These are inference interventions, not fitted
repairs. Retain full event, source/destination, count, static/camera and joint
distribution readings. Inactive-source masks refer to the existing training
domain only, never simulator object/movement labels. Preserve invalid draws.

For both original arms capture per-token pre-LN reference centered RMS, edit
centered RMS, pre-LN sample variance and final-LN sample variance. Summarize
scored35, other map cells and HUD separately, including ratios and conditional
low-reference-scale strata. The direct pre-LN output is reconstructed from its
actual copy/generation weights and candidates; paired output is b plus its
actual source and incoming writes. Original outputs must reproduce their
cached predictions exactly at identical shapes and draws.

Hard routing alone is sufficient only if joint valid movement and distribution
readings improve together; higher validity from suppressing movement is not a
repair. Failure of inference hardening is not a universal rejection of trained
hard assignment. Gate masking can causally localize variation to a source
domain but does not establish an adaptive-compute or learned-selectivity method.
Pre/post-LN magnitude correlations alone do not establish normalization as
the sole cause. Numerical/readout failures are inconclusive.

Record source/checkpoint/input hashes and atomic resumable batches under
`artifacts/eda/oracle_v1/paired_event/diagnostics/failure_localization`.
No source file, model, target, learned threshold or registered verdict changes.
