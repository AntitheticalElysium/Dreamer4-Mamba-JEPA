# F1.6 projection: loss versus readout geometry

2026-11-03. User authorized a diagnostic ablation of the global mixing/projection
and asked for its design/paper provenance. No world update, route replacement,
chooser training, P2, actor or overnight run. This protocol is fixed before fits.

## Separating question

The completed retention audit's mixed-vector null is a bounded-reader result,
not a proof of erased fine information. Repeat the old F1.5 exact projection
decomposition on F1.6's ACTUAL nonlinear joint tokens, not its raw fine tokens.
At fixation f, u is the 576D local joint crop and e=A_f u+c(p,v,mask).
The deterministic recoding A_f^+(e-c) cannot restore the discarded null component.
Positive predictive recovery after this recoding proves survival for the rescued
capability. Failed recovery still cannot prove erasure with finite readers.

## Arms and controls

Freeze selected F1.6 checkpoint (SHA256
34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045).
Use the exact existing 400 train and 400 dev episode caches and next-fine targets.
Fit identical MLP256-256 predictors, supplement padded to1024, two seeds611701/2,
2400 AdamW updates, batch512, lr.001, wd.0001, clip1; natural frame draws.
Fit first360 train episodes excluding world-monitor, last40 validation.
Dev first250 diagnosis / next150 confirmation; final split untouched.

Seven arms per seed (14 fits):
- joint: actual local joint tokens (replicate the preceding audit exactly);
- native: isolated FP32 projected joint contribution A_f u;
- retained: A_f^+ A_f u, same surviving information in crop coordinates;
- null: u-retained, discarded directions alone (not unique lost information);
- random_retained: fixed shared random rank256 subspace in crop coordinates;
- mixed: actual BF16 complete embedding (replicate preceding audit exactly);
- recovered_bf16: subtract the known quantized coarse/mask/light offset from the
  actual BF16 mixed vector, then decode with the quantized projection's inverse.

Existing base and raw-fine errors are immutable references, not refitted or
selected. Replication requires joint/mixed per-cell errors to agree to1e-6.
Main cohort: night, entity in current crop, no next scroll. Main score: whole
next-crop fine MSE. Also source entity cells, one-ring neighborhood; all frames,
day/entity, night/entity, night/empty, scroll. Report both seeds and episode-
bootstrap95% intervals; same masks and comparisons as original projection audit.

## Mechanical qualification

All49 fixation slices: rank, singular values, right-inverse, retained/null
decomposition and exact equivalence. Actual joint tokens and mixed outputs must
match the native computation on cached real percepts. No future targets may enter
transforms/inputs. FP32 references disable TF32; diagnostic fits match previous
TF32 setting. Save numerical errors and decoded BF16 differences, not just ranks.
Smoke an actual predictor backward; verify frozen world and checkpoint/source
hashes before/after the run. Source changes invalidate the result.

## Interpretation and stopping

Retained beats native: reader/representation interface explains that rescued
portion; no discarded dimensions were restored. Recovered BF16 succeeds: actual
mixed output retains that usable information conditional on the known coarse
input and fixation. The native Mamba does NOT separately receive that coarse
input: this is a generous survival control, not proof it can learn subtraction.
Joint beats retained: residual deficit, compatible with rank loss and finite
readout limits. Null utility cannot establish unique loss on the data manifold.
These tests cannot causally locate Mamba learning pressure or denoiser use.
No further fit-budget ladder or architecture change follows. Return for discussion.
