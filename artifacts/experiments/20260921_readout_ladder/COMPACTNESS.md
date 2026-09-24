# Compactness discriminator — the old `u` keeps the root-side safety signal

Run 2026-09-24, `compactness.py` (`a4933180`, committed before the run). **Exploratory**: the
observability roots, pinned by content hash (fit 7,085; judgement 2,856 on 50,000–50,274, already
inspected). The 51,000+ block is untouched here, reserved for `boundary.py`. Frozen-ladder harness
throughout: expected-risk ranking over 32-key P, 512-wide head, 3,000 updates, inner-FIT selection,
three probe seeds. Evidence: `evidence/compactness.json`.

**The question.** The frozen `u→u` world (`../20260917_state_transition`) transitioned `u`, a fixed
TRAIN-fitted label-free PCA of the 4x4 pooled patch grid (3,072 → 192). Generated, it chose safely on
23.0/36 against a 24.3 root+action control, and 10k updates raised generated-state AUC 0.620 → 0.708
without moving fatal-safe ranking. Did that fail because the 192-D compression loses the signal, or
despite a representation that carries it?

**Old encoder.** The 2026-09-16 checkpoint no longer passes the gated loader (training, gate and
data code have drifted since). Only its encoder was loaded, after asserting `d4mj/lewm.py` and every
`transformers` file in its source manifest byte-identical to what it recorded, pins, versions and
execution flags unchanged, the stored encoder settings rebuilt exactly, and the weights loaded
strictly. The persisted PCA (rank 192, fit on 20,000 samples) was applied unchanged.

## One-step death (500 opportunity roots; prior 0.628)

| arm | expected safe | vs prior | zombie | night | head params |
|---|---|---|---|---|---|
| old encoder, 4x4 grid uncompressed | 0.806 | +0.178* | 0.709 | 0.767 | 1.58M |
| **old `u` (that grid → persisted PCA-192)** | **0.804** | **+0.176*** | **0.729** | 0.761 | 108k |
| Raw H2 encoder, 4x4 grid uncompressed | 0.737 | +0.109* | 0.623 | 0.701 | 1.58M |
| Raw H2 grid → PCA-192 (FIT-fitted) | 0.783 | +0.155* | 0.686 | 0.745 | 108k |

| paired contrast | difference |
|---|---|
| **old `u` − its uncompressed grid** | **−0.001 [−0.038, +0.033]** |
| … on zombie-adjacent roots | +0.020 [−0.039, +0.070] |
| Raw H2 PCA-192 − its uncompressed grid | +0.046 [+0.013, +0.080]* |
| old encoder grid − Raw H2 grid | +0.069 [+0.031, +0.106]* |
| **Raw H2 PCA-192 − Raw H2 CLS** (both 192-D, same head) | **+0.156 [+0.104, +0.203]*** |

Two-step: the same ordering, smaller (old `u` 0.736 vs prior 0.685, +0.051*; old `u` − grid −0.003).

## Declared reading: `compression_keeps_signal`

- **The old `u` ranks root actions as well as the grid it compresses** — 0.804 against 0.806, on
  every stratum within noise. So the `u→u` world was built on a state that carries root-side safety,
  and its generated successor still failed action choice. On the review's branch, that failure points
  at **supervision and dynamics** — factual next-state MSE on a single 192-D vector — more than at a
  need for an 81-token state.
- **The 192-D bottleneck is not the problem; CLS is.** On the current Raw H2 encoder, a label-free
  192-D PCA of the patch grid reaches 0.783 where the equally sized CLS reaches 0.627, with the same
  head. Compression even *helps* the probe (+0.046 over the uncompressed grid), consistent with the
  3,072-D input overfitting at this data size.
- **The old encoder's grid carries more than Raw H2's** (+0.069), at the root. Why is not measured.

## Limits

- Root-side only. This says the old `u` *carries* the signal at the root; it does not re-measure the
  old world's *generated* `u` on these roots. The `u→u` generated failure is from the older 36-root
  panel with a different evaluator. Scoring the persisted `world_u_u.pt` generated successors in this
  harness would join the two on one population.
- Exploratory roots, already inspected.
- Different encoder checkpoints: the old grid is from the 2026-09-16 Raw joint checkpoint, not Raw H2.
- The old `u`'s scale on these hazard-root frames (std 3.7 on a sample) differs from its original
  corpus pool (2.8); the probe standardizes inputs, but the PCA was fit on a different distribution.
