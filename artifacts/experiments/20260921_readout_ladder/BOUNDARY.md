# Boundary confirmation — patch tokens over CLS, on sealed fresh seeds

Run 2026-09-24, `boundary.py` (`b810ee65`), rules committed before any seed from 51,000 was walked.
**Sealed**: seeds 51,000–51,422, collected by `observe.py` with the fork collector's exact loop; the
predeclared stop rule fired at 800 one-step opportunity roots (423 seeds, 417 with roots, 4,265 roots,
1,539 two-step opportunities). Development: the 700 FIT seeds and inner split, unchanged. Frozen Raw
H2 checkpoint. Evidence: `evidence/boundary.json`.

## Declared reading: `patch_over_cls_confirmed` — all four rules hold

One-step death; prior (always DOWN) 0.671; paired, episode-seed-clustered 95% intervals.

| rule | difference | holds |
|---|---|---|
| R0 raw pixels, 1 frame − prior (positive control) | +0.209 [+0.170, +0.258] | yes |
| R1 patch tokens (attention) − CLS | +0.080 [+0.050, +0.109] | yes |
| R2 patch tokens (attention) − CLS, 40× wider head | +0.102 [+0.060, +0.142] | yes |
| R3 R2 on zombie-adjacent roots (474) | +0.075 [+0.031, +0.120] | yes |

| arm | expected safe | vs prior | head params | day | night | zombie | lava |
|---|---|---|---|---|---|---|---|
| prior | 0.671 | — | — | | 0.687 | 0.567 | |
| raw pixels, 1 frame | **0.881** | +0.209* | 609k | 0.915 | 0.863 | 0.834 | 0.988 |
| patch tokens, attention | **0.732** | +0.060* | 115k | 0.757 | 0.719 | 0.623 | 0.974 |
| CLS | 0.652 | −0.019 | 108k | 0.699 | 0.628 | 0.544 | 0.913 |
| CLS, wide (192→2048→2048) | 0.629 | −0.042 | 4.6M | 0.673 | 0.606 | 0.548 | 0.783 |
| projected z | 0.654 | −0.017 | 108k | 0.694 | 0.634 | 0.542 | 0.943 |

Reported, not ruled on: CLS wide − CLS −0.023 [−0.056, +0.011]; z − CLS +0.002 [−0.020, +0.022].

Two-step (1,539 roots, prior 0.673): every rule holds again, with small margins — R1 +0.018*, R2
+0.029*, R3 +0.015*; pixels +0.044* over the prior; patch tokens +0.008 (unresolved) over it.

## What is now established

- **Replicated on seeds nothing had touched**: from the frozen Raw H2 encoder, within-root safe
  action ranking is readable from the patch tokens and not from CLS. The frozen-ladder result was not
  an artefact of the roots it was found on.
- **Not a capacity artefact.** A CLS head 40× the size of the patch-attention head does *worse* than
  the small one (0.629 vs 0.652) and still loses to the patch tokens by 0.102; it overfits (fit 0.83).
- **CLS and z sit at the prior**, as on the inspected block. What the world consumes carries no
  readable within-root safety signal here.
- The margins are smaller than on the inspected block (patch tokens +0.060 over the prior here, +0.103
  there), partly because this block's prior is stronger (0.671 vs 0.628).

## Read with COMPACTNESS.md

On the inspected roots, a label-free 192-D PCA of the patch grid carries the signal (0.783 on Raw H2;
0.804 for the literal old `u`), where the equally sized CLS does not. So the gap is between *what CLS
encodes* and *what the patch grid encodes*, not between one vector and many tokens. The old `u→u`
world transitioned exactly such a vector and its generated successor still failed action choice —
which points the next repair at supervision and dynamics, not at a wider state.
