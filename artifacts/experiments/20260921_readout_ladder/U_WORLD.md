# The old u→u world, joined on one population — its predictions lose part of what its input carries

Run 2026-09-24, `u_world.py` (`6784f863`, committed before the run). **Exploratory**: the
observability roots, pinned by content hash (the same ones COMPACTNESS.md used). The persisted 10k
`u→u` world (`../20260918_matched_10k/evidence/world_u_u.pt`, factual next-u MSE) over the frozen
TC-consecutive encoder and its persisted PCA, rolled out as the matched run did: u over 4 context
frames, prefill, one advance per action. Frozen-ladder harness; the per-branch arms share one head
across the 17 action branches. Evidence: `evidence/u_world.json`.

## One-step death (500 opportunity roots; prior 0.628)

| arm | expected safe | zombie (265) | lava (110) | night (281) |
|---|---|---|---|---|
| prior | 0.628 | 0.508 | 0.764 | 0.621 |
| root u — the world's input | **0.804** | 0.729 | 0.922 | 0.761 |
| **generated u** — the world's prediction | **0.727** | 0.629 | 0.940 | 0.689 |
| world history (Mamba output per action) | 0.722 | 0.617 | 0.937 | 0.684 |
| real successor u — hindsight control | 0.996 | 0.992 | 1.000 | 0.995 |

| paired contrast | difference |
|---|---|
| generated u − prior | +0.099 [+0.033, +0.170]* |
| **generated u − root u** | **−0.078 [−0.113, −0.041]*** |
| … on zombie-adjacent roots | **−0.100 [−0.152, −0.048]*** |
| world history − root u | −0.082 [−0.119, −0.046]* |
| real u − prior | +0.368 [+0.302, +0.441]* |

Two-step (986 roots, prior 0.685): the same pattern, smaller — generated u 0.718, root u 0.736,
generated − root −0.018*, generated − prior +0.033*.

## Declared reading: `u_world_loses_root_signal`

- **The world's predicted successor carries less than the state it was given.** Generated u beats
  the prior, but falls 0.078 below the root u it started from. Of root u's margin over the prior
  (+0.176), the prediction keeps about 56%. On zombie roots, the largest hazard class, it keeps
  about half (+0.121 of +0.221) and loses 0.100.
- **The loss is in the prediction, not the readout.** The same shared branch head reads the *real*
  successor u almost perfectly (0.996). What the head cannot find in the generated u is not there.
- **Terrain survives, mobs do not.** Next to lava the prediction matches or exceeds the root (0.940 vs
  0.922); the whole shortfall is on dynamic hazards — the pattern FROZEN_LADDER found in the Raw H2
  world.
- **This is consistent with the old panel, now with a population large enough to see it.** There,
  generated u scored 23.0/36 against a 24.3 root+action control; here, generated u is resolved below
  root u.

## Together with the rest of the day

1. The deciding information is in the current frame (one raw frame 0.884; OBSERVABILITY, FROZEN_LADDER).
2. The Raw H2 encoder's patch grid carries it; its CLS — the Raw H2 world's only input — does not.
   Confirmed on sealed seeds (BOUNDARY).
3. A label-free 192-D PCA of the patch grid keeps it (COMPACTNESS): the old `u` reaches 0.804.
4. A world trained on that `u` with factual next-state MSE predicts successors that keep only about
   half of it, and lose it on mobs (this document).

So the representation can be fixed by choosing the input (patch-derived `u` rather than CLS), and
what remains is the world's training: factual MSE does not preserve the action-conditioned mob
consequences its own input carries. That is the case for the all-action consequence A/B on the
`u→u` architecture, against a matched factual control, judged on a new seed block.

## Limits

- Exploratory roots, already inspected; one world, one training seed.
- Root u uses a root-level head (root → 17 scores), generated u a per-branch shared head. The branch
  head is not the bottleneck — it reads real u at 0.996 — but the two are different parameterizations.
- The world is on the TC-consecutive encoder, not Raw H2; how a u-route on Raw H2 would behave is
  inferred from COMPACTNESS, not measured.
