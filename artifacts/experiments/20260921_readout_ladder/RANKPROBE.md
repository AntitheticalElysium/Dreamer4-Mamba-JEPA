# Objective-matched probes on existing forks — one- and two-step, 4- and 32-frame

Run 2026-09-24, frozen Raw H2 checkpoint. Script `rankprobe.py`, committed before the run
(`e99d5f6f`). The 2026-09-19 ladder's readout instrument, imported unchanged: within-root pair
ranking, whole 17-action fans per batch, FIT-only standardization, mlp128, three probe seeds, with
BCE-death beside it on identical rows. Fit on 7,085 roots (1,260 one-step opportunities); judged on
the 4,047 roots of the 405 previously examined seeds — **exploratory, not sealed**. Evidence:
`evidence/rankprobe.json`, per-seed judgement scores in `evidence/rankprobe_rows.pt` (hash-bound).

## The instrument works

The hindsight positive controls, which see the realized successor, reach the target, and ranking
lifts them as the 09-19 audit found it would:

| one-step, judgement | rank | BCE-death |
|---|---|---|
| successor CLS + pooled patches | **1.000** | 1.000 |
| real successor z | **0.886** | 0.771 |
| real features | **0.891** | 0.844 |

So a failure below is a failure of the features, not the evaluator.

## No foresight representation beats the action prior

FIT-root action prior (always DOWN): **0.591** one-step, **0.756** two-step. `*` = resolved.

| rung | one-step rank | one-step BCE | two-step rank | two-step BCE |
|---|---|---|---|---|
| root features @4 | 0.564 | 0.544 | 0.744 | 0.751 |
| root features @32 | 0.554 | 0.559 | 0.757 | 0.751 |
| root pixels (CLS + 4x4 pooled) | 0.594 | 0.612 | 0.772 | 0.779 |
| u @4 | 0.498 * | 0.565 | 0.738 | 0.705 * |
| u @32 | 0.494 * | 0.568 | 0.731 | 0.715 * |
| generated z @4 | 0.496 * | 0.505 * | 0.732 | 0.688 * |
| generated z @32 | 0.518 * | 0.485 * | 0.732 | 0.690 * |
| generated features @4 | 0.510 * | 0.489 * | 0.746 | 0.669 * |
| generated features @32 | 0.518 * | 0.499 * | 0.733 | 0.666 * |

`*` here marks a resolved deficit against the prior; **no foresight cell resolves above it**.

- **The ranking objective does not rescue foresight.** It lifts every hindsight rung and none of
  the foresight ones. The 09-19 lesson was real and applied to the ladder; correcting for it does
  not change the conclusion that no foresight rung beats the prior.
- **32 frames do not help** any rung, on either outcome.
- **Root pixels are the best foresight rung** and still only tie the prior (+0.003 one-step,
  +0.016 two-step, neither resolved), though they beat the world's own root features on BCE
  one-step (+0.068) and on both two-step objectives (about +0.03).
- **The "u beats generated z" lead does not survive the objective change.** Under BCE, u ≥ generated
  z (0.565 vs 0.505 one-step); under ranking they are equal (0.498 vs 0.496). Both are below the prior
  either way.
- **Heavy overfitting under ranking.** Root features reach 0.90 safe-choice on their own fit roots
  and 0.56 on judgement roots. 1,260 fit opportunity roots may be too few for this probe to find
  a spatial rule, if one exists.

## NOOP / RIGHT / SLEEP, two-step

623 judgement roots where two-step death differs among the three. Always-RIGHT **0.907**, which is
also the FIT-best of the three. No foresight rung beats it; the generated rungs trail it by
0.02–0.13, resolved in 6 of 8 cells, worst under BCE (generated features: 0.776, −0.131). Only a hindsight rung beats it
(successor pixels, rank, +0.039).

## Read together with the replay pilot

`REPLAY.md`: given the full simulator state, one-step death is deterministic — the full-state
oracle chooses safely on 90 of 90 opportunity roots, against the same prior's 0.511.

So one-step death is a fixed function of (full state, action), yet **no representation of the
root observation tried here, under either objective, predicts it better than always choosing
DOWN**. The gap sits between the full state and what these root representations expose. Two
explanations remain, and these data do not separate them:

1. **The deciding state is not in the frames.** Mob attack cooldowns are never rendered. A
   predictor that sees only pixels would face uncertainty the oracle does not.
2. **It is in the frames and these representations lose it.** The 9x9 patch grid is one tile per
   patch — a 9x7 tile map plus two HUD rows (09-19 audit) — and pooling it to 4x4 blurs which
   tiles touch the player; the
   world's root readout is 256 wide; and the probes overfit.

Neither the export branch ("visual inputs work, projected z fails") nor the predictor branch ("u
works, generated z fails") is indicated: no foresight input works, visual or otherwise.
