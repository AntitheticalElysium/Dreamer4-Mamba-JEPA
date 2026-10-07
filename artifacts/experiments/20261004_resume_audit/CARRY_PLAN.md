# Resume audit, 2026-10-04

The October 2 read-only restriction was explicitly lifted by the user on October 4. Preserve the other agent's checkpoints,
reports and working-tree changes. This audit diagnoses existing claims and resumes interrupted, already declared runs;
it does not launch a combined architecture or apply E19's pending training intervention.

## Verified protocol correction before resuming E17

`tworld.rollout_losses` supervises output positions `0..L-2` against frames `1..L-1`. For L=16, prediction after 15 inputs
is trained; prediction after 16 inputs is not. `teval.step` returns the final input position. Lane73 requested window 16
for damage, H16 trajectory heads and teval, although its recall evaluation correctly used 15. A CPU backward pass on the
actual completed A16 s7 checkpoint gave time-row gradient norms 0.000507..0.001388 for rows 0..14 and exactly 0 for row 15.
Change the queued window-16 evaluations and their output/comparison tags to 15. Keep training unchanged. Window 16, if
ever measured, must be labelled an extrapolation sensitivity test rather than the trained-context comparison.

## Additional separating diagnostic: actual use of historical cell contents

The published recall contrast compares window 5 with window 1, changing position and scan length as well as history.
Use the existing 2,048 held-out main windows, with the same targets and existing scroll/cell labels. Select recallable
entering cells, using at most 512 cases in each same-slot/moved-slot stratum (seed 61004). Inputs end at the same frame
and retain the same window length, time positions, current frame and action sequence in every condition.

Conditions: original history; replace all prior sightings of the target world cell with a donor's held-out token from
the same view slot; replace the same number of unrelated historical map tokens with the same donor token. Donors are
rotated within source-slot and target-time groups with at least two independent held-out rows. Current inputs are never
edited. Checkpoint weights remain fixed. Compare paired target error and output displacement towards the donor token;
cluster bootstrap by held-out pool row. Report case counts, baseline capture and intervals for every arm/stratum.

This is an exploratory causal input intervention, not a sealed performance gate. Edited histories need not be natural
game trajectories. Selective sensitivity establishes use of cell content; it does not by itself establish exact carry
storage, an actor benefit, or a capacity ceiling. The unrelated-token control detects general corruption sensitivity.

## Pending queue

Completed: A16 s7/s8, full training-state update 6000 each. Interrupted: M16 s7 at logged update 1000, with no full-run
state saved; restart from its declared 36k parent. M16 s8 and all E17 final evaluations are pending. E18 fcanvas s7/s8
did not save a training state. M6 H16 trajectory comparison failed allocation twice and remains pending at batch 4.
`check_delta` has no output or running process. Finish those separating diagnostics before scheduling E19 or any combined run.

Literature check: Delta-IRIS (2406.19320) uses L1, L2 **and max-pixel** reconstruction terms, unlike our E16's uniform
latent L1. Its stochastic health examples are not evidence that our bottleneck retains ordinary damage. Mamba
(2312.00752) interprets input-dependent Delta as selecting what to retain/overwrite, not a calibrated uncertainty estimate.
VaGraM (2204.01464) motivates value-relevant model weighting; E19's hand-labelled health mask is not its value-gradient method.

## Follow-up declared after the sighting result, before execution

Past-content use alone does not locate storage in the SSM: Mamba also has a four-tap causal convolution. On 64 cases from
each stratum of the saved ledger (seed 61004), run both Mamba seeds on CPU in float32 reference mode. Split each layer's
scan just before the last input; verify that a no-reset split agrees with the unmodified scan. At that boundary, zero
only the target slot's SSM carry, convolution carry, both, or an unrelated slot's two carries. Repeat target-sighting
swaps with each target reset. Window, positions, inputs and actions remain unchanged otherwise. Record baseline error,
paired changes, and donor pull for each reset. This is a frozen-state mechanism intervention, not a training recipe or
performance gate. Other slots can still route stored history through spatial attention; no result implies a hard bound.
