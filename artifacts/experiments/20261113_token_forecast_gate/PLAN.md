# Option A gate: an all-cell periphery-token forecast from the belief (2026-11-13)

From the F5 dissection (`../20261111_p1f5/RESULTS.md` section 11): a clock formed only where a loss asked Mamba for its
timing. The reward forecast asked, for reward-relevant clocks. Nothing asked for food, drink or energy. The TSP
forecast's request was about zero because of three things: a spoiled readout, a pixel-MSE currency, and 8% coverage.
Option A (discussed with the user 11-13) replaces TSP's forecast mode. It is a separate head that predicts the next
frame's periphery tokens for all 81 cells from the belief predicted before the frame (bp). The question here is whether
such a target asks for the pixel-only clocks, before any training run. Frozen F5 checkpoints 1, 2 (before and while
recovery forms) and 10. No world training.

Design changes from Codex's review (11-13), adopted:
- **A separate token head**, comparing two currencies instead of a learned variance:
  - plain MSE;
  - MSE weighted per (cell, channel) by a fixed, floored inverse of the typical step-to-step change of that channel.
    The weight is computed once from DEV tokens; this is "precision" without a scale the model could raise.
  - We hit the scale trap with a learned pixel variance on 10-31; Seitzer et al. 2022 and Stirn et al. 2023 describe
    the same failure.
- **Bars on direction and on nuisance, not on matching the reward forecast's gradient size.** Rates scale with the
  square of the loss.
- **Early checkpoints**, copy-previous controls, and the due and not-due members scored separately.

## Instrument (`gate.py`)

Per checkpoint k in {1, 2, 10}:
1. **Weights.** sigma2[c, d] = mean over 80% of DEV episodes of (p_t - p_{t-1})^2, floored at 1% of its median. The
   precision weight is w = 1 / sigma2, normalised to mean 1. The plain currency is w = 1.
2. **Heads.** One per currency, of the form that rendered health from frozen beliefs (`readout_fit.py`'s concat:
   cell embedding + belief projection -> MLP 256-256 -> 16 per cell). Fitted on frozen bp_t -> p_t (80% of DEV
   episodes, Adam 1e-3, batch 512, 20k steps).
3. **Held-out** (the other 20% of episodes): weighted loss of each head against copying p_{t-1}.
4. **Exact-history pairs** (energy, hunger; recovery and cooldown for reference), due transition. Targets are the
   periphery tokens of the next frame.
   - **Consequence:** mean_w (x_A - x_B)^2, the cost of erasing the phase for an exact forecaster.
   - **Nuisance:** the same for a +1/+3/+6 per-255 brightness edit of member B's next frame.
   - **Floor:** the head's held-out weighted loss per frame.
   - **kappa** at the digit's cell: the fraction of the true due vs not-due token change the head reproduces.
   - **Request** through belief parameters, as in `../20261111_p1f5/credit.py`: rate = -grad(sep) . grad(loss), with
     sep = the swap cost of the two members' predictions. Also the scale-free cosine(-grad sep, grad loss) per pair.
     The reference is the reward forecast's cosine on recovery and cooldown at the same checkpoints, from
     `credit.py`'s saved pairs.
5. **Checkpoint 10 only, the user's TSP question:** how much does the reactive TSP use the belief? Held-out reactive
   TSP error with the true bp vs a bp from a random other frame.

## Fixed readings (before results)

A currency qualifies A for a training run only if, at checkpoint 1 or 2, all three hold:
1. **Consequence > nuisance:** energy and hunger erasure costs exceed the +6/255 edit in a majority of pairs.
2. **The request has the right direction:** the mean per-pair cosine for energy and for hunger is positive with an
   episode-bootstrap 95% interval excluding 0. The reward forecast on recovery is the reference, not a bar.
3. **The head is a forecaster:** it beats the same head given beliefs shuffled across held-out frames, in its own
   currency. Copying the previous tokens is reported as a reference, not a bar. (Changed after the 50-step smoke and
   before any real result: copying hands the forecast the previous frame, which it is deliberately denied. That is
   the cache's job; the forecast's job is to read the belief.)

How to read the outcomes:
- precision passes and plain fails: the currency matters as predicted;
- both pass: coverage alone suffices;
- both fail: A as specified would not ask for the clocks, and we discuss before any run.

A pass is not a repaired world. The run it earns is judged on:
- native energy and hunger timing on the exact pairs, due and not due reported separately;
- false drops held down;
- F5's recovery, cooldown and danger gains preserved;
- moving cows and arrows as guards.
