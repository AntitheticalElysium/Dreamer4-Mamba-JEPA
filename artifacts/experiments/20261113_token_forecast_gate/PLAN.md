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

---

# Round 2 (11-13 afternoon): a categorical currency, and where hunger's reset is lost

Asked by the user after round 1: run the categorical currency through the same instrument, and investigate whether
hunger's missing request is an input-side problem, whether hunger is the only clock affected, and how to verify it.
Codex's review of round 1 is adopted where marked. Frozen F5 checkpoints 1, 2 and 10, no world training.
`gate2.py` (GPU) and `d1.py` (CPU probes) write `readings/round2/`.

## Part 1: the categorical currency against plain MSE

**The currency, fixed before any fit.** HL-Gauss (Farebrother et al. 2024, ICML; their Appendix A code):
- the target of each token number x is the Gaussian mass N(x, sigma^2) on each bin, renormalised to the mass inside
  the support;
- the support is [-3, 3], the token's exact range (`out_map`: 3 tanh after a layer norm), not a choice;
- 51 bins, their Atari default; sigma = 0.75 bin width (0.088), their recommended default. Their bin ablation is
  21-201, and they report the best sigma does not depend on the bin width.
- The loss is the cross-entropy, averaged over cells and numbers as plain MSE is. The token estimate (used for kappa)
  is the expected bin centre.

The reward head uses one-hot hard bins. Codex's CPU check showed hard bins charge an adjacent-bin crossing like a
large jump, so the soft form is used. My round-1 claim that binning makes the forecast robust to brightness is
withdrawn: with Gaussian labels, an exact forecaster's cost is close to MSE / (2 sigma^2) for any shift.

**Same beliefs, data draws and trunk (Codex).** The plain head is refitted first, exactly as in round 1; its held-out
losses must reproduce `readings/ckpt_XX.json`, which validates the run. The categorical head has the same trunk with
16 x 51 outputs per cell and the same batch generator.

**Readings per pair set** (energy, hunger; recovery and cooldown for reference), at the due transition:
- **Nuisance, as the trained head pays it** (both currencies). Consequence = how much more loss the head pays when the
  drop happens than when it does not, averaged over the two members' predictions. Foil = how much more it pays for
  member B's next frame +1/+3/+6 per 255 brighter than for the real one. An exact-forecaster version needs the
  probability an exact head would put on bins it never predicts, which is not defined for a softmax head. It is
  reported for plain MSE (round 1's definition) and, with no floor, for the categorical currency (Codex's check), but
  it is not the bar.
- **Request direction** (round 1): cosine(-grad sep, grad loss) over belief parameters, per pair, with an episode
  bootstrap. sep is each head's own swap cost.
- **Request without the other 80 cells:** the same cosine with the loss restricted to the digit's cell; sep is
  unchanged (the branches' targets differ only in that cell). This separates dilution by the other cells' errors from
  a missing separation handle.
- **Matched held-out update (Codex).** Pairs are split into two folds by episode. Each fold's summed loss gradient is
  preconditioned as the trainer would (F5's AdamW second moments at that checkpoint, bias-corrected, eps 1e-8) and
  normalised to a unit step. Its effect is read on the other fold's pairs through a yardstick common to every loss:
  d(log D) per unit step, with D the squared distance between the two members' predicted beliefs at the due step,
  relative to their mean squared size. The same is computed for the reward forecast on the recovery pairs, the loss
  that did teach a clock; this is the reference scale.

**Bars, fixed now.** The categorical currency is a candidate over plain MSE for energy if, at checkpoint 1 or 2:
1. consequence > the +6/255 foil, as the trained head pays it, in a majority of energy pairs;
2. the energy cosine is positive, episode-bootstrap 95% interval excluding 0;
3. the head beats the same head given shuffled beliefs, in its own currency;
4. it adds over plain MSE: the matched held-out update on energy is larger than plain MSE's, with the paired
   per-pair difference's 95% interval excluding 0.

**Yardstick changed after the 50-step smoke, before any real result.** In the smoke, the reward forecast's held-out
update shrank the recovery pairs' raw belief distance (d log D -13.7 [-18.3, -9.5]). That is the loss that taught
the recovery clock, so raw distance does not measure the separation that matters: a loss can sharpen the relevant
difference while compressing the rest. The reward figure does not depend on the smoke's 50-step heads, so it is a
real reading, and it is reported. The matched update is now read through each loss's own head, in common units:
- kappa, the fraction of the true due / not-due contrast the head reproduces: the token change at the digit's cell
  for the token heads, and P(actual reward) due minus not due for the reward forecast;
- held-out (the other fold's step, as above) and in-pair (the pair's own step), both in AdamW geometry, unit norm;
  d log D stays reported.

**Geometry changed after a second smoke, before any real result.** In AdamW geometry (F5's second moments) the reward
forecast's own d kappa on recovery was not positive, held out (-0.0034 [-0.0078, +0.0007]) or even in-pair (-0.0045
[-0.011, +0.0017]), although its request cosine on the same pairs is +0.57 in plain-gradient geometry. F5's second
moments hold no gradient of the new loss, so parameters the old losses barely move get very large preconditioned steps;
frozen-v AdamW is therefore not the update training with a new loss would make. Plain-gradient geometry is the one the
validated round-1 cosine uses. AdamW figures stay reported.

Bar 4 uses the held-out d kappa in plain-gradient geometry, and only if the reference is valid: the reward forecast's
held-out d kappa on recovery must be positive, interval excluding 0. If it is not, the in-pair d kappa is used under
the same reference condition. If neither reference is positive, bar 4 is uninformative and is reported as such.

Hunger is reported separately, as Codex suggests; Part 2 investigates it. A pass is not a repaired world. The run it
could earn is judged as in round 1: native energy timing due / not due, false drops, F5's recovery, cooldown and
danger gains, cows and arrows.

## Part 2: is hunger's missing request an input-side problem?

**What is already measured (F5 `RESULTS.md` 5, 6, 11.4, 11.7):**
- digit value at Mamba's input (MLP probe): health 1.00, food 0.35, drink 0.49, energy 0.47;
- digit change at the input (registration AP): health 0.998, food 0.19, drink 0.24, energy 0.59 (awake);
- uptake into the state at lag 0: health +1 0.94, food 0.037, drink 0.094, energy 0.055 (base rate ~0.02-0.03);
- the reset's trace in the pairs at checkpoint 1: 0.42 / 0.12 / 0.025 when it happens (recovery / energy / hunger),
  0.014 / 0.0073 / 0.0065 at the due step;
- new from round 1's saved gradients: hunger's separation-gradient norm is not smaller than energy's (2.5e-3 vs
  1.4e-3 at checkpoint 1; 6.6e-3 vs 8.7e-3 at 2; 9.7e-3 vs 1.5e-2 at 10). That weighs against "the input stops the
  request" and is why this needs testing rather than assuming.

**D1. Where each reset is lost (natural DEV frames).** The section-6 registration protocol (40k transitions, seed
2120, logistic on |change|, 5 episode folds, AP with an episode bootstrap) applied at three stages:
- the periphery tokens, |p_t - p_(t-1)| over all 81 x 16 numbers;
- Mamba's input, |x_t - x_(t-1)| (section 6 exactly);
- Mamba's corrected state, |b_t - b_(t-1)|.

Events: health drop, health rise; food drop (awake; the timer's reset), food rise (eating, which also resets hunger
in Craftax-Classic); drink drop (awake), drink rise (drinking also resets thirst); energy drop (awake). Checkpoints
1, 2 and 10. Section 6's four any-change rows are recomputed at checkpoint 10 as the validation; they must match.
How to read it:
- **input-side loss:** high at the tokens, low at the input;
- **perception-side loss:** already low at the tokens;
- **state-side loss:** high at the input, low at the state;
- **learned versus present from the start:** the input stage across checkpoints 1, 2 and 10 (alongside F4 / HUD-Change
  from section 6). Whether hunger is the only clock affected is read off the same table.

**D2. Substitution on the exact-history pairs: is the input the cause?** Frozen model, modified inputs only.
- **Hunger, amplified:** the two members' input-embedding difference at every history step is scaled so its total
  norm equals the energy pairs' median. Gradients flow through the scaled difference, so this is what a projection
  with that much more gain on the food change would give.
- **Hunger, sham:** the same added norm, in a seeded random direction, at the same steps.
- **Energy, attenuated:** the mirror; energy's difference scaled down to the hunger pairs' median.

Per variant: D at the due step, the separation-gradient norm, the all-cell cosine (both currencies), and the
digit-cell cosine (plain).

How to read it, fixed now, at checkpoint 1 or 2:
- **input-limited supported:** amplified hunger's plain all-cell cosine becomes positive (interval excluding 0), the
  sham's does not, and attenuated energy loses its request (interval including 0);
- **not supported:** amplified hunger stays at 0 (interval including 0);
- anything else is reported as mixed.

**D3 is the digit-cell cosine above.** If hunger's digit-cell cosine is also about 0 while energy's is clearly
positive, hunger's block sits upstream of the forecast; if both are clearly positive, the all-cell average is what
removes hunger's request.

Not covered: drink has no exact-history pairs, so D2 cannot be run for it; D1 covers it.

## Part 3 (user, 11-13 evening): does a forecast mode hurt or help its reactive mode?

My round-1 reply said TSP's forecast mode "likely cost the reactive mode". The evidence for that (F5 `RESULTS.md`
11.6) compared a fresh single-mode head with F5's trained shared head, which also differs in online training and in
its window distribution, so it does not isolate sharing. The reward head has the same two-mode design and its forecast
mode is what taught the clocks. Dreamer 4 (arXiv 2509.24527, Eq. 9) instead shares the trunk and gives each MTP
distance its own output layer.

`sharing.py`: fresh heads of the original classes on F5 checkpoint 10's frozen inputs (readout_fit's), with the same
initial weights, rows, windows and budget in every regime; only the objective differs. TSP (squared error of the pixel
mean, 20k steps): reactive only, forecast only, joint. Outcome (reward log-loss, 60k steps): reactive only, forecast
only, joint. Two initialisation seeds per regime.

**Reading, fixed now:** for each mode, joint against that mode alone (held-out TSP error on uniform windows; the
forecast's health / food / drink / energy error on unchanged frames; reward log-loss overall and on reward frames).
- **Harm:** joint is worse in both seeds by more than the seed-to-seed spread within the single-mode regime.
- **Help:** joint is better in both seeds by more than that spread.
- **Otherwise:** no difference detected.

This tests sharing in the head, on frozen inputs. It does not test what a shared objective does to the belief during
training.
