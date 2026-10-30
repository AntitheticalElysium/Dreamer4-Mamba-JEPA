# M0.5 arm G: dissection (10-30, after RESULTS.md)

User, 10-30: what were the goals, were they reached, where exactly did it fail, do we know exactly or are we reading
broad metrics. Everything here uses arm G's checkpoints and evaluation; nothing is attributed to placement (arm R
not run). Scripts and outputs: `dissect/` (error_modes, glimpse_field, token_geometry, glimpse_variance; images
`dissect/img_*.png`).

## The goal, as set in the transcript
10-29: let the fovea fixate where it makes the most difference to learning (movement first, then the sparse
consequences starved of gradient), rewarded later by the model's improvement; first a cheap test with the fovea
placed by hand where the model struggles. 10-30: "our theory was that extra gradients would help these predictions;
if the gradient share is higher they should have progressed by definition". M0.5 tests that with joint training,
the generated glimpse as the primary readout.

## What the class rates hid

**1. Appearance progressed; increments did not, even below the sampling threshold.** The denoiser's clean estimate
of the glimpse at the consequence cell (real target noised at sigma; projection on the reader's real value means,
1 = the event, 0 = copy; `dissect/glimpse_field.json`), mean at sigma 20, base / 300 / 750 / 1500:
drink appearance 0.29 / 0.45 / 0.69 / 0.65; wood appearance 0.23 / 0.40 / 0.58 / 0.62; drink increment
0.35 / 0.35 / 0.35 / 0.35; wood increment 0.34 / 0.37 / 0.41 / 0.46. Drinking is deterministic (labels: DO facing
water at 1-8 goes up on 4,192 / 4,192 dev frames), so a learned field reaches 1.

**2. Appearance is a coherent, action-driven event; drink increments ignore the action.** Appearance: with the NOOP
belief the icon never appears (drink 0-3 of ~200 at every checkpoint), and when the glimpse shows it the generated
periphery agrees (27 / 27 at 1500, 58 / 64 at 750). Drink increments: the same outcomes with and without the action
at every checkpoint (1500: up 148 vs 148, down 302 vs 267, of ~1,790); a glimpse "success" agrees with the
periphery in 2-5 of 44-59 cases. Wood increments do use the action (up 153 vs 69 at 1500, 129 vs 73 at base).

**3. The drink digit is retrieved from context.** Drink increment samples (MLP verdicts, see 6): copy 66%, down 19%
(the counter's real one-step decay is 3.8%), up 4%, other 11%. Down is 26% where start - 1 was shown 2-8 frames
earlier (consecutive drinking) against 8-10% where it was not; a third of the "other" values are the counter's own
older values; other values match a neighbouring counter in 17-32% against 12-21% by chance (`error_modes.json`).

**4. In the glimpse token the denoiser is trained on, the digit is nearly invisible; in the periphery it is not.**
Real tokens of the counter's cell (`token_geometry.json`, `glimpse_variance.json`):

| | glimpse (64-d) | periphery (16-d) |
|---|---|---|
| step v -> v + 1 / within-value spread (99th pct radius) | drink 0.30, wood 0.23 | drink 3.9, wood 4.2 |
| step v -> v + 1, share of the token's squared norm | 0.9%, 0.6% | 32%, 25% |
| step 0 -> 1 (icon appears) / spread | 2.3, 2.7 | 8.1, 9.1 |
| variance explained by the digit, frames showing the icon | 17%, 5% | 97%, 96% |

Unchanged by M0.5 (1500: 0.29 / 0.24 / 17% / 5%). Neighbouring cells explain 5-7% each, daylight 6%. So the extra
gradient placement delivered (Part 2: ~8x the consequences' share in the glimpse) lands on a loss in which the icon
appearing is large and a digit change is smaller than the variation within one digit. The field moved where that
loss is large (1) and not where it is small, except wood increments (slow, action-routed): geometry is not the
whole story for drink, whose action binding is absent (2). A candidate, not established: at values >= 1 the drink
counter's recorded changes are mostly exogenous (decay 3.8% vs up 1.5% per frame), wood's are action-caused.

**5. The fovea's pixel grounding is digit-blind.** The fovea reconstruction's decoder mean renders a HUD cell as its
icon with a "1" whatever the value, real tokens included (`img_base_wood_increment.png`: real 2, 3, 4 decode as 1;
drink's icon decodes wrong as well). Its log-scale depends on lighting and position only (shared with TSP): -1.56 on
HUD counters (-0.06 to -1.27 on health, food, drink, energy) against -4.56 on map cells, a ~20x wider pixel
distribution, unchanged by M0.5. A learned scale that widens where the fit is poor down-weights those pixels and
stalls the mean there (Seitzer et al., ICLR 2022, arXiv 2203.09168); wood is digit-blind at a moderate scale
(-3.0), so this is not the whole mechanism. The one glimpse objective in pixel space therefore asks nothing of the
digit.

**6. Tables: the world places the table, the glimpse does not render it.** In the guided route, within the same
generated frame, the periphery shows the new table in 26-28% of samples (16% at 750) where the glimpse keeps the
empty cell (glimpse table 0.1-0.9%). The table's glimpse step is larger than a digit's (9-18% of the token's squared
norm), so this is open.

## What the instruments can and cannot say
- Digit values of GENERATED glimpse tokens are not verifiable: neighbouring values' supports overlap completely (a
  generated increment token lies inside the start, end and read values' supports in 97-100% of samples), so the
  verdict is the MLP reader's alone. Its verdicts carry signal (they depend on the history and the action as
  above), but the exact increment rates, and success vs other on appearances, are not verified. RESULTS.md's
  "generation vs reader misread" question is replaced by this. Icon presence (appearance vs copy) IS verified.
- The pixel route reads icon presence only (5).
- Regeneration reproduces the evaluation's readings on 97-98% of samples (GPU nondeterminism); routing 99%.

## Open
- The 750 -> 1500 decline (drink appearance samples 0.42 -> 0.19, icon present among readable 0.89 -> 0.70; field
  0.69 -> 0.65): checkpoint noise at a constant learning rate or a reversal; nothing here separates them.
- Whether placement caused the appearance gain (arm R).
- Why the glimpse does not render tables; why drink's action is not bound to its digit.

## Against P1's failure list (user, 10-30: did we help them; what is left; what regressed)
P1's failures: generation diagnosis A-H (`../20261029_generation_diagnosis/RESULTS.md`) and S1.3 (`saccade_m0/p1_world1/
s1/results.json`). guided_1500 vs base, M0.5's own instruments; not attributed to placement (arm R not run).
P1 measured drink and wood as aggregates; first units are 10.6% of drink and 25.1% of wood events on dev (DO facing
water / a tree with +1: 497 of 4,689, 2,088 of 8,314; train-400 11.4% / 24.5%), increments the rest.
- Helped (2, partial): drink first unit (glimpse 0.06 -> 0.19, 0.42 at 750; periphery 0.13 -> 0.18), wood first unit
  (glimpse 0.12 -> 0.37; periphery 0.16 -> 0.24).
- Unchanged (6): drink increments (field flat 0.35, action unused), wood increments (glimpse 0.083 -> 0.101, field
  0.34 -> 0.46: below the sampling threshold), sapling (0 vs 0.09 recorded), damage on low health (0.000-0.001 vs
  0.14-0.21), damage beside a zombie (periphery 0.002; glimpse verdicts 0.072 -> 0.035, digits unverifiable),
  zombies / cows under-generated (generated - real -0.19 -> -0.16, z 2.1; -0.22 -> -0.20, z 1.4; independent SEs).
- Worse (2): placed tables in the periphery (0.242 -> 0.224, NEGATIVE; 0.119 at 750; glimpse 0.001 -> 0.009), arrows
  over-generated (0.13 -> 0.17 vs 0.09 real, z 3.8). Outside the list: P1's held-out next-frame loss +3.6% at every
  noise level; skeletons 0.136 -> 0.120 vs 0.157 (z -2.0).
- Not measured by M0.5 (5): blocked moves predicted to shift, the hidden zombie cooldown, deaths in generated
  successors (their precursor, damage on low health, is still ~0), the production sampler's suppression (unchanged by
  construction), C2ST (0.76 at P1).
Module displacement (|ft_1500 - ckpt_10| / |ckpt_10|), beside P1's own checkpoints 5 -> 10: fovea encoder 0.012
(0.023), periphery layers 0.004 (0.009), belief 0.013 (0.026), denoiser 0.017 (0.036), pixel decoder 0.020 (0.037):
M0.5 moved every module about half as far as P1's second half did; the glimpse digit geometry did not move.
