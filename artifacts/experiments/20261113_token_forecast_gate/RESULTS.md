# Option A gate results (2026-11-13): an all-cell token forecast with an MSE currency does not ask for the pixel clocks

Instrument and readings as fixed in `PLAN.md` (`gate.py`, `readings/ckpt_{01,02,10}.json`). Fresh token heads were
fitted on frozen F5 beliefs at checkpoints 1, 2 and 10. Pairs are the exact-history energy (32), hunger (24),
recovery (32) and cooldown (32) pairs from `../20261111_p1f5`. The reference is the reward forecast's direction
cosine on the same pairs, from `../20261111_p1f5/credit.py`'s saved gradients.

## Verdict

Neither currency qualifies. Both heads are forecasters: each beats its shuffled-belief control by 4-7x in its own
currency (bar 3).
- **Plain MSE** passes the nuisance bar at checkpoint 2 and asks weakly but consistently for energy. It never asks for
  hunger, and fails hunger's nuisance bar at checkpoint 1 (the +6/255 edit wins 24/24, reproducing Codex's check).
- **The pair targets here were computed under bf16 autocast,** which inflates small differences (the +3/255 foil by
  ~24%, the +6/255 by ~8%; the consequence is unchanged). With fp32 targets (round 2), the +6/255 foil wins 0/32 energy
  pairs at checkpoint 1 instead of 15/32. Hunger still fails at 24/24, and energy's consequence is only ~1.08x the
  +6/255 foil, so the energy bar is knife-edge.
- **Fixed precision** (1 / typical step change per channel, floored at 1% of its median) fails the nuisance bar at both
  early checkpoints. The static inventory row gets 5-8x the mean weight and map cells about 0.12, so a global
  brightness edit is priced above the consequences.

| Ckpt | Energy, plain: foil wins / cosine [95% CI] | Hunger, plain | Energy, precision | Hunger, precision | Reward on recovery (ref.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 15/32, **+0.081 [+0.026, +0.138]** (23/32) | 24/24, +0.001 [-0.049, +0.053] | 32/32, -0.026 | 24/24, +0.052 | +0.588 [+0.469, +0.688] |
| 2 | 0/32, **+0.056 [+0.029, +0.082]** (25/32) | 0/24, -0.014 [-0.067, +0.038] | 32/32, -0.041 | 24/24, +0.019 | +0.207 [+0.072, +0.349] |
| 10 | 0/32, -0.021 [-0.067, +0.022] | 0/24, +0.020 [-0.037, +0.078] | 0/32, +0.028 | 11/24, -0.087 | +0.753 [+0.646, +0.846] |

- **On energy, plain MSE's request is about a tenth of the reward forecast's on recovery, and it fades by checkpoint
  10.** On hunger there is none.
- **The token head reads the clocks the belief already holds.** At checkpoint 10, kappa is 0.28 on recovery and 0.36
  on cooldown, so an all-cell forecast would carry the reward-taught clocks. It does not create the missing ones.
- **Over 98% of each pair's token difference lies in the digit's cell,** with both currencies.

## The user's TSP question (checkpoint 10)

The reactive TSP's held-out error is 0.0020 with the frame's own belief and 0.0100 with another frame's belief (58,103
frames, uniform windows). Its output depends heavily on the belief. A mismatched belief is off-distribution, so this
shows dependence, not that the belief adds information beyond the coarse tokens.

## What this does not establish

- **Whether a currency in which a rare change is a surprise would qualify.** That is a categorical prediction over
  bins, as the reward head uses. It was not tested here.
- **Why hunger draws no request in either currency.** The food reset barely perturbs Mamba's state at the moment it
  happens (`../20261111_p1f5/RESULTS.md` 11.7: 0.025, against 0.12 for energy and 0.42 for recovery), and the food digit
  reaches Mamba's input at 35% exact. Both fit an input-side limit; neither is shown to be the cause.

---

# Round 2 (2026-11-13 evening): the categorical currency fails; hunger's missing request is not explained by its input

Plan: `PLAN.md` round 2 (committed before results in dda75104; instrument 2f627464, amended after the smokes before any
real result). `gate2.py`, `d1.py`, `summarise2.py`; readings in `readings/round2/`. The plain head reproduces round 1's
held-out losses exactly at every checkpoint (0.158709 / 0.134826 / 0.130782).

## Part 1: HL-Gauss cross-entropy does not qualify, and is worse than plain MSE on energy

HL-Gauss as fixed before the fit: 51 bins on the token range [-3, 3], sigma = 0.75 bin width (Farebrother et al. 2024
defaults). Energy pairs, plain / categorical. Nuisance is the +6/255 foil beating the consequence as the trained head
pays it; the matched update is d kappa per unit plain-gradient step learned on the other fold. The reference is the
reward forecast on the recovery pairs.

| Ckpt | +6/255 foil wins (+3) | Cosine | Matched update, held out | Reference (reward, recovery) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 9/32 (0) / **32/32 (32)** | +0.076 [+0.027, +0.127] / +0.036 [-0.020, +0.091] | +0.007 [+0.002, +0.013] / +0.016 [-0.017, +0.048] | -0.003 [-0.010, +0.003]: not valid |
| 2 | 0/32 (0) / **32/32 (26)** | +0.049 [+0.018, +0.080] / -0.030 [-0.067, +0.006] | +0.030 [+0.021, +0.039] / -0.022 [-0.048, +0.002] | +6.47 [+4.62, +8.38] |
| 10 | 4/32 (0) / 32/32 (32) | -0.021 [-0.070, +0.027] / +0.010 [-0.047, +0.067] | -0.008 [-0.042, +0.020] / -0.032 [-0.098, +0.015] | +29.8 [+18.7, +41.8] |

- **Bars (categorical, energy):**
  - fails nuisance at every checkpoint: a +3/255 brightness edit already costs the trained categorical head more than
    the energy drop in 26-32 of 32 pairs;
  - fails direction at checkpoints 1 and 2;
  - is a forecaster: held-out 1.94 against 4.19 with shuffled beliefs;
  - does not add over plain MSE: the reference is valid at checkpoint 2, where categorical minus plain is
    **-0.052 [-0.080, -0.024]**.
- **Hunger, either currency:** no request at any checkpoint (cosines -0.03 to +0.02, intervals spanning 0). The trained
  categorical head fails hunger's nuisance check 24/24 at +3/255 and +6/255.
- **Why:** with sharp HL-Gauss labels, small shifts on all 1,296 numbers outweigh a large change in 16, as Codex
  predicted. My claim that binning buys nuisance robustness was wrong.
- **How weak plain MSE's request is:** at checkpoint 2, plain MSE's matched update on energy is +0.030, against the
  reward forecast's +6.47 on recovery. Both are in the same units (fraction of the true due / not-due contrast
  reproduced per unit step), so the token forecast's push is ~200x weaker than the one that taught a clock. It is gone
  by checkpoint 10.
- **Matched-update instrument:** at checkpoint 1 the reference is invalid in every geometry and yardstick tried. The
  one-step held-out effect is not a reliable early predictor even for the loss that taught recovery; it becomes valid
  at checkpoint 2, when that clock starts forming (section 11.2 of `../20261111_p1f5/RESULTS.md`).
- **Instrument caveat:** per-pair cosines of an all-cell loss correlate only 0.68 (energy) and 0.42 (hunger) between
  bf16 and fp32 targets; the means agree (+0.081 vs +0.076). Only bootstrap means carry weight.

**The request when only the digit's cell is scored** (plain / categorical; same separation):

| Ckpt | Energy | Hunger | Recovery | Cooldown |
| :--- | :--- | :--- | :--- | :--- |
| 1 | +0.058 / +0.016 | +0.020 / -0.129 | -0.050 / **+0.322** | +0.111 / **+0.231** |
| 2 | -0.074 / -0.139 | +0.184 / +0.150 | **+0.257 / +0.181** | **+0.163 / +0.287** |
| 10 | -0.019 / +0.062 | +0.158 / -0.011 | **+0.904 / +0.644** | **+0.769 / +0.632** |

Bold: interval excluding 0. The health-cell clocks get a clear request once the other 80 cells are removed, early in
the categorical case and growing as the belief holds them. Energy and hunger get none detectable; hunger's plain point
estimates at checkpoints 2 and 10 are positive with intervals spanning 0. Energy's weak all-cell request does not come
from its own cell.

## Part 2: where each reset is lost, and whether the input explains hunger

**D1, the section-6 registration protocol at three stages** (AP; drops awake; tokens / Mamba's input / Mamba's state).
The checkpoint-10 input stage reproduces section 6 exactly (0.99755 / 0.18360 / 0.23597 / 0.69090).

| Event (base rate) | Tokens, ckpt 10 | Input: ckpt 1 / 2 / 10 | State: ckpt 1 / 2 / 10 |
| :--- | ---: | :--- | :--- |
| Food drop (0.028) | 0.962 | 0.034 / 0.038 / 0.065 | 0.032 / 0.033 / 0.036 |
| Drink drop (0.031) | 0.862 | 0.083 / 0.091 / 0.112 | 0.075 / 0.086 / 0.082 |
| Energy drop (0.020) | 0.907 | 0.106 / 0.274 / 0.257 | 0.031 / 0.116 / 0.072 |
| Health +1, recovery (0.018) | 0.803 | **0.178 / 0.664 / 0.789** | 0.461 / 0.832 / 0.921 |
| Health drop (0.019) | 0.918 | 0.672 / 0.759 / 0.807 | 0.710 / 0.853 / 0.915 |
| Food rise, eating (0.005) | 0.955 | 0.037 / 0.223 / 0.519 | 0.270 / 0.302 / 0.320 |
| Drink rise, drinking (0.016) | 0.652 | 0.088 / 0.106 / 0.132 | 0.208 / 0.178 / 0.187 |

- **The periphery tokens carry every reset** (0.80-0.96 at every checkpoint). What is lost is lost in Mamba's input
  projection, the single linear map of all 81 cells to 256 numbers.
- **Hunger is not the only clock affected.** Drink's reset is lost there too, and energy's is partly registered at the
  input (0.26) but not taken into the state (0.07).
- **The input registers what a loss asks for, and learns to.** Recovery's reset goes from 0.18 to 0.66 at the input
  between checkpoints 1 and 2, the interval in which the reward forecast starts separating the recovery branches.
  Eating (rewarded the first time as an achievement) goes from 0.04 to 0.52. Nothing changes for food or drink drops.
  So the input's weakness is the same missing request one layer down, not a fixed limit of the projection.
- Section 6's food figure (0.184, any change) was mostly eating. The timer's own reset is at 0.065.
- **Limit:** a linear probe on |change| measures how distinctively a change is registered among the other cells'
  changes. Section 5's value probe (food 35% exact at the input) covers recoverability.

**D2, input substitution on the pairs** (plain all-cell cosine; D = squared distance of the members' predicted beliefs
at the due step, relative to their size):

| Ckpt | Hunger | Hunger amplified to energy's input difference | Hunger sham | Energy | Energy attenuated to hunger's |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | +0.011 [-0.030, +0.053] | x6.7: -0.012 [-0.056, +0.030] | +0.042 [-0.022, +0.105] | **+0.076** [+0.027, +0.127] | x0.15: -0.010 [-0.075, +0.055] |
| 2 | -0.010 [-0.066, +0.043] | x7.3: +0.027 [-0.016, +0.073] | +0.043 [-0.012, +0.094] | **+0.049** [+0.018, +0.080] | x0.14: +0.017 [-0.021, +0.053] |
| 10 | +0.010 [-0.053, +0.072] | x4.9: -0.043 [-0.102, +0.010] | -0.042 [-0.097, +0.012] | -0.021 [-0.070, +0.027] | x0.20: -0.022 [-0.065, +0.018] |

- **Pre-registered reading: input-limited is not supported at any checkpoint.** Giving food's reset energy's input
  size creates no request, and behaves like an equal random perturbation (the sham). Shrinking energy's to food's
  size removes energy's weak request.
- **So input size is necessary for energy's request but not sufficient for hunger's.** Amplification raises D at the
  due step 1.6-2.8x (e.g. 2.6e-5 to 5.7e-5 at checkpoint 2, comparable to energy's 4.9e-5), so the state difference
  reaches energy's size without a request appearing.
- **Reading:** amplifying food's change along the direction the untrained projection gives it is no better than a
  random direction. Energy's direction has been partly shaped (its input AP rose from 0.11 to 0.27), and recovery's
  reset enters through a dedicated channel, the reward input token. A first-order request only amplifies structure the
  belief and the readout already share, and the food and drink drops have none. This is an interpretation of D1 and
  D2 together, not a separate test.

## What round 2 establishes

1. **Neither token-forecast currency teaches the pixel clocks.** HL-Gauss is worse than plain MSE on nuisance and on
   the energy request. Plain MSE's energy request is real early but ~200x weaker than the reward forecast's on recovery,
   and gone by checkpoint 10. Neither asks for hunger.
2. **Every pixel clock's reset is lost at Mamba's input or state, not in perception.** Food and drink are lost at the
   input, energy at the state. Health and eating, which losses ask for, are registered, and the input learns them when
   asked.
3. **Hunger's missing request is not caused by the input's weakness alone (D2).** What the requested clocks have is a
   route by which the reset reaches the state in a form something already reads: the reward token for recovery, and the
   reward readout itself.

Not established: what would bootstrap a clock whose only evidence is pixels; Part 3 (sharing) is below.
