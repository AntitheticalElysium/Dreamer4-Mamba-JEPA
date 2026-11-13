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
