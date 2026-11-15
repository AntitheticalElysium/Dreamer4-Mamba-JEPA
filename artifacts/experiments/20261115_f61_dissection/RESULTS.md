# P1-F6.1 dissection: results (2026-11-15)

## Check 1: the sleep phase where no loss asks for it (`sleep_phase.py`, `sleep_phase.json`)

F6.1 and F6 at checkpoint 10, all DEV encoded on CPU (F6's natural APs reproduce its GPU reference: food 0.0372,
asleep rise 0.1703 against 0.1696), the gaze test's probes, folds 0-3 fit, fold 4 read; "beyond time and light" is the
held-out log-loss gain of a probe with Codex's time-and-light reader as a fixed offset over that reader alone, nats
per eligible frame; F6.1 minus F6 with episode-bootstrap 95% intervals.

| Asleep energy rise | Frames / rises | Time-light AP | AP F6 / F6.1 | Beyond time and light F6 / F6.1 | F6.1 - F6 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All asleep frames | 10,153 / 842 | 0.097 | 0.170 / 0.401 | 0.021 / 0.060 | 0.040 [0.030, 0.052] |
| WAKE_UP not yet paid | 2,664 / 227 | 0.106 | 0.306 / 0.823 | 0.050 / 0.169 | 0.119 [0.100, 0.137] |
| WAKE_UP already paid | 7,489 / 615 | 0.088 | 0.118 / 0.228 | 0.012 / 0.027 | 0.015 [0.010, 0.022] |
| Paid, 8 constant asleep frames (the gaze test's `after_reset`) | 2,510 / 609 | 0.254 | 0.254 / 0.338 | -0.001 / 0.011 | 0.012 [0.003, 0.025] |

- **The claim mostly holds, with a small general part.** The belief carries the phase strongly where the reward
  forecast prices it (first sleeps: AP 0.82) and weakly where nothing asks (repeat sleeps: 0.23, against F6's 0.12);
  on the strict repeat-sleep cohort it adds 0.011 nats beyond time and light, where F6 adds nothing. Competing
  explanation (i) is therefore partly right: some of what was learned on first sleeps transfers. Explanation (ii)
  (elapsed time, lighting) is ruled out for both cohorts: the time-and-light reader alone reaches AP 0.09-0.11 on
  asleep frames.
- **Awake pixel clocks: nothing, in either world.** Food, drink and awake energy drops add at most 0.0009 nats beyond
  time and light (natural, steady8, after_reset), F6.1 minus F6 within +-0.0005 everywhere. Recovery, the
  reward-priced reference: 0.070 against 0.061 (F6.1 minus F6 0.009 [0.006, 0.011]).

## Check 2: waking on repeat sleeps (`sleep_paid.py`, `sleep_paid.json`; 48 quiet scenes with WAKE_UP already earned, 357 eligible episodes)

| Checkpoint 10 | Phase advantage (steps) | Real / shuffled W1 | p | Wait ratio | Awake by 8 / 12 / 16 (simulator 0.63 / 1 / 1) | Early / late or none |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| F6.1 | 0.82 | 7.28 / 8.10 | 0.0001 | 1.96 [1.73, 2.28] | 0.23 / 0.47 / 0.73 | 0.086 / 0.859 |
| F6 | 0.34 | 15.12 / 15.46 | 0.018 | 2.98 [2.62, 3.44] | 0.23 / 0.33 / 0.44 | 0.135 / 0.823 |

Repeat sleeps are timed too, though less than first sleeps (F6.1's quiet bank, not yet paid: advantage 1.31 steps,
real W1 5.41): the general part of the belief's phase (check 1) reaches generation. F6.1's competing explanation (b)
("repeat sleeps stay untimed") is rejected in its strong form. Late or missing wakes remain the rule in both worlds.

## Check 3: where the late waking comes from (`hazard.py`, `hazard.json`)

Generated energy-rise hazard per asleep step at energy 8, by step relative to the scene's due step (the phase's own rise
step; on the repeat bank the earliest over the simulator's draws, 4 of 48 scenes having an interrupted draw):

| Bank, world | Before due | At due | 1-3 after | 4+ after | Rank corr., generated vs true rise step | Median generated / true rise step |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Quiet, F6.1 | 0.009 | 0.049 | 0.107 | 0.142 | 0.82 | 11 / 5.1 |
| Quiet, F6 | 0.038 | 0.072 | 0.043 | 0.085 | 0.28 | 11.75 / 5.1 |
| Quiet, F5 | 0.030 | 0.050 | 0.047 | 0.053 | 0.34 | 15 / 5.1 |
| Repeat, F6.1 | 0.017 | 0.049 | 0.080 | 0.134 | 0.61 | 11.25 / 5.1 |
| Repeat, F6 | 0.029 | 0.049 | 0.038 | 0.063 | 0.48 | 15.5 / 5.1 |

F6.1 is a timed clock with a late, smeared release: it almost never rises early and orders the scenes correctly, but
releases the rise only about 5% of the time at the due step and 11-14% per step after it (a flat hazard matching the
mean wait would be 0.16). F5 and F6 are near-flat hazards.

## Check 4: the belief expects the rise the denoiser does not draw (`rollout_belief.py`, `rollout_belief.json`)

Along each world's own rollouts (generation bitwise equal to the queue's run: the known answer holds), the world's own
asleep-rise probe on the belief predicted before each generated frame, while the draw is still asleep at energy 8
(step 0, whose belief is the recorded prefix's, excluded); on DEV fold 4 the same probe gives 0.27 at true rises and
0.07 elsewhere (F6: 0.12 / 0.08).

| Bank, world | Belief P(rise) before / at / 1-3 after / 4+ after due | Drawn rise, same steps |
| :--- | :--- | :--- |
| Quiet, F6.1 | 0.050 / 0.613 / 0.838 / 0.763 | 0.010 / 0.052 / 0.107 / 0.142 |
| Repeat, F6.1 | 0.139 / 0.381 / 0.500 / 0.609 | 0.017 / 0.045 / 0.080 / 0.134 |
| Quiet, F6 | 0.088 / 0.164 / 0.239 / 0.427 | 0.043 / 0.079 / 0.043 / 0.085 |
| Repeat, F6 | 0.089 / 0.114 / 0.125 / 0.211 | 0.032 / 0.045 / 0.038 / 0.063 |

**The phase stays readable in the world's own beliefs; where the delay enters is not located.** The probe's score
rises 12-fold at the due step along F6.1's rollouts and stays high after it, so the competing explanation (the belief
loses the phase along generated frames) is rejected. The probe's sigmoid is not a calibrated event probability (Codex,
NOTEBOOK 11-15: on the draws that actually rise at the due step its AP is 0.062 / 0.067 against base rates 0.052 /
0.042), so "the belief expects 0.61, the denoiser draws 0.05" (my first wording) overstated it: the reading shows a
due-or-overdue signal reaching generation, and that, at the due step, which draws rise barely depends on it. Whether
the denoiser's memory path, its cache or recursion holds the release back is the next test's question.

## Check 6: sampler or learned conditional (`sampler_steps.py`, `sampler_steps.json`; F6.1, quiet bank, same seeds)

| Sampler | Before due | At due | 1-3 after | 4+ after | Rank corr. | Median generated rise step |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| K = 4 (trained recipe) | 0.009 | 0.049 | 0.107 | 0.142 | 0.82 | 11 |
| K = 8 | 0.015 | 0.064 | 0.139 | 0.154 | 0.82 | 9.5 |

Doubling the evaluations raises the release by about a quarter and leaves it far below the belief's expectation: the
learned conditional is the main cause, the few-step sampler a minor one.

## Check 5: why the mining stage stopped (`mining_check.py`, `mining_check.json`)

Numerical: the stored frames equal the recorded ones on all 402 pairs; the largest token difference is 0.0408 for
F6.1 (one pair over the stage's 0.04; median 0.012, 99th percentile 0.032) and 0.0398 for F6 (none over), at token
magnitudes up to 2.4 where bf16's spacing is 0.016. The stage reran with the pre-check recorded (check 5b, below).

**Check 5b** (`mining_rerun.py`, `mining_rerun.json`; the stage's own code with the pre-check recorded): coal DO/DO
0.067, NOOP/DO 0.019, DO/NOOP 0.001 (F6 0.090 / 0.013 / 0.000, F5 0.055 / 0.002 / 0.001); wood 0.954 / 0.693 / 0.040
(F6 0.938 / 0.249 / 0.018). Coal's increment is rarely drawn in every world since F5; for wood, F6.1's denoiser draws
the increment from the DO member's belief even under a NOOP action (0.69 against F6's 0.25).

## Check 7: the use lags the belief (check 3's reader on the checkpoint-3 rollouts)

| Quiet bank | Before / at / 1-3 after / 4+ after due | Rank corr. | No rise in 48 steps |
| :--- | :--- | ---: | ---: |
| F6.1 ckpt 3 (belief AP 0.363) | 0.019 / 0.044 / 0.040 / 0.062 | 0.26 | 0.230 |
| F6.1 ckpt 10 (belief AP 0.471) | 0.009 / 0.049 / 0.107 / 0.142 | 0.82 | 0.016 |
| F6 ckpt 3 | 0.002 / 0.008 / 0.007 / 0.008 | 0.04 | 0.730 |
| F6 ckpt 10 | 0.038 / 0.072 / 0.043 / 0.085 | 0.28 | 0.023 |

The world learned to use the belief's phase between checkpoints 3 and 10 (ordering 0.26 to 0.82, the overdue release
doubling) while the belief's own AP rose only 0.36 to 0.47: the use lags, as recovery's did. The release at the due
step itself did not move (0.044 to 0.049).

## Check 8: each event's size in the denoiser's currency (`token_share.py`, `token_share.json`; F6.1 tokens, DEV fold 4)

| Event | Frames | Squared token change at the digit's cell | Whole frame | Cell's share of the frame |
| :--- | ---: | ---: | ---: | ---: |
| Asleep energy rise | 842 | 6.67 | 29.9 | 0.46 |
| Awake energy drop | 1,252 | 6.91 | 149.5 | 0.19 |
| Food drop | 1,580 | 8.72 | 150.4 | 0.21 |
| Drink drop | 1,837 | 11.45 | 150.1 | 0.24 |
| Recovery (+1 health) | 703 | 23.07 | 179.4 | 0.25 |

On the frame where it happens, a clock event is a fifth to a half of what changes in G1's token target; the asleep rise
is the same size as the awake energy drop on a frame with a fifth of the change (the map is dimmed and static asleep;
the HUD is rendered after the sleep tint, so it is not). The earlier "about 1%" was the share over all frames, set by
rarity, not by the size of the event where it occurs.

## Check 9: the use plateaued by checkpoint 8 (`trend.py`, `trend.json`)

Quiet bank, F6.1: rank correlation 0.26 / 0.61 / 0.81 / 0.82 at checkpoints 3 / 5 / 8 / 10; release at the due step
0.044 / 0.042 / 0.070 / 0.049. The ordering was learned by checkpoint 8; the due-step release never moved. A longer
run of this recipe is not expected to fix it.

## Check 10: on clean histories the rise ignores memory; the clean cache suppresses it (`swap.py`, `swap.json`)

32 asleep exact-history pairs (identical last 8 frames; the previous rise 10 vs 8 steps back; branch 0 due at the next
frame), frozen F6.1, one generated frame, 32 draws per cell, common random numbers:

| Cache context signal | Due (c0 m0) | Delayed (c1 m1) | Current-memory effect | Cached-memory effect |
| :--- | ---: | ---: | :--- | :--- |
| 7 (trained) | 0.032 | 0.033 | 0.000 [-0.002, 0.002] | -0.001 [-0.002, 0.000] |
| 4 (stressed) | 0.272 | 0.266 | +0.003 [0.000, 0.008] | +0.003 [0.000, 0.009] |

**Retracted as a reading of the denoiser (check 10b, `swap_belief.json`):** F6.1's own probe does not tell these
pairs' branches apart either (logit due minus delayed -0.008 [-0.029, 0.007]). The pairs were built by editing the hidden
fatigue mid-sleep, so both branches share their sleep onset, and in natural play the phase is the steps since onset
(mod 11): the belief rightly saw one scene. A zero memory effect was therefore expected and says nothing about the
denoiser's use of memory. What stands: a stressed cache raises untimed digit changes eightfold (+0.24 [0.15, 0.33]).
Check 10c rebuilds the pairs with natural histories (sleep onsets 2 steps apart). Free-running timing (rank 0.82) is carried by overdue states: on the quiet bank it
holds in the 15 scenes with no sleep onset inside the 8-frame cache (0.74; 0.37 in the 17 with one), as the hazard
ramps after the due step (check 3). The memory route responds only once the belief's signal is large.
