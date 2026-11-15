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

## Check 11: F6.1's own tokens feed a slow trace that times the awake clocks (`token_trace.py`, `token_trace.json`)

Per cell, the squared token change filtered at time constants 8 and 32; a shared label-free next-change predictor (no
episode step); AP at the after_reset clock events (DEV fold 4; digit steady 8 frames, an earlier sleep or intake):

| | Food | Drink | Awake energy |
| :--- | ---: | ---: | ---: |
| Traces over the full history | 0.306 | 0.338 | 0.449 |
| Traces started 8 frames back | 0.050 | 0.067 | 0.034 |
| Time-and-light reader (gaze test) | 0.213 | 0.241 | 0.043 |
| F6.1's belief, linear probe (check 1) | 0.052 | 0.072 | 0.044 |

The information is in F6.1's tokens, and a one-step objective uses it once a per-cell recency code exists; it never
reaches the belief. Codex's fitted luminance write on F6 tokens reached 0.11 / 0.21 / 0.06; the cell's own token change
is the better generic scalar.

## Fine-tune: SWAX's stochastic short window on the denoiser (`swax_ft.py`, `swax_ft.json`; 2,000 updates, G1 only)

| | F6.1 | 1,000 updates | 2,000 updates (bar) |
| :--- | ---: | ---: | ---: |
| Swap, due / delayed rise (trained cache) | 0.034 / 0.017 | 0.069 / 0.059 | 0.060 / 0.031 (due >= 0.10) |
| Current-memory effect | +0.012 [0.006, 0.019] | +0.008 [0.003, 0.015] | +0.024 [0.009, 0.042] (> 0.019) |
| Quiet bank: before / at / 1-3 after / 4+ after due | 0.009 / 0.049 / 0.107 / 0.142 | 0.006 / 0.052 / 0.097 / 0.123 | 0.014 / 0.059 / 0.161 / 0.184 (before <= 0.02) |
| Rank correlation | 0.82 | 0.78 | 0.81 |
| Monitor G1 (guard: within 10% of 0.00383) | 0.00383 | 0.00407 | 0.00374 |

**Below the fixed bar.** Memory's effect doubled and the overdue release rose by half, with no early rises and G1
unharmed, but the release at the due step barely moved (0.049 to 0.059; the bar asked for a due rise of at least 0.10 on
the swap, reached 0.060). A forced reliance on memory softens the cache shortcut without making the denoiser draw the
change when it is due.

## Check 12: access or incentive? (`traces_ft.py`, `traces_ft.json`; instrument check `traces_check.py`)

Per-cell recency traces (check 11's, tau 8 / 32) as a zero-initialised input to each periphery cell's denoiser token;
denoiser fine-tuned on G1 alone from F6.1, 2,000 updates. One generated frame on the exact-history pairs; due / delayed
are each branch's own memory and traces (32 draws per pair).

| Bank | update 0 (= F6.1) | 1,000 | 2,000 | Trace effect at 2,000 | Memory effect at 2,000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Awake energy (32) | 0.007 / 0.007 | 0.003 / 0.003 | 0.002 / 0.003 | -0.001 [-0.003, 0.000] | 0.000 |
| Food (24) | 0.089 / 0.089 | 0.076 / 0.074 | 0.064 / 0.064 | 0.000 [0.000, 0.000] | 0.000 |
| Recovery (32) | 0.643 / 0.000 | 0.490 / 0.000 | 0.662 / 0.000 | -0.001 [-0.003, 0.000] | +0.664 [0.530, 0.788] |
| Asleep onset (32) | 0.034 / 0.017 | 0.075 / 0.074 | 0.016 / 0.012 | +0.000 [-0.001, 0.003] | +0.003 [0.000, 0.007] (was +0.018) |

G1 0.00372 at 2,000 (SWAX fine-tune 0.00374); the map's weight norm grew steadily to 0.25.

**The "incentive" reading of the bar does not stand: checks 13-16 show the null was the input's form.** Awake energy
stays below 0.05 with no separation. Food draws at a rate that is timing-blind (0.064 in both branches and in both
exchanges, above 0.05, so "as is" under the bar). The trace effect is zero on every bank, and on the asleep onset the
drawn rise and memory's share both fell (memory +0.018 to +0.003). The traces do separate these pairs (check 13), and
the same fine-tune with each cell's exact recency in their place learns the timing (checks 16, 16b).

Instrument check (`traces_check.py`, denoiser at 2,000): the branches' traces do differ at the HUD cell (mean due
minus delayed at tau 8 / 32: energy -0.011 / -0.009, food -0.019 / -0.009, recovery -0.23 / -0.11, asleep onset
-0.14 / -0.02). After the trained map, that difference is 14% (energy), 23% (food), 58% (recovery) and 27% (asleep)
of the cell's trace input. The evidence reached the denoiser's input, and G1 did not teach it to draw timing from it
in 2,000 updates. Bounds: one seed; a fine-tune from F6.1's denoiser (its learned conditional may resist a new input).

## Check 13: the traces separate the pairs (`use_check.py separable`, `separable.json`)

Check 11's next-change predictor refitted (APs reproduced exactly: 0.3057 / 0.3377 / 0.4489), on both branches at the
due transition:

| Bank | P(change) due / delayed | Due minus delayed [95%] | Due above delayed | 8-frame traces | log(1 + trace) gap, in spreads (tau 8 / 32) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Awake energy (32) | 0.320 / 0.046 | 0.274 [0.269, 0.277] | 32 / 32 | 0.008 / 0.008 | 0.017 / 0.016 |
| Food (24) | 0.177 / 0.045 | 0.132 [0.127, 0.137] | 24 / 24 | 0.010 / 0.010 | 0.030 / 0.018 |
| Recovery (32) | 0.071 / 0.068 | 0.003 [0.001, 0.006] | 0.64 | 0.008 / 0.008 | 0.29 / 0.12 |
| Asleep onset (32) | 0.055 / 0.021 | 0.034 [0.020, 0.048] | 32 / 32 | 0.011 / 0.016 | 0.16 / 0.03 |

**Separable** (fixed bar met on awake energy and food). The timing is in the traces on every pair. But the gap in the
form check 12 fed the denoiser is 2-3% of that input's spread over cells: a digit change 22-30 frames back has
decayed to a few hundredths beside the map cells' scroll changes. (Recovery's timing is not in the traces at all: the
health digit's history does not set the recover counter.)

## Check 14: the denoiser's conditional never used them (`use_check.py expect`, `expect.json`)

The first sampler step's prediction at the digit's cell, projected on the due branch's true change (its own
probability of the change), beside the final frame and the drawn rate (both reproduce check 12 exactly):

| Bank | F6.1: first step due / delayed | F6.1 final / drawn (due) | Traces 2,000: first step due / delayed | First-step trace effect [95%] | Drawn (due) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Awake energy | 0.009 / 0.009 | 0.001 / 0.007 | 0.015 / 0.015 | -0.0000 [-0.0001, -0.0000] | 0.002 |
| Food | 0.101 / 0.101 | 0.089 / 0.089 | 0.076 / 0.076 | 0.0000 [-0.0001, 0.0002] | 0.064 |
| Recovery | 0.647 / 0.002 | 0.644 / 0.643 | 0.654 / -0.010 | -0.002 [-0.003, -0.001] | 0.662 |
| Asleep onset | 0.081 / 0.071 | 0.045 / 0.034 | 0.075 / 0.063 | -0.0005 [-0.0006, -0.0004] | 0.016 |

**Not learned.** The fine-tuned denoiser's own probability is the same with its own and with exchanged traces, so the
sampler is not what hides a learned use. Where the probability is graded (asleep onset 0.08), the few-step sampler
then roughly halves it (final 0.045, drawn 0.034); where it is sharp (recovery 0.65), the sampler keeps it.

## Check 16: given a readable form, the denoiser uses the timing (`recency_ft.py`, `recency_ft.json`, `recency_expect.json`, `recency_pairs.json`)

Check 12 with each cell's exact recency (frames since its token last changed by more than 1) through a zero-initialised
embedding. On the pairs the due frame sits 30 / 22 frames after the energy digit's last change (due / delayed), food
25 / 17, recovery 25 / 17, asleep onset 10 / 7.1. Update 0 reproduces F6.1 exactly.

| Pairs | Drawn due / delayed, F6.1 | 1,000 updates | 2,000 updates | Recency effect at 2,000 [95%] | First step due / delayed at 2,000 (F6.1) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Awake energy | 0.007 / 0.007 | 0.002 / 0.000 | 0.008 / 0.000 | +0.008 [0.004, 0.013] | 0.040 / 0.005 (0.009 / 0.009) |
| Food | 0.089 / 0.089 | 0.112 / 0.051 | 0.130 / 0.000 | +0.130 [0.105, 0.152] | 0.152 / 0.009 (0.101 / 0.101) |
| Recovery | 0.643 / 0.000 | 0.578 / 0.000 | 0.746 / 0.000 | +0.121 [0.071, 0.172] | 0.723 / -0.011 (0.647 / 0.002) |
| Asleep onset | 0.034 / 0.017 | 0.069 / 0.051 | 0.041 / 0.003 | +0.020 [0.012, 0.029] | 0.093 / 0.068 (0.081 / 0.071) |

G1 0.00362 at 2,000 (the trace fine-tune 0.00372, F6.1's monitor 0.00383); the embedding's norm was still growing
(1.83).

**Food: form** (the fixed bar: due 0.130 >= 0.10, due minus delayed above zero, positive recency effect). **Awake
energy: as is**: a learned use at the first step (0.040 against 0.005, due minus delayed 0.035 [0.032, 0.038]) that
the sampler cuts to a drawn 0.008, far below what the cue allows (the label hazard at 30 frames since the last change
is 0.96). Check 12's null was the form of the input, not the absence of an incentive: G1 does reward timing once the
timing is readable, and it rewards it for every clock here, recovery included. What remains is how far and how fast
the learned probability approaches the cue's (energy 0.04 against 0.96, food 0.15 against 0.73, after 2,000
updates and still rising), and the few-step sampler, which keeps sharp probabilities (recovery 0.72 to 0.75) and cuts
graded ones (energy 0.040 to 0.008, asleep onset 0.093 to 0.041).

## Check 15: which single-cell changes G1 draws (`use_check.py catalogue`, `catalogue.json`; DEV fold 4, 128 events per class)

The first run's two trace arms were void (autocast kept the trace map's zero bf16 copy from the F6.1 arm for the whole
episode block; own and zeroed traces came out bitwise identical); fixed in `2e35418a` and rerun, with the F6.1 arm
reproducing run 1 bitwise. F6.1's first-step probability at the event's cell and its drawn rate (at the event / on
frames without it):

| Where the cue lies | Class (Craftax inventory order) | First step | Drawn | Quiet |
| :--- | :--- | ---: | ---: | ---: |
| Action, in the window | wood +1 / placing sapling / crafting (5 tools) / placing table or stone | 0.91 / 0.83 / 0.73-0.85 / 0.62-0.76 | 0.90 / 0.80 / 0.74-0.84 / 0.59-0.75 | <= 0.07 |
| Action, in the window | stone +1 / drink +1 | 0.69 / 0.59 | 0.61 / 0.53 | 0.02 / 0.03 |
| Action, rare | coal +1 / iron +1 | 0.25 / 0.15 | 0.20 / 0.12 | 0.02 |
| Action, chance or hidden | sapling +1 (grass, p = 0.1) / food +1 (cow's hidden health) | 0.03 / 0.12 | 0.04 / 0.06 | 0.00 / 0.02 |
| Partly hidden | damage | 0.64 | 0.64 | 0.02 |
| Memory (reward-priced) | recovery | 0.48 | 0.47 | 0.04 |
| Older history only | food / drink / awake energy drop; asleep rise | 0.07 / 0.06 / 0.06; 0.09 | 0.04 / 0.03 / 0.03; 0.06 | 0.01-0.05 |

G1 learns small single-cell changes, many of them: what it draws tracks how predictable the change is from what the
denoiser reads (the window, the action, the memory), not the size of the change. The clock drops are the only
deterministic changes whose cue lies only in older history, and they are the ones not drawn.

The trace fine-tune (fixed rerun) used its traces around the player's cell (cells 30-32 and 40 lead; map 0.0018 per
cell against HUD digits 0.0013), where tokens change all the time, and not on the clock events (own against zeroed
traces: energy drawn 0.030 / 0.029, food 0.048 / 0.047); check 13's predictor scores those events at 0.18-0.19 against
0.04 on quiet frames.

## Check 16b: the recency fine-tune on natural frames (`recency_catalogue.py`, `recency_catalogue.json`)

| Class | F6.1 drawn (quiet) | Recency 2,000 drawn [95%] (quiet) | First step | Cue's hazard (labels) |
| :--- | ---: | ---: | ---: | ---: |
| Awake energy drop | 0.032 (0.011) | 0.222 [0.193, 0.250] (0.001) | 0.170 | 0.96 at 30 frames |
| after_reset | 0.006 (0.008) | 0.211 (0.002) | 0.181 | |
| Drink drop | 0.034 (0.047) | 0.274 [0.227, 0.324] (0.020) | 0.206 | 0.83 at 20 |
| Food drop | 0.042 (0.025) | 0.114 [0.087, 0.144] (0.014) | 0.122 | 0.73 at 25 |
| Asleep rise | 0.056 (0.030) | 0.046 (0.009) | 0.109 | |
| Recovery | 0.472 (0.041) | 0.395 [0.329, 0.463] (0.026) | 0.363 | |

The recency acts on the HUD digits (0.10 per cell against 0.015 on the map; drink, food, energy lead). The action-cued
classes move within about 0.05 either way (stone pickaxe 0.84 to 0.80, placing sapling 0.80 to 0.91). On natural
frames the use is far larger than on the stand-still pairs (awake energy drawn 0.22 against 0.008 there; first step
0.17 against 0.04): the pairs understate it, for a reason not measured. Recovery fell on natural frames (0.47 to 0.40)
while rising on its pairs (0.64 to 0.75).

**Synthesis (checks 12-16b).** The traces carried the timing (check 13), but in a form whose gap was 2-3% of the input's
spread: the shared map learned to use them where changes are frequent and the loss is large, not for the clocks
(checks 14, 15). The same information as an exact per-cell recency is learned within 2,000 G1 updates and drawn on
natural frames at 4-8 times F6.1's rate with fewer false alarms (check 16, 16b). The denoiser does not lack an
incentive; it lacked a readable input. What remains: the drawn rates sit far below the cues' hazards and were still
rising; one seed; a fine-tune from F6.1; the recency is hand-made (label-free and per cell, but not learned).

## Check 17: the schedule shift does not explain the pairs (`schedule_pairs.py`, `schedule_pairs.json`)

Drawn due / delayed on the old pairs and on pairs whose due branch is unedited (its drop on the episode's schedule),
same frames since the last change (energy 30 / 22, food 25 / 17):

| Pairs | Due at episode step | F6.1 | Recency 2,000 (check 16) | 4,000 (check 18) | 6,000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Energy, old | 51 | 0.007 / 0.007 | 0.008 / 0.000 | 0.704 / 0.000 | 0.793 / 0.000 |
| Energy, on schedule | 62 | 0.017 / 0.016 | 0.028 / 0.000 | 0.958 / 0.000 | 0.982 / 0.000 |
| Food, old | 46 | 0.089 / 0.089 | 0.130 / 0.000 | 0.930 / 0.000 | 0.995 / 0.008 |
| Food, on schedule | 52 | 0.075 / 0.074 | 0.107 / 0.000 | 0.942 / 0.000 | 0.996 / 0.008 |

**Not a schedule conflict** (fixed bar: F6.1's on-schedule energy rate 0.017 < 0.05, due minus delayed 0.001 [0.000,
0.003]). F6.1 is untimed on both kinds of pair; its higher rate on natural early-episode drops (0.135 before a first
sleep, mostly the 9 -> 8 drop at step 31) is not reproduced on pairs and was not traced further. The recency fine-tune
draws the on-schedule energy pairs somewhat more (0.958 against 0.704 at 4,000). Memory's effect is 0.000 on every
bank and checkpoint: these clocks are drawn from the recency input, not from Mamba's memory. The old pairs' low reading
at 2,000 was the learning stage and run-to-run variation (check 18).

## Check 18: still learning, to most of the cue (`recency_long.py`, `recency_long.json`, `recency_long_catalogue.json`)

Check 16 rerun to 6,000 updates. Its 2,000 read is a second run (kernel nondeterminism only; update 0 reproduces
exactly) and differs from check 16's by up to 0.124: energy 0.132 against 0.008, food 0.086 against 0.130, asleep
onset 0.004 against 0.041. A single run's pair read at an intermediate stage is not a reliable level; the direction
(delayed at 0, positive recency effect on energy and food) held in both.

| Pairs (due / delayed) | F6.1 | 2,000 (run 2) | 4,000 | 6,000 |
| :--- | ---: | ---: | ---: | ---: |
| Awake energy | 0.007 / 0.007 | 0.132 / 0.000 | 0.704 / 0.000 | 0.793 / 0.000 |
| Food | 0.089 / 0.089 | 0.086 / 0.000 | 0.930 / 0.000 | 0.995 / 0.008 |
| Recovery | 0.643 / 0.000 | 0.705 / 0.000 | 0.722 / 0.000 | 0.775 / 0.000 |
| Asleep onset | 0.034 / 0.017 | 0.004 / 0.000 | 0.370 / 0.080 | 0.588 / 0.067 |

G1 0.00387 / 0.00361 / 0.00349 / 0.00345 / 0.00334 / 0.00324 at 1,000-6,000 (F6.1's monitor 0.00383).

Natural frames at 6,000 (check 15's rows and seeds), drawn at the event [95%] (quiet frames):

| Class | F6.1 | Recency 2,000 (check 16b) | Recency 6,000 | Same, recency zeroed |
| :--- | ---: | ---: | ---: | ---: |
| Awake energy drop | 0.032 (0.011) | 0.222 (0.001) | 0.797 [0.739, 0.852] (0.001) | 0.011 |
| after_reset | 0.006 | 0.211 | 0.913 [0.879, 0.942] (0.000) | |
| Drink drop | 0.034 (0.047) | 0.274 (0.020) | 0.560 [0.490, 0.627] (0.026) | 0.015 |
| Food drop | 0.042 (0.025) | 0.114 (0.014) | 0.601 [0.534, 0.665] (0.029) | 0.040 |
| Asleep rise | 0.056 (0.030) | 0.046 (0.009) | 0.403 [0.354, 0.451] (0.029) | 0.241 |
| Recovery | 0.472 | 0.395 | 0.478 | 0.412 |
| Damage | 0.641 | 0.624 | 0.672 | 0.656 |
| Drink +1 / stone +1 | 0.526 / 0.613 | 0.577 / 0.584 | 0.858 / 0.854 | 0.667 / 0.738 |
| Coal +1 / iron +1 | 0.199 / 0.117 | 0.194 / 0.101 | 0.668 / 0.457 | 0.418 / 0.322 |

**Still learning** (fixed bar: awake energy +0.575 over check 16b's 0.222; drink +0.29, food +0.49). With a readable
per-cell recency and G1 alone (denoiser only; encoder and belief frozen), the denoiser draws the awake clocks on time
at 0.56-0.80 on natural frames (0.91 after a reset) and the sleep rise at 0.40, with false alarms at or below 0.03,
and no class lost ground. Zeroing the recency puts the clocks back at F6.1's level, so they come from it; the
action-cued gains (coal 0.20 to 0.67, iron 0.12 to 0.46) are partly there with it zeroed (0.42 / 0.32), so part of them
is the extra G1-only denoiser training itself. Separating that needs a plain G1 fine-tune of the same length (not run;
zeroed recency is an off-distribution input, not that control).
