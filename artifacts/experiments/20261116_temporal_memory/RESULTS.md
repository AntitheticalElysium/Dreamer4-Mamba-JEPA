# Temporal memory research pass: results

## Measurement 1: how readable is each per-cell time code? (`readability.py`, `readability.json`)

F6.1 tokens, DEV fold 4, the clock events of the `after_reset` cohort (token_trace's rows); readers fitted per digit
cell on 400 fold 0-3 episodes (120 frames each, 48,000 per cell; target: the cell's token changes at the next frame).
AP, linear reader (logistic, L2 1e-4) / nonlinear reader (LightGBM):

| Code | Food drop | Drink drop | Awake energy drop | Asleep energy rise |
| :--- | ---: | ---: | ---: | ---: |
| Base rate | 0.049 | 0.067 | 0.037 | 0.243 |
| Exact recency (check 16; complete serial compound) | 0.69 / 0.69 | 0.78 / 0.78 | 0.94 / 0.95 | 0.71 / 0.73 |
| Traces, tau 8 / 32 (check 12) | 0.14 / 0.51 | 0.16 / 0.53 | 0.07 / 0.75 | 0.29 / 0.57 |
| CogRNN f~, paper setting (8 cells, k 8) | 0.05 / 0.25 | 0.07 / 0.29 | 0.04 / 0.20 | 0.25 / 0.24 |
| CogRNN F only, paper setting | 0.27 / 0.88 | 0.24 / 0.92 | 0.14 / 0.94 | 0.31 / 0.77 |
| CogRNN f~, repository default (80 cells), k 4 | 0.35 / 0.88 | 0.41 / 0.93 | 0.23 / 0.94 | 0.36 / 0.79 |
| CogRNN f~, 80 cells, k 8 | 0.37 / 0.88 | 0.45 / 0.94 | 0.27 / 0.93 | 0.38 / 0.79 |
| CogRNN f~, 80 cells, k 16 | 0.39 / 0.88 | 0.48 / 0.94 | 0.31 / 0.93 | 0.38 / 0.79 |
| CogRNN F only, 80 cells, k 8 | 0.28 / 0.88 | 0.28 / 0.93 | 0.17 / 0.94 | 0.30 / 0.78 |

Rows: 23,838 / 19,526 / 26,585 / 2,510 with 1,176 / 1,305 / 981 / 609 events. Our scan of the Laplace layer equals
CogRNN's own (F to 1e-13; f~ to 1e-8 of its scale, the residue being float64 cancellation through Post weights up to
1.9e6 at k = 16).

**Instrument check passes** (fixed bar: exact recency's linear AP on the awake energy drop at least twice the traces'):
0.944 against 0.069, the order training found (drawn 0.80 against 0.03).

**Readings.**
- Information: every Laplace code with 80 cells holds the clocks (nonlinear AP 0.93-0.94 on awake energy) and more
  than the exact recency on food, drink and the asleep rise (0.88 / 0.93-0.94 / 0.79 against 0.69 / 0.78 / 0.73). The
  recency keeps only the last change within 63 frames; the Laplace record keeps several events over about 1,000
  frames. Craftax's counters are not a function of the last change alone: hunger and thirst grow at half rate while
  asleep, and drinking or eating at 9 resets the counter without changing the digit (`craftax_classic/game_logic.py`
  lines 1256-1273, 310, 330).
- Form: a linear reader recovers 28-33% of the recency's AP on the awake energy drop from the time cells
  (0.27 at k 8, 0.31 at k 16, against 0.94), 54-56% on food, 57-62% on drink. The inverse helps over F alone (energy
  0.17 to 0.27 at k 8), as in CogRNN's own ablation; larger k helps a little. The traces are the least readable.
- The paper's stated setting (8 cells over 1-1,000 frames) gives a broken inverse in its own code (impulse responses
  peak at 255-1,837 frames for cells meant to peak at 1-7; values up to 1.9e9): its f~ rows sit at base rate, while
  its F rows hold the information like the others.
- The idealised separation reading (distance between the codes after one event at the pair lags, as a share of the
  code's spread; `readability.json`) does not order the codes as the readers and training do (traces 0.19, time cells
  0.08, exact recency 1.45 on energy 30 vs 22): a norm ignores direction in a many-dimensional code. Not used.

Instrument follow-up after the results (convergence, energy clock only; same rows): the linear readers converged
(gradient norm below 1e-7 at 500 and 5,000 L-BFGS iterations, identical APs). With L2 weakened to 1e-6 the time cells
read better (k 8: 0.27 to 0.38; k 16: 0.31 to 0.45; F only 0.17 to 0.19) while the recency and traces do not move
(0.944, 0.069): a linear reader sharpens Weber-blurred cells only with large, finely balanced weights.

**What this says for option (b).** CogRNN's form is the brain-grounded member of the family, and on our data it holds
more information than the hand-made recency. But under the linear input map the denoiser has today, its time cells
around 20-30 frames are about 24 frames wide (k 8) and give up most of the exact clocks' readability. A fine-tune
replacing the recency with them would mostly measure that Weber cost, unless the denoiser's later layers do the
sharpening. Not run; for the discussion.

## Measurement 2: the SSM's eigenvalue structure decides time readability (`ssm_bases.py`, `ssm_bases.json`)

Measurement 1's rows, readers and references; fixed bases from their papers on each digit cell's event train. AP,
linear / nonlinear:

| Code | Dimensions | Food drop | Drink drop | Awake energy drop | Asleep energy rise |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Exact recency (reference) | 64 | 0.69 / 0.69 | 0.78 / 0.78 | 0.94 / 0.95 | 0.71 / 0.73 |
| HiPPO-LegT (LMU), theta 64 | 64 | 0.77 / 0.86 | 0.83 / 0.93 | 0.91 / 0.93 | 0.72 / 0.75 |
| S4D-Lin (complex: damped Fourier) | 64 (32 complex) | 0.76 / 0.85 | 0.78 / 0.87 | 0.90 / 0.94 | 0.63 / 0.73 |
| S4D-Real (real decays) | 64 | 0.30 / 0.89 | 0.28 / 0.94 | 0.15 / 0.94 | 0.30 / 0.76 |
| HiPPO-LegT, theta 64 | 16 | 0.52 / 0.87 | 0.54 / 0.89 | 0.73 / 0.93 | 0.44 / 0.74 |
| Laplace F (measurement 1, real) | 80 | 0.28 / 0.88 | 0.28 / 0.93 | 0.17 / 0.94 | 0.30 / 0.78 |

At equal size, step and discretisation, S4D-Lin and S4D-Real differ only in their eigenvalues: rotating modes make the
awake energy clock linearly readable at 0.90, real decays at 0.15, with the same information in both (nonlinear
0.94). The Legendre window memory reaches the hand-made recency's level (0.91 against 0.94) and exceeds it on food and
drink, where it keeps more than the last change. Real-decay codes (S4D-Real, Laplace F, CogRNN's inverse) are
readable only through a nonlinear reader. Mamba-1 is a diagonal of real decays and Mamba-2 a single real decay per
head (Mamba-3's own account): the family whose state needs a nonlinear decoder for elapsed time.

## Measurement 3: learned memories form the clocks only when the objective asks for them (`architecture.py`, `architecture_seed1.json`)

Seed 1 (the plan's amendment), 3,000 updates, 12.3M frames, every memory from scratch on the same batches. AP at the
clock events of DEV fold 4 from each model's own head (base rates 0.049 / 0.067 / 0.037 / 0.243), and the generic
next-change log-loss over all cells:

| Arm | Objective | Food | Drink | Awake energy | Asleep rise | All-cell log-loss |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| G2: belief form, global Mamba-2 | all cells | 0.049 | 0.066 | 0.038 | 0.316 | 0.202 |
| G2K: G2 read through 4 memory tokens | all cells | 0.048 | 0.064 | 0.039 | 0.302 | 0.199 |
| G3: global Mamba-3 | all cells | 0.048 | 0.066 | 0.039 | 0.338 | 0.205 |
| P2: per-cell Mamba-2 | all cells | 0.051 | 0.068 | 0.034 | 0.237 | 0.256 |
| P3: per-cell Mamba-3 | all cells | 0.050 | 0.068 | 0.048 | 0.253 | 0.255 |
| G2_digits | digit cells only | 0.063 | 0.085 | 0.451 | 0.972 | (not trained) |
| P2_digits | digit cells only | 0.181 | 0.187 | 0.196 | 0.270 | (not trained) |

Awake energy over training (1,000 / 2,000 / 3,000): G2 0.034 / 0.037 / 0.038 (flat); G2_digits 0.123 / 0.347 / 0.451;
P2_digits 0.036 / 0.061 / 0.196.

Readings against the bars fixed before results:
- Compression (P2 above G2 by > 0.2): not supported (0.034 against 0.038).
- Interface (G2K below G2 by > 0.2): not supported, and not informative: neither formed the clocks, so the memory
  tokens had nothing to lose here.
- Rotations (Mamba-3 above Mamba-2 by > 0.2): not supported, global or per cell.
- Incentive (digits-only above all-cell by > 0.2): supported for the global memory (+0.41 energy, +0.66 asleep rise);
  for the per-cell memory +0.16 energy, +0.13 / +0.12 food / drink: within 0.1 of the bar, unresolved.

Under a generic objective over all cells, no memory structure tested here forms the awake clocks within 12.3M frames,
and nothing trends (G2 flat from 1,000 to 3,000 updates). Asked through the digit cells alone, the global belief-form
memory learns them (energy 0.45 and rising, asleep rise 0.97); the per-cell one learns food and drink faster (0.18 /
0.19 against 0.06 / 0.09) and energy slower. Per-cell arms have 0.15M parameters against 2.4M (declared), and a worse
generic loss (0.256 against 0.202: a cell alone does not see its neighbours, the scroll). Together with checks 16-18
(a fixed readable record lets G1 alone draw the clocks), the limit is learning the memory under a diluted objective,
not reading it: when the record exists, the generic objective suffices to read it; when the memory must be learned,
the generic objective does not shape it, whatever its structure here.

## Measurement 4: the readable record needs no hand-made event (`ssm_tokens.py`, `ssm_tokens.json`)

Same rows and readers; fixed bases fed, per digit cell, the graded squared change (no threshold) or the raw
16-channel token (no designed feature). AP linear / nonlinear (exact recency: 0.69 / 0.78 / 0.94 / 0.71 linear):

| Input, basis | Food | Drink | Awake energy | Asleep rise |
| :--- | ---: | ---: | ---: | ---: |
| Graded change, S4D-Lin | 0.50 / 0.70 | 0.58 / 0.80 | 0.52 / 0.90 | 0.57 / 0.76 |
| Graded change, LegT 64 | 0.62 / 0.86 | 0.67 / 0.87 | 0.63 / 0.97 | 0.73 / 0.84 |
| Raw token, S4D-Lin (1,024 features) | 0.34 / 0.81 | 0.36 / 0.84 | 0.80 / 0.96 | 0.61 / 0.80 |
| Raw token, LegT 64 (1,024 features) | 0.49 / 0.91 | 0.51 / 0.87 | 0.84 / 0.98 | 0.75 / 0.90 |

Bar (awake energy linear >= 0.71): passed by both raw-token records, not by the graded change. The raw token carries
the digit's past values, not only when it changed. Food and drink are less linearly readable from the raw-token
record than from the recency (0.34-0.51 against 0.69-0.78); the information is there (nonlinear 0.81-0.91). Test 2
uses the raw token through S4D-Lin.
