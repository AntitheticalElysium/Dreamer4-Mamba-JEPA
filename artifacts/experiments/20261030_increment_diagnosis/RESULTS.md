# Increment diagnosis: results (10-30)

## Part 1: substitution probe (`increment_probe.py`, `probe/readings.json`; ran before review, see PLAN status)
P1 world 1 checkpoint 10, dev test-F rows, eye elsewhere, the same draws in every variant. Controls: MOVE's shift
0.91 -> 0.04 under NOOP (the substitution reaches the model); identity substitution changes nothing (< 0.02).

| | drink | wood |
|---|---|---|
| appearance rows (0 -> 1): generated | 0.26 | 0.31 |
| . DO -> NOOP, belief recomputed | 0.00 | 0.00 |
| . a "2" shown instead of the empty counter (belief recomputed) | 0.00 (copies the 2: 0.995) | 0.01 (copies: 0.93) |
| increment rows (v -> v + 1): generated | 0.009 | 0.018 |
| . instead | copy 0.81, v - 1 0.11 | copy 0.86, v - 1 0.05 |
| . DO -> NOOP: change of the sigma-20 field | -0.03 | -0.06 |
| . the empty counter shown (belief recomputed) | appearance 0.17 | appearance 0.33 |
| . another digit shown | copies the shown one 0.98 | 0.85 |

The world learned "action + empty element -> fixed new content", not "action + shown digit -> next digit"; it reads
the shown value and uses the action and faced block, so the missing piece is the digit-to-digit mapping. Train labels:
at v >= 1 drink goes down (thirst) more often than up; at 0 only up. Why the mapping is not learned: not pursued
(user, 10-30: a focal error, not worth the time).

## Part 2: the learning signal by class (`signal_accounting.py`, `signal/*.json`)
Train-400, 1,200 windows, the 10-29 method (split halves, Adam drive). Replication check (checkpoint 10, random
fixations, against the 10-29 run on 1,760 windows): bulk classes agree (map_still loss share 0.288 vs 0.286, cosine
0.682 vs 0.678, drive -37.8 vs -38.3), rare classes within sampling noise (table drive +121 vs +128); the split
classes sum to the 10-29 totals.

**Allocation at the fine-tune's start (checkpoint 10, the same windows and noise; random vs guided fixations on every
attempt frame).** Gradient-norm share (split-half corrected; not additive) and loss share:

| class | grad share random -> guided | loss share random -> guided |
|---|---|---|
| glimpse on drink appearance | 0.02% -> 0.44% | 0.003% -> 0.027% |
| glimpse on drink increment | 0.08% -> 0.77% | 0.015% -> 0.155% |
| glimpse on wood appearance | 0.05% -> 1.62% | 0.003% -> 0.067% |
| glimpse on wood increment | 0.13% -> 1.18% | 0.021% -> 0.136% |
| glimpse on other consequences | 0.15% -> 0.98% | 0.029% -> 0.207% |
| periphery drink / wood / table (unchanged by design) | 0.70-1.34% either way | |
| bulk: map still / HUD static / glimpse rest | 48% / 42% / 53% | |

All of the added signal is in the glimpse output (~8x the consequences' glimpse loss); the periphery's is unchanged.
First-order drive (per 1,000 steps, relative, checkpoint 10's learning rate): guided placement flips the periphery
appearances from rising to falling (drink +34 -> -15, wood +22 -> -13); table stays rising (+121 -> +131).

**What it achieved: loss per element, ckpt 10 -> ft_1500, guided fixations, same windows and noise** (highest noise
bin, sigma > 2.67, where generation is decided, in brackets):

| class | all noise levels | sigma > 2.67 |
|---|---|---|
| glimpse drink appearance | -16% | -25% |
| glimpse drink increment | -14% | -15% |
| glimpse wood appearance | -19% | -21% |
| glimpse wood increment | -19% | -20% |
| glimpse other consequences | -20% | -16% |
| glimpse rest (bulk) | -12% | -9% |
| periphery drink appearance | +3% | **+25%** |
| periphery drink increment | -4% | 0% |
| periphery wood appearance | -4% | -6% |
| periphery wood increment | -12% | -11% |
| periphery table | +9% | **+22%** |
| periphery map still (bulk) | -5% | -5% |

## Reading
- The extra share did what an extra share does where it landed: the glimpse's consequence losses fell 14-20%, at
  every noise level, more than the glimpse bulk (-9% to -14%). For the appearance this became generation (glimpse
  drink 0 -> 1: 0.088 -> 0.186); for increments it did not: their glimpse error at high noise is already near the
  bulk's (0.019 vs 0.015 per element: the digit's change is small, so copying costs almost nothing and the decline
  is general reconstruction).
- The periphery received no extra share, and at the highest noise level its appearance and table losses ROSE (+25%,
  +22%) while its losses below sigma 1 fell (drink appearance -23% to -43%, the others 0% to -26%): consistent with its unchanged or worse generation; the
  first-order flip at the start did not hold over the path (learning-rate transient, changing state).
- Limits: one run, no arm R, so the fine-tune's general effect (restored learning rate, clip-scope transient, see
  the fovea RESULTS) is not separated from placement; the glimpse bulk's own -12% is the visible part of it.
