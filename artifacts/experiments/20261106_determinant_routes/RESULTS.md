# Determinant routes: what each input route does to the blind consequences (frozen P1-F3)

2026-11-06 16:44-17:20 (plan + code amended after Codex's review, 5c3fcd0e; units routes-build, routes-run,
MAMBA_DETERMINISTIC=1). 842 simulator-edited pairs on dev backgrounds, 0 excluded: every recorded frame t - 1 replayed
bitwise, every natural successor equal to the stored frame t, every edit changing the simulator's outcome as the rule
says. Shams within the declared tolerances (natural tokens vs S1 encode max 0.024 < 0.04; same-input belief 0.012 <
0.02; cache 0.009 < 0.011; 34 repeated generations bitwise equal); all 8 readers valid; recorded successors read
correctly in 100% of pairs. Every random-label control within 0.46-0.59 (no stage gated out). `readings/routes.json`,
`run.json`, `bank.json`; raw tensors, generated percepts and reader parameters in `run.pt` (local).

Cells: (denoiser side, memory) from (d0, d1); the denoiser side is the cached frame t - 1 (sleep, energy) or the
denoiser's own action input (counts). Event rates over K = 8 futures per cell, episode-bootstrap 95%.

## 1. Sleep (asleep d0 vs woken d1, 240 pairs / 188 episodes): both routes act, the memory dominates, consistent asleep inputs still wake

| cell | awake | first-call position (0 = asleep successor, 1 = awake) |
|---|---|---|
| 00 cache asleep, memory asleep | 0.466 [0.438, 0.493] (day 0.305, night 0.627) | 0.47 |
| 10 cache awake, memory asleep | 0.790 [0.770, 0.808] | 0.73 |
| 01 cache asleep, memory awake | 1.000 | 0.96 |
| 11 both awake | 1.000 | 0.98 |

- Memory sufficiency 0.534 [0.507, 0.562] CARRIES: an awake belief draws an awake frame in every future even with an
  asleep frame in the cache. Cache sufficiency 0.323 [0.296, 0.351] UNRESOLVED (day 0.475 CARRIES, night 0.172 NO
  EFFECT). Necessity: cache 0.000 (the awake memory saturates the outcome); memory 0.210 [0.192, 0.230] NO EFFECT by the
  declared threshold, though an asleep memory keeps 21% asleep against an awake cache. Interaction -0.32 PRESENT
  (saturation at 1.0).
- The sleep state is recovered by the probe at every stage, including the generated frame's map / lighting tokens
  after every block (accuracy 0.97-1.00).
- So the asleep determinant reaches both inputs and moves the output through both routes, and with BOTH inputs asleep
  47% of futures still wake (exp 2: ~50%). Which part of the asleep-conditioned generation leaks is UNRESOLVED.

## 2. Energy (asleep, 8 d0 vs 9 d1, 200 pairs / 200 episodes): no route changes the outcome

All cells 0.340-0.344 awake (stale-8 backgrounds); every effect <= 0.004 with upper bounds <= 0.007: diagonal,
sufficiency and necessity of both routes NO EFFECT, interaction not detected; the first-call position 0.298-0.301 in
all four cells.

| stage | probe accuracy (one state) | paired ratio |
|---|---|---|
| energy token, frame t - 1 | 1.00 RECOVERED | 35.5 |
| percept embedding | 0.95 RECOVERED | 0.064 |
| belief after the percept / after the action / memory | 0.73 PARTIAL / 0.58 NOT RECOVERED / 0.61 PARTIAL | 0.05 / 0.03 / 0.03 |
| cache, energy token's keys / values (context 0.1 / stress 0.67) | 1.00 RECOVERED / 0.74 PARTIAL | 6.1 / 1.1 |
| cache, map + lighting tokens (0.1 / 0.67) | 0.68 PARTIAL / 0.55 NOT RECOVERED | 0.03 / 0.01 |
| generated frame, map + lighting tokens, blocks 1-8 | 0.50-0.53 NOT RECOVERED | <= 0.016 |

Energy 9 vs 8 is recovered at its own cached token at the inference context level, only partially after Mamba by
this probe, and not at the generated frame's map / lighting tokens at any block; changing it through either route
changes nothing. These are probe readings, not a located cause (the denoiser could read the cached energy token
without that showing in a linear probe of the map tokens; it does not change its output).

## 3. Counts (DO d0 vs NOOP d1 on identical frames, 67 per starting count): the increment is gated by the memory's action for coal and wood alike

| cell | coal increment | wood increment |
|---|---|---|
| 00 denoiser DO, memory DO | 0.134 [0.109, 0.159] | 0.863 [0.828, 0.893] |
| 10 denoiser NOOP, memory DO | 0.056 [0.039, 0.074] | 0.522 [0.481, 0.559] |
| 01 denoiser DO, memory NOOP | 0.000 | 0.003 |
| 11 both NOOP | 0.000 | 0.001 |

- Same structure in both items: with the memory's NOOP no increment whatever the denoiser's own action input; with the
  memory's DO, the denoiser's DO adds to it (interaction PRESENT in both). Wood: memory sufficiency 0.859 and
  necessity 0.522 CARRIES, denoiser-side sufficiency 0.340 UNRESOLVED (CARRIES at count 1), its necessity 0.002.
  Coal: every effect below one third of the truth (all NO EFFECT), the whole 0.134 entering with the memory.
- Within starting count (coal / wood, cell 00): n = 1: 0.157 / 0.871; n = 2: 0.118 / 0.868; n = 3: 0.127 / 0.849.
  The gap is not a count imbalance.
- The action is recovered in the belief, the memory and at each item's slot in every block (accuracy 0.99-1.00 for
  both). This is action identity, which conditions every block (Codex): it does not show that a collection event was
  computed or bound to the slot. Why coal's memory-gated increment is ~6x weaker than wood's is UNRESOLVED here.

## 4. Hidden determinants (descriptive; values already seen in the smokes, PLAN)

| target | after-action belief | current percept + action (control) | observable heuristic (reference, not a ceiling) |
|---|---|---|---|
| READY, adjacent zombie (4,000 rows / 849 episodes) | 0.79 [0.77, 0.82] (lethal 0.74, safe 0.84) | 0.67 [0.65, 0.70] | 0.92 [0.91, 0.94] |
| KILL, mob hit (3,600 / 732) | 0.89 [0.86, 0.90] (cow 0.92, zombie 0.86) | 0.84 [0.80, 0.86] | 0.99 [0.99, 0.99] |

The belief reads readiness and kills better than the current frame alone (history adds 0.12 / 0.05 AP), and less well
than the visible-history heuristics. Within truth strata the world's damage does not follow the belief reader's
decision (ready: 0.126 when the reader says ready, 0.144 when not; cooling 0.084 / 0.069); mob departure follows it
a little (kill: 0.68 / 0.58; non-kill 0.58 / 0.53). No inference-versus-use verdict (Codex).

## 5. What this establishes, and what stays open

Established, under these swaps on these backgrounds (one world seed, 16 steps):
1. **Sleep**: both routes move the sleep outcome; an awake belief alone fixes it awake; with both inputs asleep the
   world still wakes 47% (31% by day, 63% at night). The determinant is recovered everywhere. The failure to hold sleep
   is therefore not a missing input: it happens with the asleep state present at both inputs and in the generated
   tokens' states.
2. **Energy**: neither route carries 8 -> 9 into the outcome; the change is recovered at the energy token and its own
   cached keys / values, not at the generated frame's map / lighting tokens.
3. **Counts**: the increment enters through the memory's action for both coal and wood; the route structure of the
   failing item matches the working one, with ~6x less effect at every matched starting count.
4. **Hidden state**: the belief carries some history about readiness and kills beyond the current frame.

Unresolved: which computation makes asleep-conditioned generation leak; whether the energy digit reaches any token
that must change; why coal's memory-gated effect is weaker than wood's; inference versus use for the hidden
counters. No unique faulty layer is identified, and no repair is selected. Crossed cells are interventions, not
in-distribution states; 0.67 is one stress level, not the training noise distribution; wood draws on 51 episodes.
