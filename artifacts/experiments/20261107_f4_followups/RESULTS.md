# P1-F4 follow-ups: results against PLAN.md's declared readings

## 1. Danger at K = 8: the trained world, not the sampler (`steps8.py`, `readings/k8/`)

Same checkpoint, populations, seeds, reader protocol and interface classes as F4's readout; only K = 4 -> 8.

| reading | K = 4 | K = 8 | truth / recorded | declared band |
|---|---|---|---|---|
| risky-class death (interface) | 0.036 | 0.043 | 0.128 | world: within 0.02 of K = 4 |
| ready zombie damages (curves) | 0.082 | 0.086 | 0.321 | world: within 0.04 |
| lethal hits (curves) | 0.040 | 0.050 | 0.335 | - |
| sleeping agent hit by a ready zombie | 0.099 | 0.133 | 1.00 | - |
| adjacent-class damage (interface) | 0.054 | 0.056 | 0.159 | - |

Verdict (declared rule): **the trained world** draws danger rarely at any step count; doubling the steps moves it by
<= 0.01-0.03. Unlike F3 (3 -> 16 steps: risky death 0.036 -> 0.088), F4's sampler is not the bottleneck. The gains are
step-invariant: false wakes 0.008-0.024 per stratum, woken at energy 9 1.00, coal / iron 0 -> 1 0.98 / 0.92,
increments unchanged, every interface unlock within 0.015 of K = 4 (sapling 0.06 -> 0.07). Woken beside a ready
zombie 0.54 (damaged 0.13), beside a cooling one 0.18 (truth 0.007): the wake / hit decoupling also persists.

## 2. Memory versus cache routes on F4 (`routes_f4.py`, `readings/routes_f4.json`)

842 pairs (exp 3's bank), 0 excluded; shams within exp 3's tolerances (tokens 0.023 < 0.04, belief 0.016 < 0.02, cache
0.006 < 0.011, 34 repeated generations bitwise); recorded successors read correctly in 100% of pairs. Cells are
(denoiser side, memory) from (d0, d1), event rates over K = 8 futures, episode-bootstrap 95%.

| family | 00 | 10 (denoiser side d1) | 01 (memory d1) | 11 | F3 cells 00 / 10 / 01 / 11 | declared reading |
|---|---|---|---|---|---|---|
| sleep: awake (d0 asleep, d1 woken) | 0.000 | 0.046 | 0.998 | 1.000 | 0.47 / 0.79 / 1.00 / 1.00 | Mamba CARRIES |
| energy: awake (d0 8, d1 9) | 0.000 | 0.478 | 0.000 | 0.477 | 0.34 / 0.34 / 0.34 / 0.34 | Mamba BYPASSED |
| coal: +1 (d0 DO, d1 NOOP) | 0.899 | 0.234 | 0.400 | 0.002 | 0.13 / 0.06 / 0.00 / 0.00 | shared |
| wood: +1 (d0 DO, d1 NOOP) | 0.994 | 0.153 | 0.448 | 0.000 | 0.86 / 0.52 / 0.00 / 0.00 | shared |

- **Sleep: the memory alone decides it.** Memory sufficiency 0.998 and necessity 0.954 CARRIES (F3: necessity NO
  EFFECT, 0.21); the cached frame's sufficiency 0.046 NO EFFECT (F3 0.32 UNRESOLVED; night 0.09, day 0.002). With both
  inputs asleep, 0.000 wake (F3 0.47): predicted, and the leak exp 3 left unresolved is gone. The dense route offers the
  cached awake frame one step away and the world does not use it.
- **Energy: only the cache carries it.** The cached energy-9 frame wakes 0.478 whatever the memory says; the memory's 9
  does nothing (sufficiency and necessity 0.000 / -0.001 NO EFFECT). Predicted: the energy token of frame t - 1 is one
  dense step away, and the belief recovered energy only partially in F3. Not resolved: with both inputs at 9 the
  edited pairs wake 0.48, while natural energy-9 rows wake 1.00 (curves): something in natural histories beyond the
  edited digit contributes.
- **Counts: the memory gate is gone; both action inputs carry the increment.** Both DO 0.90 / 0.99 (coal / wood); the
  denoiser's own action alone 0.40 / 0.45; the memory's action alone 0.23 / 0.15; neither 0.00. In F3 a NOOP memory
  removed every increment (0.000 / 0.003). My prediction "counts keep the memory gate" is REFUTED; "coal now near wood"
  holds (0.90 vs 0.99; F3 0.13 vs 0.86). The denoiser-side route here is the action conditioning (adaLN), not the dense
  attention: the cached frames are identical in both members.
- So the dense path does not supersede Mamba wholesale: it took over the one determinant Mamba never carried (energy),
  Mamba now carries sleep alone, and counts are shared. Caveats as declared: crossed cells are interventions; one
  world seed; F4's context is corrupted to signal 7 / 8.

## 3. Movement in F4's own token space: holds (`routes_f4.py movement`, `readings/movement_f4.json`)

T1's 64 pairs (64 episodes), K = 16 futures per member through F4's sampler: the move member moved in 1.00 of futures
and the NOOP member in 0.00; all 64 pairs both ways (F3: 1.00 / 0.00). Mean projection 0.10 for the move member (0 =
the move successor's tokens) and 0.98 for NOOP (1 = the NOOP successor's). Declared reading (>= 0.95 / <= 0.05): holds.

## 4. FlexAttention: correct, deterministic in the model, ~5% faster; the split was never the cost (`flex_bench.py`, `readings/flex/`)

Two failed attempts before the reading (both OOM in `correct`, committed fixes): the loaded model leaves activation
checkpointing off (train.p1 turns it on); with it on, the check ran at the training microbatch (8 rows x 64 frames).

- **Correctness** (8 rows, 2 spliced across an episode boundary, 2 bootstrap rows, bootstrap on): flex vs SDPA in fp32
  7.0e-7 (loss per frame) / 1.8e-6 (parameter gradients): the mask is exact on the GPU. In bf16 flex vs SDPA 0.0029 /
  0.0083, below SDPA's own bf16-vs-fp32 discrepancy 0.0057 / 0.064. Declared reading: holds.
- **Determinism** (two fresh processes per path, 4 updates, bootstrap from update 2): flex model 228 / 228 tensors
  bitwise (SDPA 224 / 228, 4.7e-18), optimizer 1.1e-13 (SDPA 1.1e-19), logged terms bitwise in both. Against the
  declared rule: the model meets it (better than SDPA), the optimizer does not (larger than SDPA's pair, though at
  F3's own pre-existing cross-process level 2.3e-13 and far inside checks.py's 1e-11).
- **Cost** (sec / update, final half; peak GiB):

| configuration | 64 steps, bootstrap off | 64 steps, on | 256 steps, on | peak |
|---|---|---|---|---|
| F3 at its split (1 / 4) | 1.68 | - | 6.08 | 4.24 |
| F3 at F4's split (2 / 8) | 1.69 | - | 6.21 | 2.29 |
| F4 SDPA at 2 / 8 (the run) | 1.92 | 2.15 | 7.81 | 2.54 |
| F4 flex at 2 / 8 | 1.77 | 2.00 | 7.74 | 2.42-2.59 |
| F4 flex at F3's split | 1.74 | 1.98 | 8.38 | 4.50-4.56 |

  The 256-step flex rows are compile-contaminated (per update 20.0 / 7.56 / 7.91 and 19.3 / 9.15 / 7.61: each new
  bootstrap sub-batch shape compiles once), so their steady state is ~7.6, not the table's mean. Projected training
  (checks.py's formula): F3 11.1 h; F4 SDPA 13.9 h; flex 13.3 h at 2 / 8 (13.2 h at steady state either split).
- What this says about the 30%: halving the microbatch costs 0.4% (64 steps) / 2% (256) on F3; SDPA's window
  implementation cost ~0.15 s / update at 64 steps (8%), which flex removes; what remains of F4's +2.8 h is the
  bootstrap (~1.1 h, intrinsic) and the dense layers' own compute (~1 h). Flex fits F3's split (4.56 <= 5.0 GiB) but
  buys no speed there. Declared reading: correctness and memory hold; determinism holds for the model, not for the
  optimizer by the strict rule; the gain is ~0.6-0.7 h per run.

## 5. Danger M1: the health token under-responds to every cause (`danger_m.py m1`, `readings/danger/m1.json`)

Sham R0: exp 2's zombie population regenerated under the canonical code equals F4's saved futures bitwise. Drawn vs
true P(health change) per future, ratio with episode bootstrap 95%:

| stratum (t - 1) | rows / episodes | drop drawn / true (ratio) | rise drawn / true (ratio) |
|---|---|---|---|
| arrow adjacent, asleep | 443 / 201 | 0.005 / 0.318 (0.02 [0.01, 0.03]) | 0.027 / 0.027 |
| arrow adjacent, awake | 1,000 / 337 | 0.009 / 0.069 (0.13 [0.08, 0.20]) | 0.012 / 0.014 |
| a need at 0, awake | 1,000 / 248 | 0.010 / 0.038 (0.27 [0.16, 0.45]) | 0.011 / 0 (false) |
| a need at 0, asleep | 1,000 / 131 | 0.009 / 0.008 (1.09 [0.56, 2.60]) | 0.023 / 0 (false) |
| needs met, health < 9, awake | 1,000 / 389 | 0.006 / 0 (false) | 0.013 / 0.041 (0.31 [0.21, 0.46]) |
| needs met, health < 9, asleep | 1,000 / 288 | 0.007 / 0 (false) | 0.036 / 0.095 (0.38 [0.30, 0.48]) |

Declared reading: need damage and recovery are both <= 0.5 (awake need damage 0.27, recoveries 0.31 / 0.38; asleep
need damage, at a true rate of 0.008, is the exception) -> **the health token is weak generally**, arrows worst
(0.02 / 0.13, below zombies' 0.11-0.26). Drawn changes also run the wrong way at 0.6-2.3% (rises where a need is 0,
drops where needs are met and nothing hostile is near): the drawn changes are weakly tied to their causes.

## 6. Danger M2: the hit's wake and drop share one route, the cache (`danger_m.py m2build / m2run`, `readings/danger/m2.json`)

232 pairs (232 episodes; 8 excluded: the edited member still lost health or woke); recorded successors read right in
100%; shams: tokens 0.021, belief 0.009, 10 generation repeats bitwise. Cells = (cache, memory) from (zombie, removed):

| event | both zombie | memory only | cache only | neither | cache-only share | memory-only share |
|---|---|---|---|---|---|---|
| woken (truth 1.00 vs 0) | 0.688 | 0.014 | 0.685 | 0.012 | 0.996 [0.989, 1.003] | 0.002 [0.000, 0.006] |
| health drop (truth 1.00 vs 0) | 0.072 | 0.006 | 0.054 | 0.005 | 0.736 [0.661, 0.810] | 0.016 [0.000, 0.041] |

Fresh arrivals: woken 0.73, drop 0.07; already beside: woken 0.21 (0.10 with the zombie removed at t - 1 only: it is
still in the earlier cached frames), drop 0.06. Declared reading: **H2 (different routes) refuted** - the wake is carried
by the cache alone (not the memory, unlike part 2's sleep edit), and so is what little drop the world draws. The same
cached zombie that wakes the agent in 69% of futures moves the health digit in 7%: the failure sits at the health
token's response, not at perceiving or routing the cause. (Against the true contrast every drop class is NO EFFECT:
the world's own effect is 0.07.)
