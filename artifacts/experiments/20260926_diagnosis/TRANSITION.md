# Why every world's transition loses information, and why recursion compounds it

2026-09-27. Measurements only, nothing new trained except two controlled 4,000-update parameterization arms
(D16). Worlds: canonical H2 (Mamba on raw z, joint 10k + bridge 2k), and the readout-ladder worlds Z (Mamba, z),
sZ (block-causal transformer, layer-normed z), U (Mamba, PCA-192 of the 4x4 patch grid), W (Mamba, U / std),
T (block-causal transformer, 81 per-tile tokens). Data: 1,002 roots from 143 fresh seeds (900,000+) walked by
the corpus's BC policy, each with 4 context frames, all 17 actions x 4 keys one step ahead, and the policy's own
16-step future under 5 keys (D9); plus 8,000 corpus transitions (TRAIN, DEV/FINAL) and 256 x 96-frame corpus
windows. Errors are "fraction of the state's natural variance" unless marked "vs copy" (ratio to copying the
previous frame; < 1 beats copying).

## Part 1 — One step

### 1a. The failure is not a generalization gap: it is present on each world's own training data

One-step squared error vs copying the root, uniform over transitions (`insample.py`):

| world | TRAIN all | TRAIN moved | TRAIN blocked | TRAIN other | sleep onset | held-out all |
|---|---|---|---|---|---|---|
| H2 (canonical) | **1.34** | 1.73 | 7.4 | 2.06 | 0.11 | 1.30 |
| Z | 0.61 | 1.03 | 2.3 | 0.79 | 0.04 | 0.63 |
| sZ | **1.17** | 1.31 | 4.8 | 1.48 | 0.52 | 1.12 |
| U | 0.57 | **0.41** | **19.0** | 1.18 | 0.14 | 0.55 |
| W | 0.97 | 0.68 | 9.7 | 2.02 | 0.58 | 0.97 |
| T | 0.89 | 0.95 | 2.5 | 0.98 | 0.24 | 0.89 |

Every world gets the sleep onset (a global desaturation of the map) and fails the local transitions. Held-out
equals TRAIN within 0.05 everywhere.

### 1b. What each world gets wrong: change of the right size in the wrong direction

Per transition class, one step, all actions (`onestep.py`, lava entries counted as moves after the other agent's
correction). "Captured" = 1 - error against the 4-key mean / predictable change; negative = worse than no change.

| world | moved | blocked | interact | sleep | idle (68% of transitions) |
|---|---|---|---|---|---|
| H2 | -0.46 | -0.99 | -0.10 | 0.87 | -2.46 |
| Z | -0.04 | 0.29 | 0.18 | 0.95 | -0.40 |
| sZ | -0.20 | -0.62 | -0.08 | 0.59 | -1.52 |
| U | **0.63** | -0.87 | -0.94 | 0.83 | -1.02 |
| W | 0.34 | -2.45 | -0.79 | 0.43 | -2.79 |
| T | 0.08 | 0.33 | 0.01 | 0.87 | 0.34 |

Outside SLEEP, the imagined state-dependent change has 0.8-1.5x the true energy (T: 0.19x) but only 15-63% of it
lies along the true change (regression slope, `decompose.py`). The imagined change is nearly the same size
whether a move succeeds or is blocked: imagined moved/blocked energy 0.96-1.38x, true 17-33x.

### 1c. Why — a different binding constraint per state, each measured

**Is the change determined by the frame at all?** Yes. A rule that scrolls the root frame one tile for a
successful move, knowing nothing else, captures 67-85% of the move's error in every state space; only 11-23% of
it is the newly revealed strip (`oracle.py`). Keys agree on 100% of next-step tiles and HUD (`compound.py`).

**Is it recoverable from each world's own state?** Closed-form fits, test seeds (`learnable.py`), captured on moves:

| state | ridge per action on the state | local-linear per tile (T) | trained world | move passability AUC from the state |
|---|---|---|---|---|
| z | 0.08 | | -0.46 to -0.04 | 0.58 |
| u | 0.63 | | 0.63 | 0.69 |
| w | 0.31 | | 0.34 | 0.69 |
| tokens (T) | 0.58 (PCA-1024) | **0.72** | **0.08** | 0.97 |

- **z: the information is not in the state.** A linear map of z captures 8% of a move and 0% of a blocked move;
  z cannot say where anything is (twin states: cosine 1.000 for an adjacent vs a far zombie, D4). A scroll is a
  pure change of position, so it is not a function of z.
- **u / w: partly in the state, and the worlds reach its linear ceiling** (U 0.63 = ridge 0.63). What u lacks is
  passability (0.69): 4x4 pooling merges the neighbour tile with its neighbours, so U predicts a scroll whether or
  not the move succeeds (blocked 19x worse than copying).
- **T: the information is there, the trained world does not use it.** A per-tile linear map captures 72%, the
  transformer 8%. Its teacher-forced L1 on its own training pool is worse than copying (0.1475 vs 0.1281).

**Why T (and sZ) cannot even beat copying: the output parameterization** (`parameterization.py`, identical
architecture, data, batches, optimizer and 4,000 updates, heads removed; held-out L1):

| | copy | direct output (as trained) | residual output s_t + f(.) |
|---|---|---|---|
| sZ | 0.0768 | 0.0965 | **0.0692** |
| T | 0.1216 | 0.1411 | **0.1124** |

`spatial.World` rebuilds every next-frame token through six layers and a LayerNorm with no copy path; with one,
the same network beats copying. Nagabandi et al. 2018 name this exact difficulty ("difficult to learn when the
states s_t and s_t+1 are too similar") and predict the change instead. On moves the residual gains only 5%, so
it fixes the floor, not the move.

### 1d. Canonical H2 specifically: the bridge broke a world that worked, and the Mamba does not extrapolate
its training length

Joint world vs bridge world on joint-like TRAIN batches (`bnmode.py`): **joint 0.66x copying, bridge 2.01x.**
BatchNorm train/eval mode is not the cause (joint: 0.73 in train mode, 0.66 in eval mode).

One-step error vs copying by position, teacher-forced from a zero state over 96-frame windows (`context.py`):

| positions | 1-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64-95 |
|---|---|---|---|---|---|---|
| joint (trained on 4-frame windows) | **0.80** | 1.52 | 1.99 | 2.06 | 2.07 | 2.25 |
| bridge (trained after 50-72 burn-in frames) | 2.64 | 1.41 | 1.55 | 1.43 | 1.36 | 1.48 |
| Z (trained on <= 6-frame windows) | 0.86 | **0.69** | 0.94 | 1.00 | 1.17 | 1.43 |
| U (trained on <= 6-frame windows) | 0.67 | **0.63** | 0.68 | 0.69 | 0.73 | 0.88 |

- Every Mamba world is best inside its training window and degrades beyond it. The joint phase (`joint.frames = 4`)
  never trains past 4 frames.
- The bridge feeds windows after 50-72 burn-in frames. Its first-update dynamics loss was 0.16 against the joint's
  0.018. 2,000 updates brought it to ~0.066, summed over its two terms (teacher and depth-2 generated), against a
  copy MSE of ~0.025 per term. From a zero state it is worse than copying at every position, and the short
  regime is lost (2.64).
- The recurrent SSM state keeps growing past the training length: 3.2-4.4x its t=4 RMS by t=96, not saturated
  (`memory.py`). This matches Chen et al. 2025 (*Stuffed Mamba*): Mamba trained on contexts too short for its
  state does not learn to forget, and the minimum training length scales with state size. Growth and degradation
  are measured; the forgetting gates themselves are not.
- The gate's recursive_dynamics check passed at depth 1 (0.032 vs 0.048) on a DEV sample whose persistence MSE is
  twice the corpus-uniform one (0.048 vs 0.024). The world beats copying only on rare high-change transitions
  (sleep onset 0.11x). That the gate sample over-weights these is inferred from that 2x, not recomputed.

## Part 2 — Why recursion compounds

`compound.py`, `persistence.py`, factual futures, depth k = 1..16:

| world | error at k=1 | at k=16 | copy root, k=16 | inherited share at k=16 | gain / step | per-direction error coef. | fresh-error cosine step to step | growth exponent | k=16 / sum of fresh |
|---|---|---|---|---|---|---|---|---|---|
| H2 | 0.048 | 1.66 | 0.94 | 0.91 | 0.99 | 0.96 | 0.67 | 1.28 | 2.46 |
| Z | 0.031 | 0.81 | 0.94 | 0.90 | 0.98 | 0.97 | 0.48 | 1.18 | 1.84 |
| sZ | 0.047 | 0.74 | 1.02 | 0.89 | 0.90 | 0.91 | 0.72 | 0.99 | 0.96 |
| U | 0.035 | 1.21 | 0.88 | 0.90 | 0.99 | 0.94 | 0.50 | 1.28 | 1.85 |
| W | 0.174 | 2.20 | 1.20 | 0.81 | 0.87 | 0.82 | 0.53 | 0.92 | 0.66 |
| T | 0.165 | 0.93 | 1.03 | 0.92 | 0.97 | 0.96 | 0.15 | 0.62 | 0.37 |

1. **81-92% of the depth-16 error is inherited** from earlier imagined steps. The step's own fresh error
   (teacher-forced) is flat with depth.
2. **Inherited error is passed on almost intact:** gain 0.87-0.99 per step, 0.82-0.97 per eigen-direction.
3. **Because the true latent process is a near-integrator.** One-step autocorrelation of the true states,
   variance-weighted: z 0.985 (99.9% of variance at >= 0.95), u 0.976, T 0.915, w 0.919. Map, HUD and inventory
   persist, so a faithful world must carry its state forward, and carries its errors with it. Lambert et al.
   2022: with poles well inside the unit circle error "rapidly reaches a steady-state value"; as poles approach
   instability (they test up to 0.95) it compounds. We are at 0.92-0.99.
4. **Fresh errors are correlated across steps** (cosine 0.48-0.72), so they add coherently: error at k=16 is
   1.8-2.5x the sum of the fresh errors for H2, Z and U (growth exponent 1.18-1.28; independent errors give 1).
   W and T saturate instead: W's imagined states collapse toward the data mean (across-root spread 0.39x the true
   by k=8), and T barely moves (its imagined step is 0.02-0.12x the true one).
5. **Fed its own predictions, a world moves less:** imagined vs teacher-forced step size at k=8 is Z 0.61 vs 0.84,
   U 0.67 vs 0.94, W 0.26 vs 1.09, sZ 0.36 vs 0.82. The imagined world stalls while the real one moves.
6. **The errors land where the slow facts live.** Per-step error falls in each state's low-variance directions
   at 22x (H2), 19x (sZ), 8.7x (Z), 6.1x (U), 3.4x (T), 1.5x (W) their variance share. This tracks how unequally
   each loss weighs its components; W, the one equal-weight state, is lowest. With per-direction coefficients
   0.63-0.92 there, the HUD, inventory and light (near-constant over 16 steps) are corrupted below copying the root.
   U at k=4: food R^2 0.35 imagined vs 0.79 copying the root; inventory 0.06 vs 0.53. At k=16 the imagined HUD
   reads R^2 -2.4 to -6.0.
7. **The recurrence also runs past its training length:** a 16-step rollout uses context positions 4-19 (1d).
8. **Off the data manifold:** Mahalanobis radius (true states ~1): H2 5.2 at k=1 and 16.5 at k=16, Z 1.4 -> 4.4,
   U 0.9 -> 3.5, T 0.8 -> 2.0.

What recursion does NOT show: no amplification (gain <= 1 everywhere), no fresh error growing with depth, no
aleatoric floor. The noise across keys is 0.01-0.065 of variance by k=16, against errors of 0.7-2.2.

## Part 3 — Corrections to earlier claims

- **"W's reading of asleep deaths is LeJEPA Lemma 1 on the head" — not supported.** A fresh head on the FROZEN U
  world, raw vs whitened input, 3 seeds (`headonly.py`): SLEEP-death AUC +0.009 to +0.013 for whitening, against
  W's 0.092 advantage over U. Head conditioning is real and about a tenth of the effect.
- **"The transition universally degrades" is too broad** (the other agent was right). Every world gets sleep, and
  Z, U and T beat copying overall. The losses are local: moves, blocked moves, interactions, idle.
- **Earlier "the U world reproduces 95.5% of each action's effect":** measured again, U captures 63% of a
  successful move's change, at the linear ceiling of u.

## Sources

- Lambert, Pister, Calandra 2022, *Investigating Compounding Prediction Errors in Learned Dynamics Models*,
  arXiv 2203.09637, §5.1 and Remark 2.
- Chen et al. 2025, *Stuffed Mamba: Oversized States Lead to the Inability to Forget*, COLM 2025, arXiv 2410.07145.
- Nagabandi et al. 2018, *Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning*,
  arXiv 1708.02596, §IV-A.
- You et al. 2026, *A Control Theory of Predictability in Latent World Models*, arXiv 2607.10362 (single-step
  validation error does not track control; off-manifold divergence is the binding term).
- Hafner et al. 2021, *DreamerV2*, arXiv 2010.02193 (discrete latents); Venkatraman et al. 2015 (*DaD*).
