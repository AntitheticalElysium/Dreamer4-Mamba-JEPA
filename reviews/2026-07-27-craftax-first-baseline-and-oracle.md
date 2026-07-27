# Craftax first baseline (T-JEPA / M-JEPA) + representation oracle

Date: 2026-07-27
Branch: `craftax-clean-baseline`
Deviations: none. No architectural change from a pinned source; the capacity
fix below was a defect and the runners are runners.

## Data

`d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt`
sha256 `7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b`
320 episodes / 696,746 transitions. Split 80/10/10 by whole episode
(`split_seed=20260727`): train 256 (553,145), dev 32 (73,182), sealed 32 (never
loaded).

Audited properties (measured, not read off the manifest):

| property | value |
|---|---|
| episodes truncated at the 2,500-step generation cap | **252 / 320 (78.8%)** |
| episodes ending in a real terminal | 68 (66,746 transitions, 9.6%) |
| terminal episodes in the train split | 58 / 256 |
| mean reward / step | 0.00939 (train), 0.00900 (dev) |
| nonzero-reward fraction | 2.83% |
| mean continue / step | **0.99990** (1 terminal in ~9,500) |
| dev majority action | `do` at **0.1492** |
| action label entropy | 2.62 nats |

The 2,500-step cap is still in force. Expert survival is ~9,285 steps, so ~73%
of expert play is absent from the dataset. Regenerating uncapped is ~36 GB at
64x64, which does not fit in RAM as one `.pt` on this machine.

## Defect fixed

`load_episode_replay` defaulted to `capacity_steps=500_000` and FIFO-dropped the
oldest episodes past it, silently. The train split alone is 553,145 transitions,
so this run would have lost training data with no error. Now defaults to `None`
(sized to the file), errors on an explicit too-small capacity, and asserts
episode/transition counts survived the load. Two regressions in
`tests/test_baseline.py`.

## Results

| | T-JEPA | M-JEPA |
|---|---:|---:|
| world dev cosine | 0.7314 | 0.7350 |
| final JEPA loss | 0.1861 | 0.1421 |
| online std (collapse monitor) | 0.0970 | 0.0978 |
| world time / peak VRAM | 62.4 min / 451 MB | 75.9 min / 761 MB |
| BC dev accuracy @3k (inherited) | 0.1594 | 0.1557 |
| **BC dev accuracy @30k** | **0.1833** | **0.1901** |
| imagined mean reward | 0.00051 | −0.0259 |
| imagined mean continue | 0.9999977 | 0.8440 |
| imagined return std | 0.0162 | 0.0611 |
| PMPO pos/neg | 1030/1018 | 1177/871 |

It runs, cheaply and stably, and it does not work.

**Mamba vs Transformer is indistinguishable** on every measure: +0.004 dev
cosine, and a BC difference of 0.007 against a dev standard error of ~0.009.

**Reward heads are wrong in both arms**: T-JEPA under-predicts 18x
(0.00051 vs 0.00939 empirical); M-JEPA has the wrong sign. M-JEPA gives the
actor 3.8x more return spread, but manufactured from a mis-signed reward.

**Continuation is NOT broken.** An earlier reading of `mean_continue≈1.0` as
majority-label collapse was wrong: the true base rate is 0.99990, so ≈1.0 is
nearly correct. The real point is structural — at horizon 32 with a 1e-4 death
rate, continuation cannot carry signal at all. M-JEPA's 0.844 is a large error
in the *opposite* direction.

### BC budget control (`reviews/artifacts/craftax_bc_budget.py`)

Both conclusions hold at once. The inherited 3k budget was under-training (cost
~0.03 accuracy). And a ceiling is real: T-JEPA is flat from 10k onward
(0.1833 → 0.1797 → 0.1818 → 0.1818 → 0.1833), train CE flattening at 2.426.
Fully trained, BC beats the majority floor by +0.034 (T) / +0.041 (M),
capturing ~0.19 of 2.62 nats.

## Oracle verdict — the encoder is the bottleneck

`reviews/artifacts/craftax_oracle_run.py` on 3,802 expert frames / 48 episodes
(`d4_mamba_jepa/expert/probe.py`, stride 30). Probe frames come from the same
PPO expert that produced the training replay, so a `lost`/`degraded` verdict is
not distribution shift.

**Self-audit PASSES both arms**: perfect 1.000, constant −0.037, misaligned
−0.040, shift −0.004.

| target | pixel linear R² | T-JEPA latent | M-JEPA latent |
|---|---:|---:|---:|
| food | 1.000 | 0.165 | 0.243 |
| drink | 0.999 | 0.418 | 0.359 |
| energy | 1.000 | 0.541 | 0.600 |
| wood | 0.990 | 0.097 | 0.146 |
| stone | 0.999 | 0.082 | 0.184 |
| sapling | 1.000 | 0.041 | −0.066 |
| wood_sword | 0.825 | 0.051 | 0.012 |
| stone_pickaxe | 0.850 | 0.074 | 0.067 |
| **health** | 0.796 | **0.866** | **0.879** |

Verdicts over 16 continuous targets — **T-JEPA: 0 preserved / 14 degraded /
2 lost. M-JEPA: 1 preserved / 12 degraded / 3 lost.**

These quantities are drawn as HUD icons at fixed pixel locations, so a linear
pixel probe reads them at R² ≈ 1.0. The latent (4x64 = 256 dims vs 12,288
pixels) keeps almost none of it.

**Health is the one exception, and the tell.** It is the only one of the sixteen
that enters the training signal (reward = +1/achievement + 0.1 x health_delta).
Recorded as a mechanism consistent with the data, NOT a proven cause: the latent
may retain only what a loss term demands.

### Caveat on the achievement AUROCs

The reported 0.953 / 0.930 are **not usable as evidence**. Unlike the continuous
groups, `achievement_group` scores the latent with no constant, timestep, or
pixel reference, and achievements are cumulative and monotone within an episode,
so an episode-progress probe would score high. Instrument gap, not a result.

## Direction

The oracle localizes the failure to the encoder. The next two runs test why:

1. **Capacity ablation** — `d_bottleneck` 16 → 32 → 64 (256/512/1024 latent
   dims), world phase only, oracle on each. Single axis.
2. **T-BASE** — reconstruction-trained encoder, same probe and oracle.

T-BASE is a **diagnostic control only**. The JEPA line is deliberately
reconstruction-free and the end goal is SIGReg (D036, rejected on CartPole for
low intrinsic dimension; Craftax is exactly the higher-dimensional task its
rejection note said to re-open it on). If T-BASE preserves what JEPA loses, that
identifies what a reconstruction-free objective must supply by other means — it
is not a route to adopt reconstruction.
