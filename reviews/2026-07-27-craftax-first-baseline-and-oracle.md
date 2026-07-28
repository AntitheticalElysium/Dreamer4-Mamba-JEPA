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

## Random-encoder floor — training REMOVES the information

`reviews/artifacts/craftax_oracle_random.py`, same probe, same oracle, untrained
(randomly initialized) encoders. The oracle had constant/timestep/pixel floors
but no random-encoder floor, so part of every reported latent R^2 could have
been Johnson-Lindenstrauss projection rather than anything training produced.

Untrained versus trained, at IDENTICAL geometry (`n_latents=16, d_bottleneck=16`,
latent 256 dims):

| target | random | trained T-JEPA | delta |
|---|---:|---:|---:|
| food | 0.633 | 0.165 | **−0.47** |
| wood | 0.630 | 0.097 | **−0.53** |
| sapling | 0.664 | 0.041 | **−0.62** |
| wood_sword | 0.479 | 0.051 | −0.43 |
| stone | 0.407 | 0.082 | −0.33 |
| drink | 0.663 | 0.418 | −0.25 |
| energy | 0.558 | 0.541 | −0.02 |
| iron_pickaxe | 0.832 | 0.622 | −0.21 |
| **health** | 0.465 | **0.866** | **+0.40** |

Training moves every target DOWN except health, which is the only one with a
gradient path from a loss term (reward = +1/achievement + 0.1 x health_delta).

The geometry is not the constraint. Untrained encoders at larger `n_latents`
approach the pixel ceiling: at `n_latents=256`, food 0.944, sapling 0.932,
diamond 0.931, wood 0.879, iron 0.943 (4 targets `preserved`).

So the earlier framing was wrong in an important way: this is not "the encoder
fails to acquire task state". It is "the encoder is initialized with it and
training removes it".

Still open, and the reason T-BASE matters: this does not yet distinguish the
JEPA objective specifically from ANY training under this setup. If T-BASE also
falls below the random floor at matched geometry, the cause is something common
to our training (e.g. the tanh bottleneck saturating, or joint-training
dynamics), not the non-generative objective.

## Closest action-conditioned baseline: V-JEPA 2-AC

Read 2026-07-27 because I-JEPA/V-JEPA are not action-conditioned and are
therefore the wrong comparison. From `facebookresearch/vjepa2`
(`configs/train/vitg16/droid-256px-8f.yaml`, `app/vjepa_droid/train.py`):

| | V-JEPA 2-AC | ours |
|---|---|---|
| encoder init | pretrained V-JEPA 2 ViT-g (`pretrain_checkpoint: vitg.pt`, `load_encoder=True`) | **random** |
| encoder trained? | yes, full LR (`enc_lr_scale` default 1.0) | yes |
| bottleneck | **none** — full patch-token grid | 16 latents -> 4 packed tokens |
| tokens / clip | 256px, patch 16, tubelet 2, 8 frames | 64 patches -> 4 spatial |
| autoregressive steps | `auto_steps: 2` | `jepa_jumps: 5` |
| normalize reps | true | true (L2-normalized MSE) |

The AC objective there never has to build a representation from scratch; it
refines one already trained by masked latent prediction at scale. Registered as
a note on D032, untested here.

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

## Unattended run queue launched 2026-07-27 (all parameters)

Shared by every stage: expert replay
`7e5cdfc8...` (320 ep / 696,746 tr), split seed 20260727 -> train 256 / dev 32 /
sealed 32 (sealed never loaded), run seed 20260727, batch 8, lr 1e-4, AdamW wd
1e-2, grad clip 1.0, warmup 1,000, `sequence_length=16`, `jepa_jumps=5`,
`jepa_terminal_fraction=0.5`, `jepa_terminal_weight=8`, EMA tau 0.99 -> 0.999,
20,000 updates, device cuda. Probe: `expert_probe_v1.probe_only.pt`, 3,802
frames / 48 episodes, stride 30. Oracle split seed 20260726, margin 0.05.

| stage | what | key parameters |
|---|---|---|
| A | `n_latents` ladder | 64:16 and 256:16 (`d_bottleneck` pinned at the paper's 16) |
| B | oracle on both A rungs | — |
| C | T-BASE @ `n_latents=16` | MAE tokenizer, 20k x batch 8, objective control at EXACT baseline geometry |
| D | oracle on C | — |
| E | T-BASE @ `n_latents=64` | objective x geometry interaction |
| F | oracle on E | — |
| G | encoder time-course | one baseline world, oracle at steps 0/250/500/1k/2.5k/5k/10k/20k |

Scripts: `reviews/artifacts/craftax_{capacity,tbase,oracle_run,oracle_random,timecourse}.py`,
driven by `craftax_queue.sh` and `craftax_queue2.sh`. Logs in
`outputs/d4_mamba_jepa/queue/`.

Stage G exists because the random-encoder floor showed training REMOVES the
targets; the curve shape distinguishes an early optimization transient from the
objective grinding the information away over 20k updates, and those imply
different fixes. No architectural change is made by any stage.

## Stage A/B result: `n_latents` is eliminated too

Ladder 16/64/256 with `d_bottleneck` pinned at the paper's 16, all else at the
baseline. Dev cosine DECREASES with capacity (0.7314 / 0.6415 / 0.5219) while
training JEPA loss falls (0.186 / 0.151 / 0.141).

Trained vs the random floor at each geometry (linear R^2):

| target | n=16 rand -> trained | n=64 rand -> trained | n=256 rand -> trained |
|---|---|---|---|
| food | 0.633 -> 0.165 | 0.872 -> 0.219 | 0.944 -> 0.216 |
| sapling | 0.664 -> 0.041 | 0.845 -> 0.000 | 0.932 -> 0.335 |
| wood | 0.630 -> 0.097 | 0.783 -> 0.160 | 0.879 -> 0.302 |
| stone | 0.407 -> 0.082 | 0.719 -> 0.111 | 0.797 -> 0.201 |
| energy | 0.558 -> 0.541 | 0.855 -> 0.628 | 0.920 -> 0.451 |
| health | 0.465 -> 0.866 | 0.770 -> 0.916 | 0.843 -> 0.907 |

Preserved counts stay at 1/16 (health) at every rung. The random floor converts
extra capacity into retained state; the trained encoder does not, so the gap
WIDENS with capacity (food 0.47 -> 0.65 -> 0.73).

Qualification: trained values do rise somewhat (wood 0.097 -> 0.302, sapling
0.041 -> 0.335), so capacity is not irrelevant -- it is insufficient, and it
never changes a verdict.

INSTRUMENT CAVEAT: the nonlinear latent probe is unreliable at high latent
dimension and must not be quoted there. At n=256 it repeatedly falls BELOW the
linear probe (food 0.216 lin / 0.057 non; stone_pickaxe 0.542 / -0.053), which
is a 128-unit MLP overfitting 4,096 features on ~1,900 training frames. Verdicts
use max(linear, nonlinear) so they are unaffected, but the "MLP ~= linear,
therefore genuinely absent" reading established at n=16 does NOT transfer up the
ladder.

Both capacity axes are now eliminated (`d_bottleneck` 16/32/64, `n_latents`
16/64/256). What remains under test is the training itself; T-BASE at matched
geometry is the discriminator between the JEPA objective specifically and any
training in this setup.

## Stage C/D: T-BASE separates the objective effect from a training-common effect

T-BASE at `n_latents=16` (identical geometry to the baseline JEPA arm), MAE
tokenizer 20k x batch 8, reconstruction loss 0.0697 -> 0.0094. In T-BASE the
encoder is trained by the tokenizer phase and then FROZEN, so this is exactly
the encoder T-BASE deploys -- tokenizer-only is the correct isolation, not a
confound.

Inventory mean over all 12 targets: **random 0.661 > T-BASE 0.521 > T-JEPA 0.320**

| target | random | T-JEPA | T-BASE |
|---|---:|---:|---:|
| sapling | 0.664 | 0.041 | 0.576 |
| wood | 0.630 | 0.097 | 0.464 |
| stone | 0.407 | 0.082 | 0.383 |
| diamond | 0.733 | 0.514 | 0.863 |
| iron | 0.693 | 0.559 | 0.790 |
| iron_sword | 0.793 | 0.816 | 0.949 |
| food | 0.633 | 0.165 | 0.141 |
| health | 0.465 | 0.866 | 0.568 |

Both pre-registered possibilities are true and separable:

1. The OBJECTIVE matters: reconstruction retains ~63% more inventory information
   than self-prediction at matched geometry (14x on sapling).
2. A TRAINING-COMMON effect also exists: T-BASE is below the random floor on 11
   of 16 targets, so the JEPA objective is not the whole story.

Health confirms the "loss demands it" mechanism in reverse. T-BASE is
tokenizer-only and has NO reward head; its health falls from T-JEPA's 0.866 to
0.568, toward the random floor of 0.465. Health is high precisely when a loss
term supervises it.

## Refuted: dimensional collapse (`craftax_latent_rank.py`)

Hypothesis: the training-common effect is the latent collapsing to a low-rank
subspace, which `online_std` cannot see because it measures per-dimension
variance rather than rank.

REFUTED, in the opposite direction:

| encoder | dims | effective rank | var top-1 | dead dims |
|---|---:|---:|---:|---:|
| RANDOM n16 | 256 | 4.0 | 0.563 | 0.00 |
| T-JEPA n16 | 256 | 12.0 | 0.242 | 0.00 |
| M-JEPA n16 | 256 | 13.1 | 0.240 | 0.00 |
| T-BASE n16 | 256 | 30.3 | 0.217 | 0.00 |
| RANDOM n256 | 4096 | 3.0 | 0.740 | 0.00 |
| T-JEPA n256 | 4096 | 7.7 | 0.278 | 0.00 |

Training RAISES effective rank, and the encoder retaining the most information
(random, 0.661) has the LOWEST rank (4.0). No dead dimensions anywhere.

The metric was also the wrong instrument for the question: effective rank
measures variance concentration, not extractable information. A random
projection concentrates variance in few directions while staying near-injective,
so ridge recovers targets from low-variance directions. Rank and retention are
not the same axis. Recorded as a refuted mechanism, not a finding.

## Causal status (only what is demonstrated)

- Capacity eliminated on both axes (`d_bottleneck` 16/32/64, `n_latents` 16/64/256).
- BC under-training eliminated (30k ladder, flat from 10k).
- The objective demonstrably matters (reconstruction >> self-prediction).
- A training-common effect also exists and is NOT rank collapse.
- Per-target retention tracks loss supervision (health).

No single cause is claimed for the residual; nothing measured identifies it
uniquely.

## Stage E/F: the objective x capacity interaction

T-BASE at `n_latents=64`, everything else identical. Inventory mean R^2 over all
12 targets:

| encoder | n=16 | n=64 | gain from capacity |
|---|---:|---:|---:|
| random (untrained) | 0.661 | 0.803 | +0.14 |
| T-BASE (reconstruction) | 0.521 | 0.712 | **+0.19** |
| T-JEPA (self-prediction) | 0.320 | 0.348 | **+0.03** |

Per target, e.g. food: T-BASE 0.141 -> 0.726, T-JEPA 0.165 -> 0.219. Sapling:
T-BASE 0.576 -> 0.832, T-JEPA 0.041 -> 0.000.

This sharpens the capacity conclusion rather than contradicting it. Capacity is
eliminated FOR THE SELF-PREDICTION OBJECTIVE, and that is now a non-trivial
claim, because the same capacity is demonstrably usable: both the random encoder
and the reconstruction objective convert it into retained state.

It also revises the "training-common effect" recorded at stage D. T-BASE's
deficit against the random floor SHRINKS with capacity (0.14 at n=16 -> 0.09 at
n=64) while T-JEPA's GROWS (0.34 -> 0.455). One account covers both without
needing a second mechanism:

  reconstruction's incentive is proportional to pixel area, so the 4-pixel HUD
  digits are under-weighted when capacity is scarce and retained when it is not;
  self-prediction has no comparable incentive at any capacity.

Recorded as the LEADING READING, not a conclusion. It is consistent with every
measurement so far (including health, which is retained by whichever arm has a
loss term touching it), but no single measurement isolates it.
