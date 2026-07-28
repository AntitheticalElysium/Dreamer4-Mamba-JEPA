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

## Stage G: the time-course — flat, then monotone decay

One baseline world (`n_latents=16`, every parameter the baseline's), oracle at
steps 0/250/500/1k/2.5k/5k/10k/20k. Latent linear R^2:

| step | dev_cos | health | food | drink | energy | wood | stone | sapling | wood_sword | iron_sword | diamond |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.03 | 0.47 | 0.63 | 0.66 | 0.56 | 0.63 | 0.41 | 0.66 | 0.48 | 0.79 | 0.73 |
| 250 | +0.29 | 0.48 | 0.62 | 0.65 | 0.55 | 0.61 | 0.40 | 0.67 | 0.48 | 0.81 | 0.72 |
| 500 | +0.46 | 0.48 | 0.60 | 0.64 | 0.54 | 0.62 | 0.40 | 0.65 | 0.48 | 0.83 | 0.70 |
| 1,000 | +0.46 | 0.82 | 0.65 | 0.63 | 0.59 | 0.59 | 0.37 | 0.55 | 0.31 | 0.81 | 0.59 |
| 2,500 | +0.60 | 0.94 | 0.30 | 0.50 | 0.47 | 0.38 | 0.27 | 0.28 | 0.13 | 0.68 | 0.51 |
| 5,000 | +0.70 | 0.94 | 0.21 | 0.42 | 0.40 | 0.36 | 0.29 | 0.14 | −0.00 | 0.59 | 0.67 |
| 10,000 | +0.72 | 0.91 | 0.19 | 0.37 | 0.37 | 0.30 | 0.25 | 0.02 | 0.07 | 0.49 | 0.50 |
| 20,000 | +0.73 | 0.89 | 0.15 | 0.36 | 0.52 | 0.35 | 0.19 | −0.01 | 0.06 | 0.44 | 0.39 |

Against the shapes pre-declared in the script before it ran:

* NOT an early transient. Through step 500 the encoder learns to self-predict
  (dev cosine 0 -> 0.46) with task state completely flat.
* IS monotone decay over the remaining 19,000 steps -- the pre-declared reading
  for which is "the objective grinding it away, and the loss is the thing to
  change".

Two further readings:

* The objective and the task state are in DIRECT COMPETITION. Dev cosine rises
  monotonically 0.46 -> 0.73 across exactly the interval in which food, sapling,
  wood_sword and diamond fall.
* The encoder REALLOCATES rather than merely forgetting. Health rises 0.48 ->
  0.94 over steps 1,000-2,500, the same window in which everything else starts
  to fall. It acquires what the reward head supervises while shedding what
  nothing supervises -- the health mechanism seen a third time, now with a time
  axis.

Onset coincides with the end of the 1,000-step linear warmup, i.e. with full-LR
optimization beginning. The EMA tau ramp is smooth and has no discontinuity
there, so this is not a separate schedule effect.

### What stage G does and does not establish

It isolates THE OBJECTIVE as the cause, holding architecture, capacity, the tanh
bottleneck, data and budget fixed -- the T-BASE control at identical geometry is
what licenses that. It does NOT identify WHICH COMPONENT: the self-prediction
loss, the SPR/BYOL EMA anti-collapse (D031), and the task heads are not
separated. Stage H tests that.

## Stage H (running): component decomposition

`craftax_timecourse.py --jepa-weight 0.0`: identical run with the
self-prediction term removed, so the encoder is trained ONLY by the
reward/continuation heads. If the decay vanishes, self-prediction is the cause;
if it persists, the heads/dynamics are. Diagnostic on an existing code path; no
architectural change is proposed by it.

## Stage H, first attempt: VOID — a dead config field

The first `--jepa-weight 0.0` run reproduced stage G BIT-FOR-BIT, every value
including `dev_cos`. That is not a result; it is proof the flag did nothing.

Cause: `cfg.jepa_weight` was declared in `D4LiteConfig` (config.py:72) and
validated in `__post_init__`, but **read by no code**. The live knob is
`LossWeights.jepa`, consumed by `world_loss`. Anyone setting the config field --
including this diagnostic -- silently trained the ordinary baseline.

Fixed: `_jepa_world_loss` now honours `cfg.jepa_weight` as a multiplier
alongside the caller-supplied `LossWeights`. Both default to 1.0, so every
result recorded above is bit-unchanged. Regression added
(`test_cfg_jepa_weight_is_honoured_not_a_dead_field`): default leaves total ==
raw term, 0.0 removes the contribution, and the reported raw term is unaffected.
Suite: 108 passed.

Re-run with the live knob confirms it now bites: at step 250, `dev_cos` +0.069
against +0.292 for the baseline, with targets holding or improving (food 0.68 vs
0.62, wood 0.67 vs 0.61, stone 0.49 vs 0.40). The real stage H is running.

The void run is retained as evidence in the log; no conclusion was drawn from it.

## Stage H (corrected): self-prediction is NOT the culprit

`LossWeights.jepa=0.0` -- encoder trained ONLY by the reward/continuation heads.
`dev_cos` stays 0.03-0.11 throughout, confirming no self-prediction occurred.

| target | init | baseline @20k | heads-only @20k |
|---|---:|---:|---:|
| food | 0.63 | 0.15 | 0.08 |
| wood | 0.63 | 0.35 | 0.10 |
| diamond | 0.73 | 0.39 | -0.19 |
| iron_sword | 0.79 | 0.44 | 0.61 |
| stone | 0.41 | 0.19 | 0.14 |
| sapling | 0.66 | -0.01 | -0.02 |
| health | 0.47 | 0.89 | 0.80 |

REFUTED: that the self-prediction loss (or the D031 EMA anti-collapse) is what
discards task state. Removing it makes retention WORSE on most targets, so it is
partially PROTECTIVE. Health still rises, because health is still in the reward.

### Leading account (not a conclusion)

Retention tracks the DIMENSIONALITY AND COVERAGE of the training target, not the
objective family:

| training target | target dims | inventory mean R^2 |
|---|---:|---:|
| full frame (T-BASE reconstruction) | 12,288 | 0.521 |
| next latent + heads (T-JEPA baseline) | 256 + 2 | 0.320 |
| reward + continue only (stage H) | 2 | lowest on most |
| none (random init) | -- | 0.661 |

This fits every measurement today: the objective swap, the capacity interaction
(a full-frame target can use extra capacity; a 2-scalar target has nothing to
use it for), the time-course, and health.

NOT excluded: with `jepa=0` there is no anti-collapse at all and rank was not
measured on that run, so partial collapse could contribute to its position.

### Consequence for direction (for the user to weigh)

SIGReg is an anti-collapse regularizer on the embedding DISTRIBUTION; it does
not increase target coverage. If coverage is what determines retention, SIGReg
addresses a different failure mode than the one measured here. This should be
settled before committing to it.

## 2026-07-28 — second-agent claims verified

### CORRECTED: V-JEPA 2-AC freezes its encoder (my D032 note was wrong)

I recorded on 2026-07-27 that V-JEPA 2-AC "also trains its encoder (full LR by
default)" because the encoder appears in `init_opt`'s param groups. That was
WRONG. In the training step (`app/vjepa_droid/train.py:408-449`) the encoder is
absent from the loss graph:

```python
def forward_target(c):
    with torch.no_grad():
        h = target_encoder(c)      # the ONLY encoder call in the step
h = forward_target(clips)
z_tf, z_ar = forward_predictions(h)   # predictor only
loss = loss_fn(z_tf, h) + loss_fn(z_ar, h)
```

`grep "encoder("` over the file returns exactly one hit. So V-JEPA 2-AC
effectively FREEZES its pretrained encoder. Dreamer 4 likewise freezes its
pretrained tokenizer during dynamics learning. Our arm jointly trains a RANDOM
encoder at full LR through a 4-token bottleneck. Ledger D032 corrected.

### Other claims, all verified against pinned sources

| claim | verdict | evidence |
|---|---|---|
| Predictor collapses to a 64-d channel | CORRECT | `CDPPredictor.forward` does `context = agent_tokens.mean(dim=2)`, then `Linear(2*64, 64)`; the `n_latents` ladder widened the OUTPUT only |
| `WorldLossNormalizer` omits JEPA | CORRECT | `_jepa_world_loss` normalizes reward/continuation only; no `"jepa"` EmaRms term is even registered |
| Dreamer-CDP separates encoder LR | CORRECT | `enc_lr: 6e-6` vs `dyn_lr: 4e-4` = 66.7x (`dreamerv3/configs.yaml:85`) |
| SPR is not a standalone precedent | CORRECT | `src/algos.py:131` optimizes RL + reward + SPR jointly through `stem_parameters()` |

The predictor point invalidates my own framing: the `n_latents` 16/64/256 ladder
did NOT test the predictor/projection bottleneck, because both stay 64-d
regardless. My description of the self-prediction target as "256 dimensions" was
misleading.

### Sampler audit (`craftax_sampler_audit.py`) — mechanism confirmed, magnitudes overstated

Measured by running the REAL sampler over exact episode structure (true lengths,
rewards, continues; 1x1 dummy pixels). The operative slice is the HEAD window:
`_jepa_world_loss` reads heads at `[context, context+K)` with K=`jepa_jumps`=5,
and a terminal window places its terminal at the final position, inside it.

| quantity | claimed | measured (head window) | replay |
|---|---:|---:|---:|
| terminal fraction in head loss | 16.68% | **10.01%** | 0.0105% |
| amplification | 1,590x | **955x** | — |
| continuation BCE mass on terminals | 61.6% | **47.1%** | — |
| negative-reward frequency | 17.78% (21.8x) | **11.49% (14.1x)** | 0.815% |
| positive-reward frequency | ~unchanged | 2.10% (1.04x) | 2.02% |
| mean reward label | +0.0094 -> −0.0238 | +0.0094 -> **−0.0127** | +0.0094 |

At `terminal_fraction=0.0`: terminal fraction 0.019%, BCE mass 0.15%, reward
mean +0.0143.

The claimed 16.68% double-counted MTP expansion (a terminal occupies several
head slots, but so does every other target, so the FRACTION barely moves:
3.34% -> 3.44% over the full sequence). The real route to 10.01% is the 5-position
head window, which they did not state.

### CORRECTION to an earlier finding of mine

I reported the reward heads as "18x under-predicting" and "wrong sign" by
comparing imagined reward against the REPLAY mean of +0.00939. That is the wrong
reference: the model never trained on that distribution. Against the labels it
actually saw (−0.0127), T-JEPA's imagined +0.00051 is wrong in the opposite
direction and M-JEPA's −0.0259 is roughly right in sign. The heads remain
miscalibrated; my stated diagnosis of HOW was measured against a distribution
the model never saw.

## Paired sampler control (2,500 updates, same init) — mixed verdict

Pre-declared criterion (second agent): "if removing forced terminal windows
prevents the health spike AND state erosion, the primary cause is established."

| target | init | tf=0.5 | tf=0.0 |
|---|---:|---:|---:|
| health | 0.47 | **0.94** | **−0.06** |
| food | 0.63 | 0.45 | 0.24 |
| drink | 0.66 | 0.51 | 0.48 |
| energy | 0.56 | 0.59 | 0.40 |
| wood | 0.63 | 0.34 | 0.47 |
| stone | 0.41 | 0.37 | 0.32 |
| sapling | 0.66 | 0.37 | 0.52 |
| wood_sword | 0.48 | 0.20 | 0.27 |
| iron_sword | 0.79 | 0.52 | 0.70 |
| diamond | 0.73 | 0.63 | 0.66 |
| dev_cos | — | 0.548 | 0.528 |

CONFIRMED: the health spike is ENTIRELY the sampler. Without forced terminal
windows health is not merely un-spiked, it is destroyed (0.47 -> −0.06). The
health signature previously read as evidence of "reward supervision preserves
what it touches" is a product of the terminal oversampling.

NOT CONFIRMED (their own criterion): erosion persists without the sampler --
wood 0.63->0.47, sapling 0.66->0.52, food 0.63->0.24, iron_sword 0.79->0.70.
Inventory-target mean 0.617 init -> 0.405 (tf=0.5) -> 0.490 (tf=0.0), so the
sampler accounts for roughly 40% of the erosion at this budget, not all of it.

NEW: removing the sampler makes VITALS worse (mean 0.62 -> 0.27), because death
prediction was the only thing making vitals task-relevant. The sampler was
simultaneously distorting the objective and supplying the only pressure keeping
vitals in the latent.

This refines the "what the loss demands" account rather than overturning it:
inventory is unsupervised under both samplers and erodes under both.

### Reproducibility caveat, quantified

i1 (tf=0.5) is a fresh rerun of the baseline configuration and differs
materially from stage G at the same step (food 0.45 vs 0.30, sapling 0.37 vs
0.28, diamond 0.63 vs 0.51) despite nominally identical settings. Per-target
magnitudes are single-run evidence; only the paired i1/i2 comparison is valid,
not i2 against stage G. The direction of degradation is stable across all runs.

## Running: self-prediction-only arms to complete the 2x2

| | all losses | heads only | self-prediction only |
|---|---|---|---|
| tf=0.5 | i1 (done) | h2 (done, 20k) | j2 (running) |
| tf=0.0 | i2 (done) | — | j1 (running) |

## Completed 2x2: self-prediction is the primary cause of inventory erosion

All cells 2,500 updates from a shared initialization. Inventory mean over
{wood, stone, sapling, wood_sword, iron_sword, diamond}; vitals over
{health, food, drink, energy}.

| arm | inventory | vitals | dev_cos |
|---|---:|---:|---:|
| INIT (shared) | 0.618 | 0.580 | −0.03 |
| tf=0.5, all losses (baseline) | 0.404 | 0.622 | 0.548 |
| tf=0.5, self-prediction only | 0.405 | 0.359 | 0.629 |
| tf=0.0, all losses | 0.491 | 0.264 | 0.528 |
| tf=0.0, self-prediction only | 0.474 | 0.356 | 0.520 |

1. THE TASK HEADS CONTRIBUTE ~NOTHING TO INVENTORY EROSION: 0.404 vs 0.405 at
   tf=0.5, and 0.491 vs 0.474 at tf=0.0 (within measured run-to-run variance).
2. SELF-PREDICTION ALONE, CLEAN SAMPLER, STILL ERODES: 0.618 -> 0.474, ~67% of
   the baseline's total erosion. This is the pre-declared condition for blaming
   SPR/global prediction and it is MET.
3. The sampler owns the remaining ~40% of the baseline's extra erosion
   (0.404 -> 0.491) and, with the heads, the entire vitals story.
4. Sign flip on vitals: under terminal oversampling the heads PRESERVE vitals
   (0.622); under uniform sampling they DEGRADE them (0.264 vs 0.356 without
   heads), because uniform sampling makes continue~1 and reward~0 nearly
   everywhere, so the heads supply a near-constant target.

BUDGET CAVEAT: the earlier heads-only run was 20,000 updates and did erode badly
(food 0.08, wood 0.10). These cells are 2,500. "Heads contribute nothing" is
established at 2,500 updates only; the two budgets are not comparable.

### Mechanism candidate, now specific

`CDPPredictor.forward` mean-pools the agent tokens to ONE 64-d vector
(`context = agent_tokens.mean(dim=2)`), passes through a 64-d hidden layer, and
the SPR loss compares 64-d projections. The self-prediction target therefore
cannot require the encoder to retain twelve independent inventory counts --
there is no channel through which they would be needed. This is consistent with
every cell above and with the `n_latents` ladder having produced no improvement
(it widened the output, never this channel). Not yet isolated by an experiment.

### Status of the ranked diagnosis

| candidate | status after these runs |
|---|---|
| Terminal sampling | CONTRIBUTOR, ~40% of extra erosion; fully owns the health signature; NOT the primary cause by its own pre-declared test |
| Self-prediction / global 64-d predictor | PRIMARY cause of inventory erosion, isolated with heads off and sampler clean |
| Task heads | ~zero contribution to inventory at 2.5k; own the vitals story; untested at 20k under a clean sampler |
| Unanchored full-LR encoder | UNTESTED; strongest remaining structural divergence (both references freeze a pretrained encoder) |
| Loss-scale mismatch (JEPA unnormalized) | UNTESTED, verified present |
| SIGReg / collapse | deprioritized by the user; rank collapse already refuted |

## D045: widening the predictor channel — NEGATIVE

`jepa_predictor_context`: pooled agent (64-d) -> spatial stream + agent tokens
(384-d), hidden width scaled with it. Default unchanged and bit-identical;
108 existing tests plus 2 new ones pass.

| arm | channel | inventory | vitals | dev_cos |
|---|---|---:|---:|---:|
| INIT | — | 0.618 | 0.580 | — |
| tf=0.0, jepa-only, pooled | 64 | 0.474 | 0.356 | 0.520 |
| tf=0.0, jepa-only, spatial+agent | 384 | **0.477** | 0.417 | 0.496 |
| tf=0.5, all losses, pooled | 64 | 0.404 | 0.622 | 0.548 |
| tf=0.5, all losses, spatial+agent | 384 | **0.439** | 0.527 | 0.608 |

A 6x wider channel moves the isolated cell by +0.003. The baseline cell's +0.035
is NOT claimed: stage G and i1, nominally identical, gave 0.375 and 0.404 (spread
0.029), so one run per cell cannot separate it.

The change works mechanically (dev cosine 0.548 -> 0.608 in the baseline cell),
so the predictor does exploit the wider context; it just does not reduce erosion.

WHY, per the pre-registered first suspect in D045: the SPR loss projects through
`JepaProjector` to a GLOBAL 64-d vector (`jepa_projection_dim=64`). The encoder's
gradient is therefore a 64-d global comparison regardless of the predictor's input
width. Widening one side of a two-sided bottleneck predictably did nothing.

Live candidate: the loss-side projection. Not run; awaiting approval.
