# Architecture

One block per component, in dataflow order. `div:` lists divergences from the
pinned source. Absent `div:` means faithful. Config values are the live
`craftax_jepa_config()`.

Status: `OK` faithful · `BY DESIGN` deliberate divergence · `OPEN` unresolved ·
`BUG` known defect.

---

### 1. Data — `craftax_env.py`, `expert/generate.py`, `data.py`
method: offline expert replay, Craftax-Classic pixels, PPO expert (Crafter score 91.2)
params: 320 episodes / 696,746 transitions · 64x64x3 uint8 (63 native, zero-padded) · 17 actions · 80/10/10 whole-episode split (seed 20260727)
source: craftax 1.6.1, digest-pinned in `source.py`
div: **D047 BY DESIGN** — episodes capped at 2,500 steps; expert survives ~9,285, so 73% of expert play is absent. Uncapped is ~36 GB, exceeds RAM.
div: **D048 OPEN** — 58 of 256 train episodes end in a real terminal (0.0105% of transitions).

### 2. Sampler — `cartpole_baseline.py:sample_cartpole_sequences`
method: episode-bounded windows; `terminal_fraction` of rows forced to an episode's LAST window
params: `sequence_length=16` · `jepa_terminal_fraction=0.5` · batch 8
div: **D035 BY DESIGN** — terminal oversampling so the continuation head sees episode ends.
div: **D049 OPEN** — measured effect: after MTP expansion 16.67% of continuation targets are terminals (955x the replay rate), 61.5% of BCE mass, and the mean reward LABEL flips +0.0094 -> -0.0238.
div: **D050 OPEN** — `terminal_fraction=0` is episode-uniform, not transition-uniform; it over-weights short episodes.

### 3. Encoder / tokenizer — `model.py:build_tokenizer`
method: MMBench2 causal ViT tokenizer, loaded by path + digest
params: `patch_size=8` -> 64 patches · `tokenizer_d_model=64` · depth 4 · heads 4 · `n_latents=16` · `d_bottleneck=16` · MAE `p_min=0, p_max=0.9`
source: MMBench2 `model.py` (D000)
div: **D001 BY DESIGN** — Dreamer 4 uses `N_b=512 x D_b=16` from 960 patch tokens; we use `16 x 16` from 64. `d_bottleneck` matches; `n_latents` is 32x below.
div: **D051 OPEN** — trained from random init at full LR. Every reference anchors it: Dreamer 4 and V-JEPA 2-AC freeze a pretrained encoder; Dreamer-CDP uses `enc_lr 6e-6` vs `dyn_lr 4e-4`; SPR anchors with Q-learning. `enc_lr=6e-6` removes ~all representation erosion (ABL-16).

### 4. Packing — `config.py`
method: reshape bottleneck to spatial tokens for the dynamics
params: `packing_factor=4` -> `n_spatial=4` x `d_spatial=64` (256 floats/frame)
div: covered by D001. Dreamer 4 packs to `N_z=256 x 32`.

### 5. Dynamics — `model.py:D4LiteWorld.dynamics` (MMBench2 `Dynamics`)
method: block-causal stack; spatial attention + temporal mixer + MLP
params: `dynamics_d_model=64` · depth 4 · heads 4 · `time_every=2` (2 temporal modules) · `n_register=2` · `n_agent=2` · `k_max=4`
source: MMBench2, unmodified classes
div: **D002 BY DESIGN** — 17-way discrete action embedding replaces the continuous 16-D action MLP.

### 6. Temporal operator — `temporal.py:MambaTimeMixer`
method: `TimeSelfAttention` (T arm) or official `Mamba2` (M arm), swapped in place
params (M): `d_state=64` · `headdim=64` · `expand=1` · `d_conv=4` · `ngroups=1`
source: `state-spaces/mamba`, digest-pinned
div: **D004/D037 BY DESIGN** — the single research axis.
div: **D052 OPEN** — `d_inner = d_model x expand = 64` and `headdim=64`, so the M arm runs with **exactly one head**. Not chosen; a consequence of the D022 state expansion.
div: **D053 OPEN** — arms are not parameter-matched: 986,348 (T) vs 996,202 (M) total, temporal modules **+29.6%**. D037 says "single moved axis" without disclosing the size change.
div: **D046 BUG (FIXED)** — the Mamba swap ran before the JEPA modules were built, consuming RNG, so 16 shared tensors differed at init between arms. Swap now runs last; regression pins 227/227 identical.

### 7. JEPA predictor — `model.py:CDPPredictor`
method: deterministic action-conditioned next-latent MLP; **this is the entire rollout**
params: context = mean over 2 agent tokens (64) + action (64) -> hidden 64 -> `n_spatial*d_spatial`
source: shaped after Dreamer-CDP's predictor
div: **D030 BY DESIGN** — non-generative; no denoiser, decoder, or MAE at rollout.
div: **D045 REJECTED** — widening the context to 384 does not reduce erosion.
div: **D054 OPEN** — Dreamer-CDP predicts from an **8192-d** deterministic state (`configs.yaml:93`) with a **token-wise** cosine (`axis=-1`); ours is a 64-d pooled channel compared through a global projection. 128x narrower.

### 8. Anti-collapse — `model.py:JepaProjector`, `objectives.py:jepa_self_prediction_loss`
method: SPR/BYOL stop-grad EMA target encoder; asymmetric online projection+prediction vs EMA target projection; L2-normalized MSE
params: `jepa_projection_dim=64` (global) · `jepa_jumps=5` autoregressive steps · EMA tau 0.99 -> 0.999 linear
source: `mila-iqia/spr` `do_spr_loss`; ramp form from I-JEPA
div: **D031/D033/D034/D044 BY DESIGN** — `renormalize` omitted deliberately (D033); ramp endpoints local (D044).
div: **D055 OPEN** — SPR's visual augmentation (`kornia` RandomAffine et al.) is omitted entirely.
div: **D056 OPEN** — SPR is **auxiliary** in its source (`algos.py:131` optimizes RL + reward + SPR jointly); ours runs self-prediction as the primary encoder signal.

### 9. Loss composition — `training.py:_jepa_world_loss`
method: `jepa_weight*jepa + reward*reward_n + continuation*continuation_n`
params: all weights 1.0 · AdamW lr 1e-4, wd 1e-2 · clip 1.0 · warmup 1,000
div: **D057 OPEN** — `WorldLossNormalizer` normalizes reward and continuation but **not** JEPA; no `"jepa"` EmaRms term is registered. The three terms are not on a common scale.

### 10. Task heads — `model.py:RewardHeadMTP`, `ContinuationHeadMTP`
method: MTP heads read post-transition agent tokens of the **imagined** rollout
params: `reward_horizon=8` · `reward_bins=255` · symlog `[-10,10]` · `continuation_horizon=8` · `jepa_terminal_weight=8`
div: **D003/D012/D043 BY DESIGN** — heads trained on the same pass they are read on at deployment.

### 11. BC policy — `cartpole_baseline.py:CartPoleBCPolicy`
method: attention-pooled categorical head; world frozen
params: 3,000 updates · batch 16 · lr 1e-4
source: MMBench2 `PolicyHeadMTP`
div: **D014 BY DESIGN** — head shape and gradient boundary follow source.
div: **D058 OPEN** — the Dreamer-4 reproduction optimizes `p["dyn"]` during BC (`train_bc_rew_heads.py:374`), i.e. it tunes **dynamics** jointly with policy and reward heads. We freeze the whole world. D014 verified the head, never the phase scope.

### 12. Imagination — `imagination_actor_critic.py`
method: Dreamer-4 actor/value; actor init from BC, frozen BC prior, TD-lambda value, balanced PMPO + reverse KL
params: 500 updates · batch 64 · `context=8` · `horizon=32` · `gamma=0.997` · `lambda=0.95` · `alpha=0.5` · `beta=0.3`
div: **D016/D017/D018/D020 BY DESIGN** — reduced context/horizon vs Dreamer 4's 192 frames.
div: **D059 OPEN — TOP PRIORITY.** `imagine_trajectory:517` re-slices `[:, -context:]` every step and `rollout._sample_next_jepa` carries no recurrent state, so the imagined rollout is a **sliding 8-state window re-scanned from scratch**. The M arm therefore never accumulates recurrent state over the 32-step horizon: Mamba is run as a fixed-window model, which removes the only axis on which it can differ from attention. Sliding and recurrence are incompatible — carrying state requires a **growing** context, which changes the D017 budget.

### 13. Executed evaluation — `craftax_achievement.py`
method: frozen policy in live Craftax; official geometric-mean Crafter score; paired episode bootstrap
params: 30 seeds (100000-100029) · `max_steps=2500` · `context=8` · sampled at temperature 1 · `policy_seed_base=7000000`
div: **D060 OPEN** — exploratory seeds, not a sealed tier; one training seed and one policy-sampling schedule per condition, so CIs cover environment-seed variance only. **No Craftax T-BASE policy exists**, so "M-JEPA vs T-BASE" is untested.

### 14. Oracle (instrument, not architecture) — `craftax_oracle.py`
method: per-target linear ridge + MLP probes on frozen latents vs constant / timestep / raw-pixel references; episode bootstrap; self-audit
params: 3,802 expert frames / 48 episodes, stride 30 · margin 0.05 · split seed 20260726
div: `preserved` means "as recoverable as from pixels" — a strict ABSOLUTE standard the random init also fails. It is not a necessary condition for control success and must not be used as a binary gate.
div: **D061 OPEN** — `achievement_group` has no constant/timestep/pixel reference, so its AUROCs are unusable.

### 15. Provenance — `source.py`, `checkpoint.py`
method: digest-pinned sources recorded into every checkpoint
div: **D062 OPEN** — `source_report()` emits MMBench2, Mamba-2, and Gymnasium CartPole only. It omits **Craftax** (though `verify_installed_craftax` exists) and every JEPA source (SPR, I-JEPA, LeJEPA).
div: **D063 OPEN** — Craftax BC and imagination reports keep only final metrics; no loss history, RNG, or optimizer state. The world phase does save these.

---

## Not divergences

Dreamer-CDP and DRAMA are **sources we borrow components from, not architectures
we reproduce**. Their stochastic/categorical RSSM latents, online joint
world+control learning, reconstruction/KL terms, and recurrent caches are
deliberately absent. V-JEPA 2-AC is a reference for the AC-JEPA *training
regime* only; our encoder is not modelled on it, so its predictor size and
teacher-forced loss are not a standard we are held to.
