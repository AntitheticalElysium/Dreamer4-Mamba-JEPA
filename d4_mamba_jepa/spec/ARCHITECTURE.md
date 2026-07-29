# Architecture

What the system IS, right now. Present tense, overwritten in place, no history.
One block per component in dataflow order. Shapes are for the live
`craftax_jepa_config()`.

- `diff` — departure from the pinned source. Omitted means faithful.
- `inherit` — where each knob's VALUE came from. Omitted means `craftax`.
  - `craftax` chosen or measured on Craftax
  - `cartpole` selected on CartPole (2 actions, 500-step cap) and carried over unexamined
  - `default` `D4LiteConfig` default; never selected on any task
  - `source` fixed by a pinned paper or repo

`cartpole` and `default` are the untested surface: every such value was set
against a task with two actions and three informative scalars.

Rule: if a line would need a date, a result, or the word "corrected", it belongs
in `ABLATIONS.md`, not here.

---

### 1. Environment — `craftax_env.py`
out    `[3,64,64]` uint8 CHW
knobs  Craftax-Classic pixels · native 63x63 (9x9 tiles at 7px) zero-padded to 64 · 17 actions · 22 achievements
src    `craftax` 1.6.1; `game_logic.py`/`constants.py`/`renderer.py` digest-pinned in `source.py`
diff   `continues` uses Crafter's `1 - dead`: timeout truncation keeps `continues=1`

### 2. Replay — `expert/generate.py`, `data.py`
out    episode-bounded `EpisodeReplay`
knobs  320 episodes / 696,746 transitions · PPO expert · 80/10/10 whole-episode split, seed 20260727
src    `Craftax_Baselines@7ce36fa` `ppo_rnn.py`
diff   episodes capped at `max_steps=2500`

### 3. Sampler — `common.py:sample_sequences`
out    `SequenceBatch`: `[B,T,3,64,64]` + led-to actions/rewards/continues/valid
knobs  `sequence_length=16` · `terminal_fraction=0.5` · batch 8
inherit  code CartPole-era but env-agnostic · `sequence_length` default (CartPole ran 12) · `terminal_fraction` cartpole · batch cartpole
diff   `round(B*fraction)` rows are forced to the LAST window of a terminal episode; the remainder is episode-uniform, not transition-uniform

### 4. Encoder — `model.py:build_tokenizer`, called at `model.py:243`
in     `[B,16,3,64,64]` uint8 -> `temporal_patchify(8)` -> `[B,16,64,192]`
out    bottleneck `[B,16,16,16]`, tanh
knobs  patch 8 (64 patches) · d_model 64 · depth 4 · heads 4 · time_every 2 · n_latents 16 · d_bottleneck 16 · **MAE OFF**
src    MMBench2 `Encoder`, unmodified
inherit  every knob `default` — no encoder geometry has ever been selected for Craftax
diff   `training_mask=False` forces `mae_p_min=mae_p_max=0`, so `MAEReplacer` takes its no-op path; the config's `mae_p_max=0.9` is unused by the world
diff   trained from random init at full LR unless `encoder_learning_rate` is passed; Dreamer 4 and V-JEPA 2-AC freeze a pretrained encoder, Dreamer-CDP uses `enc_lr 6e-6` vs `dyn_lr 4e-4`
diff   `n_latents=16` vs Dreamer 4's 512 latent tokens; `d_bottleneck=16` matches its `D_b=16`

### 5. Packing — `model.py:encode_frames`
in     `[B,16,16,16]`
out    `[B,16,4,64]` (`n_spatial=4` x `d_spatial=64`) = 256 floats/frame
knobs  `packing_factor=4`
src    MMBench2 `pack_bottleneck_to_spatial`, unmodified
inherit  default

### 6. Dynamics — `model.py:forward_dynamics` (MMBench2 `Dynamics`)
in     packed `[B,T,4,64]` + led-to actions
out    tokens, agent tokens `[B,T,2,64]`
knobs  d_model 64 · depth 4 · heads 4 · time_every 2 (2 temporal modules) · n_register 2 · n_agent 2 · k_max 4
src    MMBench2, unmodified classes
inherit  every knob `default`
diff   17-way discrete action embedding (`DiscreteActionEncoder`) replaces the continuous 16-D action MLP, in the same token slot

### 7. Temporal operator — `temporal.py:replace_dynamics_time_attention`
knobs  T arm: upstream `TimeSelfAttention` · M arm: official `Mamba2`, `d_state=64 · headdim=64 · expand=1 · d_conv=4`
src    `state-spaces/mamba`, digest-pinned
inherit  `d_state`/`headdim` cartpole (selected there over a parameter-matched 16/32) · `expand`/`d_conv` default
diff   the single research axis; swap runs LAST in `D4LiteWorld.__init__` (`model.py:317`, after `_build_jepa` at `:310`) so both arms draw identical shared init
diff   `d_inner = d_model*expand = 64` with `headdim=64` gives the M arm exactly one head
diff   arms are not parameter-matched; the temporal modules differ in size

### 8. Predictor — `model.py:CDPPredictor`
in     mean over 2 agent tokens (64) + next action token (64)
out    `[B,1,4,64]` — **this is the entire rollout**; no denoiser, decoder, or MAE
knobs  `jepa_predictor_context="pooled_agent"` · `hidden_ratio=1.0`
src    shaped after Dreamer-CDP's predictor
inherit  default
diff   Dreamer-CDP predicts from an 8192-d deterministic state (`configs.yaml:93`) with a per-token cosine; ours is a pooled 64-d channel

### 9. Self-prediction loss — `objectives.py:jepa_self_prediction_loss`
in     online: predictor rolled autoregressively K steps · target: EMA target encoder on the real future frame
out    scalar
knobs  `jepa_jumps=5` · `jepa_projection_dim=64` · EMA tau 0.99 -> 0.999 linear over world steps · `jepa_anticollapse="ema"`
src    `mila-iqia/spr` `global_spr_loss` (`F.normalize(dim=-1)` then MSE); ramp FORM from I-JEPA `train.py:228`
inherit  `jepa_jumps` cartpole (ladder 5/8/11 run there) · `projection_dim` default · EMA endpoints default
diff   scores ONE global vector per frame: `[B,K,4,64]` is flattened to `[B,K,256]`, projected to 64, normalized once. SPR's per-position branch (`local_spr_loss`) is shipped but off by default (`run.py:106`); I-JEPA and V-JEPA 2 normalize and score per token, and have no global variant
diff   EMA ramp endpoints are local (I-JEPA publishes `[0.996, 1.0]`)
diff   SPR's visual augmentation is not ported
diff   SPR runs self-prediction as an auxiliary beside Q-learning and reward (`algos.py:131`); here it is the primary encoder signal
diff   `sigreg` alternative exists (LeJEPA, digest-pinned) and is not the default

### 10. Loss composition — `training.py:_jepa_world_loss`
out    `jepa_weight*jepa + reward*reward_n + continuation*continuation_n`
knobs  all weights 1.0 · AdamW lr 1e-4, wd 1e-2 · clip 1.0 · warmup 1,000 · 20,000 updates · batch 8 · optional `encoder_learning_rate` splits the encoder into its own param group
inherit  every optimizer knob and the whole budget cartpole
diff   `WorldLossNormalizer` registers EmaRms for flow/reward/continuation/cdp/reconstruction — there is no `"jepa"` term, so the JEPA loss is unnormalized while the other two are

### 11. Task heads — `model.py:forward_task_heads`
in     post-transition agent tokens of the IMAGINED rollout (`return_rollout_agents`)
out    reward logits + centers, continue logits
knobs  `reward_horizon=8` · `reward_bins=255` · symlog `[-10,10]` · `continuation_horizon=8` · `jepa_terminal_weight=8`
src    MMBench2 MTP heads; `bins=255` from DreamerV3 `configs.yaml:101`
inherit  horizons and bins default/source · symlog range default · `terminal_weight` cartpole
diff   heads receive `jepa_jumps` positions per batch, not `sequence_length`, and only positions >= context get gradient

### 12. BC policy — `common.py:BCPolicy`
in     agent tokens (world frozen)
out    17-way categorical
knobs  3,000 updates · batch 16 · lr 1e-4
src    MMBench2 `PolicyHeadMTP` — attention pooling, MLP shape, small output init, head-only gradient
inherit  code CartPole-era but env-agnostic · every budget cartpole
diff   the Dreamer 4 reproduction optimizes `p["dyn"]` during BC; we freeze the whole world

### 13. Imagination — `imagination_actor_critic.py`
in     replay contexts, frozen world/heads/BC prior
out    actor (`BCPolicy`) + `ValueHead`
knobs  500 updates · batch 64 · `context=8` · `horizon=32` · gamma 0.997 · lambda 0.95 · alpha 0.5 · beta 0.3
src    Dreamer 4 eqs 10-11: actor init from BC, PMPO + reverse `KL(actor||BC)`, TD-lambda symexp-twohot value, one rollout per context
inherit  gamma/lambda/alpha/beta source · updates, batch, context, horizon all cartpole (the 500 was selected on CartPole dev seeds)
diff   `imagine_trajectory` (`:512`) re-slices `[:, -context:]` each step and `_sample_next_jepa` carries no recurrent cache, so the rollout is a sliding 8-state window re-scanned from scratch; the M arm never accumulates state across the 32-step horizon
diff   reduced vs Dreamer 4's 192-frame context; no task tokens, no `tau_ctx` corruption

### 14. Executed evaluation — `craftax_achievement.py`
knobs  frozen policy in live Craftax · sampled at temperature 1 · official Crafter geometric-mean score · paired seed bootstrap
diff   seeds are exploratory, not a sealed tier; one training seed and one policy-sampling schedule per condition, so CIs cover environment-seed variance only

### 15. Oracle — `craftax_oracle.py` (instrument, not architecture)
knobs  per-target ridge + MLP probes on frozen latents vs constant / timestep / raw-pixel references · episode-level splits and bootstrap · self-audit on perfect/constant/misaligned inputs
diff   `preserved` means "as recoverable as from raw pixels" — an absolute standard, not a necessary condition for control. Not a gate
diff   `achievement_group` has no constant/timestep/pixel reference

### 16. Provenance — `source.py`, `checkpoint.py`
knobs  digest-pinned sources recorded into every checkpoint; strict atomic saves with full RNG
diff   `source_report()` emits MMBench2, Mamba-2 and Gymnasium CartPole only — it omits Craftax (though `verify_installed_craftax` exists) and every JEPA source
diff   `POLICY_FORMAT`/`FORMAT` still carry `cartpole` in their strings; they are serialized identifiers compared on load, so renaming them would invalidate existing checkpoints

---

## Inherited-value summary

Nothing in components 4-13 was selected on Craftax. Encoder geometry, packing,
dynamics width and the predictor are `D4LiteConfig` defaults never chosen for
any task; the sampler, optimizer, budgets, `jepa_jumps`, `terminal_weight`,
Mamba state expansion and the whole actor/value schedule were selected on
CartPole. Craftax-native choices exist only at the boundaries: the environment,
the replay, the executed evaluation and the oracle.

## Transition convention

`(obs_t, action_t) -> (obs_{t+1}, reward_{t+1}, continue_{t+1})`. In the
block-causal sequence the action stored at state position `t+1` is `action_t`,
the action that led to that state. Position 0 gets the start action. Head
distance 0 predicts the transition that led to the current position.

## Not reproductions

Dreamer-CDP, DRAMA and NE-Dreamer are sources we borrow components from, not
architectures we reproduce. Their RSSM latents, online joint world+control
learning, reconstruction/KL terms and recurrent caches are deliberately absent.
