# M0 code plan (2026-10-27): files, signatures, checks

Small by intent: four modules, one config, one test file. Numbers come from `saccade/m0.yaml`; reasons from the specs.
Shapes: B streams, T frames per chunk (+ P = 8 context-only prefix frames), N = 81 periphery cells, G = 9 glimpse cells,
K = 4 memory tokens, W = 256.

## saccade/data.py — the corpus as training streams
- `load_split(cfg) -> Split`: train / monitor (fixed 5% of train episodes) / dev episode index from the support-v2
  manifest, with the collector's exogenous flags regenerated per step (as `eval/labels.py`).
- `class Streams`: `__init__(split, cfg, seed)`, `next(steps: int, n: int) -> Chunk`. B episode streams, each played in
  order; a chunk carries the previous P frames of its stream as context only; a finished episode is replaced by the
  next one in the stream's shuffled list. `Chunk`: frames uint8 [n, P+T, 63, 63, 3], actions, rewards, terminated,
  valid (padding) and episode-start masks, fixations [n, P+T, 2], stream ids.
- `fixation_prior(rng, shape) -> int [.., 2]`: window centres uniform over the 49 positions of a 3x3 window on the 9x9
  screen (HUD included).
- `change_targets(split) -> memmap float16 [frames, 7, 9]`: soft change probabilities from the frozen A9b teacher
  (map rows), precomputed once (build step 2).
- `exogenous_windows(split, h) -> [(episode, t0)]`: windows of h consecutive exogenous actions (P2, S2).

## saccade/model.py — perception, belief, denoiser, heads
- `class Perception`: `forward(frames, fixations) -> Percept(p [.., N, 16], g [.., G, 64], v [.., 8], fix)`. Lighting
  latent from pooled frame statistics; periphery = shared MLP on each cell downsampled to the configured acuity, FiLM
  by v, tanh clamp; fovea = small conv on the full-resolution 3x3-cell crop, one token per cell, tanh clamp.
- `class Belief` (Mamba-2, ported from `d4mj/mamba_recurrence.py`, its tests kept): `scan(x [B, 2L, W], carry,
  seq_idx) -> (y, carry)`; `percept_token(Percept, reward, terminated) -> [.., W]` (flattened percept, linear);
  `action_token(a) -> [.., W]`; `memory(y_predicted) -> [.., K, W]`. Interleaved order: percept_t, action_t.
- `class Denoiser`: `forward(x_noisy: Percept, sigma [B, T], action [B, T], memory [B, T, K, W], fix, mask, cache=None)
  -> Percept` (x-prediction). Tokens per frame: N periphery + G glimpse (at their fixation cells) + 1 lighting + K
  memory; spatial attention every layer, temporal attention every 4th layer over the past 4 frames (block-causal,
  per token slot); RoPE + absolute 2D positions; adaLN-Zero on (sigma, action). `step(...)` = streaming form with
  per-layer caches.
- EDM: `precondition(sigma) -> (c_skip, c_out, c_in, c_noise)`; `denoise_loss(model, x0, sigma, cond) -> loss per
  element`; `sample(model, cond, rng) -> Percept` (Euler, 3 steps, churn).
- Heads: `TSP.forward(memory, p_window, v, fix) -> (mean, log_scale)` over the glimpse pixels; `ChangeMap.forward(p_t,
  p_t1, memory) -> logits [.., 7, 9]`; `Reward.log_prob(r, cond)`, `Reward.sample(cond)` (mixture of logistics over
  the 256 symlog bins); `Continuation.forward(cond, r) -> logit`.
- `sigreg(tokens) -> statistic` (port of LeWM's module; samples 4,096 tokens).

## saccade/train.py — phases
- `p1(cfg, world_seed)`: the one-step loop (streams, accumulation at 256 steps, frame-indexed lr, harmonised groups,
  SIGReg, checkpoints every 10% of frames, the fixed monitor). `harmonise(losses, log_sigma) -> total`.
- `p2(cfg, world_seed, checkpoint)`, `p3(cfg, world_seed, checkpoint)`: written after S1 / S2, against these interfaces.

## saccade/eval/s1.py — the S1 readings (written before P1 ends)
Token probe on periphery tokens (S0 probe definition, the configured acuity) beside the untrained-encoder copy and the
pixel reference; belief probes (TEST_SPEC S1.2) with their two controls; frozen reader; matched two-sample tests;
reward / continuation calibration on the common bins (`eval.calib.reward_bin`).

## saccade/tests/test_m0.py — checks before any long run
Streams never cross episodes and are deterministic; the periphery encoder is equivariant to whole-cell shifts; the
Mamba scan equals step-by-step stepping across chunks and resets; the streaming denoiser equals the chunked one; EDM
coefficients match the reference formulas; the reward head can put near-all mass on one bin; gradients reach the
encoder only through the declared routes (targets stop-gradient); a resumed run equals an uninterrupted one;
P2 gradients reach the denoiser and the update, not the encoder or outcome heads.

## Build order
1. data.py + tests. 2. A9b teacher retrained label-free on train day pairs, frozen, targets precomputed; checked
against A9b's dev precision. 3. model.py + tests. 4. train.p1 + tests; overfit a few episodes; resume test; the
200-update full-model throughput and memory check. 5. eval/s1.py. 6. P1, world seed 1, then seed 2.
