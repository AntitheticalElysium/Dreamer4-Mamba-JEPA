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
  frame` (raw-output MSE, mean over the frame's 1,880 scalars); `sample(model, cond, cache, rng) -> (Percept, cache)`
  (EDM Algorithm 2, order 1, 3 evaluations at sigma_hat, then the context pass that writes the cache; STEP_SPEC 7).
- Reward decoding: `Reward.sample` draws a bin, then emits the bin's train-split mean reward (a buffer computed once
  from the train split and saved with the checkpoint).
- Heads: `TSP.forward(memory, p_window, v, fix) -> (mean, log_scale)` over the glimpse pixels; `ChangeMap.forward(p_t,
  p_t1, memory) -> logits [.., 7, 9]`; `Reward.log_prob(r, cond)`, `Reward.sample(cond)` (mixture of logistics over
  the 256 symlog bins); `Continuation.forward(cond, r) -> logit`.
- Fovea reconstruction: `PixelDecoder` shared by TSP and the fovea (per cell, discretised logistic): TSP feeds it
  (memory, periphery window, v), the reconstruction feeds it the glimpse tokens and v.
- Token scale: `out_map(x) = 3 * tanh(F.layer_norm(x, (x.shape[-1],)) / 3)` per token (I-JEPA / V-JEPA 2-AC, then
  JEDI), shared by the three encoders and every denoised estimate in the sampler. No SIGReg in M0 (COLLAPSE.md).

## saccade/train.py — phases
- `p1(cfg, world_seed)`: the one-step loop (streams, accumulation at 256 steps, frame-indexed lr, harmonised groups,
  stop-gradient routing, checkpoints every 10% of frames, the fixed monitor). `harmonise(losses, log_sigma) -> total`.
- `p2(cfg, world_seed, checkpoint)`, `p3(cfg, world_seed, checkpoint)`: written after S1 / S2, against these interfaces.

## saccade/eval/s1.py — the S1 readings (written before P1 ends)
Token probe on periphery tokens (S0 probe definition, the configured acuity) beside the untrained-encoder copy and the
pixel reference; belief probes (TEST_SPEC S1.2) with their two controls; frozen reader; matched two-sample tests;
reward / continuation calibration on the common bins (`eval.calib.reward_bin`).

## saccade/tests/test_m0.py — checks before any long run
Streams never cross episodes and are deterministic; the periphery encoder is equivariant to whole-cell shifts.
Causal perception (Codex D1): replacing future frames, other episodes or padding leaves a percept unchanged, and
perceiving frame by frame equals perceiving the chunk; every encoded token and every sampled denoised estimate is
bounded by 3 (D2). Mamba: the scan equals step-by-step stepping across chunks; with a nonzero carried state the
continuing segment's local sequence index is 0 and each in-chunk episode start increments it; the convolution history
is zeroed at episode starts; contiguous execution equals chunked execution WITH the 8-frame prefixes, whose memory
tokens are the cached historical ones and which are never replayed through Mamba; frame / action / reward alignment
and the episode-start reset as declared (CURRICULUM §1). Denoiser: the streaming form equals the chunked one; the
whole cached sampler equals a recompute-the-history sampler under identical draws over several generated frames, an
episode reset, branch duplication and checkpoint resume; the sampler matches the EDM Algorithm 2 equations (sigma_max-
scaled start, evaluation at sigma_hat). EDM coefficients match the reference formulas, and the raw-output loss equals
the clean-space loss weighted by 1/c_out^2 on fixed inputs. Reward: a head with all mass on one bin samples exactly 0
for a deterministic 0 and exactly 1.0 for a deterministic 1; empirical bin frequencies of many samples agree with
exp(log_prob). Gradients, tested PER LOSS: the denoising loss reaches the encoder only through the memory tokens,
never through a denoiser frame input or target; TSP, the change map, the fovea reconstruction and the outcome heads
through their declared routes. A resumed run equals an uninterrupted one. P2: gradients reach the denoiser and the
update, not the encoder or the G2 heads; the outcome heads train only by the continuing G3 loss.

## Build order
1. data.py + tests. 2. A9b teacher retrained label-free on train day pairs, frozen, targets precomputed; checked
against A9b's dev precision. 3. model.py + tests. 4. train.p1 + tests; overfit a few episodes; resume test; the
200-update full-model throughput and memory check. 5. eval/s1.py. 6. P1, world seed 1, then seed 2.
