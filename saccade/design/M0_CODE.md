# M0 code plan (2026-10-27): files, signatures, checks

Small by intent: data, model, training and evaluation modules, one config and contract tests. Numbers come from `saccade/m0.yaml`; reasons from the specs.
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
  p_t1, memory) -> logits [.., 7, 9]`; `Reward.log_prob(r, cond)`, `Reward.sample(cond)` (categorical over the 256
  symlog bins; revised at the overfit check, LOSSES G3); `Continuation.forward(cond, r) -> logit`.
- Fovea reconstruction: `PixelDecoder` shared by TSP and the fovea (per cell, discretised logistic): TSP feeds it
  (memory, periphery window, v), the reconstruction feeds it the glimpse tokens and v.
- Token scale: `out_map(x) = 3 * tanh(F.layer_norm(x, (x.shape[-1],)) / 3)` per token (I-JEPA / V-JEPA 2-AC, then
  JEDI), shared by the three encoders. `clean_estimate` applies EDM preconditioning plus a range clamp to [-3, 3]
  in both the deployed sampler and S1. It preserves every valid target; reapplying `out_map` did not. No SIGReg in M0
  (COLLAPSE.md).

## saccade/train.py — phases
- `p1(cfg, world_seed)`: the one-step loop (streams, accumulation at 256 steps, frame-indexed lr, harmonised groups,
  stop-gradient routing, checkpoints every 10% of frames, the fixed monitor). `harmonise(losses, log_sigma) -> total`.
  Checkpoints bind configuration, world seed, source/config hashes, manifest and target hashes, and torch version;
  resume rejects differences before loading state. Each checkpoint logs raw per-loss encoder gradient norms on
  one stream capped at 64 new frames (separate recomputation after training backward, before the optimizer step),
  plus raw projection RMS/channel standard deviation on the fixed monitor. Diagnostics preserve RNG,
  gradients and training mode; nuisance drift remains a separate measurement.
- `p2(cfg, world_seed, checkpoint)`, `p3(cfg, world_seed, checkpoint)`: written after S1 / S2, against these interfaces.

## saccade/eval/s1.py — the S1 readings (written before P1 ends)
Token probe on periphery tokens (S0 probe definition, the configured acuity) beside the untrained-encoder copy and the
pixel reference; belief probes (TEST_SPEC S1.2) with their two controls; frozen reader; matched two-sample tests;
reward / continuation calibration on the common bins (`eval.calib.reward_bin`). Threshold-grouped AP handles ties.
The memory target is visible kind-level history and last-seen side (TEST_SPEC), independent of hidden lifecycle.
C2ST includes the full eight-frame conditioning window and predicted beliefs, current action and fixation, with
identical FP16 precision for both outcomes and physical-episode train/validation/test splits. C2ST reuses the existing
MLP probe and fit schedule with train-only standardisation and equal energy per input group; a matched-marginal
dependence control prevents a coordinate-wise classifier from silently missing joint changes. Unlock probability sums reward bins matching
the health-adjusted unlock event. The reader is saved with schema, seed, label identity and encoder-checkpoint hash;
S2 reloads the exact weights. A missing blocker metric returns unmeasured, never passed.

`eval/after_p1.py`: the one-shot completion handler (user requested after launch). Native systemd OnSuccess provides
the training exit metadata; the handler requires the bound successful invocation and complete, comparable monitor
readings, applies CURRICULUM's checkpoint rule, verifies source/data identity and execs full S1. It records the
selection and refuses an automatic repeated attempt. Its CPU tests cover checkpoint selection, incomplete runs,
wrong/failed/stopped invocations, full-S1 command and exactly-once guard. Live throwaway service tests verify attaching
the hook to a running unit without restart, success triggering and failure suppression (checks/auto_s1).

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
episode reset, branch duplication and checkpoint resume (15 frames, including a segment longer than 8); an exact
raw denoiser with zero loss returns the encoder target; the sampler matches the EDM Algorithm 2 equations (sigma_max-
scaled start, evaluation at sigma_hat). EDM coefficients match the reference formulas, and the raw-output loss equals
the clean-space loss weighted by 1/c_out^2 on fixed inputs. Reward: a head with all mass on one bin samples exactly 0
for a deterministic 0 and exactly 1.0 for a deterministic 1; empirical bin frequencies of many samples agree with
exp(log_prob). Gradients, tested PER LOSS: the denoising loss reaches the encoder only through the memory tokens,
never through a denoiser frame input or target; TSP, the change map, the fovea reconstruction and the outcome heads
through their declared routes. A resumed run equals an uninterrupted one; recipe, seed or source mismatches reject.
S1 regression checks cover tied AP, history sensitivity without target leakage, precision parity, null C2ST, observable
absence/side targets, the unlock event, reader roundtrip and unmeasured blockers. P2 checks are DEFERRED until P2 is
implemented: gradients must reach the denoiser and update, not the encoder or G2 heads; outcome heads train only by G3.

## Build order
1. data.py + tests. 2. A9b teacher retrained label-free on train day pairs, frozen, targets precomputed; checked
against A9b's dev precision. 3. model.py + tests. 4. train.p1 + tests; overfit a few episodes; resume test; the
200-update full-model throughput and memory check. 5. eval/s1.py. 6. P1, world seed 1, then seed 2.
Status 10-27: steps 1-5 built. The implementation audit and correction checks are linked from
artifacts/saccade_m0/checks/CHECKS.md. Production P1 remains unstarted during this correction turn. `exogenous_windows` (P2, S2) is written with P2.
