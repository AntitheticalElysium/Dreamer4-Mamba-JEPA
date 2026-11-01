# T2 — joint training: does it erase entity content, and how do anti-collapse and unroll length interact? (PROTOCOL)

Written 2026-10-23 before any T2 code ran. A design test for the system discussion with the user, not a candidate model.

## Why this test (our own failures)
- R63 (10-17): in joint encoder + predictor training, mob content was retained at ~1k updates and destroyed by ~4k in all
  four joint runs; a frozen encoder kept it. We are now committed to joint training, so this must not recur.
- 09-03/04: generated states were not decision-equivalent although latent cosine was 0.98; training the Direct world on
  16-step recursion improved distribution drift at depth but worsened decisions at every depth.
- Reading (READING.md): the decoder-free systems that work align unrolled latents to an encoder target and train heads on
  unrolled latents; they differ in anti-collapse (stop-gradient, EMA, SIGReg) and use short unrolls (2-5). SIGReg's fixed
  geometry was reported to bottleneck a multi-object task (2608.17542). None of this is evidence about our system.

## The system slice under test
Pixels 63x63 -> encoder (7x7 patches, 81 tokens, small transformer) -> per-token latent z (LayerNorm). A recurrent,
action-conditioned latent predictor holds a per-token hidden state h: C = 4 real frames warm it up (teacher forced), then it
unrolls H steps on its own predictions with the recorded actions. From h it reads (a) the predicted next latent and
(b) grounded heads: reward and termination (native signals), trained on every step, teacher-forced and unrolled.
Self-prediction target: the encoder's latent of the real frame at that step. Deterministic predictor (no explicit chance
variable in T2; that is T2b, conditional on these results).

## Factors
- Anti-collapse (5): `sg` stop-gradient target, same encoder; `ema` EMA target encoder (tau 0.995); `sigreg` target with
  gradient + SIGReg (LeJEPA estimator, 256 directions) on encoder outputs, computed over the batch separately at every token
  position (not across positions: that axis rewarded position codes in R56-R68); `vicreg` target with gradient + VICReg
  variance and covariance terms on the same axis; `none` target with gradient, no regularizer (control).
- Unroll H (3): 1, 4, 16; self-prediction and head losses averaged uniformly over all steps (no temporal decay, so H is a
  clean factor).
- Seeds: 2. All else identical: data order, initialization per seed, optimizer, 10k updates, batch 32 sequences.

## Data
TRAIN-split support-v2 episodes for training. Evaluation on DEV-split episodes (never trained on): probes are fitted on half
of the DEV episodes and scored on the other half (episode-disjoint). Simulator labels from exact replay, evaluation only.
FINAL split never touched.

## Measurements (checkpoints 1k, 3k, 6k, 10k)
1. Retention in the encoder (the R63 failure): ridge probes from frozen encoder tokens: per-cell mob presence (zombie, cow,
   skeleton; AP), per-cell terrain class (accuracy), vitals (health, food, drink, energy; R^2 from all tokens).
   References: the same architecture at random initialization; raw 7x7 pixel patches.
2. Collapse: effective rank of tokens per position across frames.
3. Imagined readability by depth k = 1..16 (the 09-03 failure): from DEV contexts, unroll with recorded actions;
   apply the probes fitted on real encoder tokens to the imagined latents; score against the real labels at t+k, next to
   the copy baseline (probes on the context's last real latent) and the probe on the real encoder latent at t+k (ceiling).
   Grounded heads at depth: termination and reward error on the unrolled path.

## Pre-declared readings
- **Erasure occurs** in an arm if mob AP of the encoder at 10k is below its 1k value by more than 0.05 (mean of 2 seeds) and
  below the random-encoder reference.
- **Anti-collapse matters** if the arms' 10k mob AP differ by more than 0.05 with the seeds' ranges not overlapping.
- **Unroll matters** if, within an arm, H changes 10k mob AP or depth-8 imagined mob AP by more than 0.05 with
  non-overlapping seed ranges.
- **Interaction**: the H with the best depth-8 imagined readability differs between anti-collapse arms.
- Anything else is reported as observed. No arm is "selected" by this test; it informs the design discussion.

## Limits
Small models and budget (10k updates); deterministic predictor; no fovea, no memory study, no actor. A negative result for an
arm here is about this slice, not a verdict on the mechanism.

## Amendment 1 (2026-10-23, written after the first three T2 results and before T2c ran): T2c, counter-mechanisms to erasure

**What prompted it.** At H = 16, seed 1, all three finished anti-collapse arms erase entity content from the encoder:
zombie AP 0.61 at initialization -> 0.018 (sg) / 0.047 (ema) / 0.072 (sigreg) at 10k; cow 0.97 -> 0.07 / 0.33 / 0.46.
The arms change the rate, not the outcome. T2 itself continues unchanged.

**Two counter-mechanisms from theory / working systems** (`t2c_joint.py` = `t2_joint.py` + two options, same classes,
data and schedule):
- *Slow encoder* (`--enc-lr-scale 0.02`): Tang et al. 2023's condition (semi-gradient target + faster predictor);
  Dreamer-CDP trains its encoder ~67x slower than its dynamics.
- *Grounded dense target* (`--glimpse 10`): at every predicted frame, the real pixels of one uniformly random 3 x 3-cell
  window are decoded linearly from the predictor's tokens at those cells (MSE x 10). The fovea idea without learned
  attention. `--glimpse-all` (all 81 cells = full pixel reconstruction) is a diagnostic retention reference only.

**Arms (H = 16, seeds 1 and 2):** sg + glimpse; sg + slow; sg + slow + glimpse; ema + glimpse; sigreg + glimpse;
sg + glimpse-all (reference). Evaluation: `t2_eval.py` unchanged.

**Pre-declared readings:**
- A mechanism **prevents erasure** if encoder zombie and cow AP at 10k are within 0.05 of the same arm's random
  initialization or above it, both seeds.
- **Grounding is sufficient** if sg + glimpse prevents erasure; **the slow encoder is sufficient** if sg + slow does.
- **Imagination carries entities** if imagined zombie AP exceeds the copy baseline at depths 4 and 8, both seeds.

## Amendment 2 (2026-10-23, before T2c ran): the SLIM repair as a third counter-mechanism
Strohm et al. (arXiv 2610.03137) report that LeWM drops all movable bodies on a small-object benchmark (3 / 1200 trials)
because they contribute little to the frame change the prediction loss measures, that SIGReg "prevents global collapse
but does not choose which variance survives", and that an encoder inverse-dynamics loss repairs it (0.003 -> 0.348; the
gradient into the encoder carries the effect). In Craftax most entities are NOT controlled by the agent, so whether
inverse dynamics retains them is open. Added arms (H = 16, seeds 1, 2): `sg_idm` (sg + IDM) and `sigreg_idm`
(sigreg + IDM, the LeWM+I analogue). `--idm 1`: a 3-layer MLP predicts a_t from the token-mean of LN(z_t), LN(z_t+1)
and their absolute difference; cross-entropy x 1; gradient into the encoder. Same readings as Amendment 1.
