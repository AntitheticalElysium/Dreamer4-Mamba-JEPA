# M0 build checks (2026-10-27): build order steps 1-4 of saccade/design/M0_CODE.md

Implementation checks, not experiments. P1 is NOT launched (Codex reviews first).

## Step 2: change-map targets (saccade/data.py, `python -m saccade.data targets artifacts/saccade_m0/targets`)
- The A9b corrupted-day detector refit exactly as `eda_change_night_v2.fit` (same cells, episode split, Otsu threshold,
  LightGBM settings) reproduces A9b's saved held-out scores to every printed digit (`targets/change_teacher.json`):
  day AP 0.943, precision 0.848, recall 0.917; night AP 0.612, precision 0.630, recall 0.510; tau 6.1005.
- Vectorised features equal A9b's per-cell loop exactly (max difference 0.0, one episode, 5,154 cells).
- `targets/change_targets.npy`: 2,439,791 rows (all train frames, monitor episodes excluded), ~58.6 defined map cells
  per frame (player cell and entering cells undefined), mean probability 0.022 (A9b change base rate 1.7-2.9%), every
  episode's first frame undefined.

## Step 3-4: tests (`.venv/bin/python -m pytest saccade/tests -q`): 17 passed (6 S0 + 11 M0)
streams follow episodes, are deterministic, prefix = previous chunk's tail, reward / termination / action alignment
against the store; perception causal, frame-local, bounded by 3; periphery equivariant to whole-cell shifts; Mamba:
chunked = contiguous = step-by-step, an in-chunk episode start = carried scan before and fresh scan after (carry too),
a chunk starting with a reset drops the carried state, Triton = pure-torch reference with resets; P1 bookkeeping
(cached predicted beliefs for the 8-frame prefix) equals one contiguous scan in fp32 (atol 2e-3) and Mamba scans only
the chunk's own 2T tokens; streaming denoiser = chunked (atol 1e-4, with an episode reset); the whole cached sampler
(3 churned Euler evaluations at sigma_hat + one context pass per frame) = a recompute-the-history sampler under
identical draws (6 frames, a reset, branch duplication; atol 1e-4); EDM coefficients, raw-output loss = clean loss
weighted 1 / c_out^2, sampler = EDM Algorithm 2 equations; reward: deterministic 0 and 1 sample exactly, sampled bin
frequencies = exp(log P) (200k draws, < 0.005); gradients per loss: denoising reaches the encoder and Mamba ONLY
through the memory tokens (zero with them detached), TSP / change / reward / continuation through their declared
routes, the reconstruction through the fovea without Mamba, none into the denoiser; resumed run = uninterrupted.

## Step 4: overfit 4 episodes, 400 updates (`checks/overfit/log.jsonl`)
First version: TSP and reconstruction stalled (5.58 -> 5.46 nats / pixel) while the gradient norm grew to 750-2,750.
Per-term gradient norms: the reward term 300-800 on its head (25-50 on Mamba), every other term < 1. Control without
the reward term: TSP 4.56, reconstruction 4.54 and G1 0.17 at update 200 (vs 5.50 / 5.49 / 0.59), norm ~10.
Mechanism: a discretised logistic mixture's tail gradients scale as 1 / scale; as components sharpened, global clipping
shrank every update ~1,000-fold and Adam's second moment (beta2 0.999) kept the earlier, larger gradients.
Fix (LOSSES G3): a categorical over the same 256 symlog bins (DIAMOND / JEDI / IRIS predict reward categorically).
Rerun: every term falls, norm 8-10:
| update | G1 | change | TSP | recon | reward | cont |
|---|---|---|---|---|---|---|
| 1 | 0.945 | 0.686 | 5.582 | 5.582 | 5.608 | 0.678 |
| 100 | 0.385 | 0.095 | 5.231 | 5.206 | 0.294 | 0.014 |
| 400 | 0.140 | 0.083 | 4.051 | 3.993 | 0.118 | 0.011 |

## Step 4: throughput and memory (full model, every loss, real streams, optimizer state)
- 16 streams x 64 steps: 1.35 s / update, peak 4.0 GiB -> 12,500 updates = 4.7 h.
- 4 x 4 streams x 256 steps (accumulated): 5.05 s / update, peak 3.8 GiB -> 3,125 updates = 4.4 h.
- P1 ~ 9.1 h per world seed plus checkpoints / monitor; data loading 0.02-0.2 s per chunk (negligible).

## Pinned outputs (sha256)
- artifacts/saccade_m0/targets/change_teacher.txt: `69862f6b98e3ba770395d07bdb516c2cc0e326917c0f848c4a8aedadd109406f`
- artifacts/saccade_m0/targets/change_targets.npy: `f1a79ed7dba730db02b5efb537cc8f44fd580406c2da7894005b4e3e84c87b97`

## Step 5: S1 readings (saccade/eval/s1.py), smoke test only
Run end to end on a 20-update checkpoint (15 dev / 30 train-400 episodes; numbers meaningless by design): token probes
(trained and untrained copy, 2 seeds), the S3.2 perception blocker, belief probes with both controls, one-step sampling
from real histories, the two-sample classifier with its power on corrupted real tuples (shuffled history 0.77, frozen
time 0.85, mismatched actions 0.69; the 20-update model's samples 1.00), reader counts, reward / continuation
calibration, TSP beside copy-previous and position-mean baselines. Definition check: "unlock" equals reward > 0.45 on
99.98% of dev steps (the reward event used for the model's unlock probability).
Needs lightgbm on PYTHONPATH (two-sample classifier), as the S0 harness did.
