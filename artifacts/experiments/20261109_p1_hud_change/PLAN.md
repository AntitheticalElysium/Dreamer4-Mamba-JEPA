# P1-HUD-Change: Full-Frame Change Grounding (9x9x2) for Saccade World Model

Date: 2026-11-09  
Objective: Unmask the HUD (rows 7-8) in periphery encoder grounding by training Saccade with full-frame change supervision over all 81 screen cells, supervising both same-position identity and dominant-translation correspondences.

## 1. Context and Motivation

In P1-F4, change map supervision was restricted to the 7x9 map cells (`map_only`), excluding the bottom two (HUD) rows and the player's cell. What was measured on F4 (11-07/08):
- The periphery HUD tokens themselves encode every digit exactly (reader accuracy 1.00), but the belief's input embedding
  (`percept_embed`) keeps the exact food and energy digits only 42% of the time (drink 52%, health 92%), and the
  post-action memory source less (food 28%) (`20261107_f4_followups/RESULTS.md` 14).
- HUD digits are unchanged on about 96% of transitions, so the denoiser can predict them by copying from its cache of
  past frames; F4's health output under-draws even hits whose cues it holds (RESULTS 12, 16).
- On Episode 376 (the natural thirst alias: identical 8-frame observations at $t=654$ and $t=658$ with true next drinks 5 vs 4), F4 failed to track the timer phase, predicting decrements at 2/128 for both.

This experiment tests whether full-frame change map grounding across all screen cells ($9\times 9\times 2$) provides sufficient perceptual signal to allow Mamba to maintain hidden timer counters without the denoiser shortcut dominating.

## 2. What Changes, Exactly

### Model Architecture
- `saccade/model.py: ChangeMap`:
  - Position embeddings: $81 \times 32$ (covering all $9 \times 9$ cells, expanded from $63 \times 32$).
  - Net output: 2 channels (`same_position` and `dominant_translation`), expanded from 1 channel.
  - Parameter delta: $+833$ parameters (from 165,569 to 166,402).
- Everything else is strictly identical to P1-F4:
  - Perception: coarse periphery MLP + glimpse conv + lighting token.
  - Recurrent Belief: Mamba-2 ($W=256$, interleaved $[p, a]$ tokens).
  - Denoiser: Dreamer 4 shortcut forcing (k_max 8, bootstrap fraction 0.25, bootstrap start 5000, ramp [0.9, 0.1], context signal 7).
  - Attention: Dense temporal attention in long-context layers (blocks 4 and 8, window 4).
  - Fixation Chooser: Original coarse-only learned controller (101,584 parameters).
  - Training Budget: 25.6M frames (12.8M at 64 steps, 12.8M at 256 steps), world seed 1.

### Target Supervision
- Precomputed target store: `artifacts/saccade_full_frame_change/targets/change_targets.npy`
  - Shape: `(2439791, 9, 9, 2)`, dtype: `float16`, size: 754 MB.
  - Verified across 7,672 training episodes against `artifacts/craftax_support_v2/manifest.json`.
  - Format: `CHANGE_FORMAT = 'full_frame_a9b_v2'`.

### Control Baseline
- Control checkpoint: P1-F4 (`artifacts/experiments/20261106_p1f4/run/ckpt_10.pt`)
  - SHA-256: `1b70edcc432daf3fb337c37c67af8d709d3728e25d02ac94523f8a7322ba1379`
  - Training log: `artifacts/experiments/20261106_p1f4/training.log`

## 3. Preflight Protocol (`checks.py` -> `checks.json`)

1. **Learned Resume Determinism across Bootstrap Start:**
   - 4-update straight run vs 2+2 resumed run across update 2 (bootstrap warm-up boundary).
   - Strict bitwise equality: `streams`, `carry`, `cache`, `state`, `rng`, `fixation_controller`, `gaze_bytes`, `logged_terms`.
   - Float tolerances: optimizer state $< 1\times 10^{-11}$, model parameters $< 1\times 10^{-11}$.
2. **Update 1 Alignment vs F4 Control:**
   - Verify bitwise identical `g1` ($0.32771176$), `tsp`, `rec`, `reward`, `cont`.
   - Verify expected divergence on `change` due to full-frame targets ($9\times 9\times 2$ vs $7\times 9$).
3. **Launcher Process Test:**
   - Subprocess execution of `train_hud_change.py` for 2 updates with `--no-checkpoints`.
4. **Sustained Benchmarks & Memory Profiling:**
   - `sustained64` (200 updates @ 64 steps): throughput and monitor loss.
   - `sustained64_bootstrap` (100 updates @ 64 steps, bootstrap on).
   - `phase256` (3 updates @ 256 steps).
   - `phase256_bootstrap` (3 updates @ 256 steps, bootstrap on).
   - Verification that peak memory $\le 2.6$ GiB on RTX 3060 (6 GiB card).
   - Projected training time: $\sim 14.5$ hours.

## 4. Run Execution

- Worker: `run.py` (F2.1/F4 one-shot worker pattern, validating source pins and `checks.json`).
- Launch command:
  ```bash
  systemd-run --user --unit=saccade-p1-hud-change \
    -p MemoryMax=24G \
    -E MAMBA_DETERMINISTIC=1 \
    -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --working-directory=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA \
    .venv/bin/python -u -m artifacts.experiments.20261109_p1_hud_change.run
  ```

## 5. Morning Readout Plan (`readout.py`)

1. **S1 Evaluation:**
   - Belief-path probing (`belief_path_probe.py`): exact HUD digits in the belief's input embedding, the corrected belief
     and the post-action belief, against F4's 42% (food) / 42% (energy) / 52% (drink) / 92% (health); hidden counters.
   - Belief probing (latent state decodability).
2. **Consequences:**
   - Determinant curves and crafting evaluation at frame level.
3. **Actor Interface:**
   - Random policy rollouts (40 episodes).
4. **Recursion:**
   - 4-step own-output recursion.
5. **Decisive Timer Diagnostic:**
   - Natural thirst alias evaluation on Episode 376 ($t=654$ vs $t=658$).
   - Deterministic contrast: if the generated decrement rates diverge (truth 0 vs 1), the corrected grounding let the
     world use the older reset. If both stay near F4's 2/128, the grounding alone does not teach the timer; the cause
     (memory content, the denoiser's use of it, or its cache) would still have to be isolated.
