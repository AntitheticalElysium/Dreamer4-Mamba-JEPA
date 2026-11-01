# Decision 1's build checks (11-01; code 8facf375 + d90330a6; nothing trained)

Decision 1 (learned-fixation TEACHING 10 / 10a, agreed 11-01) is built behind `denoiser.factorised` (absent = P1 world
1's joint denoiser). CPU tests (saccade/tests/test_m0.py, random NONZERO adaLN, final-modulation and output weights):
physical outputs identical under changed glimpse inputs, glimpse level and fixations on every frame; factorised
streaming = chunked; a tied pair = the single-level loss; the cascade's periphery and lighting identical across
fixations under common draws; the choose hook. These GPU checks complete TEACHING 10a's list ("checks before calling it
built", Codex) on the real recipe and real weights. `checks.py` -> `checks.json`, one process at a time on the GPU.

1. **Disabled path, bitwise, against the code before decision 1** (60b98881's model.py and train.py, loaded from saved
   copies): at P1-F's initialization and at its checkpoint 5 (with its Adam moments), on 2 guided streams x 64 steps:
   the initial parameters drawn under the world seed, every loss term, every parameter gradient and the parameters
   after one clipped AdamW step at a nonzero LR are equal.
2. **The real launcher, 25 updates** (`train_guided --updates 25 --no-checkpoints`, faithful, guided): P1-F's logged
   updates 1 and 25 reproduced in every logged value but time and memory (the beta-TSP standard).
3. **Physical invariance on P1-F's checkpoint-10 weights, switched to factorised** (the state dict is unchanged; no new
   parameters): the denoiser's real inputs captured from p1_losses on 2 streams, under training's autocast; the
   physical outputs (periphery, lighting) at every frame are equal when every frame's fixation is replaced by each of
   the 49 legal windows, and when the glimpse inputs are replaced by noise at another level; the glimpse outputs differ.
4. **Cascade on real weights:** M.sample's periphery and lighting equal across all 49 fixations under common draws, the
   glimpse not; the choose hook returns its fixation and the frame drawn at it.
5. **Training path, switched on:** p1_losses at P1-F's initialization: every term finite; the set of parameters
   receiving gradient equal to the joint path's.
6. **Cost:** per-update time and peak memory of a real P1 update (16 streams x 64 steps; 4 x 4 streams x 256 steps),
   joint vs factorised from the same initialization; M.sample per call at 64 streams with an 8-frame context, joint vs
   cascade (stage 2 recomputes the physical tokens at each of its 3 steps; caching their keys / values is deferred
   unless the cost matters).

Not a reading of any trained factorised world: none exists. Known gap (8facf375): the joint evaluation samplers
(saccade/eval/s1.one_step, Codex's sampler factorial) would read a factorised world's tied joint mode; P1-F2's readings
plan routes them through the cascade.

## Results (11-01 evening; `checks.json`, MAMBA_DETERMINISTIC=1 unless keyed `_autotuned`)
1. **Disabled path: bitwise.** Initial parameters under the world seed equal; at P1-F's initialization and its
   checkpoint 5 every term, every gradient and the parameters after one AdamW step equal (max abs difference 0).
2. **Launcher: within mamba_ssm's own run-to-run noise, not bitwise.** Deterministic mode: update 1 exact, update 25
   within 1.3e-6 relative (grad norm 6.431418 vs 6.431410). Default (autotuned) mode, as P1-F trained: update 1
   already differs at ~1e-6 (grad norm 6.528726 vs 6.528735), update 25 within 2.2e-6. mamba_ssm autotunes its
   Triton block sizes per process, and deterministic mode also switches its backward to a deterministic reduction
   (ssd_combined `DETERMINISTIC_REDUCTION`), so neither mode re-creates P1-F's process; the beta-TSP launch check's
   exact match (11-01 afternoon) was the same picks by chance. Item 1, in one process, is the decisive check. Timing
   as P1-F's (1.36 s / update; peak 4.0 GiB). The default-mode re-run first ran out of memory beside the checks'
   own CUDA context (fixed: the parent no longer initialises CUDA).
3. **Physical invariance on P1-F's checkpoint-10 weights, switched on:** periphery and lighting outputs equal under
   all 49 fixations (every frame of 2 streams x 72 frames, 46 distinct real fixations) and under replaced glimpse
   inputs and level; the glimpse outputs differ in every case; output magnitudes 0.63-0.80.
4. **Cascade:** periphery and lighting equal across all 49 fixations under common draws, the glimpse distinct; the
   choose hook returns its fixation and exactly the frame drawn at it.
5. **Training path switched on:** every term finite; the same parameters receive gradient as on the joint path.
6. **Cost:** a real P1 update 1.32 -> 1.60 s at 64 steps (+21%), 4.97 -> 5.88 s at 256 (+18%); peak 4.00 -> 4.10
   GiB. Sampling at 64 streams: 90 -> 198 ms per frame (x2.2; stage 2 recomputes the physical tokens).
For any later bitwise re-run across processes: MAMBA_DETERMINISTIC=1, and compare old against new code in one process.
7. **Stage-2 cache** (added 11-01 night, user's request; code bd: `glimpse_step`, the step after 09488790's checks):
   against the code before it (a4f2c2a2), the factorised training step is bitwise (terms, gradients, one AdamW step at
   P1-F's initialization), the joint path still bitwise against 60b98881 (step 1 re-run), and on P1-F's checkpoint-10
   weights switched on, 8-frame context caches equal and sampled periphery / lighting bitwise under common draws; the
   sampled glimpse within 0.006-0.010 max abs (std 0.46) under bf16 autocast, 7e-7 in float32 (diagnostic): the
   cached path is exact, the difference is bf16 rounding over different matmul shapes. Cost per sampled frame at 64
   streams: joint 87 ms, cascade 192 ms -> 139 ms (x2.2 -> x1.6 the joint sampler).
