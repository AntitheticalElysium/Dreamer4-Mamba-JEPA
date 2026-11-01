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
