# P1.5 (2026-11-16 night; named by the user: far enough along to warrant P1.5 on the way to P2)

From scratch, F6.1's recipe and schedule (25.6M frames, world seed 1), with five changes agreed by the user on 11-16
(bundled by the user's choice: one run, readings by component afterwards; ablations later). Launcher: `train.py`.
Code in the canonical package, every change recipe-gated: F6.1's checkpoint loads strictly into the changed code and
the 70 CPU unit tests pass.

## Changes and their sources

1. **Spatial history** (`cfg.record`; `M.Record`, the denoiser's `record_read`, the belief's `record_in`). Per periphery
   cell, a fixed S4D-Lin record of every channel of its own coarse token (S4D, Gu et al. 2022: A_n = -1/2 + i pi n,
   32 modes, B = 1, the S4D listing's bilinear step at dt = 1/32), reset at episode starts, built from stop-gradient
   tokens (a denoiser input, LOSSES 7a). The denoiser adds a read of the record through the previous frame to each
   cell's token through test 3's per-cell MLP (Linear -> GELU -> Linear, last layer zero-initialised;
   `20261116_temporal_memory`): by test 3's bars the two reads tied, but the user chose the MLP (ahead on the awake
   clocks at a fraction of the cost: one 256-wide layer); the linear form is not built; Mamba reads the state after each frame through a per-cell map to the token width and a linear
   map beside the flattened percept (the user's split: the history feeds both). Evidence: tests 2-3 (clocks partial
   at 4,000 updates, G1 -22% against the recency run at matched updates, coal / iron drawn better); measurements 2 and
   4 (rotating bases make elapsed time linearly readable, 0.90 against 0.15 for real decays).
2. **Route by role** (`losses.route_by_role`). The reward, continuation, forecast and gaze heads read a second Mamba
   pass over detached percepts: their gradients train Mamba and its input maps, never the encoder; G1 (through the
   memory tokens) and TSP / reconstruction keep the attached pass. The encoder, the record's source and G1's target, is
   owned by objectives that ask for everything visible; Mamba, selective, by the heads. Precedent (sources read):
   - Dreamer 4 (2509.24527, source): "The dynamics model operates on the interleaved sequence of actions and
     representations produced by the frozen tokenizer"; reward / policy MTP heads fine-tune the dynamics transformer
     while "we continue to apply the video prediction loss". Reward trains the dynamics model, never the tokenizer:
     the same split.
   - IRIS (2209.00588, source, Algorithm 1): "Update E and D" from the autoencoder's own losses (L1, commitment,
     perceptual), then "Update G" from transition, reward and termination: reward never trains the encoder.
   - ATC (Stooke et al., ICML 2021, source): an encoder trained only by its unsupervised loss, "fully detached from any
     RL gradient update", matches or beats end-to-end RL on DMControl and DMLab; on Atari 5 of 8 games, Breakout and
     Space Invaders worse (a caveat: small task-relevant features).
   - Counter-precedent: DreamerV3's code (`agent.py`: `sg(feat, skip=config.reward_grad)`, `reward_grad: True`) lets
     reward train its representation by default; the switch to stop it is in Hafner's own code.
   - Keeping G1's gradient into the encoder: JEDI's ablation (removing the denoising gradient into the encoder hurts;
     LOSSES principle 3). Our combination (grounding + G1 own the encoder, reward owns the belief) joins two
     established halves; it has no single precedent.
3. **Harmoniser at its fixed point** (`losses.harmoniser_fixed_point_rate: 0.01`). sigma_g set each training call to
   HarmonyDream's stationary point for the group's running mean loss (App. B: sigma^2 = mu (1 + sigma)); running mean
   at DreamerV3's normaliser rate 0.01 (danijar/dreamerv3 `configs.yaml`); no gradient. Measured need: F6.1 ended with
   G1 / G2 / G3 at 2.7 against fixed points 15.5 / 21.3 / 12.8, G4 at its own (1.35 / 1.4)
   (`20261116_loss_share/RESULTS.md`).
4. **Gaze forecast** (`heads.gaze_forecast`, group G5). The chooser's window distribution (the live chooser's, computed
   in `Controller.prepare` before the world's update) forecast from the chooser's own inputs and the belief predicted
   before the frame, cross-entropy (`20261115_gaze_learnability`, gaze.py's head and loss). Evidence: it taught the
   asleep phase beyond time and light (+0.035 nats at 3,000 updates, still accelerating), no awake clock; recovery
   decayed without its reward request, so it sits beside the reward forecast. Under route by role it trains Mamba only.
5. **Mamba-3** (`belief.mixer: mamba3`, chunk 64, mamba_ssm's SISO recommendation). `M.FunctionalMamba3`: the official
   module's forward with state passing (its kernel's Input_States / return_final_states / cu_seqlens), episode starts
   as variable-length segments. Its single-step decode (`step`) is "only tested on H100" and needs CuteDSL; the world
   never uses it (training, evaluation and imagination run the chunked forward with a carry). A defect found and fixed
   by the pre-launch check: the kernel casts Angles to bf16 and returns the final angle state in bf16, so the carried
   rotation angle was re-rounded (~0.01-0.025 rad) at every chunk boundary (0.065 rad after 8 chunks; keys rotated
   wrongly; chunked against contiguous 0.3-0.55%). The wrapper now recomputes the final angle with the same angle
   kernel on fp32 copies of the bf16 values the kernel used: carried angle, key and value exact, SSM state 2e-4,
   outputs 0.4-0.5% per chunk under bf16 without growth (the kernel's own bf16 arithmetic; our Mamba-2 wrapper: 0.2-0.35%
   under bf16). The module's own inference path never feeds states back into this kernel, so upstream never met it.
   Screen evidence
   (`20261116_temporal_memory` measurement 3): no cost, no gain on the clocks; the paper: state tracking Mamba-2 lacks.

## Pre-launch checks (`checks.py`, GPU after test 3; results in `checks.json`)

- Mamba-3: wrapper = official forward on one fresh sequence; chunked = contiguous with the carry; episode resets
  (before / after); three streams with a carried state and mixed resets = each alone; finite gradients. fp32 and bf16.
- Units: the record equals test 2's; the harmoniser equals its fixed point and has no parameters; gaze rows map to
  their streams and frames; route by role: per-loss encoder gradient norms 0 for reward, continuation, forecasts and
  gaze, positive for G1, TSP and reconstruction; every head's gradient reaches Mamba.
- Throughput at run scale (the trainer itself, 16 streams, F5's micro-batching): seconds per update and peak GiB at 64
  and 256 steps, route by role on and off; F6.1's own logged values beside them. The run launches only if the peak at
  256 steps stays under the 6 GB card's budget (F6.1: 2.4-3.5 GiB) and the time per update keeps 25.6M frames within
  about 14 hours.

Results (`checks.json`; logs beside it). All pass.
- Mamba-3 (after the angle fix): wrapper = official 0.0; chunked against contiguous 6e-4 fp32, 4.5e-3 bf16; eight
  chained chunks: angle exact, state 3-4e-4, outputs flat (no growth); resets and streams match; gradients finite.
- Units: record = test 2's (0.0); harmoniser at its fixed point (0.0), no parameters; gaze rows correct; route by role:
  encoder gradient exactly 0 for reward, continuation, every forecast distance and the gaze forecast, positive for
  G1 (0.0022), TSP (0.018) and reconstruction (0.060); every head reaches Mamba (G1 0.006, reward 1.19, continuation
  0.15, reward forecast 1.28, gaze 0.31). G1 is checked with the denoiser's zero-initialised outputs randomised: at
  initialisation they pass no gradient (0.0 in the first run, which is why it read false).
- Throughput (30 updates from scratch, the trainer itself; F6.1's log at comparable points):

  | steps | P1.5 route on | route off | F6.1 | peak GiB on / off / F6.1 |
  |---|---|---|---|---|
  | 64 | 2.03 s | 1.99 s | 2.06 (update 25), 2.17 (late) | 3.08 / 2.94 / 2.60 |
  | 256 | 7.65 s | 7.44 s | 8.0 (steady) | 3.02 / 2.88 / 2.60 |

  Route by role costs 2-3% (the second Mamba pass); the whole bundle runs at F6.1's speed. At F5's schedule (12,500
  updates at 64 steps, 3,125 at 256): about 13.7 h of training, against 14.5 h at F6.1's logged rates, plus the
  same monitor and checkpoint overhead. Peak allocated 3.1 GiB (+0.5 over F6.1) on the 6 GB card. Neither grows with
  run length: the record state is fixed per stream (81 x 1,024), the second pass is per update.

## Launch (after Codex's review; user 11-16)

One segment to the full budget, F6.1's environment; the trainer pins the saccade sources' hashes, so `saccade/` must
not change during the run (the evaluation adapter is written in this folder).

    systemd-run --user --unit=saccade-p15 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" bash -c 'D=artifacts/experiments/20261116_p15
        .venv/bin/python -u -m artifacts.experiments.20261116_p15.train $D/run >> $D/training.log 2>&1'

## Readings after the run (fixed before results; one seed)

Against F6.1 (and test 2 / 3 for the record):
- Clocks drawn natively (check 15's natural catalogue, check 17's pair banks): awake energy, drink, food drops and
  the asleep rise; the plain question is whether a from-scratch world with the record draws them beyond F6.1 (0.03).
- The encoder under route by role: periphery effective rank and token variance against F6.1 (4.88; 0.26) and F4
  (5.96; 0.38); late token drift (the loss-share measurement); coal / iron 0->1.
- The belief: recovery and cooldown (reward-priced) kept; the asleep phase (the gaze forecast); reward calibration.
- G1 (monitor), TSP, generated-vs-real classifier, danger and the standard S1 readout.
The evaluation stack must feed the record to every sampling path (the denoiser's conditioning and Mamba's input);
that adapter is written during the run, before any reading.
