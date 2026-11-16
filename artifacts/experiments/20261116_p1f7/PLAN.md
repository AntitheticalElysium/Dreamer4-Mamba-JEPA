# P1-F7 (2026-11-16 night; the run the user called "P1.5")

From scratch, F6.1's recipe and schedule (25.6M frames, world seed 1), with five changes agreed by the user on 11-16
(bundled by the user's choice: one run, readings by component afterwards; ablations later). Launcher: `train.py`.
Code in the canonical package, every change recipe-gated: F6.1's checkpoint loads strictly into the changed code and
the 70 CPU unit tests pass.

## Changes and their sources

1. **Spatial history** (`cfg.record`; `M.Record`, the denoiser's `record_read`, the belief's `record_in`). Per periphery
   cell, a fixed S4D-Lin record of every channel of its own coarse token (S4D, Gu et al. 2022: A_n = -1/2 + i pi n,
   32 modes, B = 1, the S4D listing's bilinear step at dt = 1/32), reset at episode starts, built from stop-gradient
   tokens (a denoiser input, LOSSES 7a). The denoiser adds a read of the record through the previous frame to each
   cell's token (read: test 3's verdict, `20261116_temporal_memory`, linear or MLP, decided on its fixed bars; the other
   form is not built); Mamba reads the state after each frame through a per-cell map to the token width and a linear
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
   never uses it (training, evaluation and imagination run the chunked forward with a carry). Screen evidence
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
