# P1-F3: F2 with the physical G1 term as an x-space loss (user 11-05; for Codex's review; not launched)

**Question.** Does moving the physical denoising objective's weight from near-readable inputs to noise levels where
only the conditioning can supply a consequence improve the world's rare consequences without unacceptable regressions?
(Codex's framing, 11-05.) The consequence trace (`../20261105_consequence_trace/RESULTS.md`) localised the counts'
failure to the field's own estimate, ~2e-4 of the objective under our EDM weighting (sigma_data 1 = x-loss + eps-loss,
Kingma & Gao 2023); the x-space loss raises that share ~3.5x on F2's measured term (RESEARCH_N1.md 3, 5). F2 does not
fit first coal even on training histories seen 10.5 times (11-04 train-vs-dev): an under-fitting, learning-signal
problem, the part of the ceiling this lever targets. Hidden state (zombie readiness, mob health), head ceilings,
self-feeding drift and false positives are NOT targeted (RESULTS.md 8; tomorrow's investigation).

**One change.** F2's recipe (F1.6 world + the original online cheap-head fixation; F2's trainer, chooser, corpus, world
seed 1, 25.6M frames, checkpoint rule), with periphery and lighting G1 per element = raw-output squared error x 3.425 x
c_out(sigma_physical)^2 (= 3.425 x the x-space error; Dreamer 4, 2509.24527, "we formulate the objective in data space";
its largest single ablation gain, FVD 326 -> 151). Glimpse G1 unchanged (EDM; user 11-05: the glimpse check found no
conditioning-use deficit; EDM's optimality is recorded as up for debate, DECISIONS.md). 3.425 = sum(raw) / sum(raw x
c_out^2) of the physical part on F2 ckpt_10's measured training-form term (B4, 2,000 random frames x 4 draws; bootstrap
3.29-3.62; F2.1 3.46): G1's physical magnitude and the glimpse's 23.4% share match F2 at calibration (they evolve in
training). Applied only inside TR.p1_losses / TR.encoder_diagnostics: the checkpoint monitor stays in F2's EDM units,
so F2's selection rule reads comparable numbers. Noise draws, context noise, sampler and data are unchanged.
Launcher-scoped (`train_p1f3.py`): saccade/model.py, train.py and m0.yaml are pinned by F2.1's evaluation seal (its
deferred uptake must still validate); cfg edm.g1_weighting records the weighting in every checkpoint (user 11-05:
canonical after the results).

## Declared readings (single world seed; vs F2 at the same seed and data order; read patterns and large effects)

Instruments are the consequence trace (fields: first call, field across sigma, B3 sampler steps, B4), the actor
interface at 3 AND 16 sampler steps (F2's 16-step reading: `../20261105_consequence_trace/readings/evening/`), the
own-history pilot at 3 and 16 steps, S1, the glimpse check. Worlds are compared at >= 16 steps or on fields (the 3-step
vote amplifies small differences: RESULTS.md 8).

Primary (the lever), trace B2 field at sigma 20 (conditioning only), F2 / threshold:
- coal HUD 0.12, iron HUD 0.05: lever CONFIRMED if F3 coal >= 0.25 AND iron >= 0.15 (each beyond 3 SE of F2);
  NO measurable lever if coal <= 0.15 AND iron <= 0.08; otherwise partial.
- plateau: wood 0.70, stone 0.63 HUD: both >= 0.80 -> signal-limited plateau; both <= 0.72 -> not; else unresolved.
- descriptive: drink increments (0.41-0.44), first call per class, posthoc blind-denoiser comparison, B4 on F3's errors (its
  "as_trained" column is the EDM convention; F3's physical term is the "x_space_weighting" column, descriptive).

Actor-facing, interface at 16 steps (native heads on generated futures), F3 - F2, "improved" = +0.05 beyond 3 SE
(unpaired SE): first coal (F2 0.364), first iron (0.193), risky death (0.062; recorded 0.128), adjacent damage (0.056;
0.159), risky damage (0.115; 0.127). Sapling context (0.142 vs 0.135) must stay calibrated (|F3 - recorded| <= 0.02).

Guards (harm; each reported, any breach is a regression to weigh, not an automatic verdict):
- low-noise fidelity: trace R0 |bias| at sigma 0.05 < 0.01 on every class / part;
- S1.1 perception presence recall / precision within 0.02 of F2; S1.4 TSP nats/pixel within 5% of F2 (2.209);
- interface at 16 steps: wood (0.77), stone (0.77), table (0.93), place stone (0.94), drink (0.86), wood / stone tools
  (0.87-0.89 / 0.77-0.79) not down by > 0.05 beyond 3 SE;
- false rewards: random-history unlock probability at 16 steps (F2 0.048, recorded 0.034) not up by > 0.01;
- own-history at 16 steps: S2.1 count gate fires on <= 2 of 8 kind x mode checks (F2: 2).

## Run

`run.py` = F2.1's one-shot worker (`../20261104_p1f21/run.py: main`, unchanged) with this launcher, gate checks.json,
then `readout.py` on the selected checkpoint: S1 (writes run/s1 and its frozen reader) -> consequence trace (all
classes incl. stone, readers) -> sampler steps -> actor interface 3 / 16 steps -> own-history 3 / 16 steps -> glimpse
check -> readings / posthoc / allocation (CPU). Launch (user's approval after review): `systemd-run --user
--unit=saccade-p1f3 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
--working-directory=REPO .venv/bin/python -u -m artifacts.experiments.20261105_p1f3.run`. ~11.5 h training + ~1.7 h
readout. One GPU job at a time; F2.1's deferred CPU uptake is not run beside it.

## Preflight (`checks.py`)

F2.1's arm checks, unchanged (`../20261104_p1f21/checks.py: run`, with `prepare = install`): exact resume (model,
streams, carry, cache, state, RNG, chooser, optimizer, gaze bytes), the real launcher in a fresh process, 200 updates at
64 steps and 3 at 256 (finite, speed, memory), the monitor. Added for F3: (1) loss algebra on GPU: inactive and
unit-weight calls equal model.denoise_loss bitwise with the same noise; the active value equals 3.425 c_out^2 x the
physical squared errors + the glimpse's, computed separately; (2) update 1 vs F2's logged update 1: every logged term
except g1 and grad_norm identical (same init, data, gaze, RNG; only G1 changed); (3) the monitor bitwise identical with
the patch installed vs removed; (4) G1's magnitude at update 1 vs F2's (descriptive). `pin_sources.py` seals F2's pin
list + these files + the readout's instruments.

## Preflight result (11-05 20:57-21:05, `checks.json`, 464 s, peak 4.3 GiB)

All passed. F2.1's arm checks: exact resume (model, streams, carry, cache, state, RNG, chooser, gaze bytes; optimizer
max |diff| 5.7e-14), real launcher in a fresh process, 200 x 64-step and 3 x 256-step updates finite at 1.68 / 6.14 s
per update (F2 1.71). F3's: loss algebra (inactive and unit-weight calls bitwise equal to model.denoise_loss; parts sum
to it within 1.1e-7; the active value equals 3.425 c_out^2 x physical + glimpse within 1.0e-7); update 1 vs F2's
logged update 1 identical in update, frames, steps, lr, sigma, change, tsp, rec, reward, cont, noise; g1 0.9457 ->
1.1325 and grad norm 6.50 -> 6.74 (only G1 changed); monitor bitwise identical with the patch installed vs removed; G1
(descriptive, its own units) 1.13 -> 0.55 over 200 updates vs F2's 0.95 -> 0.65.

## For the reviewer

The launcher-scoped patch (reimplements model.denoise_loss with per-part weights; consumes the RNG identically); the
scale's calibration on F2's converged errors (early-training magnitudes differ; the harmoniser adapts, rate-limited);
the monitor staying in EDM units; the readout's 16-step reads; the declared thresholds; the YAML-rule deviation.
