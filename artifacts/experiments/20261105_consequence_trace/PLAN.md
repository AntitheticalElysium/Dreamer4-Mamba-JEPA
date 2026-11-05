# Stage B: where each failing consequence stops (plan + code for review; not run)

User 11-05: research first (RESEARCH.md = Stage A), then this measurement, written up for Codex's review before
anything touches the GPU. No training, no repair, no follow-up queued.

## Question

On the frozen F2 world, for each failing consequence: is it (H1 / H2) missing from what the generator receives (lost
at the 6,569 -> 256 projection, or in Mamba), (H3 / N1) received but missing from the field's own estimate at the
production decision point, (N2) in that estimate but lost by the sampler; and (N3) which route carries it?

Why this is not a repeat. Measured on F2 already: what the SAMPLES contain (actor interface; M0.5 battery's readers:
drink appearance 67-83%, increment 8-25%, wood 31-72%, table 35-43%, adjacent / risky damage 0-1.7% vs recorded
14-22%), the action's route for drink / wood / table (belief only: NOOP in the belief removes them), memorisation (not
memorised). Never measured on any factorised world: the field's estimate across noise levels, the field at the first
sampling call, the sampler's own loss relative to that call; never on coal, iron or damage. The field instruments
(10-29 A / F, 11-01 first call) ran on joint-denoiser worlds only.

## World, data, classes

F2 checkpoint 10 (`20261103_p1f2/run/ckpt_10.pt`, sha256 e0f6a24e...), dev labels (1,009 episodes), S1-seeded random
fixations (F2's outcomes are gaze-insensitive, own-gaze reading 11-04). Second world, same script and readings, read
for agreement only: F2.1's selected checkpoint after Codex's evaluation queue. One world seed each.

Classes from the state at t - 1 and a_{t-1} (game_logic rules, RESEARCH.md 1), two draws per context:
coal 1,184 contexts (491 start at 0), iron 669 (382), wood <= 1,500, drink <= 1,500 (all: awake, DO, faced block,
tool, count <= 8, no mob on the faced cell; checked on the labels: count + 1 and the faced cell becoming path / grass
in 100% of rows), field_calibration's risky <= 2,000 and adjacent <= 2,000 (with the cooldown split) and safe <= 500
(negative control). Functional: catalogue's linear f per part, 0 = no consequence, 1 = consequence: the HUD count
cell (start -> consequence value, mean real tokens) and, for mining, the faced map cell (its token at t - 1 -> the mean
real token of path / grass at that screen cell). By the law of total expectation the field's mean f over a class must
equal the recorded mean f at every noise level if the field is the conditional mean (10-29 A).

## Measurements (trace.py)

- B0 consequence size |b - a| per part and sigma* = |b - a| / 2 (above it the noisy input no longer shows the
  consequence: detection d' < 2); training draws and objective above sigma*, ours vs Dreamer-4-style (noise_profile).
- B1 readers (information_probe 10-29 E, unchanged protocol) for coal, iron (awake, faced block, any action; label
  count + 1), drink and zombie attack (10-29's populations), on: (a) the full conditioning, (b) predicted belief +
  action + fixation (the memory route's source), (c) last frame + action (the direct route's last frame),
  (e) frame t - 1's percept embedding (after the projection) + action, (d) the rule's inputs (positive control).
- B2 the field on real successor + noise at 0.05 ... 20 plus the 8 monitor levels and the production first level;
  the first call itself (the cascade's first evaluation from pure noise, asserted bitwise equal to the sampler's own
  first evaluation); the production sample from that same start.
- B3 routes, same draws: beliefs recomputed (control); NOOP at t - 1 in both action channels (in support; the M0.5
  routing kept the denoiser's own action, so belief-only); both again with the 8 context frames entered at sigma 5
  instead of 0.1 (direct route mostly removed, memory intact). The route reading is the action contrast DO - NOOP
  with and without the direct context, so a drift that the noised context causes in every row cancels.
- B4 the consequence cells' share of the objective on their event frames, at the 8 equal-mass monitor levels.

## Declared readings (per class and part; episode-bootstrap SE; "beyond 3 SE")

- R0 instrument checks, else the class is not read: recorded outcome as defined >= 99% (mining, drink); |bias| < 0.01
  at sigma 0.05; SAFE |f| < 0.05 at every level <= the first level, first call and production; |rerun control| < 0.01;
  first call = the sampler's first evaluation (asserted).
- R1 information: rule inputs AP >= 0.95 qualifies the population. Per input: AP >= 0.9 accessible, <= 0.7 not
  accessible to this reader, else partial. Accessible in (c) but not (e): lost at the projection (H1); in (e) but
  not (b): lost in Mamba (H2). A negative is a reader result, not proof of absence (Codex).
- R2 decision point: rho = first call / recorded f. >= 0.8 the field carries it; <= 0.5 the field lacks it.
- R3 leak: field(real + noise at the first level) - first call. If R2 lacks and leak > 0.2 beyond 3 SE: the field
  answers from the successor leaking through the noise, not from the conditioning (Lin et al. 2024 pattern; N1).
- R4 sampler: production - first call. < -0.1 beyond 3 SE: the sampler loses the field's estimate (N2); within
  +-0.05: the deficit is the field's.
- R5 profile: the lowest level with bias < -0.05 beyond 3 SE, against sigma*. N1 predicts calibration below ~sigma*
  and underweighting above it.
- R6 action route: rerun - NOOP first call vs the first call (the action reaches the consequence through the inputs
  it changes); NOOP first call ~0 for mining / drink (control); ready risky / adjacent rows under NOOP are a
  deterministic hit (game_logic.update_mobs: zombie adjacent before its move, player stays, cooldown <= 0): NOOP first
  call against 1.
- R7 direct route: C(0.1) = first call(DO) - first call(NOOP) with the context at 0.1, C(5) the same with the context
  at sigma 5; contrast_change = C(5) - C(0.1). Within +-0.1 (and 3 SE): the action-conditional consequence does not
  depend on the direct context (memory carries it, or carries its absence). C(5) above C(0.1) by > 0.1 beyond 3 SE:
  the memory predicts the consequence and the direct context SUPPRESSES it (the field copies the context's unchanged
  cell); below by > 0.1: the direct context supports it.
- R8 objective share: descriptive.

Interpretation, declared now, per class:
- R1 accessible in (b), R2 lacks, R3 large: the information reaches the generator; the field did not learn to use it
  at the decision level. Points at the objective / noise allocation (N1). Stage C would then be a training test of the
  noise recipe, designed and reviewed separately.
- R1 not accessible in (b) but in (c) / (e): lost before the generator's memory input, at the node R1 localises; with
  R7 showing no direct-route use, the factorised memory-only path is the bottleneck for that consequence.
- R2 lacks while R7 shows suppression (C(5) well above C(0.1)): the memory route carries the consequence and the
  direct context overrides it: route competition (N3), not missing information.
- R2 carries, R4 loses: the sampler (N2): the user deferred shortcut forcing; report only.
- R2 carries, R4 within +-0.05, yet the samples' readers show low rates: readout mismatch between f and the readers;
  stop and reconcile before reading anything else.
- Anything else: "not localised", reported as such.

What this cannot show: why training produced the field (only a training intervention can, Stage C); that a change
would fix it; generality beyond these two single-seed worlds.

## Run

After F2.1's evaluation queue (one GPU job at a time; Codex owns that queue), on the user's go after review:
`systemd-run --user --unit trace-f2 -p MemoryMax=16G -E MAMBA_DETERMINISTIC=1 --working-directory=$PWD
.venv/bin/python -m artifacts.experiments.20261105_consequence_trace.trace
artifacts/experiments/20261105_consequence_trace/readings/f2 artifacts/experiments/20261103_p1f2/run/ckpt_10.pt`,
then the same for F2.1's selected checkpoint into `readings/f21`.

## Smoke (CPU, 11-05; not a reading)

40 dev episodes, 6 contexts per class, every module's device on the CPU, Mamba on its reference scan, no GPU visible
(driver in Claude's scratchpad, not committed). The whole pipeline ran (B0-B4, all seven classes, all five
readers per population); the first-call assertion held on every class and variant; recomputed beliefs differ from
the stored ones by ~3e-6 in f; the field at sigma 0.05 has bias ~0 (coal 0.0003). One amendment came from it,
recorded here: the first version read the direct route as first call(context 0.1) - first call(context 5), declared
for a DROP only; the smoke showed values moving UP under the noised context, also on the SAFE control, so the route
is now read as the change of the DO - NOOP contrast (a fifth variant, NOOP with the noised context), in both
directions (R7). Opportunity rates were computed after capping (fixed: uncapped counts). No smoke number is read.

For the reviewer: (a) context sigma 5 on all 8 frames is in support frame by frame under diffusion forcing but
jointly rare; (b) the NOOP variant changes both action channels; (c) B4 uses the inference context (0.1), not the
training form's noisy context; the exact training form is loss_accounting (10-29 H), which has no coal / iron
classes; (d) real successor + noise contains the answer at low sigma: an instrument, never a repair.
