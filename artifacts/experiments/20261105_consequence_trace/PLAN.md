# Stage B: where each failing consequence stops (plan + code; revised after Codex's review)

User 11-05: research first (RESEARCH.md = Stage A), then this measurement; Codex reviewed be525f12
(`../20261105_consequence_trace_review/REVIEW.md`); the user asked for B to run automatically after F2.1's evaluation
queue. No training, no repair, no follow-up queued beyond the declared GPU smoke -> F2 -> F2.1 sequence.

## Question

On the frozen F2 world, for each failing consequence: is it (H1 / H2) missing from what the generator receives (lost
at the 6,569 -> 256 projection, or in Mamba), (H3 / N1) received but missing from the field's own estimate on
correctly distributed inputs, (start) present there but not at the sampler's pure-noise start, (N2) in the first
estimate but lost by the sampler; and how sensitive is it to each route (N3, descriptive)?

Why this is not a repeat. Measured on F2 already: what the SAMPLES contain (actor interface; M0.5 battery's readers:
drink appearance 67-83%, increment 8-25%, wood 31-72%, table 35-43%, adjacent / risky damage 0-1.7% vs recorded
14-22%), the action's route for drink / wood / table (belief only: NOOP in the belief removes them), memorisation (not
memorised). Never measured on any factorised world: the field's estimate across noise levels, the field at the first
sampling call, the sampler's own loss relative to that call, the actual training term's allocation; never on coal,
iron or damage. The field instruments (10-29 A / F, 11-01 first call) ran on joint-denoiser worlds only.

## World, data, classes

F2 checkpoint 10 (`20261103_p1f2/run/ckpt_10.pt`, sha256 e0f6a24e...), dev labels (1,009 episodes), S1-seeded random
fixations (F2's outcomes are gaze-insensitive, own-gaze reading 11-04). Second world, same script and readings, read
for agreement only: F2.1's selected checkpoint 10 (`20261104_p1f21/run/ckpt_10.pt`). One world seed each.

Classes from the state at t - 1 and a_{t-1} (game_logic rules, RESEARCH.md 1), two draws per context:
coal 1,184 contexts (491 start at 0), iron 669 (382), wood <= 1,500, drink <= 1,500 (all: awake, DO, faced block,
tool, count <= 8, no mob on the faced cell; checked on the labels: count + 1 and the faced cell becoming path / grass
in 100% of rows), field_calibration's risky <= 2,000 and adjacent <= 2,000 (cooldown split descriptive: hidden) and
safe <= 500 (negative control); random <= 1,000 frames for B4 only. Functional: catalogue's linear f per part, 0 = no
consequence, 1 = consequence: the HUD count cell (start -> consequence value, mean real tokens) and, for mining, the
faced map cell (its token at t - 1 -> the mean real token of path / grass at that screen cell). For a class measurable
from the conditioning, the exact conditional mean's average f over the class equals the recorded average at every
noise level on real successor + noise (10-29 A); not within subgroups defined by hidden state (Codex 2).

## Measurements (trace.py)

- B0 geometry: |b - a| per part; sigma_d2 = |b - a| / 2 is a declared SNR threshold (d' = 2: two known modes still
  ~84% separable, Codex 6), with |b - a| / sigma at the first level; draws and SURROGATE objective above sigma_d2.
- B1 readers (information_probe 10-29 E, unchanged protocol) for coal, iron (awake, faced block, any action; label
  count + 1), drink and zombie attack (10-29's populations), on (a) the full conditioning; (b) predicted belief + action
  + fixation (the memory route's source); (c) frame t - 1's raw percept (periphery, glimpse, fixation, lighting) +
  a_{t-1}, the projection's input; (e) frame t - 1's percept embedding + the same a_{t-1}, the projection's output;
  (f) the 8 context frames' coarse tokens (periphery, lighting) + a_{t-1}, what the physical tokens can read directly
  (never the glimpse); (d) the rule's inputs. (c), (e), (f) carry no predicted belief (the first version's "last frame"
  arm carried that frame's 256-D belief: Codex 1, verified on s1.frame_features).
- B2 inference: the field on real successor + noise at 0.05 ... 20 plus the 8 monitor levels and the production first
  level; the first call (the cascade's first evaluation from pure noise, asserted bitwise equal to the sampler's own
  first evaluation) beside the exact two-outcome conditional mean on the same pure-noise input, computed in each row's
  measured geometry (|a|, |b|, |b - a|) with the class's recorded rate (Codex 2's analytic control; for a class whose
  outcome the conditioning determines, p = 1 and the exact value is 1 from any input); the production sample from the
  same start.
- B3 routes, same draws: beliefs recomputed (control); NOOP at t - 1 in both action channels (in support; the M0.5
  routing kept the denoiser's own action, so belief-only); both again with the 8 context frames entered at sigma 5
  (a stress test: all eight training sigmas above 5 has probability ~2e-11, Codex 5).
- B4 the actual P1 G1 term: model.denoise_loss's computation on [8 context + target] windows, every frame its own
  independent physical and glimpse sigma ~ p_train (factorised P1's diffusion forcing), the chunked training forward,
  the UNCLIPPED raw output against its preconditioned target, 4 draws per row (Codex 3). Reported per part (periphery
  + lighting by the physical sigma, glimpse by its own): the whole term's allocation by band on random frames, the
  consequence cells' share on their event frames by band and above sigma_d2, each as trained and with the same errors
  weighted as an x-space loss, and the constant that matches the two magnitudes.

## Declared readings (per class and part; episode-bootstrap SE; "beyond 3 SE")

- R0 instrument checks, else the class is not read: recorded outcome as defined >= 99% (mining, drink); |bias| < 0.01
  at sigma 0.05; SAFE |f| < 0.05 at every level <= the first level, first call and production; |rerun control| < 0.01;
  first call = the sampler's first evaluation (asserted). Measurability: R2 and R5 are read as conditional-mean
  statements for coal / iron / drink only if reader (b) or (c) reaches AP >= 0.95 on that population; otherwise
  descriptively.
- R1 information (one-sided): rule inputs AP >= 0.95 qualifies the population. Per arm: AP >= 0.9 accessible to this
  reader, <= 0.7 not accessible to it, else partial. (c) accessible and (e) not: the projection's output is not
  accessible to this reader (H1-consistent); (e) accessible and (b) not: H2-consistent. A weak reader is not proof of
  absence; (f) says whether the physical route could read it directly, not that the network does.
- R2 decision point: rho = first call / exact two-outcome first call. >= 0.8 the field's first estimate is what an
  exact denoiser would give from that start; <= 0.5 it lacks it.
- R5 calibration (the N1 reading): bias(sigma) on real successor + noise; the lowest level with bias < -0.05 beyond
  3 SE, against sigma_d2. N1 predicts calibration below ~sigma_d2 and underweighting above it.
- R3 descriptive: field(real + noise at the first level) - first call. Not evidence of under-training by itself
  (Codex 2: an exact denoiser can show a large gap when p < 1).
- R4 sampler: production - first call. < -0.1 beyond 3 SE: the sampler loses the field's first estimate (N2); within
  +-0.05: the deficit is the first estimate's. (The interpolation coefficient 0.958 does not decide the outcome; later
  calls can, which is what R4 measures.)
- R6 action route: rerun - NOOP first call vs the first call; NOOP first call ~0 for mining / drink (control); ready
  risky / adjacent rows under NOOP are a deterministic hit (game_logic.update_mobs), read descriptively (the cooldown
  is hidden from the conditioning unless inferable).
- R7 route stress, descriptive only: contrast_change = C(5) - C(0.1), C = first call(DO) - first call(NOOP), C(5) with
  every context frame at sigma 5. Sensitivity, not a route identification; N3 stays unresolved without a
  route-specific control (Codex 5).
- R8 B4, descriptive: the measured allocation of the training term (replaces RESEARCH.md's surrogate).

Interpretation, declared now, per class:
- N1-consistent: (b) accessible, R5 underweighting at levels >= sigma_d2 while calibrated below, R2 <= 0.5. The
  conditioning carries the consequence; the field does not use it where the input hides it. Names the objective's
  allocation as the candidate for a training test (Stage C); does not prove it.
- Start-distribution: R5 calibrated at the first level (|bias| < 0.05) but R2 <= 0.5: the field is right on correctly
  distributed inputs at that level and wrong from the sampler's pure-noise start: a sampler / schedule mismatch
  (Lin et al.'s finite terminal SNR), not under-training.
- H1 / H2-consistent: R1 as above, the consequence not accessible after the node it localises.
- N2: R2 >= 0.8, R4 < -0.1.
- R2 >= 0.8, R4 within +-0.05, yet the samples' readers show low rates: readout mismatch between f and the readers;
  stop and reconcile before reading anything else.
- Anything else: "not localised", reported as such.

What this cannot show: why training produced the field (only a training intervention can, Stage C); that a change
would fix it; generality beyond these two single-seed worlds.

## Run (automatic, user 11-05)

**Priority amendment, user 11-05 at approximately 14:08 AEDT:** pause the remaining F2.1 evaluation and run the
unchanged trace measurement first. `priority_queue.sh`, unit `trace-priority-queue`, invokes
`launch.sh --before-evaluation`: verify the selected checkpoint, wait for the GPU, then smoke -> F2 -> F2.1.
Only after all three exit successfully does it start the serial, resumable evaluation worker in
`saccade-p1f21-eval-recovery4`. Previously completed S1 and both interface readings are retained; the interrupted
M0.5 stage is rerun. The old waiting queue and evaluation unit were stopped before starting this queue.
The default `launch.sh` mode below remains available, but is not the current order. No measurement or training
setting changes in this amendment.

`launch.sh` (systemd unit `trace-queue`, its own cgroup) waits until F2.1's `STATUS.json` says `complete` and its
evaluation unit has stopped (any other end state: it stops without launching), then, each under
`MAMBA_DETERMINISTIC=1`: (1) a GPU smoke, 40 dev episodes, into `readings/smoke_gpu` (time, host / GPU peak, the
first-call assertion on the GPU kernels; must exit 0), (2) F2 into `readings/f2`, (3) F2.1 into `readings/f21`.
Logs and `QUEUE.json` (state, times, exit codes, peaks) beside them.

## Smoke (CPU, 11-05; not a reading)

40 dev episodes, 6 contexts per class, every module's device on the CPU, Mamba on its reference scan, no GPU visible
(driver in Claude's scratchpad, not committed). The whole pipeline ran (B0-B4, all classes, all reader arms); the
first-call assertion held on every class and variant; recomputed beliefs differ from the stored ones by ~3e-6 in f;
the field at sigma 0.05 has bias ~0. One amendment came from the first smoke, recorded: the direct route was first
read as first call(context 0.1) - first call(context 5), declared for a DROP only; values moved UP under the noised
context, also on SAFE, so it became the change of the DO - NOOP contrast (fifth variant), now descriptive (R7).
Opportunity rates had been computed after capping (fixed). The revised script smoked again after the review (3 min
55 s, 1.8 GB). No smoke number is read.

## Review (Codex, 11-05) and what changed

1. Last-frame arm contained that frame's predicted belief: correct (s1.frame_features + bp). Rebuilt (c), added (f);
   R1 wording one-sided.
2. Pure noise vs real + noise can flag an exact denoiser: correct when p < 1; its counterexamples use separation /
   sigma 2-4 and an outcome at the origin. Added the exact two-outcome reference in each row's measured geometry; R2
   is now relative to it; R3 descriptive; R5 (real + noise calibration) is the N1 reading. For p = 1 classes the exact
   value is 1 from any input, so the original concern does not apply to them; measurability gated by B1. Hidden-state
   subgroups (cooldown) descriptive.
3. B4 used the clipped estimate, one sigma and clean context: correct. Replaced by the training-form term (unclipped,
   independent sigmas, noisy context, glimpse separate).
4. Wording (sigma_data, surrogate, Dreamer-4 mixture, 0.958 coefficient): conceded, RESEARCH.md amended.
5. Context-sigma-5 is a stress test: conceded (R7 descriptive). Recomputed NOOP inputs now reused for noop_ctx5. A GPU
   smoke precedes the full runs (time and memory not claimed from the CPU smoke).
6. sigma* renamed sigma_d2 with the separation / sigma curve reported.

## Stage B2: stone (user 11-05 ~15:20: "launch it"; "pause F2.1 to run this, it's more important")

**Claim.** On F2 and F2.1 the coal / iron counts ignore the conditioning that the same rows' mined cell uses, and the
counts' use of the conditioning is monotone in per-slot training exposure (RESULTS.md 2: wood 682k events -> 0.72 /
0.64, drink 414k -> 0.44, coal 102k -> 0.10, iron 63k -> 0.02-0.10; shared cell rule ~1.8M -> 0.94 / 0.84).
**Competing explanations.** (a) per-slot exposure; (b) the count needs the event from elsewhere on the screen (a route /
locality limit); (c) the digit increment's composition (copy d, render d + 1); (d) something material-specific.
**Separating control.** Stone: the most exposed count (929k events at F2's budget, ~1.4x wood, ~9x coal), the same HUD
row as coal (slot 69, between wood 68 and coal 70), the same faced-cell geometry, the same tool (wood pickaxe), the
same rule (block -> PATH, count + 1; verified 100% on dev's 10,903 contexts, 18% start at 0), increments at every
digit. It differs from coal in exposure (and the block's look), not in route or composition.

**Measurement.** trace.py with stone added (HUD 69 = s_inv column 1, renderer column 6; MINING stone = (STONE 4, PATH,
69, wood_pickaxe); appended last in CAP so the other classes' draws are unchanged; `KINDS` argument = stone only, no
B1 readers). Same B2 / B3 / B4, 1,500 contexts x 2 rows, F2 ckpt_10 then F2.1 ckpt_10, MAMBA_DETERMINISTIC=1.
`stone_queue.sh` (unit trace-stone-queue): runs both, then resumes F2.1's evaluation with Codex's recovery command
(serial worker, completed stages kept) whatever the traces' exit codes.

**Declared reading** (R0 as before: recorded >= 99%, |bias| at 0.05 < 0.01, rerun < 0.01, else not read). Primary:
the stone count's field mean on real successor + noise at sigma 20 (the conditioning's use; coal 0.12 / 0.11, wood
0.70 / 0.62 overall), both worlds:
- >= 0.6 in both: exposure-consistent. Rules out (b) and (c) as blocking at this exposure (stone shares coal's route
  and composition); per-slot exposure is the leading account (not proven: stone's slot also changes by placing and
  crafting, which counts as exposure of the slot, and its look differs).
- <= 0.2 in both: per-slot exposure does not govern at these magnitudes (stone > wood); (b) / (c) / (d) remain and wood's
  success needs another account.
- otherwise, or the worlds disagree: unresolved, reported as such.
Descriptive: first call, production, the cell, the joint cell / count outcome, appearance vs increment, the blind
reference (`posthoc.py`), B4. One world seed each; no further run queued on any outcome.
