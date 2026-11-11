# P1-F5: forecast modes on TSP and the outcome heads, change map removed (2026-11-11)

Agreed with the user on 11-11 (`../20261111_change_forecast_gate/DISCUSSION.md` and the follow-up conversation). The
three changes are bundled in one overnight run on purpose: "It's less clean scientifically, but it maximizes our
progress chance, and runs take 15h." Order: (1)-(2) design and code, (3) the TSP forecast gate (CPU), results discussed
with the user, (4) F5 launched only after Codex's review.

## Why

- **Nothing in P1 asks the belief to anticipate in a currency where one event counts.** Every surprise-priced head
  (reward, continuation, change map) reads the real outcome; only G1 anticipates, through the denoiser, beside the cache
  and the noisy target, under squared error over 1,880 numbers. Dreamer 4 trains its reward and policy heads to predict
  ahead from the current representation (multi-token prediction) while imagination keeps reading rewards off the
  generated frames.
- **The change map was a hedge that was never tested,** and it is the most Craftax-tailored component: tile-aligned
  cells, five whole-tile camera shifts, exact pixel equality, a LightGBM teacher fitted on Craftax pairs. It entered
  on 10-25 as a target for the record design (later set aside) and was repurposed on 10-27 as a second anchor for the
  periphery tokens. It was never ablated. F4's change map excluded the HUD rows, yet F4's HUD tokens encode every digit
  exactly, so TSP and the reward head alone grounded that content. Removing it returns to the agreed grounded targets
  (10-24): reward, termination, fovea pixels.
- **The change map's own forecast mode failed its gate** (`../20261111_change_forecast_gate/RESULTS.md`).

## Changes

1. **Change map removed** in the experiment launcher (`heads.change_map: null`). G2 = TSP + fovea reconstruction. No change targets are read.
   The class stays so historical checkpoints build from their own configuration.
2. **TSP forecast mode** (`heads.forecast`). The same TSP network and pixel decoder, with the current frame's periphery
   tokens and lighting latent replaced by one learned mask vector (zero-initialised; no random draw). It predicts a 3x3
   window's pixels of frame t from the belief predicted before frame t (bp, after action t-1) and the window's cell
   positions. Loss: the faithful mean loss (squared error in [-1, 1]), mean over the window, no noise-model term.
3. **Outcome forecast mode.** The same reward/continuation trunk with frame t's percept embedding replaced by a learned
   mask vector (zero-initialised): reward categorical log-loss and continuation BCE from bp alone. Continuation keeps
   its chained conditioning on the recorded reward (a valid teacher-forced factorisation p(r|b) p(d|b, r)).

Imagination, sampling and the reactive heads are unchanged; the forecast modes are training-only.

## Forecast schedule

- **Every valid frame, both modes.** The forecast terms are computed on every valid, non-first frame alongside the
  reactive terms; no masking fraction. Reasons: the reactive TSP is now the periphery's only fixed anchor and keeps its
  full rate; Dreamer 4 trains its multi-step heads on every step; no new fraction to tune. First frames of an episode
  are excluded (their belief is zeroed), as the reward head already does.
- **TSP forecast window: independent of the gaze.** One window per frame, drawn uniformly over the 49 centres from its
  own seed (`numpy default_rng([forecast seed, world seed, update])`, resumable, the training RNG untouched), cropped
  from the recorded frame. Not the chooser's fixation: the chooser's surprise term compares its forecast with the
  actual periphery of the target frame (`saccade/chooser.py`, `s = (y - taken)^2`), so it looks at the HUD more often
  exactly when a digit changes, and a forecast at that window could infer the change from where the eye went. This
  adds one fovea-sized training target per frame at a window where the eye did not necessarily look. This uses the
  offline corpus's full frames. The original learned-gaze crop still supplies the normal TSP/reconstruction targets
  and the observed glimpse, so the inference-time foveal budget is one crop as before.
- **Weighting: a fourth harmonised group, G4 = TSP forecast + reward forecast + continuation forecast.** Summing the
  forecasts into G2 / G3 would let their larger losses raise those groups' sigma and shrink the reactive anchors' and
  outcome readers' effective weight. HarmonyDream's logic (one learned scale per task) treats anticipation as its own
  task. Harmoniser init sigma 1 as the others.
- **From update 0**, like every other term; bootstrap rows, the chooser, the curriculum (chunk 64 -> 256) and the
  learning-rate schedule unchanged.
- **Routes.** Forecast terms reach the belief (Mamba, its percept projection and, through past percepts, the encoder at
  0.3x lr); never the frame they forecast. Test: changing frame t's pixels changes the forecast targets only, never the
  forecasts.
- **Not in F5:** multi-step forecasts (Dreamer 4's per-distance outputs, kept for later), forecast terms in P2's
  one-step groups (decide at P2).

Accepted confounds of the bundle: removing the change map module changes the random draws for every module built after
it (outcome heads, harmoniser order unchanged), so F5's initialisation differs from F4's beyond the changed heads; one
world seed.

## Step 3: TSP forecast gate (CPU, before review)

Same oracle protocol, pairs, foils and natural states as `../20261111_change_forecast_gate/PLAN.md` (C1 hunger 23 pairs,
C2 zombie readiness 32 pairs, C3 +1/+3/+6 brightness, 240 natural states x 8 keys), for the two new forecast targets:
- **TSP forecast:** pair cost = ||y_A - y_B||^2 / (4 n) over the window's n = 1,323 pixel values in [-1, 1], averaged
  over the 49 uniform windows; natural floor = the same over 8 keys.
- **Outcome forecast:** reward-bin and termination log-score gaps (exact, as the 11-11 observable gate) and their
  natural floors. Termination is conditioned on the observed reward, matching the implemented factorisation.
Reported beside G1 at pure noise and the change target from the earlier gate (same scenes, same states: reproduced and
asserted). R is a discrimination diagnostic, not a go/no-go (weight-invariant). What would change F5: if the TSP
forecast prices both consequences below G1 at pure noise relative to the floors, or its foils out-price the
consequences in most scenes, we discuss dropping the TSP forecast from F5 and keeping the outcome forecast.

## Review and preflight

The existing F4/HUD recipe remains the generic default. The F5 launcher takes its exact training configuration
and learned fixation controller from the pinned HUD checkpoint, resets weights and optimizer as a fresh seed-1
run, and applies the three changes above. Its reward head starts with different RNG draws after removal of the
change map. It uses the 25.6M-frame budget and the ten checkpoint monitor readings with the existing selection
rule. It starts training and then runs the checkpoint-bound, resumable evaluation queue in one service.

The CPU oracle gate finished on 23 hunger pairs, 32 cooldown pairs and 240 natural states. The median TSP/G1
contrast-to-natural-variation ratios are 4.73 for hunger and 2.26 for cooldown. The largest +6/255 edit is
stronger than the cooldown TSP signal in 18/32 pairs, so the original discussion trigger fires. At +1 and +3
no foil beats either consequence. Hunger also passes at +6, and the reward forecast separately gives all
32 cooldown pairs ln(2) nats of contrast while carrying none for passive hunger. Keeping both forecast modes
is the reviewed recommendation for the bundled arm, subject to the user's launch approval. This gate prices
the target. It does not measure learned timing, belief retention or deployment uptake.

The independent uniform forecast windows are used in the gate, the proposed run and the held-out forecast
score. The regular learned fovea stays active for ordinary TSP, reconstruction and Mamba input. The current
chooser's realized surprise reads the target frame, so reusing its selected location would give the forecast
information about that frame. A future causal alternative could choose with the preceding frames and incoming
action, using the existing contingency score plus uniform exploration and no realized target error.

## F5 judged on (fixed before launch; readout plan reviewed with the launch)

- Native exact-history timing under the full cache: hunger pairs and natural thirst alias, due vs not-due rates
  separately (truth 1 / 0); zombie readiness pairs, hit when ready vs cooling.
- The standard readout against F4 and HUD-Change: S1 entity retention (tests the small-entity hedge the change map was
  for), consequence determinant curves and crafting, actor interface, 4-step recursion, belief probes (timers, sleep).
- The forecast heads' own held-out scores, reported but not used as success: an improved forecast without native
  generated improvement is not a repaired world.

## Launch and evaluation

From the repository root, after the approved preflight and source seal:

```sh
systemd-run --user --unit=saccade-p1f5 -p MemoryMax=24G \
  -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
  -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --working-directory="$PWD" "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261111_p1f5.run
```

The worker binds the evaluation to the selected checkpoint hash after all ten monitor records exist. Queue
order: S1, paired hunger and cooldown generation, held-out forecast scores, determinant and crafting curves,
actor interface, four-step recursion, the natural thirst pair, timer-event and belief-path probes, then the
same hunger and cooldown generation on the pinned F4 and HUD controls. The cooldown pair's older hit falls
within the visual cache, so the native comparison measures the assembled world, not causal attribution to
Mamba. Use the unchanged full-cache sampler throughout.

After an evaluation interruption, resume from the same selected checkpoint with
`python -m artifacts.experiments.20261111_p1f5.readout CHECKPOINT` under the launch environment. The readout
verifies the checkpoint hash and skips only stages with completed status and their expected artifact.
