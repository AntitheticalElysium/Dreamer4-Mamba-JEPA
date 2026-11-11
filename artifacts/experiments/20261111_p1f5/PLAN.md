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

1. **Change map removed** (`heads.change_map: null`). G2 = TSP + fovea reconstruction. No change targets are read.
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
  keeps one fovea-sized window per frame (the agreed pixel budget) but places it where the eye did not necessarily look:
  flagged for review.
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
  natural floors.
Reported beside G1 at pure noise and the change target from the earlier gate (same scenes, same states: reproduced and
asserted). R is a discrimination diagnostic, not a go/no-go (weight-invariant). What would change F5: if the TSP
forecast prices both consequences below G1 at pure noise relative to the floors, or its foils out-price the
consequences in most scenes, we discuss dropping the TSP forecast from F5 and keeping the outcome forecast.

## F5 judged on (fixed before launch; readout plan reviewed with the launch)

- Native exact-history timing under the full cache: hunger pairs and natural thirst alias, due vs not-due rates
  separately (truth 1 / 0); zombie readiness pairs, hit when ready vs cooling.
- The standard readout against F4 and HUD-Change: S1 entity retention (tests the small-entity hedge the change map was
  for), consequence determinant curves and crafting, actor interface, 4-step recursion, belief probes (timers, sleep).
- The forecast heads' own held-out scores, reported but not used as success: an improved forecast without native
  generated improvement is not a repaired world.
