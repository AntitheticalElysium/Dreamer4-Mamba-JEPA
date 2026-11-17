# Does a horizon target give the pixel clocks a share? (2026-11-14 night, data only)

Candidate (from the research pass, not adopted): predict the future over a horizon, discounted and averaged
(successor features of the periphery: (1 - gamma) sum_k gamma^(k-1) x_{t+k}), instead of the next frame alone.
Reasoning: a lawful state change persists (the food digit stays changed), unpredictable flicker reverts (night
noise), so averaging over the future keeps the first and divides the second's variance by about (1 + gamma) /
(1 - gamma); and every frame of a clock's interval, not only the due frame, depends on its phase. Precedent:
GVF networks, predictive-state decoders, multi-step CPC for belief states, gamma-models and TD flows.

Claim: under such a target the food / drink / energy digits' share of the residual (what transport cannot
predict) rises well above the ~1% they hold in every next-frame loss we measured. Competing explanation: the
horizon target adds as much unpredictable content (future movement, entering terrain, mobs) as it removes, so the
share stays near 1%.

Reading (`horizon.py`, CPU, 400 DEV episodes; cells at the periphery's own acuity: each 7x7 cell average-pooled to
3x3, 27 numbers, as the encoder sees it): for gamma = 0 (next frame), 0.8, 0.9, 0.95, the residual energy of the
target against two baselines that carry no memory: copy the present frame (future movement unknown) and transport
the present frame along the recorded displacement (future movement known; screen-fixed rows not moved). The
displacement and the split into map / HUD cells are used only to build the baseline and to attribute the residual,
standing in for what the learned denoiser already does (moved-map error 0.052 against copying's 0.258). Shares:
map cells by day and by night, the need digits (food, drink, energy cells), the same on frames whose horizon holds
need drops and no rises (the timer-only part), health, other HUD cells.
