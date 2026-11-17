Frozen descriptive study, not a causal training comparison. Checkpoints 1,3,5,6,8,10 in HUD and MultiStep
(five ends the 64-frame phase; six is after the 256-frame phase starts). One fixed 192-episode DEV bank,
same fixation draws and all true prefixes across checkpoints, four episode folds. Read corrected and
post-action beliefs, before and after conditioning on a fixed recent *semantic* baseline (body, inventory,
sleep, action, rewards, lighting over eight steps). That baseline is not the complete visual cache.
Use standardized ridge throughout and a shared nonlinear reader for post-action belief plus recent cues.
Compare same metrics within this study, not against different cohorts in prior reports.

Clock cohort: awake, non-extreme body value, latest visible decrement >8 steps ago, awake throughout since
that change. Evaluate exact hidden phase on the same rows and report agreement of the visible-reset-age
positive control; do not select rows by successful decoding. Sleep duration >8 is a separate control.
Read last-seen zombie side after 9–64 steps absence, and two already-earned achievement flags.
A failed reader is not proof of erased information. No new native generation run unless an unexpected
qualified gain would materially change the prior native verdict.

Separate gradient check: original vs rebased split, same sampled real chunks, noise, weights, horizons and
carry; compute individually weighted parameter gradients on shared belief at checkpoints 1,5,10. This
isolates the immediate correction and objective's gradient, not the causal value of learning with it.

Post-trajectory qualification: added the two exact untrained constructors to test whether the early
sleep-duration readout gap already existed at initialization. Same bank, readers and episode folds;
no further model fit. Plot shows the maximum of the stated awake-clock readers (a tested upper bound,
not the average or a claim about all possible decoders). No unexpected clock gain triggered further generation.
