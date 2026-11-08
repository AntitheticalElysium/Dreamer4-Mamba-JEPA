# P1-F1.5 readiness diagnosis

2026-11-02. Owner: Codex, at the user's request. This is a diagnostic protocol, not a new training recipe.
The controller, loss changes and P2 launch remain discussion points. Preserve the completed negative results.

## Questions and separating measurements

1. **Attribution:** P1-F changed both guided fixation and joint generation relative to F1.5. The missing full
   training arms are joint/uniform/faithful and factorized/guided/faithful. No percentage attribution is possible
   from the two endpoints. An inference-time mask toggle is not a substitute for those training arms.
2. **Fovea gradients:** distinguish crop exposure, weighted native loss gradients, and useful evidence uptake.
   Use native checkpoints, carries, losses, harmoniser and data. Compare checkpoint 5 and 10 of P1-F and F1.5
   on matched scenes, with each world's native fixations and history. First validate the instrument on two
   streams; then use up to eight disjoint two-stream blocks, 64 new frames each, common random seeds.
   Split G1 into periphery/glimpse/lighting without changing its sum. Report weighted gradient norms and
   signed projections onto the full gradient by module. Norm shares are not causal importance. If needed,
   compare exact frozen one-step Adam updates with and without fine-view terms, including clipping and moments;
   this is a local optimizer differential, not evidence of long-run benefit.
3. **Generated evidence versus readout:** use each world's own outcome head on its own real/generated evidence.
   The real-evidence control checks recognition. Keep outcome-selected achievement cohorts separate from
   pre-outcome risk cohorts and natural random scenes. Keep unknown external-reader outputs in denominators.
   For F1.5 test production e3c1 versus e8c0 and e32c1, four independent futures, first 200 dev episodes,
   at most 256 scenes per cohort (random, risky, wake, coal and iron). Exact reward-marginal death readout.
   A solver change tests a numerical repairable component at this frozen checkpoint, not curriculum attribution.
   Follow-up selected after seeing the solver readings: test the encoder's exact token constraints on real and
   generated tokens (inverse tanh has approximately zero channel mean and unit second moment before clamping).
   This is independent of any semantic reader and can establish a coordinate-distribution mismatch, not its
   decision cost or a valid fix. Do not project/renormalize predictions as an unreviewed intervention.
4. **Own-history robustness:** if the above instruments pass, use at most 128 canonical exogenous windows,
   four futures, horizons 1/4/8/16. Compare free-running with real-history conditioning. Full real prefixes,
   native reward/termination sampling, one belief advance per frame, absorbing termination. Compare conditional
   rates/distributions; do not score independently sampled futures pointwise against a single recorded future.
   After observing free-running count decline, add a mechanical real-percept replay control: use the same
   stepwise belief-update code as free running but feed recorded percepts/outcomes. Check its predicted beliefs
   against full-episode encoding. Preserve the initial outputs. This separates an update-code discrepancy from
   generated-input drift; it does not validate semantic readers on arbitrarily drifted tokens.
5. **P2 credit and coverage:** inspect the actual proposed objective, detached discrete outcomes, last-two-step
   gradient cutoff, and exogenous data coverage. Distinguish a proper population score from the gradient actually
   optimized. Read JEDI, Dreamer 4, DIAMOND, Self Forcing, Pacchiardi and NeuralGCM at the relevant claims.
   Follow-up chosen after the pilot had no recorded deaths: count termination by position in all eligible
   complete windows, including all sliding starts. Check whether the window-admission rule conditions on future
   survival before interpreting rollout scores as calibrated prediction. This audits a proposed P2 dependency;
   P2 has not run and cannot explain the current P1 checkpoint's regression.

## Execution and interpretation

One GPU job at a time; inspect processes before using it and do not interrupt Claude's work. No training run or
bundled repair. Save source/checkpoint identities and raw outputs under this directory. Bounded diagnostics may
be shortened for resource limits; report the actual scope and do not promote an unrun endpoint to a result.
The living synthesis is `saccade/design/P2_READINESS.md`. Confirmed measurement corrections also go into the
original results and TEACHING; historical raw outputs stay unchanged. Scientific decisions return to the user.
