# Bounded prelaunch checks for decision 2

2026-11-02. Approved for diagnosis, then exercised under UPTAKE_PROTOCOL.md.
Results: [UPTAKE_RESULTS.md](UPTAKE_RESULTS.md). No controller or new training recipe selected here.
These qualify the chooser's signals, behavior and implementation. They do not prove long-run teaching benefit.
Use native full models and real replay; no frozen toy learner and no simulator labels as chooser inputs.

1. **Signal reliability and allocation.** Fixed real scenes and full prefixes at early/mid/final native checkpoints.
   Repeat first-call predictions with common noise for actual/NOOP actions, and repeat renderer static at the same
   underlying scene. Measure selection probabilities, noise sensitivity, consequence/threat coverage and allocation
   to well-predicted regions. This informs estimator draw count, normalization and the role of persistent action
   contingency; it does not identify optimal learning weights or prove learning-progress selection.
2. **Actual sequential gaze behavior.** Replay complete real episodes with the candidate chooser, broad random
   floor, and a predeclared comparison with/without the proposed recency trace. Each policy gets its own legitimate
   glimpse and belief history. Measure repeat streaks, spatial coverage, revisit timing and event/night allocation.
   This can expose fixation traps and assess a trace's behavioral effect, not establish its training benefit.
3. **Localized useful-evidence uptake.** On real scenes where the periphery misses an entity, first verify that
   the fine encoder exposes it. Compare looking there with a matched control crop, same physical history/action,
   and score local next-step denoising rather than only whole-map loss. A positive control is required; otherwise
   a null cannot diagnose memory uptake. This tests current use of evidence, not future curriculum benefit.
4. **Full-path preflight and throughput.** Once the controller contract is settled, test the actual P1 objective,
   every gradient route, one belief advance, cache invalidation, independent chooser RNG, exact fixation-history
   replay, and identical chooser input versions in real/generated sensing. Disabled chooser must reproduce the
   baseline. Benchmark actual collection plus training before committing to an overnight frame budget. Short
   optimization smoke checks are mechanical checks, not a scientific improvement gate.

Predeclare scenes, candidate policies, readouts and budgets before execution. Keep a held-out replay subset for
the final behavior check. Preserve negative/null results. Overnight F2 alone tests whether the selected curriculum
improves full training relative to F1.5. Corpus, losses, architecture and primary sampler remain matched.

Separately audit existing advanced-state random-action support and terminal-window admission for P2. Any new
dataset or outcome-objective change remains outside tonight's chooser comparison.
