# F2 review — chooser first, 2026-11-04

User requests a meticulous review of the completed F2 run, especially actual
looking behavior and surprise-to-contingency development. Previous fine-path
proposals are set aside. No new learning ablation, world/head refit, architecture
change, P2, or launch belongs to this review.

Original contract read first: 20261103_p1f2/PLAN.md and chooser.json,
TEACHING.md sections15, E1–E3, chooser-history decision and F2 preparation;
actual controller, prediction head, streams, evaluation scripts and outputs.
Endpoint summaries were inspected before this review scope was written. The
following are descriptive analyses, not claimed as preregistered efficacy tests.

Bounded work:
1. Verify actual F2/F1.6 configs, source pins, budget, checkpoint selection and
   evaluation completeness. Preserve source or instrument discrepancies.
2. Re-read the complete saved training trace. Separate where crop centers land
   from what their 3x3 crops include; pair actual gazes with the original uniform
   draws on the same frames. On the already labelled train400 subset, report
   map/HUD, each entity, action consequences, entering terrain and changed HUD,
   day/night and training deciles. Repeats are grouped by physical source episode,
   not treated as millions of independent examples. No true-state label enters
   the controller. Background looks are not automatically noise/waste.
3. Replay the TEN saved online heads with their OWN encoders on the SAME existing
   saved/confirmation dev cohorts. No refitting or best-checkpoint selection.
   Track combined-policy remote/contact coverage, component rankings, target-
   counter-only C/S, factual counter error versus copying, and actual increment
   energy. Interpret raw energies across changing encoders cautiously; normalize
   by same-checkpoint target-change energy and report actual event counts. Zero
   C on NOOP is a tie, not learned upper-left targeting. Endpoints must reproduce
   the queued fixed-head evaluation, with numerical discrepancies retained.
4. Dissect the already queued world evaluations beside F1.6, class by class,
   real versus generated and fixed random/guided gaze. Retain regressions,
   weak-reader domains, sample counts and intervals. Chooser acquisition,
   world teaching, native fine use and actor utility remain different claims.

Whole-screen held-out behavior uses the natural random-frame cohorts, never
the event-balanced benchmark as a population distribution. C and S alone are
component diagnostics, not proposed replacement policies. No policy/weight
search, new night-noise campaign or sequence of follow-up ablations follows.

Final report: actual run lineage, interpretable plots, numerical evidence and
limits. Return for discussion. No job is queued after the bounded replay.

Additional existing-data read, declared before its values: the queued Codex panel
already contains eight independent real renders of each of16 bright and16 dark
fixed states. Reuse that bank with the frozen final head and its unchanged real
history to quantify fresh-render sensitivity of S and the combined probabilities.
No new simulator source or noise intervention is collected. Report the finite-
bank variance decomposition and policy TV, not a population wasted-gaze rate.
