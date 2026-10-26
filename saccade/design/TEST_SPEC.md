# Staged bounded test v1.1 (2026-10-27) — M0 first, increments as diagnosed comparisons

What it must show: the learned belief + denoiser preserve (1) entities and coupled events through recursive
generation, (2) places and resources through revisits, (3) uncertainty through weak observations and returns, (4) the
information decisions need. A downstream failure does not identify its component (perception, memory, generation or
decision); probes locate it.

Rules
- Readings, budgets (updates, seeds) and probe definitions fixed before each stage; simulator state for EVALUATION
  only; dev split for readings, final split untouched until the end.
- Gate rule (saccade/eval/gates.compare): reject when |difference| > 3 joint episode-bootstrap SE + 1e-9, so a
  deterministic difference with zero estimated variance rejects; a gate without exposure is reported as unmeasured,
  never as passed. Classifier two-sample tests split by physical episode, both domains of an episode on one side.
- Absolute metrics with uncertainty (episode-bootstrap intervals), not ratios to "floors" or "ceilings": references
  (random features, a pixel probe of the SAME architecture) are reported beside them.
- One frozen reader (a probe trained on real percepts against simulator labels) reads real and generated percepts
  alike; its own error against simulator labels is reported, and its outputs on generated percepts are diagnostics.
  The reader is fitted to a SPECIFIED encoder checkpoint (the end of P1); the encoder is frozen during P2, otherwise
  representation drift would masquerade as lost information. S0 builds the harness and the pixel references; the
  reader itself is fitted after P1. A shared reader makes real and generated readings comparable; its errors need not
  cancel under the shift to generated percepts, and its real-data error does not bound its error on generated ones.
- Bug fixes are allowed and documented; a scientific miss is kept as a result and leads to a bounded, declared
  revision, never to tuning until the gate passes.

## S0 — harness (no model)
- References on dev: conditional event probabilities (arrival given light and edge cell, attack given adjacency and
  time since last attack, arrow given a skeleton at range, vanish given position) and re-entry gaps, with intervals;
  revisit frequency by time since last observed; all measured through the same observation path the model gets.
- Evaluator power: every gate must REJECT deliberately broken predictions — shuffled histories, mismatched actions,
  duplicated entities, misplaced consequences, frozen time. A gate that cannot is not used.
- Not used as gates: A10 (LightGBM; the new probe architecture needs its own pixel reference) and A3 (its helper lacks
  the attack stop; A14b's corrected helper is the one to reuse).
- DONE 10-27 (artifacts/experiments/20261027_s0_harness/RESULTS.md; code in saccade/eval/): one-step gates (43; real half passes, five corruptions rejected),
  revisit references (terrain static: agreement >= 0.9992), multi-step gates with a full-state simulator oracle (six
  corruptions rejected; m = 8 per window, all 2,236 exogenous 16-step dev windows), the per-cell probe definition and
  its pixel reference (day perfect; night 0.85-1.0 by ring), the outcome evaluator with an information ladder.
  Every reference is recomputed THROUGH the frozen reader on real trajectories at the end of P1 and the power checks
  repeated; the energy score is reported beside the oracle and marginal references, never as pass / fail.
  Reviewed by Codex 10-27; evaluator defects fixed and pinned by regression tests (saccade/tests): zero-variance
  rejection, a cardinal-motion gate, revisit ages from the compared sighting, physical-episode classifier splits,
  power of the operational one-step gates on multi-step rollouts.

## S1 — M0 one-step (phase P1 of CURRICULUM)
1. Entity retention in percept tokens: presence, kind and position per cell, by light and ring, beside random-feature
   and pixel-probe references.
2. Belief contents: unlocked achievements, HUD values, entities that just left the view. Controls: the current
   observation INCLUDING reward and termination; the 4-percept window without Mamba (isolates what Mamba adds).
3. One-step generation: classifier two-sample tests on MATCHED (history, action, fixation, outcome) tuples, with power
   checked on the S0 corruptions; entity counts read by the frozen reader; reward and continuation calibration.
4. TSP: glimpse likelihood vs copy-previous and per-position-mean baselines.

## S2 — M0 recursion (phase P2; free-running rollouts on dev exogenous windows)
1. Conditional event probabilities vs S0 references with intervals (arrivals, vanishings, cardinal motion, arrow
   after a shot, damage timing after adjacency); duplicated and missing entities read by the frozen reader.
2. Revisits: returns within the rollout AND to locations seen in the real prefix, stratified by time since observed
   (<= 16, 16-64, 64-256, > 256).
3. Uncertainty: proper scores (energy score), bias, coverage of real outcomes, and coupled-event checks; spread alone
   is not a pass.
Coverage limit: 16-step exogenous windows come almost only from random play.

## S3 — decisions (phase P3)
1. Imagination-trained actor vs BC with matched inputs and the SAME fixation controller (fixation learning evaluated
   separately afterwards); >= 3 independent training seeds, paired environment seeds, intervals across training
   runs. Its budget is reserved from the start.
2. What blocks S3 is declared now: an S1.1 retention failure or S2.1 duplicate / missing entities beyond the declared
   bound; other misses are reported and S3 still runs.
3. Learned fixation afterwards: anticipation vs reaction; night robustness.

## Increments are candidate comparisons after diagnosis, not automatic responses
- Wrong damage timing can come from perception, forgotten cooldown or a denoiser ignoring retained information; a bad
  revisit from forgetting or from generation not using what was retained. The probes decide which; only then is I1
  (store + self-location) or I2 (records) the candidate, and its benefit is claimed only by a matched retraining.

## Open
Budgets per stage; the belief-content probes (S1.2); the declared bounds for the two S3 blockers. (Fixed in S0: the
per-cell probe for S1.1, the S2 sample counts.)
