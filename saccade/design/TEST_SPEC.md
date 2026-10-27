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
  The reader is fitted to the encoder checkpoint selected at the end of P1 by CURRICULUM's fixed monitor rule;
  the encoder is frozen during P2, otherwise
  representation drift would masquerade as lost information. S0 builds the harness and the pixel references; the
  reader itself is fitted after P1 and saved with its schema, seed, label hash and encoder-checkpoint hash.
  Real and generated reader inputs use the same FP16 storage precision. A shared reader makes real and generated readings comparable; its errors need not
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
2. Belief contents (specified 10-27, before any P1 result). Features: the corrected belief b_t (Mamba output after
   the percept, reward and termination of t: what the actor reads). Probe: the S0 family (MLP 256-256 GELU), fit on
   train-400, read on all dev, 2 probe seeds, episode-bootstrap intervals; reported, not a gate. Targets:
   (a) achievements unlocked before t (22 binary): AP and balanced accuracy each;
   (b) the displayed HUD quantities (health, food, drink, energy; the 12 inventory counts, capped at 9): exact
       accuracy each; read beside control (i): at 3 x 3 acuity a supervised reader still reads HUD values from the
       periphery (0.9999), so this is not automatically a memory test (Codex);
   (c) recent sightings: per kind, none currently visible but that kind was visible 1-64 steps ago, by absence
       stratum {1-4, 5-8, 9-16, 17-64} (beyond 8 the denoiser's own context cannot hold it). No hidden survival or
       identity condition. Side is the nearest screen edge at the LAST VISIBLE sighting (4-way), not an unseen exit.
       Multiple same-kind sightings choose the one closest to an edge; ties use row-major cell then top/bottom/left/right.
       The player-covered centre cell is excluded, as in S0. Reappearance resets the kind-level absence clock.
   Controls with the same probe: (i) the current percept (periphery, glimpse, fixation, lighting) with its reward
   and termination; (ii) the last 8 steps with everything the belief saw in them (percepts with fixations and
   lighting, actions, rewards, terminations: the denoiser's effective reach), without Mamba. A gap measures accessible
   readout, not an information bound (Codex).
   What Mamba adds = belief minus (ii), read per stratum.
3. One-step generation: classifier two-sample tests on MATCHED (history, action, fixation, outcome) tuples, with power
   checked on the S0 corruptions; entity counts read by the frozen reader; reward and continuation calibration.
   Classifier inputs include every recent periphery/glimpse/lighting percept, fixation, action, reward/termination,
   predicted belief, validity flag, plus current predicted belief/action/fixation; current target outcomes are excluded
   from the conditioning side. Both outcome domains are rounded to FP16. Power donors stay within their train/test
   episode partition (train, validation and test). C2ST reuses the existing MLP 256-256 GELU and probe fitting
   schedule (seed 0); column means/std use training examples only (std floor 1e-3), then history/current-belief/action/
   fixation/outcome groups are scaled to equal expected energy. Raw-coordinate trees failed the matched-marginal
   dependence control; the neural classifier must reject it and the declared power corruptions. Unlock calibration uses the same event as the S0 labels, round(reward - 0.1 * health change) = 1;
   evaluation-only health change selects the corresponding reward bins, never enters world-model training.
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
   separately afterwards); 3 actor and 3 BC seeds per world seed, paired environment seeds (512 DEV), intervals across
   training runs, reported PER WORLD SEED (two independently learned worlds, not six replications). Its budget is
   reserved from the start (CURRICULUM, Budgets).
2. Blockers (agreed 10-27), evaluated ONCE on the declared checkpoints, per world seed, per entity kind (zombie, cow,
   skeleton, arrow); a pooled check passed the removal of every arrow (-4.2% count, persistence up), so none is pooled:
   - perception (S1.1): token probe (S0 definition, mean of 2 probe seeds, seed spread reported) on rings 1-2, day and
     night separately: blocks if a kind's presence recall or precision falls more than 0.10 below the pixel reference
     AT THE PERIPHERY'S ACUITY (3 x 3, rings 1-2, mean of 2 probe seeds; recall by day 1.00 / 1.00 / 0.999 / 0.998,
     at night 0.965 / 0.959 / 0.901 / 0.767; precision by day 1.000 for every kind, at night 0.999 / 0.994 / 0.993 /
     0.977, for zombies / cows / skeletons / arrows; acuity_precision.json, PERCEPTION acuity section), or below an untrained copy of the same encoder by > 0.02 beyond 3 SE
     (learned erasure). The same 0.10 tolerance as the earlier absolute 0.90, now measured from what the input allows
     (10-27: the absolute bound would have fired on night arrows from acuity alone).
   - bookkeeping (S2.1): frozen reader on prior-sampled rollouts (all 2,236 dev exogenous windows, m = 8) vs the same
     reader on the recorded futures: blocks if a kind's mean visible count differs by > 30% AND by > 3 SE, or if 4-step
     persistence of zombies, cows or skeletons falls > 0.10 below the recorded (0.996 / 0.978 / 0.906 on labels; arrows,
     short-lived at 0.32, get the count check only). Support in those windows: 584 / 1,566 / 339 / 244 windows.
   The S0 corruption checks, including all-arrows-removed, are rerun through the reader before use. Other misses are
   reported and S3 runs. If both pass, S3 runs at once with its reserved budget; if one fires, the miss goes to
   diagnosis (below), never to longer training. Any missing required exposure returns unmeasured, preventing a false pass.
3. Learned fixation afterwards: anticipation vs reaction; night robustness.

## Increments are candidate comparisons after diagnosis, not automatic responses
- Wrong damage timing can come from perception, forgotten cooldown or a denoiser ignoring retained information; a bad
  revisit from forgetting or from generation not using what was retained. The probes decide which; only then is I1
  (store + self-location) or I2 (records) the candidate, and its benefit is claimed only by a matched retraining.

## Open
P2/S2 implementation and its gradient checks are deferred until the S1 reading. S1 probe definitions, budgets and
S3 blocker bounds are fixed above and in CURRICULUM.
