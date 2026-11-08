# Search for a learned behavior with a useful fine cue — 2026-11-04

User explicitly requests candidates satisfying BOTH native learned competence
and incremental fine evidence; stop honestly if none qualify. Frozen F1.6ck10,
SHA34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045.
Do not interpret F2 results, train a world/probe, alter sensors, or queue a new run.

## Bounded source search, before native consequence outputs

Existing first400 dev episodes and checkpoint-bound instruments/cached percepts.
1. Counter/resource prerequisites: inspect exact current HUD readability and
   native learned conditional behavior. If coarse tokens already recover the
   prerequisite, it is not a fine-required cue. No artificial coarse masking.
2. Requested player movement blocked by a cow/zombie/skeleton. The cue is CURRENT
   occupancy of the requested adjacent cell, on otherwise walkable terrain.
   Heading, entity health and attack cooldown do not determine this event.
   Awake, alive, action1..4. Exclude arrow-overlaid cues, inconsistent mob_map,
   non-walkable terrain, and unobservable camera translations.
3. Deterministic resource interactions (DO at water/tree/stone) and static
   movement blockers: source-local terrain identity. Exclude occluding entities;
   the TASK fact, rather than arbitrary class confusion, must be misread coarsely.
   Use fixed coarse terrain reader, existing fine readers/decoder only.

All source counts retained. For each family with source coarse misses, at most
12 different-episode coarse-missed candidates and12 different-episode clear
references, plus documented rejections. No selection on generated predictions.
Fine acquisition requires present/alternative discrimination, not just correctly
calling the factual scene. Existing readers' failure is not proof of absence.
Inspect recent coarse/fine history rather than assume present fine is necessary.

## Qualification before route interpretation

- Simulator replay exactly reproduces selected source and actual successor.
  Same action/key with cue removed/changed must change the named event. Keep all
  other state fixed except the bookkeeping needed to represent a removed mob.
  Counterfactual edits are valid renders but not guaranteed a shared feasible past.
- Native competence must distinguish BOTH alternatives on clear references:
  production3-step/churn1,32 paired samples per case. Require mean correct event
  classification>=.75 in EACH branch. A high achievement score on successes,
  low latent loss or successful copying alone does not establish this skill.
- Event instruments must distinguish both real successors. For movement, read
  camera translation from the fixed terrain reader on discriminating static
  terrain cells; require>=8 such cells and correct classification of both real
  successors. No HUD/entity locations enter this movement score. Future labels
  are for scoring, never inputs to the world.
- Only families passing the competence and fine-evidence qualification proceed
  to full causal route tracing. Use genuine focused/disjoint gazes and same-gaze
  semantic fine swaps under both coarse contexts,64 common random futures. Full
  real prefix, no future evidence, fixed query, full Mamba carry, cache invariance,
  and pair/episode-level reporting as in the preceding campaign.
- Source coarse-reader mistakes are operational evidence, not proof of complete
  absence of the fact from coarse/history tokens. If that is the only support,
  preserve the limitation and do not call a null definitive memory erasure.

If no candidate satisfies the conjunction, stop at qualification and report
which premise fails for each family. No successor repair or diagnostic ladder.

## Source-driven extension, before native consequence outputs

The local source catalogue finds zero task-relevant coarse misses in17518
frames. Therefore expand the SAME qualification search to newly visible distant
content, where the known acuity gap actually occurs:

- Cow/zombie/skeleton one-step semantic persistence, scoring presence anywhere
  in that entity's one-step reachable neighborhood rather than an exact sampled
  destination. Source alive/awake, non-DO action, light<.5, Manhattan distance>=3,
  no other same-kind source entity within3 cells. Require both fixed fine readers
  correct on present AND removed alternatives, and source coarse reader calling
  background. Report previous8 visibility; prioritize no-visibility cases.
- Static terrain semantic persistence: tree/stone/coal/iron/table/furnace at
  distance>=3, source not occluded, non-DO/non-placement action, light<.5. Count
  source coarse misses across all6 classes; qualify at most the2 classes with
  most missed source episodes, selected by SOURCE counts only. Existing native
  fine reconstruction decoder must distinguish both valid rendered alternatives.

At most12 different-episode missed and12 clear references per family, source-
only selection. Keep failed fine or predictive-competence qualification. Presence
at the next step is a world-state fact, not evidence for action consequences or
decision value. The native skill gate must pass BOTH alternatives before tracing.
This extension is justified by local input redundancy, not by a native model
prediction result. No additional candidate family after this bounded search.
Before any native outcomes, restrict the source catalogue to non-sleep actions
0..4. For mobs, EVERY one-cell move must remain in view under both a successful
and blocked player move; for terrain the source world cell must remain visible
under both. This avoids mistaking uncertain visibility for failure of persistence.

Before native outputs: run capability gates only for families with at least one
source case passing fine acquisition, alternate coarse readability, and future
instrument qualification. This avoids testing skills that cannot serve the
requested conjunction. The .75 mean-correct criterion applies separately to
present and absent branches, over every qualified clear reference; report the
number of cases and case-bootstrap intervals, not a binomial interval treating
all draws as independent episodes. No outcome-based individual case selection.

Post-result instrument audit (not a new family or a revised selection gate):
night noise is resampled in the native renderer with the actual successor world
held fixed,64 keys per branch, for the24 clear terrain references and12 traced
cases. This checks whether one successful real-successor reading overstated
the instrument's expected recall at night. Preserve all primary results.
Low recognition on true worlds limits interpretation of generated-event rates.
No further family search, model repair, or training follows this audit.

Final source qualification, declared after the routing result: compare the
existing native pre-glimpse TSP reconstruction with the same two source
alternatives, using the common real-prefix prior and current periphery only.
This checks whether external-reader misses actually conceal a distinction the
model already knows. Report EVERY traced case and preserve the original
cohort/results; no new case selection or follow-up training. The pixel metric
is an operational contrast readout, not a complete test of latent information.
