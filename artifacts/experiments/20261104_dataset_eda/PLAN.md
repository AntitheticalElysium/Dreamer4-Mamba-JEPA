# Full candidate dataset EDA — 2026-11-04

User requested the entire dataset and evidence-based expectations before tonight.
Descriptive CPU census, no model fit, training run, sampler/acuity change or
architecture choice. Inspect original support-v2 and both fresh stores. Exclude
the fixed monitor episodes from all training figures. Keep original dev/final
unchanged. Collection is still running at initial inspection; pin committed
manifest snapshots and refresh the final census after its clean completion.

Questions fixed before the expanded census:

1. How many independent episodes/transitions/terminal events, and how much of
   the stream has short versus long history? How do complete source episodes
   differ from actual per-frame training frequency?
2. All22 achievements: unique unlocks, episode support, day/night, and source
   policy. Separate unlocks from repeated inventory/body consequences. Break
   increments down by previous numeric value; report remaining holes.
3. Damage/death: health, night, sleep, empty food/drink, projectiles, cooldown,
   equipment, action and survival controls. Contexts overlap and do not attribute
   causes. Compare the same exact definitions across all sources.
4. Creatures and terrain: actual visible creature instances, near-player/night
   exposure and first-visible transitions; terrain view-cell composition and
   camera-aligned edits. Background terrain can be occluded; do not call all
   labeled terrain fully visible pixel evidence.
5. Action support in tool-qualified resource-facing states: DO, NOOP, others;
   success versus no increment, actual random-branch outcomes. Behavior support
   is not matched randomized causal identification by itself.
6. Exogenous2/4/8/16-step windows, including true-terminal absorbing suffixes but
   excluding unavailable truncated suffixes. Report unique covered events and
   independent episodes alongside overlapping window counts. This is P2 support
   accounting, not adoption of a new P2 objective.
7. Approximate training exposure at the existing25.6M-frame budget. Distinguish
   independent data, empirical frequencies and optimization repetitions. Raw
   concatenation is a candidate accounting baseline, not an approved mixture.

Checks: reconcile the original census and every new manifest's event totals;
assert state/transition alignment, held-out exclusion, terminal handling, positive
and negative event support, and a synthetic terminal-window check. No claimed
world-model improvement before training. Predeclare likely beneficiaries,
remaining weak points and unchanged architectural limitations from the census.
