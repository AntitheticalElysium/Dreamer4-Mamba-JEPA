# Learned behavior with a useful fine cue — 2026-11-04

**Verdict: this bounded search did not find a clean case satisfying BOTH
learned native behavior and a demonstrated need for incremental fine evidence.**
Stone and trees were much better positive-control families than arrows: the
model predicts them well when the input is clear. However, native pre-glimpse
TSP already distinguishes both alternatives in10/12 of the challenging terrain
cases. In the other2, the particular outcome associated with TSP's mistake is
already generated correctly about95–97% without looking at the cue. An external
coarse-reader mistake was an inadequate criterion for fine necessity.

Thus the terrain routing null is NOT evidence locating a fault in the mixer,
Mamba, or denoiser. It also does not establish that fine sensing never helps.
The result supports the concern that our selected probe targets often repeat
information already recoverable through the coarse/belief path.

Frozen F1.6ck10:
`34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045`.
First400 existing dev episodes, no final split, no world/probe training, no
architecture or sensor change. F2 scientific results were not opened.

```text
Candidate                                    Missing qualification
HUD / nearby resources and blockers -------- coarse already supplies task facts
Distant creatures -------------------------- native competence not established
Stone / trees -- good native clear-input skill
                 -- fine distinguishes the cue
                 -- but native pre-glimpse TSP usually distinguishes it too
                               |
                 no clean fine-dependent positive control found
```

## 1. Source search, not outcome-based case picking

Local action catalogue:17,518 eligible frames, comprising11,598 DO-facing
water/tree/stone frames,1,356 movement attempts into an otherwise walkable cell
occupied by one cow/zombie/skeleton, and4,564 terrain-blocked movement attempts.
The existing coarse reader had zero TASK-relevant misses in those frames.
For mob blockage the fact is occupancy, not exact species; for static blockage
it is walkability. This is not a claim that every adjacent semantic distinction
or every successful action is learned. Current occupancy determines player
blocking before mobs move; attack heading/cooldown is not required for this cue.

HUD audit:4,388 deterministic sampled/value-stratified natural frames ×16 fields
=70,208 values. Native renderer glyphs matched ALL natural HUD patches exactly;
enumerating their frozen native periphery codes under each frame's lighting
recovered ALL tested values from the saved actual coarse tokens. No fitted
weights. The smallest runner-up squared-distance gap across the fields was
0.0460. This is a finite-sample information-preservation check, not proof that
Mamba or the actor uses the values. It does not cover unobserved values or
achievement flags, which are not displayed in these HUD patches.

Distant-night source catalogue, awake/alive, action0..4, source distance>=3:

| family | eligible cell/entity-frames | coarse-reader misses | missed episodes |
|---|---:|---:|---:|
| stone |141,255|1,938|181|
| tree |27,313|468|123|
| cows |2,044|90|45|
| zombies |1,993|68|41|
| skeletons |988|47|32|

Coal, iron, table and furnace were also counted; the predeclared source-count
rule chose the2 terrain types with most missed episodes, stone and tree.
Maximum12 different episodes per family × missed/clear stratum,120 source pairs.
Source selection used source data/readers only, not generated predictions.

Exact replay reproduces all120 selected source and factual-successor images.
The alternate world removes the selected mob, or replaces the terrain cell with
grass, with the same action and environment key. Simulator checks verify the
named future event differs and the two player positions agree. These are valid
renders; the alternate current frame is not guaranteed a shared feasible past.

## 2. Fine acquisition and instruments

Each alternative must be distinguished, rather than correctly classifying only
the factual image. Existing2 fine-kind readers are used for mobs. For terrain,
the native fine reconstruction is compared with both real alternatives on the
pixels that differ. Both native future images must also be distinguished by the
fixed physical-percept reader. Fresh encodings verify the cached coarse miss;
4 source cases changed that classification under fresh inference and are not
treated as misses. Maximum fresh/cache token difference0.02283; no byte-parity
claim. All clear references are freshly checked too.

| family | missed-source cases passing all acquisition/instrument checks |
|---|---:|
| cows |7|
| zombies |6|
| skeletons |7|
| stone |9|
| tree |3|

These qualifications do NOT establish absence of information from coarse
tokens or history. They merely qualify a practical candidate and its readers.

## 3. Does the native model already perform the behavior?

Production3-step/churn1 sampling,32 samples per branch,12 clear-reference cases
per family. Both event-present AND event-absent branches must average>=75%
correct. Creature event: matching species somewhere in its one-cell reachable
region; all those cells stay visible under either player-motion possibility.
No demand to predict the exact stochastic mob destination. Terrain event:
correct semantic content at the world cell's next screen location.

| family | present predicted correctly | absent predicted correctly | gate |
|---|---:|---:|---|
| cows |41.41%|100%|fail|
| zombies |14.32%|100%|fail|
| skeletons |16.41%|100%|fail|
| stone |90.63%|100%|pass|
| tree |97.66%|100%|pass|

Native generated-percept/readout competence for creatures is not established;
we do NOT run their route tests. This does not localize their failure or prove
the semantic reader is perfect over generated distributions.

Stone present case-bootstrap95% interval74.74–100%; tree93.23–100%. Stone is
heterogeneous:1 clear case scores6.25%, the others84.38–100%. The rule was a
family-mean gate, not selection of individually successful generated cases.
This is one-step semantic persistence, not an action-consequence or actor-value
result. All60 clear cases and their3,840 futures are retained.

## 4. Same route test on9 stone and3 tree cases

64 common random futures per condition. Genuine focused versus disjoint gazes
in each of the2 worlds; separately, same-gaze semantic fine swaps under EACH
coarse context. Real complete prefix, full Mamba carry, fixed next query, shared
context/sampling randomness. Physical temporal keys/values are exactly equal
within each fine intervention pair. Thus raw fine temporal attention does not
leak into the tested physical-percept route. Prefix replay RMS is<0.03 in every
case (clear-reference maximum0.01135). No future evidence enters the model.

| case family | present: focused | present: disjoint | absent: focused | absent: disjoint |
|---|---:|---:|---:|---:|
| stone,9 cases |55.03%|55.21%|81.42%|81.25%|
| tree,3 cases |3.125%|3.125%|98.96%|98.96%|

Mean balanced event-correctness change from focusing:0pp for both families.
Stone case-bootstrap95% interval−0.35 to+0.35pp. This conditions on the finite
sampled seeds/cohort; it is not a universal bound on gaze value. Tree has only3
cases and zero paired classification changes; a zero bootstrap width is not
proof the true effect is exactly zero.

Same-gaze semantic fine swaps have a balanced expected-direction effect of
+0.087pp for stone (case-bootstrap interval−0.35 to+0.61pp),0pp for tree.
Only9 of1,152 paired stone classifications change under content swaps;0/384
tree classifications change. Continuous reader scores also move slightly; these
are classifier scores, not calibrated world probabilities.

All24 content contrasts change the projected input and complete recurrent SSM
carry/emission; no exact collision. Projected-input RMS difference0.000606–
0.003050; recurrent emitted-state RMS0.004286–0.006079. These demonstrate
sensitivity, not a semantically usable code. They do not exonerate the mixer.
The complete route comparison contains6,144 generated one-step futures.

Every traced case lacks source-cell visibility in the preceding8 frames.
7/12 have never had that world cell inside the view in the entire prefix;
5 were in view11–61 steps earlier. Counts are geometric visibility, not a claim
that earlier observations were reliable or that memory retains them.

## 5. Two necessary post-result validity audits

These were explicitly added AFTER the route results, recorded in PLAN.md.
They do not replace the cohort, tune a gate, or change any primary result.

**Night instrument reliability.** Hold the real successor world fixed and vary
only the native renderer's `state_rng`:64 new night-static draws per branch for
the24 clear and12 traced terrain cases,4,608 encoded real frames.

| family/cohort | present recognition | absent recognition |
|---|---:|---:|
| stone/clear |99.61%|100%|
| stone/traced |93.92%|100%|
| tree/clear |100%|100%|
| tree/traced |67.19%|98.44%|

Therefore the hard-case generation shortfall cannot simply be equated to a
perfect-reader100% ceiling. Still, the real-rendered stone reference93.92%
is substantially above generated55.03%. Tree's natural-noise readability is
imperfect, but substantially above generated3.125%. This check holds the world
fixed; it does not calibrate complete model dynamics or diagnose camera-motion
errors. A low score at the expected location can include incorrect scrolling.

Native fine reconstruction recovers80.5–98.6% of the stone-versus-grass pixel
contrast and60.6–72.7% of tree-versus-grass contrast. Here contrast is the
projection of the decoded A-minus-B image onto the true A-minus-B difference,
divided by squared true contrast;1 is exact contrast,0 identical reconstructions.
It is not percentage semantic accuracy. The two-alternative classification is
correct for all24 traced current-frame reconstructions.

**Does native pre-glimpse TSP already distinguish the cue?** Use the SAME
common real-prefix predicted belief and current coarse percept, with no current
fine evidence. Decode TSP and apply the SAME two-alternative pixel metric:

-8/9 stone and2/3 tree pairs are already correctly distinguished on BOTH
  alternatives. Pre-glimpse recovered contrast is69–109% for stone and60–96%
  for trees. The external coarse classifier understated native recoverability.
- Stone exception:episode42,t203. TSP's factual mean is closer to grass;
  fine reconstruction selects stone. But native disjoint-gaze generation
  ALREADY preserves the factual stone61/64 times(95.31%); focused60/64.
- Tree exception:episode1,t209. TSP incorrectly selects tree in the
  removed-tree frame; fine reconstruction selects grass. But native disjoint
  generation ALREADY predicts absence62/64 times(96.875%); focused62/64.

TSP's means are an operational source-content readout, not a calibrated belief
distribution or proof that every detail is known. Nonetheless, the original
claim of *fine-only* semantic evidence is not sustained. The2 remaining source
TSP errors do not define outcomes that depend on fine input in the measured
generator. Preserve these results rather than select another cohort on them.

## 6. What this settles, and what it does not

The old arrow null was unsuitable for isolating fine use because the native
predictive skill was unqualified. Stone/tree correct that premise. But testing
the next premise reveals a different issue: apparent coarse misses often are
reader misses, not a fact absent from the native coarse/belief path. Fine
reconstruction fidelity and availability are not automatically new information.

There IS a measured clear-input versus difficult-input generation gap, and
focusing does not rescue the selected physical-percept outcomes. It is not
established that this gap requires unavailable fine information. It could also
involve native generation/readout, pose prediction, trust, or redundant routes.
Do not relabel this null as proof of semantic destruction in the projection or
as proof the denoiser alone is responsible. No repair is justified by this test.

This distinction follows the activation-patching literature's warning that
redundant routes and contrast choice limit what a null can identify:
[Heimersheim & Nanda2024](https://arxiv.org/html/2404.15255v1).

The campaign stops here for user discussion. No further family search,
fine-tuning, architecture change, or overnight job has been queued.

## Evidence

- `PLAN.md`, `SOURCE_PINS.json`, `FINAL_PINS.json`: declaration/timing/lineage.
- `catalogue.json`, `far_catalogue.json`, `hud_codebook.json`: source search.
- `sources.pt`, `source_qualification.json`: exact rendered simulator pairs.
- `qualification.json`: per-source acquisition and reader qualifications.
- `gates.pt`, `gate_decisions.json`: all clear-reference native predictions.
- `traces.pt`, `trace_summary.json`, `trace_cases.json`: all routing results.
- `source_history.json`: entire-prefix geometric visibility of the terrain.
- `observation_check_before_tsp.json`, `observation_check.json`: post-result
  noise/instrument check and native pre-glimpse qualification, preserved stages.
- Scripts: `catalogue`, `far_catalogue`, `hud_codebook`, `sources`, `native`,
  `analyse`, `observation_check` (run as modules from repository root).

Read generated-reader rates as operational measurements with the stated
qualification. Neither this12-case terrain cohort nor the bounded source
catalogue proves that no genuinely fine-dependent behavior exists anywhere.
