# Bounded foveal uptake and F2 prelaunch campaign

2026-11-02, written before this campaign's runs. Native F1.5 checkpoint 10 is frozen.
No overnight launch, architecture repair, or automatic follow-up is authorized by this protocol.
Report the results and discuss them with the user before extending this campaign.

## Priority: useful evidence to future physical prediction

Use the previous uptake reading's exact 250 dev episodes and sampled frames (saved in
`20261101_p1f15/readings/uptake/uptake.npz`). Retain all night frames and the first 256
random frames, deduplicating (episode,time); no selection by effect size or future outcome.
Replay each complete real prefix, keeping the carry just BEFORE the intervention frame.
Compare all 49 legitimate real crops of that SAME frame. Keep all earlier observations,
actions, rewards, denoiser noise and future query fixed. No zeroed or shuffled evidence.

1. Fit two fixed-budget (1,200 updates, 256-256 MLP, balanced kind sampling) fine-token
   readers on 400 training episodes, with the last 40 episodes used only for reporting.
   Labels train instruments only, never the world or chooser. Report ordinary held-out
   errors and confusion matrices. Compare with the native frozen periphery reader.
   Primary useful-evidence subset: native periphery reader misses the entity AND both
   fine readers identify it. A null without this positive control is inconclusive.
2. Trace legitimate crop contrasts through percept embedding, corrected belief,
   post-action belief and projected memory. Decompose the input difference algebraically
   into crop-value and fixation-mask contributions. These are measurements of trained
   weights, not invalid mask-only world interventions.
3. Score the next real percept's native denoising loss at all eight monitor levels,
   with two independent noise sets common across crops. Primary region is the union of
   radius-two neighborhoods around useful source entities (clipped to map rows); it is
   defined from the source frame, before the future outcome. Report whole-map too.
   Compare crops containing useful evidence against entity-free crops, paired per scene;
   stratify night/day, all noise levels/highest two. Estimate intervals by episode bootstrap.
   This is evidence use under P1's actual objective, not a full sampler or actor result.
4. Also report first-call pure-noise prediction sensitivity to crops, without scoring a
   stochastic sampled future against one realized future. This localizes conditioning
   sensitivity; it cannot establish correct future probabilities.

## Four agreed F2 checks, bounded scope

* Reliability: same scene/prefix and checkpoint 1/5/10, two independent forecast noises,
  actual versus NOOP with the action fork applied to BOTH belief and denoiser. Flat
  contingency (actual action NOOP) must produce exactly uniform priorities. Report rank
  stability and crop allocation; no labels in scores. Renderer-repeat sensitivity may
  only be reported if the exact renderer/state reconstruction is checked.
* Sequential behavior: 12 fixed dev episodes, first 256 frames, under uniform and two
  diagnostic policies (equal mean-normalized contingency/surprise, softmax temperature
  one, 20% uniform; same plus recency penalty exp(-overlap trace), trace half-life four).
  Policies have their OWN real glimpse/belief history. These are working test policies,
  not canonized F2 choices. Last four episodes are the held-out behavioral report.
* Useful evidence: the priority experiment above, with its positive control.
* Full-path preflight: bounded collector prototype outside canonical code. Disabled
  chooser must reproduce baseline chunks and random streams; enabled collection must
  retain actual fixations in prefixes, use the pre-chunk carry for native loss replay,
  reset on episode boundaries and invalidate denoiser caches after parameter changes.
  Use full native `p1_losses`, not G1 alone; verify each module's gradients and timings at
  both 64 and 256 chunk lengths. If the actual chooser contract remains unsettled,
  report the missing preflight rather than claiming an unbuilt launcher is qualified.

## Data audit

Verify the collector's full 2,500-step seed tree against every stored exogenous action.
Count terminal-inclusive windows by admission using that precommitted flag sequence,
including flags AFTER recorded death, with an absorbing reference and precommitted
random actions for simulated branches that survive longer. Never infer post-death
exogeneity from survival. Count existing advanced-state random stretches separately.
Do not modify the P1 corpus. New data collection is a distinct intervention.

All scripts, fixed rows, source hashes and raw arrays are saved beside the report.
Instrument bugs may be fixed with a written explanation; scientific nulls are retained.

Renderer-scene selection note (before model scoring): taking the first dark frame selects
night onset (light .495), with almost no static. Preserve those exact-replay results in
`static_onset_initial/`; use the lowest-light frame within each fixed episode's first 256
steps for the noise stress check. This selection is by source lighting, not model scores.

Final uptake control amendment, after the primary physical null: test the native TSP
head on the SAME 58 fine-readable night scenes. Query the next frame at the window
centered on the first useful source cell (source-defined, no future-outcome selection).
Compare its faithful pixel MSE across the same 49 legitimate past crops. TSP reads the
post-action belief plus next real periphery; it has no direct past-glimpse context path.
Thus a positive benefit establishes a native consumer can use the fine evidence through
Mamba even if the coarse denoiser does not. A null still cannot prove erasure. No training,
no architecture change, no further automatic follow-up after this bounded control.
