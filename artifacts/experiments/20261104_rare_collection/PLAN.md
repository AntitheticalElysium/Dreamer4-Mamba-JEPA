# Broad coverage collection, authorized 2026-11-04

Purpose: collect new independent worlds that better cover advanced skills AND
exploration near competent play. This tests neither architecture nor the cause
of the death regression. No world-model training is authorized by this launch.

The old corpus has only184 eat-plant and187 diamond unlocks in actual P1 train,
versus2,249 iron and3,398 coal. Coverage must include all22 achievements, repeated
inventory transitions, combat, survival, sleeping/waking, damage and death.

## Fixed acquisition policy

Frozen archived PPO expert (same parameters as support-v2), categorical action
sampling as in the existing collector. Fresh seed202611040 and independent
reset keys, checked against ALL existing support-v2 reset keys before collection.
No model-generated pixels, hand-edited state, success filtering or terminal
acceptance filter. Full episodes through first true termination or2,500 actions.
Both survivors and failures are retained. All new episodes are TRAIN sources;
the existing stable5% episode-id monitor rule still applies when integrating.
No old dev/final source is used for collection, and old data are not modified.

Two interleaved components,24 environments/batch,128 batches maximum:

1. **Competent play:** half the batches, epsilon cycling0,.02,.05,.10. This lets
   the expert reach deep technology, grow/eat plants and collect diamonds rather
   than repeatedly dying before those states. Epsilon0 still samples the expert
   categorical policy; it does not take an argmax.
2. **Competent prefix plus exploration:** half the batches, base epsilon .02,
   with16-action fully random bursts every128 steps after a fixed512- or1,024-
   step prefix. The prefix alternates across batches. Burst timing is fixed
   before collection, not chosen after seeing damage/reward. This provides
   non-expert actions, failed actions and exogenous sequences at later stages.

Save per-transition epsilon AND the actual uniform-random flag, not a misleading
single scalar epsilon for scheduled policies. Preserve original actions and
reward/termination/truncation alignment. BC eligibility remains false.

Upper bounds:3,072 episodes,7.68M simulated transitions, about92GB raw pixels
before truncation; two-hour collection wall budget checked at batch boundaries.
Budget expiry is reported explicitly, not presented as all coverage goals met.
The final coverage audit must show which rare families remain poorly covered.

## Checks before sustained launch

- CPU contract checks: epsilon schedules, event/terminal alignment, survivor
  retention, no post-terminal transitions in saved episodes, distinct ids,
  explicit propensity flags and checkpoint/resume identity.
- GPU parity against the existing support collector at constant epsilon using
  the same seed, action and environment random-key construction.
- A full2,500-step real-path batch: saved format/hash/load check, event counts,
  peak memory, elapsed time and throughput. Scientific collection is not gated
  on achieving rare successes in this batch; malformed data is a blocker.
- First/last/transition frames checked against exact replay on fresh sources;
  replayed actions/rewards/termination must match stored values.

## Outputs and interpretation

Separate `artifacts/craftax_coverage_v1/` store, atomic per-batch shards and atomic
manifest. Per-shard audit sidecars contain body/inventory/achievement, light,
position, sleep, creature state and action propensities for all retained states.
These labels are collection diagnostics, never model inputs. Source, expert,
environment and old-manifest hashes are recorded. A standalone summary updates
after each batch and finalizes with actual new counts, episode support, inventory
value bands, day/night, component and fixed monitor partition.

No count target proves that the world will learn an event. Do not cherry-pick
successful episodes or call a larger transition count independent rare coverage.
At a fixed future training-frame budget, adding data also reduces repetitions;
the matched data-only run tests the resulting dataset recipe as a whole.
Keep3x3 acuity, architecture and chooser fixed for that first run.2x2 can be a
subsequent matched sensing comparison; it is not silently bundled into collection.

Literature: Craftax (Matthews et al., ICML2024)
https://proceedings.mlr.press/v235/matthews24a.html motivates the deep exploration/
skill hierarchy; the actual collection mechanics and provenance are verified
against local simulator, expert and support-v2 collector source.
