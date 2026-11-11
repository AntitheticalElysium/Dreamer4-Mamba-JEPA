# Adversarial review of the post-F4 danger synthesis

2026-11-07. User authorized adversarial checks. No training, source changes to
Claude's work, or automatic follow-up campaign. One bounded native GPU diagnostic
only after a live idle-GPU check; CPU analysis uses one thread.

## CPU checks

1. Recompute **per-history** damage Brier scores on the SAME saved zombie panel
   for F3/F4, with the finite-ensemble U-statistic correction. Claude's published
   score uses ready/cooling group means and cannot stand in for the full forecast
   score. Episode-bootstrap paired F4-F3 differences. Read panel averages and,
   separately, inverse-selection-weighted DEV-stratum averages. Score a simple
   training-only adjacency/sleep base-rate reference beside them. This is a
   label-based evaluation control, never a training target.
2. Split timer due/early saved curves; distinguish discrimination, calibration
   and proper score. R1 age is time since the last same-direction HUD change, not
   the actual internal timer counter. Source-audit resets, reversals and saturation.
3. Check whether the native temporal cache contains historical belief tokens.
   A CPU property test may vary past memory while holding observations/current
   memory fixed, using trained denoiser weights. This tests dependency only,
   not scientific benefit or necessity of an OOD intervention.

## One native sleep-duration test

Question: does the one-step energy-increment ratio actually imply 3.7x longer
sleep in the recursive model? Fixed F4 checkpoint 10, native K=4 sampler, full
Mamba feedback, sampled rewards/termination, committed generated context once.

Select up to 32 DEV episodes, one source per episode, from current situations:
asleep, energy=8, health>=7, food/drink>=3, no active arrows and no zombie or
skeleton within Manhattan distance 6; source step >=16. Choose with fixed RNG
110711, no filtering on recorded/generated future or phase. NOOP intervention
at the source and throughout, fixed uniform sensing prior. Eight branches per
source, H=48. Process eight sources at a time (64 branches) for memory.

Reference: exact simulator replay at each source, fresh independent environment
keys, same NOOP actions, source render verified bitwise. Record actual energy,
sleep/health/termination. Compare first energy=9, first awakening, termination,
waking without energy9, and occupancy/timing curves. No pairing of independently
sampled outcomes as if they were the same hidden future.

The model's corrected/predicted belief is rebuilt over its whole real prefix,
with the final prospective action set to NOOP. Cache construction uses the real
past actions/beliefs and native context pass. Real-percept replay versus stored
predicted belief is a mechanical sham before unrolling. Use exact-source energy=8
reader check. A reader trained on real F4 tokens is refitted on the same train400
protocol; report its DEV accuracy. Reader confidence is not correctness evidence
on generated tokens; cross-check event timing with the native wake/death interface.

At H=48, report medians, P(event by 12/16/32/48), and restricted mean waiting time
with source-episode bootstrap. Right-censored means are not unrestricted sleep
durations; do not report a precise full-duration multiplier. This can support or
refute the practical delay claim without assuming iid constant hazards. Source
selection may return fewer scenes; report that instead of weakening criteria.

No extra samplers, fine-tunes, route changes or automatic follow-ups. Stop after
the report and return interpretation to the user.
