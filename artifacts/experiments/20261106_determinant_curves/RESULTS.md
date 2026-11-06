# Determinant curves: P1-F3 conditions none of the failing consequences on what determines them

2026-11-06 15:17-15:35 (PLAN.md and curves.py, f5400a15; unit det-curves, MAMBA_DETERMINISTIC=1). F3 checkpoint 10, 16
sampler steps, the unchanged actor interface, K = 4, 18,169 dev histories, 72,676 futures. `readings/curves.json`,
raw per-future outputs `readings/raw.npz` (local). R0 passed: under the capture, interface.futures reproduced F3's
16-step dev COLLECT_COAL reading exactly. Readers: all eight valid (100.0% on every dev frame, every value; asleep
reader 305,907 / 305,907). Deviation from the plan, no reading affected: the hit population's zombie strata were capped
at 1,000, not 800 (the cap lookup matched the zombie population's prefix).

## 1. Declared contrasts: all BLIND

T = world contrast / true contrast (frame readers; heads beside them agree within 0.02 everywhere); 95% episode bootstrap.

| consequence | world d1 / d0 | recorded d1 / d0 | T [95%] | Q = world / truth at d1 | false rate at d0 | verdict |
|---|---|---|---|---|---|---|
| wake (energy 9 vs 8 fresh) | 0.538 / 0.517 | 1.000 / 0.013 | 0.02 [-0.00, 0.05] | 0.54 | +0.50 | BLIND |
| cooldown (ready vs cooling, awake, safe) | 0.142 / 0.069 | 0.321 / 0.005 | 0.23 [0.18, 0.28] | 0.44 | +0.06 | BLIND |
| lethality (death, ready: health <= 2 vs >= 3) | 0.090 / 0.001 | 0.335 / 0.001 | 0.27 [0.24, 0.30] | 0.27 | 0.00 | BLIND |
| sleep attack (damage, asleep: ready vs cooling) | 0.381 / 0.303 | 1.000 / 0.005 | 0.08 [0.05, 0.11] | 0.38 | +0.30 | BLIND |
| cow kill (cow leaves the faced cell: kill vs not) | 0.745 / 0.644 | 0.999 / 0.381 | 0.16 [0.12, 0.21] | 0.75 | +0.26 | BLIND |
| zombie kill (same) | 0.589 / 0.460 | 0.979 / 0.205 | 0.17 [0.13, 0.20] | 0.60 | +0.26 | BLIND |

**World vs head: WORLD-DRAWN everywhere.** When the head reports a wake (WAKE_UP not yet unlocked, energy <= 8), the
generated frame is awake in 100.0% of futures (1,200 / 1,200; at energy 9, 785 / 785); death 1,913 / 1,913; damage
96.7%. The false wake rewards of 11-06's post-hoc are drawn wakes, not a head reading a wake into an asleep frame.

## 2. Wake: the world does not keep the agent asleep

- From asleep histories the world draws an awake frame in 42-54% of futures at EVERY energy (<= 5: 0.42; 6: 0.48; 7:
  0.49-0.51; 8: 0.48-0.52; 9: 0.54), against 1-2% recorded below 9 and 100% at 9. It keeps awake agents awake (99%).
  Training: asleep -> awake 5.1% per asleep step (471,922 asleep transitions); the world is 10x over.
- Not the action (every one of the 17 recorded actions 0.41-0.53; NOOP 0.505), not the time asleep (0.41-0.52 from 1 to
  60+ steps), strongly the light: 0.71 at light < 0.3, 0.61 at 0.3-0.5, 0.46 at 0.5-0.8, 0.36 at >= 0.8. Per history
  the four futures split around one half: a per-step coin flip, not a per-history decision.
- Coherent frames (`coherence.py`, `readings/coherence.json`; 300 energy-8 histories regenerated, other seeds): the
  reader is bimodal on generated frames (31 of 19,188 between 0.05 and 0.95); single-cue readers agree with it, player
  sprite 99.8%, mean map token 99.8%, lighting latent 85% (the generated lighting leans awake: asleep 0.38 vs 0.50). The
  whole frame is redrawn awake; recorded frames: all cues 99.3-100%.
- Declared value vs change: V = 0.021 [-0.004, 0.046], C = 0.040 [0.014, 0.067]: CHANGE-TRIGGERED by the declared rule
  (C's bound > 0 and C >= V), but both are under 5 points on a 50% base: the world reads neither the value nor the change.
- **Correction to 11-06's post-hoc** (`../20261105_p1f3/RESULTS.md` 7, my reading): "world P per step at energy 6 / 7 /
  8 = 0.00 / 0.16 / 0.20" was the HEAD's P(reward > 0.5), not the world's wake rate. Here, on one-step futures, the
  head's unlock is graded in energy (<= 5: 0.15 ... 9: 0.47) while the frame's wake is flat: the world draws the wake
  at ~50%, the head then grades a drawn wake as WAKE_UP by the energy it reads (unlock / awake ~0.34 at <= 5
  -> ~0.87 at 9).

## 3. Damage and death: readiness-blind, arithmetic partly right

- Damage when ready 0.14 vs 0.32 recorded; when cooling 0.07 vs 0.005: the hidden cooldown barely moves the world.
- Given drawn damage, the HUD arithmetic: health 1 -> 0 in 100% (154 / 154); health 2 -> 0 in 65% (208 / 320; 1 in 35%);
  safe health h -> h - 2 in 52-86% (h - 1 next). So most of the death shortfall is damage not drawn (readiness), then
  the 2 -> 0 step; reading the health digit is not the limit.
- Asleep: a ready zombie always hits (7 damage, wakes); the world draws damage 0.43 / 0.31 (lethal / safe) ready and
  0.30 cooling, and drops of 1-7 spread rather than 7.

## 4. Kills: hidden health ignored, the frame inconsistent with itself

The world removes the hit mob about as often whether the hit kills or not (cow 0.75 vs 0.64; zombie 0.59 vs 0.46).
Food rises after a cow hit in 15% of futures whether or not the cow is drawn gone (0.153 vs 0.158): the HUD does not
follow the map.

## 5. Counts: each (item, digit) transition learned on its own exposure, over a shared token geometry

- Increment success (count n -> n + 1): coal 0.28 at n = 0, 0.10 at n >= 1; iron 0.18 / 0.09; wood 0.78 / 0.59 (n 1-3:
  0.82; n 5: 0.10); stone 0.88 / 0.63 (n 5: 0.29). Copy (count unchanged) is the error; the mined cell is cleared
  0.92-0.99 everywhere (iron 0.73-0.81).
- `geometry.py` (`readings/geometry.json`, EDA): across the 27 (item, n) cells with >= 25 histories, success follows that
  transition's own training exposure (mining increments in the train split): Spearman 0.91, logit R^2 0.81; the token
  step |mu(n + 1) - mu(n)| adds little (R^2 0.83) except the smallest step, 5 -> 6 (distance 0.93-1.52 in every slot):
  stone at 5 succeeds 0.29 with 8,281 examples, where 8,000 elsewhere give 0.6-0.8.
- The coarse tokens are compositional for digits: the step mu(n + 1) - mu(n) points the same way in every count slot
  (cosine 0.85-0.99 wood / stone / coal / iron at equal n) and a linear digit reader fitted on wood + stone tokens reads
  coal at 93% and iron at 99.7% (same-slot 100%). Perception offers a shared "+1"; the generator does not use it.
- Perception is not the limit for any visible determinant: every HUD value is read at 100% from the coarse token plus
  lighting, adjacent digits separated 6-49x their within-value spread (energy 8 -> 9: distance 1.38, the smallest step
  of any slot, separation 10.7).

## 6. Reading

1. Every failing consequence is BLIND to its determinant (T <= 0.27), for visible (energy, health) and hidden
   (cooldown, mob health) determinants alike, and the generated frame draws what the heads report.
2. Wake is not an energy-digit failure in F3: the world does not hold the asleep state (~50% per step, worse in the
   dark). The energy rule cannot be read until sleep persists.
3. Counts are a generator that learns each (item, digit) transition separately, at a rate set by its own exposure,
   although its own perception gives the "+1" one shared direction and the belief carries the coal event (trace B1 AP
   1.0). This is the user's "struggling to understand relationships": a learner without transfer, so data looks short.
4. Damage / death / kills are limited by the hidden state (readiness, mob health); the health arithmetic is mostly right.
5. Per PLAN: all six enter the third experiment; it localises where each determinant stops (plan for Codex's review).

Limits: one world seed; one-step futures from real histories; 16 steps; the fixed random gaze; readers fitted on real
tokens (valid on generated frames where the generated token is on the real support: share of reads at probability >=
0.9 is >= 0.96 in every stratum with >= 100 histories, except the mined slot at stone 4 / 5: 0.92 / 0.90).
