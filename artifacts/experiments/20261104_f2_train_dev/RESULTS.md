# F2 on its own training histories: the consequence failures are not memorisation-limited

2026-11-04 22:57-23:13 (PLAN.md, approved by the user before the F2.1 mixture decision). F2 checkpoint 10, actor
interface unchanged, fixed random gaze, K = 4: 1,009 training episodes (seed 612403 from the 7,672 non-monitor train
episodes; 397 monitor episodes excluded) vs the archived dev reading of the same instrument (reproduced in process to
the last field by `../20261104_f2_own_gaze`). One world seed; intervals are evaluation uncertainty (episode bootstrap,
unpaired, combined).

**Answer (declared rule).** All four primary events are NOT MEMORISED: F2 fails to generate them on histories it
trained on (~10.5 exposures each) just as on dev. More distinct instances at fewer repeats is therefore not predicted
to help them; what remains is exposure (optimisation pressure per event) or the learner itself.

| Primary event | metric | recorded train / dev | generated train / dev | train - dev [95%] | closure [95%] | verdict |
|---|---|---|---|---|---|---|
| COLLECT_COAL | P(reward > .5) | 100 / 100 | 22.41 / 24.37 | -1.97 [-6.65, +2.72] | -0.026 [-0.088, +0.036] | not memorised |
| COLLECT_IRON | P(reward > .5) | 100 / 100 | 7.23 / 9.18 | -1.95 [-4.80, +0.90] | -0.021 [-0.053, +0.010] | not memorised |
| risky: death | P(termination) | 12.02 / 12.77 | 1.00 / 1.27 | -0.26 [-0.57, +0.04] | -0.017 [-0.045, +0.010] | not memorised |
| adjacent: damage | P(reward < -.05) | 14.43 / 15.90 | 1.05 / 1.23 | -0.18 [-0.51, +0.15] | -0.005 [-0.030, +0.020] | not memorised |

Closure = (rho_train - rho_dev) / (1 - rho_dev), rho = generated / recorded; MEMORISED needs closure >= 1/3 with the
difference's interval above 0; NOT MEMORISED needs closure's upper bound < 1/3.

**Every other class** (same rule, none dropped; `readings/train_dev.json`):
- Memorised: COLLECT_DRINK (92.1 vs 86.4, closure +0.42 [+0.13, +0.72]), PLACE_FURNACE (87.4 vs 79.2, +0.39
  [+0.17, +0.62]).
- Not memorised: sapling context, EAT_COW, COLLECT_SAPLING, DEFEAT_ZOMBIE (+5.2 pp but closure +0.07 [+0.02, +0.12] of
  a 77 pp gap), DEFEAT_SKELETON, COLLECT_STONE, PLACE_TABLE, MAKE_STONE_PICKAXE, WAKE_UP, COLLECT_DIAMOND (n = 31).
- Unresolved (small dev gaps, wide closure intervals): COLLECT_WOOD, wood pickaxe / sword, PLACE_PLANT, PLACE_STONE,
  stone sword, iron pickaxe / sword, EAT_PLANT (n = 28: +23 pp, closure +0.28 [+0.11, +0.46]).
- `random`: no gap (generated 3.8 / 4.0% vs recorded 3.2 / 3.4%).

**Real-successor heads** (the heads' own memorisation, beside): higher on training histories for the rare stochastic
classes (EAT_COW 89.0 vs 80.9, DEFEAT_ZOMBIE 85.8 vs 77.6, DEFEAT_SKELETON 74.7 vs 63.9, EAT_PLANT 79.9 vs 36.1), near
ceiling elsewhere. The heads fit training events the generator does not produce.

## What this means for F2.1 (the mixture decision it was run for)
- A fixed-budget union run (diversity at fewer repeats: unique instances ~2.6-3x, per-frame exposure 0.75x for first
  coal, 0.77x death / risk, 0.86x damage) is predicted NOT to improve coal, iron, damage or death: F2 does not
  reproduce them even where it has seen them ten times.
- It does not rule out exposure: F2 might produce them with many more repeats. It does not rule out the learner
  either (the earlier diagnoses: single-cell consequences are a tiny share of a uniform denoising loss, and the
  field misses them under every sampler: `saccade/design` P1 consequence diagnosis, E14). Separating the two is a
  diagnosis question, not a data run.
- Untouched by this reading: whether the new data's states (late game, off-expert actions, failure tails) matter for
  the actor's own distribution. That needs the new held-out episodes read, with labels not yet built.

## Limits
- Under random gaze the training histories' glimpses are not those F2 trained with; the physical stage never reads
  glimpses, so this can only understate memorisation through the belief (conservative for the verdict).
- One-step futures, outcome-head readings; one world seed.
