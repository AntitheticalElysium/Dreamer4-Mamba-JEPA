# P1-F3 on its own training histories at 16 steps: coal is NOT memorised (learner-limited, not a vote artefact)

2026-11-06 13:30-13:50 (PLAN.md, c0a9d6bd; unit f3-train-dev, MAMBA_DETERMINISTIC=1). F3 checkpoint 10 at
edm.sampler.steps = 16, actor interface unchanged, fixed random gaze, K = 4, the same 1,009 training episodes as 11-04
(asserted). R0 passed: dev COLLECT_COAL recomputed in process equals F3's readout exactly. `readings/train_dev.json`.

| Primary event | metric | generated train / dev | train - dev [95%] | closure [95%] | verdict (F2, 3 steps, 11-04) |
|---|---|---|---|---|---|
| COLLECT_COAL | P(reward > .5) | 0.427 / 0.444 | -0.017 [-0.069, +0.035] | -0.03 [-0.12, +0.06] | not memorised (not memorised) |
| COLLECT_IRON | P(reward > .5) | 0.235 / 0.240 | -0.005 [-0.043, +0.033] | -0.01 [-0.06, +0.04] | not memorised (not memorised) |
| risky: death | P(termination) | 0.082 / 0.088 | -0.006 [-0.017, +0.004] | -0.03 [-0.32, +0.26] | not memorised (not memorised) |
| adjacent: damage | P(reward < -.05) | 0.092 / 0.097 | -0.005 [-0.016, +0.007] | +0.08 [-0.13, +0.28] | not memorised (not memorised) |

Declared reading: NOT MEMORISED -> learner-limited even at 16 steps: F3 does not produce coal, iron, death or damage on
histories it trained on (~10 exposures each) any better than on unseen ones. More distinct instances are not predicted
to help cheaply; the deficit is in how the consequence is learned (signal, route or representation). The 3-step vote
was not hiding memorisation in the 11-04 reading.

Every other class (same rule): memorised: PLACE_TABLE, COLLECT_DRINK, PLACE_STONE, PLACE_FURNACE, EAT_PLANT (n = 25);
not memorised: COLLECT_WOOD, COLLECT_STONE, EAT_COW, COLLECT_SAPLING, DEFEAT_ZOMBIE (+0.057, closure 0.09),
DEFEAT_SKELETON (+0.088, closure 0.16), WAKE_UP (0.424 / 0.439), COLLECT_DIAMOND (n = 12); unresolved: wood / stone /
iron tools, PLACE_PLANT; no gap: random (0.043 / 0.045), sapling context (0.167 / 0.168). Real-successor heads fit
training events better than dev for the hidden-state classes (EAT_COW 0.889 vs 0.804, DEFEAT_ZOMBIE 0.873 vs 0.768,
DEFEAT_SKELETON 0.746 vs 0.615, EAT_PLANT 0.842 vs 0.385): the heads memorise what the generator does not produce.
One world seed; one-step futures; outcome-head readings.
