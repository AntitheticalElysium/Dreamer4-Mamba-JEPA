# P1-F2.2: F2 with 2 x 2 periphery acuity (user 11-04: prepare for review; not launched)

**Why.** At 3 x 3 the coarse periphery already carries the tested Craftax task facts (HUD 70,208 / 70,208 values; no
task-relevant coarse misses in 17,518 local resource / blockage frames: `../20261104_task_information`, which also
found one small natural witness where a fine observation corrects a mistake every tested coarse observer makes). F2's
native fine-to-future uptake is null (+0.002%, `../20261104_p1f2_review` section 5) and its generated outcomes are
gaze-insensitive (own closed-loop vs random gaze within 1.3 pp in every class: `../20261104_f2_own_gaze`). At this
acuity the fovea's informational role is small and rare (it may still matter for learning efficiency, untested). The
acuity EDA (`saccade/design/PERCEPTION.md`, same pooling as the model: adaptive average pooling
rounded to 8 bits) gives, at r = 2: HUD values exact (1.000); day presence 0.96 (fails the declared 0.01 rule); night
rings 1-2 recall zombie .87 / cow .83 / skeleton .53 / arrow .42 (r = 3: .96 / .96 / .90 / .77). At 2 x 2 the crop
becomes the only full source of near-player entity identity at night, and partly by day.

**Claim.** When the periphery loses entity information, the world uses its learned fovea: glimpse evidence reaches
the belief and the generated futures, and the learned gaze goes where the periphery is now poor.
**Competing.** The world does not integrate fine evidence whatever the need (the fused token's access deficit at the
global projection, `../20261103_fusion_projection`; F2's null native uptake). F2.2 is then F2 with less information:
poorer entity perception, outcomes still gaze-insensitive.
**Separating readouts** (F2.2 vs F2, and within F2.2 own vs random gaze; declared before launch):
1. Native uptake: the predeclared 58-scene / 48-episode past-look audit (informed vs control glimpse, local-future
   loss). F2: +0.002% [-0.005, +0.010]. Use: a clear improvement (negative, interval below 0).
2. Gaze sensitivity: F2.2 under its own closed-loop gaze vs random gaze (`../20261104_f2_own_gaze` instrument) on the
   interface classes. F2: at most 1.3 pp. Use: own gaze better, by clearly more than F2's 1.3 pp on entity classes
   (defeat zombie / skeleton, eat cow, adjacent, risky).
3. Night entities: S1.1 periphery recall (expected lower: the input) and S1.2 belief / S1.3 reader counts (does the
   belief or generation recover what the periphery lost?), F2.2 vs F2, night rings 1-2.
4. Costs: the full interface table (random gaze) and S1.3 C2ST vs F2. Not a pass / fail; it prices the change.
Neither outcome selects an architecture by itself: it answers whether this system uses foveal evidence when it is
needed, which the 3 x 3 runs cannot.

**What changes.** `perception.periphery.acuity` 3 -> 2 only (`train_p1f22.py`, a recipe override; m0.yaml untouched).
Held fixed: corpus (original store), change-map targets, budget (25.6M frames), F2's trainer and chooser (online
cheap head, same policy and seeds), losses, world seed 1. Every provenance field equals F2's (checked). Consequences of
the one change: the periphery's first layer reads 12 inputs instead of 27, so the world's initial parameters differ
from F2's beyond that layer (a different draw from the same seed); the chooser reads the same 2 x 2 coarse tokens as
the world (checked bitwise on real frames).
**Risk, part of the test.** The chooser sees only coarse vision; at 2 x 2 it may not find what the periphery no longer
shows (F2's night entity coverage was already 38% vs 55% by day). One seed.

**Preflight** (`checks.py`; F2.1's bounded checks): exact learned-gaze resume, the real launcher, 200 updates at 64
steps and 3 at 256, the monitor, the acuity checks (first layer 12 inputs; chooser coarse = world periphery). Then
`pin_sources.py` seals the sources. **Launch** (user's approval after review): `run.py` one-shot (preflight gate,
training, F2's fixed checkpoint rule); the evaluation queue against F2 is written and reviewed while it trains.
One GPU job at a time: F2.2 and F2.1 run one after the other, in the order the user chooses.
