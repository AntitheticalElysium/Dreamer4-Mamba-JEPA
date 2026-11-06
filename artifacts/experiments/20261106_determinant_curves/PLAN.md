# Determinant curves: does P1-F3 condition each failing consequence on what determines it? (user 11-06: "the second experiment")

**Why.** P1-F3 (canonical x-loss world) still misses the consequences that hinge on an exact value: wake-up (energy
digit), coal / iron increments (the count digit), death (health digit), damage (the zombie's hidden cooldown), mob kills
(hidden mob health). The coal reading (`../20261106_f3_train_dev`) found these learner-limited, not memorised. Existing
evidence is observational and read by the heads only: the own-history post-hoc (`../20261105_p1f3/posthoc_wake.py`) shows
false wake rewards graded in the energy digit (6 / 7 / 8 -> 0.00 / 0.16 / 0.20 per step); the trace's cooldown strata show
F3 separating ready / not-ready damage 1.6x against 18x recorded. Nobody has read whether the generated FRAME draws these
events, nor the world's event rate across each determinant's values on situation-defined populations.

**Claim (per consequence).** The world's event rate follows the determinant as the simulator does:
P_world(event | d) tracks P_true(event | d) across d.
**Competing explanations.** (B) blind: the world uses a base rate, flat in d; (X) proxy: graded through a correlate of
d rather than d itself (for wake: the energy digit CHANGING, or time asleep, instead of its value 9); (H) head-only:
the heads report an event the generated frame does not draw (or the reverse), so the failure is the head's, not the
world's. **Separating observations.** (1) the curve across d against the truth on the same rows; (2) for wake, a
value x change design: asleep at energy 9 (always a fresh digit), at 8 with the digit changed on the previous step
(fresh), at 8 unchanged (stale): truth 1.0 / 0.02 / 0.02; a value reader separates 9 from 8-fresh, a change trigger
separates 8-fresh from 8-stale; (3) every future read twice, by the outcome heads and by frozen readers of the
generated frame, with the same readers on the recorded successor.

## Instrument

- P1-F3 checkpoint 10 (`../20261105_p1f3/run/ckpt_10.pt`), `edm.sampler.steps = 16` (where worlds are compared since
  11-05), all weights frozen, MAMBA_DETERMINISTIC=1.
- Generation: the actor interface's production path UNCHANGED (`../20261030_actor_interface/interface.py`, `futures`: 8
  real context frames, streaming cascade, the fixed random evaluation gaze, K = 4 futures per history). The generated
  percepts are captured from `model.sample`'s return inside `futures` (no copy of the sampler loop).
- **R0** (before anything else): with the capture active, `futures` on dev COLLECT_COAL (interface classes, rng 2031,
  seed 3000 + its class index) reproduces F3's 16-step reading (`../20261105_p1f3/readings/interface_steps.json`,
  f3_16steps) exactly; otherwise stop.
- Readers (frozen, fitted before any generation, on F3's tokens of the train400 frames, `s0_train400_labels.npz`; S0 probe
  protocol `s1.fit`: MLP 256-256, AdamW 1e-3, held-out = episode % 10 == 9, early stop 3, <= 30 epochs, seed 0):
  - asleep: [player cell token 31, lighting v, mean map token] -> s_sleep;
  - HUD value per cell, [cell token, v] -> value 0..9 (health <= 0 pooled as 0, the empty-heart frame): health 63,
    food 64, energy 66, wood 68, stone 69, coal 70, iron 71 (renderer row 7);
  - the faced cell's entity kind and terrain: F3's own frozen S1 reader (`../20261105_p1f3/run/s1/results.reader.pt`).
  **Reader validity (declared):** on ALL dev real frames, accuracy >= 0.99 overall and >= 0.95 for every value with >=
  100 dev frames; a reader that fails marks its events reader-limited (the heads' reading then stands alone).
  Generated-token readability (share of reads with max probability >= 0.9) reported per population and cell.

## Populations (dev, t >= 8; the state at t - 1 and a_{t-1} only; caps sampled with rng 2106)

| population | definition | strata (determinant) | truth (game_logic.py) |
|---|---|---|---|
| asleep | asleep at t - 1 | energy <= 5, 6, 7 fresh / stale, 8 fresh / stale, 9 | wake iff energy >= 9 or hit |
| mining | awake, DO, faced coal / iron / tree / stone with its tool, no mob on it, count <= 8 | kind x count n | count n + 1 always; cell -> path (grass for tree) |
| zombie | a zombie at Manhattan distance 1 | asleep / awake x ready (min adjacent cooldown <= 0) / not x lethal (health <= 2 awake, <= 7 asleep) / not | attack iff adjacent after the move and cooldown <= 0; damage 2 awake, 7 asleep (wakes) |
| hit | awake, DO, faced cell holds a cow or a zombie | kind x kill (mob health <= the sword's damage) / not | mob removed iff health <= damage; cow: food + 6 (cap 9) |

Coal and iron are taken whole (1,184 / 669 dev rows); wood and stone (the same relation at high exposure: the positive
control for "+1 is learnable in this architecture") capped at 150 per count; every other stratum capped at 300-1,200.

## Events (per future; generated frame, recorded frame through the same readers, labels)

awake (asleep reader < 0.5); damage (health read < the context's health); death (health read 0; heads: P(termination));
count = n + 1 / = n (copy) / other at the mined slot; cell cleared (faced cell: no entity, terrain = the block left
behind); mob gone (faced cell no longer reads that kind); food up; energy value. Heads, as the interface: P(reward >
0.5) (an unlock: read only on rows where that achievement is not yet unlocked), P(reward < -0.05), P(termination).

## Declared readings

Primary contrasts, d1 vs d0 on the same population, tracking T = (P_world(d1) - P_world(d0)) / (P_true(d1) - P_true(d0)),
episode bootstrap 2,000 (seed 2107), world = the frame readers (heads beside it):

| consequence | event | d1 | d0 | recorded (dev) |
|---|---|---|---|---|
| wake (value) | awake | asleep, energy 9 | asleep, energy 8 fresh | 1.00 / 0.02 |
| cooldown | damage | awake, ready, not lethal | awake, not ready, not lethal | ~0.3 / ~0.006 |
| lethality | death | awake, ready, lethal | awake, ready, not lethal | ~0.33 / ~0.005 |
| sleep attack | damage | asleep, ready | asleep, not ready | 1.00 / ~0.005 |
| cow kill | mob gone | cow, kill | cow, not kill | ~0.93 / ~0.03 |
| zombie kill | mob gone | zombie, kill | zombie, not kill | ~0.96 / ~0.15 |

TRACKS: T's lower 95% bound >= 2/3. BLIND: T's upper bound < 1/3. PARTIAL otherwise. Beside T: production
Q = P_world(d1) / P_true(d1) and the false rate P_world(d0) - P_true(d0).

- **Wake, value vs change** (truth: V = P(9) - P(8 fresh) ~ 0.98, C = P(8 fresh) - P(8 stale) ~ 0): VALUE-READ if V's
  lower bound > 0 and C < V / 3; CHANGE-TRIGGERED if C's lower bound > 0 and C >= V; otherwise MIXED. Time asleep within
  8-stale is reported descriptively.
- **World vs head** (wake on WAKE_UP-not-unlocked rows; death and damage on zombie rows): P(frame event | head event >
  0.5) and its reverse. WORLD-DRAWN if P(frame | head) >= 2/3; HEAD-READ if <= 1/3. This settles 11-06's open question:
  whether the false wake rewards are drawn wakes or a head reading a wake into an asleep frame.
- **Mining** (descriptive; truth is 1 at every n): per kind and n, P(n + 1), P(n), other, cell cleared; increments
  (n >= 1) vs appearance (n = 0) per kind, and coal vs stone at n = 1..3.

**What each outcome feeds (no automatic follow-up).** BLIND / PARTIAL consequences with a VISIBLE determinant (wake,
lethality, mining) and the hidden ones (cooldown, kills) are the third experiment's populations and contrasts; its plan
is written for Codex's review, not launched by this reading. A TRACKS verdict removes that consequence from the
determinant question. A HEAD-READ verdict moves that consequence to the outcome heads.

**Limits.** One world seed; one-step futures from real histories; 16 sampler steps (production uses 3: lower rates,
consequence trace RESULTS.md 7); the fixed random gaze; readers are fitted on real tokens, so generated-frame reads are
valid only where the generated token is on the real support (readability reported; heads beside it). Hidden
determinants are inferable only from history (cooldown <= 5 steps, mob health from earlier hits): their truth is the
simulator's, not the best a history-reader could do.
