# Decision levels: results

PLAN.md (929bb069) and code (8dda93b8) were committed before any measurement. Frozen P1-F3 checkpoint 10 for T1 / T2;
T3 is in section 3.

## 1. T1: where the world stops following the target, and what it predicts from context alone

272 pairs (the audit's 208 + 64 MOVEMENT pairs: all 64 replayed bitwise, natural successors equal to the stored
frames), K = 8 shared draws, first evaluation only. Reliance = |f(conflicting target) - f(own target)| (1 = follows
the noisy target, 0 = follows its context); sigma* = where it falls through 0.5; ideal = an equal-prior two-point
reader at the pair's own separation d.

| family | d (median) | reliance at sigma = 1 / 2 / 5 / 12 / 20 | sigma* network | sigma* ideal | pure-noise first evaluation at 20 (member 0 / 1) |
|---|---|---|---|---|---|
| sleep (asleep / woken) | 28.9 | 0.80 / 0.58 / 0.43 / 0.12 / 0.04 | 3.3 | 15.9 | 0.06 / 0.98 (0 = asleep) |
| energy (8 / 9, asleep) | 29.4 | 0.99 / 0.99 / 0.90 / 0.26 / 0.09 | 8.7 | 16.1 | 0.06 / 0.06 (both asleep) |
| coal (DO / NOOP) | 3.7 | 0.48 / 0.18 / 0.03 / 0.00 / 0.00 | 0.9 | 2.0 | 0.93 / 1.01 (1 = copy) |
| wood (DO / NOOP) | 4.6 | 0.60 / 0.18 / 0.03 / 0.01 / 0.00 | 1.2 | 2.5 | 0.12 / 1.00 |
| movement (move / NOOP) | 18.0 | 0.36 / 0.07 / 0.01 / 0.00 / 0.00 | 0.85 | 9.9 | 0.10 / 0.98 (0 = moved) |

Spearman(sigma*, d) over pairs 0.78.

- **H-dec (the level is set by the consequence's size): REFUTED by the declared rule.** Movement sits 11.6x below
  its ideal level and out of order (largest-but-two d, lowest sigma*); sleep 4.8x. The level tracks d only where the
  context says nothing about the outcome (energy: 1.9x); wherever the world has a confident context prediction
  (movement, persistence; counts at small d) it stops consulting the target far earlier. The governing variable is
  the confidence of the learned context prediction together with d, which is what a posterior-mean denoiser
  combining a learned conditional prior with the noisy target does (EDM: D = E[x | x_sigma, c]).
- **H-ctx (above that level the output is the context-only prediction): SUPPORTED.** The pure-noise first evaluation
  at sigma 20 matches the generated rates of the prefix-20 recipe (audit) in every family: sleep 0.06 vs 0.08 woken;
  energy 0.06 / 0.06 vs 0.06 / 0.07 woken; coal increment 0.07 vs 0.09; wood 0.88 vs 0.89; movement 0.90 moved. The
  energy-9 context-only prediction is the marginal (asleep), not the rule.
- **Movement, the case that could break H-start: not broken, and it explains the start effect.** Generated movement
  is 100% correct at both starts (move member moved 1.00, NOOP member 0.00, K = 16). At the original start the
  world's reliance on the target is 0.01 for movement, 0.03 for coal / wood, but 0.43 for sleep and 0.90 for energy:
  pure noise at sigma 5 replaces a target only where the world still consults one, and those are exactly the families
  the higher start changes (sleep -37 pp, energy -28 pp; coal / wood / movement within 4 pp).

## 2. T2: the determinant curves with the prefix-20 start (same tuples, seeds; sham bitwise)

| declared reading | original start (exp 2) | prefix-20 | prediction |
|---|---|---|---|
| woken from asleep, energy <= 8 (truth 0.01-0.03) | 0.42-0.52 | 0.06-0.11 | <= 0.15: holds |
| energy value contrast T | 0.02 | 0.04 (woken at 9: 0.54 -> 0.15; truth 1.00) | < 0.10: holds |
| cooldown T | 0.23 | 0.24 | change < 0.10: holds |
| lethality T | 0.27 | 0.22 | holds |
| sleep-attack T | 0.08 | 0.04 | holds |
| cow / zombie kill T | 0.16 / 0.17 | 0.20 / 0.20 | holds |
| coal / wood increments (n >= 1) | 0.105 / 0.587 | 0.093 / 0.607 | holds |

The start lever is confined to the sleep / wake state and what rides on it: a ready zombie hitting a SLEEPING agent
(truth: damage 1.00, wakes 1.00) is drawn with damage 0.43 -> 0.18 and woken 0.63 -> 0.23. All six BLIND verdicts of
the determinant curves survive the corrected start. The heads still read what the frame draws (wake unlocks 321 / 321
and 284 / 284 on awake frames, death 1,003 / 1,003). The false-wake reward exploit shrinks (head unlock per asleep step
0.15-0.36 -> 0.03-0.10), but true waking at energy 9 falls to 0.15: with the start fixed, imagined sleep becomes a
state the agent rarely leaves.

## 4. Which relations the world learns (user 11-06: "let's see that analysis"; `need.py`, `readings/need.json`, existing data only)

Per relation: training events per pass (labels); the change size D at the consequence token(s) (F3's own tokens);
need = events x D^2 x the share of training noise draws at which the target does not already reveal the outcome
(the squared-error reduction the determinant offers when the model otherwise predicts the default: copy / persist /
marginal; the x-space weight is the same at every level, so it drops out). Success at the frame level (T2 prefix-20,
T1) unless marked.

| where the determinant lives | relation (events / pass, need -> success) |
|---|---|
| own past + the action | movement (905k, 5.7M -> 1.00); faced cell cleared when mining wood (86k, 217k -> 0.98), stone (130k, 164k -> 0.94), coal (11k, 23k -> 0.96), iron (8k, 9k -> 0.81) |
| another token (the count slot must learn what happened at the faced cell) | wood 0 -> 1 (17k, 67k -> 0.82), wood n -> n+1 (51k, 149k -> 0.80), stone (16k / 77k, 65k / 164k -> 0.89 / 0.66), coal (4k / 6k, 16k / 16k -> 0.25 / 0.10), iron (3k / 3k, 12k / 8k -> 0.13 / 0.08) |
| another token, global consequence, smallest cue | energy 9 -> wake (16k, 101k -> 0.15) |
| history (hidden) | damage when a zombie is ready (50k, 103k -> 0.42 of recorded); cow killed leaves the cell (13k, 42k -> 0.75, but non-kill hits 0.62); food +6 after a kill (13k, 14k -> 0.17 of recorded) |
| own past + a specific action (head only, frame not measured) | crafting a tool: slot 0 -> 1 (iron pickaxe / sword 1.4k per pass, need 5.5k): head 0.92 / 0.89 of its ceiling; placing table / furnace / plant 0.94-0.97 |

- The natural control: the SAME coal event, same frequency and cue, clears the faced cell 0.96 and increments the
  count 0.10-0.25 (needs within 1.4x). What differs is where the determinant lives: the cell's new value is its own
  past (coal) + the action; the count's new value needs what happened at ANOTHER token.
- Within "another token", need orders success (wood / stone work at 65-164k; coal / iron fail at 8-16k), except the
  energy rule: wood-level need, 0.15. It stacks every difficulty: a global consequence (every map token and the
  lighting must change), the smallest cue of any determinant (8 -> 9, 1.38), half-hidden in noised contexts (exp 3),
  decided at the highest level (T1).
- "Own past + action" relations work at low need (coal cell at 23k, iron cell 9k). Crafting would be the strongest
  case (1.4k events per pass) but it is measured only by the head, which also reads the belief that knows the craft
  action; the frame was not checked. That is the one gap in this table.

## 5. Crafting at the frame level (PLAN, last section; `crafting.py`, `readings/crafting/`)

Frozen F3, dev, prefix-20, K = 4: 6,051 histories, 24,204 futures (strata in `run.json`). Tool-slot readers fitted on
train400: values 0 and 1 read 1.00 in all six slots (wood sword 5 and iron sword 3-4 fail, never seen in train400, so the
declared rule flags those two readers for those values only; appearance uses 0 and 1). Recorded successors read
correctly in every stratum (tool +1 on feasible-from-0: 1.00 in all six; 0.00 on infeasible). Per pass = counts per
pass over the train split (need.json's unit); appearances only (the need row's 1,444 for the iron pickaxe included
increments; appearances are 1,103).

| tool | appearances / pass (need) | feasible from 0: tool drawn | infeasible: station missing / materials missing / both | T [95%] | verdict | tool already held: +1 drawn (incr. / pass) | materials spent, per slot (wood / stone / coal / iron) |
|---|---|---|---|---|---|---|---|
| wood pickaxe | 5.4k (20.9k) | 0.90 | 0.08 / 0.04 / 0.00 | 0.85 [0.82, 0.87] | TRACKS | 0.04 (1.6k) | 0.84 |
| stone pickaxe | 3.2k (12.5k) | 0.81 | 0.10 / 0.00 / 0.00 | 0.77 [0.74, 0.80] | TRACKS | 0.01 (0.7k) | 0.70 / 0.67 |
| iron pickaxe | 1.1k (4.3k) | 0.83 | 0.04 / 0.03 / 0.00 | 0.80 [0.75, 0.84] | TRACKS | 0.02 (0.3k) | 0.77 / 0.46 / 0.72 / 0.82 |
| wood sword | 4.2k (16.5k) | 0.91 | 0.03 / 0.04 / 0.00 | 0.88 [0.86, 0.91] | TRACKS | 0.04 (0.9k) | 0.86 |
| stone sword | 3.1k (12.5k) | 0.79 | 0.21 / 0.02 / 0.00 | 0.71 [0.67, 0.74] | TRACKS | 0.07 (1.2k) | 0.80 / 0.62 |
| iron sword | 1.1k (4.6k) | 0.86 | 0.04 / 0.08 / 0.00 | 0.82 [0.77, 0.87] | TRACKS | 0.19 (0.2k) | 0.75 / 0.44 / 0.67 / 0.78 |

- **Declared prediction: refuted.** Both iron tools TRACK (T lower bounds 0.75 / 0.77 >= 2/3) at 1.1k appearances per
  pass. Cross-token dependence at low exposure is not enough to fail: the gate (a table / furnace among the 8 cells
  around the player = map tokens; the materials = other HUD tokens) is read, with 3-8% false tools for the iron tools.
- **Declared route reading: crafting tracks while the coal / iron counts' appearance fails at higher exposure and need**
  (0.25 / 0.13 at 4.1k / 3.1k per pass, need 16k / 12k), so the hard part is another token's past, not cross-token
  dependence as such. The plan named one two-step order (temporal at that token, then spatial); model.py also allows
  the other (spatial within frame t - 1, carried forward by the slot's own cache entry). The reading, one attention step
  against two, is unchanged. **Not separated:** DO is shared by every collection, while MAKE_X names its slot. In
  mining the two coincide (which item was collected lives only in the faced cell's past), so these data cannot
  separate "past vs present" from "generic vs slot-specific action". No relation in the data has a generic action with
  its cause present in frame t at low exposure (drink: 41k per pass).
- **A second difficulty, separate from the route:** with the same action and the same gate, a held tool's count is
  drawn +1 in 0.01-0.19 of futures (copied 0.77-0.99), at 0.2-1.6k increments per pass, while the appearance is drawn
  0.79-0.91 at 1.1-5.4k. Counts point the same way at matched items (appearance -> increment: coal 0.25 -> 0.10, iron
  0.13 -> 0.08, stone 0.89 -> 0.66, wood 0.82 -> 0.80).
- **A craft is drawn as one coupled event.** P(wood spent | tool drawn) 0.79-0.94 against 0.18-0.51 when the tool is not
  drawn; when the tool is already held, the materials are spent in 0.01-0.35 of futures.
- Heads: WORLD-DRAWN for all six (P(frame | head) 0.90-0.99); the unlock head on recorded frames T 0.98-1.00.
- **Correction to section 4:** crafting was classed "own past + a specific action (head only)". Its gate lives in other
  tokens that are still present in the frame being generated, and its frame-level success is now measured (this
  section).
