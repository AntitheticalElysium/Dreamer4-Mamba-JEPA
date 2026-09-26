# What is wrong, measured — 2026-09-26

No training in any of this. Every number comes from the simulator's own state, from swapping a part of the
pipeline for ground truth, or from twin game states differing in one tile. Rows: the one-step opportunity
roots of seed blocks 55k–58k (already opened; 3,218 roots, 1,971 zombie-adjacent). Scripts and JSON are in
this directory; the per-world dump is `artifacts/eda/diagnosis_dump_v1` (`dump.py`).

## 0. The decision is simple, visible and deterministic (`decision.py`)

Game code (`craftax_classic/game_logic.py`): the player moves first; a zombie then hits iff its Manhattan
distance is exactly 1 **after** the move and its hidden cooldown is ≤ 0. So on a zombie root, any move into
a free tile is safe, and staying put (a stay-type action, or a move into a solid tile or mob) is what gets hit.

| zombie roots | expected safe choice |
|---|---|
| rule on visible tiles only (neighbour passability, mob positions, lava), nothing fitted | **0.977–0.982** |
| same + hidden cooldown | 0.984–0.989 |
| always DOWN | 0.50–0.60 |

The information needed is the passability of the 4 neighbour tiles. The hidden cooldown barely matters for
the choice. It does decide *which* roots die under a given stay action (AUC 0.978 from cooldown, 0.57 from
visible health: `followups.json/noop_death`), but that is irrelevant to choosing.

## 1. Ground-truth substitution: where each system loses it (`choices.py`, `dump.py`)

Each world's **own** head, reading the TRUE next state (`observe_latent`: same Mamba history, latent replaced
by the encoded real successor) vs the IMAGINED one. Zombie roots:

| world | true next state | imagined | chooses SLEEP | chooses a blocked move | chooses a free move |
|---|---|---|---|---|---|
| H2 (canonical) | 0.60–0.66 | 0.19–0.26 | **74–79%** | 8–11% | 12–15% |
| Z | 0.72–0.77 | 0.39–0.46 | 32–37% | 21–29% | 32–39% |
| U | **0.975–0.981** | 0.49–0.55 | 17–23% | 27–40% | 41–50% |
| W | **0.990–0.999** | 0.54–0.64 | 0% | 36–47% | 51–61% |

- For U and W, the state and the head are sufficient: fed the true next state they choose as well as the rule.
  The whole loss is imagination.
- For Z and H2, the state already costs ~0.25 before imagination.
- Nearly every "stay" choice is one action, SLEEP (6). SLEEP is 3.9% of training actions, as common as NOOP,
  so this is not out-of-distribution action error.

## 2. Why z / CLS fails on zombies (`twins.py`, `followups.json/zombie_position`, `count_vs_adjacency`)

Twin states differing in one tile, encoded by the Raw encoder at joint steps 0 and 10k.

- **z is position-blind for a zombie.** Adding a zombie moves z along the same vector wherever it stands:
  cosine between the adjacent-zombie and far-zombie changes = **1.000** (init and 10k). Adjacent vs far from z:
  held-out AUC **0.51**. z therefore cannot represent adjacency at all.
- **After JEPA training, z barely registers a zombie.** Same scene with vs without a zombie, from z: AUC 0.947
  at init → **0.528** at 10k. Fisher d′ of a neighbour zombie against natural variation: 1.79 → 0.51. The
  edited tile's patch token keeps it (d′ 4.2–4.4 throughout).
- **The zombie is the least visible object.** Pixel change of the edit: zombie 2.5 (1.1 at night) vs tree 2.6,
  cow 3.4, table 4.1, stone 4.5, water 5.6, lava 6.5. Its patch-token d′ is also the lowest (4.4 vs 6–28).
- **Lava is not special either; its adjacency is readable from its amount.** The number of lava tiles in view
  predicts lava adjacency at AUC **0.97**, the number of zombies predicts zombie adjacency at only **0.76**
  (8,600 roots). z's 0.95 for lava and 0.60 for zombies (reviewer, `20260925_h2_bound_challenge`) are what a
  position-blind "amount of X" code gives.

Literature consistent with this (not the proof): Littwin et al. 2024 (JEPA prefers high-influence,
predictable features); Shekhar et al. 2023 (joint-embedding CLS weaker on small objects, COCO AP_small
MoCo-v3 15.5 vs MAE 18.7). The suppression is measured; its cause inside training is not isolated here
(static edits do not uniformly rise: tree 0.67→0.38, lava 1.45→0.89).

## 3. Why the transition fails (`transition.py`, `coverage.py`)

Recognising an outcome after the fact is easy; predicting it is not. The true successor shows a move's
outcome as a whole-view scroll (moved vs blocked from ‖true − root‖: AUC 0.86 in u) and damage in the HUD
(damage AUC 0.99 from the true successor). Predicting whether a move succeeds needs the passability of one
specific neighbour tile:

| moved vs blocked (AUC, fit 55k+56k, test 57k+58k) | Z | U | W |
|---|---|---|---|
| root patch tokens (encoder output) | | **0.970** | |
| root state the world receives | 0.58 | 0.69 | 0.69 |
| Mamba output h(a) | 0.58 | 0.69 | 0.69 |
| imagined latent, linear probe | 0.58 | 0.65 | 0.67 |
| size of imagined change ‖ĝ − root‖ | 0.49 | **0.49** | **0.47** |

The world imagines the same change whether the move succeeds or walks into a wall. Passability is in the
encoder's patch tokens and is lost in the state the world is given (CLS: position-blind; u: a 4×4 pool mixes
the neighbour tile with ~8 others). Under next-state MSE the prediction is then the average of scroll and
no-scroll. Consequence: every world picks the free move among its move choices only 0–10 points above chance
(`followups.json/counterfactual`: e.g. W 0.52–0.63 vs random 0.51–0.54).

**Ruled out: coverage.** 22% of the corpus's move transitions are blocked (7,713 of 35,180 decided in 6
expert shards; 23,620 of 109,396 in 40 support shards; the pixel-shift test is 98.8% accurate against
simulator truth). Walking into a tree or stone is how a player faces it to mine it.

## 4. The heads

**H2's depth alias** (`../20260921_readout_ladder/evidence/h2_alias.json`): P(dead) 0.728 on a generated
death at depth 2 and 0.063 at depth 1; at depth 1, death vs 10 frames earlier AUC 0.51. A shortcut: the label
was predictable from recursion depth. Fixed on branch `h2-terminal-depth`.

**SLEEP next to a zombie looks safe to every head, from a rendering quirk in the data**
(`sleep.py`, `followups.json/asleep_heads`, `asleep_deaths`):
- A zombie hit wakes a sleeping player before the frame is drawn, so death frames render awake. Of 4,835
  corpus death frames, **154 (3.2%)** render asleep, while ~20% of all frames are asleep. An asleep-looking
  frame is ~8× less likely to be a death frame.
- SLEEP next to a zombie is the exception: the player falls asleep in the same step the zombie hits. 96–98% of
  those successors render asleep, and 85% are fatal.
- Heads on the TRUE successor: U reads death on awake frames at AUC 0.990 (NOOP) but on asleep frames at
  0.857, with mean P(dead) 0.455 against a true 0.85. H2's imagined P(dead | SLEEP) is 0.024, its lowest action.
- Not a base-rate effect: in the corpus, asleep frames precede death *more* often than awake ones (0.87% vs 0.18%).

## 5. Why W beats U, exactly (`followups.json/counterfactual`, `asleep_heads`, `death_direction`)

1. **W's gain is not choosing SLEEP.** Replace U's stay-type choices with a random move and U scores
   0.550 / 0.600 / 0.604 / 0.571, against W's actual 0.541 / 0.593 / 0.636 / 0.626: about 79% of the gap.
   Among moves, neither beats chance at avoiding blocked ones by much.
2. **W's head reads the rare asleep death frame; U's does not.** True SLEEP successors: AUC 0.949 vs 0.857,
   mean P(dead) 0.616 vs 0.455. Awake frames: both 0.99.
3. **Why, given W = U / std is an invertible linear map with identical information:** 90–92% of the
   fatal-vs-survived information (d²/var) lies in u's low-variance ranks 43–192, which hold only 20% of the raw
   mean difference. The raw mean-difference direction — where weight decay pulls a readout on u — separates at
   held-out AUC 0.61; the same direction in w separates at 0.82. With abundant awake deaths (97% of training
   deaths) U's head learns the tail anyway; with the rare asleep deaths (3%) it does not. This is LeJEPA's
   Lemma 1 (ridge bias under anisotropy) in a specific, measured case.

## What this changes

- The "zombie problem" is two separate faults: z cannot localise anything small (Section 2), and imagination
  cannot predict whether a move is blocked (Section 3). The second one affects U and W too and costs ~0.4.
- The information the transition needs exists at the encoder's output (patch tokens, 0.97) and is discarded
  by the state interface. A state that keeps per-tile resolution — Dreamer 4 uses 256 spatial tokens — is the
  direct target.
- SLEEP is a data artefact that any head trained on factual corpus deaths will inherit. It is visible only
  through counterfactual roots such as these.
