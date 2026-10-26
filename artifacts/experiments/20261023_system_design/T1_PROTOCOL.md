# T1 — which generic state structure can hold returning entities and revisited terrain? (PROTOCOL)

Written 2026-10-23 before any T1 code ran. A design test for the system discussion, not a candidate model.

## Question
In Craftax, more than half of all mob appearances are an already-seen entity re-entering view (median gap 2-5 steps;
`entity_stats.py`), and the view scrolls with the player. The user rejects hand-coded world coordinates as Craftax-specific.
Which generic structures can still hold (a) entities across exits, (b) terrain across revisits, (c) history-determined
counters? The structural alternatives are: no structure (one global vector); a screen-anchored per-cell state; content
retrieval by attention over past frames; object slots on top of the per-cell state; and a memory anchored by LEARNED
self-motion (no position labels; EgoMap / TEM style).

## Isolation
Perception and the target space are removed from the question: inputs and targets are the symbolic per-cell observation
(block id and mob type for each of the 7 x 9 view cells, vitals, inventory), obtained by exact replay. Every structure is
trained with the same proper objective (cross-entropy of the next observation given history and action), same width,
sequence length 64, batch, optimizer and updates. A structure that fails here with perfect perception cannot be rescued
by a better encoder; one that succeeds still has to be shown with pixels later.

## Structures (about 1M parameters each)
- `G` global: frame embedding -> 2-layer Mamba over time -> decoder of all cells.
- `S` screen-anchored: per-cell tokens + spatial transformer, one Mamba sequence per token position, spatial transformer.
- `A` retrieval: per-cell tokens; each token attends to all tokens of the last 16 frames (causal), plus spatial attention.
- `O` slots: `S` plus 8 recurrent object slots (slot attention over the tokens each step, read back by cross-attention).
- `E` learned ego-motion memory: `S` plus a 25 x 25 feature canvas shifted each step by a learned soft shift over the 9
  one-cell offsets (predicted from consecutive frames and the action), written at the centre, read into the tokens.
- `E*` (diagnostic only, uses simulator positions): `E` with the true shift. Upper reference for anchoring.
Seeds 2.

## Data
TRAIN-split support-v2 episodes (exact replay labels) for training; DEV-split episodes for evaluation. Labels are the
model's inputs and targets here (symbolic world model), never pixels. FINAL untouched.

## Measurements (DEV, teacher-forced next-step prediction within 64-frame windows)
- Mob re-entry: cells where a mob that was visible earlier in the episode (same entity, continuously alive) re-enters
  view; mean predicted probability of its type at that cell; by gap (1-2, 3-5, 6-15, 16+ steps).
- Mob first sight: the same for entities never seen before (not predictable from history; a reference).
- Terrain revisit: cells entering view whose world position was seen earlier in the window vs never seen: top-1 accuracy.
- Counters: NLL of vitals at steps where one changes.
- Overall NLL per field.

## Pre-declared readings
- A structure **holds returning entities** if its mean probability on re-entries exceeds its first-sight value by at least
  0.2 and is at least half of `E*`'s re-entry value, both seeds.
- **Anchoring is needed** if `E*` exceeds every non-anchored structure (G, S, A, O) on terrain-revisit accuracy and
  re-entry probability by at least 0.1, both seeds; **learned anchoring suffices** if `E` is within 0.05 of `E*` on both.
- Otherwise reported as observed.

## Limits
Symbolic inputs (perfect perception, no night darkness); teacher-forced (not open-loop imagination); small budget; the
structures are minimal instances of each family, not tuned systems.
