# Full-frame change correction and timer diagnosis

2026-11-08, user-authorized implementation and bounded diagnostics. Historical checkpoints and their stored
teachers remain untouched. No overnight training is authorized by this work.

## Questions and stopping rules

1. Remove semantic map/HUD/player exclusions from canonical change supervision. A full-frame target must distinguish
   screen-fixed and camera-moving content without fixed region rules. First audit a generic local-surround choice
   between identity and the frame's dominant translation. The center cell cannot choose its own correspondence:
   otherwise moving objects can be explained away. Evaluate the existing detector on held-out actual HUD changes,
   unchanged HUD cells, and map entity/terrain changes. If this fails, do not call a shape-only extension a fix.
2. Establish whether recent observations can uniquely determine timer events using natural exact-history aliases.
   Compare raw full-resolution frame windows plus actions/rewards/termination, not probes of lossy embeddings.
   Conflicting next outcomes under identical observable histories prove that recent evidence is insufficient on
   that subset. No conflicts do not establish sufficiency. Use only dev episodes; no final split.
3. If natural aliases provide enough support, compare native world predictions on them under real episode prefixes,
   without contradictory cache/memory swaps. Reuse qualified digit readers, common generation randomness and a
   recent-history-only bound. If aliases are absent, report the bounded search result rather than replace it with
   another unqualified negative probe.

The hidden hunger/thirst/fatigue/recovery clocks are NOT displayed. Their visible resets and body digits are.
Full-frame change grounding can improve those cues, but recognizes observed changes and does not itself require
anticipation. Existing data cannot causally establish the effect of supervision omitted throughout training;
that attribution requires a matched learning comparison.

## Controls

- Existing old teacher on its unchanged correspondence, reproduced before comparison.
- Full-screen translations, stationary overlays, independently moving objects, and a digit change during camera
  movement; no semantic region is supplied to the new algorithm.
- Actual dev daytime/nighttime changes separated; simulator labels are evaluation only.
- Exact-history matches verified with array equality after hashing, never approximate nearest neighbours.
- No production checkpoint migration or silent reuse of 7x9 target arrays.

## Amendments, recorded after candidate checks

Local winner selection and both detector refits failed the map-preservation control and are rejected. Final
correction supervises BOTH same-position and dominant-translation comparisons over the whole screen, using the
original detector with exact-identity zero correction. The head adds one output channel; metadata and target
validation change to9x9x2. Historical checkpoints load for evaluation only by extending this unused auxiliary;
bitwise generation is explicitly checked, rather than assumed. A new training run is not a historical resume.

Final qualification is a separate80-episode slice (480--559). Same-input historical scores are reused only for
constructing the full new array, with F4's old target hash/data configuration verified and direct recomputation
checks. This is a computation cache, not a mask controlling new supervision.

The expanded search found one natural thirst alias in episodes320--479. Its exact recent histories are verified
with direct array equality and native predictions evaluated under matched fixations; no candidate was selected
by its generated result. This supplements, rather than replaces, the constructed24-scene hunger test.

## Qualified fallback after the exact-match search

The first 160 dev episodes contain no exact recent-history duplicates. That is a search failure, not evidence that
recent observations suffice. Use a simulator construction instead: same real dev background, two hidden hunger
phases, identical NOOP actions and RNG keys. After each branch has visibly decremented food once, select a later
step where the next decrement differs but the last eight FULL frames and incoming rewards/terminations/actions
are exactly identical. Older observed food decrements reveal the phase. Every pair must replay the unedited dev
background exactly, have identical recent observations, and have different simulator-confirmed next food values.

Run the native frozen F4 with the actual full history of each branch and the same random draws. A next-food reader
must first read both real successors correctly. Primary result is the native difference in decrement probability
versus the deterministic true contrast. This avoids swapping contradictory memories/caches or asking a probe to
establish that the recent window is insufficient. A weak native contrast establishes a failure to exploit older
visible evidence on these constructed histories, not whether the memory has irreversibly erased it or which
training loss caused it. Backgrounds are dev only; this is a controlled diagnostic, not an unbiased population rate.
