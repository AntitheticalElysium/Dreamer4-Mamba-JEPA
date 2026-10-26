# Option 3 (working state + location-keyed episodic memory) — exploration notes, 2026-10-23

User decision: option 3 is the coherent direction; explore it rigorously before committing. Nothing here is a commitment.

## Facts (EDA; simulator replay, TRAIN split, evaluation-only state)
| | finding | source |
|---|---|---|
| M1 v2 | Expert (data-generating PPO-RNN): memory rebuilt from only the last 64 observations: achievements over 300 steps equivalent within +-0.15 at t0 = 200 / 500 / 1000; survival equivalent or slightly higher; water-seeking equivalent or faster. First-step action agreement 0.72-0.85. Last-16 / no memory: -0.14 / -0.16 achievements at t0 = 200 (margin crossed), equivalent later; no memory: +20.8 steps [3.1, 40.3] to water last seen 64-256 steps before, at t0 = 1000. Alignment check exact (0.0). | `eda_expert_amnesia_v2.py` |
| M2 | 16-step windows: entering terrain last seen never 41% / <=16 27% / 16-64 14% / 64-256 12% / >256 5%; within 23 cells of the start. Entities: ~97% of previously seen ones within 64 steps. | `eda_window_memory.py` |
| S1 | Move outcome = f(visible target cell, mob on it, asleep): rule accuracy 100% (402 episodes); successful moves change the 7x9 view 99.95% of the time. | `eda_selfloc.py` |
| S2 | Identical 7x9 view at a different position: 0.03% of frames. | same |
| S3 | Remembered terrain changed when revisited: 0% (gaps <= 256), 0.07% (> 256). | same |
| S4 | Entity displacement while out of view: zombies <=4 steps median 0 (65% unmoved), <=16 median 2 (p90 5), <=64 median 3 (p90 8); cows similar. | same |
| S5 | Re-entering terrain block: none 2.28 bits / 49%; visible neighbours 1.24 bits / 77%; memory 0.98 / 78% (includes never-seen cells); memory + neighbours 0.49 / 91%. | same |

Implications: in Craftax, self-location is exact from action + visible target (S1) and relocalisation from content is
unambiguous (S2); terrain memory never goes stale (S3); recall turns re-entering seen terrain from ~77% (context) into
~100%; entities need a position belief that widens with absence (S4). Competent behaviour does not use >64-step memory
(M1); imagination fidelity does (M2, S5). Caveat: because self-location is trivial in Craftax, Craftax cannot test the
robustness of learned self-location that robotics would need.

## Modelling families for option 3
1. **Factorised location + episodic store (TEM / TEM-t).** Location code g by learned path integration (action, gated
   by what the target cell shows), corrected by content->location retrieval; store binds g to content; imagination
   queries with imagined g. Store as attention over (g, content) pairs (TEM-t) or as fast weights (TEM Hebbian; TTT /
   Titans are the modern parametric version). General (non-grid, non-spatial relations); most parts.
2. **Latent graph (CSCG / transformer with discrete bottleneck).** Places = discrete latent states learned from
   observation/action sequences; aliasing resolved by multi-step and next-encoding prediction. CSCG learns each map from
   scratch; TDB generalises in-context with >5,000 training rooms; categorical inputs; no moving objects tested.
3. **Allocentric canvas with learned ego-motion (EgoMap, Neural Map, Active Neural SLAM).** The grid special case of 1:
   exact and cheap when motion is translation on a grid; needs projective transforms in 3D; no relocalisation in ANS.
4. **Position-encoded retrieval (Scene Memory Transformer, pose-attention).** Attention over stored observations with
   (relative) pose encodings: family 1 in transformer form when the pose comes from learned self-motion; the cited
   pose-attention work is GIVEN poses.
5. **Parametric fast memory keyed by location (Titans, TTT, Vector-HaSH).** f(location) -> content updated online;
   graceful capacity (Vector-HaSH: scaffold separate from content, no memory cliff).

## Design dimensions inside option 3 (open unless stated)
- D1 location: representation (continuous code / discrete state / grid index) and learning (path integration from action
  + visible blocking; correction from content). Craftax: easy (S1, S2).
- D2 what is stored and when: per-cell content latent; predictive state (MERLIN's "right format"); write every step vs at
  surprise / event boundaries.
- D3 read path: by location for prediction / imagination; by content for relocalisation; how recalled content enters the
  working state (cross-attention, prior for re-entering cells).
- D4 entities: object records with last-seen location and time, a position belief widening with absence (S4), identity
  matching on re-entry.
- D5 imagination: branch-local copy-on-write overlay, tagged imagined, never merged into real memory (source monitoring).
- D6 uncertainty: no entry -> sample from the prior, write to the overlay (WHITEBOARD row 4); old entry -> trust (S3).
- D7 training: real writes in teacher-forced steps, overlay writes in unrolled steps; losses on re-entering content.

## User decisions 2026-10-23 (round 2)
Family 1 with attention over (location, content) pairs; biologically inspired pruning later; may move towards family 5
(parametric fast memory) if cost requires. Entities: object records "seem cleaner", pending research + EDA. Self-location:
not a focus (untestable in Craftax), but the system must be coherent so it can later work with noisy movement.

## Entities vs places
Brain (verified): items (perirhinal -> lateral entorhinal) and space (parahippocampal -> medial entorhinal) converge in the
hippocampus, which binds item to context; MEC object-vector cells code allocentric distance/direction to objects from
the start in new environments (Hoydal 2019); LEC codes egocentric bearing of items (Wang 2018) and has object-TRACE cells
firing where an object used to be (Tsao, Moser & Moser 2013); object files keep identity of moving objects by
spatiotemporal continuity, persist after attention leaves, reactivate on re-attention (Kahneman, Treisman & Gibbs 1992);
tracking capacity ~4-5 (Pylyshyn & Storm 1988), VWM ~4 (Luck & Vogel 1997, contested).
EDA `eda_entities.py` (402 episodes): entities alive within 12 cells and seen within 64 steps: <=2 60%, <=4 93%, <=6 99.5%,
<=8 99.99%; re-entries with no other plausible same-type candidate 87-92% (one candidate 7-12%, two <1%); habitat
P(block | type): zombies grass .53 / path .25 / sand .23, cows grass .82 / sand .14, skeletons path .63 / grass .28;
hidden state at re-entry: zombies cooldown > 0 0.1%, damaged 1.4%; skeletons cooldown > 0 32%; cows damaged 3%.
Reading: entities are few, re-identified by continuity almost unambiguously, loosely habitat-bound (not place-bound),
and their per-entity hidden state matters mostly during interaction (and skeleton cooldown at re-entry).
