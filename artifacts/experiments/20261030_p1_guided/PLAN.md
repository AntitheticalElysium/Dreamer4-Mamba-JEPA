# Guided P1 from scratch (plan, 2026-10-30; for Codex's review before launch)

## Question
User, 10-30: does fixation guided to consequence attempts from the FIRST update shape a better world than P1 world 1,
where M0.5 only fine-tuned a converged checkpoint? The fovea stays (evidence stream later, VoI); this tests its
training role while representations and dynamics first form.

- **Claim.** P1's recipe with the attempt frames fixated by M0.5's rule (Q = 1, 13% of frames) yields a world whose
  generated futures carry more of the actor-relevant consequences than P1 world 1's.
- **Expectations declared now** (from M0.5 at the actor interface; descriptive, no gate): more mob interactions
  (defeat zombie, eat cow, skeleton), more first units / appearances (wood, sapling); risks: crafting unlocks (M0.5:
  stone pickaxe -0.20, wood pickaxe -0.08, table -0.05), drink (-0.11); death not expected to move (M0.5: ~0).
- **Competing explanations.** (a) Seed / GPU-nondeterminism variation between two P1 trajectories: one guided run
  against one random run cannot separate it; P1 world seed 2 (random) is the planned reference for that spread, and
  a guided replicate for the guided arm's (Codex). (b) The rule's window is itself a cue (it is decided from the
  action, faced block, zombie adjacency and sleep at t - 1): the readings below use RANDOM fixations (S1-seeded), so
  the cue is absent at evaluation; a world that leans on it would read worse, not better.
- **Control.** P1 world 1 (`artifacts/saccade_m0/p1_world1`): same source (hashes of saccade/model.py, train.py,
  data.py, m0.yaml identical to its provenance), config, world seed 1, initialization, episode order, Streams random
  draws, sigma / noise draws, schedule (25.6 M frames, 64-step then 256-step chunks), checkpoints and monitor. The
  ONLY difference: on frames whose state t - 1 is an attempt context of the rule, the fixation is the rule's window
  instead of the prior's draw; the prior is drawn first on every frame (v2's GuidedStreams), so every later random
  draw is unchanged.

## Build (`train_guided.py`)
`saccade.train.p1` itself, with `saccade.data.Streams` replaced by v2's GuidedStreams bound to M0.5's placement file
(`../20261030_m05_guided_fovea/place_q1_train.npy`, the rule of `../20261030_fovea_placement/map.json`, Q = 1).
No change to the saccade package. An identity file (git commit, script / GuidedStreams / placement / map hashes, the
recipe's source hashes) is written at the first start and asserted on resume.

## Checks before launch
1. Streams (CPU): GuidedStreams with no placement emits records identical to D.Streams over chunks of 64 and 256
   steps; with the placement, records differ only in the fixation of placed frames; placed share ~13%.
2. Match (GPU): the launcher with the placement disabled reproduces P1 world 1's logged update 1 (learning rate 0
   there: the losses are a function of initialization and data only).
3. Smoke (GPU): 2 updates with the placement.
Results (10-30): (1) PASS, `check_streams.json`: 360 chunks (300 x 64 steps, 60 x 256), 552,960 frames: unguided
records identical to D.Streams; guided records identical except the fixation on the 13.10% placed frames, each equal
to the rule's window. (2) PASS, `check_match_update1.jsonl`: update 1 with the placement disabled equals P1 world 1's
logged update 1 in every field (grad norm 13.07340145, g1 0.94510591, change, tsp, rec, reward, cont). (3) PASS,
`check_smoke_update1.jsonl`: 2 guided updates run; update 1 differs where the glimpse windows differ (rec / TSP 5.05
vs 5.61, g1 0.94477 vs 0.94511), peak 3.9 GiB.
Launch (after review): `systemd-run --user --unit=saccade-p1-guided1 -p MemoryMax=24G --setenv=PYTHONPATH=$PWD
--working-directory=$PWD $PWD/.venv/bin/python -m artifacts.experiments.20261030_p1_guided.train_guided
artifacts/experiments/20261030_p1_guided/run`

## Readings after the run (declared now; descriptive, intervals over evaluation rows only)
Checkpoint by P1's own selection rule, against P1 world 1 checkpoint 10:
1. Actor interface (`../20261030_actor_interface/interface.py`, primary): per-achievement unlock on generated vs
   recorded successors, false unlocks, death and damage in risky / adjacent contexts.
2. S1 (the design's readings): one-step C2ST, entity counts, belief probe, TSP.
3. M0.5's evaluator (glimpse / periphery consequence rates, perception, regression panel) and the dissection's glimpse
   token geometry (does guidance from the start change how the glimpse encodes values?).
4. P1's monitor curve, guided vs world 1.
Not attributable to guidance without the seed-2 reference (a); one seed per arm.

## Limits
Decoder digit-blindness unchanged (not addressed tonight). Costs ~9.5 h (world 1: 9 h 22 min). If the pixel grounding
or any other component changes later, neither world 1 nor seed 2 remains a matched control for that version (Codex).
