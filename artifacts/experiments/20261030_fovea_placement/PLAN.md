# Guided fovea placement: does looking teach the world? (plan, 2026-10-30)

Not run: Codex reviews the code and this plan first (user, 10-30).

## Question
User, 10-30: if the fovea looked where the world's missed consequences land, chosen before the outcome, would the
world learn them, also when the eye is elsewhere? This is a substitution test of the premise behind a learning-progress
fovea (the eye goes where looking makes the world improve). It does not test that chooser.

- **Claim.** Directing glimpses to the cells where drink, wood, sapling, table and zombie-hit consequences land, on
  frames whose context makes them possible, makes P1's denoiser generate them, in its periphery output too.
- **Competing explanations.** (a) Any further training at a restored learning rate improves them (P1 ended at its 1e-5
  floor with every step clipped, accounting H). (b) What placement teaches stays in the glimpse output and does not
  carry over to the periphery output imagination uses when the eye is elsewhere. (c) The extra glimpses on consequences
  cost the bulk, movement or entities what they gain.
- **Control.** The same fine-tune with the random prior (arm R) against the guided rule (arm G). Same base, data order,
  noise draws, updates, learning rate and clipping; only the fixation of attempt frames differs. Evaluation on identical
  inputs for both arms, crossed: eye elsewhere (random fixations) and eye on the consequence (rule fixations).
- **No loss reweighting.** The loss is P1's G1 term unchanged; placement only decides which glimpse tokens exist (user,
  10-30).

## Fine-tune (`finetune.py`; one seed, user 10-30)
- Base: P1 world seed 1, checkpoint 10 (model, AdamW moments, stream state, Mamba carry, torch RNG).
- Only the denoiser trains, on P1's own G1 term (`train.p1_losses`) times the checkpoint's 1 / sigma_G1 (2.42), clipped
  at 1.0 over the denoiser. Encoder, belief, heads and harmoniser are frozen: the targets stay fixed (no shortcut through
  the encoder making its targets easier, Codex), the readers stay valid for every checkpoint, and the deficit is the
  denoiser's (generation diagnosis A, E).
- Learning rate 5.8e-5, P1's own at checkpoint 5 where wood, drink and table still had drive (accounting H); linear from
  the checkpoint's 1e-5 over the first 100 updates, then constant.
- 1,500 updates of 16 streams x 256 steps in 4 microbatches (P1's second half): 6.1 M frames, 24% of P1. Checkpoints
  at 300 (pilot read, descriptive only; the run does not stop on it), 750 and 1,500, each followed by P1's fixed monitor.
- Order: R to 300, G to 300, evaluation at 300; then both to 1,500 and the final evaluation.

## The guided rule (`placement_map.py`; privileged: built from simulator labels, Codex)
- Targets: the five consequences the field misses (A, F, G, H): drink + 1, wood + 1, sapling + 1, table placed, zombie
  hit, on the cells `loss_accounting.cell_classes` assigns them.
- The context of frame t comes from state t - 1 only, in the denoiser's conditioning: action a_{t-1}, faced block,
  zombie at Manhattan distance 1, sleeping.
- On the train split (monitor episodes excluded): per context, frames n and, per window of the prior's 49, the frames
  whose target cells it covers. An attempt context has n >= 100 and a best window covering a target on more than the
  base rate of target frames. Its window is that best one; ties go to the window centred nearest its target cells.
- Arm G: an attempt frame takes its window with probability 1/2 (one fixed coin per frame), otherwise the prior's draw.
  Every other frame keeps the prior's draw. The prior is drawn in both arms, so everything else matches.
  - Why half, not all: the agreed random floor; attempt contexts with the eye elsewhere stay in G's training, so the
    periphery route is in distribution for both arms; exposure of the consequences in the glimpse still rises several
    fold (map.json reports by how much).
- Not where the model's error is: that is known only after the outcome, and the denoiser is conditioned on the fixation,
  so it would learn "the change is where I look" (E14: weighting successes only hallucinated 24-30%).
- Manipulation check, logged during training: the share of target-consequence frames whose cells the glimpse covered,
  per arm (G must exceed R clearly; otherwise nothing is read).
- The map as built (`map.json`, from the full train-split labels, 2.43 M frames): base rate of target frames 6.6%; 129
  attempt contexts holding 13.1% of frames and 95% of target frames; 6.6% of frames guided. Windows chosen: the drink
  counter (7, 2), wood (7, 5), sapling (7, 4), health (7, 1) for adjacent-zombie contexts, and the player-centred (3, 4)
  for PLACE_TABLE (it covers all four faced cells). Predicted coverage of target frames, prior -> guided: drink 0.12 ->
  0.56, wood 0.12 -> 0.56, sapling 0.12 -> 0.55, table 0.18 -> 0.59, zombie hit 0.04 -> 0.41. The largest context is DO
  facing grass (80 k frames, sapling on 7.4%): mostly matching negatives at the sapling counter.

## Readouts (`evaluate.py`)
Dev split; the first 1,000 rows per class of tests F (move, blocked, wood, sapling, table, drink) and A (adjacent,
risky, safe, random), as test G; the same rows, noise draws and one shared encoding for every checkpoint (base, R and G
at 300 and 1,500); only the denoiser differs, which is asserted.
1. **Periphery route (primary).** S1-seeded random fixations. Field projection at sigma 0.05 (instrument check: |bias|
   < 0.01), 0.7, 2 and 6.67, the production projection, and production samples read by a qualified reader. Primary
   subset: eye elsewhere (the target frame's glimpse does not cover the consequence cell).
2. **Glimpse route.** Every attempt frame of dev fixated by the rule (all, not half); the same readouts (primary subset:
   eye on the consequence cell), plus the production glimpse tokens at that cell: a projection on the real start / end
   value means of that cell and window (read only if, on real tokens, it reproduces the recorded rate within 0.05) and
   a reader. There is no field readout on glimpse tokens (the instruments project periphery tokens).
3. **Readers.** Periphery HUD values (drink, wood, sapling, health): nearest value mean of real train-400 tokens. Glimpse
   HUD values, per cell and window: a multinomial logistic reader on real train-400 glimpse tokens (the fovea mixes its
   crop, so a token also varies with the neighbouring counters: a nearest-mean reader reached only 0.34-0.85 in the smoke
   test). Every reader is qualified on dev real tokens: below 0.99 accuracy it is not read. Faced-cell table: the S1
   reader's terrain head; its dev real-token rate is reported beside the recorded one.
4. **Hallucination.** A's random rows: a generated change at each HUD cell, or a table on the faced cell, where the
   recorded cell did not change.
5. **Costs.** P1's monitor G1 by level (fine-tune logs), dev G1 loss per element by accounting class, MOVE / BLOCKED,
   generated entity counts (S1.3 one-step, 5,000 tuples, S1 reader).

## Declared readings
Primary quantity: G - R on the periphery route, eye elsewhere, paired by rows (episode-bootstrap SE), at update 1,500:
the field projection at sigma 6.67 and the reader's production success rate.
- **Minimal effect of interest** per deterministic class: a quarter of the base's gap, 0.25 x (recorded - base field at
  6.67): drink ~0.20, wood ~0.19, table ~0.13 (test G's checkpoint 10).
- **Positive.** G - R reaches the minimal effect and exceeds 2 SE on at least two of drink, wood and table, with
  hallucination on those cells up by no more than 0.01 absolute. Costs are reported beside it, not gated.
- **Glimpse only.** Route 2 positive, route 1 not: the learning does not carry over to the periphery within this budget,
  so a chooser must be rewarded for improvement that persists after the eye moves away (Codex).
- **Bounded null.** Both routes below the minimal effect, with the manipulation check passed and arm R itself moving
  something measurable (monitor or any class): "guided placement on half the attempt frames changes the learning of
  these consequences by less than the minimal effect within 1,500 updates at this learning rate". Not: allocation is
  irrelevant, or a cause for the deficit (Codex).
- **Manipulation failure.** Nothing is read.
- Sapling (stochastic, rate 0.1) and damage (hidden cooldown) are read descriptively: whether the field approaches the
  recorded rate.

## Limits
- Privileged rule (labels), one seed, one base checkpoint, frozen encoder and belief, half placement, one budget and
  learning rate. A positive result says guided placement can teach this model these consequences. It does not show that
  a label-free chooser would find these places, nor what joint training would do.
- Training noise is not in the standard errors. The SE is an episode bootstrap over evaluation rows. The GPU's
  nondeterministic arithmetic makes two runs of the same arm differ like a small seed change: after 4 updates two
  straight runs differ by 1.8e-5 in the denoiser, against 1.8e-4 moved from the base. A positive result is therefore
  provisional until replicated, which is the plan (user, 10-30: one seed first, decide on a positive).

## Checks done before review (no experiment run)
- Labels: the full train split replayed (8,069 episodes incl. monitor, 2.56 M transitions); its first 400 episodes equal
  `s0_train400_labels.npz` exactly.
- Fine-tune smoke (3 updates per arm): same frames and target frames in both arms; coverage of target frames 0.08
  (random) vs 0.53 (guided); 4.8 s per update, 3.1 GiB peak (1,500 updates ~2 h per arm).
- Exact resume: 4 straight updates vs 2 + resume + 2: identical stream state, counters and CUDA RNG; denoiser
  difference 1.78e-5, the same as between two straight runs (1.79e-5), i.e. GPU nondeterminism, not the resume.
- Evaluation smoke (30 dev episodes, 50 rows, base + an unchanged copy + a perturbed copy): the unchanged copy
  reproduces the base exactly on both routes (matched rows and noise); base values agree with test G's checkpoint 10
  (wood field 0.25 vs 0.24); periphery HUD readers 1.00 on dev real tokens; the table reader matches the recorded rate
  (0.98 vs 0.98).

## Cost
GPU, one job at a time: fine-tune ~2 h per arm; evaluation ~1 h for three checkpoints (base, R, G) at 300 and again at
1,500. About 6 h in all.
