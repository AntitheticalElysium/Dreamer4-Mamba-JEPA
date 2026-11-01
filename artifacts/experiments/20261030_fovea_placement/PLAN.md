# Guided fovea placement: does looking teach the world? (plan, 2026-10-30; v2 after Codex's review)

Status: Codex reviewed v1 (commit d46471e8; `../20261030_fovea_placement_review/REVIEW.md`): the training mechanics
passed, the evaluator needed corrections (all accepted, section "Changes after Codex's review"). The guided arm was
launched first on the user's instruction (10-30, 09:38): its code is the reviewed v1, which the review left unchanged.

## Question
User, 10-30: if the fovea looked where the world's missed consequences land, chosen before the outcome, would the
world learn them, also when the eye is elsewhere? This is a substitution test of the premise behind a learning-progress
fovea (the eye goes where looking makes the world improve). It does not test that chooser.

- **Claim.** Directing glimpses to the cells where drink, wood, sapling, table and zombie-hit consequences land, on
  frames whose context makes them possible, makes P1's denoiser generate them, in its periphery output too.
- **Competing explanations.** (a) Any further training at a restored learning rate improves them (P1 ended at its 1e-5
  floor with every step clipped, accounting H). (b) What placement teaches stays with the guided conditioning or the
  glimpse output and does not reach the periphery output imagination uses when the eye is elsewhere. (c) The extra
  glimpses on consequences cost the bulk, movement or entities what they gain.
- **Control.** The same fine-tune with the random prior (arm R) against the guided rule (arm G). Same base, data order,
  noise draws, updates, learning rate and clipping; only the fixation of attempt frames differs. Evaluation on identical
  inputs for every checkpoint, crossed: eye elsewhere (random fixations) and eye on the consequence (rule fixations).
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
  at 300, 750 and 1,500, each followed by P1's fixed monitor.
- Order (user, 10-30): **arm G first**, to 1,500; stage 1 compares it with the base; arm R runs only on a stage-1
  positive (below).

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
    fold.
- Not where the model's error is: that is known only after the outcome, and the denoiser is conditioned on the fixation,
  so it would learn "the change is where I look" (E14: weighting successes only hallucinated 24-30%).
- Manipulation check, logged during training: the share of target-consequence frames whose cells the glimpse covered
  (union of the five; G must exceed R clearly, otherwise nothing is read).
- The map as built (`map.json`, full train-split labels, 2.43 M frames): base rate of target frames 6.6%; 129 attempt
  contexts holding 13.1% of frames and 95% of target frames; 6.6% of frames guided. Windows: the drink counter (7, 2),
  wood (7, 5), sapling (7, 4), health (7, 1) for adjacent-zombie contexts, and the player-centred (3, 4) for
  PLACE_TABLE (it covers all four faced cells). Predicted coverage of each class's OWN cells, prior -> guided (corrected
  after Codex: v1's map.json credited a simultaneous other consequence; verified independently): drink 0.122 -> 0.561,
  wood 0.122 -> 0.561, sapling 0.122 -> 0.547, table 0.184 -> 0.578, zombie hit 0.041 -> 0.409. The largest context is
  DO facing grass (80 k frames, sapling on 7.4%): mostly matching negatives at the sapling counter.

## Readouts (`evaluate.py`)
Dev split; the first 1,000 rows per class of tests F (move, blocked, wood, sapling, table, drink) and A (adjacent,
risky, safe, random), as test G; the same rows, noise draws and one shared encoding for every checkpoint; only the
denoiser differs, which is asserted.
1. **Periphery route, eye elsewhere (primary).** S1-seeded random fixations; rows whose target-frame glimpse does not
   cover the consequence cell. Periphery outputs: field projection at sigma 0.05 (instrument check: |bias| < 0.01), 0.7,
   2 and 6.67; production projection; production samples read by a qualified reader.
2. **Guided conditioning, eye on.** Every attempt frame of dev fixated by the rule (all, not half). The SAME periphery
   outputs under the guided fixation history and target fixation (Codex: a difference between 1 and 2 shows dependence
   on the fixation conditioning, not a glimpse-only effect).
3. **Glimpse output, eye on.** The production glimpse tokens at the consequence cell, read only by qualified glimpse
   instruments: a multinomial logistic reader per (cell, window), and a ridge value readout (v_hat - start) /
   (end - start) qualified per class (its mean on the class's real eye-on rows within 0.05 of the recorded rate). v1's
   mean-based glimpse projection failed this test on real data (wood 0.926, drink 0.944; Codex) and is replaced; the
   tolerance is unchanged. Instruments that fail make the reading unmeasured. **No glimpse instrument exists for the
   table: unmeasured.**
4. **Failed attempts (the hallucination test).** Attempts whose cell did not change: drink (DO facing water, counter
   unchanged; 141 contexts on dev), table (PLACE_TABLE, no table; 4,097, first 1,000 used), wood (DO facing a tree,
   unchanged; 9 contexts, descriptive only). Reading: the generated-change rate by the qualified reader, per stratum,
   on both routes. A's random rows remain as a population diagnostic only (Codex: they hold one failed drink).
   Sapling and damage are stochastic: a correct generator emits them at their rate on histories whose recorded outcome
   is negative, so they are read as calibration (generated vs recorded rate), not as hallucination.
5. **Readers.** Fitted on train-400 real tokens, qualified on dev real tokens before any model is read: periphery HUD
   values (nearest value mean), accuracy >= 0.99; faced-cell table (S1 reader's terrain head), table recall and
   non-table specificity >= 0.99; glimpse readers as in 3. Enforced row by row: a reading exists only where its
   instrument is qualified and the row's start and end values are in its support; excluded rows are counted.
6. **Costs.** P1's monitor G1 by level (fine-tune logs), dev G1 loss per element by accounting class, MOVE / BLOCKED,
   generated entity counts (S1.3 one-step, 5,000 tuples, S1 reader).

## Declared readings
Differences are paired by rows, with episode-bootstrap SE and a 95% interval (est +- 1.96 SE); the SE covers
evaluation rows only (Limits). Per deterministic class (drink, wood, table), two minimal effects of interest, from
this evaluation's base, eye elsewhere:
- projection: MEI_p = 0.25 x (recorded projection - base field projection at sigma 6.67) (test G: drink ~0.20, wood
  ~0.19, table ~0.13);
- event probability: MEI_r = 0.25 x (recorded rate - base reader production success).
A class is POSITIVE when the field-projection difference at 6.67 is >= MEI_p with its 95% interval above 0, AND the
reader production-success difference has its 95% interval above 0. A class is a BOUNDED NULL when the upper end of
the field-projection interval is below MEI_p and the upper end of the reader interval is below MEI_r. Otherwise it is
INCONCLUSIVE.

**Stage 1: G (update 1,500) vs base, primary route.**
- Stage-1 positive: at least two of drink, wood and table POSITIVE; the failed-attempt false-change rate (drink, table)
  not up by 0.05 or more (upper end of its 95% interval below 0.05). Then arm R runs to 1,500 (same code and settings).
- Stage-1 null: all three classes BOUNDED NULL: arm R is not run. Reading: "guided placement on half the attempt frames,
  plus the restored learning rate, does not raise the generation of these consequences by the minimal effect within
  1,500 updates". Caveat: if fine-tuning alone would degrade them, placement could still protect them; that is a
  different claim from teaching and is not tested.
- Otherwise inconclusive: reported to the user, who decides on R.
- Pilot reads at 300 and 750: descriptive only.

**Stage 2 (only if R runs): G - R at 1,500, primary route,** by the same class rule.
- **Positive:** at least two of drink, wood and table POSITIVE, with the failed-attempt false-change rate of G - R
  below +0.05 at the upper end of its interval. Costs are reported beside it, not gated.
- **Conditioning dependence:** route 2 positive, route 1 not: the gain shows only under the guided fixation history,
  so it is tied to that conditioning.
- **Glimpse output only:** route 3 positive (qualified instruments) while route 1 is not: the glimpse output learned
  the consequence and the periphery output did not; a chooser must be rewarded for improvement that persists after the
  eye moves away (Codex). Unmeasured for any class whose glimpse instruments fail.
- **Bounded null:** all three classes BOUNDED NULL on route 1, the manipulation check passed and R moved something
  measurable (monitor or any class): "guided placement on half the attempt frames changes the learning of these
  consequences by less than the minimal effect within 1,500 updates at this learning rate". Not: allocation is
  irrelevant, or a cause for the deficit (Codex).
- **Manipulation failure:** nothing is read.
- Sapling (rate 0.1) and damage (hidden cooldown): calibration, descriptive.

## Limits
- Privileged rule (labels), one seed, one base checkpoint, frozen encoder and belief, half placement, one budget and
  learning rate. A positive result says guided placement can teach this model these consequences. It does not show that
  a label-free chooser would find these places, nor what joint training would do.
- Training noise is not in the standard errors. The SE is an episode bootstrap over evaluation rows. The GPU's
  nondeterministic arithmetic makes two runs of the same arm differ like a small seed change: after 4 updates two
  straight runs differ by 1.8e-5 in the denoiser, against 1.8e-4 moved from the base. A positive result is therefore
  provisional until replicated, which is the plan (user, 10-30: one seed first, decide on a positive).
- Stage 1 compares G with the base, so it cannot separate placement from the restored learning rate; only stage 2 can.

## Changes after Codex's review (10-30; each verified, all accepted)
1. Glimpse instruments: v1's projection failed its own tolerance on real data and was reported anyway; the wood glimpse
   reader at (7, 5) is at 0.989 (< 0.99) for 989 of 1,000 wood contexts; the table had no glimpse readout. Now: ridge
   readout replacing the projection, both glimpse instruments qualified on real data and enforced; the table's glimpse
   reading declared unmeasured. Tolerances unchanged.
2. Hallucination: failed-attempt strata added (counts reproduced: 141 failed drinks among 4,830 drinking attempts on
   dev); the random rows kept as a population diagnostic only; stochastic events read as calibration.
3. Routes: route 2 renamed "guided conditioning" (periphery outputs); the glimpse-output reading moved to qualified
   glimpse instruments only.
4. Qualification enforced row by row for every reader (an unqualified reader no longer yields readings; unsupported
   values are excluded and counted).
5. Readings: 95%-interval rules, separate projection and event-probability thresholds, inconclusive as a declared
   outcome; staged for the guided-first order.
6. Coverage: per-class predicted coverage on each class's own cells (numbers above; `placement_map.py` corrected; the
   map and placement files themselves are unchanged and are the ones in use by the running arm).

## Checks done before review (v1)
- Labels: the full train split replayed (8,069 episodes incl. monitor, 2.56 M transitions); its first 400 episodes equal
  `s0_train400_labels.npz` exactly.
- Fine-tune smoke (3 updates per arm): same frames and target frames in both arms; coverage of target frames 0.08
  (random) vs 0.53 (guided); 4.8 s per update, 3.1 GiB peak (1,500 updates ~2 h per arm).
- Exact resume: 4 straight updates vs 2 + resume + 2: identical stream state, counters and CUDA RNG; denoiser
  difference 1.78e-5, the same as between two straight runs (1.79e-5), i.e. GPU nondeterminism, not the resume.
- Evaluation smoke (v1, 30 dev episodes): an unchanged denoiser copy reproduces the base exactly on both routes;
  base values agree with test G's checkpoint 10; periphery HUD readers 1.00 on dev real tokens.
- v2 evaluator: smoke-tested after the guided arm finishes (one GPU job at a time), before any reading.

## Cost
GPU, one job at a time: arm G ~2 h (running); evaluation ~1 h (base, G at 300, 750, 1,500); arm R ~2 h and a final
evaluation ~40 min only on a stage-1 positive.
