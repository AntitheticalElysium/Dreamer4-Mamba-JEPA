# P1-F: guided fixation + faithful pixel decoder, from scratch (plan 10-30; revised 10-31 after Codex's review of
3d5eea60 and the user's decision to switch in the faithful decoder)

## Question
User, 10-30 / 10-31: does a world trained from the FIRST update with fixation guided to consequence attempts AND a
pixel decoder whose mean is not starved by its learned scale come out better than P1 world 1? The fovea stays
(evidence stream later, VoI); this tests its training role while representations and dynamics first form, with the
grounding repaired. User, 10-31: two changes at once make the experiment less clean, but maximise the chance of a
better model; ablations later, once something works.

- **Claim.** P1's recipe with (i) the attempt frames fixated by M0.5's rule (Q = 1, 13% of frames) and (ii)
  `heads.pixel_decoder.mean_loss: faithful` yields a world whose generated futures carry more of the actor-relevant
  consequences than P1 world 1's, and whose pixel grounding renders HUD values.
- **Attribution.** The comparison is of the COMBINED recipe against world 1: an improvement cannot be assigned to
  guidance or to the decoder (Codex). A negative result would not settle whether the fovea can work.
- **Expectations declared now** (descriptive, no gate). Decoder: HUD digits and icons rendered (frozen-token refits:
  95-100% on both paths), rare blocks and zombies better. From M0.5 at the actor interface: more mob interactions
  (defeat zombie, eat cow, skeleton), more first units (wood, sapling). Risks: crafting unlocks (M0.5: stone pickaxe
  -0.20, wood pickaxe -0.08, table -0.05), drink (-0.11); death not expected to move; common-block sharpness (refits:
  ~0.5-1 grey level); faithful's squared error does not down-weight night static, so the encoder may spend glimpse
  capacity on it (read: night token statistics, S1.1 night recall); the grounding terms change magnitude, so the
  harmoniser re-weights G2 and the encoders' grounding share changes (read: the encoder gradient norms logged at
  every checkpoint, beside world 1's).
- **Competing explanations.** (a) Seed / GPU-nondeterminism variation between two P1 trajectories: P1 world seed 2
  (random) is the planned reference for the random arm's spread; a guided replicate for the guided arm's (Codex). (b)
  The rule's window as a cue: read under BOTH gaze policies (below); a gain confined to the guided policy stays
  policy-specific and does not resolve cue exploitation (Codex).
- **Control.** P1 world 1 (`artifacts/saccade_m0/p1_world1`): same config apart from the two declared changes
  (asserted on first launch against its checkpoint), world seed 1, initialization, episode order, Streams random
  draws, sigma / noise draws, schedule (25.6 M frames, 64-step then 256-step chunks), checkpoints and monitor.

## Build
- `saccade` (10-31): `heads.pixel_decoder.mean_loss` (m0.yaml; default nll = world 1's recipe, LOSSES G2 row).
  faithful: TSP's and the reconstruction's means by squared error in [-1, 1] (the G2 terms); the shared scale by the
  discretised logistic NLL on the DETACHED means with the lighting latent DETACHED in the scale's input (`noise`,
  weight 1, outside the harmonised groups, reaching only the scale network: Stirn et al.'s second modification, the
  route Codex found through the lighting latent closed).
- `train_guided.py`: `saccade.train.p1` itself, with `saccade.data.Streams` replaced by v2's GuidedStreams bound to
  M0.5's placement and `--decoder faithful`. Identity (Codex's finding 1): the guided placement's identity and the
  decoder choice are IN the config, which every checkpoint stores and `validate_resume` compares (with source, corpus
  and target hashes); on first launch the config minus the two declared changes must equal world 1's checkpoint
  config; a directory holding a checkpoint without an identity file is refused.

## Checks before launch (results)
1. Streams (CPU, `check_streams.json`, 10-30): PASS. 552,960 frames: unguided records identical to D.Streams; guided
   records differ only in the fixation of the 13.10% placed frames, each the rule's window. (Codex reproduced the
   regeneration independently: 319,344 placed frames, the pinned array exactly.)
2. Match after the saccade change (GPU, 10-31): PASS. `--unguided --decoder nll` reproduces world 1's logged update 1
   in every field (grad norm 13.07340145, g1 0.94510591, change, tsp, rec, reward, cont): the option is inert by
   default. (Codex also matched world 1's update 25 on the earlier code.)
3. Identity (GPU, 10-31, Codex's reproduction): PASS. World 1's ckpt_10 as latest.pt in a fresh directory: refused
   ("a checkpoint without an identity file"); with an identity file from a fresh guided start: refused by
   validate_resume ("Resume configuration or world seed differs").
4. Faithful routes (GPU, `check_faithful.json`): PASS. The noise term reaches only the scale network (gradient norm
   0.258; mean head, adapters, encoders including the lighting latent, belief, denoiser: 0); the squared-error terms
   never reach the scale. Encoder gradient per loss at initialisation, faithful vs nll on the same chunk: TSP 0.023 vs
   0.041, reconstruction 0.056 vs 0.044 (same order; the harmoniser re-weights over training).
5. TSP under each loss (frozen checkpoint-10 inputs, `../20261030_decoder_diagnosis/refit_tsp_results.json`): faithful
   renders every HUD value on TSP's path (digits 1.000, icons 1.000) and rare content best; beta_nll partial (0.78-0.94).
Launch: `systemd-run --user --unit=saccade-p1f -p MemoryMax=24G --setenv=PYTHONPATH=$PWD --working-directory=$PWD
$PWD/.venv/bin/python -m artifacts.experiments.20261030_p1_guided.train_guided artifacts/experiments/20261030_p1_guided/run`
(training only: evaluation is run by hand afterwards, Codex's operational note).

## Readings after the run (declared now; descriptive, intervals over evaluation rows only)
Checkpoint by P1's own selection rule, against P1 world 1 checkpoint 10, both read the same way:
1. Actor interface (`../20261030_actor_interface/interface.py`, corrected 10-31, primary), under BOTH gaze policies
   (`--policy=random`, `--policy=guided`) for both worlds: reward > 0.5 on both sides; held-out per-history reliability
   for unlock, damage and death; context-defined death / damage; per-achievement outcome-selected rows as sensitivity
   diagnostics only; the outcome split as description only.
2. S1 (the design's readings; S1-seeded random fixations by design): one-step C2ST, entity counts, belief probe, TSP.
3. M0.5's evaluator (it has both a guided and a random route): glimpse / periphery consequence rates, perception,
   regression panel; the dissection's glimpse token geometry.
4. Decoder rendering of the trained world, both paths (`../20261030_decoder_diagnosis/refit.py`'s evaluate): HUD
   digits / icons, blocks, mobs, day / night.
5. P1's monitor curve and per-loss encoder gradient norms at every checkpoint, against world 1's.

## Limits
One seed per arm. The faithful decoder was demonstrated on frozen checkpoint-10 inputs only (one seed per arm); its
noise model's calibration is not established (Codex), though night likelihood matched the NLL arms (3.03-3.14
nats / pixel); nothing downstream uses the scale (TSP and the reconstruction are training-time grounding, never used in
imagination). Neither world 1 nor seed 2 is a matched control for either change alone. ~9.5 h (world 1: 9 h 22 min).
