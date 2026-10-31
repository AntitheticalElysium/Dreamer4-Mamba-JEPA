# Overnight loss test: scale-weighted TSP, from initialization (11-01; agreed by the user, Codex and Claude 10-31)

Contract and readings: Codex's `../20261031_grounding_loss_review/OVERNIGHT_PROPOSAL.md` (the recommended candidate).
This note records the exact recipe, the pre-launch checks and the readings as they will be applied.

## Recipe
P1-F exactly (`../20261030_p1_guided/train_guided.py`: guided Q = 1 placement, faithful pixel decoder, world seed 1,
the same initialization, streams, RNG, budget of 25.6M frames, optimizer, clipping, encoder LR, checkpoints), with one
declared change, `heads.pixel_decoder.tsp_mean_loss: scale_weighted_nll` (saccade/train.py, 4 lines; absent = the
previous code path): TSP's mean teacher becomes `stopgrad(exp(clamp(log_scale, -7))) * discretized_logistic_NLL`,
Codex's BM arm (Seitzer et al. 2022's beta = 0.5 in a logistic-scale form, not their Gaussian loss). Fovea
reconstruction stays faithful squared error; scale fitting (NLL on detached means, detached lighting), G1, the change
map, reward and continuation unchanged. Launcher: `train_beta.py` (config minus the change asserted equal to P1-F's
checkpoint config; identity file; resume validation compares the stored config).

## Pre-launch checks (`checks.py` -> `checks.json`, `launch_checks.json`; all passed)
- Disabled path: HEAD's p1_losses vs the edited one without the key, at P1-F's initialization and at its checkpoint 5
  (with its Adam moments): every term, every parameter gradient and the parameters after one AdamW step at a nonzero
  LR, bitwise equal.
- Enabled path vs Codex's BM wrapper (`beta_fork.losses`): the same, bitwise.
- Routing: the TSP term reaches the same 57 modules in both forms; none of the pixel decoder's scale network.
- Resume (TR.p1 at a shrunken budget, a checkpoint every 2 updates): 4 straight updates vs 2 + resume + 2 bitwise equal,
  as are two straight runs in separate processes; a scale-weighted checkpoint is refused without the key, P1-F's with it.
- The real launcher, 25 updates: `--teacher faithful` reproduces P1-F's logged updates 1 and 25 exactly (24 nonzero-LR
  updates); the new teacher matches P1-F at update 1 in every term but TSP (5.15 vs 0.56, units of the treatment) and
  the gradient norm (8.70 vs 6.53; the update is clipped at 1 in both, so TSP's share of each step grows: part of the
  treatment, as Codex states); 1.34 s / update, peak 4.0 GiB, as P1-F.

## Run (11-01)
Unit saccade-beta-tsp, 00:08-09:32 (9 h 25 min), 25.6 M frames, clean exit. The monitor fell at every checkpoint
(0.0932 -> 0.0272; P1-F 0.1061 -> 0.0286), so P1-F's rule selects checkpoint 10. The readings run by hand in
`queue.sh` (unit beta-readings): S1; the actor interface under both gazes and M0.5's evaluator with P1-F re-run beside
this run (the same rows and draws; the re-run must reproduce P1-F's 10-31 numbers); the decoder render; Codex's
readings through `readings_codex.py` (its code unchanged, outputs redirected to readings/codex/).

## Claim, competing explanations, control
- **Claim.** From initialization, the scale-weighted TSP teacher lowers the renderer static carried by the periphery
  tokens at night and improves generated physical outcomes against P1-F, while keeping faithful's digit / icon
  grounding.
- **Competing.** (1) Noise falls, physical generation does not improve: filtering nuisance is not sufficient. (2) Gains
  appear only at the rule's crop: the gate (a separate deficiency) is untouched. (3) Noise does not fall from scratch:
  the 256-update forks' effect was a continuation property.
- **Control.** P1-F itself, verified identical to this run up to the declared change (above).
- **About the environment / data.** Whether Craftax's night renderer static enters the world's state through the
  periphery's pixel teacher in full training, and whether removing it matters for imagined consequences.

## Readings (declared now; both worlds, the same contexts and draws)
P1-F's battery (its PLAN's order): the actor interface under random and guided gaze; S1; M0.5's evaluator; the decoder
render. Codex's: the re-render noise panel by light (manipulation check, with retained digits / entities); counter
consequences at the rule's crop AND another crop containing the counter, DO and NOOP; entity counts; sampler
production Euler-3 / churn 1 primary, Euler-8 / churn 0 and Euler-32 / churn 1 secondary for both worlds (never
a best-of). Checkpoint by P1-F's rule (10 unless the monitor rose at three successive checkpoints). Interpretations
as OVERNIGHT_PROPOSAL.md: noise down + evidence kept + physical deficits down -> the replacement helps this recipe;
noise down, generation unchanged -> filtering insufficient here; only G1 / C2ST better -> no physical claim; noise
not down -> the manipulation failed. One seed: no claim across world seeds; no claim that faithful caused the
historical regression; the learned fixation is not part of this run.
