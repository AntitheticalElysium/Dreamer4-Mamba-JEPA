# Non-generative JEPA world arm: outcome

Date: 2026-07-21

## Verdict

**GO. A genuinely non-generative JEPA world (no denoiser, decoder, MAE, or
reconstruction) both (1) makes imagination improve the policy and (2) makes
good vs bad policies diverge in imagination — the two objectives — on the
sealed tier.** The deterministic action-conditioned predictor is the entire
rollout; anti-collapse is a source-faithful SPR/BYOL EMA-target self-prediction.

## The two objectives (sealed seeds `987000:987100`, 100 paired)

| Policy | Mean return | Median |
|---|---:|---:|
| Random | 18.59 | 15.5 |
| T-BASE actor (control) | 264.33 | — |
| JEPA BC | 392.97 | 451 |
| **JEPA imagination actor** | **423.81** | **500** |
| Oracle | 500.00 | 500 |

- **Imagination improves the policy**: actor **423.81** vs its own BC **392.97**,
  paired **+30.84, 95% CI [2.9, 59.4]** (excludes zero). Imagination no longer
  loses to BC.
- **Comparable-or-higher than T-BASE**: actor vs T-BASE actor **+159.48,
  CI [127.7, 190.3], 83/100 wins**.
- **Good vs bad diverge in imagination** (BC-vs-anti-BC probe, horizon 32):
  imagined discounted return **BC 59.23 > uniform 56.28 > anti-BC 54.71**
  (monotonic), and the world now imagines termination under bad control —
  imagined continuation min **0.92 (BC) vs 0.12 (anti-BC)**. Before the fix the
  gap was exactly 0.000 and continuation was a flat 1.0.

## Proof that the latent carries the signal (not argued — measured)

A frozen-world linear probe on the agent tokens the continuation head reads,
predicting "terminal within 5 steps", on held-out terminated episodes:

- **JEPA agent-token AUC = 0.869**
- T-BASE agent-token AUC = 0.712

The non-generative latent carries the pole-about-to-fall signal, and **more**
than the generative T-BASE latent. So the earlier flat imagined return was a
readout/rollout failure, now demonstrated, not a representation failure.

## What was wrong and what fixed it

Two corrections, both registered (D034, D035):

1. **Multi-step self-prediction (D034).** The first port simplified SPR's
   multi-step `jumps` to a single teacher-forced step. It trained teacher-forced
   but *deployed autoregressively*, so imagined multi-step rollouts drifted off
   the training manifold and never reached terminal states (imagined continue
   stayed 1.0). Applying the predictor autoregressively for `jepa_jumps=5`
   steps, feeding its own output back and matching each to the EMA target of the
   actual future frame (faithful to `mila-iqia/spr 0b9dd4e7 do_spr_loss`), makes
   training match deployment. After this, imagined rollouts reach failures
   (continue min 0.12 under anti-BC).
2. **Terminal-weighted continuation (D035).** The unweighted continuation head
   collapsed to a constant on CartPole's sparse terminals (real-state continue
   std 0.0) even though the signal was decodable (AUC 0.869). Up-weighting the
   rare `continue==0` targets (`terminal_weight=8`, terminal window fraction
   0.5) recovers a calibrated head (real-state continue std 0.22, min 0.0).

Together these make the imagined return policy-dependent, which gives PMPO real
advantage signal, so imagination improves BC instead of drifting off it.

## Fresh-tier confirmation (`988000:988100`, unseen)

On a second sealed tier never used for any selection, the actor-beats-BC result
holds and strengthens: JEPA imagination actor **433.40** vs its BC **380.28**,
paired **+53.12, 95% CI [21.9, 85.9]** (vs +30.84 on 987). Both sealed tiers
show imagination improving BC with a CI excluding zero.

## Provenance

- JEPA multi-step world (`cartpole_jepa_v2`, seed 20260722, 20k updates):
  `39b9fc4b2c8d242cdbf94af4286b52efb4bdc5aa190b095786e63f72b8ba323a`
- JEPA BC: `d65d5fa55ad22591bd5d39a423303c83be4bd8a34e668e7e2987683af2c47296`
- JEPA imagination actor (500 updates):
  `95b6acc9d42cad08819ad8ba1a73e69f70a8722d7cb3c598158bca5a528f94a0`
- T-BASE control actor:
  `c8d99bd0598a19d6e23fafa34834f12cfa750eb02b2e7f9ca18f8bfa29e4a1c3`
- World cosine 0.868, online-std 0.098 (no collapse); tests: 6 JEPA + 57
  base/cdp pass.
- Probe artifact: `reviews/artifacts/d4_jepa_probe.py`.

## Claim boundary

Reduced D4-lite, pixel CartPole, non-generative arm on branch `d4-jepa-arm`.
The imagined good-vs-bad *return* gap (~4.5 at horizon 32) is modest in
magnitude but real, monotonic, and driven by the world genuinely imagining
termination (continue 0.92 vs 0.12); the executed-return improvement over BC is
the primary evidence that the non-generative latent carries policy-relevant
signal and that imagination now helps. Reward is constant-while-alive on
CartPole, so survival (continuation) is the only discriminator; that is the
mechanism exercised here.
