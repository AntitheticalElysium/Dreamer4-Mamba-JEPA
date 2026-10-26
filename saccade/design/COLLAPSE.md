# Collapse prevention: decision record (2026-10-27)

**Decision.** M0 has no distribution regulariser (SIGReg, VICReg, Barlow) on its percept tokens. Collapse is
countered by the training graph and by grounding:
1. Routing: every frame input of the denoiser (noisy frames and context-prefix frames) and its targets are
   stop-gradient; the memory tokens stay differentiable. The denoising loss reaches the encoder only through the belief
   (past percepts -> Mamba -> memory tokens), never through the frame being predicted; the encoder learns 0.3x slower
   than the denoiser.
2. A fixed-target owner per token type: periphery tokens by trans-saccadic prediction (TSP) and the change map; glimpse
   tokens by the fovea's own reconstruction of its crop; the lighting latent through both pixel likelihoods; all of them
   through the reward and termination heads.
3. Scale: each token is layer-normalised over its channels without affine (I-JEPA's target convention; V-JEPA 2-AC
   normalises every world-model state this way, encoded and predicted), then soft-clamped by JEDI's 3 tanh(z/3), so
   sigma_data = 1 holds while the encoder trains. Per token, so stateless and causal: chunk-wide BatchNorm, the first
   choice, made a past percept depend on future frames and broke the bound (Codex's audit, 10-27).
This is the chosen training mechanism, not a guarantee: entity retention is measured in S1 (token probe against the
untrained-encoder copy; the S3 perception blocker).

**How it was decided.** I proposed dropping SIGReg on 10-23 ("working systems don't use it"); the user rejected that
argument. Collapse prevention stayed open; SIGReg re-entered on 10-25 inside a larger approved message. On 10-27 the
user asked for its own investigation and delegated the call; a second research pass followed the same day.

**Why no SIGReg on the tokens**
- Precedent sits elsewhere. LeJEPA and LeWM apply SIGReg to one global embedding per frame, through a projector, and
  use the layers before the projector downstream; R2-Dreamer aligns a projection of its global world-model state with
  a stop-gradient encoder embedding ("this detach is important"). The space the self-supervised loss shapes is the
  worse one downstream (Guillotine regularisation, Bordes et al., up to 30 points). Our tokens are used directly.
- Our history: where patch-level SIGReg acted in October (the deployed projector output), macro recall was 0.48
  against 0.83 in the backbone beneath it and day nuisance drift about 2x the Raw encoder's; patch SIGReg opposed
  camera-aligned transport at 4/4 paired checkpoints. Causes not isolated: a warning, not proof.
- Fit to the data: by day a Craftax cell has 44 meanings (2.7 bits on average); filling a 16-dimensional isotropic
  Gaussian per cell requires continuous variation, which here is light level, sleep tint or night static: what the
  lighting latent is meant to discount. Temporally Centered SIGReg (arXiv 2607.26924) measured the related bias on
  LeWM: SIGReg spends variance on temporally persistent content and suppresses the residual that carries dynamics.
- It does not address our actual failure, selective erasure: it "prevents global collapse without choosing which
  variance survives" (SLIM), whose plain LeWM stayed collapsed at 3x the SIGReg weight.

**Why it is safe to drop here, and not in LeWM**
LeWM and PLDM have no stop-gradient and no fixed targets: a variance regulariser is their only anti-collapse, and
without it LeWM collapses (median latent distance 17-19 -> 0.006; ACPC, arXiv 2608.12939). M0 has both. The danger
specific to training an encoder jointly with a latent diffusion model is the diffusion loss reaching the encoder
through the latent: REPA-E measured spatial variance 17.06 -> 0.02 and FID 444 without the stop-gradient. JEDI, the
closest precedent, uses no regulariser (stop-gradient target and own input, encoder lr 0.3x, tanh clamp); its appendix
relates SIGReg to the aggregate prior term of its information bottleneck that it leaves out.

**What was weighed on the other side**
- The user's July aim: a reconstruction-free JEPA with SIGReg as the anti-collapse. Reconstruction-free stands; the
  fovea's crop reconstruction is the limited pixel grounding the user intended, never full-frame.
- Local distribution regularisers do help where per-patch content is rich: VICRegL improves detection and segmentation
  on natural images. PLDM found its variance/covariance terms essential (its only anti-collapse).
- LeWM's lambda is forgiving (0.01-0.2 keeps success above 80%).

**Revisit when**: per-patch content is rich (natural or 3D images); S1 shows dimensional collapse or erasure that the
grounding does not prevent; or a frame-level embedding enters the design (where LeWM's precedent applies).
**Planned comparison**: per-token SIGReg (LeWM's module, lambda x N held at 0.09 x 128) and the temporally centred
variant, by matched retraining, judged on S1 retention and S3 decisions.

Sources: LeJEPA (arXiv 2511.08544, vendored MINIMAL.md), LeWM (arXiv 2603.19312), R2-Dreamer (vendored code),
Guillotine regularisation (arXiv 2206.13378), SLIM (arXiv 2610.03137), Temporally Centered SIGReg (arXiv 2607.26924),
ACPC (arXiv 2608.12939), REPA-E (ICCV 2025), JEDI (arXiv 2605.13013, vendored), VICRegL (arXiv 2210.01571), HamJEPA
(arXiv 2605.20107), TD-MPC2 (arXiv 2310.16828: normalised latents, SimNorm), I-JEPA and V-JEPA 2 (vendored code:
`F.layer_norm` on target and world-model states), our R57 / backbone review (NOTEBOOK).
