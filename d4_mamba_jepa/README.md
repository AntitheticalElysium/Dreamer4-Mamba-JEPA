# D4-Mamba-JEPA

This directory is a clean, separately runnable research track. It does not
modify `m3_hjwm_compact` and it does not claim that the compact architecture is
Dreamer 4.

The reference system is the pinned, MIT-licensed MMBench2 PyTorch world model.
The baseline arm imports its model implementation without editing or copying
it. Local code is limited to:

1. a 6 GB-scale configuration;
2. Crafter's discrete action and continuation interfaces;
3. a switchable temporal Mamba-2 replacement;
4. a switchable CDP-shaped predictive representation objective;
5. tests, provenance, checkpointing, and evaluation adapters.

The experiment is a two-by-two factorial:

| Arm | Temporal operator | Representation objective |
|---|---|---|
| `T-BASE` | upstream temporal attention | upstream reconstruction/frozen-latent objective |
| `M-BASE` | Mamba-2 | upstream reconstruction/frozen-latent objective |
| `T-CDP` | upstream temporal attention | CDP-shaped predictive auxiliary plus frozen-decoder anchor |
| `M-CDP` | Mamba-2 | CDP-shaped predictive auxiliary plus frozen-decoder anchor |

The Transformer baseline must run first. Mamba and CDP are never silently
enabled and may not be used to repair a broken baseline.

## Evidence files

- `SOURCE_MANIFEST.md`: exact upstream revisions, hashes, licenses, and reuse
  boundaries.
- `DEVIATION_LEDGER.md`: every local architectural deviation, why it exists,
  how it is isolated, and what failure it could cause.

## Claim boundary

MMBench2 states that its world model largely follows Dreamer 4, but it is not
the original authors' canonical Dreamer 4 release. Accordingly this directory
uses the terms **D4-style** and **D4-lite**, not "faithful Dreamer 4
reproduction".

The CDP arm is an auxiliary attached to a D4-style latent denoiser. The future
target is stop-gradient, the encoder uses a lower learning rate, the ordinary
flow loss receives detached tokenizer latents, and a frozen decoder supplies a
reconstruction anchor. It is not a faithful Dreamer-CDP reproduction and it is
not reconstruction-free.

## Current state

The source and implementation gates, four-arm CUDA smokes, controlled overfit,
a real-Crafter `T-BASE` preflight, and the matched real-Crafter Stage M1 have
run. The baseline demonstrably uses actions and can execute the random-shooting
planner. M-BASE passes every frozen relative feasibility gate against T-BASE,
and both arms reproduce exactly under the pinned deterministic contract.

Exact results and claim boundaries are recorded in
`reviews/2026-07-20-d4-mamba-jepa-reboot-implementation-and-preflight.md` and
`reviews/2026-07-20-d4-stage-m1-matched-backend-outcome.md`.

Both backends still fail terminal calibration and have weak K=4 generated
reward ranking. Mamba is viable, not superior: it is slower and uses more
memory at sequence length 16. CDP has not yet been trained on real Crafter.
The next isolated stage is the matched `BASE`-versus-`CDP` factorial.
