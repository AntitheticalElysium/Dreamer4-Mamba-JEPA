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

There is now a positive, reproducible executed-control baseline before either
research modification:

- official, source-pinned Gymnasium `CartPole-v1`;
- the Transformer tokenizer and shortcut world model only (`T-BASE`);
- a 20,000-update world checkpoint;
- a categorical port of the upstream MMBench2 BC policy head, trained with the
  tokenizer and world frozen;
- 30 fresh evaluation seeds, repeated exactly.

The frozen pixel policy averages **288.70** return versus **17.60** for matched
uniform random, wins all 30 pairs, and has a paired bootstrap delta of
**+271.10 [234.07, 309.77]**. Its held-out demonstration action accuracy is
87.04%. The repeated evaluation has identical substantive rows, summaries,
confidence interval, and gate decision. The world checkpoint, policy
checkpoint, hashes, raw rows, and commands are recorded in
`reviews/2026-07-20-d4-lite-cartpole-working-baseline.md`.

The imagined exhaustive planner is separately positive but weaker: 44.87
versus 19.00 random on 30 fresh seeds, paired delta +25.87
[15.80, 36.73]. It misses its deliberately stricter absolute mean-return gate
of 50. Therefore the project now has a working learned control baseline, but
does not yet claim that imagination planning itself is solved.

This remains a D4-style reduced baseline, not a full Dreamer-4 reproduction.
The working policy is behavior-cloned from demonstrations; it is not an
actor/value pair trained in imagination. Mamba and CDP/JEPA are still disabled,
so subsequent swaps have a genuine positive control.
