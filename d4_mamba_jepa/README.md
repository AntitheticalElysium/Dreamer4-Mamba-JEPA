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
[15.80, 36.73].

The missing Dreamer-4 actor/value phase is now implemented as a separate
source-pinned runner. It copies the actor exactly from BC, freezes an exact BC
prior and the complete world, generates one 32-decision rollout from each
replay context with four shortcut steps, learns a zero-output symexp-twohot
value head from TD-lambda targets, and updates the actor with balanced PMPO
plus reverse `KL(actor || prior)`. It never calls the shooting planner.

The initial batch-16 checkpoint passed development but failed its first sealed
set and remains recorded as a negative result. Correcting only replay-context
diversity to the inspected Dreamer-4 reproduction's batch of 64 produced the
selected 250-update actor. On 100 fixed fresh seeds, direct greedy execution
scores **281.33** versus **249.32** for the paired frozen BC policy and
**18.32** random. The paired actor-minus-BC delta is
**+32.01 [6.66, 57.94]**. Actor, value, training history, and frozen
invariants reproduce exactly.

This completes the reduced Transformer Dreamer-4-style control baseline. It
does not reproduce the paper's scale, Minecraft task interface, 192-frame
context, or `tau_ctx=0.1` context corruption. The frozen world still predicts
continuation near one in imagined rollouts, so this result establishes a
working end-to-end actor/value baseline, not that CartPole imagination is
perfectly calibrated. Mamba and CDP/JEPA remain disabled and can now be
introduced one at a time against a positive actor-critic control.
