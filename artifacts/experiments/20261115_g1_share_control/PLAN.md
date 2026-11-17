# Is G1 starved by the reward heads' dominance of Mamba? (2026-11-15)

User 11-15: "For G1 - this is more of what I was after, partly because if reward dominates Mamba too much that might be
bad? ... run what you need." Before designing a minimum share for G1 at Mamba, this asks whether G1 gains anything when
the reward heads stop dominating Mamba.

- **Measured fact.** At F6's checkpoint 10 the reward forecast's weighted gradient into Mamba's module is about 150
  times G1's, its cosine with the total update 0.997 (`../20261114_f6_dissection/readings/batch_10.json`).
- **Claim.** That dominance costs G1: given Mamba to itself, G1 reaches a lower loss than under F6's mix.
- **Competing explanation.** Yesterday's G1-alone continuation (`../20261114_learnability/`, arm A5: training-stream
  G1 .00420 -> .00386 over 2,000 updates) only shows more training from a fresh optimiser; the same continuation with
  F6's full loss would gain as much.
- **Separating control (`control.py`).** A5's loop exactly (F6 checkpoint 10, its stream state, carried Mamba state
  and cache, the same 2,000 chunks and bootstrap rows, 16 streams in 2 microbatches, AdamW 1e-4 with A5's warm-up,
  clip 1.0, Mamba's module and the denoiser trainable, everything else frozen), with one change: the loss is F6's own
  weighted sum (the harmoniser's final divisors, frozen) instead of G1 alone. In both arms the denoiser's parameters
  receive only G1's gradient, so a difference in G1 can come only from what Mamba carries (or from Adam's coupling
  through Mamba).

## Readings (fixed before results)

1. **Primary:** the held-out monitor G1 (`train.monitor`, F6's monitor episodes and windows) of F6 checkpoint 10, A5 at
   1,000 and 2,000 updates (saved worlds) and the control at 1,000 and 2,000, with its noise levels. Starvation is
   supported if A5's held-out G1 at 2,000 updates is at least 3% below the control's; not supported if within 1%.
2. Training-stream G1 every 250 updates, both arms on identical chunks.
3. **What A5 paid:** recovery and cooldown in native generation (`../20261111_p1f5/trajectory.py`) for the control
   (A5's are in `../20261114_learnability/g1_run/readings/`: recovery drawn .466 -> .192).

Bounded: a continuation of a reward-shaped world, not training from scratch; one seed. A supported claim motivates
designing a G1 share floor at Mamba (its own reviewed arm); an unsupported one says a floor would buy little for G1.

## Stopped (user, 11-15, before any result)

Stopped during its first 250 updates; nothing was read. The user: a G1 share at Mamba belongs in an actual overnight
run from scratch, not a continuation of a reward-shaped world; the learnability A5 arm (G1 with the whole budget) was
already the extreme of this question.
