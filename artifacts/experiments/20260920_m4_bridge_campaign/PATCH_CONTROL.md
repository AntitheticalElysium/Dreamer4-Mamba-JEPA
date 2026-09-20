# Patch control — design (audit action item 3)

**Not built. Designed, costed, and predeclared so it can be run or declined deliberately.**

## The question it answers

`RISK_REVIEW.md` risk 7 is the nearest analogue of the old export-bottleneck mistake:

> Source policy's CLS plus pooled spatial tokens → only projected 192-D state plus Mamba history
> and a new readout. **High information risk.** The paper predicts a projected latent but gives its
> downstream policy richer visual features. Our imagined states cannot recover missing patch
> information by reading the real encoder.

TC-LeWM freezes the encoder and hands its BC policy **unprojected CLS + a 4×4 pooled patch grid**
per camera. Our path is `CLS → projector → z`, then `z + Mamba history → 256-D readout → heads`.
Patch tokens exist in `LeWMEncoder.export()` but reach only diagnostics and M03, never the policy.

So if the campaign's actor underperforms, the cause could be the world, the critic, the bridge —
or simply that the 192-D projected `z` never carried enough to act on. **Nothing in the campaign as
designed separates that last possibility from the others.** This control does, before the fact.

## Design

Three behaviour-cloning policies on **observed** frames only. Identical corpus, identical splits,
identical head architecture, identical budget and optimizer. They differ in one thing: what visual
features the policy reads.

| arm | features | dimension | what it represents |
|---|---|---:|---|
| **A** | projected `z` + Mamba history | 192 + 256 | the campaign's own path |
| **B** | unprojected CLS + 4×4 pooled patch grid | 192 + 16×192 | **paper-faithful ceiling** (TC-LeWM Appendix A.1) |
| **C** | unprojected CLS alone | 192 | isolates the projector from the loss of spatial tokens |

All three read `LeWMEncoder.export()` from the **same frozen joint encoder**, so no arm benefits
from a different representation — only from how much of it the policy is allowed to see.

### Predeclared contrasts

- **B − A** — what the projector-plus-`z` bottleneck costs against the paper's own feature set.
- **C − A** — how much of that cost is the projector specifically.
- **B − C** — how much the spatial patch grid adds beyond CLS.

### Measurements

1. **Action agreement** on held-out DEV expert transitions (top-1, and the relevant/uniform split
   separately), which is cheap and needs no execution.
2. **Real Craftax**, the campaign's own protocol: seeds 30000–30511, native 10,000-step cap,
   paired bootstrap over episodes, reported as achievements with a 95% interval.

### What each outcome licenses

| result | reading |
|---|---|
| B ≈ A | the `z` path is **not** the binding constraint; a downstream failure is genuinely the world, bridge or critic, and the campaign's attribution stands |
| B ≫ A | the bottleneck bounds everything built on it. No amount of world-model work recovers information the policy never receives, and the campaign's actor-vs-BC result must be read inside that ceiling |
| B ≫ C ≈ A | the **spatial tokens** carry the missing decision information, and a patch-aware world is the indicated redesign |
| C ≫ A | the **projector** is discarding it, which is a much cheaper fix than a spatial world |

## Why this is a ceiling and not a rival actor

The paper's BC always has a real observation. **An imagined successor has no patch grid** — the
world predicts `z`, not tokens. So B and C cannot be turned into imagination actors as they stand;
they bound what any imagination actor built on this encoder could reach.

A patch-aware imagination actor would need a **propagated spatial state**: the world would have to
predict the patch grid, not just `z`. That is a different architecture, out of scope here, and this
control is what tells us whether designing it is warranted.

## Cost

No world model, no imagination, no recursive training. The encoder is frozen and forward-only, so
features can be cached once and reused by all three arms.

| item | estimate |
|---|---|
| feature cache (archive TRAIN+DEV, CLS + 4×4 grid + z) | ~25 min, ~12 GB |
| three BC fits | ~20 min each |
| real Craftax, 3 policies × 512 seeds | ~1.2 h |
| **total per encoder arm** | **~2.5 h** |

Run it against the Raw arm's joint encoder first; TC only if Raw's result is ambiguous.

## Recommendation

**Run it, and run it early.** It is independent of G1, the bridge and the actor, it costs about a
quarter of the main campaign, and it makes the main result interpretable whichever way that lands.
Run after the joint phase produces an encoder, and before or alongside the bridge — its answer
changes how the actor-vs-BC number should be read, so learning it afterwards is worth strictly less.
