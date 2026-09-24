# Frozen-representation ladder — the safety signal is in the patch tokens, not in CLS

Run 2026-09-24, `frozen_ladder.py` (`5844222f`, committed before the run). **Exploratory**: the
observability test's roots, asserted identical by content hash (fit 7,085 roots; judgement 2,856 on
seeds 50,000–50,274, now inspected). Raw H2 bridge checkpoint, asserted equal to CONFIRM's. Every arm:
the same expected-risk ranking objective, optimizer, batches, 3,000 updates, inner-FIT selection and
512-wide hidden layer; three probe seeds. Evidence: `evidence/frozen_ladder.json`, per-seed
judgement scores in `evidence/frozen_ladder_rows.pt` (hash-bound).

## One-step death (500 opportunity roots; prior always-DOWN 0.628; oracle 1.000)

| arm | expected safe | vs prior | fit | params | day | night |
|---|---|---|---|---|---|---|
| actions only (last 4) | 0.653 | +0.025 | 0.82 | 46k | 0.639 | 0.663 |
| **raw pixels, 1 frame** | **0.884** | **+0.256*** | 1.00 | 609k | 0.913 | 0.862 |
| raw pixels, 4 frames | 0.702 | +0.074* | 0.95 | 628k | 0.721 | 0.688 |
| raw pixels, 4 frames + 4 actions ¹ | 0.744 | +0.116* | 0.99 | 665k | 0.773 | 0.721 |
| **encoder patch tokens, flat** | **0.778** | **+0.150*** | 1.00 | 8.0M | 0.859 | 0.715 |
| encoder patch tokens, attention readout | 0.731 | +0.103* | 0.89 | 115k | 0.787 | 0.688 |
| encoder CLS + 4x4 pooled grid | 0.722 | +0.093* | 0.93 | 1.7M | 0.774 | 0.681 |
| **encoder CLS, 1 frame** | **0.627** | −0.001 | 0.74 | 108k | 0.629 | 0.625 |
| encoder CLS, 4 frames | 0.632 | +0.004 | 0.73 | 402k | 0.585 | 0.668 |
| projected z, 1 frame | 0.625 | −0.003 | 0.76 | 108k | 0.616 | 0.632 |
| projected z, 4 frames | 0.625 | −0.003 | 0.80 | 402k | 0.622 | 0.627 |
| world root features | 0.627 | −0.001 | 0.80 | 140k | 0.625 | 0.629 |
| u (per branch, shared head) | 0.654 | +0.026 | 0.74 | 132k | 0.663 | 0.647 |
| generated z (per branch) | 0.611 | −0.017 | 0.62 | 99k | 0.634 | 0.593 |
| generated features (per branch) | 0.630 | +0.002 | 0.72 | 132k | 0.643 | 0.621 |

`*` = paired, episode-seed-clustered 95% interval excludes zero.

| paired contrast along the chain | difference |
|---|---|
| pixels 4 frames + actions − pixels 4 frames | +0.041* |
| pixels 4 frames − pixels 1 frame | **−0.182*** |
| patch tokens (flat) − pixels 1 frame | −0.106* |
| CLS + pooled − patch tokens (flat) | −0.057* |
| **CLS − CLS + pooled** | **−0.095*** |
| projected z − CLS (1 frame / 4 frames) | −0.001 / −0.007 |
| world root features − z (4 frames) | +0.002 |
| u − world root features | +0.027 |
| generated z − u | −0.043* |
| generated features − generated z | +0.020 |

### By hazard (one-step, expected safe; non-exclusive strata from the pre-darkness visible state)

| | lava adjacent (110) | zombie adjacent (265) | arrow / skeleton (151) | none of these (85) |
|---|---|---|---|---|
| prior | 0.764 | 0.508 | 0.689 | 0.741 |
| raw pixels, 1 frame | 0.989 | **0.816** | 0.895 | 0.965 |
| patch tokens, flat | 0.870 | **0.691** | 0.815 | 0.890 |
| CLS + pooled | 0.892 | 0.600 | 0.773 | 0.843 |
| CLS | 0.846 | **0.503** | 0.682 | 0.682 |
| projected z | 0.822 | **0.498** | 0.660 | 0.722 |
| world root features | 0.822 | **0.506** | 0.664 | 0.686 |
| u | **0.968** | 0.456 | 0.753 | 0.765 |
| generated z | **0.986** | **0.362** | 0.746 | 0.816 |

Chosen actions over 1,500 choices (500 roots × 3 probe seeds): raw pixels spread across
RIGHT / LEFT / UP (≈300 each); CLS and z choose DOWN on 553 and 575; generated z chooses **SLEEP on
686**.

## Two-step death (986 roots; prior 0.685; oracle 0.829)

The same shape, compressed: raw pixels 1 frame 0.753 (+0.068*), patch tokens 0.720 (+0.035*),
CLS + pooled 0.710 (+0.025*), and CLS 0.683, z 0.674, root features 0.675, u 0.692, generated z
0.685 — all within noise of the prior. CLS − CLS + pooled −0.027*.

## What the numbers say

1. **The current frame carries the one-step decision.** One raw frame, no actions, reaches 0.884 —
   level with the structured visible state (0.872 in OBSERVABILITY.md) and holding 0.862 at night.
   The observability test's pixel figure (0.768, four stacked frames + actions) understated it:
   stacking frames as channels *costs* this CNN 0.182 at this data size, and actions recover only
   +0.041 of that. Actions alone are worth nothing (+0.025, unresolved).
2. **The frozen encoder keeps much of it — in its patch tokens.** Flattened patch tokens reach 0.778;
   they lose 0.106 against raw pixels, most of it at night (0.715 vs 0.862 for pixels) and on zombie
   roots (0.691 vs 0.816).
3. **No head here generalizes a safe-action ranking from root CLS.** CLS alone scores 0.627 against a
   0.628 prior (probe seeds 0.565–0.673), on every hazard, at 1 or 4 frames; the largest step down
   the chain is CLS + pooled → CLS (−0.095). The gap is not a capacity artefact: the patch-attention
   head, about the size of the CLS head (115k vs 108k parameters), beats CLS by **+0.105 [+0.067,
   +0.149]**. But a stronger CLS readout has not been tried, so this shows a usable-information gap,
   not an irreversible loss. And because CLS already sits at the decision floor, a further loss in the
   projector would be invisible here (z − CLS −0.001): CLS is the *first observed weak interface*, not
   an acquittal of what follows — u → generated z still falls 0.043, resolved.
4. **Everything the world model sees comes from CLS** — it transitions `z = projector(CLS)` and never
   receives the patch tokens (`lewm.py:LeWMEncoder.export`: "the world transitions z alone. Patch
   tokens are never regularized by SIGReg and never predicted"). World root features sit at the prior
   (0.627), as its input does.
5. **The dynamics learned terrain, not mobs.** Next to lava, u reaches 0.968 and generated z 0.986 —
   well above what z itself gives (0.822). Next to a zombie, the largest hazard class, every rung
   from CLS on is at or below the prior (0.50 → u 0.456), and generated z falls well below it (0.362,
   −0.147 [−0.217, −0.071] against the prior). Generated z also chooses SLEEP on 686 of 1,500
   choices — but on 222 of 330 near lava too, where staying put is safe. The SLEEP preference is a
   general symptom of the predicted latent, not a zombie-specific rule; the finding is poor
   zombie-conditioned action ranking.

## How far this goes

- **Not a nested chain, so not a verdict.** CLS and the patch tokens are parallel outputs of one
  ViT, not successive stages. What is established: from the model's own frozen representations,
  within-root safety is readable from the patch tokens and is not readable from CLS, z, or anything
  the world computes from them — except for lava.
- **CLS may hold it in a form this probe cannot read.** A comparably small head (CONFIRM's adapter,
  a different design) read 0.887 from the *successor's* CLS in hindsight, so CLS is readable in
  principle; but hindsight CLS sees the outcome. That does not settle what the root CLS holds.
- **Why CLS lacks it is not measured.** A candidate, not a finding: JEPA's objective rewards what is
  predictable, and mob movement is drawn from the step RNG, so a CLS trained to be predicted may
  discount mobs. Lava is static and survives.
- **Reproduction is not bit-exact.** The observability arm rerun through its own code gave 0.744
  against 0.768 (probe seeds 0.745 / 0.713 / 0.773 against 0.777 / 0.775 / 0.752): cuDNN convolution
  training is nondeterministic on this GPU. Every decision-level reading of that arm is unchanged.
- **Every arm reaches ~1.0 on its fit roots** where it has the capacity (pixels, tokens flat): the
  judgement numbers carry the result.
- **Exploratory roots.** A repair motivated by this is judged on a new seed block before H16.

## What this points at

The review's branches were "patches fail → check whether they decode the visible scene; patches pass
and z fails → the projector". Patches pass, but z fails *because CLS already fails*: the projector is
not where it goes. The place to look is the interface: only `z = projector(CLS)` enters the world. This needs one
correction to how it was first framed. **TC-LeWM's world predictor also transitions projected CLS**; it
is TC-LeWM's *downstream policy on real observations* that reads CLS plus pooled patch tokens, and our
specification already records using the projected latent for the policy as "a major deviation"
(DECISIONS.md TC-07). TC-03 lists "loss of … spatial relations" as a stop signal. So feeding real patch
tokens to the policy would follow the source more closely but would not fix imagination: future real
patches do not exist inside a rollout. A repair has to carry a patch-aware state **through the
prediction itself**, and through the outcome readout of the predicted state — a new world-model design,
not something TC-LeWM validated. The decodability check on patch tokens is not triggered, since they did
not fail.
