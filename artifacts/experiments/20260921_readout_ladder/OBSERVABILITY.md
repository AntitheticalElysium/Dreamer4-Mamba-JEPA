# Observability test — the current observation carries much of what decides death

> **CORRECTED 2026-09-24, after review.** The first version was titled "the deciding state is on
> screen" and overclaimed in four places, all corrected below: (1) +0.022 is what hidden fields add
> *to this fitted probe*, not a bound on their value — even the full-state probe leaves 0.106 to the
> oracle; (2) the pixel arms also receive the last k actions, with no actions-only control, so their
> gain is not all pixels; (3) "history does not help" was broader than the evidence, which is one
> overfit CNN and one hand-picked history trace; (4) the "visible" arm reads exact simulator tiles and
> mobs *through night darkness* the renderer draws over them — see the day/night table. And the
> proposed next step, "the first rung that falls is where the loss is", is withdrawn: these rungs are
> not a nested information chain, so a falling score is a place to look, not a verdict.

Run 2026-09-24. Data: `observe.py` (`13891572`); probes: `observability.py` (`3de6a3fd`), both
committed before any fresh-seed result was read. Evidence: `evidence/observability.json`, per-seed
judgement scores in `evidence/observability_rows.pt` (hash-bound).

- **Development:** the 700 FIT seeds, 7,085 roots, replayed with every stored root reproduced
  exactly, each root carrying the full simulator state and P(death) over 32 independent key pairs.
- **Judgement:** fresh seeds from 50,000, a range nothing in this codebase had used, collected with
  the fork collector's exact loop and retention rule (verified byte-identical on existing seeds). The
  predeclared stop rule fired at seed 275 with exactly 500 one-step opportunity roots: 272 seeds with
  roots, 2,856 roots, 986 two-step opportunities. **Sealed, untouched by any earlier analysis.**

Six matched arms — root → [trunk] → 512 → 17, same objective (expected-risk pair ranking), optimizer,
batches and budget, three probe seeds, selection on an inner 15% of FIT seeds — scored as expected
safe choice, 1 − P(death | chosen), against the FIT-root action prior (always DOWN).

## This time the positive control works

The full-state arm clears the prior by a wide margin on both outcomes (below). The 2026-08-19 Direct
test (`artifacts/eda/probe_observability.py`) failed exactly here: its full-state probe scored 0.734
within-root AUC, identical to its visible probe, so its visible ≈ full result could not tell
"hidden fields do not matter" from "the probe cannot decode even the full state". Its commit cited
an AUC of 1.0000, but that was S35's simulator oracle, not the probe.

## One-step death

Prior 0.628; full-state oracle 1.000; 500 opportunity roots. `*` = resolved.

| arm | expected safe | vs prior | fit-root safe |
|---|---|---|---|
| **visible** — drawn state now | **0.872** | +0.244 [+0.185, +0.314]* | 1.000 |
| visible + 32-step visible history | 0.857 | +0.229 [+0.169, +0.296]* | 1.000 |
| **full** — visible + hidden (positive control) | **0.894** | +0.266 [+0.204, +0.335]* | 1.000 |
| **raw pixels, 4 frames** (Nature-DQN CNN) | **0.768** | +0.140 [+0.075, +0.205]* | 0.994 |
| raw pixels, 32 frames | 0.648 | +0.020 [−0.060, +0.102] | 1.000 |
| raw pixels, 32 frames + hidden vector | 0.799 | +0.171 [+0.111, +0.240]* | 0.992 |

| paired contrast | difference |
|---|---|
| full − visible (what the hidden fields add) | **+0.022 [+0.007, +0.039]*** |
| visible history − visible | −0.015 [−0.040, +0.008] |
| pixels32 − visible history | −0.209 [−0.254, −0.159]* |
| pixels32 + hidden − pixels32 | +0.150 [+0.100, +0.193]* |
| pixels32 − pixels4 | −0.119 [−0.173, −0.062]* |

## Two-step death (NOOP second step, 32-key expected risk)

Prior 0.685; oracle 0.829 (two-step outcomes are partly random); 986 opportunity roots.

| arm | expected safe | vs prior |
|---|---|---|
| visible | 0.755 | +0.070 [+0.048, +0.100]* |
| visible + history | 0.762 | +0.077 [+0.056, +0.104]* |
| full | 0.764 | +0.079 [+0.056, +0.109]* |
| raw pixels, 4 frames | 0.720 | +0.035 [+0.013, +0.064]* |
| raw pixels, 32 frames | 0.713 | +0.028 [+0.006, +0.056]* |
| raw pixels, 32 frames + hidden | 0.730 | +0.045 [+0.019, +0.074]* |

Full − visible is +0.009 [+0.001, +0.016]; visible closes about half of the prior-to-oracle gap.

## Declared reading: `drawn_state_suffices`, both outcomes

- **The current observation is highly informative.** The structured visible state reaches 0.872 with
  no action history at all. Adding every hidden field adds **+0.022 to this probe** — but the
  full-state probe itself reaches only 0.894 against an oracle of 1.000, so the hidden fields' value
  to a better predictor is not bounded by this.
- **A from-scratch image model gets a useful share of it.** A DQN CNN on the last four frames reaches
  0.768, +0.140 over the prior, on sealed fresh seeds — but it also receives the last four actions,
  and there is no actions-only control, so not all of the gain can be credited to pixels. Everything the ranking probes in `RANKPROBE.md`
  tried on the model's own representations — CLS + pooled patches, root features, `u`, generated
  `z` — sat at or below the prior.
- **These particular history inputs did not help**: the hand-picked structured trace (−0.015,
  unresolved), and 32 raw frames, which do *worse* than 4 (−0.119, resolved). The 32-frame CNN has 8× the input channels and the
  same 7,085 roots, reaches 1.000 on fit roots and 0.722 on inner roots — overfitting, not evidence
  that history hurts. Handing it the hidden vector rescues it (+0.150), because a short vector is a
  far easier route than 96 channels of pixels.

## Night: the "visible" arm sees through darkness the pixels cannot

281 of the 500 one-step opportunity roots are at night (light level < 0.5), where the renderer
darkens the view and overlays random static. The structured arm is built from exact simulator tiles
and mobs, so night costs it nothing; the pixel model loses about 5 points:

| one-step expected safe | day (219) | night (281) |
|---|---|---|
| prior (DOWN) | 0.637 | 0.621 |
| structured visible | 0.874 | 0.870 |
| full state | 0.895 | 0.893 |
| raw pixels, 4 frames | 0.795 | 0.746 |
| raw pixels, 32 frames + hidden | 0.791 | 0.805 |

So "visible" here means *what the renderer draws before night obscures it* — an overestimate of
what is actually on screen for most of these roots. Any further ladder should stratify by light.
No opportunity root had the player asleep.

## What this does not establish, and the confound that blocks the next claim

- **Every arm overfits** (fit-root safe ≈ 1.0). The judgement numbers are what count, and they are
  on sealed roots, but the gaps say these probes are data-limited, pixels most of all.
- **The structured visible arm has a strong built-in bias**: an egocentric, tile-aligned grid. It is
  an upper reference for what is drawn, not what an image model should be expected to reach from
  7k roots. The honest pixel number is 0.768.
- **The comparison with the model's representations is not yet matched.** RANKPROBE scored them
  with a different head (per-row mlp128 on `[17, D]` rows) on different, previously examined roots,
  against realized outcomes. "Raw pixels reach 0.768 and the model's representations do not beat
  the prior" is therefore suggestive, not a localization. The reviewer's branch — *visual history
  clears the prior → localize the loss through encoder, projector and transition* — needs those
  representations scored **in this harness, on these roots**: encoder patch tokens unpooled,
  CLS + pooled grid, projected `z`, world root features, `u`, generated `z`. These roots have now
  been inspected, so that ladder is exploratory; any repair it motivates is judged on a new seed
  block.
