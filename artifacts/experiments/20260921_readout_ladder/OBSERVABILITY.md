# Observability test — the deciding state is on screen

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

- **Hidden state is a small part of the problem.** The full-state oracle is perfect one-step; what is
  drawn on screen *now* reaches 0.872, and adding every hidden field — cooldowns, mob health, the
  player's accumulators — adds only **+0.022**. Partial observability exists, and is resolved, but it
  is not what separates the models from the target.
- **It is learnable from raw pixels.** A from-scratch DQN CNN on the last four frames reaches 0.768,
  +0.140 over the prior, on sealed fresh seeds. Everything the ranking probes in `RANKPROBE.md`
  tried on the model's own representations — CLS + pooled patches, root features, `u`, generated
  `z` — sat at or below the prior.
- **History does not help here**, in structured form (−0.015, unresolved) or as more frames: 32 raw
  frames do *worse* than 4 (−0.119, resolved). The 32-frame CNN has 8× the input channels and the
  same 7,085 roots, reaches 1.000 on fit roots and 0.722 on inner roots — overfitting, not evidence
  that history hurts. Handing it the hidden vector rescues it (+0.150), because a short vector is a
  far easier route than 96 channels of pixels.

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
  representations scored **in this harness, on these sealed roots**: encoder patch tokens unpooled,
  CLS + pooled grid, projected `z`, world root features, `u`, generated `z`. The first rung that
  falls from the pixel level to the prior is where the loss is.
