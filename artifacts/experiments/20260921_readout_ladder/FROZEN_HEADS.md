# Frozen-T head comparison and the H2 position audit

Both predeclared in `a98d9d77`, after the review of SPATIAL.md.

> **Correction, 2026-09-26 (found by the reviewing agent, verified).** The configuration labelled
> "gate protocol" generated the 4th frame from 3 observed frames. The real evaluator
> (`d4mj/lewm_diagnostics.py:_fork_readouts`) observes 4 frames and generates the 5th. At that position
> (L5, depth 1, cross-checked with this script's reader), generated death P(dead) is 0.054 (TRAIN) and
> 0.058 (DEV), and death vs 10-frames-earlier AUC is 0.51 / 0.53. The agent's independent run
> (`20260925_h2_independent`) gets 0.055 / 0.505. The conclusion stands.

## H2 audit (`h2_alias.py`, `36e52337`): declared reading `position_alias_in_H2` — the alias is recursion depth

Canonical H2 world, its trained heads and its frozen latent cache. 400 TRAIN and 400 DEV terminal
episodes. Fixed transitions:

- **death:** the terminal frame;
- **pre-death:** the frame before it;
- **alive-10:** ten frames before death.

Each is scored as the last block of a window, under the configurations below:

| TRAIN episodes, generated P(dead) | death | pre-death | alive-10 | real death frame |
|---|---|---|---|---|
| H2's exact terminal configuration (L32, burn-in 96, depth 2) | **0.73** | 0.70 | 0.56 | 0.60 |
| same window, depth 1 | 0.063 | 0.050 | 0.063 | 0.60 |
| L32 / L16 / L8, no burn-in, depth 1 | 0.055 / 0.046 / 0.041 | | | |
| ~~gate protocol~~ L4, one advance (3 observed + 1 generated; **mislabelled**, see correction) | 0.027 | 0.022 | 0.029 | 0.51 |

Window length and burn-in barely matter (0.03–0.06). **The generation depth does.** At depth 2 the
reading is not about death: the frame before death reads 0.70 and ten frames before reads 0.56.
The head recognizes the second recursively generated step, the step `paired_terminal_loss` always
labels dead (with the step before it labelled alive).

At depth 1, a generated death cannot be told from a state ten frames earlier (AUC 0.51). The real
death frame is read moderately (AUC 0.84 against pre-death). DEV episodes behave the same.

This is a structural defect of the canonical bridge's terminal supervision. It is the same family as
the per-tile worlds' alias (SPATIAL.md step 3): there too the high reading sat at the second
generated step.

## Frozen-T head comparison (`frozen_heads.py`): declared readings `readout_fails`, `no_evidence_transition_adds`

Sealed seeds 54,000–54,395: 804 one-step opportunity roots, collected after the commit and read once.
Heads were fitted on the FIT-train seeds and selected on the FIT-dev seeds; the gate-reserved and
unallocated seeds were never touched. The T world was frozen. Three head seeds per arm.

| head | reads | overall | zombie | stay kills, a move survives | head seeds |
|---|---|---|---|---|---|
| logged (BCE, the collector's action) | generated T | 0.638 | 0.520 | 0.473 | .631 .640 .645 |
| uniform (BCE, one random action per root) | generated T | 0.660 | 0.464 | 0.391 | .642 .662 .675 |
| all_bce | generated T | 0.674 | 0.508 | 0.449 | .671 .666 .685 |
| **all_rank** (primary) | generated T | **0.676** | 0.564 | 0.519 | .678 .676 .674 |
| root_bce (control) | root tiles + action | 0.673 | 0.535 | 0.494 | .681 .673 .665 |
| root_rank (control) | root tiles + action | 0.684 | 0.533 | 0.497 | .695 .724 .633 |
| prior (always DOWN) | | 0.613 | 0.533 | 0.515 | |
| actions_only | last 4 actions | 0.635 | 0.547 | 0.520 | .633 .622 .649 |
| root_tokens (boundary readout) | raw root patch tokens | 0.746 | 0.650 | 0.629 | .811 .673 .753 (**unstable**) |

**Readout: `readout_fails`.** all_rank beats the prior by +0.063 [+0.010, +0.112]\*. It does not beat
actions_only: +0.041 [−0.007, +0.091]. It does not beat the prior on zombie roots: +0.031
[−0.028, +0.094].

**Transition: `no_evidence_transition_adds`.** all_rank − root_rank is −0.008 [−0.050, +0.029]
(zombie +0.031, not resolved). A same-capacity head on the root's tiles plus the action does as
well as T's generated state.

**Against the stronger root readout** (reported): all_rank − root_tokens is −0.070 [−0.108, −0.032]\*,
and −0.086\* on zombie roots. The exploratory "generated = root" of step 6 (0.724 vs 0.724, 50k roots)
**does not replicate on sealed seeds**.

**Labels** (no declared contrast resolved overall):

| contrast | overall | zombie |
|---|---|---|
| uniform − logged | +0.022 | |
| all_bce − uniform | +0.014 | +0.044\* (reported only) |
| all_rank − all_bce | +0.002 | |

Calibration does show the logged action's selection bias. Mean predicted P(death) on opportunity
branches is 0.378 against a true 0.497 for logged, against 0.509 (uniform) and 0.522 (all_bce). On
ordinary roots every BCE head predicts ≤ 0.027 against a true 0.

**Chosen actions:** the generated-state heads still choose SLEEP often: all_rank 524 of 2,412
choices (804 roots × 3 head seeds); uniform 854.

## What this establishes

- **The canonical H2 world-and-head system reads death only as "the second generated step".**
  Paired terminal supervision labels exactly that step dead, so the head learns the step, not the
  consequence. At the gate's one-step protocol it assigns ~0.03 to generated deaths, and it cannot
  separate them from states ten frames earlier.
- **T's generated per-tile state gives no sealed evidence of decision value beyond its input.** With
  fork supervision it supports a modest readout: above the prior, but not above actions_only or on
  zombie roots, and equal to root tiles + action. It falls below the root readout. Together with the
  unchanged health tile (SPATIAL.md step 3), the per-tile Transformer world as trained is **not a
  repair**, and no fork-supervised safety readout passes either.
- **Not established:**
  - that removing the alias fixes H2; that is the next test;
  - anything about Mamba per-tile worlds;
  - why the root_tokens reference is so seed-unstable (0.67–0.81).

## Next (not run)

**An alias-free H2 bridge.** Same joint parent, same recipe, same 2,000 updates. The only change is
the terminal supervision layout: terminal and matched non-terminal states are supervised at the
*same* generation depth and at the gate's evaluation depth, so neither depth nor position predicts
the label. It stays inside the factual-training contract (no fork data). Judge it on:

- the gate protocol;
- a world-fidelity check: do depth-1 generated successors now separate death from alive?

## H2 input bound (`h2_bound.py`, `1fdb30a5`): declared reading `cls_input_bound_blocks_zombie_criterion`

Run after the reviewing agent proposed an alias-free H2 bridge judged on zombie action choice against
DOWN. H2's bridge keeps the encoder frozen, so every CLS-based H2 world sees only z over 4 frames plus
the actions between them. Heads were all-action ranking, the most favourable supervision. Fit on
FIT-train, selected on FIT-dev, judged on 54k (**exploratory**; the block was read before).

| head reads | overall | **zombie (479)** | lava |
|---|---|---|---|
| DOWN (prior) | 0.613 | 0.533 | 0.675 |
| actions_only | 0.635 | 0.547 | 0.675 |
| z4 (H2's latent input) | 0.645 | 0.542 | 0.892 |
| z4 + actions (H2's full input) | 0.646 | **0.554** | 0.812 |
| H2 root readout | 0.643 | 0.544 | 0.894 |
| H2 generated z | 0.606 | **0.434** | 0.976 |
| H2 generated features | 0.610 | 0.453 | 0.971 |
| root patch tokens (positive control) | 0.748 | 0.659 | 0.929 |

On zombie roots the positive control works: patch tokens beat DOWN by +0.126 [+0.062, +0.198]. None of
H2's input arms beats DOWN:

- z4 +0.009 [−0.047, +0.072]
- z4 + actions +0.022 [−0.034, +0.083]
- root readout +0.011 [−0.049, +0.072]

This agrees with z (last frame) on both sealed blocks: 0.542 vs 0.567 at 51k, 0.555 vs 0.538 at 53k.
**An alias-free CLS-H2 bridge cannot pass a zombie criterion against DOWN; its input already sits there.**

The world then makes it worse. Generated state against its own input on zombie roots:

- generated z −0.121 [−0.176, −0.076]
- generated features −0.102 [−0.146, −0.054]

Both are below DOWN, −0.099\* and −0.080\*, **even under all-action ranking supervision**. On lava the
transition helps: generated 0.97 against input 0.81–0.89. H2's transition trades mob consequences
for terrain consequences. So on zombie roots the H2 gate failure is the interface plus the world, not
the head: no head, supervision layout or alias fix can recover a hazard the input does not carry.
The depth alias remains a real defect of the trained head everywhere else.
