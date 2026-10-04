# The revised 2×2 — declared result, and why it failed

Predeclared `spatial.py` (`0b3324e0`). P0 check `boundary_v4.json` (`ddda2b21`). Declared result
`spatial.json` (`41ddb109`). Post-hoc localization `spatial_why*.py`, each committed before it ran.

## Declared result (sealed seeds 53,000–53,402, 800 one-step opportunity roots): `spatial_health_fails`

P0 held: on this block the root's patch tokens beat CLS by +0.126 [+0.094, +0.159], and on zombie
roots they beat the 40×-wider CLS head by +0.093 [+0.049, +0.134].

Then, for the head each world was trained with, reading its own generated successor:

| trained head | overall | zombie | stay kills, a move survives |
|---|---|---|---|
| action prior | 0.558 | 0.538 | 0.509 |
| actions_only | 0.633 | 0.580 | 0.559 |
| root tokens (root + action) | 0.744 | 0.678 | 0.664 |
| Z | 0.437 | 0.167 | 0.069 |
| ZH | 0.443 | 0.174 | 0.078 |
| T | 0.485 | 0.254 | 0.167 |
| TH | 0.429 | 0.169 | 0.076 |

- **S1 fails:** TH − ZH on zombie roots −0.005 [−0.045, +0.032].
- **S2 fails:** TH − prior −0.129\*; TH − actions_only −0.204\*.
- **S3 fails:** TH − root tokens −0.315\*.
- **S4 fails:** Brier is equal across arms (0.541, which is what near-zero P(dead) everywhere scores),
  and TH's health cross-entropy is worse than ZH's (5.33 vs 2.81).
- **S5 holds:** TH's movement error is 0.97 against T's 0.63.

Every trained head chooses *below the prior*, mostly by staying put where that kills. Reported
contrasts:

- T − Z +0.049\* (zombie +0.087\*)
- TH − T −0.056\* (zombie −0.085\*): the health head hurt
- ZH − Z +0.006, not resolved

## Why — eight post-hoc steps

Steps 1–5 use the sealed block, TRAIN windows and DEV windows. Steps 6–8 use the exploratory
observability roots, which have been inspected many times.

| step | question | answer |
|---|---|---|
| 1 `spatial_why` | Head or world? | **T/TH: generation fails.** Trained heads read death on the REAL successor at AUC 0.998 / 0.996, and TH's health head is 98% right on real damaged pairs. On GENERATED successors P(dead) is ~1e-4 and health accuracy on damage is 0%. **Z/ZH: the head fails even in hindsight** (0.58). |
| 2 `spatial_why2` | Memorized, shifted, or never? | **Never.** The same happens on the world's own TRAIN terminal windows (one step ahead, P(dead) 1e-4 against 0.64 on the real frame) and on DEV. |
| 3 `spatial_why3` | How did training loss look fine? | **Position shortcut.** In the pool a death can occur only at a window's last frame. TH reads P(dead) 0.97 in exactly the trained configuration (second generated step, position 5), 0.36 at position 5 from real frames, and 0.0002 one step ahead at position 4. **The generated health tile stays nearer the copy** than the real successor's on 100% of damaging transitions. |
| 4 `spatial_why4` | Is damage forecastable, and computed? | **Yes and yes.** A probe on frame t + action reaches DEV AUC 0.909; the predictor's hidden state 0.90. |
| 5 `spatial_why5` | Is it written into the generated tile? | **Weakly.** A fresh probe on the generated health-tile pair reaches 0.91 (real pair 1.00): the world writes the forecast as a small displacement, not the large change the heads were fitted to. |
| 6 `spatial_why6` | Does the generated STATE carry the within-root decision? | **Yes, fully.** A fresh per-branch ranking probe reads 0.724 on T's generated state against 0.724 for the root's tokens (−0.000 [−0.047, +0.047]). TH 0.711; Z 0.682; ZH 0.651\* below the root. |
| 7 `spatial_why7` | Can a head trained on FACTUAL generated states, with no position cue, read it? | **No.** T 0.628 = prior (0.659 with history). Factual data teaches the head whether the *context* is doomed, and the context is the same for all 17 actions. |
| 8 `spatial_why8` | Population or labels? | **Labels.** On the same fork roots and with the same BCE head: factual labels 0.650, all-17 counterfactual labels 0.696 (T). Z: 0.554 / 0.582. |

## What this establishes, and what it does not

> **Corrected 2026-09-26, after review.** The first version's headline, "the per-tile world works as
> a world", overclaimed, and is withdrawn.
>
> Step 6 shows that T's generated tokens hold a **recoverable safety signal**: a ranking probe finds
> it. But a successor generated from the root and the action can carry root-plus-action information
> by passing its input through, without ever drawing the right consequence. The rest of the evidence
> says T does not draw it:
> - the generated health tile stays nearer the unchanged tile (step 3);
> - T's trained head, which reads real successors at AUC 0.998, reads generated ones at 0.557
>   (step 1).
>
> The reviewer's exploratory re-fit of the step-6 probe on the already-opened 53k block gave 0.682
> against a 0.558 prior (+0.124 [+0.066, +0.180]). Its three seeds varied (0.732 / 0.589 / 0.725),
> against a root-token reference of about 0.737 there. So "keeps *everything* the root knows" is not
> shown. No matched root-tokens-plus-action per-branch control has been run, so it is also not shown
> that the transition adds anything to its input.

- **Established, as a failure of the trained world-and-head system:**
  - **A position alias.** All 8,155 training deaths sit at window position 5. The trained TH system
    says 0.97 there and 0.0002 one step earlier.
  - **No visible consequence in the generated successor.**
- **Recoverable safety signal, exploratory, in two populations:** T's generated tokens hold a
  within-root safety signal that fresh ranking probes find (0.724 on the 50k roots, 0.682 on 53k).
- **Labels:** heads trained on factual labels choose at or near the prior (steps 7–8). Step 8's
  factual → all-action gain (0.650 → 0.696) also changes the number and distribution of labelled
  branches, and has no paired interval; it does not isolate label type.
- **The added health head gave nothing and cost movement** (TH vs T).
- **Scope:** T is a six-layer **Transformer**, not Mamba. Nothing here establishes a Mamba-LeWM
  repair.
- **Contract:** the per-tile state is a declared deviation from the canonical recipe (TC-07 / TC-19:
  patch tokens do not enter the world). Fork-label heads are outside it (TC-17). A fork-supervised
  head that selects safely would be a **fork-supervised safety readout**, not a repaired world.
- **The H2 alias is untested.** H2's terminal sampler is tail-aligned too, and its paired loss scores
  the last two positions.

## Next steps (after review; not run)

1. **Frozen-T head comparison.** No retraining. Same heads and budget, on the partition's FIT-train
   seeds (fit) and FIT-dev seeds (selection) only, never the gate-reserved or unallocated seeds:
   - logged-action labels;
   - one uniformly sampled action per root;
   - all-action BCE;
   - all-action ranking;
   - a matched **root-tokens-plus-action** per-branch head.

   Rules committed before 54k is collected or read. Judged on:
   - within-root safe choice, overall, on zombie roots and on stay-versus-move roots;
   - calibration on ordinary as well as opportunity roots;
   - action histograms;
   - variability across head seeds.
2. **A separate world-fidelity verdict.** For a native repair: retrain with terminal and comparable
   non-terminal transitions at the *evaluation* prediction position (no alias). Then test whether the
   generated successor expresses damage and death: the health tile, and a real-fitted head reading
   generated successors.
3. **H2 position audit.** Fixed transitions scored at matched positions in the canonical H2 world.
