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

- **The per-tile world works as a world.** Its imagined successor keeps the root's decision
  information (step 6). The u→u world's generated state lost it; z-state worlds lose part of it. This
  is exploratory and unconfirmed on a sealed block.
- **The decision heads are what fail, for two measured reasons.**
  - **The tail-aligned terminal layout creates a position shortcut** (step 3). H2 tail-aligns its
    terminal windows the same way; whether H2 learned the same shortcut is **untested**.
  - **Factual outcome labels teach a context shortcut, not the action's consequence** (steps 7–8).
    Counterfactual labels, all actions from one state, are what teach a head to read it.
- **The added health head gave nothing and cost movement** (TH vs T). Reading only (s_t, s_{t+1}) did
  not make the predictor write the consequence visibly.
- **Not established:**
  - any sealed confirmation of steps 6–8;
  - that BCE heads can match the ranking probe (0.696 against 0.724; the probe objective matters, see
    memory);
  - more than one world seed;
  - anything about two steps or H16.

## Next test implied (not run)

Freeze the T world. Fit its decision head on its own generated states with counterfactual outcome
labels:

- the hazard-fork corpus (`broad_forks_v2`: 15,016 roots, all 17 successors and outcomes) at the
  evaluation position;
- a ranking-aware objective.

Judge it once on a new sealed block (54,000+) against the root tokens, the prior and actions_only. In
parallel, test cheaply whether H2's paired terminal loss is satisfied by position in H2 as well.
