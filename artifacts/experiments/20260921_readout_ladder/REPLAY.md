# Repeated-key replay pilot — is within-root death random given the full state?

Run 2026-09-24. 60 of the 405 judgement seeds (seeded sample), re-walked with the fork collector's
own frozen BC policy, loop and keys; 578 stored roots; 32 independent key pairs per root, each an
all-17-action step followed by the stored NOOP second step. Exploratory: these seeds were already
examined post hoc. Script `replay.py`, committed before the run (`d681f00f`). Evidence:
`evidence/replay_pilot.json`, per-root probabilities in `evidence/replay_pilot_rows.pt` (hash-bound).

## The replay is exact

At all 578 roots the replayed 32-frame history matched the stored frames with **zero** drift, and
the original key reproduced every stored one-step death and every stored NOOP-second-step death —
**zero** mismatches. These are the trajectories that produced the fork store.

## One-step death is deterministic given the full state

| | one-step death | two-step death |
|---|---|---|
| roots / opportunity roots | 578 / 90 | 578 / 114 |
| (root, action) pairs with 0 < P < 1 — all roots | **0.17%** | 6.5% |
| … on opportunity roots | **1.1%** | 20.7% |
| opportunity roots with any random action | **2 of 90** | 106 of 114 |
| full-state oracle, expected safe choice | **1.000** | **0.894** |
| full-state oracle, realized safe choice | 1.000 | 1.000 ¹ |
| FIT-root action prior (DOWN), realized | 0.511 | 0.702 |
| always-RIGHT, realized | 0.544 | 0.746 |
| oracle − prior, paired seed-clustered | +0.489 [+0.348, +0.637] | +0.298 [+0.209, +0.414] |

On every one of the 90 one-step opportunity roots, some action has P(death) = 0 over 32 draws, and
the realized outcome agrees with the majority-P outcome on 99.9% of pairs. **Simulator randomness
does not explain the generated path's one-step failure**: with the state known, the right action
is knowable essentially every time. TRANSITION.md's explanation (a) — "termination given (state,
action) is substantially random" — is refuted for one step.

What this does not settle is **hidden state**. The oracle knows everything, including mob attack
cooldowns the frames never show. A predictor reading pixels could still face irreducible
uncertainty the oracle does not. That is what the 32-frame, objective-matched probes in
`rankprobe.py` measure from the other side.

Two-step death carries real randomness — a fifth of opportunity pairs — which is where mob
movement enters. SLEEP's delayed hazard is plain in the probabilities: mean P(two-step death) is
**0.253 after SLEEP against 0.141 after NOOP** and 0.131 after RIGHT; one step, they are 0.109,
0.112 and 0.069.

## ¹ A selection effect in how every panel here is scored

On two-step death the oracle's *expected* safety is 0.894 but its *realized* safety is 114/114.
19 of those roots have no certainly-safe action (oracle choice P between 0.25 and 0.84), yet the
chosen action survived the stored draw on all 19. That is not luck and not a bug. All 17 actions
at a root share one draw, and a root is only scored if that draw left some action alive and some
dead. When the hazard is nested — the least-exposed action dies only in draws that kill the rest
— those draws never produce a scorable root, so a good ranker's realized score is inflated.

This touches every safe-choice number in this directory scored against the stored single draw.
For one-step death it is inert, because the outcomes are deterministic. **For two-step outcomes,
realized safe-choice overstates any ranker that tracks P, the oracle most of all; compare rankers
on the same roots, and prefer probability-scored metrics where P is available.**
