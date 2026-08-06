# M-JEPA: the combined Mamba + non-generative JEPA arm

Date: 2026-07-23
Branch: `d4-jepa-arm`
Deviations: `D037` (the arm), `D038` (development-seed actor-budget selection)

## What was built

`M-JEPA` replaces only the dynamics `TimeSelfAttention` modules of the screened
`T-JEPA` actor-critic with official Mamba-2 modules at the D022 state expansion
(`d_state=64`, `headdim=64`, `expand=1`, `d_conv=4`). Every other axis is held
identical: online encoder, SPR/BYOL EMA anti-collapse (D031), multi-step
`jumps=5` (D034), terminal weighting (D035), deterministic predictor-as-rollout
(D030), reward/continuation heads, actor/value, BC prior, pixel adapter,
schedules, and the sealed protocol. This is the combined thesis architecture and
it had never been run: every previous JEPA world in `outputs/` is `T-JEPA`.

Checkpoints:

| Artifact | SHA-256 |
|---|---|
| world | `ece982ada8212b978f5949da3d16337be60fc759b1e31bb9409ce634f15dc5fd` |
| BC policy | `a0d0601315a774ecab00ccf81488b22b8a3e7e96738e1607c719c148ff35f4bd` |
| actor (500) | `bbca73bb56819ec61b4ea79de9a5f0ce9f57118b2a50c9ca5c13feffe7e0a9bb` |

## Gate 2 — good vs bad imagination: PASS, and better than T-JEPA

| | M-JEPA | T-JEPA |
|---|---:|---:|
| imagined return, BC | **16.41** | 59.23 |
| imagined return, uniform | 12.29 | 56.28 |
| imagined return, anti-BC | 9.89 | 54.71 |
| good − bad gap | **6.52** | 4.52 |
| BC / anti-BC ratio | **1.66x** | 1.08x |
| imagined continuation (mean) | **0.284** | 0.998 |

Both arms order the policies correctly, but `M-JEPA` separates them roughly
eight times more strongly in relative terms, and its continuation is not
saturated. `T-JEPA` still predicts continue ≈ 1.0, the long-running D018/D035
pathology; `M-JEPA` is the first world in this project whose imagined episodes
actually end.

## Gate 3 — latent fidelity probe (new instrument)

A probe is fitted on **real** agent tokens to predict the four CartPole state
variables and terminal-within-5, then applied to **imagined** tokens. Ground
truth at each horizon is the same action sequence replayed through the pinned
CartPole dynamics from the true state at the end of the context.

Real held-out probe quality (the probe itself is sound on both arms):

| | cart_x | x_dot | theta | theta_dot | terminal AUC |
|---|---:|---:|---:|---:|---:|
| M-JEPA | **0.612** | 0.790 | 0.724 | 0.810 | 0.954 |
| T-JEPA | 0.255 | 0.755 | 0.810 | 0.826 | 0.965 |

Imagined, by horizon:

| h | NRMSE M / T | pred-state variance M / T | \|BC−antiBC\| M / T |
|---:|---|---|---|
| 1 | 1.40 / 0.70 | 0.213 / 0.149 | 0.536 / 0.430 |
| 4 | 1.67 / 0.85 | 0.132 / 0.133 | 0.687 / 0.348 |
| 8 | 2.00 / 1.30 | 0.137 / 0.205 | 0.423 / 0.405 |
| 16 | 2.54 / 1.59 | 0.078 / 0.213 | 0.209 / 0.445 |
| 32 | – / – | 0.036 / 0.178 | 0.164 / 0.440 |

Read honestly, this is the one place `T-JEPA` looks better: `M-JEPA`'s imagined
latents drift further from real state, their across-batch variance contracts
(0.213 → 0.036), and BC-vs-anti-BC separation decays (0.536 → 0.164), where
`T-JEPA` stays flat. The likely reason is benign and is exactly what gate 2
shows — `M-JEPA`'s imagined episodes terminate early (continuation 0.284), so
the h=32 latent is mostly post-termination and carries nothing. But the probe as
built cannot yet separate "the rollout ended" from "the representation
contracted", and that separation is the obvious next instrument.

A measurement limit worth recording: ground truth is an **open-loop** replay, and
open-loop CartPole terminates quickly, so NRMSE is only measurable to h≈16
(alive counts fall to 3/191 by h=32). The variance and divergence curves need no
ground truth and span the full range.

## Gate 1 — imagination vs its own BC: NOT CLEARED, and why

| tier | actor | BC | delta | 95% CI | CI > 0 |
|---|---:|---:|---:|---|---|
| 987 | 465.42 | 454.87 | +10.55 | [−6.93, 29.37] | no |
| 988 | 455.93 | 454.30 | +1.63 | [−22.41, 24.86] | no |

Both tiers return `DREAMER4_ACTOR_CRITIC_PARITY` under the evaluator's own
non-inferiority rule, but neither clears a strictly positive CI.

The budget is not the cause. D038 ran the pre-declared 250/500/1000/1500 ladder
on the reserved development seeds: 500 is the argmax (+29.33 [0.47, 64.30]) and
every other rung is worse. The selected checkpoint is the one already evaluated.

The cause is a task ceiling, and it is measured. `M-JEPA`'s BC is so strong that
it reaches the CartPole-v1 500-step limit on **64/100** (987) and **62/100**
(988) sealed seeds, and **both** actor and BC sit at the cap — mechanically
forcing the paired delta to exactly zero — on **50/100** and **52/100** seeds.
About half the sealed evidence therefore carries no signal at all, and the
remaining half is dominated by a few catastrophic episodes on either side.

This is the uncomfortable structural point: **the better the world model, the
stronger its BC, and the less headroom remains for imagination to prove itself
against that BC.** Gate 1 is self-limiting on a task with a hard return cap.

### Is the actor actually better, or just untestable? (D039)

The ceiling hypothesis was tested directly on development seeds by lifting only
the `TimeLimit` truncation:

| cap | actor | BC | delta | 95% CI | BC at cap | both at cap |
|---:|---:|---:|---:|---|---|---|
| 500 | 474.43 | 445.10 | +29.33 | [0.17, 64.27] | 17/30 | 13/30 |
| 1500 | 910.90 | 788.57 | **+122.33** | [−54.97, 300.90] | 3/30 | 0/30 |

The actor **is** genuinely better: removing the cap quadruples its advantage
(+29 → +122; 6.6% → 15.5% relative) and empties the saturation bucket. But the
confidence interval *widens* rather than narrows, because episode-return variance
grows faster than the mean. **Lifting the cap confirms the effect and still
cannot deliver a positive CI**, so it is not a route to gate 1, and no sealed
tier was re-run under it.

### How much evidence would gate 1 actually need?

From the observed per-seed spread of the sealed deltas:

| tier | mean | sd | half-width at n=100 | seeds needed for CI > 0 |
|---|---:|---:|---:|---:|
| 987 | +10.55 | 93.5 | 18.3 | **301** |
| 988 | +1.63 | 120.7 | 23.7 | **21,074** |

Tier 987 is within reach. Tier 988 is not: its true effect is about +1.6, which
is indistinguishable from zero at any tractable sample size. Because gate 1
requires *both* tiers, it is not attainable for this arm on this benchmark. That
is a demonstrated limit, not an unfinished search.

## Head-to-head vs T-JEPA (paired, identical sealed seeds)

| tier | comparison | M-JEPA | T-JEPA | delta | 95% CI | CI > 0 |
|---|---|---:|---:|---:|---|---|
| 987 | actor | 465.42 | 423.81 | **+41.61** | [19.32, 64.74] | **yes** |
| 988 | actor | 455.93 | 433.40 | +22.53 | [−7.02, 52.00] | no |
| 987 | BC | 454.87 | 392.97 | **+61.90** | [35.97, 88.12] | **yes** |
| 988 | BC | 454.30 | 380.28 | **+74.02** | [42.20, 106.02] | **yes** |

`M-BASE`/`T-BASE` remain historical context only.

### Why the actor gains so little: the mechanism (D040)

The obvious suspect was the continuation head. `M-JEPA` imagines continuation
0.284 over 32 steps while its BC really survives ~227 steps, which looks like
gross over-prediction of termination and would have justified a per-arm terminal
weight. Measured on held-out **real** states, that hypothesis is refuted:

| on real states | M-JEPA | T-JEPA |
|---|---:|---:|
| mean predicted P(continue) | 0.9946 | 0.9994 |
| empirical continuation rate | 0.9896 | 0.9896 |
| calibration error | **+0.0050** | +0.0097 |
| implied vs true episode length | 186 vs 96 | 1552 vs 96 |

`M-JEPA`'s head is *better* calibrated than `T-JEPA`'s. The 0.284 therefore
arises only on **imagined** states, and it lines up exactly with the fidelity
curves above (imagined NRMSE 1.40 → 2.54 against `T-JEPA`'s 0.70 → 1.59, state
variance contracting 0.213 → 0.036).

The complete mechanism is therefore:

1. `M-JEPA` predicts one step better (dev cosine 0.675) and represents state
   better (BC 0.9105, executed 454).
2. Its multi-step imagined rollouts nevertheless drift off the real manifold
   faster than `T-JEPA`'s.
3. The correctly-calibrated continuation head reads those drifted latents as
   terminal, so imagined survival collapses.
4. The actor consequently trains on an effective horizon of roughly 8 of 32
   steps, and can only nudge an already-excellent BC.

The lever this identifies — reducing multi-step rollout drift — is a change to
the predictor and its training, i.e. the architecture this arm holds fixed.

### Can the drift be trained away? (D041)

D040 identified rollout drift as the residual cause, and there was one untried,
source-faithful lever: the world is trained on `jepa_jumps=5` autoregressive
steps but the actor imagines 32, so drift past step 5 is unconstrained. The SPR
`jumps` mechanism (D034) already parameterises exactly this. A pre-declared
5/8/11 ladder was selected on development seeds (11 is the ceiling under the
existing `jepa_jumps < sequence_length` check at `sequence_length=12`).

| jumps | dev cos | BC acc | imagined continuation | imagined return | dev actor | dev BC | delta | gate-2 gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5** | 0.675 | 0.9105 | 0.444 | 6.25 | 474.43 | 445.10 | **+29.33** | 6.52 |
| 8 | 0.721 | 0.9134 | **0.978** | **68.54** | 245.87 | 368.40 | −122.53 | **23.07** |
| 11 | 0.556 | 0.8089 | 0.840 | 8.16 | 42.07 | 50.03 | −7.97 | 0.64 |

The drift mechanism is **confirmed**: at `jumps=8` imagined continuation rises
0.444 → 0.978 and imagined return 6.25 → 68.54, the world improves on every
metric, and gate-2 divergence becomes the best in the project (23.07; BC 33.8 vs
anti-BC 10.7, a 3.15× separation). **And the actor collapses anyway**, to −122.53
against its BC — a long, now-survivable imagined horizon simply lets it exploit
the model's residual long-horizon error. At `jumps=11` world training
destabilises outright.

So the ladder selects 5 and the inherited value stands. The result is a genuine
tradeoff, not a tuning failure:

- **too short** (5): imagined survival collapses, the actor gets little signal,
  and improves only marginally;
- **matched** (8): the actor gets abundant signal and exploits it — the
  compounding-error regime of *On Training in Imagination* (`2605.06732`),
  reproduced here;
- **too long** (11): the world itself destabilises.

Fixing the drift is therefore necessary but nowhere near sufficient; what the
actor needs is a rollout that is both survivable *and* accurate, which this
architecture does not deliver at 32 steps.

## Verdict

The combined architecture works. Swapping the temporal operator to Mamba-2
improves the world (dev cosine 0.675 vs 0.648, JEPA loss 0.236 vs 0.373 with no
collapse), the representation (BC 0.9105 vs 0.9077; executed BC +62 and +74 with
CIs above zero), executed control (actor +41.61 with CI above zero on 987), and
imagination quality (gap 6.52 vs 4.52, and the first non-saturated continuation
head in the project).

Gate 1 as specified is not achievable on CartPole for this arm, not because
imagination fails but because the arm's own BC saturates the benchmark. Deciding
what to do about that — a higher time limit, a harder task, or a gate defined
against a fixed reference policy rather than the arm's own BC — changes the
sealed protocol and is the maintainer's call, not an implementation change.
