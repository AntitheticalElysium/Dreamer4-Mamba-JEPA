# Frozen Raw readout ladder — result

Run 2026-09-21 on the sealed Raw H2 bridge checkpoint (`raw/bridge/step-002000.pt`). Nothing in
the world model was trained; only fresh probe heads were fitted, on all-action fork roots disjoint
from the gate's. Rules and protocol were fixed in `PREDECLARATION.md` before the numbers existed.

- 7,085 fit roots / 3,369 judgement roots, disjoint by episode seed (sealed partition
  `aab9f3e483bf6d2c`), 17 actions each: 120,445 and 57,273 rows.
- Model selection every 100 updates on 15% of the **fit** roots, held out by root.
- Evidence: `evidence/ladder.json`. Rerun with the added control reproduced every shared number
  from the first pass bit for bit.

## Declared verdict

**`transition_or_generated_feature`** — rule 2: real-successor features succeed on held-out roots
where generated ones fail. This is what the rules, fixed in advance, return. The rest of this
document is what the numbers say beyond that, including two things the rules could not see.

## The matrix (judgement roots)

`vs_marg` is against the action-marginal control; `vs_context` is against the `context_action`
head, which sees the root and the action but **no successor at all**. Brackets are 95%
seed-clustered bootstrap intervals.

| family | metric | score | marginal | assoc | vs_marg | vs_context |
|---|---|---|---|---|---|---|
| context_action | reward regret | 0.1694 | 0.2460 | +0.350 | **+0.077** [+0.056,+0.100] | reference |
| context_action | safe-choice | 0.5922 | 0.6203 | +0.159 | −0.028 [−0.114,+0.055] | reference |
| successor_cls | reward regret | 0.2012 | 0.2460 | +0.231 | **+0.045** [+0.029,+0.060] | −0.032 [−0.054,−0.009] |
| successor_cls | safe-choice | 0.9469 | 0.6203 | +0.692 | **+0.327** [+0.270,+0.383] | **+0.355** [+0.302,+0.410] |
| successor_z | reward regret | 0.2333 | 0.2460 | +0.072 | +0.013 [−0.008,+0.033] | −0.064 [−0.091,−0.037] |
| successor_z | safe-choice | 0.7016 | 0.6203 | +0.227 | **+0.081** [+0.011,+0.150] | **+0.109** [+0.044,+0.170] |
| real_features | reward regret | 0.1608 | 0.2460 | +0.478 | **+0.085** [+0.068,+0.104] | +0.009 [−0.012,+0.029] |
| real_features | safe-choice | 0.8359 | 0.6203 | +0.547 | **+0.216** [+0.149,+0.273] | **+0.244** [+0.190,+0.299] |
| generated_features | reward regret | 0.2380 | 0.2460 | +0.242 | +0.008 [−0.002,+0.018] | −0.069 [−0.089,−0.048] |
| generated_features | safe-choice | 0.5000 | 0.6203 | +0.129 | **−0.120** [−0.201,−0.042] | **−0.092** [−0.152,−0.035] |

Cross cells, judgement roots: a head fitted on real features scores **−0.150** [−0.234,−0.067]
against the marginal on generated features — the worst cell in the matrix — and a head fitted on
generated features scores −0.072 [−0.155,+0.008] on real features. Transfer fails in both
directions.

Train-minus-DEV gaps are small (reward −0.042 to +0.007, terminal −0.071 to +0.054). Selection
bit: selected steps are 700–3,900 of 6,000. These are not memorization artifacts.

## What the numbers say

**1. The whole signal is in termination. Reward carries nothing the successor adds.**
Against the action-marginal, `real_features` looks like a win on reward (+0.085). Against
`context_action` it is **+0.009 [−0.012,+0.029]** — indistinguishable from zero. Every other
family is significantly *worse* than root+action on reward. So the reward ranking that beats the
marginal is a root-and-action effect; seeing the successor adds nothing to it. Had the ladder
only carried the action-marginal control, this would have read as a reward success. It is not one.

This agrees with, and explains, the H2 gate's true-successor substitution, which found the trained
reward head no better with the real successor (0.2879) than the generated one (0.2781).

**2. On termination, the real path carries it and the generated path is worse than not looking.**
`real_features` reaches 0.836 safe-choice against a 0.620 marginal, +0.244 over root+action.
`generated_features` reaches 0.500 — below the marginal (−0.120) *and* below root+action (−0.092).
The generated successor does not merely lose the termination signal; conditioning on it is worse
than ignoring the successor entirely. The cross cells say why this is not just extra noise: a head
that reads real features well scores its worst result on generated ones, so the two live in
different regions of feature space.

**3. The projector is a large bottleneck, and `observe_latent` is not.**
`successor_cls` and `successor_z` come from the same forward pass and both lack root context, so
they are directly comparable. On termination, CLS is +0.355 over root+action and z is +0.109, with
non-overlapping intervals; within-root association falls 0.692 → 0.227. Most of the termination
information in the encoder's CLS does not survive the projection into the `z` the world
transitions.

`observe_latent` then *recovers* much of it: raw z is +0.109, but `real_features` —
`observe_latent(root, a, z_true)` — is +0.244, because it recombines z with the root state. So the
world's own readout is not the bottleneck. **Declared rule 3 is refuted by its own measurement.**

**4. The trained bridge head reads less termination than a fresh head does from the same features.**

> **CORRECTED 2026-09-23, and the correction matters.** This section originally claimed the
> trained bridge head *fails* to read termination present in its own features, resting on the H2
> gate's true-successor safe-choice of 0.559 against a 0.657 marginal — below the control. The
> confirmation run (`CONFIRM.md`) scored that same trained head on 4,047 roots from the 405
> unallocated seeds and got **0.700 against a 0.609 marginal, +0.091 [+0.030,+0.158]** — *above*
> the control, resolved. The trained head does **not** fail. The original claim relied on the
> gate's 512-root, 102-terminal-opportunity population, and the sign flips on a larger matched one.
>
> What survives is weaker and specific: on identical roots, an exact-capacity fresh head reaches
> **0.848** where the trained head reaches **0.700**. See `CONFIRM.md` for the direct paired test.

The reason the declared rules could not see this at all is that they only compare fresh heads to
each other; nothing in the ladder scored the trained head.

## What this does and does not establish

Established, on held-out roots with the action-marginal and no-successor controls, at small
train/DEV gaps:

- Termination is a genuine state-conditioned consequence here; within-root reward ranking largely
  is not.
- The real-successor route carries termination; the generated route is worse than ignoring the
  successor.
- The projector loses most of CLS's termination information; `observe_latent` is not the bottleneck.
- Fresh heads read termination from real features better than the trained bridge head does
  (see the correction in section 4: the trained head does beat the marginal, it just extracts
  less).

**Not** established, and not to be claimed:

- That repairing the transition would make the bridge head work. Two separate deficits are visible
  (the generated path, and the trained head's reading of the real path); neither is shown to be
  the cause of the gate failure, and fixing one does not predict the other.
- Anything licensing H16 or actor training. This is a localization instrument. The stop decision
  stands.
- That the reward null generalizes beyond this fork population's reward structure (marginal regret
  is only 0.246, a small absolute scale).
- `successor_cls`/`successor_z` carry no root context, so their reward deficit against
  `context_action` is expected and is **not** evidence about the projector on reward. The
  projector claim rests on the CLS-vs-z terminal comparison alone, which is matched.

## Reproduce

```
./artifacts/experiments/20260921_readout_ladder/run.sh
```

`TRITON_F32_DEFAULT=ieee` is required: `sources.py` records the numeric execution flags in every
checkpoint and `verify_lewm_sources` is strict, so loading the bridge parent without it fails with
`source/dependency drift in ['execution']`. No file in the source closure was edited.
