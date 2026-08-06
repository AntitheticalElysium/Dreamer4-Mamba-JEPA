# Audit of the Craftax encoder-LR experiment + two initialization defects

Date: 2026-07-29
Deviations: D046 (new), D045 (corrected), D037 (retrospectively confounded)

Independent verification of a second agent's queue9 experiment and its audit.
Every claim below was re-derived from code or artifacts, not accepted.

## Claims checked

| claim | verdict | evidence |
|---|---|---|
| T/M shared-init RNG confound, 16 tensors | **CONFIRMED** | rebuilt both arms at seed 20260727: 227 shared tensors, 16 mismatched, all four JEPA MLPs; encoder bit-identical (62 tensors) |
| BC/value heads not reseeded | **CONFIRMED** | `CartPoleBCPolicy` and `CartPoleValueHead` constructed with no preceding `torch.manual_seed` |
| Arms not parameter-matched | **CONFIRMED** | 986,348 (T) vs 996,202 (M), +1.00% total, temporal modules +29.6% |
| Imagination context truncated to 8 | **CONFIRMED** | `imagination_actor_critic.py:517` re-slices `[:, -context:]` every step |
| Dreamer-CDP Crafter `deter` is 8192 | **CONFIRMED** | `configs.yaml:93`; 4096 is the `rssm.py:19` dataclass default. D045 cited the default, so the divergence from our 64-d channel is 128x, not 64x |
| D021-D023 dangling | **CONFIRMED** | 2 citations each, 0 rows |
| "provenance consistent" | **FALSE** | see below |

## What the audit missed: the full-LR arms are cross-implementation

`craftax_lr_experiment_summary.json` records `implementation_drift_allowed:
true` for `full_t` and `full_m` ONLY, with `stored_implementation_sha256`
`c513c752…` against `b178cbf8…` for the slow arms.

The full arms are the ORIGINAL 2026-07-27 baseline checkpoints
(`outputs/.../craftax_expert_v1`, world `c6742b6d…`); the slow arms were trained
fresh in `craftax_slowenc_v1`. So the headline "slow beats full" comparison
moves encoder LR AND code version together, and the audit reported provenance as
consistent when the summary says otherwise.

VERIFIED BENIGN, post hoc rather than at the time:

* the D045 predictor change is RNG-inert — rebuilding a world with the
  pre-D045 `CDPPredictor` monkeypatched in gives 237/237 (transformer) and
  243/243 (mamba2) identical tensors;
* the `jepa_weight` wiring multiplies by exactly 1.0 at the default;
* the only config delta between the two world reports is the new
  `jepa_predictor_context` field.

So the executed comparison stands. It was bypassed, not checked, and that is the
part worth recording.

## The two defects, and the fix (D046)

`replace_dynamics_time_attention` ran BEFORE `_build_jepa`, consuming RNG, so
every `mamba2` world drew different JEPA predictor/projector weights than the
`transformer` world at the same seed. Neither Craftax head runner reseeded, so
BC and value heads inherited backend-dependent RNG state as well.

Together these void D037's "the temporal operator is the single moved axis"
contract for EVERY T-vs-M comparison — Craftax and CartPole alike, since
`train_jepa_world` seeds then builds under the same construction order.

Fixed: backend substitution moved to the end of `__init__`; both head runners
reseed before constructing their head. After the fix, 227 shared tensors and 0
mismatched, with every backend-specific key containing `.time.`. Two regressions
pin it; suite 115 passes.

The magnitude of the confound on past results is UNKNOWN — no initial head
states were saved. D037's T-vs-M deltas are therefore uninterpretable as
single-axis measurements, not known-wrong.

## Executed results as they stand (queue9, one training seed)

| arm | random | BC | actor | actor − BC (95% CI) |
|---|---:|---:|---:|---|
| T full | 1.640 | 2.606 | 2.481 | −0.124 [−0.862, +0.456] |
| T slow | 1.640 | 2.353 | **4.395** | **+2.042 [+0.960, +2.623]** |
| M full | 1.640 | 3.604 | 2.948 | −0.656 [−1.316, +0.120] |
| M slow | 1.640 | **3.554** | 3.292 | −0.262 [−1.177, +0.626] |

The pattern that matters for the current priority: **M has the stronger BC in
both LR conditions (3.55-3.60 vs 2.35-2.61) and the weaker actor.** Slow-T's
actor is the best absolute policy on the branch (4.395), and it is the only
actor that beats its own BC.

Reading these CIs correctly: they bootstrap ENVIRONMENT seeds only. There is one
training seed and one policy-sampling schedule per condition, so they do not
cover training-seed or deployment-noise variance.

## Runs launched

1. **Deployment-noise check** — re-evaluate the queue9 slow checkpoints under two
   further policy-sampling seed bases (7100000, 7200000). Cheapest test of
   whether the slow-T gain survives stochastic deployment. Uses
   `--allow-implementation-drift`, justified because the only change since those
   checkpoints is construction ORDER: `state_dict` keys and shapes are identical
   (243 tensors), so a strict load fully determines every parameter.
2. **Corrected paired T/M pipeline** at `enc_lr=6e-6` with D046 applied
   (`outputs/.../craftax_fixedinit_slow`), then executed evaluation on the SAME
   30 seeds (100000-100029, context 8, sampled, `policy_seed_base` 7000000).

Purpose of (2): whether M-JEPA's imagination failure survives a comparison in
which the two arms actually share an initialization.

## Not pursued

SIGReg, per instruction. Recording only that the queue9 SIGReg cells are a
single-seed throwaway and should not be cited as a Craftax rejection of D036.
