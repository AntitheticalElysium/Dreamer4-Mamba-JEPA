# Is the information missing from generated latents, or is the decoder failing on them?

Status: **complete**, 2026-09-17. Read-only refit over published M03 features; no gate
output changed, nothing authorized. Runner: [`generated_readout.py`](generated_readout.py),
evidence in [`evidence/readout.json`](evidence/readout.json).

## Question

M03 fits one decoder on `train_observed_successor` and applies it unchanged to observed,
generated and reset latents (`gate.py:1784`; `_fit_probe_many` defends the design in its
docstring). That is the right instrument for asking whether generated latents land in the
*same* decodable space as observed ones. It cannot distinguish:

- the information is **absent** from generated latents, from
- the information is **present**, in a subspace this decoder was never fitted to.

Every "generated" number in the M03 report, and in the memory supplement, inherits that
ambiguity. So: refit, on the same rows, against the same labels and floors.

## Answer: the information is present, and the decoder was failing badly

Outcomes, mlp, mean AUC over measured targets:

| arm | obs-fit → gen | **gen-fit → gen** | gen-fit → obs | obs-fit → obs | root+action | action |
|---|---|---|---|---|---|---|
| TC consecutive | 0.6375 | **0.7549** | 0.6588 | 0.7409 | 0.7882 | 0.6825 |
| TC strided | 0.6098 | **0.7462** | 0.6095 | 0.7495 | 0.7350 | 0.6825 |
| Direct-Mamba | 0.5686 | **0.7107** | 0.6007 | 0.7084 | 0.6972 | 0.6825 |
| Direct-Attention | 0.5667 | **0.7134** | 0.6046 | 0.7084 | 0.6972 | 0.6825 |

Refitting recovers +0.12 to +0.15 mean AUC in **every arm, including both Direct anchors**.
After refitting, generated latents decode as well as observed ones do (0.7549 against 0.7409
for the consecutive arm) -- they are not information-poor.

The recovery is concentrated, and its signature is unmistakable:

| arm | death | damage |
|---|---|---|
| TC consecutive | 0.413 → **0.696** | 0.422 → **0.683** |
| TC strided | 0.409 → **0.703** | 0.422 → **0.707** |
| Direct-Mamba | 0.441 → **0.788** | 0.431 → **0.767** |
| Direct-Attention | 0.440 → **0.782** | 0.430 → **0.770** |

All eight are separated by non-overlapping bootstrap intervals. The observed-fit decoder
scored **below 0.5** on every one -- it was anti-correlated on generated latents, which is
mis-decoding, not absent information. `inventory_changed` and `tile_changed` were already
0.83-0.95 and barely move; `reward_positive` and `achievement_event` stay low either way.

Neither fit transfers to the other distribution: gen-fit read on observed drops to 0.60-0.66.
The two latent families each carry the information, in **different representations**, so the
failure is symmetric and is a property of the measurement, not of one distribution.

## But it buys nothing over persistence

The honest counterweight. Against a root-latent-plus-action floor -- what is knowable from
the present state and the chosen action, with no world-model prediction at all:

| arm | target | root+action | gen-fit | Δ | separated |
|---|---|---|---|---|---|
| TC consecutive | death | 0.683 | 0.696 | +0.013 | no |
| TC consecutive | damage | 0.675 | 0.683 | +0.008 | no |
| TC strided | death | 0.705 | 0.703 | −0.002 | no |
| TC strided | damage | 0.713 | 0.707 | −0.006 | no |
| Direct-Mamba | death | 0.787 | 0.788 | +0.001 | no |
| Direct-Mamba | damage | 0.775 | 0.767 | −0.008 | no |

**Not one comparison separates.** Once measured fairly, the one-step generated successor
carries about what the current state and the action already carry, and no more -- in every
arm, TC and Direct alike.

## What this changes

The generated-condition deficits reported across M03 and the memory supplement are
**substantially a measurement artifact**, and death/damage numbers below 0.5 should have been
read as a decoder failing, not as a world model failing. Conclusions drawn from them need
re-reading, this experiment's own floors included.

What survives is narrower and better grounded: the one-step world model adds nothing over
persistence on consequence decoding. That is a statement about the *setup* -- one-step
horizon, deterministic MSE target, frozen encoder -- and it holds for Direct too, so it is
not a TC-LeWM verdict.

## What it does not establish

One dataset, one split, 256 train and 128 dev roots fanned over 17 actions. Refitting on
generated features risks a decoder learning generator artifacts; the root and action floors
bound that, but do not eliminate it. Nothing here evaluates multi-step rollout, and nothing
here is a capability claim: M03's status is unchanged and `m4_authorized` remains false.
