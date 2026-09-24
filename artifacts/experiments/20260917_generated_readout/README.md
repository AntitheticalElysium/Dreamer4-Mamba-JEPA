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

Neither fit transfers to the other distribution: gen-fit read on observed drops to 0.60-0.66,
so the failure is symmetric rather than a property of one distribution. It does **not** follow
that this is "the same information, merely rotated": refitting moves the TRAIN-derived
normalization and the decoder weights together, so a rotation and a genuinely different
encoding are not separated here.

## But it shows no advantage over a trained root+action predictor

The honest counterweight, stated carefully. The comparison baseline is **not** persistence
or copying the current state: it is a *separately trained* predictor of the same future
labels from the current `z` plus the action, which can learn one-step consequences on its
own. Matching it means no demonstrated advantage on these probes -- not that the world model
is useless.

| arm | target | root+action | gen-fit | Δ | separated |
|---|---|---|---|---|---|
| TC consecutive | death | 0.683 | 0.696 | +0.013 | no |
| TC consecutive | damage | 0.675 | 0.683 | +0.008 | no |
| TC strided | death | 0.705 | 0.703 | −0.002 | no |
| TC strided | damage | 0.713 | 0.707 | −0.006 | no |
| Direct-Mamba | death | 0.787 | 0.788 | +0.001 | no |
| Direct-Mamba | damage | 0.775 | 0.767 | −0.008 | no |

**Not one comparison separates**, and the intervals are wide: TC-short death is +0.013 with
a 95% interval of [−0.040, +0.056]. That is uncertainty, not demonstrated equivalence. Once
measured fairly, the one-step generated successor shows **no advantage** over a trained
root+action predictor on these probes, in any arm; it does not show that it carries nothing.

Nor does better global risk decoding imply better action selection. After refitting,
within-root death-ranking AUC is 0.813 / 0.818 for the TC arms against 0.847 for action-only,
over just 36 informative roots.

## Completing the readout checks: the artifact is confined to outcomes

Extended to the Mamba-state readouts and to continuous targets, on the same rows.

**Mamba state** (mlp, c4, outcomes). The joint carries the same artifact as `z`, and
refitting recovers it. `h` needs no refit -- the memory panel already fits it on TRAIN `h`,
which is why its observed and generated scores are identical -- so it is a control, not a
condition:

| arm | z obs-fit | z gen-fit | joint obs-fit | joint gen-fit | h (native) | root+action |
|---|---|---|---|---|---|---|
| TC consecutive | 0.6375 | 0.7549 | 0.6468 | 0.7498 | 0.7538 | 0.7882 |
| TC strided | 0.6098 | 0.7462 | 0.6064 | 0.7451 | 0.7735 | 0.7350 |

(These `h` numbers use this experiment's probe, with actions concatenated; they are **not**
protocol-identical to the memory panel's zero-padded equal-parameter inputs and do not
overturn its `h` comparison.)

**Successor state does not recover, and that is the point.** Binary targets, mlp:

| arm | obs-fit | gen-fit |
|---|---|---|
| TC consecutive | 0.6353 | 0.6293 |
| TC strided | 0.6305 | 0.6268 |
| Direct-Mamba | 0.7518 | **0.7935** |
| Direct-Attention | 0.7509 | **0.7941** |

Continuous targets, mean R² -- starker still:

| arm | obs-fit | gen-fit |
|---|---|---|
| TC consecutive | 0.005 | −0.006 |
| TC strided | −0.144 | −0.220 |
| Direct-Mamba | 0.075 | **0.183** |
| Direct-Attention | 0.089 | **0.173** |

Refitting rescues Direct's state decoding and does nothing for either TC arm. So the
decoder artifact is **confined to outcome decoding**; the state-representation weakness is
upstream of the decoder, and it is specific to LeWM rather than general to the setup.

## What this changes

The generated-condition deficits reported across M03 and the memory supplement are
**substantially a measurement artifact**, and death/damage numbers below 0.5 should have been
read as a decoder failing, not as a world model failing. Conclusions drawn from them need
re-reading, this experiment's own floors included.

What survives is narrower: on these probes the one-step generated successor shows no
advantage over a trained root+action predictor, in any arm including Direct.

A separate weakness is **not** explained by any of this, and is the more important finding
here. Refitting does not rescue LeWM's successor *state* decoding: it stays near 0.63 against
Direct's 0.79. And the gap is already present on **observed** successor `z` -- 0.658 and 0.681
for the TC arms against 0.848 for Direct -- so it is an upstream export/representation
weakness, upstream of any decoder question. This experiment does not isolate its cause among
CLS pooling, the projection, the joint objective or capacity.

## What it does not establish

One dataset, one split, 256 train and 128 dev roots fanned over 17 actions. Refitting on
generated features risks a decoder learning generator artifacts; the root and action floors
bound that, but do not eliminate it. Nothing here evaluates multi-step rollout, and nothing
here is a capability claim: M03's status is unchanged and `m4_authorized` remains false.
