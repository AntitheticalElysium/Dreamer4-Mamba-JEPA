# Do the 2×2 worlds move the rows the gate actually fails on?

Status: **complete**, 2026-09-17. **No.** The `u` successor-state advantage does not reach
action selection, which is the row Direct wins by a wide margin. Runner:
[`gate_facing.py`](gate_facing.py), evidence in
[`evidence/gate_facing.json`](evidence/gate_facing.json).

Produced by the gate's **own** `_outcome_report`, so these sit on one scale with the sealed
panel. Direct features are reused from the published run, not recomputed. Generated
successors are produced at three memory lengths, since the worlds trained on three
action-pairs: `standard` (4 pairs), `trained_length` (3), `reset` (1).

## Within-root action choice — the row that matters for control

`fatal_safe_ranking`, 36 opportunity roots, uniform baseline 0.2222:

| condition | arm | selected | lift | 95% interval |
|---|---|---|---|---|
| trained_length | `z→z` | 0.1944 | −0.028 | [0.069, 0.341] |
| trained_length | **`u→u`** | **0.3333** | +0.111 | **[0.171, 0.500]** |
| trained_length | **Direct-Mamba** | **0.8611** | **+0.639** | **[0.742, 0.970]** |
| standard | `z→z` | 0.2778 | +0.056 | [0.128, 0.429] |
| standard | `u→u` | 0.3889 | +0.167 | [0.222, 0.552] |
| reset | `u→u` | 0.3611 | +0.139 | [0.212, 0.531] |

**`u→u` is better than `z→z` at every memory length, and its interval still contains the
uniform baseline.** With 36 roots it does not significantly beat picking at random. `z→z` is
at or below uniform.

**Direct-Mamba reaches 0.8611**, an interval nowhere near either arm. Whatever `u` fixed
upstream, this is not it.

`reward_ranking` separates nothing: `z→z` and `u→u` both 0.3077, Direct 0.2308, uniform
0.0679, all intervals wide and overlapping, identical across memory conditions.

## Coarse outcomes — neither arm beats the action floor

mlp, `trained_length`:

| arm | generated | observed | root+action | action-only | shuffled |
|---|---|---|---|---|---|
| `z→z` | 0.6532 | 0.7409 | **0.7882** | 0.6825 | 0.6781 |
| `u→u` | 0.6782 | **0.7741** | 0.7043 | 0.6825 | 0.5954 |
| Direct-Mamba | 0.5686 | 0.7084 | 0.6972 | 0.6825 | 0.6160 |

Neither generated column clears action-only at 0.6825. The generated figures use the gate's
observed-fit transfer decoder, so they carry the artifact established earlier and understate
all arms equally — but that does not change the ordering against a floor that shares it.

**A tension worth recording.** The root-plus-action floor is computed in each arm's own
space, and `z`'s root predicts coarse outcomes *better* than `u`'s (0.7882 against 0.7043) —
the reverse of the successor-state result, where `u` led. The two target families disagree
about which representation is richer, so "`u` is the better state" is target-dependent, not
general.

## What this settles

The one-step successor-state win was real and it does **not** propagate to the gate-facing
rows. On action ranking `u→u` improves on `z→z` without clearing uniform; on coarse outcomes
neither arm clears action-only; and Direct's fatal-safe advantage is untouched by anything
tested here.

Memory length is not the explanation: `u→u` leads `z→z` at all three lengths, and the
ranking numbers barely move between them.

## What it does not establish

36 opportunity roots is a thin instrument, and every interval here is wide; this rules out a
large effect, not a small one. Direct receives a different input protocol and its numbers are
reused rather than re-derived under these conditions. The historical stress panels
(`exact961`, `policy104`, `hazard5402`, `legacy751`) are **not** covered — these are primary
roots only. Nothing here is a capability verdict: coverage remains 13/25 static and 18/25
successor binary targets, M03 reports measurability rather than a numeric threshold, and
`m4_authorized` stays false.

Both arms' encoder is the TC consecutive checkpoint (`variant: tc`), and the fresh worlds
train on MSE alone, so none of this is a Raw-versus-TC objective comparison.
