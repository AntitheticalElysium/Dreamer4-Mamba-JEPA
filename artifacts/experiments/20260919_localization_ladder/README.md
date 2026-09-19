# Counterfactual failure localization — executed ladder, 2026-09-19

Execution of [PLAN.md](../20260919_counterfactual_localization/PLAN.md). Frozen features and
published checkpoints only: no encoder, world model or actor was trained. Machine-readable
record: [evidence/diagnosis.json](evidence/diagnosis.json).

Primary metric is **safe-action choice on the 36 DEV roots that offer both a fatal and a safe
action**, three probe seeds. `death` is exactly `successor health <= 0` (100% agreement).

> **Status.** This package establishes three measured **symptoms**. Only the TC CLS→z
> degradation is tightly component-localized. The objective and head were selected on the same
> 36-root DEV panel the headline is reported on, so these results are **exploratory** until the
> phase-4 confirmation panel lands. Section "Withdrawn claims" lists what an earlier draft of
> this README overstated.

## Symptom 1 — the gate's readout objective understated real LeWM states

On **identical real-z rows**, changing only the supervision from six-target BCE to within-root
pair ranking:

| successor-only | BCE | rank |
|---|---:|---:|
| Mamba Raw z, logistic | 7.0/36 | **27.0** |
| Mamba Raw z, mlp128 | 17.7/36 | **29.7** |
| Mamba TC z, mlp128 | 10.7/36 | **21.0** |
| Direct, mlp128 | 34.0/36 | **35.7** |

Capacity barely matters (logistic 27.0 vs mlp512x2 29.7); the objective does. The earlier
conclusion that Raw's real latent *cannot* support safe selection was wrong.

This does not make the BCE numbers meaningless — they test globally calibrated decoding and
transfer, ranking tests within-root ordering. They are different capabilities.

## Symptom 2 — the export path weakens a readable signal

At **matched 192 dimensions**, same readout, same seeds:

| rung | dim | mamba_raw | mamba_tc |
|---|---:|---:|---:|
| z (what the world transitions) | 192 | 29.7 | **21.0** |
| CLS | 192 | 31.3 | 33.0 |
| patch_mean | 192 | 34.7 | 32.7 |
| pooled2_pca192 | 192 | **36.0** | **36.0** |
| pooled4_pca192 | 192 | **36.0** | **36.0** |

- **TC's loss is tightly localized**: CLS 33.0 → z **21.0** (AUC .842 → .562). That implicates
  the TC projector/export path for this signal, and it is the one component-exact result here.
- **Raw declines more gently** across the whole path (36.0 → 31.3 → 29.7).
- **Memorization excluded**: permuting death labels *within* root collapses the same rung to
  8–10/36 (chance 8) while real labels give 56/56 TRAIN and 36/36 DEV.
- **Width**: 192 *post-hoc* dimensions suffice on this panel. This does **not** show that an
  end-to-end trained 192-d latent has adequate capacity.

### This is not evidence for spatial layout

`patch_mean` discards token position and still reaches 34.7 (raw) / 32.7 (tc). Mean pooling
removes explicit spatial arrangement, so these data do **not** support "directional geometry
needs a spatial grid". The supported claim is narrower:

> Final patch-token outputs carry a readable signal that the learned CLS/projector export does
> not preserve equally well.

Those tokens are contextualized ViT outputs; their advantage could be terminal cues, HUD/health
information, redundancy, or other globally distributed features.

## Symptom 3 — generated states add no demonstrated decision skill

| arm | real state | generated | root/action control |
|---|---:|---:|---:|
| Mamba Raw z | 29.7 | **15.3** | 19.0 |
| Mamba TC z | 21.0 | **16.3** | 23.7 |
| Direct | 35.7 | **31.7** | 23.7 |
| frozen u→u | **36.0** | **23.0** | 24.3 |

Memory does not rescue it: Raw `h` ≈ 19.3, `[z,h]` ≈ 21.3 against a history control of 19.7.

Action derangement (same magnitudes, permuted action→effect map): Raw ≈ 0.0 cost, TC 1.3,
`[z,h]` 4.0, **generated u 23.0 → 15.0 (8.0)**, Direct 31.7 → 19.3 (12.4). So the LeWM
predictors are **not action-blind** — u→u carries appreciable action correspondence — but not
enough to beat the current-state/action control.

## The u→u negative result

`world_u_u` is a **diagnostic for Raw/TC, not a canonical architecture**. Observed u is perfectly
readable (36/36); generated u does not beat its control (23.0 vs 24.3). So replacing `z` with
this existing pooled/PCA export does not by itself solve the gate, and another bare `u` export
experiment is not presently justified.

It does not rule out every patch-derived world: frozen TC encoder, one pooled representation,
MSE-only objective, one seed, no jointly adapted encoder, no propagated token grid. (Unsealed
diagnostic: patched loader, primary-only, both arms share one frozen consecutive-TC encoder.)

## Withdrawn claims

An earlier draft of this README asserted each of these. They are withdrawn:

| claim | why it fails |
|---|---|
| "Spatial layout is the missing representation" | `patch_mean` is position-free and nearly solves the panel |
| "The sequence mixer is exonerated" | Transformer arms were never run through this rank/dynamics/derangement ladder |
| "Health is not the explanation" | Direct's poor *scalar* R² does not exclude a sharp dead/alive boundary; binary/HUD controls not run |
| "The predictor is not action-conditional" | it is architecturally conditioned; generated u loses 8.0 to derangement |
| "192 dimensions rule out capacity" | only *post-hoc* width, on this panel |
| "Three separable causes" | better described as three symptoms |
| "Train Mamba on the spatial export" | the u→u world already ran that and its generated states fail |

## Controls

| control | mean /36 |
|---|---:|
| uniform over 17 actions | 8.0 |
| best TRAIN constant action (DOWN) | 22.0 |
| fitted action-only | 21.0 |
| mamba_raw root+action / history+action | 19.0 / 19.7 |
| mamba_tc root+action / history+action | 23.7 / 23.3 |
| direct root+action | 23.7 |
| shuffled successor | 9.7–13.0 |

## Known package limitations

- Probe logits and per-root predictions were not retained by phases 1–3, preventing independent
  paired bootstrap. Phase 4 retains them.
- Cached stages resume on file existence alone; phase 4 adds input hashing.
- Patch/pooled caches are **float16**; z and cls are float32.
- `choice_by_true_health` originally passed `health` to an argmin, selecting the *most* fatal
  fork and scoring 0/36. Fixed to `-health`; the inverted value is retained as an anti-oracle.

## Reproduction

```
python extract.py            # taps for roots + 17 successors, parity-checked against M03
python phase1.py             # anchors, timing, 108-fit ladder, health decodability
python phase1_controls.py    # analytic + fitted controls
python phase2.py             # dimension-matched rungs, TRAIN-fitted PCA (bases saved)
python memorization.py       # within-root label permutation on the winning rung
python phase3.py             # four-condition dynamics table, derangement, affine alignment
python phase3b.py            # full state interface (h, [z,h]) and the z->z / u->u worlds
python phase4.py             # transformer ladder, derangement distribution, health controls,
                             # u-world memory, paired intervals, confirmation panel
python synthesize.py         # evidence/diagnosis.json (refuses if inputs are missing)
```

Contract and hashes: [contract.json](contract.json). `m4_authorized: false`; no gate output was
changed and `d4mj/` was not modified.
