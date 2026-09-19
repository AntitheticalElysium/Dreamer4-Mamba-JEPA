# Counterfactual failure localization — executed ladder, 2026-09-19

Execution of [PLAN.md](../20260919_counterfactual_localization/PLAN.md). Frozen features and
published checkpoints only: no encoder, world model or actor was trained. Machine-readable
record: [evidence/diagnosis.json](evidence/diagnosis.json).

Primary metric is **safe-action choice on the 36 DEV roots that offer both a fatal and a safe
action**, three probe seeds. `death` is exactly `successor health <= 0` (100% agreement).

> **Status.** Three measured **symptoms**. Symptoms 1 and 3 are now **confirmed on an untouched
> 100-root historical panel** with paired episode-cluster intervals (phase 4). Symptom 2 is
> confirmed on DEV but could not be scored on the confirmation panel, which publishes z only.
> Section "Withdrawn claims" lists what earlier drafts overstated — including several claims
> phase 4 forced me to correct again.

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

Action derangement was measured properly in phase 4 over **20 draws**; the single-draw numbers
that first appeared here were noise. See "Derangement distribution" below — the corrected
ordering is mamba_raw −3.9 (no action information at all) < mamba_tc 0.9 < u→u 2.8 <
transformer_raw 7.2 < direct 16.1.

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
| "The sequence mixer is exonerated" | **REFUTED in phase 4**: Transformer action-conditioning cost 7.2 vs Mamba −3.9 |
| "Health is not the explanation" | **REFUTED in phase 4**: HUD tokens alone give dead/alive AUC 1.0000 and 36/36 |
| "The predictor is not action-conditional" | arm-dependent: Mamba −3.9 (none), Transformer 7.2, u→u 2.8 over 20 draws |
| "192 dimensions rule out capacity" | only *post-hoc* width, on this panel |
| "Three separable causes" | better described as three symptoms |
| "Train Mamba on the spatial export" | the u→u world already ran that and its generated states fail |

## Phase 4 — confirmation and self-correction

### Untouched confirmation panel (100 opportunity roots, 68 historical seeds, fit on unchanged TRAIN)

| condition | /100 |
|---|---:|
| mamba_raw real z, **rank** | **93.0** |
| mamba_raw real z, BCE | 74.7 |
| mamba_raw generated z | 51.3 |
| mamba_raw root+action control | 55.7 |

Predeclared contrasts, paired episode-cluster bootstrap, all three seeds:

| contrast | DEV | confirmation panel |
|---|---|---|
| C1 rank > BCE | +0.333, all seeds exclude 0 | +0.16/+0.22/+0.17, **all exclude 0** |
| C2 pooled-PCA > z at matched width | +0.176, all seeds exclude 0 | not scorable (no patch taps in shards) |
| C3 generated > root+action | −0.102, does not exclude 0 | −0.02/−0.06/−0.05, does not exclude 0 |

### Derangement distribution (20 draws, not one)

| arm | intact | deranged | cost |
|---|---:|---:|---:|
| mamba_raw | 14.0 | 17.9 | **−3.9** |
| mamba_tc | 18.0 | 17.1 | 0.9 |
| world_u_u | 23.0 | 20.2 | 2.8 |
| **transformer_raw** | 22.0 | 14.8 | **7.2** |
| direct | 32.0 | 16.0 | **16.1** |

**My single-draw result was noise.** I reported mamba_raw "cost 0.0"; over 20 draws it is −3.9.

**The sequence mixer is not neutral.** The Transformer predictor is substantially more
action-conditional than Mamba's (7.2 vs −3.9) — yet its intact score (22.0) still only matches
its own root+action control (21.3). Action-conditioning improved; decision utility did not.

### Health controls — the audit was right, I was wrong

| mamba_raw rung | dead/alive AUC | safe choice |
|---|---:|---:|
| z | 0.765 | 29.7 |
| CLS | 0.780 | 31.3 |
| patch_mean | 0.898 | 34.7 |
| **hud_tokens_mean** | **1.0000** | **36.0** |
| map_tokens_mean (HUD removed) | 0.833 | 34.7 |
| structured state (16 sim fields) | — | 36.0 |

Health is drawn in pixel rows 49–62 (measured: max r=0.379 at row 50; rows 0–48 mean |r|=0.018),
so patch tokens 63–80 are HUD. **HUD tokens alone are a perfect dead/alive readout and a perfect
36/36.** The patch advantage over CLS is substantially a retained HUD health readout. Map-only
tokens still reach 0.833 / 34.7, so it is not *exclusively* HUD.

**Scope caveat.** 34/36 DEV roots are terminal-tail with root health 1–2, so this decision largely
reduces to "which action takes health to zero". It may not transfer to non-threshold decisions.

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
