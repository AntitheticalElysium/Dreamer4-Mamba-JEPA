# Counterfactual failure localization audit — 2026-09-19

Recommendation: run the real-successor ladder, but treat it as an observed-state
readout diagnostic rather than a representation ceiling. Establish adequate
readouts and observation/context controls, then compare real and generated states.
Use the result to select one matched training intervention. The current evidence
does not establish that CLS has destroyed the relevant information or that
dynamics is innocent.

The executable work here reads published features/predictions and fits a small
CPU readout screen. It trains no encoders, world models, or actors. The forwardable
execution contract is [PLAN.md](PLAN.md).

## Independently reproduced results

[verify_cached_readouts.py](verify_cached_readouts.py) opens SQLite in read-only
mode, validates feature-pointer manifests and cached payload checksums, and
matches probe training/evaluation inputs by exact tensor-byte hashes. It does not
refit the probes. Results: [cached_readout_audit.json](cached_readout_audit.json).
The original M03 primary panel has 256 TRAIN / 128 DEV independent episode roots,
with 56 TRAIN / 36 DEV roots containing both fatal and safe actions. Both backend
runs use the same sidecar hash `ead287f0…`.

| Arm | Real successor: safe choices, published MLP | Real successor: all-action within-root AUC | Real successor: movement-only within-root AUC | Generated successor: safe choices, native-fit MLP |
|---|---:|---:|---:|---:|
| Mamba Raw | 14/36 | 0.756 | 0.567 | 22/36 |
| Mamba TC | 8/36 | 0.723 | 0.486 | 21/36 |
| Transformer Raw | 14/36 | 0.778 | 0.475 | 17/36 |
| Transformer TC | 6/36 | 0.534 | 0.456 | 20/36 |
| Direct-Mamba | 36/36 | 0.958 | 0.975 | 33/36 |

The generated-choice column is read from the separate published
`20260918_action_selection/evidence/{mamba,transformer}.json` reports. Its Direct
paired advantage over its own root+action control is +11/36 = +0.306,
95% episode-cluster bootstrap interval [0.125, 0.487]. The control chooses safely
on 22/36 roots. This corrects the earlier global-AUC-based claim that no world
model adds skill. Direct's demonstrated advantage is specifically one-step safe
choice on this panel; its reward choice does not separate from that control.

Uniform choice among all 17 actions succeeds at 0.222 here, rather than 0.5.
Uniform choice among the four movement actions succeeds at 0.569. Restricting the
published readout to movement raises Raw safe selection to 26/36, but its
directional AUC remains 0.567. Restricted-action results are diagnostic and do
not replace the original 17-action score or establish a paired improvement over
a restricted-action prior.

Raw's published MLP chooses SLEEP on 24/36 opportunity roots, TC on 30/36;
SLEEP is safe on only 6/36. None of the published real-successor argmin results
has an exact minimum-score tie. All 36 roots also have variation among movement
actions. In DEV, 34/36 opportunity roots come from the terminal-tail stratum,
34/36 have root health 1 or 2, and 0/36 have `front_lava` true. This does not
prove that no lava is nearby, but the supplied account of directional lava loss
was not established by this panel's results.

## The readout screen conducted here

[action_input_screen.py](action_input_screen.py) holds encoders/features frozen
and refits both successor-only and successor-plus-action readouts on CPU using
the default M03 recipe: 200 AdamW steps, hidden width 128, one fixed seed,
TRAIN-only standardization, and the six original binary outcome targets.
Results: [action_input_screen.json](action_input_screen.json).

| Arm, MLP | Successor only: safe choice / within-root AUC | Successor + action: safe choice / within-root AUC |
|---|---:|---:|
| Mamba Raw | 8/36 / 0.493 | 12/36 / 0.742 |
| Mamba TC | 7/36 / 0.496 | 7/36 / 0.721 |
| Direct-Mamba | 36/36 / 0.950 | 36/36 / 0.953 |

Removing the explicit action input alone did not rescue the deficit under this
small screen. Much of LeWM's all-action ranking ability can be supplied by action
categories. This remains a one-seed, fixed-capacity, short-optimization result,
not an information-absence theorem. CPU minibatch RNG differs from the original
CUDA run, so compare the two refitted CPU conditions to each other; they are not
bit-exact replications of the published GPU fits.

## Corrections to the attached interpretation

1. **The finite real-successor probe is not a ceiling.** Real successor encoding
   plus a trained decoder bypasses learned dynamics and reveals an upstream
   readout problem. It does not distinguish irrecoverable loss from a weak decoder,
   optimization, training objective, limited labels, or missing observation context.
   Generated Raw selection actually exceeds this particular real-probe result,
   which already prevents treating the latter as a hard numerical upper bound.

2. **Direct's observed-state comparison includes temporal encoding.** In
   `d4mj/m03/gate.py::_encode_legacy`, each real successor is appended to the
   native 64-frame prefix before Direct encoding. In `_encode_lewm`, successors
   are encoded as individual one-frame images. Direct therefore differs in
   encoder context as well as spatial structure, dimensionality, and training
   objective. Its world dynamics is bypassed in the real-successor test, but its
   temporal encoder is not. A shortened Direct input is an ablation, not a fair
   replacement for its native result.

3. **Linear and MLP do not settle capacity.** Real Raw choices change 8/36 to
   14/36; Direct changes 31/36 to 36/36. A linear/MLP tie on global generated AUC
   says nothing conclusive about adequate real-successor directional readout.

4. **Magnitude is not action fidelity.** A generated/observed action-spread ratio
   near 1 can survive permuting action assignments. It does not establish the
   correct direction or semantic effect of each action, nor rule out dynamics
   failure. Likewise poor decoder cross-back does not exclude a rotated space;
   a decoder is not invariant to an unapplied rotation.

5. **The old `u` experiment cannot settle this outcome question.** Its `u` is
   TRAIN-fitted PCA-192 of the 4×4 pooled patch grid, a sibling of CLS, not bare
   CLS. Fresh z→z and u→u worlds shared a frozen consecutive-TC encoder and
   MSE-only training. They improved generic state decoding but did not improve
   the reported transfer-based safe-action metric. This did not measure native-fit
   death safe selection from real `u` versus generated `u`. A representation can
   improve inventory/prerequisite readouts while remaining poor for this safety
   distinction; conversely a good real `u` can still be predicted poorly.

6. **Outcome labels have different information requirements.** `death` is
   successor health≤0 or lava at the successor player's location. `damage` and
   inventory change compare root and successor. `tile_changed` compares the
   complete simulator map, including potentially invisible regions. An isolated
   successor frame is not generally sufficient for every transition label.
   For transition targets, provide matched root+successor inputs, and separate
   visible changes from privileged simulator-wide events.

7. **Coverage is now separately demonstrated.** The declared
   `20260918_coverage_panel` replays the 129 audited addresses and measures all
   48 arm×rare-label cells. It reports 0/48 separated global-AUC gains over each
   arm's root+action control. Those persistent state-predicate results do not
   answer within-root action choice. The canonical completed gate still records
   insufficient coverage; the stress panel changes neither that report nor M4
   authority. Count informative action-opportunity episodes separately.

The full-simulator flattened prestate learner also remains a learned control,
not an oracle information ceiling. The new plan separates exact simulator label
sanity, successor-state structured controls, and root-state dynamics controls.
The attachment's proposed explanation of historical actor failure is not used
as a causal premise for the present world-model diagnosis.

## Research grounding

Spatial features are a justified candidate, not a proven repair: the
[DINO-WM paper, §4.4](https://arxiv.org/html/2411.04983v2#S4.SS4) reports better
planning with DINO patches than CLS on several spatial tasks. Its pretrained
representations and continuous-control environments differ from this project.
The [TC-LeWM paper, §5.1 and Appendix A.1](https://arxiv.org/html/2607.26924v1)
uses both CLS and pooled patches in its downstream BC policy; that evaluation
does not demonstrate counterfactual planning from CLS-only predicted states.

The [JEPA-WMs study, Appendix E.3](https://arxiv.org/html/2512.24497v1#A5.SS3)
finds that validation prediction metrics and planning success are imperfectly
aligned and explains observation, objective, planner, and action-distribution
effects. This supports checking actual choices and multiple horizons rather
than selecting a repair from latent loss alone.

[Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/) show why probe
capacity and memorization controls matter for interpreting learned readouts.
[Alain and Bengio](https://arxiv.org/abs/1610.01644) motivate intermediate-layer
probes as diagnostics. Their language/classification results motivate the audit
methodology, not a specific Craftax failure mechanism.

## Scope and reproducibility

`evidence_manifest.json` pins consulted local reports, source files, attachments,
and new outputs. It is an audit manifest, not a regenerated M03 source seal.
Original report files were not rewritten. Readout screens are exploratory;
architecture, objective, and long-horizon causal claims require the controlled
experiments in the plan. No M4 authorization follows from this audit.
