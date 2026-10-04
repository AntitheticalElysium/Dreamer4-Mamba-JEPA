# Independent challenge to the H2 input-bound reading (2026-09-25)

**Status: exploratory.** The 54k judgment block had already been inspected. These runs freeze the canonical Raw H2 checkpoint and do not retrain the encoder or world. Scripts were written before their respective runs; each JSON stores the SHA256 of its exact script, checkpoint, and judgment manifest. FIT-train roots fit heads, FIT-dev or an inner FIT split selects them, and judgment uses roots from seeds 54,000–54,395. Run with `TRITON_F32_DEFAULT=ieee`.

## Matched decision-head comparison

`matched.py` repeats the agent's z4-plus-action 17-output readout and adds a root-input control with **the same shared scalar branch-head family, all-action ranking loss, 3 seeds, 3000 steps, and judgment roots** as the generated-successor readout. Each root branch gets four observed z vectors, the previous three actions, and its candidate action. The agent's earlier root-versus-generated contrast had compared vector and branch head forms.

| Opportunity stratum | n | DOWN | root branch | generated H2 feature | root minus generated, paired seed-cluster CI |
|---|---:|---:|---:|---:|---:|
| All | 804 | .613 | .634 | .610 | +.024 [-.017, +.065] |
| Zombie adjacent | 479 | .533 | .561 | .453 | **+.108 [.055, .157]** |
| Zombie only | 467 | .544 | .557 | .440 | **+.117 [.069, .167]** |
| Lava adjacent | 126 | .675 | .839 | .971 | **-.132 [-.236, -.039]** |
| Lava only | 114 | .737 | .854 | .974 | **-.120 [-.233, -.026]** |

The repeated vector root head scored .554 on zombie roots versus .561 for the matched branch root head; their difference is unresolved. Head form does not explain the large root-to-generated zombie drop. The disjoint hazard strata establish the *performance asymmetry* without attributing a mechanistic resource trade-off to training. Root branch exceeds DOWN by only +.028 [-.031, +.089] on zombies. Thus the current probe has weak headroom there, but it does not establish an impossibility bound. The agent's z4-plus-actions interval was +.022 [-.034, +.083] versus DOWN, which does not rule out a meaningful gain.

## Visible-hazard information in frozen encoder outputs

`hazard_presence.py` and `lava_presence.py` fit 3-seed balanced-BCE probes on the same frozen Raw H2 encoder. Labels are current-frame visible-state adjacency to a zombie or lava, not future death. FIT has 7,085 roots, with an inner 974-root selection set; 54k has 4,072 judgment roots. z1 and z4 use MLPs; patch tokens use attention. `linear_presence.py` independently fits a standardized ridge-linear control, selecting regularization on the inner FIT split. All judgment AUCs are on the previously inspected 54k roots.

| Visible current-frame label | z1 MLP, 3 seeds | z4 MLP, 3 seeds | z1 linear | z4 linear | patch-token attention, 3 seeds |
|---|---|---:|---:|---:|---:|
| Zombie adjacency | .561–.601 | .545–.586 | .612 | .603 | .99999–1.00000 |
| Lava adjacency | .938–.940 | .925–.930 | .949 | .959 | .998–1.000 |

This is a controlled reason to suspect **selective encoding**: zombie adjacency is common in FIT (56%) yet barely decodable from z; lava adjacency is rare (2.4%) yet well decodable. It cannot prove z contains zero zombie information, nor that visible adjacency alone suffices to choose a safe action. The patch-token arm shows that this encoder's spatial output retains the visual cue under the same data split.

## Interpretation and next decision

The agent's headline "alias-free CLS-H2 cannot pass by construction" is stronger than the evidence. A finite selected probe cannot certify a Bayes input ceiling, and a nonsignificant improvement over DOWN is not evidence of a small upper bound. However, the converging sealed 51k/53k choice results, this 54k physical-label test, and the matched 54k root-to-generated contrast make a standalone alias-free CLS retrain an unattractive *repair* bet. They support testing patch-derived input next. The current H2 has at least three separable defects: a depth/terminal-label shortcut, weak CLS encoding of moving hazards, and further zombie-choice degradation across the transition. Earlier independent frozen-head refitting improved zombie choice by +.152 yet left it below DOWN, so head supervision also contributes materially.

A clean retrain should compare alias-free CLS-H2 to alias-free patch-derived compact-input H2 under identical frozen encoder, corpus, windows, world budget, head protocol, and judgment set. The patch arm should fit its pooling/compression on TRAIN only. Judge root and generated hazard fidelity, one-step within-root safe choice, zombie/night/lava and disjoint strata, and real-versus-generated readout. This compares the **representation interface** while holding the known terminal-layout repair fixed: a u-to-u arm changes both the world input and its prediction target, so the contrast cannot isolate input alone. Old u-to-u and full-spatial results motivate the arm, but differ in encoder/loss/world and do not prove this variant will work. Predeclare on a new judgment block rather than reusing 54k.

## Reproduce

From the repository root:

```sh
TRITON_F32_DEFAULT=ieee .venv/bin/python artifacts/experiments/20260925_h2_bound_challenge/matched.py
TRITON_F32_DEFAULT=ieee .venv/bin/python artifacts/experiments/20260925_h2_bound_challenge/hazard_presence.py
TRITON_F32_DEFAULT=ieee .venv/bin/python artifacts/experiments/20260925_h2_bound_challenge/lava_presence.py
TRITON_F32_DEFAULT=ieee .venv/bin/python artifacts/experiments/20260925_h2_bound_challenge/linear_presence.py
```
