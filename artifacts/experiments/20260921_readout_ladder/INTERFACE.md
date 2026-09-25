# Representation-interface comparison: alias-free u→u Mamba vs alias-free z→z Mamba

Predeclared `interface.py` (`9aefa86d`), designed by the reviewing agent (`a2c1aee3`). Sealed block:
seeds 55,000–55,462, collected after the commit and read once. It holds 804 one-step opportunity
roots, 510 of them next to a zombie.

A machine reboot interrupted the run midway (arm U was in phase 2; collection was at 716 of 800
opportunities). Both were relaunched unchanged: the collector resumed by counting the files on disk,
and arm U retrained from scratch, reproducing its pre-reboot loss exactly (0.0854 at update 5,000).

Both arms share the frozen Raw H2 encoder, the canonical LeWMWorld (Mamba), the same windows, the same
loss rule, A/B/C's joint recipe followed by H2's bridge with the alias-free terminal layout, one
training seed, and the same probes. Generation depth came out balanced (4,748 draws at depth 1,
4,585 at depth 2).

## Declared readings

| rule | result | reading |
|---|---|---|
| P0: patch tokens beat DOWN on zombie roots | +0.099 [+0.034, +0.161] | ok |
| interface: generated_U − generated_Z, zombie | **+0.136 [+0.087, +0.187]** | **patch_interface_better** |
| retention U: generated_U − root_U, zombie | −0.006 [−0.042, +0.032] | **retains** |
| retention Z: generated_Z − root_Z, zombie | −0.049 [−0.083, −0.015] | world_degrades |
| world U: generated_U vs DOWN / DOWN zombie / actions_only | +0.107\* / +0.114\* / +0.071\* | **u_world_state_passes** |
| trained U vs DOWN / DOWN zombie / actions_only | +0.038 / −0.006 / +0.002 (all not resolved) | trained_system_fails (as predicted) |
| trained Z vs DOWN / DOWN zombie / actions_only | −0.007 / −0.078 / −0.043 (not resolved) | trained_system_fails (as predicted) |

## Expected safe choice (opportunity roots)

| head reads | overall | zombie | night | lava | stay kills, a move survives |
|---|---|---|---|---|---|
| DOWN | 0.586 | 0.496 | 0.551 | 0.761 | 0.470 |
| actions_only | 0.623 | 0.541 | 0.601 | 0.715 | 0.509 |
| root patch tokens | 0.708 | 0.596 | 0.644 | 0.928 | 0.579 |
| root_U | 0.715 | 0.616 | 0.625 | 0.957 | 0.591 |
| **generated_U** | **0.693** | **0.610** | **0.647** | 0.899 | **0.585** |
| features_U | 0.677 | 0.576 | 0.587 | 0.952 | 0.540 |
| trained_U (own head, no refit) | 0.624 | 0.490 | 0.563 | 0.899 | 0.430 |
| root_Z | 0.622 | 0.523 | 0.593 | 0.937 | 0.486 |
| generated_Z | 0.576 | 0.474 | 0.511 | 0.928 | 0.429 |
| trained_Z | 0.580 | 0.418 | 0.515 | 0.964 | 0.353 |

Additional contrasts:

- generated_U − root patch tokens: −0.015 [−0.049, +0.019], not resolved.
- trained_U − trained_Z on zombie roots: +0.073 [+0.015, +0.128]\*.
- SLEEP chosen by the trained head: U 192 / 804, Z 292 / 804.

## Fidelity (reported)

| | U | Z | canonical H2 |
|---|---|---|---|
| trained head, within-root death AUC on REAL successors | 0.993 | 0.717 | |
| same head on GENERATED successors | 0.649 | 0.612 | |
| DEV, 4 observed + 1 generated: death vs alive-10 AUC (generated) | 0.822 | 0.719 | 0.51 |
| same, death vs pre-death (generated) | 0.637 | 0.577 | |
| same on the real successor, death vs alive-10 / vs pre-death | 0.998 / 0.998 | 0.875 / 0.873 | |

## What this establishes

- **The patch-derived u→u Mamba world's imagined successor keeps the root's decision information on
  sealed seeds.** Overall 0.693 against a matched root at 0.715, not resolved apart; on zombie roots
  0.610 against 0.616. It beats DOWN and actions_only, and matches the root patch-token readout. The
  CLS z→z world, identical in every other respect, falls below its own root (−0.049\*) and is
  +0.136\* worse on zombies. This is the first world in the campaign whose generated state passes a
  sealed retention test.
- **The scope is a world-state claim, read with fork-supervised probes.** The world was trained
  factually, inside the contract. The probe is a diagnostic.
- **Both trained systems fail, as declared in advance.** U's own head reads real successors at
  within-root AUC 0.993 but generated ones at 0.649. The consequence is present in the generated
  state (the probe finds it), but it is not drawn the way a real consequence is, so a head fitted
  largely on real outcomes does not fire. This is the remaining gap.
- **The alias-free layout moved factual death discrimination** from canonical H2's 0.51 to 0.72 (z) and
  0.82 (u). This is not an isolated test of the layout: the worlds differ from canonical H2 in
  training data and schedule too.
- **Not established:**
  - more than one training seed;
  - two steps or H16;
  - a working trained decision head.
- **Contract:** u as the world's state is a declared deviation from TC-07/TC-19. The PCA was fitted on
  TRAIN only.

## Next, not run

The world-state question is answered on sealed seeds for u→u. The open problem is the trained head's
fidelity gap on generated states, which has to be fixed inside the factual contract:

- **The generated consequence's appearance:** does the generated health component move like the real
  one?
- **How the head sees generated states in training:** it currently gets 0.25 generated-suffix weight
  plus the paired terminal term.

Replication with a second world seed should precede any H16 step.
