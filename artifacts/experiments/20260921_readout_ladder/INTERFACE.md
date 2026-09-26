# Representation-interface comparison: alias-free u→u Mamba vs alias-free z→z Mamba

> **Superseded in part, 2026-09-26: the world-state pass did NOT replicate.** On a fresh block (56k),
> with a second world seed AND the seed-1 worlds, the declared replication reading is
> `does_not_replicate`. The U-over-Z interface advantage and the action-specific content of U's
> generated state replicate. The 55k "u_world_state_passes" and "no resolved loss against the root"
> do not. See "Replication" at the end.

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
| retention U: generated_U − root_U, zombie | −0.006 [−0.042, +0.032] | "retains" = **no resolved loss**; non-inferiority was not tested and a loss of ~4 points remains compatible |
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

> **Tempered after review (2026-09-26).** The generated-state probe also received an explicit
> candidate-action token, and U's root + action scores 0.715 against generated 0.693. The world-state
> pass therefore does not by itself show that the transition writes the action's consequence; the probe
> could be reading preserved root context plus the token. Tested in `interface_shuffle.py`.

## What this establishes

- **The patch-derived u→u Mamba world's imagined successor shows no resolved loss of the root's decision
  information on sealed seeds** (non-inferiority not tested; up to ~4 points of loss remain compatible). Overall 0.693 against a matched root at 0.715, not resolved apart; on zombie roots
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

## Shuffle control (`interface_shuffle.py`, `95a0d264`; post hoc on the same sealed block)

Readings: **U `state_alone_carries_consequence`**, **Z `root_plus_action_shortcut`**.

Probes were refitted as in `interface.py`. "Permuted" means the 17 generated states were shuffled among
the actions within each root (5 seeded permutations, tokens kept fixed); "mean" means every state was
replaced by its within-root mean.

| probe | intact | permuted | mean | intact − permuted (zombie) |
|---|---|---|---|---|
| U, state + token | 0.693 | 0.531 | 0.651 | +0.162\* (+0.280\*) |
| **U, state alone** | **0.684** | 0.475 | 0.442 | **+0.209\* (+0.350\*)** |
| Z, state + token | 0.576 | 0.580 | 0.589 | −0.004 (−0.009) |
| Z, state alone | 0.567 | 0.479 | 0.442 | +0.089\* (+0.156\*) |
| DOWN | 0.586 | | | |

- **U's generated state writes the action's consequence.**
  - A probe that never sees the action ranks actions from the imagined state alone: DOWN +0.098\*,
    zombie +0.089\*.
  - That ranking collapses when successors are shuffled within a root (−0.209\*).
  - It equals the token probe (−0.009, not resolved).
  - The token probe also falls to 0.531 under permutation, below DOWN: it relies on the action-specific
    state, not on the token.
- **Z's token probe is a root-plus-action shortcut.** Shuffling costs it nothing. Z's state alone does
  carry action-specific structure (+0.089\* over permuted), but it ranks zombie roots below DOWN
  (−0.104\*): the direction is wrong.
- **Limit:** the mean-replaced state is off-distribution, so "token + mean = 0.651" is not a clean
  measure of how much root-plus-token alone is worth.

## Replication (`replicate.py`, `ff4c83ae`): declared reading `does_not_replicate` (both seeds)

The second world seed (init 8, other batch orders) went through the unchanged recipe. Seed 2 and the
seed-1 worlds were each scored once on seeds 56,000–56,423: 801 opportunity roots, 528 of them next
to a zombie, collected after the commit.

| on the 56k block | seed 1 | seed 2 | 55k (seed 1, for reference) |
|---|---|---|---|
| P0: patch tokens − DOWN, zombie | +0.120\* | +0.120\* | +0.099\* |
| **interface: generated_U − generated_Z, zombie** | **+0.069\*** | **+0.076\*** | +0.136\* |
| generated_U − generated_Z, overall | +0.041\* | +0.050\* | +0.117\* |
| **retention: generated_U − root_U, zombie** | **−0.046\*** | **−0.055\*** | −0.006 |
| generated_U − DOWN, overall / zombie | +0.081\* / +0.068\* | +0.079\* / +0.058\* | +0.107\* / +0.114\* |
| **generated_U − actions_only, overall** | **+0.035 [−0.003, +0.071]** | **+0.033 [−0.003, +0.070]** | +0.071\* |
| token-free probe on U's state − DOWN, overall / zombie | +0.058\* / +0.033 | +0.043 / +0.011 | +0.098\* / +0.089\* |
| token-free probe: intact − within-root permuted, zombie | +0.334\* | +0.309\* | +0.350\* |
| trained_U − DOWN, overall / zombie | +0.065\* / +0.007 | +0.066\* / +0.021 | +0.038 / −0.006 |
| trained_U − trained_Z, zombie | +0.149\* | +0.081\* | +0.073\* |

Expected safe choice on 56k:

| | root patch tokens | root_U | generated_U | trained_U | generated_Z | DOWN | actions_only |
|---|---|---|---|---|---|---|---|
| seed 1 | 0.743 | 0.708 | 0.672 | 0.656 | 0.631 | 0.591 | 0.637 |
| seed 2 | 0.743 | 0.708 | 0.670 | 0.658 | 0.620 | 0.591 | 0.637 |

### What holds, and what does not

- **Replicates in every seed and block (3 of 3):**
  - The patch interface beats CLS in the imagined state (+0.07 to +0.14 on zombie roots).
  - U's imagined state beats DOWN.
  - Probes use U's action-specific change: shuffling successors within a root costs 0.31–0.35 on
    zombie roots.
  - U's trained system beats Z's on zombie roots (+0.07 to +0.15).
- **Does not replicate:**
  - **Retention.** On 56k U's transition loses a resolved ~5 points on zombie roots against its own
    root (−0.046\*, −0.055\*). The 55k "no resolved loss" was the optimistic block; the reviewer's
    warning that about 4 points of loss remained compatible was right.
  - **The margin over actions_only** misses resolution in both seeds (+0.035, +0.033).
  - **The token-free probe** no longer beats DOWN on zombie roots.
- **The seeds agree closely** (generated_U 0.672 vs 0.670; root and controls identical); the
  difference from 55k is block-to-block variation. **Where the information goes:**
  - U's root compression costs little against the full patch tokens (−0.035 overall, −0.006 zombie).
  - The transition then loses another ~0.04–0.06: generated_U − patch tokens is −0.071\* and −0.073\*.
- **The trained U head now beats DOWN overall** (+0.065\*, +0.066\*) but not on zombie roots or
  against actions_only. The real/generated gap is unchanged (within-root AUC 0.996 real against 0.65
  generated).

### Standing conclusion

- **A better interface, not a repaired world.** The patch-derived state is robustly better than CLS
  for a Mamba world trained on logged transitions, and its imagined successor carries action-specific
  safety information.
- **The transition still loses part of the zombie consequence** (~5 points), and the gain over
  actions-only is not established.
- **The trained head's real-to-generated transfer is unsolved.**
- **H16 and any move into d4mj stay gated.**

## Where the transition loses it (`transition_diag.py`, `df25ceed`; post hoc, FIT fit / 56k judge)

DIAGNOSE.md's fatal-direction test, applied to all four worlds. The direction is fitted on real FIT
successors.

| world | real AUC along fatal dir. | generated AUC (all / zombie) | error ratio (dir. / all) | generated magnitude along dir. | correlation with real | share of effect energy along dir. |
|---|---|---|---|---|---|---|
| U seed 1 | 0.9997 | 0.636 / 0.771 | 5.9 | 0.85 | 0.18 | 0.11% |
| U seed 2 | 0.9997 | 0.614 / 0.782 | 4.1 | 0.64 | 0.22 | 0.11% |
| Z seed 1 | 0.972 | 0.517 / 0.521 | 6.2 | 0.74 | 0.18 | 0.07% |
| Z seed 2 | 0.972 | 0.504 / 0.523 | 6.1 | 0.69 | 0.20 | 0.07% |
| old u→u world (DIAGNOSE.md) | 0.9975 | 0.495 | 25 | 0.39 | 0.02 | 0.04% |

All four read **misaligned, not shrunk**: the world moves along the fatal direction at near-real
magnitude, but mostly uncorrelated with what really happens. The damage-direction readings are void
(real AUC 0.72–0.82).

**U's error is spectral.** Grouped by the variance rank of U's PCA coordinates (seed 1; seed 2 alike):

| variance ranks | share of the action-effect energy | the transition's effect R² | share of the fatal direction |
|---|---|---|---|
| 0–10 | 73.7% | 0.83 | 0.0% |
| 10–30 | 15.1% | 0.66 | 0.5% |
| 30–60 | 6.3% | 0.55 | 5.5% |
| 60–100 | 2.9% | 0.37 | 42.6% |
| 100–192 | 2.2% | 0.23 | 51.4% |

**94% of the fatal direction sits in components that carry 5% of the effect energy**, and those are
exactly where the transition predicts worst. The pooled grid dilutes the health tile into a
low-variance tail. The world learns the dominant movement and scroll components well and the tail
poorly, and the loss weighting (var^−½) raises the direction's share of the loss only from 0.11% to
0.56%.

Z is different. Its effect R² is uniform across its isotropic coordinates (~0.78–0.81), yet its
generated AUC is at chance: there the consequence was barely encoded to begin with.

The longer-training test (`longer.py`, running) asks directly whether the tail catches up with more
training.
