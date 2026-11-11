# P1-MultiStep-Mamba Results: Multi-Step Mamba Belief Self-Consistency

**Run:** 25.6M frames (15,625 updates, 15h 18m). NVIDIA GeForce RTX 3060 Laptop GPU (2.56 GiB peak VRAM).
**Selected Checkpoint:** `ckpt_10.pt` (SHA256: `c9c00e53565424ac3ed0f00beeb60a26f1ed40b2d4e381d8d197c405a2cd1e70`).
**Readout Completed:** S1 entity readers, consequence determinant curves, crafting strata, actor interface rollouts,
4-step own-output recursion, natural thirst alias on DEV Episode 376 ($t=654$ vs $t=658$), and belief probes.
Matched three-arm values (F4 / HUD-Change / MultiStep) are from `../20261111_two_arm_audit/REVIEW.md`, which uses the
same populations, sampling IDs and readers for all three checkpoints.

---

## 1. This run is not an isolated test of the auxiliary

Two implementation defects confound the comparison with HUD-Change, so the idea is neither confirmed nor refuted:

1. **The split forward pass reset the carry at artificial forks.** `multistep_losses` splits the observed sequence
   right after percept $t$ and passes the suffix's cumulative episode indices unchanged (`seq[:, split_idx:]`).
   Functional Mamba continues the carry only for segment 0, so in every row with an episode start earlier in the
   chunk the suffix began from a zero carry. That suffix feeds the auxiliary's target AND every ordinary loss, and its
   final carry enters the next chunk. Exact stream replay: 33,999 of 243,387 valid auxiliary targets (13.97%) and
   1,786,546 of 25.6M G1 frames (6.98%) were affected. On the trained weights the affected row equals a zero-carry
   continuation (RMS against the uninterrupted pass .304; .004 with the indices rebased).
2. **The initialization differs from HUD-Change.** `MultiStepM0` builds the standard belief and then replaces it with
   a newly constructed `MultiStepBelief`, so 43 belief tensors (and the new mask token) come from a different RNG draw.

The auxiliary as run: one sampled endpoint per microbatch, $H \in \{2, 4, 8, 16\}$, masked intermediate observations,
the detached observed post-percept belief as target, MSE weight 0.1, the starting carry attached; it has no direct
denoiser term. Near the end its median weighted contribution to the total loss was 0.0175, against G1 0.0088
(scalar contributions, not gradients), so the 0.1 coefficient alone does not make it weak.

---

## 2. Belief probes: do the timers appear?

| Timer Category | Target | F4 Baseline | P1-HUD-Change | P1-MultiStep-Mamba | Reading |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Awake timers** | Food Down Due (AP / Share) | 0.037 / 0.01 | 0.038 / 0.01 | 0.042 / 0.054 | absent |
| | Drink Down Due (AP / Share) | 0.045 / 0.01 | 0.047 / 0.03 | 0.048 / 0.036 | absent |
| | Energy Down Due (AP / Share) | 0.033 / 0.09 | 0.038 / 0.16 | 0.036 / 0.139 | absent |
| | Recovery Due (AP / Share) | 0.043 / 0.02 | 0.046 / 0.04 | 0.049 / 0.069 | absent |
| | Need Damage Due (AP / Share) | 0.043 / 0.11 | 0.040 / 0.07 | 0.042 / 0.098 | absent |
| **Counters** | Hunger Awake ($R^2$ in $bp_t$) | $-0.007$ | $-0.005$ | $-0.003$ | absent |
| | Thirst Awake ($R^2$ in $bp_t$) | $0.019$ | $0.016$ | $0.014$ | absent |
| | Fatigue Awake ($R^2$ in $bp_t$) | $0.028$ | $0.045$ | $0.046$ | absent |
| **Visible state duration** | Steps Asleep ($R^2$ in $bp_t$) | $0.766$ | $0.797$ | $0.865$ | present (see below) |
| | Fatigue Asleep ($R^2$ in $bp_t$) | $0.139$ | $0.261$ | $0.301$ | partial |
| | Energy Up Asleep (AP / Share) | 0.155 / 0.96 | 0.160 / 1.03 | 0.167 / 1.136 | weak trace |
| **Zombie readiness / hits** | Zombie Ready Awake (AP / Share)| 0.872 / 0.88 | 0.912 / 0.97 | 0.891 / 0.924 | present |
| | Zombie Hit Awake (AP / Share) | 0.700 / 4.35 | 0.781 / 5.22 | 0.746 / 4.721 | present |
| | Zombie Hit Asleep (AP / Share) | 0.807 / 0.66 | 0.863 / 0.78 | 0.826 / 0.704 | present |
| **Natural Alias Contrast** | Thirst Drop Contrast ($\Delta$) | $0.000$ | $0.000$ | $0.000$ | no separation |

- Awake timers and counters stay absent in all three arms ($R^2 \le 0.046$, shares $\le 0.14$).
- The steps-asleep gain is in the post-action export only: post-percept $R^2$ is .779 / .811 / .812 (F4 / HUD /
  MultiStep), so it does not show better retention in the underlying carry. Zombie readiness and hit probes peak in
  HUD-Change, not here. Energy-up-asleep's "share" is measured against a weak binned reference, not an exact observer.
- The awake timer events are deterministic given the visible history (`20261107_f4_followups/RESULTS.md` 14), so
  their absence is a learning failure, not missing information. Why the auxiliary did not install them is not
  established by this run; `../20261111_bootstrap_target_audit/` measures how little its MSE target separates due
  from not-due histories.

---

## 3. Native memory contrasts

Natural thirst alias (DEV episode 376, identical recent 8 frames, true next drink 5 vs 4):

| Metric | Episode 376 ($t=654$) | Episode 376 ($t=658$) | Target Behavior |
| :--- | :--- | :--- | :--- |
| **Current Thirst Digit** | 5 | 5 | — |
| **True Next Thirst Digit** | 5 | 4 | Decrement at $t=658$ |
| **Reader Holdout Accuracy** | $1.000$ | $1.000$ | Perfect ground-truth read |
| **Mean Reader Confidence** | $0.99932$ | $0.99869$ | Confident digit extraction |
| **Generated Digits ($K=128$)** | `[0, 0, 0, 0, 2, 126, 0, 0, 0, 0]` | `[0, 0, 0, 0, 2, 126, 0, 0, 0, 0]` | Distinct distributions |
| **Generated Drop Rate** | $2/128$ | $2/128$ | High drop rate at $t=658$ |
| **Contrast ($\Delta$)** | — | — | $\mathbf{0.000}$ (F4 0.000, HUD 0.000) |

24 exact-history hunger pairs (1,536 futures per checkpoint), generated decrement when due / not due:
F4 5.99% / 5.86%, HUD 6.51% / 6.64%, MultiStep 5.34% / 5.47%, truth 100% / 0%.

---

## 4. Training & Monitor Loss Progression

Held-out next-percept loss ($g_1$) per checkpoint. Each model is scored on its OWN learned percept targets, so the
columns are not a common-unit quality score: the final values split into pure-noise contributions .002162 / .002093 /
.002098 (F4 / HUD / MultiStep); MultiStep's lower total comes from nonzero signal levels.

| Checkpoint | Frames | P1-F4 Monitor Loss | P1-HUD-Change Loss | P1-MultiStep-Mamba | Relative Difference vs F4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2.56M | 0.01358 | 0.012670 | 0.013129 | $-3.3\%$ |
| 2 | 5.12M | 0.00845 | 0.007625 | 0.008432 | $-0.2\%$ |
| 3 | 7.68M | 0.00672 | 0.005868 | 0.004997 | $-25.6\%$ |
| 4 | 10.24M | 0.00591 | 0.005385 | 0.004295 | $-27.3\%$ |
| 5 | 12.80M | 0.00492 | 0.004159 | 0.003993 | $-18.8\%$ |
| 6 | 15.36M | 0.00389 | 0.003370 | 0.003409 | $-12.4\%$ |
| 7 | 17.92M | 0.00371 | 0.003303 | 0.003391 | $-8.6\%$ |
| 8 | 20.48M | 0.00355 | 0.003187 | 0.003070 | $-13.5\%$ |
| 9 | 23.04M | 0.00334 | 0.003093 | 0.003189 | $-4.5\%$ |
| **10 (Final)** | **25.60M** | **0.00323** | **0.003077** | $\mathbf{0.003001}$ | **$-7.1\%$** |

Periphery effective rank rose from 4.49 to 5.80; no collapse by this measure (rank does not exclude selective loss).

---

## 5. Belief Path Probe ($R^2$, MLP reader)

| Target | HUD tokens | Percept Embed | $b_{t-1}$ | $bp_t$ | F4 $bp_t$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Health Value** | $1.00$ | $1.00$ | $0.97$ | $0.94$ | $0.76$ |
| **Food Value** | $1.00$ | $0.78$ | $0.56$ | $0.37$ | $0.34$ |
| **Drink Value** | $1.00$ | $0.90$ | $0.82$ | $0.73$ | $0.70$ |
| **Energy Value** | $1.00$ | $0.56$ | $0.65$ | $0.57$ | $0.53$ |
| **Hunger Awake (Timer)** | $0.12$ | $0.01$ | $0.01$ | $-0.00$ | $-0.01$ |
| **Thirst Awake (Timer)** | $0.12$ | $0.04$ | $0.04$ | $0.01$ | $0.02$ |
| **Fatigue Awake (Timer)**| $0.12$ | $0.04$ | $0.16$ | $0.05$ | $0.03$ |
| **Fatigue Asleep (Timer)**| $0.07$ | $0.02$ | $0.26$ | $0.30$ | $0.14$ |
| **Recover Awake (Timer)**| $0.29$ | $0.22$ | $0.26$ | $0.21$ | $0.13$ |
| **Steps Asleep (Duration)**| $0.34$ | $0.31$ | $0.81$ | $0.87$ | $0.77$ |

Exact food digit in the belief's input embedding: F4 .417, HUD .610, MultiStep .518.

---

## 6. Consequences (matched three-arm values)

| Reading | F4 | HUD-Change | MultiStep |
| :--- | ---: | ---: | ---: |
| Risky generated termination (recorded .1277) | .0362 | .0810 | .0555 |
| Awake cooldown contrast, fraction of real | .159 | .189 | .294 |
| Sleeping damage contrast, fraction of real | .053 | .114 | .066 |
| Coal first unit | .986 | .545 | .821 |
| Coal increments at 1-3 | .912 | .109 | .057 |
| Iron first unit | .923 | .105 | .198 |
| Iron increments at 1-3 | .902 | .035 | .059 |
| Stone increments at 1-3 | .956 | .792 | .789 |
| Stone increments at 5-8 | .715 | .720 | .661 |
| Wood increments at 5-8 | .083 | .334 | .105 |
| Free-running entities step 16 (real 1.273) | .861 | .617 | .756 |

Waking at energy 9 is 1.0 in all three. MultiStep keeps part of HUD-Change's danger gain and recovers part of its
coal-first and long-recursion losses, but coal and iron increments stay collapsed relative to F4. Crafting: recipe 11
(wood pickaxe) appearance $T = 0.894$, spent $T = 0.889$ (both track); head unlock $T = 0.926$ / $0.999$. Hazard
determinant curves: cooldown partial ($T = 0.294$); lethality ($T = 0.197$), sleep attack ($T = 0.066$), zombie kill
($T = 0.261$) and cow kill ($T = 0.241$) blind. Hazard consequences remain under-drawn although the belief linearly
predicts zombie readiness (AP .891) and hits (.746); which part of the system fails to use them is not identified here.
