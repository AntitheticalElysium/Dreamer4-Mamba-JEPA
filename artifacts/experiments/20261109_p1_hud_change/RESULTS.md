# P1-HUD-Change Results (2026-11-10): Full-Frame Change Map Grounding (9x9x2)

Run: 25.6M frames, world seed 1, training 09:36-00:01 (14h 25m, 15,625 updates). Held-out monitor loss ($g_1$)
decreased monotonically across all 10 checkpoints without any three-increase trigger, selecting checkpoint 10
(`ckpt_10.pt`, SHA256 `967d1a15...`). The first readout attempt failed in the consequence stage; the stage was rerun
and every substage completed (`readings/STATUS.json`). Readout: S1 entity and belief probes, the belief-path probe,
consequence determinant curves, crafting strata, actor interface, 4-step own-output recursion and the natural thirst
alias (DEV episode 376, $t=654$ vs $t=658$). F4 comparisons below use F4's own saved outputs on the same instruments
(`readings/comparison.json`, `compare.py`), one world seed per arm.

---

## 1. Summary

1. **The belief's input now carries health exactly and food better; the timers are still absent.** The raw HUD
   periphery tokens were already read exactly in F4 (reader accuracy 1.00 in both arms). What changed is what the
   belief receives and keeps (`readings/belief_path.json` vs `20261107_f4_followups/readings/danger/belief_path_f4.json`,
   exact digit accuracy, belief input / corrected belief / post-action belief):

   | Digit | F4 | HUD-Change |
   | :--- | :--- | :--- |
   | Health | 0.92 / 0.83 / 0.67 | 1.00 / 1.00 / 0.99 |
   | Food | 0.42 / 0.34 / 0.28 | 0.61 / 0.40 / 0.32 |
   | Drink | 0.52 / 0.42 / 0.34 | 0.57 / 0.42 / 0.35 |
   | Energy | 0.42 / 0.47 / 0.41 | 0.49 / 0.43 / 0.39 |

   Hidden counters stay absent (MLP R² from the corrected belief: hunger 0.01, thirst 0.03, fatigue 0.16, recovery
   0.21; F4 0.00 / 0.03 / 0.06 / 0.20).
2. **The natural thirst alias is unchanged.** Both histories generate a decrement in 1/128 futures (F4: 2/128 each);
   contrast 0.000 against a true contrast of 1 (section 2).
3. **Danger forecasts improve; mining regresses severely.** Risky-class generated termination 0.081 vs F4 0.036
   (recorded 0.128); sleeping hits draw fewer impossible wakes (cooling 0.064 vs 0.177) but also fewer true ones
   (ready 0.441 vs 0.560). Coal and iron collapse: coal 0->1 0.545 vs 0.986, coal 1-3 0.05-0.16 vs 0.85-0.95, iron
   0->1 0.105 vs 0.923. Stone 5->6 and wood at counts 5-8 improve (section 4).
4. **Long recursion is worse than F4.** Free-running visible entities fall to 0.617 at step 16 vs F4 0.861 (truth
   1.273); rewards per step stay matched (section 5).
5. **The monitor loss is not a like-for-like comparison.** HUD-Change ends at 0.003077 vs F4's 0.003233, but each
   model is scored on its own learned percept targets (section 3).

What this arm does not establish: why the world ignores the older reset. It does not separate missing memory content,
the denoiser not using it, and the denoiser's cache; no cache-masking test was run.

---

## 2. Natural Thirst Alias Contrast (DEV Episode 376)

Context: identical 8-frame observations, identical actions, identical rewards/terminations up to $t=654$ and $t=658$.

| Metric | Episode 376 ($t=654$) | Episode 376 ($t=658$) | Target Behavior |
| :--- | :--- | :--- | :--- |
| **Current Thirst Digit** | 5 | 5 | — |
| **True Next Thirst Digit** | 5 | 4 | Decrement at $t=658$ |
| **Reader Holdout Accuracy** | $1.000$ | $1.000$ | Perfect ground-truth read |
| **Mean Reader Confidence** | $0.999993$ | $0.999993$ | Confident digit extraction |
| **Generated Digits ($K=128$)** | `[0, 0, 0, 0, 1, 127, 0, 0, 0, 0]` | `[0, 0, 0, 0, 1, 127, 0, 0, 0, 0]` | Distinct distributions |
| **Generated Drop Rate** | $1/128$ ($0.0078125$) | $1/128$ ($0.0078125$) | High drop rate at $t=658$ |
| **Contrast ($\Delta$)** | — | — | $\mathbf{0.000}$ (F4: 0.000) |

---

## 3. Training & Monitor Loss Progression

Held-out next-percept loss ($g_1$, shortcut flow term) per checkpoint (2.56M frames each). Each model is scored on
its OWN learned percept targets, so the columns are not a common-unit quality score; Codex's matched audit splits
the final values into their pure-noise contributions, 0.002162 (F4) vs 0.002093 (HUD-Change)
(`20261111_two_arm_audit/REVIEW.md`).

| Checkpoint | Frames | P1-F4 Monitor Loss | P1-HUD-Change Loss | Relative Difference |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 2.56M | 0.01358 | 0.012670 | $-6.7\%$ |
| 2 | 5.12M | 0.00845 | 0.007625 | $-9.8\%$ |
| 3 | 7.68M | 0.00672 | 0.005868 | $-12.7\%$ |
| 4 | 10.24M | 0.00591 | 0.005385 | $-8.9\%$ |
| 5 | 12.80M | 0.00492 | 0.004159 | $-15.5\%$ |
| 6 | 15.36M | 0.00389 | 0.003370 | $-13.4\%$ |
| 7 | 17.92M | 0.00371 | 0.003303 | $-11.0\%$ |
| 8 | 20.48M | 0.00355 | 0.003187 | $-10.2\%$ |
| 9 | 23.04M | 0.00334 | 0.003093 | $-7.4\%$ |
| **10 (Final)** | **25.60M** | **0.00323** | $\mathbf{0.003077}$ | **$-4.8\%$** |

---

## 4. Consequences & Determinant Curves (matched instruments, `readings/comparison.json`)

### Sleep & Waking

| Reading | Truth | F4 | HUD-Change |
| :--- | :--- | :--- | :--- |
| Waking at energy 9 | 1.000 | 1.000 | 1.000 |
| Waking at energy <= 5 | 0.01-0.03 | 0.007 | 0.005 |
| Woken beside a ready zombie | 1.000 | 0.560 | 0.441 |
| Woken beside a cooling zombie | 0.007 | 0.177 | 0.064 |

### Danger in the actor interface (generated P(negative reward) / P(termination))

| Class | Recorded | F4 | HUD-Change |
| :--- | :--- | :--- | :--- |
| Risky | 0.127 / 0.128 | 0.042 / 0.036 | 0.089 / 0.081 |
| Adjacent zombie | 0.159 / 0.000 | 0.054 / 0.000 | 0.091 / 0.001 |

### Inventory increments (generated next = count + 1)

| Transition | F4 | HUD-Change |
| :--- | :--- | :--- |
| Coal 0->1 | 0.986 | 0.545 |
| Coal 1->2 / 2->3 / 3->4 | 0.945 / 0.845 / 0.943 | 0.161 / 0.049 / 0.068 |
| Iron 0->1 | 0.923 | 0.105 |
| Stone 5->6 / 6->7 | 0.332 / 0.702 | 0.605 / 0.690 |
| Wood 5->6 / 6->7 / 7->8 / 8->9 | 0.004 / 0.177 / 0.270 / 0.083 | 0.388 / 0.214 / 0.370 / 0.167 |

Full per-count tables for all four materials: `readings/comparison.json`.

---

## 5. Recursion & Rollouts (16 Steps, 128 Windows x 4)

| Metric | Truth | F3 (3 steps) | F3 (16 steps) | P1-F4 (4 steps) | P1-HUD-Change (4 steps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Visible Entities Step 1** | 1.109 | 0.797 | 1.127 | 1.020 | 1.020 |
| **Visible Entities Step 4** | 1.219 | 0.422 | 1.266 | 0.973 | 0.893 |
| **Visible Entities Step 8** | 1.328 | 0.217 | 1.387 | 0.891 | 0.762 |
| **Visible Entities Step 16** | 1.273 | 0.131 | 1.633 | 0.861 | 0.617 |
| **Teacher Reward / Step** | 0.014 | 0.039 | 0.050 | 0.015 | 0.014 |
| **Free Reward / Step** | 0.014 | 0.016 | 0.018 | 0.007 | 0.009 |
| **Free Terminations by 16**| — | 0.004 | 0.033 | 0.016 | 0.020 |
