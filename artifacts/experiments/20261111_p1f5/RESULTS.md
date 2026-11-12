# P1-F5 results (2026-11-13): forecast modes on TSP and the outcome heads, change map removed

Run: launched by Codex with the user's approval, `saccade-p1f5.service` 2026-11-12 08:57 AEDT, 25.6M frames, 15,625
updates, 15 h 07 min. The monitor loss fell at all ten checkpoints, so checkpoint 10 was selected (`run/ckpt_10.pt`, SHA256
`b232d2d7...`). Every stage of the checkpoint-bound evaluation queue completed (`readings/STATUS.json`).

The follow-up queue (`followups.py`) completed sleep. Movement and mining failed on a stale assertion: `routes_f4.setup`
checks `model.denoiser.shortcut/.dense`, attributes that were removed when `df702338` made shortcut forcing with dense
attention the only denoiser. `followups.py` now does the same load and encode with a config-level check (shortcut
forcing, 4 sampler steps); both stages were rerun on 11-13. The dissection readings (sections 3, 4, 6 and 7) are
`dissect.py`, run on trained checkpoints with no training. The matched table against F4, HUD-Change and the MultiStep
arm is `compare.py` -> `readings/comparison.json`. One world seed per arm.

## 1. Summary

1. **Hidden state that the reward forecast prices is now learned, carried by Mamba and drawn in generation.**
   - Zombie cooldown on exact-history pairs: hit drawn at 0.862 when ready vs 0.025 when cooling. F4 was 0.065 vs
     0.058, HUD-Change 0.223 vs 0.108.
   - The contrast is carried entirely by the belief stream: effect 0.837, against −0.0005 for the visual cache, which
     holds the earlier hit (section 3).
   - The recovery clock (26 steps) went from chance to AP 0.585 in the belief.
   - Recovery on exact-history pairs (identical last 8 frames and recent rewards): +1 health drawn at 0.508 when due vs
     0.000 when not, 29/32 pairs in the right direction and none reversed. F4 and HUD-Change are flat at 0.01 / 0.01
     (section 4).
   - This is the first world in the project to time a hidden clock of this length in its own generation. Credit
     through ~26 recurrence steps and the denoiser's use of the belief both work.
2. **Timers that only the pixels show are unchanged.** Hunger pairs: 0.069 due vs 0.068 not due (F4 0.060 / 0.059).
   Natural thirst alias contrast: −0.008. Food, drink and energy drop APs from the belief are at their base rates.
3. **Why: the TSP forecast cannot render what the belief holds (section 7).**
   - On frames where the digit does not change, its error at the health cell is 0.094. That is against 0.129 for a
     digit-blind forecaster and 0.010 for the reactive mode, although the belief holds health at 97% exact.
   - Food is 0.172 vs 0.188 blind.
   - The HUD cells carry 14–25% of the window's forecast error, so the loss priced them on every frame. It could not
     buy them through this readout.
   - Energy separates the remaining suspects. Its awake drops register at Mamba's input (AP 0.587, 23x the base rate,
     section 6), yet the belief has no energy timer. Registration is therefore not energy's binding link.
   - Food is additionally poorly registered (awake AP 0.191; F4 0.268, HUD-Change 0.369).
4. **What removing the change map, bundled with the forecasts, cost (section 8):**
   - TSP fidelity: held-out 2.571 vs 2.161–2.218 nats/pixel; training MSE doubled.
   - Periphery effective rank fell from 5.96–6.03 to 5.22.
   - Food at the belief's input fell from 0.61 to 0.35 exact.
   - Stone increments are worse; waking at energy 9 fell to 0.906.
   - One-step entity count is lower (0.951 vs 1.020); free recursion at step 16 is 0.650 (F4 0.861, HUD-Change 0.617).
   - Coal and iron stay collapsed as in HUD-Change: coal DO/DO +1 at 0.055 on the mining pairs, against F4's 0.899.
     The collapse appears with the full-frame change map and without one, never with F4's map-only map. Its cause is
     open.

## 2. Native exact-history pairs (`readings/native_pairs*.json`, 32 draws per member)

| Pairs | F4 due / not due | HUD-Change | F5 | F5 paired contrast, 95% CI |
| :--- | :--- | :--- | :--- | :--- |
| Hunger (24): food drop | 0.060 / 0.059 | 0.065 / 0.066 | 0.069 / 0.068 | [−0.003, 0.005] |
| Zombie cooldown (32): health −2 | 0.065 / 0.058 | 0.223 / 0.108 | **0.862 / 0.025** | [0.757, 0.901] |

F5's cooldown pairs all point the right way (32/32). Generated digits are mostly exact: the due member draws the true
next digit in 0.760 of futures.

## 3. Cooldown route swap (`readings/dissection/swap.json`)

Each member is generated with the 8-frame cache's percepts from member v and the belief stream (the cache's memory
tokens and the generating memory) from member b, under the same draws. Member 0 is due.

| | (v, b) = 00 | 01 | 10 | 11 | belief effect | visual effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F5 | 0.862 | 0.025 | 0.863 | 0.025 | **0.837** [0.757, 0.902] | −0.0005 [−0.0015, 0.0] |
| HUD-Change | 0.223 | 0.110 | 0.224 | 0.108 | 0.114 [0.074, 0.159] | 0.0005 [−0.001, 0.002] |

Native cells reproduce section 2 exactly. The earlier hit sits inside the visual cache and contributes nothing; the
contrast comes from Mamba.

## 4. Recovery pairs (`dissect.py build_recovery`, `readings/dissection/recovery*.json`)

Built like `../20261108_full_frame_change/phase_pairs.py`: `player_recover` set to 20 vs 12 at a recorded daytime
awake state, followed by NOOPs.
- Selection: health ≤ 7, food, drink and energy ≥ 3; 32 pairs from 32 DEV episodes.
- Checks: no damage and no termination; frames, rewards and terminations identical over the last 8 steps. The earlier
  rise (reward +0.1) is at step 6 vs 14, i.e. 26 vs 18 steps back, both outside the cache. The due rise is at step 32.

| | due | not due | contrast, 95% CI | pairs due > not due / reversed |
| :--- | :--- | :--- | :--- | :--- |
| F4 | 0.009 | 0.008 | 0.001 | 1 / 0 |
| HUD-Change | 0.012 | 0.013 | −0.001 | 0 / 1 |
| **F5** | **0.508** | **0.000** | 0.508 [0.370, 0.637] | 29 / 0 |

## 5. Belief probes (`readings/belief_probe.json`, `readings/belief_path_probe.json`)

Event AP from the belief predicted before the frame (`bp`). The percept-plus-action control is 0.04–0.05 for the
first six rows and 0.53–0.83 for the zombie rows in every arm.

| Event | prevalence | HUD-Change | MultiStep | F5 |
| :--- | :--- | :--- | :--- | :--- |
| Food drop, awake | 0.037 | 0.038 | 0.042 | 0.042 |
| Drink drop, awake | 0.044 | 0.047 | 0.048 | 0.048 |
| Energy drop, awake | 0.027 | 0.038 | 0.036 | 0.035 |
| Energy rise, asleep | 0.086 | 0.160 | 0.167 | 0.184 |
| Recovery (+1 health), awake | 0.041 | 0.046 | 0.049 | **0.585** |
| Need damage, awake | 0.035 | 0.040 | 0.042 | **0.250** |
| Zombie ready | 0.502 | 0.912 | 0.891 | **0.992** |
| Zombie hit, awake | 0.161 | 0.780 | 0.746 | **0.960** |

Counter R² from `bp` (MLP), F4 / HUD-Change / F5:
- recover (awake): 0.131 / 0.186 / **0.525**; at the belief's input it is 0.20–0.22 in all arms, so the clock is
  computed by the recurrence;
- fatigue (awake): 0.027 / 0.045 / 0.099;
- hunger: −0.007 / −0.005 / −0.007;
- thirst: 0.018 / 0.016 / 0.026.

Exact digit accuracy at the belief's input (`percept_embed`, MLP), F4 / HUD-Change / F5:
- health 0.92 / 1.00 / 1.00;
- food 0.42 / 0.61 / **0.35**;
- drink 0.52 / 0.57 / 0.49;
- energy 0.42 / 0.49 / 0.47.

## 6. Reset registration at Mamba's input (`readings/dissection/registration*.json`)

Corrected M4 protocol (`../20261108_denoiser_teaching_continuation`): 40k DEV transitions (seed 2120), logistic on
|Δ percept_embed|, 5 episode folds, AP. F4 reproduces its archived values exactly (health 0.99865, food 0.26796, drink
0.19100, energy 0.06281).

| Digit change | F4 | HUD-Change all / awake / asleep | F5 all / awake / asleep |
| :--- | :--- | :--- | :--- |
| Health | 0.999 | 0.995 / 0.994 / 0.998 | 0.998 / 0.997 / 0.999 |
| Food | 0.268 | 0.366 / 0.369 / 0.389 | **0.184 / 0.191** / 0.119 |
| Drink | 0.191 | 0.194 / 0.196 / 0.232 | 0.236 / 0.237 / 0.267 |
| Energy | 0.063 | 0.572 / 0.414 / 0.816 | 0.691 / **0.587** / 0.852 |

## 7. What the TSP forecast can render (`readings/dissection/pricing.json`)

F5's own TSP heads at fixed HUD windows over 120 DEV episodes (30,703 frames). The table gives per-cell squared error
in [−1, 1] pixel units. Digit-blind is that cell's pixel variance across frames.

| Cell | forecast, unchanged | forecast, changed | reactive, unchanged | digit-blind | share of window forecast error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Health | 0.094 | 0.280 | 0.010 | 0.129 | 0.14 |
| Food | 0.172 | 0.199 | 0.010 | 0.188 | 0.24–0.25 |
| Drink | 0.147 | 0.197 | 0.006 | 0.202 | 0.21 |
| Energy | 0.121 | 0.206 | 0.007 | 0.160 | 0.17 |

The forecast head is a per-cell MLP. It reads a learned mask, the cell's position embedding and one 128-number
projection of the belief copied to every cell. Its pixel decoder is shared with reactive mode. The denoiser reads the
belief through attention over Mamba's memory tokens instead, and draws recovery and hits (sections 3 and 4). Both
forecast losses were still falling at the end (TSP forecast 0.0511 -> 0.0495, reward forecast 0.077 -> 0.0625 over
the last two segments).

## 8. Standard readout (`readings/comparison.json`)

**Danger and sleep (recorded or truth / F4 / HUD-Change / F5):**
- woken beside a ready zombie: 1.000 / 0.560 / 0.441 / **0.959**;
- woken beside a cooling one: 0.007 / 0.177 / 0.064 / **0.018**;
- risky class P(negative reward) / P(termination): 0.127 / 0.128, then 0.042 / 0.036, 0.089 / 0.081, **0.118 / 0.102**;
- adjacent zombie P(negative): 0.159 / 0.054 / 0.091 / **0.143**;
- WAKE_UP P(termination): 0.077 / 0.016 / 0.035 / 0.070;
- waking at energy 9: 1.000 / 1.000 / 1.000 / **0.906**;
- waking at energy ≤ 5: 0.01–0.03 truth; F5 0.015.

Sleep rollout (32 sources × 8 draws × 48 steps), restricted wait ratio against the simulator: energy 9 at 3.24
[2.70, 4.07] and awake at 3.06 [2.61, 3.68], against F4's 4.78 and 4.20. Generated energy 9 is followed by waking in
0.51 of 410 eligible transitions (F4 0.97 of 165). This is biased low, because a failed wake makes the next step
eligible again.

**Achievements** (P(unlock) in the interface; recorded 1.0):
- crafting MAKE_* 0.96–0.99 (F4 0.90–0.97);
- DEFEAT_ZOMBIE 0.48 (F4 0.29, HUD-Change 0.34);
- EAT_COW 0.29 (F4 0.24);
- PLACE_FURNACE 0.81 (F4 0.91, HUD-Change 0.97);
- COLLECT_COAL 0.82 (F4 0.98, HUD-Change 0.61);
- COLLECT_IRON 0.51 (F4 0.90, HUD-Change 0.13).

**Inventory increments (generated next = count + 1), F4 / HUD-Change / F5:**
- coal 0->1: 0.986 / 0.545 / 0.569;
- coal 1->2 .. 4->5: 0.79–0.95 / 0.05–0.16 / 0.03–0.07;
- iron 0->1: 0.923 / 0.105 / 0.321;
- stone 1->2 .. 4->5: 0.91–0.97 / 0.74–0.85 / 0.61–0.74;
- wood 0->1 .. 4->5: 0.90–1.00 / 0.81–0.96 / 0.69–0.90.

**Recursion (free-running visible entities at steps 1 / 4 / 8 / 16):**
- truth 1.109 / 1.219 / 1.328 / 1.273;
- F4 1.020 / 0.973 / 0.891 / 0.861;
- HUD-Change 1.020 / 0.893 / 0.762 / 0.617;
- F5 0.951 / 0.852 / 0.729 / 0.650.
Rewards per step are matched (F5 teacher 0.015, free 0.009, truth 0.014).

**S1:**
- Perception: entity recall 1.000 in daytime in every arm; at night within ±0.03 of F4 and HUD-Change.
- Belief memory of departed zombies: AP 0.21 at 1–4 steps (0.11–0.13 in the others) and 0.35 at 17–64 (0.26).
- Generated-vs-real classifier accuracy: 0.988 (F4 0.999, HUD-Change 0.996).
- TSP nats/pixel: 2.571, day 2.412 (F4 2.218 / 1.950; HUD-Change 2.161 / 1.875).

**Training (last 40 logged updates; medians over the ten checkpoints for gradient norms into the encoder):**
- reactive TSP MSE 0.00232 (F4 0.00117, HUD-Change 0.00108);
- reactive reward 0.0066 (0.0040 / 0.0024);
- periphery effective rank 5.22 (5.96 / 6.03);
- encoder gradients: reward forecast 4.15, reward 1.19, TSP forecast 0.64, TSP 0.41, G1 0.04. In F4 the change map
  was largest (1.11), then TSP 0.54 and reward 0.55. In HUD-Change: change 1.01, reward 0.84, TSP 0.37.
- Harmoniser sigmas at the end: 0.384 / 0.371 / 0.388 / 0.469 (G4).

## 9. Follow-ups (`readings/followups/*/result.json`)

- **Movement** (64 move-vs-NOOP pairs, 16 futures each): every move future moved and every NOOP future stayed still,
  in 64/64 pairs (F4 had the same result).
- **Mining, coal (201 pairs, 8 futures; cells are denoiser action / memory action):**

  | Cell | F5 | F4 |
  | :--- | :--- | :--- |
  | DO/DO | 0.055 | 0.899 |
  | NOOP/DO | 0.002 | 0.234 |
  | DO/NOOP | 0.001 | 0.400 |
  | NOOP/NOOP | 0.000 | 0.002 |

- **Mining, wood (201 pairs):**

  | Cell | F5 | F4 |
  | :--- | :--- | :--- |
  | DO/DO | 0.914 | 0.994 |
  | NOOP/DO | 0.348 | 0.153 |
  | DO/NOOP | 0.025 | 0.448 |
  | NOOP/NOOP | 0.000 | 0.000 |

  F5 moved the wood increment onto Mamba: the memory's action alone carries 0.348 and the denoiser's own action alone
  0.025, the reverse of F4. The recorded successors read correctly in 100% of pairs.

## 10. What this does not establish

- **Which bundled change caused what.**
  - The reward forecast is the only anticipatory, surprise-priced teacher of health events in the bundle. The TSP
    forecast cannot render even health, and removing a reactive head cannot add anticipation. So attributing the
    recovery and cooldown gains to the reward forecast is an inference from consistency, not an ablation.
  - The costs in section 8 cannot be divided between the change map's removal and the forecast modes, which share
    TSP's network and decoder.
- **Why the TSP forecast head renders belief content poorly.** Candidates are capacity, its sharing with the reactive
  mode, or a global-to-cell readout that a per-cell MLP fits slowly; none is measured.
- **Whether fixing that readout would teach the awake timers.** Food would still face weak registration.
- **The coal collapse mechanism,** inherited from the HUD-Change era.
