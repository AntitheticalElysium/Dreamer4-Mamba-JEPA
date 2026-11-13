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
3. **Leading explanation, not established: the TSP forecast cannot render what the belief holds (section 7).** The
   readout problem is measured. Its role in the timer failure is a hypothesis.
   - On frames where the digit does not change, its error at the health cell is 0.094. That is against 0.129 for a
     digit-blind forecaster and 0.010 for the reactive mode, although the belief holds health at 97% exact.
   - Food is 0.172 vs 0.188 blind.
   - The HUD cells carry 14–25% of the window's forecast error, so the loss priced them on every frame. It could not
     buy them through this readout.
   - Energy's awake drops are visible at Mamba's input: AP 0.587 from the input-embedding difference, 23x the base
     rate (section 6). Every awake energy change in DEV is a decrement, i.e. the clock's reset. Yet the belief has no
     energy timer. Whether the reset registers in Mamba's state is not measured.
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

Hunger pairs share the last 8 frames. Cooldown pairs share only the last 4: the earlier hit lies inside the 8-frame
cache, so section 3's swap establishes the route. F5's cooldown pairs all point the right way (32/32). Generated digits
are mostly exact: the due member draws the true next digit in 0.760 of futures.

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
- **Why the TSP forecast head renders belief content poorly:** answered in 11.6. It is its joint training, not its
  form.
- **Whether fixing that readout would teach the awake timers:** see 11.3 and 11.8. The readout is one of three factors
  in a request that is about zero.
- **The coal collapse mechanism,** inherited from the HUD-Change era.

## 11. Second dissection (11-13): who asks Mamba for a clock

Frozen readouts on the ten F5 checkpoints and the F4/HUD-Change controls, no training. Scripts: `data_counts.py`,
`trajectory.py`, `credit.py`, `credit_g1.py`, `credit_readout.py`, `uptake.py`, `entities.py`, `readout_fit.py`, and the
energy pairs in `dissect.py`. G1's credit and the readout split were added to the approved list. Both are frozen
measurements completing the credit test.

### 11.1 How often each loss saw each clock's events (`readings/dissection/data_counts.json`)

F5's exact training stream was rebuilt from `run/gaze.bin` (25.5M frames, 10.46 passes). Each row's forecast window
was rebuilt from its seed, against the full training labels and counters. "Usable" means the clock's previous reset
fell inside the same training chunk; gradients stop at chunk boundaries.

| Event | In the stream | Usable | Covered by the TSP forecast window | Covered and usable | Covered by the learned gaze |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Food drop, awake | 690,380 | 488,950 | 56,164 | 39,899 | 147,252 |
| Drink drop, awake | 792,250 | 601,371 | 96,977 | 73,782 | 197,724 |
| Energy drop, awake | 550,717 | 384,153 | 67,036 | 46,534 | 118,304 |
| Energy rise, asleep | 379,319 | 338,368 | 46,519 | 41,482 | 128,934 |
| Health +1 (recovery) | 430,497 | 334,744 | 17,476 | 13,589 | 141,675 |

The reward forecast reads every frame, so it saw all 334,744 usable recovery events. The TSP forecast saw 40-74k
usable events per pixel clock over the whole run. That is about the number the reward forecast had seen of recovery by
checkpoint 2 (55,397), when recovery was beginning to form in the belief (11.2). The learned gaze covers 2-3x more
change events only because of its leak: its surprise term reads the target frame.

### 11.2 Checkpoint trajectory (`readings/dissection/trajectory/`)

Same exact-history pairs at every checkpoint. Head separation: reward forecast, log-odds; TSP, the fraction of the true
pixel difference reproduced. Native draws are due / not due.

| Ckpt | Recovery: reward sep | Recovery: belief AP | Recovery native | Cooldown: reward sep | Cooldown native | Energy native / TSP | Hunger native / TSP |
| :--- | ---: | ---: | :--- | ---: | :--- | :--- | :--- |
| 1 | -0.04 | 0.05 | 0.146 / 0.146 | 0.23 | 0.867 / 0.866 | 0.917 / 0.914, 0.000 | 0.867 / 0.867, 0.000 |
| 2 | 2.26 | 0.20 | 0.301 / 0.291 | 1.21 | 0.772 / 0.773 | 0.694 / 0.696, 0.000 | 0.760 / 0.763, 0.000 |
| 3 | 5.17 | 0.28 | 0.021 / 0.009 | 3.92 | 0.096 / 0.040 | 0.016 / 0.016, 0.000 | 0.042 / 0.043, 0.000 |
| 5 | 7.82 | 0.43 | 0.177 / 0.000 | 7.08 | 0.555 / 0.060 | 0.000 / 0.000, 0.000 | 0.034 / 0.034, 0.000 |
| 7 | 12.87 | 0.54 | 0.268 / 0.000 | 10.12 | 0.856 / 0.069 | 0.005 / 0.005, 0.000 | 0.122 / 0.121, 0.000 |
| 10 | 15.62 | 0.59 | 0.508 / 0.000 | 11.36 | 0.862 / 0.025 | 0.009 / 0.009, 0.001 | 0.069 / 0.068, 0.000 |

- The reward forecast's branch separation and the belief's recovery AP rise together from checkpoint 2.
- Generation uses the clocks about two checkpoints later.
- Energy and hunger stay at zero on every measure throughout.
- Energy pairs (32, `dissect.py build_energy`) on the controls: F4 0.004 / 0.004, HUD-Change 0.003 / 0.003. No arm
  shows even a weakly timed energy response.
- The energy pairs all fork at episode step 14 with the due drop 8 -> 7: a narrow, early-episode sample.

### 11.3 Who asks the belief to tell the branches apart (`readings/dissection/credit/`)

Per pair at the due transition, for each loss as trained, the rate is d(separation)/d(eta) under one SGD step on that
loss. Separation is the loss rise from swapping the two members' beliefs. The rate is reported over belief parameters,
mean over 32 pairs, with the number of positive pairs.

| Ckpt | Reward: recovery | Reward: cooldown | TSP: recovery | TSP: energy | TSP: hunger |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | +499 (30) | +91 (27) | -0.0005 (15) | -0.0003 (15) | -0.000002 (11) |
| 2 | +2,011 (20) | +496 (28) | +0.003 (20) | -0.0007 (12) | +0.0002 (13) |
| 3 | +4,676 (16) | +5,030 (32) | +0.025 (24) | +0.0009 (23) | -0.0001 (12) |
| 5 | +10,263 (23) | +5,198 (31) | +0.022 (24) | -0.0001 (16) | -0.0019 (6) |
| 10 | +78,544 (30) | +5,005 (27) | +0.026 (25) | -0.0001 (12) | -0.0016 (7) |

Both heads are in G4 with one harmoniser weight, so the units are comparable.
- **For energy and hunger, the TSP forecast asks nothing at any checkpoint:** about zero, with a random sign.
- **On recovery it asks only once the belief already holds the clock,** and then 10^6 x below the reward forecast. It
  follows the belief; it did not teach it.
- **The gap has two measured factors:**
  - the separation gradient of TSP's head is 10^3 x smaller (0.004 vs 4.9 at checkpoint 1);
  - the currency: a few pixels inside a 1,323-number window mean against a whole categorical outcome. The rate scales
    with the square of the loss magnitude.

G1 (`credit_g1.py`, its flow term as trained over all 8 signal levels; the cache and the generating memory are swapped
between members), rate in objective units:

| Ckpt | G1: recovery | G1: cooldown | G1: energy | G1: hunger |
| :--- | ---: | ---: | ---: | ---: |
| 1 | -1.3e-6 (16) | -8.0e-7 (21) | -9.7e-7 (17) | -1.3e-6 (11) |
| 3 | +4.0e-4 (25) | +2.0e-4 (28) | -4.8e-6 (16) | +2e-8 (16) |
| 5 | +8.8e-3 (30) | +2.8e-3 (31) | +3.9e-7 (21) | +7e-8 (12) |
| 10 | +4.3e-2 (28) | +6.5e-3 (24) | -2.2e-7 (15) | -2.8e-7 (13) |

- **At checkpoint 1, the reward forecast is the only loss that asks for any clock.** Recovery: reward +499, TSP
  -0.0005, G1 -1.3e-6.
- **G1 and TSP turn positive on recovery and cooldown only once the belief holds them.** G1's rising request is what
  teaches the denoiser to read the clock, consistent with generation lagging the belief by about two checkpoints.
- **No loss asks for energy or hunger at any checkpoint.**

Readout vs currency (`credit_readout.py`, checkpoint 10, TSP forecast loss with fresh readouts fitted on the frozen
beliefs):
- **Recovery:** trained head 0.026, fresh same form 8.3 (28/32), fresh attention 15.8 (30/32), against the reward
  forecast's 78,544. A working readout buys ~2.5 of the gap's 6-7 orders; the pixel-MSE currency the remaining ~4.
- **Energy and hunger:** about zero, random sign, for every readout. Their branches' states at the due step are nearly
  identical (11.4), and the separation gradient is (J_A - J_B)^T (y_A - y_B), which vanishes when they coincide.

### 11.4 Reset uptake and retention (`readings/dissection/uptake.json`)

Did event X happen exactly L steps before frame t? Read from Mamba's corrected belief b_t (linear logistic, 5 folds,
40k DEV frames); F5 / HUD-Change. The input-embedding control is at the base rate for every row at L >= 1.

| Event (base rate ~0.02-0.03) | L = 0 | 1 | 2 | 4 | 8 | 16 | 28 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Health +1 (recovery's reset) | **0.944** / 0.421 | 0.758 / 0.112 | 0.485 / 0.041 | 0.131 / 0.042 | 0.078 / 0.043 | 0.053 / 0.035 | 0.166 / 0.030 |
| Health drop | 0.973 / 0.895 | 0.842 / 0.398 | 0.844 / 0.157 | 0.346 / 0.088 | 0.149 / 0.068 | 0.063 / 0.044 | 0.038 / 0.028 |
| Energy drop, awake | 0.055 / 0.048 | 0.055 / 0.049 | 0.049 / 0.046 | 0.041 / 0.044 | 0.035 / 0.037 | 0.033 / 0.032 | 0.027 / 0.028 |
| Food drop, awake | 0.037 / 0.046 | 0.040 / 0.040 | 0.037 / 0.038 | 0.033 / 0.035 | 0.041 / 0.043 | 0.039 / 0.036 | 0.030 / 0.029 |
| Drink drop, awake | 0.094 / 0.046 | 0.058 / 0.039 | 0.041 / 0.039 | 0.044 / 0.044 | 0.039 / 0.037 | 0.041 / 0.040 | 0.037 / 0.038 |

- **The pixel-only resets are not taken into the state.** Energy's reset is visible across two consecutive inputs
  (section 6, AP 0.587) yet reaches b_t at 0.055. Food's does not reach it at all.
- **HUD-Change is the natural control for recovery.** Its reset also arrives as a +0.1 reward input, but nothing there
  asks to anticipate recovery: uptake is 0.42 and is gone after one step. In F5 uptake is 0.94, and the counter forms
  (section 5).
- **F5's recovery row rises again at L = 28 (0.166).** That is one period after a rise, i.e. the belief represents
  phase.
- **Reading:** the request taught uptake as well as counting. A reset that is merely visible at the input is not taken
  in unless some loss asks for the timing it sets.

### 11.5 Entities (`readings/dissection/entities.json`; recursion decomposition from the saved rollouts)

One-step generation from real histories (S1 protocol, 20,000 identical tuples per arm), each arm read by its own S1
reader. True entities are classed by simulator slot. Recall is within one cell, against the reader's ceiling on the
real frame.
- **F5's one-step deficit is in persisting, moving entities.** By day, moved entities are recalled at 0.705 against
  F4's 0.757 and HUD-Change's 0.759; stationary ones at 0.879 against 0.912 and 0.906. At night, moved ones are at 0.543
  against 0.580 and 0.599.
- **By kind, it is cows and arrows.** Persisting cows: 0.823 / 0.586 day / night, against 0.874-0.880 / 0.657-0.688.
  Persisting arrows: 0.167 / 0.095 against 0.282-0.381 / 0.164-0.194. Zombies and skeletons are unchanged. F5 also
  invents more cows (0.071 per frame by day, against 0.056-0.057).
- **Entities entering view (scroll-in or walk-in) and spawns are rarely drawn in any arm (0.04-0.10).**
- **Free-running decline (`recursion_4`, corrected by each model's reader on the real frames):** at step 16 F4 is at
  -0.373, HUD-Change -0.656 and F5 -0.600, mostly cows and zombies. That decline arrived with HUD-Change, not with F5.
  Paired F5 - F4: -0.061 [-0.143, +0.015] at one step and -0.227 [-0.356, -0.098] at step 16, matching Codex's numbers.
- **Reading:**
  - F4's map-cell change map, and HUD-Change's, are the only grounding in those arms that is about where things moved
    between frames, and F5's own loss is moving cows and arrows. This matches the small-moving-entity hedge the change
    map was added for (10-26). It does not show up in perception (S1 recall unchanged).
  - The free-running decline and the coal collapse both appear with HUD-Change's change target (F4 -> HUD-Change is a
    one-change comparison), and the MultiStep arm replicates the coal collapse with a different initialisation.

### 11.6 What a readout can extract from the same belief (`readings/dissection/readout_fit.json`)

Fresh readouts were trained only to forecast, on checkpoint 10's frozen pre-frame beliefs (uniform windows, 80% of DEV
episodes, 20k steps). They are scored on held-out episodes at the HUD windows, on frames where the digit does not
change.

| Cell | Digit-blind | Trained F5 head | Fresh, same form | Fresh, attention over memory tokens |
| :--- | ---: | ---: | ---: | ---: |
| Health (the belief holds it at 97%) | 0.118 | 0.091 | **0.0095** | 0.0063 |
| Food | 0.193 | 0.177 | 0.152 | 0.151 |
| Drink | 0.210 | 0.149 | 0.124 | 0.121 |
| Energy | 0.167 | 0.131 | 0.091 | 0.096 |

- **The trained head's failure is not its form.** A fresh head of the same form renders health almost perfectly; it
  fails because of its training, shared with the reactive mode. Attention adds little.
- **Pair separation** (fraction of the true change reproduced):
  - recovery: trained 0.052, fresh 0.25 / 0.32-0.37;
  - energy and hunger: about 0 for every readout.
- **Reproducibility:** a rerun reproduces the same-form numbers to 6 decimals. Attention varies slightly (recovery
  0.37 vs 0.32).
- **Sharing check (fresh single-mode heads of the original classes, same frozen inputs):**
  - TSP reactive-only reaches MSE 0.00044 against the trained shared head's 0.00199 on held-out uniform windows. The
    trained reactive mode learned mostly on the chooser's windows, so part of that gap may be window shift.
  - The fresh outcome heads underfit within the same budget (reward NLL 0.033 / 0.228 against the trained 0.014 /
    0.085), so the outcome head's sharing cost is not determined.

### 11.7 How much of a reset survives until the due step (`readings/dissection/trace.json`)

Relative distance between the two branches' corrected beliefs. The branches differ in input only between steps 6 and
14.

| Ckpt | Recovery: step 6 / 15 / due | Energy: step 6 / 15 / due | Hunger: step 6 / 15 / due |
| :--- | :--- | :--- | :--- |
| 1 | 0.420 / 0.084 / **0.014** | 0.122 / 0.034 / **0.0073** | 0.025 / 0.009 / **0.0065** |
| 3 | 0.698 / 0.227 / 0.152 | 0.230 / 0.086 / 0.0128 | 0.040 / 0.013 / 0.0077 |
| 10 | 0.854 / 0.387 / 0.250 | 0.269 / 0.085 / 0.0150 | 0.050 / 0.016 / 0.0087 |

- **Before learning, every reset leaves a faint trace of the same order at the due step.** Energy's is half of
  recovery's; hunger's is similar. So the trace is not what separated recovery from the pixel clocks.
- **The reward forecast's request amplified recovery's trace 18x by checkpoint 10;** energy's and hunger's barely
  moved.
- **Food's reset barely perturbs the state even at the moment it happens (0.025)**, consistent with its poor
  registration (section 6).

### 11.8 What this round establishes

1. **A loss's request is what makes a clock.**
   - At checkpoint 1, before anything was learned, the reward forecast asks Mamba to tell the recovery branches apart
     at +499 and cooldown at +91. The TSP forecast asks -0.0005 and G1 -1.3e-6.
   - Asked, the whole chain forms: the reset is taken in (recovery uptake 0.42 -> 0.94), a counter forms (R² 0.19 ->
     0.53), and the head separates from checkpoint 2. G1 then begins asking, and generation uses the clock about two
     checkpoints later.
   - Credit through ~26 recurrence steps and the denoiser's use both work. The faint starting trace is comparable for
     every reset.
   - The requested clocks are exactly the hidden variables that change future reward, as reward-predictive
     representation theory says.
2. **No loss asks for the pixel-only clocks at any checkpoint,** so their resets are never taken in, though visible at
   the input. Energy pairs are flat in every arm.
3. **The TSP forecast's request is about zero for three measured reasons:**
   - its trained readout was spoiled by joint training (about 2.5 orders of magnitude of the gap);
   - the pixel-MSE currency, a few pixels in a window mean against a whole categorical outcome (about 4 orders);
   - its uniform window covered 8% of events, 40-74k usable per clock over the run. That matches what recovery had by
     checkpoint 2, about 5-8x below recovery's total, and below the ~139k recovery needed before generation used it.
4. **The user's rarity concern holds, separately.** Fixing the readout alone would leave each pixel clock with the event
   budget of recovery's first two checkpoints. Rarity is not the cause of the readout's failure: the trained head had
   about 1M covered frames of the current health digit and still could not draw it.
5. **The one-step entity regression of F5 is moving cows and arrows.** The free-running decline and the coal collapse
   arrived with HUD-Change.

Not established:
- **Which objective would teach pixel-only clocks.** The measurements say what such a request must provide: a currency
  in which one rare change is a surprise, coverage of the cells where it happens, and an unspoiled readout. They do not
  test a candidate.
- **The attribution of the recovery and cooldown gains to the reward forecast** rests on it being the only loss that
  asks for them before they exist, not on an ablation.
- One world seed; the energy pairs are a narrow, early-episode sample.
