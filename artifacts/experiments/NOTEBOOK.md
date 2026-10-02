# Lab notebook — LeWM + Mamba on Craftax-Classic

The single index of what was run, where, with which parameters, and what it established. Each entry points to
the script and evidence file that hold the exact numbers. Newest campaign first. Corrections are kept, not
deleted; each names the claim it retires.

## Conventions

**Data**
- Corpus: `artifacts/craftax_expert_store_v1` + `artifacts/craftax_support_v2`, contract in
  `artifacts/lewm_m4_canonical/raw/dataset.json`. 10,400 episodes: 8,325 TRAIN / 1,041 DEV / 1,034 FINAL.
- Latent cache: `lewm_m4_canonical/raw/cache` (eval-mode z of the Raw joint encoder).
- Pools:
  - `artifacts/eda/interface_pool_v1`: u, z; 24,576 main + 8,071 terminal 6-frame windows, TRAIN.
  - `spatial_pool_v1`: Raw layer-normed tokens.
  - `spatial_pool_tc_v1`: the same windows, TC encoder.

**Judgement blocks**
- `artifacts/eda/observe_fresh_v4..v12` = seeds 53k..61k. Each root has frames, visible/hidden simulator state,
  and 32-key P(death) for all 17 actions.
- 55k-58k are opened (post hoc use only). 59k, 60k and 61k were each read once, sealed.
- Diagnostic futures: `artifacts/eda/diagnosis_futures_v1`, seeds 900,000+, 1,002 roots.
  - Per root: 4 context frames, all 17 actions x 4 keys one step ahead, and the policy's 16-step future x 5 keys.
  - Diagnosis split: 70/30 by seed, `randperm(seed 0)`.

**Models**
- Encoders: Raw and TC joint step-10,000 (`lewm_m4_canonical/{raw,tc}/joint`). The bridge's encoder equals
  Raw joint's.
- Canonical world: Mamba-2 (6 layers, width 256, d_state 64, headdim 64, 4 heads).
- Joint recipe: 4-frame windows, batch 128, AdamW 5e-5, wd 1e-3, 500 warmup, cosine to 5e-6, 10,000 updates,
  pred MSE + 0.09 SIGReg (1,024 projections, 17 knots).

| world | architecture and state | trained by |
|---|---|---|
| H2 | canonical Mamba, raw z | joint 10k + bridge 2k |
| Z, U, W | canonical Mamba; z, u (PCA-192 of the 4x4 patch grid), w (u / std) | `interface.py` |
| sZ, T | block-causal transformer (`spatial.World`); LN z, or 81 LN patch tokens | `spatial.py` |

**Metrics**
- Expected safe = 1 - P(death1 | chosen action).
- "x copy" = squared error / error of copying the previous frame; < 1 beats copying.
- "/ V" = squared error / total natural variance of that state.
- Captured = 1 - error / predictable change.
- Historical facts in the existing levers eval JSON files are closed-form ridge probes fitted on TRUE
  TRAIN-seed states and transferred to imagined TEST-seed states. This measures decoder transfer as well
  as what is readable from imagination. The corrected teval schema v2 reports both facts_true_fit and
  facts_generated_fit (same probe family fitted on each world's generated TRAIN states), plus an all-action
  generated-fit one-step control. Neither is an actor or information-theoretic ceiling; see the
  2026-09-29 correction below. Existing JSON files retain their historical meaning. The generated-fit factual readout uses a true root token followed by generated successors; the all-action readout uses generated one-step successors. For the 18k corrt and fmamba worlds, version-2 physical and historical true-fit metrics reproduce the old JSON exactly (maximum absolute delta 0.0); all five LDAD historical true-fit fact grids also reproduce exactly.

## Canonical pipeline status (2026-09-27)
- Raw: joint done; H2 bridge trained; **H2 gate failed** (5/7 checks).
  - The checks that fail are 1 broken, 1 borderline and 3 unmeasurable at their sample size (`raw/gates/h2`).
- TC: joint done, never bridged. It stopped on a cache tolerance: 1.9e-5 vs an allowed 1e-5.
- **Bugs found in `d4mj`:**
  1. H2 terminal depth alias. Fixed on branch `h2-terminal-depth` (9e42b7a2), not merged.
  2. Context length (2026-09-27; see below). Not yet fixed.
- The actor has never been trained on LeWM.
- 2026-09-30: no world has been trained on more than two self-fed steps, and none has been tested on
  safe-action choice (see the audit below). The length-64 continuation stopped at update 200 of 3,600.

---

## 2026-10-02 17:20 — reviewed at the user's pause request (results 15:45-16:22); PAUSE: no new trainings

E14c, both seeds (mask1 + skip vs the same-seed teacher 18k):

| reading | s7 | s8 |
|---|---|---|
| c_learned (held caught) | **0.991** (pass) | **0.950** (pass; baseline 0.271) |
| c_cost (onestep_all) | 0.149 → 0.256 (fail) | 0.152 → 0.256 (fail) |
| blocked / idle | 0.30 → 0.93 / 0.44 → 0.88 | 0.32 → 0.93 / 0.47 → 0.90 |
| c_position (ever wrong) | 0.449 → 0.466 (fail) | 0.493 → 0.477 (fail) |
| c_depth16 (gen_16) | +0.160 (fail) | +0.100 (fail) |
| c_decision: gen1 | -0.050 [-0.064, -0.035] (fail) | -0.037 [-0.051, -0.022] (fail) |
| c_decision: gen2 | -0.007 [-0.014, -0.001] | ns |
| transfer1 | **+0.072** [+0.054, +0.094] | **+0.042** [+0.024, +0.060] |
| transfer2 | +0.013 [+0.002, +0.024] | ns |

- Why (s7, measured):
  - check_e14c_grad: the mask term's backbone gradient is 5.4x the uniform term's in the trained world (cos 0.05).
  - check_e14c_refit: backbone_damage TRUE.
    - A fresh linear uniform head on the E14c backbone keeps the damage: idle static +34%, sleep static +134%.
    - Only the near-player error on interact steps was the skip head's.
    - That head catches 0.000 consequences: they need the dose even on this backbone.

E14d s7, mask0.1 + skip (dose x31): consequences caught **1.000** (hallucinated 0.006); onestep_all 0.206 (+38%; blocked
0.859, idle 0.783, sleep back to normal 0.017); gen_16 +0.030 [+0.013, +0.046]; excess16 0.590; ever position-wrong 0.410 (-9%).
d_cost, d_position and d_depth16 fail at s7. The dpanel and seed 8 are pending; E14d s8 is held by the pause.

E14b' s7, 36k plain recipe:

| reading | 18k | 36k |
|---|---|---|
| consequences caught | 0.002 | 0.562 |
| onestep_all | 0.149 | 0.126 (-0.023, resolved) |
| every one-step class | | better: moved -0.034, blocked -0.031, interact -0.039, idle -0.032 |
| gen_16 | 0.622 | 0.556 (-0.067, resolved) |
| excess16 | 0.560 | 0.493 |
| ever position-wrong | 0.449 | **0.298 (-34%; E13's oracle consequence substitution: -37%)** |

- b_cost, b_position and b_depth16 pass at s7. The dpanel is pending (an OOM at 16:10, retrying); s8 36k lands ~18:00.
- The leading reading at s7 (one seed, decision panel pending): the 18k worlds were under-trained.
  - The uniform loss learns the consequences late and abruptly (s8 12-18k, s7 18-24k) and improves everything else.
  - The dose buys the consequences but damages the backbone on blocked / idle / sleep.

---

## 2026-10-02 afternoon — E14c seed 7 (interim; seed 8 and the linear arm pending)

corrt_raw_teacher_s7_mask1_skip_u18000 vs corrt_raw_teacher_s7_u18000:

| reading | teacher s7 | E14c s7 | rule |
|---|---|---|---|
| consfit held caught / hallucinated | 0.002 / 0.003 | **0.991** / 0.011 (train 0.998) | c_learned **pass** |
| teval onestep_all (err / copy) | 0.149 | 0.256 (+72%; interact 0.863 → 0.652, blocked 0.301 → 0.926, idle 0.442 → 0.882) | c_cost **fail** |
| subst16 ever position-wrong / at 16 | 0.449 / 0.464 | 0.466 / 0.499 | c_position **fail** |
| subst16 depth-16 excess (static) | 0.560 (0.439) | 0.712 (0.553) | — |
| compare gen_16 | 0.622 | 0.783 (+0.160 [+0.135, +0.186]) | c_depth16 **fail** |
| dpanel gen1 / gen2 (E14c - teacher) | | -0.050 [-0.064, -0.035] / -0.007 [-0.014, -0.001] | c_decision **fail** |
| dpanel transfer1 / transfer2 | | **+0.072** [+0.054, +0.094] / **+0.013** [+0.002, +0.024] | reported |

- End-to-end, the backbone learns the consequences almost perfectly (head-only on a frozen h: 0.63).
- The rest of the world degrades broadly (`check_e14c_cost.py`, held, by action x token class):
  - no false scrolls (blocked 0.0095 → 0.012);
  - static tokens are 68% of the increase, in every action class;
  - player token on moved steps 0.091 → 0.249; HUD +72% on moved steps.
- Against the baseline's own snapshots:
  - moved / blocked player, HUD and static errors sit at the 6k level, as if under-trained;
  - idle and sleep static, and near-player tokens on interact steps, are WORSE than the 6k baseline: active damage.
- The broad degradation offsets the gain E13's oracle predicted (-37% position failures), so position failures do not
  drop.
- Open: does the mask term dominate the gradient (`check_e14c_grad.py`)? Is the skip readout behind the active damage
  (the linear arm)? Seed 8.

---

## 2026-10-02 midday — E14c PREDECLARED: end-to-end training with the head-only fix (labelled upper bound)

Head-only facts that set it up (E14a addenda, both corrt seeds unless noted):

| s7 (s8) head, frozen backbone | caught | all-token L1 vs uniform | hallucinated |
|---|---|---|---|
| any readout, uniform loss | 0.002 (0.26-0.28) | — | — |
| linear, mask0.1 (x31) | 0.002 | +0% | 0.003 |
| linear, mask1 (x304) | 0.541 | +21% | 0.046 |
| linear, mask3 / mask10 | 0.581 / 0.579 (0.568) | +38% / +59% (+38%) | 0.05 |
| MLP, mask1 | 0.583 | +7% | 0.049 |
| **skip (raw local neighbourhood), mask1** | **0.628 (0.653)** | **+7% (+9%)** | **0.022 (0.027)** |
| SimPLe dead zone q50 / q75 / q90 | 0.002 | ≤ +4% | 0.001 |
| hard-token (own error) 1% x3 | 0.237 (0.200) | +83% (+71%) | 0.058 (0.054) |
| skip + hard 1% x3 | 0.633 (0.529) | +72% (+84%) | 0.155 (0.154) |
| linear, calm-frame events c60 x2 (label-free) | 0.565 | +119% | 0.105 |
| skip, calm-frame events c60 x1 / x2 | **0.925** / 0.934 | +85% / +108% | **0.315** / 0.378 |

Seed 8 (where not in brackets above), addendum:
- dead zone q50 / q75 / q90: 0.255-0.263, null at both seeds;
- mask0.1 (x31): 0.439, already effective on s8's sharper h (check_allprobe AP 2x s7's);
- mask1: 0.554 (+14%); mask3: 0.567 (+26%);
- mlp_mask1: 0.588 (+5%, passes); mlp_mask10: 0.595 (+26%).

Readings at both seeds: H_capacity false, H_representation true.

The dose needed scales with the backbone's precision. Calm-frame events catch 92-93% with a skip head but hallucinate on 32-38% of
failed attempts.
- The event dose up-weights only the changes that happened, about x300. That moves the change / no-change threshold to
  P > ~1/300, so every plausible change is drawn. This is our reading of the numbers, not yet tested.
- The labelled mask doses the faced tile of every attempt, successful or not, so the prior within the attempt set is
  unchanged.
- A selective dose therefore needs its matching negatives: the context where a change could happen, not only the changes.
- Tested (addendum 5, predeclared in 909a5b45): dose ONLY the strict consequence tiles at mask1's per-token dose (x304),
  skip readout.

  | | positives only (s7 / s8) | attempt mask, skip_mask1 (s7 / s8) |
  |---|---|---|
  | caught | 0.966 / 0.946 | 0.628 / 0.653 |
  | hallucinated | 0.297 / 0.240 | 0.022 / 0.027 |
  | all-token L1 | x1.06 / x1.07 | x1.07 / x1.09 |
  | HUD | x0.98 / 0.97 | x0.99 / 0.97 |

  - H_prior_shift is TRUE at both seeds (hallucination 11-13x).
  - Positives-only dosing costs almost nothing elsewhere. So the calm-event arms' +85-108% L1 and HUD x3-4 come from
    dosing the OTHER events (HUD, mobs), and their hallucination from the prior shift.
  - With this h, catching versus hallucinating is a threshold trade-off: success depends on inventory, which the head
    must read from h.
- A label-free dose WITH matching negatives, an action-effect map counted per (action, k-means state of the agent token)
  (`check_effectmap.py`, predeclared), is not selective enough.
  - Best: recall 0.57 at 1.3% of tokens dosed, with consequences only 2.5% of the dosed set (the attempt mask: 0.33% /
    17%). The rule failed.
  - Status of label-free selection: by loss, by reducible loss, by calm events and by count maps, none matches the
    attempt mask. E14c decides first whether the consequence fix is worth this search.

- Where the linear mask1 cost lands (`check_costwhere.py`, held):
  - 70% on static tokens. The trained head uses "generate" at 12% weight to refine copies; the dose pulls the single linear
    proj toward consequence content.
  - 14% on the 4 tokens next to the player (+69%: h cannot tell an attempt's faced tile from its neighbours).
  - An MLP readout cuts the static part (+20% → +5%); the near-player part stays (+58%).
- Generic doses fail for a measured reason (`check_toperr.py`). Consequences do sit in the error tail (median rank
  0.32% / 0.53%; top-1% recall 0.998 / 0.72), but 93% of the tail is something else: entering cells 52%, static 25%,
  HUD 12%. A x300 dose on the tail distorts everything.
- E14c asks the decision-relevant question before searching further for a generic selective dose: if catching the
  consequences (with Craftax labels) does not improve imagination and decisions end-to-end, the search is moot.
- RHO-LOSS (Mindermann et al., ICML 2022, eq. 3, verified in the PDF; `check_rho.py`) does not rescue a label-free
  selection. Its irreducible-loss model, a local MLP on the raw neighbourhood trained with the uniform L1, copies the
  consequences too: IL 0.900 vs the world's 0.919.
  - The reducible-loss tail is 71% static tokens; consequence recall is 0.000 / 0.002 in the top 0.3%.
  - Every model trained with the uniform per-token L1 on this data copies them: all transformer heads, MLP / skip
    readouts, and a standalone local MLP whose input carries the conjunction.
  - A classifier on the same input with balanced positives singles them out (AP 0.667). The obstacle is rarity under
    the uniform objective, not the information.

Arms (lane35, then lane36 evaluates): corrt, Raw, teacher, 18k, seeds 7 and 8, the u18000 recipe except
`--weight mask1 --skip` (primary) or `--weight mask1` (linear head). The 36k budget control (E14b, declared in lane32.sh)
runs in the same lane between them.

Readings (fixed in lane35.sh before training; comparator = the same-seed teacher u18000 world; two-seed rule):
- c_learned: consfit held caught ≥ 0.5.
- c_cost: teval onestep_all ratio ≤ 1.10.
- c_position: subst16 ever_position_wrong at least 20% lower. E13's oracle substitution gave -37%.
- c_depth16: gen_16 not resolved worse.
- c_decision: dpanel gen1 or gen2 resolved better.
- skip_needed: the skip arm passes c_learned and c_cost while the linear arm fails one.

GPU (11:15):
- lane23 stops after deepeval group1 (group3 and group2 resume after the trainings);
- the direct / categorical addenda are dropped. Those worlds never scroll; the two corrt seeds decide.

---

## 2026-10-02 morning — E14a / E14m launched; E12 port correction (primary source)

**E12 correction: our rollout2 was not an exact port, and my "RoPE vs learned absolute positions" explanation is refuted.**
Read at facebookresearch/jepa-wms 13cf1d9:
- The Metaworld 2-roll config (`configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_2roll.yaml`)
  uses `pred_type: dino_wm`, `use_rope: null`.
- Its predictor `ViTPredictor` (app/plan_common/models/vit.py) has a learned absolute (frame x patch) table,
  `pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))`, applied as `pos_embedding[:, :n]`. That is
  the same class of encoding as our learned space + time tables, so RoPE is not the difference.
- The difference is the context window:
  - That config sets `ctxt_window_train_rollout: 3` (code default 8). `video_wm.rollout` feeds the predictor
    `vid_feats[:, -ctxt_window:]`, and the positions are indexed from 0, so the generated frame sits at slot 1 or 2 of a
    window of at most 3 frames.
  - Its evaluation rollouts also use 3 (`data_traj_eval_ctxt_window: 3`).
  - Our port (`tworld.rollout_losses`) fed the full prefix: the generated frame was at slots 1-4 of a window of up to
    6 frames, and we evaluate with a 5-frame window.
- E12's measured mechanism was blocked moves scrolled from TRUE frames when the current frame sits in slots 1-4; with a
  1-frame window the world is fine. That is consistent with this train/eval slot mismatch, which FAIR's setup avoids by
  construction. Consistent only, not tested: an E12 arm with a 3-frame rollout window would test it.
- E12 negative is not noise: both seeds are resolved worse at depth 16, and the effect is larger than the seed spread.
  It stays lowest priority.

**E14a launched (lane30, `headfit.py`, readings predeclared in 72a4ca77).** Two questions decide which fix attacks the right
problem:
- Can the output head alone learn DO / place consequences from the frozen backbone? This is Kang et al.'s cRT.
- What does the gradient look like at the trained checkpoint?

First fact, corrt teacher s7 at 18k, mean mixture weights:

| tokens | self | up / down / left / right | generate |
|---|---|---|---|
| consequence (faced tile of a DO / place that changed it) | 0.971 | 0.0001-0.0002 each | 0.029 |
| all other tokens | 0.745 | 0.028-0.031 each | 0.141 |

- The head copies the faced tile of an attempt, and copying is right on 83% of attempts. Labels:
  - 26.7% of transitions are attempts and 4.6% are strict consequences;
  - 4,647 training windows hold at least one consequence.
- Gradient share of consequence tokens in the head:
  - 0.11% under L1 and 0.76% under L2 (x6.8);
  - cos(g_cons, g_rest) -0.14 per batch and -0.37 summed over 20 batches.

**E14a result (lane30, `headfit.py`, 4 worlds): with the backbone frozen, the head alone cannot express the consequences
without paying everywhere else. A fresh head re-learns exactly the trained catch rate.** Held-out strict consequences
(556; train within 0.02 of held everywhere):

| arm (dose on consequence tokens) | corrt teacher s7 | corrt teacher s8 | categorical s7 | direct suffix s7 |
|---|---|---|---|---|
| trained head, +3,000 updates (`continue`) | 0.002 | 0.257 | 0.004 | 0.000 |
| fresh head, uniform (cRT) | 0.002 | 0.257 | 0.004 | 0.000 |
| L2 | 0.002 | 0.264 | — | 0.000 |
| EAWM same-position, w 0.5 (x1.7) | 0.002 | 0.270 | 0.034 | 0.000 |
| EAWM copy-compensated, w 0.9 (x4.7) | 0.002 | 0.273 | 0.113 (L1 +7%) | 0.000 |
| EAWM copy-compensated, w 0.99 (x8.8) | 0.002 | 0.270 | 0.464 (L1 +61%, halluc 0.175) | — |
| resample, half the batch from rich windows (x3.7) | 0.002 | 0.261 | 0.007 | 0.000 |
| mask10, faced tile of every attempt (x3,034) | 0.579 (L1 +59%) | 0.568 (L1 +38%) | 0.653 (L1 +36%) | 0.570 (L1 +59%) |

- Predeclared readings:
  - H_uniform_copies true wherever the trained world copies; H_converged true everywhere: the fresh uniform head reaches
    the trained head's all-token L1.
  - H_head_fixable, H_generic and H_geometry false everywhere: mask10 catches but fails the L1 guard.
  - G_starved true everywhere: consequence tokens carry 0.11% (s7), 0.32% (direct) and 0.9% (categorical) of the head
    gradient norm.
  - G_l1_sign: 6.8 (s7), 4.3 (s8), 6.5 (direct). L2 changes nothing at the head, so the per-token sign of L1 is not the
    cause.
  - G_cancel false (batch cos -0.14 to +0.05).
- The seed difference lives in the backbone. The s8 backbone gives 0.257 with a fresh head, s7's gives 0.002.
- Doses up to x9 do nothing to the L1 heads. The CE head (categorical) responds, but always at a large cost.
- The L1-median explanation is refuted: on the faced cells of attempts, a linear probe on the same h gives
  P(change | changed) 0.634 / 0.652 (s7 / s8; E13e). 69% of true changes are above 0.5, where the L1-optimal decision is
  already "change", yet the head copies 99.8% (s7).
- Doses (from the labels, `headfit_labels_v1`):
  - consequence tokens are 0.0569% of tokens; faced tiles of attempts are 0.33%;
  - 23% of consequence tokens are not copy-residual events: the new tile matches a neighbour's token;
  - EAWM's published w = 0.5 gives x1.7.
- Open (lane32 addendum, predeclared in e741a46c): an MLP readout on the same frozen h (capacity or representation?),
  the mask dose response x31 / x304 / x911, and SimPLe's dead zone.

**Why the head cannot single them out (`check_allprobe.py`): the backbone dilutes a conjunction that the raw local input
carries.** Probes for "this token is a consequence" among ALL held tokens at the natural rate (829,440 tokens, 556
positives, 0.067%):

| features | AP linear | recall @ precision 0.5 (linear) | AP MLP | fresh-head caught (E14a) |
|---|---|---|---|---|
| h, corrt s8 | 0.338 | 0.451 | 0.442 | 0.257 |
| h, corrt s7 | 0.164 | 0.020 | 0.281 | 0.002 |
| h, direct s7 | 0.043 | 0.000 | 0.115 | 0.000 |
| h, categorical s7 | 0.020 | 0.000 | 0.091 | 0.004 |
| raw local input (token, 4 neighbours, action) | 0.214 | 0.086 | **0.667** (r@p0.5 0.917) | — |

- AUC is 0.95-0.9997 everywhere; E13's AUC 0.92-0.95 hid this. With a 0.067% positive rate, precision is the binding
  quantity.
- The fresh-head catch rate tracks h's precision across the 4 worlds. The seed difference is this: s8's h singles out 45%
  of consequences at precision 0.5, s7's 2%.
- Readings:
  - `representation` true for s7, direct and categorical (h MLP AP < 0.3);
  - `linear_bottleneck` true only for direct and categorical;
  - `seed_difference` true.
- Every backbone is worse than an MLP on the raw local input. The conjunction (faced tile x action x tile type) is in the
  input but diluted in h, which was trained with a uniform loss where it carries 0.06% of the tokens.
- Substitution test launched (lane33, addendum 2, predeclared in 045c4564): the head reads [h, raw local neighbourhood,
  action]. Under the uniform loss, then with mask doses.
  - This is a locality prior with no Craftax semantics, as Δ-IRIS's decoder is conditioned on x_t and a_t.

**Literature for these facts (primary sources read 2026-10-02):**
- SimPLe (Kaiser et al., ICLR 2020, sec. 4): "clipped loss max(Loss, C) ... crucial ... decreases the magnitude of gradients
  stemming from fine-tuning of big areas of background ... concentrate on small but important areas (e.g. the ball in
  Pong)". C = 10 for pixel L2, 0.03 for softmax. tensor2tensor applies it per element: `relu(|pred - target| - cutoff)`.
- CGSReg (arXiv 2607.15142, sec. 4 / Table 6):
  - the loss is `sum m (x - x̂)² / sum m`, i.e. our mask form;
  - λ in {0, 0.01, 0.1, 1}; selected 0.1 (DreamerV3), 0.01 (DIAMOND), 1.0 (TWISTER);
  - λ = 1 hurts DreamerV3 (-21.0 vs -11.9);
  - no measurement of quality outside the concept regions, so the trade-off we measured is not reported there.
- Δ-IRIS (Micheli et al., ICML 2024, sec. 2.2 and Fig. 3, Crafter):
  - the autoencoder, conditioned on past frames and actions, encodes only "what has changed and that cannot be inferred
    from actions, i.e. the stochastic delta";
  - the next frame's Δ-tokens are sampled autoregressively (the joint law);
  - with RANDOM Δ-tokens, the deterministic dynamics stay correct ("wood level increasing, crafting table appearing"):
    crafting consequences are learnable by a deterministic, frame-conditioned decoder in Crafter;
  - it is also the template for E14m's requirement: deterministic known content, jointly sampled unseen content.

**E14m (lane31, `monotone.py`, readings predeclared in 6512dfb6; `check_monotone_why.py`): deterministic imagination IS monotone,
exactly on content the world cannot see, and naive sampling is not the fix.**

Question (the user's): is option 3 (generative) needed so imagination is not monotone? Method: the class content of imagined
frames vs the simulator, on cells revealed after the root vs cells observable at the root; the categorical world decoded both
argmax and sampled.

Diagnosis futures, test roots, aligned depths (imagined camera = true camera); corrt teacher 18k s7 / s8:

| depth 16 | observable at root | revealed after root |
|---|---|---|
| TV vs truth (probe floor) | 0.019 / 0.038 (0.011 / 0.028) | 0.133 / 0.170 (0.001 / 0.017) |
| entropy imagined / true (bits) | 2.00 / 2.11, 1.84 / 2.04 | 1.64 / 2.29, 1.52 / 2.25 |
| grass share imagined / true | 0.541 / 0.535, 0.589 / 0.567 | 0.611 / 0.483, 0.663 / 0.493 |
| class accuracy vs simulator | 0.933 / 0.875 | 0.663 / 0.604 |

- Readings: M_monotone on revealed cells at both seeds; observable cells faithful; M_known_drift false.
- Revealed class ratios at depth 16 (imagined / true share), s7 / s8:

  | class | s7 | s8 |
  |---|---|---|
  | tree | 0.011 | 0.020 |
  | coal | 0.06 | 0.0 |
  | iron | 0.0 | 0.18 |
  | lava | 0.0 | 0.0 |
  | water | 0.63 | 0.48 |
  | grass | 1.26 | 1.35 |

- Distinct classes per frame in the revealed area: 2.2 vs 3.55. It starts at the first revealed step: depth 1 grass
  0.539 vs 0.478, entropy 1.86 vs 2.27.
- Why (check_monotone_why, 27,976 held one-step entering cells): the world's entering content equals the local deterministic
  predictor's (an MLP on the 3 visible edge tokens):

  | class | true | world s7 | MLP |
  |---|---|---|---|
  | grass | 0.476 | 0.528 | 0.527 |
  | tree | 0.037 | 0.0012 | 0.0002 |
  | stone | 0.172 | 0.186 | 0.186 |

  - Tree recall: world 0.003 / 0.0, MLP 0.0.
  - Accuracy: world 0.760 vs MLP 0.756.
  - Tokens are on-manifold: median nearest-code distance 2.75 vs 2.69 true. Where a tree enters, the world draws a crisp
    grass token (0.92 vs 1.98).
  - So the monotony is the deterministic mode under uncertainty the model cannot remove from what it sees: entering trees
    are never the most likely class. More deterministic training of the same objective cannot produce them.
- Naive generative decoding (the categorical world, every token sampled independently from its K = 4,096 code softmax,
  T = 1):
  - On the rows that can be compared, it restores the class histogram: revealed grass 0.559 vs 0.555 at depth 4, entropy
    2.12 vs 2.12.
  - It destroys known content: observable aligned accuracy 0.956 → 0.871 at depth 1, 0.949 → 0.721 at depth 4,
    0.942 → 0.467 at depth 16.
  - The predeclared sampling readings could not be computed: the categorical world never scrolls (below), so it has no
    aligned revealed cells.
- The categorical and direct worlds never draw a scroll: imagined scroll rate 0.0002 / 0.0 vs true 0.369. Their
  monotone rows describe a frozen view, so the monotone answer rests on the two corrt worlds (0.276 / 0.323).
- Answer to the user: option 3 is needed for diversity of unseen content (the trees, ores and water an agent would explore
  for). It must be a generative component confined to what the world cannot know. Independent per-token sampling corrupts
  what it does know. This is a measured requirement, not yet an intervention.

---

## 2026-10-02 — Independent E13 review and primary-source pass (read-only diagnostics; no GPU)

Full review: [`20260927_levers/E13_REVIEW.md`](20260927_levers/E13_REVIEW.md). Source/results pinned to `9fabf964`.
- Verified TRAIN underfit (teacher s7: 1/870 strict changed tiles caught), h change AUC 0.9446 and class readout 0.8633,
  and content-substitution effects across the four worlds. E13 is strong evidence for an **output underfit and content-to-move
  error pathway** in the current per-tile worlds.
- Scope correction: loss dilution is a justified intervention target, not yet a measured gradient-starvation mechanism.
  The 0.892% share is alive-transition diagnostic **L1 loss**, not gradient mass; rare coefficient mass is 0.0553% of tile targets.
- `check_enterpred.py`'s 0.760 world / 0.756 three-edge-token MLP does **not** establish that 24% is unknowable. E11 replay varies
  future RNG from one fixed full simulator state/map; its variance does not cover hidden-map uncertainty conditional on pixels.
- The 28/36/3/33 oracle decomposition is sequential and intervention-dependent; realignment also fills exposed cells with true
  tokens. `R4_target_tile_causal` remains false under its original conjunction (all-but-target repair 0.346 > 0.30), although
  target repair 0.604 versus random 0.030 establishes a substantial, non-exclusive effect. No threshold changed.
- Primary papers support testing interaction-focused output loss first; EAWM is not interchangeable with an arbitrary h-only
  auxiliary head. For (2), supervise the emitted state or a gate used in that output. For (3), learn calibrated, coherent
  hidden-content distributions and retain sampled/observed map content across reentries. Preserve failed attempts and check
  inventory/reward consistency, H16 choice and real-to-generated head transfer. No intervention launched or predeclared here.

---

## 2026-10-01 night — Literature on the diagnosed problem (E13); nothing implemented

Read from the papers' text (PDFs) unless noted. Problem 1 is the main one: a rare, deterministic, sparse action consequence (one
token in 81, 4.5% of transitions, < 0.9% of the objective) that the input and the backbone encode, yet every trained head copies.

**Problem 1, the same diagnosis in the literature: uniform per-token objectives starve sparse interaction regions.**
- CAER (arXiv 2608.30897, Aug 2026): "abundant background tokens dominate the gradient while sparse interaction dynamics remain
  under-optimized". It reweights tokens by the model's OWN action effect (prediction under the real vs a learned null action, read
  at a fixed noise level), normalized to unit mean per sample. Action dropout (p = 0.1) keeps a recall floor. First-order result:
  at equal coefficient mass, focused weighting lowers interaction risk iff Cov(weight, token utility) > 0.
  Caveats for us, from our numbers: our heads' faced-tile prediction is action-INSENSITIVE (they copy), so the effect map starts
  blind exactly there; in Craftax a move shifts the whole view, so the map would concentrate on scroll steps.
- IMPACT (arXiv 2609.00161, same group): the same "supervision-allocation mismatch". It weights candidate regions by DETACHED
  LOCAL PREDICTION ERROR, which does fire on our missed consequences (copy error ~160 squared distance). Its prior is language
  cross-attention, which we do not have.
- CGSReg (arXiv 2607.15142, Sep 2026): five SOTA world-model agents (DreamerV3, DIAMOND, TWISTER, Simulus, STORM) are weak worlds.
  Policies trained from scratch in the frozen world collapse in Pong (DreamerV3 -5.5 -> -20.9, STORM 18.7 -> -21.0), and the gap
  holds in 22-26 of 26 Atari100K games. The failures are interactions: ball disappearance, invalid ball-paddle contact.
  Fix: an auxiliary mask-normalized MSE on task-critical regions (lambda 0.01-1.0), which helps 4 of 5 models. It needs masks; for
  us the faced tile is known exactly.
- EAWM (ICLR 2026, arXiv 2601.19336): auxiliary prediction of EVENTS (statistically significant per-pixel changes; for token inputs,
  a type change) on top of the world model, +10-45% across Atari 100K, Craftax-1M and DMC.
- Gradient Starvation (Pezeshki et al., NeurIPS 2021): dominant easy features starve the gradient of less frequent predictive
  ones (theory for cross-entropy; fix: spectral decoupling). Matches our late, abrupt, seed-dependent learning (teacher s8:
  0 at 12k, DO 0.47 at 18k).
- "Modeling What Changes" (arXiv 2609.02046): a per-object change gate + residual wins only by not corrupting the static
  majority; delta regression on the movers stays at no-op quality. The same pattern as our copy heads.
- Dreamer 4 (arXiv 2509.24527, vendored): its world model learns placing / breaking blocks (14 of 16 interaction tasks) with a
  uniform objective, via a generative shortcut-forcing objective, more spatial tokens (capacity) and 2,541 h of data. Oasis
  "hallucinates large structures" after a few placed blocks, cf. teacher s8's hallucinated DO effects.
- Sample-level prioritization: Curious Replay (ICML 2023; DreamerV3 on Crafter 14.5 -> 19.4) and Simulus's prioritized
  world-model replay (Simulus's ablations exclude Craftax). Caveat: our error is one token in 81, so window-level priority dilutes
  it; token-level weighting is the direct form.
- AGWM (arXiv 2605.06841): world models learn frequent action-outcome co-occurrences as rules and ignore preconditions; it adds
  explicit affordance tracking. Its own Crafter / Craftax imagination MSE does not improve.
- Delta-IRIS (ICML 2024): the tokenizer encodes deltas conditioned on the past; the decoder carries deterministic dynamics.
  Reconstruction-based.
- Background (read, broader framing only): "Imagined rollouts are kinematic, not dynamic" (2607.05966; DreamerV3 DMC);
  "The planning limits of latent world models" (2609.39235); DWM (2607.18715, action vs world effects); MV2MAE motion-weighted
  patch loss (pretraining).

**Problem 2 (revealed terrain, ~24% unpredictable from what is visible).** The game-world-model agents that train in imagination
on Crafter / Craftax (IRIS, Delta-IRIS, Dedieu et al.'s TWM, DreamerV3) all SAMPLE the next state. A deterministic regressor
must commit to one guess. No paper found quantifies this for revealed terrain specifically.

**What this implies (candidates only; none run, each would need predeclared rules and >= 2 seeds):**
(a) token-level weighting of changed / interaction tokens: CGSReg-style on the faced tile, error-calibrated (IMPACT),
action-effect (CAER, with the cold-start caveat); (b) auxiliary event prediction (EAWM); (c) a generative / sampling objective
for unobservable content (Dreamer 4, IRIS-family). Judged on consfit (caught rate on training and held-out), E13 position
failures and depth-16 excess, and the decision panels.

---

## 2026-10-01 night — E13: the imagined-error chain by substitution; every link now causal (after the user's audit request)

The user asked: no intervention until the diagnosis is certain, then a literature sweep. E11 had built the chain from correlations
and onset attribution. E13 tests each link causally. E12 is paused: rollout2 was resolved worse than teacher-only at both seeds
(depth 16 +0.097 [+0.065, +0.132] s7, +0.088 [+0.069, +0.108] s8; blocked moves scrolled from TRUE frames 23-31%).

**Population substitution (`subst16.py`, readings in its docstring; diagnosis futures, 1,002 roots).** Depth-16 excess / share of
roots ever position-wrong:

| arm | teacher s7 | teacher s8 | suffix s7 | suffix s8 |
|---|---|---|---|---|
| no oracle | 0.560 / 0.449 | 0.619 / 0.493 | 0.627 / 0.542 | 0.624 / 0.545 |
| true consequence tiles (DO / place) | 0.493 / 0.282 | 0.522 / 0.309 | 0.608 / 0.475 | 0.599 / 0.478 |
| true entering cells | 0.380 / 0.341 | 0.415 / 0.359 | 0.447 / 0.427 | 0.451 / 0.455 |
| both | 0.250 / 0.121 | 0.289 / 0.179 | 0.416 / 0.364 | 0.408 / 0.381 |
| exact position (realign) | 0.404 / 0.025 | 0.446 / 0.025 | 0.453 / 0.058 | 0.450 / 0.049 |
| exact position + entering | 0.203 / 0.009 | 0.221 / 0.007 | 0.240 / 0.029 | 0.247 / 0.035 |
| exact position + entering + consequences | 0.184 / 0.009 | 0.186 / 0.009 | 0.225 / 0.025 | 0.232 / 0.036 |
| world's own gate replayed from TRUE input | 0.568 / 0.372 | 0.697 / 0.346 | 0.590 / 0.407 | 0.585 / 0.397 |
| gate saturated +-20 (not a faithful oracle) | 0.992 / 0.140 | 1.089 / 0.182 | 0.874 / 0.075 | 0.975 / 0.387 |

- Depth-16 error decomposition (teacher s7; all four worlds within a few points): view position 28% (realign; 1,539 corrections,
  12,769 filled cells), entering-cell content 36%, consequence pixels ~3%, residual 33% (HUD 0.044, mobs ~0.02, static drift,
  mostly sleeping steps per E11n).
- Triggers of position failures: consequences alone prevent 37% (teacher) / 12% (suffix); entering content 24-27% / 17-21%; both
  64-73% / 30-33%. The suffix worlds have a third trigger, their slot-4 scroll bias: a fully true current frame restores only
  66-68% of their flipped decisions (teacher 98%).
- The saturated gate corrupts content (HUD 0.046 -> 0.370), so it is not an oracle; the true-input gate adds nothing: the decision
  error is in the drawn content, not the gate.

**The decision step (`subst16.py` part B).** At the first wrong move decision, re-run with the current frame altered:

| restores the decision | teacher s7 | teacher s8 | suffix s7 | suffix s8 |
|---|---|---|---|---|
| only the target tile true | 0.60 | 0.61 | 0.33 | 0.36 |
| one random other tile true | 0.03 | 0.04 | 0.04 | 0.02 |
| all tiles true except the target | 0.35 | 0.35 | 0.32 | 0.34 |
| whole current frame true | 0.98 | 0.98 | 0.66 | 0.68 |

- False scrolls (teacher): the target tile alone restores 93-98%. Missed scrolls: the target tile and the rest of the frame share it.

**Consequences are never learned (E13b-f).**
- `consfit.py` (training and held-out pool windows, the worlds' own training data; strict change = probe class change AND token
  change above the 99th percentile of unchanged): DO / place change the faced tile in 4.5% of transitions (16% of DO / place
  transitions), carrying < 0.9% of the objective. Caught on TRAINING windows: corrt teacher s7, suffix s7 / s8 0.001 (copied
  0.999). The same for direct, residual, gated, corr, corrg, corrt 6k, ITC generator loss (gl, gl_itc, gl_itc 18k), categorical
  (CE on 4,096 codes), fmamba (Mamba), rollout2: 0.000-0.001. Only teacher s8 catches some (DO mining 0.47, placement 0), and only
  between 12k and 18k updates (0 at 6k and 12k).
- Not alignment (`check_align.py`): faced-tile change after NOOP 0.0016, after DO 0.217.
- Not information (`check_infoprobe.py`): an MLP on the INPUT faced token + action predicts the next class of changed tiles at 0.977
  held-out (with HUD 0.984; copying 0.000).
- Not a backbone failure (`stageprobe.py`, `conscalib.py`): the world's backbone state at the faced tile encodes it (linear class
  probe 0.79-0.86 on changed; unweighted change probe AUC 0.92-0.95, mean P(change) 0.60-0.65 on true changes vs 0.11-0.12).
- The output head does not express it: the categorical world puts 0.067 of its mass on the true next class of changed tiles and
  0.828 on the current class (`check_catmass.py`). Not a threshold effect.
- One-step, not self-feeding (`subst16.py` part C): from the TRUE window the faced-tile change is missed 100% (teacher s7) / 35%
  (s8) / 76-100% (suffix). Hallucinated changes are the self-feeding part: 1% from true windows vs 9-19% from imagined ones.

**Entering terrain is an observation limit (`check_enterpred.py`).** Class of cells entering the view, held-out: world (true
frames) 0.760, MLP on the 3 adjacent visible edge tokens 0.756, copy the adjacent tile 0.714, majority 0.476. About 24% of new
terrain cannot be inferred from what is visible, so a deterministic guess is wrong that often.

**The chain, causal:**
1. Interaction consequences (DO / place on the faced tile) are deterministic, present in the input, and encoded by the backbone,
   but every trained output head copies the tile. They are 0.05% of token targets; the one world that learns some does so late
   (12k-18k) and on one seed only. A starved minority signal, not missing information.
2. With newly revealed terrain (an observation limit), the wrong content of the target tile flips the move decision (target-tile
   repair restores 60% in the teacher worlds; a random tile 3%).
3. These two triggers cause 64-73% of position failures (teacher worlds); the suffix worlds add the slot-4 bias.
4. Position errors cause 28-29% of depth-16 error; entering content 36%; the residual 33-37% is HUD, mobs and slow drift.

Open (not this campaign): the H16 decision panel is not yet tied to this chain; the sealed deepeval groups are paused.

---

## 2026-10-01 — E10 stage 3, first sealed world (judge block 62,000-62,399; rules in 8c969101)

corrt teacher 18k s7.
- Roots: fit / dev / judge = 6,393 / 3,096 / 3,631. Judge opportunity roots: k=1 760, k=4 1,562, k=16 2,957.
- Fidelity err/copy at k = 1 / 4 / 16: 0.204 / 0.529 / 0.673.

| k | prior | root_rank | gen | transfer | real (one draw) |
|---|---|---|---|---|---|
| 1 | 0.591 | 0.692 | 0.900 | 0.531 | 0.999 |
| 4 | 0.647 | 0.655 | 0.675 | 0.635 | 0.718 |
| 16 | 0.586 | 0.588 | 0.597 | 0.569 | 0.652 |

- **E9's H1 reading, sealed confirmation: CONFIRMED.**
  - gen1 − root_rank1 = +0.208 [+0.174, +0.246];
  - gen1 − prior1 = +0.310 [+0.255, +0.363].
- Carries the H16 decision: **formally yes, but small.**
  - gen16 − prior16 = +0.011 [+0.003, +0.019];
  - gen16 − root_rank16 = +0.009 [+0.003, +0.016];
  - this is 17% of the prior-to-real-draw margin (32% at k = 4).
- Usable in imagination at H16: **no.** transfer16 − prior16 = −0.017 [−0.028, −0.008].
- The other worlds are running (lane23).

**Second sealed world, corrt suffix 18k s7 (landed before an unplanned restart at ~18:43).**
- Same 3,631 roots; every world-independent reference arm (prior / root_rank / real at k = 1, 4, 16) is bit-identical
  to the teacher run's.
- Fidelity err/copy: 0.225 / 0.560 / 0.619.

| k | gen | transfer |
|---|---|---|
| 1 | 0.950 | 0.531 |
| 4 | 0.684 | 0.626 |
| 16 | 0.602 | 0.569 |

- H1 confirmation: **confirmed** (gen1 − root_rank1 = +0.258 [+0.224, +0.293]; gen1 − prior1 = +0.359).
- Carries H16: **yes, small** (gen16 − prior16 = +0.016 [+0.008, +0.025]; gen16 − root_rank16 = +0.015 [+0.007, +0.022];
  24% of the margin to the real draw).
- Usable at H16: **no** (transfer16 − prior16 = −0.017 [−0.026, −0.008]).
- Teacher − suffix, paired (`deep_compare.json`, seed 7 only, so NOT attributed under the two-seed rule):
  - gen1: −0.049 [−0.066, −0.033]; gen4: −0.009 (ns); gen16: −0.005 [−0.011, −0.000];
  - transfer: ns at every depth.
  - The 6k pairs showed no gen1 difference on the opened blocks (−0.006, −0.002 ns).
**Third sealed world, int_corrg_raw_suffix_s7_fmamba_u18000 (Mamba-2 backbone; landed 2026-10-02 12:56).**
- Same roots and reference arms. Fidelity err/copy 0.220 / 0.546 / 0.631.
- Expected safe at k = 1 / 4 / 16:
  - gen 0.915 / 0.678 / 0.595;
  - transfer 0.597 / 0.625 / 0.576.
- H1 confirmation: **confirmed** (gen1 - root_rank1 = +0.224 [+0.190, +0.260]; gen1 - prior1 = +0.325).
- Carries H16: **marginal**.
  - gen16 - prior16 = +0.0095 [+0.0003, +0.0185];
  - gen16 - root_rank16 = +0.008 [-0.001, +0.015], not resolved.
- Usable at H16: **no** (transfer16 - prior16 = -0.0095 [-0.018, +0.0003]).
- Paired A - B, seed 7 only (not attributed under the two-seed rule; deep_compare.json, merged with the earlier pairs):

  | | gen1 | transfer1 | gen16 | transfer16 |
  |---|---|---|---|---|
  | suffix - fmamba | +0.034 [+0.018, +0.051] | -0.066 [-0.095, -0.038] | +0.007 [+0.001, +0.013] | -0.008 [-0.014, -0.001] |
  | teacher - fmamba | -0.015 (ns) | -0.066 [-0.093, -0.040] | ns | -0.008 [-0.016, -0.000] |

  The Mamba world's imagined states take real-fitted heads better (transfer) at H1 and H16, but carry less of the decision in
  their own imagined-state heads than the attention suffix world.

- Restart: lanes 19d / 23 / 27 were killed.
  - deepeval_group1 was imagining fmamba; its partial memmaps are rewritten on rerun.
  - E12 training had never been admitted: it needed 3.6 GB beside deepeval's 2.1 GB.
  - Relaunch order changed: the seed-8 18k pair (needed by E12's two-seed rule) now runs before the 6k pairs.
  - E12 training admission is 3.1 GB (E8's selffed training measured 2.9 GB), so it can run beside deepeval.

---

## 2026-10-01 — E11 / E11b result: imagined error is ~90% the model's own; its growth is wrong move decisions on imagined inputs, not randomness

Asked: is a stochastic head the fix, or an assumption? It was an assumption. This measures it.

**E11 (`stochdiag.py`, readings declared in c9bdc65a).**
- The diagnosis futures have FIVE sampled 16-step futures per root under the same actions, so their per-token variance
  is the exact floor of ANY deterministic world.
- Estimator checked on synthetic data: floor 3.920 vs 3.920 true; excess 0.0003 at the true mean, 1.999 for a 0.5
  offset (2.0).

| world | aleatoric share at 16 | excess-growth share: static / hud / mob / entering / player | mob blur | code ratio |
|---|---|---|---|---|
| corrt teacher 18k s7 | 0.092 | 0.813 / 0.082 / 0.067 / 0.031 / 0.008 | 4.21 | 1.00 |
| corrt teacher 18k s8 | 0.084 | 0.822 / 0.075 / 0.068 / 0.026 / 0.009 | 4.67 | 1.04 |
| corrt suffix 18k s7 | 0.083 | 0.822 / 0.077 / 0.067 / 0.025 / 0.009 | 4.70 | 0.99 |
| corrt suffix 18k s8 | 0.084 | 0.817 / 0.079 / 0.067 / 0.026 / 0.012 | 4.63 | 1.04 |
| fmamba corrg 18k | 0.085 | 0.820 / 0.078 / 0.068 / 0.026 / 0.008 | 4.63 | 1.10 |
| corrt teacher 6k | 0.076 | 0.806 / 0.085 / 0.069 / 0.028 / 0.012 | 5.25 | 1.08 |
| noise | 0.066 | 0.816 / 0.074 / 0.070 / 0.029 / 0.010 | 5.92 | 0.92 |
| selffed | 0.073 | 0.805 / 0.083 / 0.072 / 0.029 / 0.012 | 5.68 | 1.18 |

- Every world: aleatoric_dominated = false, growth = deterministic_growth, mode_averaged = false.
- Imagined mob tokens are 4-6x farther from the conditional mean than a typical true sample, and as close to real tokens
  (codebook distance) as true ones. They are wrong, not averaged.
- Teacher 18k s7, by depth:
  - floor 0.009 → 0.057; excess 0.027 → 0.560;
  - entering cells carry 60% of depth-1 excess (0.0158 of 0.0265) and barely grow;
  - static excess goes 0.0055 → 0.037 (depth 2) → 0.439 (depth 16).

**E11b (`driftanat.py`, readings declared in c154db33, amended before any run in ee2ca978).** The view position is
tracked by scroll.estimate offsets.

| world | map excess in position-wrong cases | realigning removes | aligned observable | aligned revealed | wrong at 16 | roots ever wrong | first error: missed / false scroll |
|---|---|---|---|---|---|---|---|
| teacher s7 | 0.657 | 0.346 | 0.155 | 0.188 | 0.464 | 447 | 0.86 / 0.11 |
| teacher s8 | 0.710 | 0.333 | 0.127 | 0.163 | 0.558 | 491 | 0.49 / 0.49 |
| suffix s7 | 0.705 | 0.345 | 0.119 | 0.176 | 0.587 | 540 | 0.52 / 0.45 |
| suffix s8 | 0.712 | 0.347 | 0.127 | 0.161 | 0.581 | 543 | 0.59 / 0.39 |
| fmamba corrg | 0.751 | 0.371 | 0.109 | 0.141 | 0.612 | 553 | 0.52 / 0.46 |
| noise | 0.914 | 0.472 | 0.079 | 0.007 | 0.807 | 800 | 0.96 / 0.02 |
| selffed | 0.662 | 0.281 | 0.161 | 0.177 | 0.536 | 510 | 0.60 / 0.36 |

- None of the declared readings is met.
  - Position-wrong cases hold 66-91% of depth-16 map excess, but realigning removes only 28-47% of it: after a wrong
    scroll the content corrupts too. At depth 2, realigning removes 78% (teacher s7).
- The trigger, in 97-98% of roots, is a wrong move decision: a missed scroll on a successful move, or a false scroll on a
  blocked one, with a world-dependent mix.
  - From true frames this almost never happens (0.3% at depth 1).
  - From depth 2 on, ~3-4% of remaining roots go wrong per step.
- Correction: an earlier message to the user said "86% missed scrolls" from teacher s7 alone. Across worlds the split
  is about half and half.
- **Answer to the question: the measured growth of imagined error is deterministic, mainly wrong move decisions on
  imagined inputs. Randomness is 7-9% of depth-16 error, and mob tokens are not mode-averaged. This diagnostic does not
  indicate a stochastic head.**
- What it does not cover:
  - revealed cells (14-19% of map excess) carry terrain no observation contained; a deterministic world can only guess it;
  - the decision value of sampling, which is untested here.
- E11c (`missedscroll.py`) localizes the wrong decisions (imagined vs true vs hybrid windows), both kinds; results below.

**E11c (`missedscroll.py`, readings declared in 20e067ed, extended to false scrolls before any run).**
- Setup: at the FIRST wrong step, the head's move logit (frame + target gate) is read on four windows: imagined; true;
  true current frame with imagined history; imagined current frame with true history.
- Share deciding right, written true / imagined / img_hist / img_cur:

| world | missed scroll (n) | false scroll (n) | target-tile distance, imagined vs true (case vs control): missed, false |
|---|---|---|---|
| teacher s7 | 1.00 / 0.37 / 1.00 / 0.38 (381) | 0.97 / 0.02 / 0.97 / 0.02 (58) | 69.3 vs 6.9, 180.9 vs 13.1 |
| teacher s8 | 0.99 / 0.69 / 0.99 / 0.69 (235) | 1.00 / 0.13 / 1.00 / 0.13 (245) | 51.3 vs 9.6, 173.5 vs 13.6 |
| suffix s7 | 0.87 / 0.60 / 0.88 / 0.61 (272) | 0.57 / 0.21 / 0.54 / 0.22 (252) | 55.3 vs 21.2, 70.8 vs 18.9 |
| suffix s8 | 0.91 / 0.53 / 0.91 / 0.58 (309) | 0.53 / 0.21 / 0.55 / 0.20 (216) | 56.4 vs 18.2, 66.7 vs 16.2 |
| selffed | 0.89 / 0.43 / 0.90 / 0.44 (299) | 0.55 / 0.06 / 0.54 / 0.07 (181) | 70.4 vs 22.6, 87.2 vs 24.6 |

- Squared distances; a layer-normed token has |x|^2 ≈ 192. Frame-mean distances are only 1.2-1.9x those of the controls.
- fmamba is excluded: its frame logit is negative on every input, so this readout cannot express its scroll.
- Declared readings:
  - false scrolls, teacher s7 / s8: input_caused and current_frame both true;
  - missed scrolls everywhere, and false scrolls in suffix / selffed: thresholds not met, because the move logit is
    only part of the corr mixture (an imperfect proxy for the actual scroll);
  - target_tile: true in every world and both kinds.
- Direction, identical in every world: swapping in the TRUE current frame restores the true-window decision rate;
  swapping in the imagined current frame reproduces the imagined rate; history does not matter.
- Provenance of the target tile (re-run reproduced run 1 exactly):
  - observable at the root: 69-82% of missed and 55-87% of false scrolls;
  - mob-occupied: ≤ 1% of missed, 3-12% of false scrolls.
  - Within the observable class, the target tile is far more corrupted on wrong decisions than on correct ones:
    49-77 vs 3-18 (missed), 41-184 vs 12-21 (false).
  - Revealed target tiles are equally corrupted on wrong and correct decisions (missed 37-56 vs 36-59).

**E11d (`adjdrift.py`, reading declared in 88aaf506).** Per-cell excess of known cells (observable, no mob, position
right), move targets vs Chebyshev ring 3+:

| world | depth 1 | depth 8 |
|---|---|---|
| teacher s7 | 6.4x | 1.28x |
| teacher s8 | 9.0x | 1.53x |
| suffix s7 | 5.5x | 1.68x |
| suffix s8 | 5.5x | 1.72x |

- Declared adjacent_drift (≥ 2x at depth 8): false. As measured: the player's neighbours drift 5.5-9x faster at the
  first imagined step, when the first wrong decisions start, and the drift then spreads over the map.
- Explanation tested in E11e-E11l below: contextual tokens, through the player MOVING next to a tile (not turning).

**E11e-E11l (2026-10-01 evening, after a restart; each reading declared and committed before its run). Why do known tiles drift
on imagined inputs?** Seven conjectures were tested in sequence; the data rejected six.
- **E11e `ctxtok.py`: contextual tokens when the player TURNS.**
  - Content-unchanged move targets change 0.29 (squared token distance) when the facing changed vs 2.67 when it did not;
    ring 3+ changes 4.80.
  - contextual_neighbour: false, and I first wrote this up as "contextual tokens refuted". **CORRECTED by E11l: refuted for
    turning only; moving is a different matter.**
- **E11f `targetdrift.py`: the FACED tile (where DO / place act), depth 1, no-scroll steps.**
  - Unchanged cells barely drift (1-4e-5 per cell); the faced tile is not worse in 3 of 4 worlds.
  - teacher s8 alone hallucinates DO effects on the faced tile (0.0049 vs ~2e-5): a seed-specific defect.
- **E11g `occlusion.py`: the tile the player just LEFT.**
  - It is predicted 4-12x worse than the tile ahead after a move, but barely better when a context frame had shown it
    (1.15-1.35x). occlusion_driven: false.
  - Consistent with these worlds using ~only the current frame.
  - Separately, previously occupied tiles are UNDER-represented among the decision-flipping targets (5-14% of missed vs
    16-19% of controls).
- **E11h `scrolldrift.py`: drift added at SCROLL steps?** No: scroll / no-scroll per-cell increments 0.48-1.02x. The drift
  appears once inputs are imagined (~1.5-3e-4 per cell per step vs ~1-4e-5 from true frames).
- **E11i `copyconf.py`: the copy head loses confidence on its own outputs?**
  - No: correct-source weight imagined vs true agrees within 0.004 in all 5 worlds (0.76-0.94).
  - The mixture is soft even on true inputs.
- **E11j `scrolldrift.py --hard`: repeated soft re-mixing (diffusion)?** No: exact argmax copying INCREASES known-cell drift
  (soft / hard 0.60-0.63). So the TRUE tokens of content-unchanged cells must move.
- **E11k: the light change per step?** No: increments 0.00029 / 0.00017 / 0.00030 at light change < 0.001 / 0.001-0.01 /
  0.01-0.03.
- **E11l `truechange.py`: do the true tokens move?** YES (true_target_moves).
  - Measure: deterministic change of the true token of content-unchanged cells (sample-mean change, noise-corrected), / V.

| step | all cells | move targets | ring 2 | ring 3+ | day | night |
|---|---|---|---|---|---|---|
| no scroll | 0.000256 | 0.000287 | 0.000242 | 0.000240 | 0.000294 | 0.000111 |
| scroll | 0.000817 | **0.003696** | 0.000219 | 0.000226 | 0.000762 | 0.001101 |

  - No-scroll: the true change is 0.93x the soft worlds' mean drift increment (0.000256 vs 0.000275).
  - ~~After a move, content-unchanged tiles beside the player change token 16x more than distant ones~~ **CORRECTED the
    same evening.** Splitting the scroll-step targets shows the whole effect is the tile the player LEFT, whose drawn pixels
    change from the player sprite to terrain (the visible-state tile class ignores the sprite):

    | scroll-step cell | true change per cell | cells |
    |---|---|---|
    | behind (the tile the player left) | 0.0132 | 3,658 |
    | ahead | 0.000256 | 3,143 |
    | sides | 0.000199 | 6,898 |
    | ring 2 / ring 3+ | 0.00022 / 0.00023 | |

    There is no special contextual effect at the player's neighbours. E11d's depth-1 neighbour ratio (5.5-9x) likely also
    comes mostly from the left tile (E11g: that tile 4-12x the tile ahead).
  - What stands: true tokens of content-unchanged cells move ~0.00026 per cell per step everywhere (day 0.00029, night
    0.00011), about as fast as the worlds drift. Their source is not established: global-attention context or lighting
    (E11k argues against lighting for the world's increments).

**E11m-E11o: what moves tokens, and what corrupts the DECISIVE tile.**
- **E11m `truecause.py` (exploratory): sleeping steps carry ~73% of the true token motion of unchanged cells.**
  - Per cell: 0.0013 asleep vs 0.00006 awake.
  - The renderer draws the whole map grayscale at half brightness while the player sleeps (renderer.py "Apply sleep").
  - Night, HUD changes, turning and mobs all show SMALLER means.
- **E11n (`scrolldrift.py` sleep split; reading declared): the worlds' known-cell drift is sleep_driven.**
  - Sleeping steps hold 66-72% of the summed increment.
  - Per cell: 0.00136-0.00151 asleep vs 0.00007-0.00010 awake.
- **Sleep does NOT trigger decision failures** (`missedscroll.py` slept provenance; declared reading false in all 5 worlds).
  - 5.5-7.5% of wrong decisions had a sleeping frame earlier, vs 2.4-8.9% of controls.
  - Missed scrolls do follow sleep more (7-11% vs 0.7-1.7%); false scrolls follow it less.
- **E11o `onset.py` (exploratory, re-run reproduced): the decisive tile's corruption is a single-step event, mostly an
  interaction with that tile.**
  - Method: for each first wrong decision and each control, the target tile's world cell is traced back through the
    imagined frames. The onset is the step with the largest error increase.

| onset event | teacher s7 | teacher s8 | suffix s7 | suffix s8 | controls |
|---|---|---|---|---|---|
| faced tile under DO / place, content CHANGED (real consequence missed) | **0.460** | 0.117 | 0.115 | 0.145 | 0.005-0.047 |
| faced tile under DO / place, content UNCHANGED (consequence hallucinated) | 0.016 | **0.271** | 0.111 | 0.088 | 0.014-0.052 |
| DO at the onset step | 0.46 | 0.43 | 0.21 | 0.23 | 0.11-0.13 |
| entered the view | 0.15 | 0.15 | 0.14 | 0.13 | 0.03-0.05 |
| mob within one cell | 0.22 | 0.20 | 0.19 | 0.20 | 0.17 |
| asleep | 0.07 | 0.06 | 0.06 | 0.06 | 0.04-0.06 |
| tile the player left | 0.04 | 0.01 | 0.03 | 0.04 | 0.03-0.05 |
| share of final error added at onset (median) | 0.98 | 0.96 | 0.94 | 0.92 | 0.68-0.83 |

  - The onset is ~3 steps before the decision (wrong 3.0-3.6, controls 3.3-3.5).

**The chain, measured (revised):**
- *Trigger:* the world mispredicts the consequence of acting on the faced tile (DO / place), in one step. teacher s7 misses
  real changes; teacher s8 hallucinates them; the suffix worlds do both. Cells entering the view are a second trigger
  (13-15% vs 3-5%).
- About 3 steps later the player moves onto or through that tile; the move decision reads the wrongly drawn tile and
  flips. Then the view position goes wrong and the trajectory corrupts (steps 1-3 below).
- *Background (not the trigger):* known tiles also drift slowly, mostly on sleeping steps (grayscale rendering). That
  carries 66-72% of the known-cell drift VOLUME but rarely flips a decision.

0. (E11l, corrected) The true tokens of content-unchanged cells move ~0.00026 per cell per step (source not established).
   From true frames the world tracks this (depth-1 error on unchanged cells ~1-4e-5); from imagined frames it tracks
   almost none of it (increment 0.93x the true change). The tile the player leaves changes most (sprite to terrain, 0.0132)
   and is predicted 4-12x worse, but it is NOT over-represented among decision-flipping tiles (5-14% vs 16-19%).
1. Known static content at the tiles beside the player drifts first.
2. The move decision, which reads the target tile, flips (missed or false scroll).
3. The view position goes wrong (46-81% of roots by depth 16).
4. Those trajectories hold 66-91% of depth-16 map error.
5. Randomness is 7-9% of the error; revealed terrain is 14-19% of map excess.

So the lever is keeping known content stable at decision-critical tiles under self-feeding, not stochasticity. E12
(FAIR's rollout loss) trains the predictor on its own outputs, which targets exactly that.

---

## 2026-10-01 evening — E12 first arm (rollout2 s7; not ruled until seed 8 and the decision panels)

corrt_raw_rollout2_s7_u18000, against the 18k references:

| world | one-step all / blocked / moved / idle | depth 1 / 16 | gain k = 2 / 4 / 16 | blockwin w4 / w5 | blocked TF w1 / w4 / w5 |
|---|---|---|---|---|---|
| teacher s7 | 0.149 / 0.301 / 0.142 / 0.442 | 0.035 / 0.622 | 1.17 / 1.10 / 1.01 | 0% / 0% | 0.024 / 0.023 / 0.024 |
| teacher s8 | 0.152 / 0.316 / 0.141 / 0.473 | 0.036 / 0.678 | 1.17 / 1.11 / 1.01 | 0% / 0% | |
| suffix s7 | 0.173 / 0.326 / 0.180 / 0.454 | 0.041 / 0.683 | 1.02 / 1.03 / 1.00 | 0% / 15.5% | 0.025 / 0.025 / 0.122 |
| **rollout2 s7** | 0.213 / **1.648** / 0.189 / 0.484 | 0.060 / 0.720 | 1.02 / 1.01 / 0.99 | **29.4% / 30.7%** | 0.031 / **0.146 / 0.154** |

- rollout2 s7 scrolls on ~30% of blocked moves from TRUE frames: a blocked-move error worse than copying (1.648).
- It is fine with one frame (w1); the damage appears when the current frame sits in slots 1-4. FAIR's random prefix puts
  predicted frames in exactly slots 1-4 during training; slot 0 is always true.
- This generalizes the suffix finding: suffix had predicted frames in slot 4 only and was biased only at w5.
- Reading (consistent with E11c, blocklevel and E11l): in every slot that held a drifted predicted frame during training,
  the move decision learns to discount the drawn target tile (unreliable there) and fall back on the action prior
  ("moves usually succeed").
- Gain at 16 is 0.99: the error stops compounding, but every depth starts worse (depth 1: 0.060 vs 0.035).
- Seed 8, rollout4, driftanat, dpanel and deepeval are running (lane27).

---

## 2026-10-01 — E12: FAIR's multistep rollout recipe, ported exactly (PREDECLARED, not yet run)

Why: Terver et al., "What drives success in physical planning with JEPA world models?" (TMLR 2026, arXiv 2512.24497,
read in full, code read at facebookresearch/jepa-wms 13cf1d9). What it found:
- A k-step rollout loss (their Eq. 5, TBPTT) helps on average from 1 to 2 steps, then hurts in simulated environments;
  DROID's optimum is 6 steps (Fig. 3b).
- Per environment it is not uniform: Metaworld success is 29.7 ± 3.8 at 1 step vs 28.7 ± 5.8 at 2 steps (Table 13).
- The best variant is "2-step 'Last-gradient only' with random initial context"; "what matters is to train the predictor
  to receive as input a mix of encoder outputs and predictor outputs" (App. C).
- Theory (App. D, Remark 1): K raises one-step error δ_K and is conjectured to lower the effective Lipschitz constant
  Λ_K; test error ≤ δ_K (Λ_K^H − 1)/(Λ_K − 1). This is DaD's bound.
- They found V-JEPA-2-AC's official 2-step loss miscomputed (App. C).
- Our `suffix` is not their recipe: fixed prefix (slot 4 only), gradient through the generated frame, frame 4 counted
  twice. Our `selffed` is a longer single-term relative. Their recipe has never been run here.

E11/E11b (below) measured the failure such a loss should address:
- ~90% of depth-16 error is the model's own.
- The first view-position error is a wrong move decision on imagined inputs (missed or false scroll; 86% missed for
  teacher s7, about half and half in the other worlds; corrected the same day), at ~3-4% of remaining roots per step from
  depth 2.
- A loss on the predictor's own inputs targets exactly that.

Arms (tworld.py `--loss rolloutK`): corrt, Raw, full backbone, 18,000 updates, recipe identical to
corrt_raw_{teacher,suffix}_s{7,8}_u18000 except the loss.
- rollout2 at seeds 7 and 8 (their simulated optimum).
- rollout4 at seed 7 (trade-off probe; reported only).
- Port, per their code: prefix t ~ U{0..W−K−1}; the teacher-forced prediction of frame t+1, detached, then K−1 rollout
  steps, each input detached; weights 1/(K+1) for the teacher term and 1/K per rollout step.
- CPU unit test: losses finite, gradients flow; teacher / suffix / selffed bit-identical to before the edit.

Evaluation:
- Measures: teval (w5, w4), blockwin, posprofile, driftanat (position-error rate), stochdiag, dpanel (H1/H2, opened
  55k-56k), deepeval (H16, sealed 62k; these worlds are added to E10 stage 3's sealed set here, before any of them is
  read).
- Comparators: teacher 18k s7 / s8.

Decision rules, fixed now (two-seed rule):
1. rollout2 "**improves self-feeding**" iff, at BOTH seeds against teacher of the same seed:
   - compare.py depth-16 imagined error is resolved lower;
   - driftanat wrong_rate at depth 16 is lower;
   - deepeval gen16 is not resolved lower.
2. rollout2 "**helps the H16 decision**" iff deepeval gen16 (rollout2 − teacher) > 0, resolved at both seeds.
3. Otherwise report as measured. rollout4 is reported only.

---

## 2026-10-01 afternoon — E9 groups 2-3 (head ablation), seed-8 18k replication, a hung summary

**E9 head ablation (lane19c, 6k, seed 7, suffix; same blocks and protocol): the copy head is what makes the one-step
decision readable.**

| head | err/copy h1 | gen1 (zombie) | gen1 − root_rank | transfer1 |
|---|---|---|---|---|
| corrt (copy) | 0.344 | 0.926 (0.912) | +0.194 [+0.171, +0.217] | 0.626 |
| residual | 0.494 | 0.647 (0.502) | −0.085 [−0.107, −0.065] | 0.551 |
| direct (spatial.py T's output) | 0.797 | 0.592 (0.485) | −0.139 [−0.167, −0.111] | 0.465 |

- The direct head's imagined successor reads WORSE than root + action, which explains why T carried no decision.
- Copying the tile from its neighbour keeps the successor's geometry sharp: player beside the zombie, player on lava.

**E9 group 2 (E8 arms and seed 8, 6k):**

| world | gen1 | gen1 − root_rank | gen2 − root_rank |
|---|---|---|---|
| suffix s8 | 0.922 | +0.190 | +0.009 [−0.000, +0.018] |
| teacher s8 | 0.919 | +0.187 | +0.019 |
| selffed | 0.916 | +0.184 | +0.043 |
| noise | 0.792 | +0.060 | +0.015 |

- noise's blur costs 0.127 of gen1 against teacher s7.
- Teacher − suffix on gen1: −0.006 [−0.017, +0.005] (s7 6k), −0.002 [−0.012, +0.008] (s8 6k), so there is no recipe
  effect on the one-step decision at 6k.
- At 18k s7 it is −0.020 [−0.032, −0.007]. Seed-8 18k panel queued (lane19d).
- transfer1 teacher − suffix: +0.066 (s7 6k) but −0.000 (s8 6k), another seed effect.

**Seed-8 18k pair (lane22, rule declared in 8c969101: same sign, resolved, at both seeds):**
- One-step all: suffix → teacher −0.024 [−0.026, −0.022] at s8; −0.024 at s7. **Teacher-only better, attributed.**
- Depth 16: −0.004 [−0.016, +0.009] at s8 vs −0.061 at s7. **Not attributed.**
  - teacher s7 18k (0.622) is the outlier; the other three sit at 0.678-0.683.
  - teacher s7 → s8: +0.055 [+0.024, +0.088].
- blockwin w5: suffix 15.4% (s7) / 15.4% (s8); teacher 0% / 0%.
- Recipe at 18k, on two seeds: teacher-only has better one-step error, no slot bias, and equal depth-16 error.

**Hung summary (operations):**
- The FIT replay wrote all 700 seeds (7,085 roots = the FIT root count) by about 13:19. It then sat in disk sleep in
  its summary, which loaded every depth frame (RSS 9.4 GB under MemoryHigh 8 GB).
- Killed at 14:51 (logged FAILED, exit 143).
- Fixed in 368a60a1: the summary keeps labels only, and deepeval's token cache streams in two passes. The streamed
  cache is bit-identical to the old one on the smoke rows.
- Relaunched; the summary completed in 35 s:
  - FIT opportunity: k=1 0.178, k=2 0.349, k=16 0.679 (noop) / 0.752 (recorded);
  - one-step spread 0.994 (a few random one-step deaths, cf. the replay pilot's 0.17%).
- lane23 (deepeval) started at 14:56.

---

## 2026-10-01 — E9 result (group 1, exploratory on opened blocks 55k-56k): every per-tile world's imagined successor carries the one-step decision; none draws it like a real successor

1,605 H1-opportunity and 3,097 H2-opportunity judge roots (zombie-adjacent 1,038 / 1,435).
- Fit / dev / judge = 2,493 / 1,040 / 3,125 roots.
- Arms are per-root means over 3 head seeds; contrasts are paired, seed-clustered.
- References:
  - H1: prior (DOWN) 0.589, actions_only 0.626, root_rank 0.732, root_tokens 0.737, real 0.999; zombie: prior 0.518,
    root_rank 0.603.
  - H2: prior 0.639, actions_only 0.647, root_rank 0.654, root_tokens 0.668, **real2 0.739**.

| world | err/copy h1 / h2 | gen1 (zombie) | gen1 − root_rank | transfer1 | gen2 | gen2 − root_rank | transfer2 | gen2 − gen1_h2 |
|---|---|---|---|---|---|---|---|---|
| corrt suffix 18k | 0.253 / 0.534 | 0.937 (0.927) | +0.205 [+0.182, +0.227] | 0.658 | 0.711 | +0.057 [+0.048, +0.066] | 0.633 | −0.001 |
| corrt teacher 18k | 0.232 / 0.566 | 0.917 (0.913) | +0.185 [+0.162, +0.210] | 0.643 | 0.702 | +0.049 [+0.040, +0.057] | 0.638 | −0.004 |
| fmamba corrg suffix 18k | 0.247 / 0.532 | 0.925 (0.909) | +0.193 [+0.171, +0.215] | 0.643 | 0.705 | +0.051 [+0.041, +0.061] | 0.644 | −0.004 |
| corrt suffix 6k | 0.344 / 0.567 | 0.926 (0.912) | +0.194 [+0.171, +0.217] | 0.626 | 0.695 | +0.041 [+0.032, +0.050] | 0.639 | −0.023 |
| corrt teacher 6k | 0.274 / 0.580 | 0.920 (0.904) | +0.188 [+0.166, +0.211] | 0.692 | 0.682 | +0.028 [+0.019, +0.037] | 0.625 | −0.033 |

Readings (measured, exploratory):
- **The transition adds on the one-step decision, in every per-tile world: +0.19-0.21 over the same root + action
  read without it (zombie roots +0.31-0.32).**
  - This is the first time in this project. Earlier worlds did not add:
    - spatial.py's T: frozen_heads 0.676 vs root_rank 0.684 on 54k;
    - interface U: 0.693 vs root 0.708 on 55k.
  - It holds across the corrt / corrg heads, attention / Mamba backbones, 6k / 18k, and suffix / teacher.
- **Heads fitted on real successors do not read imagined ones.**
  - transfer1 0.63-0.69 vs gen1 0.92-0.94; the real-fitted head reads real successors at 0.999.
  - The imagined successor carries the consequence, but drawn differently (step-1 err/copy 0.23-0.34).
  - Imagination training must fit its heads on the world's own imagined states, or close this drawing gap.
- **H2 is capped by randomness:** one real step-2 draw reaches only 0.739 against 32-key P(death2). gen2 recovers
  0.028-0.057 of root_rank's 0.085 gap to that ceiling.
- **The second imagined step adds nothing** (gen2 − gen1_h2 −0.001 to −0.033). At 6k it is worse than the first step.
- Recipe contrasts (suffix vs teacher, corrt vs fmamba) are single-seed here. They are not attributed (E8's two-seed
  rule); dpanel --compare runs after group 2.
- Sealed confirmation: E10 stage 3's k = 1 reading on the fresh block 62,000-62,399 (rules committed in 8c969101).

---

## 2026-10-01 — E9/E10: decision panels with action-dependent death (DECLARED before results; exploratory)

Why: the diagnosis futures (teval, 1,002 roots) hold 1 opportunity root in TRAIN and 0 in TEST, so no per-tile world
has ever been judged on a decision. Everything the levers campaign measured is token error and fact probes.
- **E9, H1/H2** (`20260927_levers/dpanel.py`, lane19b): the readout ladder's judgement blocks carry all 17 first
  actions and 32-key P(death1) / P(death2 = within two steps, NOOP second step).
  - Partition: frozen_heads.py's (fit = FIT-train 700 seeds with 32-key P; dev = FIT-dev 350 seeds, realized death,
    selection only). Judge = observe_fresh_v6 + v7 (55k-56k), OPENED blocks, so exploratory.
  - Tokens: raw bridge encoder, verified bit-identical to joint step 10,000 (max |diff| 0.0 over 209 tensors), which
    is the levers Raw token space.
  - Imagination follows teval: step 1 from 4 context frames and the branch action; step 2 from a 5-frame window and NOOP.
  - Heads: frozen_heads.train_head's ranking arms (per-branch attention probe, soft_rank, 3 seeds, dev selection).
    - gen1/gen2: fitted on imagined states.
    - gen1_h2: step-1 state fitted on P(death2).
    - transfer1/2: real-successor heads read on imagined states.
  - References: real1/2, root_rank (no transition), root_tokens, actions_only, prior. Token fidelity err/copy at h = 1, 2.
  - Worlds:
    - corrt suffix 18k, corrt teacher 18k, fmamba corrg suffix 18k (Raw);
    - corrt suffix 6k, corrt teacher 6k;
    - the E8 arms (noise, selffed, suffix s8, teacher s8).
  - Reported, not ruled.
  - CPU smoke (6 seeds per split): end to end; token check err/copy 0.277 at h1 on 14 hazard roots (teval's
    one-step all for the same world is 0.149 on diagnosis roots), so branch actions are aligned.
- **E10 stage 1, H16 label statistics** (`deeppanel.py`, lane19a): the collector's walk and retention on 10 fresh seeds
  (69,000-69,009; seeds 62,000-68,999 stay untouched for sealed blocks).
  - Per root: 17 first actions x 32 key sequences x 16 open-loop steps in the real simulator.
  - Continuations, identical across branches: 15 NOOPs, or the BC policy's own next 15 actions.
  - Measures, per depth k: opportunity rate, within-root spread, prior/oracle safe, and how often the depth-16 argmin
    differs from depth 1.
  - This decides whether an H16 panel carries decision content that H2 does not. No world is involved.
  - **Stage-1 result (83 roots, 10 seeds; small, exploratory):**
    - Sanity checks:
      - k = 1 opportunity 15.7% (judgement blocks 18.6%);
      - k = 1 within-root spread exactly 1.000 (one-step death deterministic, as the replay pilot found);
      - k = 1 prior = DOWN.
    - Opportunity by depth k:

      | Continuation | k=1 | k=2 | k=4 | k=8 | k=16 |
      |---|---|---|---|---|---|
      | noop | 0.157 | 0.289 | 0.253 | 0.747 | 0.675 |
      | recorded | 0.157 | 0.289 | 0.253 | 0.771 | 0.735 |

      Mean P(dead by 16): 0.48 (noop), 0.54 (recorded).
    - At k = 16, of the roots whose death varies over first actions (56 noop / 61 recorded), only 5% / 7% vary at
      k = 1 and 14% / 16% at k = 2.
    - Expected safe at k = 16, choosing uniformly among the actions that minimize a shorter horizon's P:

      | Continuation | By H1 | By H2 | Uniform | Fixed prior | Oracle |
      |---|---|---|---|---|---|
      | noop | 0.663 | 0.665 | 0.662 | 0.718 | 0.764 |
      | recorded | 0.533 | 0.539 | 0.529 | 0.600 | 0.690 |

      The oracle is optimistic: a minimum over 17 noisy 32-key estimates.
    - Reading: H1/H2 knowledge carries almost nothing about the depth-16 decision; an H16 panel is a different panel,
      not an extension.
- **E10 stage 2 (declared before collection):** `deeppanel.py --mode replay|collect`.
  - Deep labels (both continuations, K = 32) for:
    - the FIT-train seeds (lane20a);
    - the FIT-dev seeds (lane20b). Replayed roots are checked against the fork store exactly as observe.py replay
      does (frames within 1, identical one-step deaths).
  - A fresh judgement block, seeds 62,000-62,399 (lane20b, collect mode), untouched until an evaluation with
    predeclared rules is committed.
  - Each row also stores, under the recorded continuation and key sequence 0:
    - the real frames at depths 1, 2, 4, 8, 16 for all 17 branches, and that draw's dead-by-k (checked equal to key
      sequence 0 of the label rollout);
    - the root's last 8 frames and actions;
    - the visible state.
  - Tests before launch:
    - smoke mode reproduces the committed seed-69,000 labels bit for bit (15/15 roots);
    - a one-seed collect (69,010) wrote 10 complete rows, with zero realized-vs-P contradictions and cumulative deaths.
- **E10 stage 3, the H16 evaluation (`deepeval.py`; rules fixed before the judgement block 62,000-62,399 is read):**
  - Data:
    - fit / dev = the replayed FIT-train / FIT-dev deep rows; judge = the fresh block.
    - Kept: roots whose P varies at some judged depth.
  - Continuation: `recorded` (the BC policy's next 15 actions, open loop, identical across the 17 branches).
  - Labels: P(dead by k) over 32 key sequences, k ∈ {1, 4, 16}.
  - Imagination: teval's convention (4 frames, then a 5-frame window), with the continuation's actions.
  - Heads: dpanel.fit_head, 3 seeds.
    - gen_k: fitted on imagined depth-k states.
    - transfer_k: real_k head on imagined states.
  - References:
    - real_k (real depth-k tiles, ONE draw);
    - root_rank_k (root tiles + first action; it does NOT see the continuation);
    - prior_k.
  - Fidelity: err/copy at each depth.
  - Worlds:
    - corrt teacher / suffix 18k (s7, and s8 from lane22);
    - fmamba corrg suffix 18k;
    - corrt teacher / suffix 6k (s7, s8).
  - Declared readings, per world, at k = 16 on judge opportunity roots (paired seed-clustered 95% intervals):
    1. **carries the H16 decision**: gen16 − prior16 > 0 resolved AND gen16 − root_rank16 > 0 resolved.
    2. **usable in imagination at H16**: transfer16 − prior16 > 0 resolved.
    3. Recipe contrasts (teacher vs suffix, corrt vs fmamba) are attributed only if resolved with the same sign at
       BOTH init seeds; otherwise reported as measured.
  - k = 1 and 4 are reported with the same contrasts, not ruled.
- **lane22 (seed-8 18k teacher / suffix pair; declared before results):** the 18k recipe effect counts only if
  suffix − teacher is resolved with the same sign at seeds 7 and 8 (compare.py, depth 16 and one-step all).
- **Sealed confirmation of E9's H1 reading (declared before the 62k block is read):**
  - deepeval at k = 1 is E9's gen1 question exactly: the continuation starts at step 2, and P(dead by 1) is one-step
    death over 32 fresh key sequences.
  - For each world, "**the transition makes the one-step decision readable (confirmed)**" iff gen1 − root_rank1 > 0
    AND gen1 − prior1 > 0, both resolved on the 62k block.
  - deepeval GPU smoke (10 stage-2 rows from seed 69,010 as fit/dev/judge, teacher 18k): end to end; depth
    err/copy 0.225 / 0.316 / 0.553 at k = 1 / 4 / 16.
- **lane19c (E9 head ablation, declared):** dpanel on direct and residual 6k (seed 7, suffix) vs corrt 6k. spatial.py's T,
  which carried no decision (frozen_heads 0.676 vs root_rank 0.684), used the direct output. Reported, not ruled.
- Also committed: `20260929_mamba_integration/memceil.py` / `memceil.json`, the recall ceiling for cells entering view
  (2,000 Raw long-pool windows, 126,000 steps).
  - Scroll rate 0.351. Entering cells are 4.49% of map-cell predictions.
  - Share of entering cells seen within the last N frames: N = 4: 9.2%, 8: 16.1%, 16: 23.3%, 32: 29.3%, 63: 31.6%.
  - Seen AND unchanged (token distance below the no-scroll median): 3.1%, 3.9%, 4.4%, 4.6%, 4.7%.
  - So long-context recall could inform at most ~1.4% of map-cell predictions (0.316 x 0.045); unchanged re-entries
    are ~0.2%.

---

## 2026-10-01 — E8 result: neither recipe improves self-feeding; the suffix harm does NOT replicate (only its slot-4 scroll bias does)

All arms are corrt, Raw, 6k, run in lanes 18a/18b. Comparisons are paired over walk seeds (`compare_e8.log`).
**These intervals are within-model; they do not include init-seed variance.**

| arm | one-step all | blocked | idle | depth 1 / 4 / 16 | gain k = 2 / 4 / 16 | blockwin w4 / w5 | blocked TF w4 → w5 |
|---|---|---|---|---|---|---|---|
| teacher s7 | 0.191 | 0.391 | 0.492 | 0.045 / 0.209 / 0.760 | 1.13 / 1.06 / 1.00 | 0% / 0% | 0.027 → 0.027 |
| teacher s8 | 0.237 | 0.895 | 0.821 | 0.053 / 0.216 / 0.745 | | 0.1% / 0.05% | 0.027 → 0.027 |
| suffix s7 | 0.302 | 0.913 | 0.846 | 0.054 / 0.229 / 0.810 | 1.05 / 1.02 / 1.00 | (18k: 0% / 15.5%) | 0.029 → 0.103 |
| suffix s8 | 0.257 | 0.907 | 0.842 | 0.053 / 0.217 / 0.753 | 1.05 / 1.03 / 1.00 | 0.05% / 11.2% | 0.029 → 0.071 |
| noise s7 | 0.364 | 0.530 | 0.527 | 0.106 / 0.341 / 0.873 | **0.97 / 0.97 / 0.99** | 5.4% / 4.4% | 0.036 → 0.036 |
| selffed s7 | 0.219 | 0.799 | 0.564 | 0.052 / 0.215 / 0.788 | 1.06 / 1.03 / 1.00 | 10.1% / 13.4% | 0.072 → 0.094 |

Rule 1 ("improves self-feeding" vs teacher s7 6k needs all three of: depth 16 lower, blockwin w5 < 2%, one-step not
worse by more than 0.02):
- noise fails all three. Depth 16 +0.113 [+0.090, +0.139]; blockwin 4.4%; one-step +0.174.
- selffed fails all three. Depth 16 +0.028 [+0.004, +0.052]; blockwin 13.4%; one-step +0.029.
- **Neither is adopted.**

Rule 2 ("the suffix harm replicates" at seed 8 needs both depth 16 and one-step all resolved worse):
- One-step all: suffix s8 → teacher s8 −0.020 [−0.022, −0.018].
- Depth 16: −0.008 [−0.025, +0.008].
- **Does not replicate.**

**Correction to the 2026-09-30 audit, Finding 2:**
- What stands, replicated on 2 seeds: the suffix creates the slot-4 scroll bias.
  - blockwin w5: suffix 15.5% (s7 18k) and 11.2% (s8); teacher ≤ 0.05% (s7, s8).
  - Blocked TF error w4 → w5: suffix 0.029 → 0.071; teacher flat at 0.027 on both seeds.
- What is withdrawn: "net harmful for corrt at both budgets, depth 16 included" and "teacher-only better on all
  paired statistics".
  - At 6k, teacher s7's one-step blocked/idle advantage is a seed effect. teacher s8 sits at 0.895 / 0.821, like
    both suffix seeds.
  - teacher s7 learned facing/idle rendering early: imagined moved/blocked ratio 9.6 at 6k, vs 55-75 for the other
    three; every run reaches ~9 by 18k.
  - The depth-16 gap does not replicate at seed 8. The 18k teacher-vs-suffix contrast is single-seed and unproven
    for the same reason.
- Init-seed spread of one recipe: corrt suffix s7 vs s8 differ by 0.057 at depth 16 (−0.072, −0.041); residual
  raw 0.004, corr raw 0.026, residual TC 0.001. One-step all differs by 0.04-0.05 for every pair.
  - The rule-1 tolerance of 0.02 is below that spread.
  - **From now on, a recipe effect needs ≥ 2 init seeds per arm before it is attributed.**

What the arms show (measured, not ruled):
- noise reproduces DaD's L < 1 regime (Venkatraman et al. 2015, below Theorem 1: "predictions converge to the mean"):
  - gain < 1 at every depth (0.97-0.99): the first world here that contracts its own errors;
  - but blurred: moved one-step 0.547 vs 0.196, imagined moved/blocked 4.55 vs true 8.0;
  - GameNGen itself warns that with noise augmentation "small local changes get ignored".
- selffed (DaD-style: L1 from self-generated history to the TRUE next frame):
  - scroll bias at every window (10.1% at w4, where no other arm exceeds 0.05% except noise);
  - is WORSE with more true context (TF w1 0.047, w4 0.053).
  - Self Forcing (arXiv 2506.08009 s3.3) instead matches the DISTRIBUTION of self-rolled videos to real ones
    (DMD/SiD/GAN, post-training). Paired regression to the true next frame asks a deterministic model for an
    average wherever the generated history has drifted on stochastic content (mobs).
- `blocklevel.py` (noise world, clean frames labelled with noise level k):
  - blocked moves scrolled at w4: k = 0: 5.35%, 3: 7.42%, 6: 7.72%, 9: 7.98%; mean logit −4.28 → −3.71;
  - w5: 4.39% → 6.81%;
  - k = 0 reproduces the original exactly.
  - Partial support for "unreliable inputs → action-prior fallback": the label alone moves the logit +0.57, but
    most of the harm (0 → 5.35%) is present at k = 0, i.e. in the shared weights.

---

## 2026-10-01 — E8: self-feeding recipes that expose every slot (PREDECLARED, not yet run)

Why: the depth-2 suffix (V-JEPA 2-AC's T = 2 rollout loss) puts a generated frame only in time slot 4. It creates
the slot-4 bias and is net harmful for corrt (audit, Finding 2; "net harmful" withdrawn 2026-10-01, see E8 result). Every world passes its own errors forward at
gain ~1 (0.94-1.17, teval `gain`), so none corrects its imagined history. The sources' remedies expose every
slot to imperfect inputs:
- GameNGen (arXiv 2408.14837): Gaussian noise on context frames, level bucketed (max 0.7, 10 buckets) and
  embedded; "significantly improved" even with clean inference; without it quality "degrades fast after
  20-30 steps".
- Diffusion Forcing (arXiv 2407.01392) / Dreamer 4 (arXiv 2509.24527 s3.2):
  - independent noise level per timestep in training;
  - context "slightly corrupted" at inference. Dreamer 4 writes τ_ctx = 0.1 while defining τ = 1 as clean;
    both reimplementations read it as mostly clean (edwhu/dreamer4-jax `ctx_signal_tau` 0.9;
    nicklashansen/dreamer4 context at index k_max-1).
- DaD (Venkatraman, Hebert, Bagnell, AAAI 2015, Alg. 1):
  - pair the model's predicted states with the TRUE next states and retrain;
  - Theorem 1: multi-step error is exponential in the horizon for Lipschitz L > 1, linear for L = 1.
- Self Forcing (arXiv 2506.08009): self-generated history with gradient truncated to the current step (frame-wise
  VBench: teacher forcing 78.12, diffusion forcing 80.56, Self Forcing 84.26).

Arms (tworld.py `--loss`; corrt head, Raw tokens, full backbone, 6,000 updates, recipe otherwise unchanged):
- `noise` (seed 7): GameNGen-style per-frame noise with level embedding; clean inference.
- `selffed` (seed 7): teacher L1 + one self-fed prediction per update. Anchor a ~ U{0..3}, target k ~ U{a+2..5},
  generated history without gradient.
- Replication (seed 8): `suffix` and `teacher`.
- Comparator: teacher-only seed 7 at 6k (snapshot of lane 17; the schedule is constant after warmup).

Evaluation (lanes 18a/18b): teval (5- and 4-frame windows), posprofile, blockwin; paired walk-seed intervals over
143 seeds (`compare_e8`).

Decision rules, fixed now:
1. An arm "improves self-feeding" if all three hold against teacher-only 6k: depth-16 imagined error resolved
   lower; blocked moves scrolled on true 5-frame windows below 2% (no slot bias); one-step all not resolved worse
   by more than 0.02 x copy.
2. The suffix harm "replicates" if, at seed 8, suffix is resolved worse than teacher at depth 16 AND on one-step
   all.
3. Anything else is reported as measured, with no adoption.

---

## 2026-09-30 — Audit of the 09-28..30 runs: LDAD move routing, the suffix-induced slot-4 bias, self-feeding

Scope: everything run after db1bfd8e. That is my 09-28 lanes 6-16, and the other agent's 09-29/30 work:
the readout-v2 correction, the stage-2 evaluations and diagnostics, LDAD λ=1 10k, and the patch-token Mamba
integration (Subruns 0-1, motion-carry audit, length-64 preparation). Committed stage by stage: 943f2265,
377c4fdc, 68857605, 8a183340, b44eb8b8, 9da04b45; older uncommitted records in 8f6961ba; then this audit.

**Integrity (all pass).**
- 59/59 eval JSONs agree with their own per-root tensors (<= 1e-5).
- The 7 v2 rescores reproduce every legacy physical and true-fit field exactly (247 fields each, max |Δ| 0).
- Every pinned source matches the committed file (tworld a7e3d3…, teval 7eb04d…, short_train, spatial,
  d4mj/train, d4mj/mamba_recurrence, tc_pool).
- Historical versions were reconstructed exactly:
  - tworld 5e4c2fa2… is the stage-2 pin;
  - teval c6c64bed… is the pre-integration pin;
  - my 09-28 teval, re-run on corrt18k, reproduces its 09-28 JSON byte for byte.
- Long pools: identical ledger and labels across encoders. Raw long-pool tokens equal the six-frame pool's bit
  for bit on 400 overlapping frames.
- `short_train.py` is the levers recipe (held-out rule, sampler seed 11, phase optimizer, constant post-warmup
  LR, suffix loss). 6k → 12k → 18k is one continuous run.

**Run state.**
- The length-64 stage completed 200 of 3,600 updates of its first arm (LDAD1 full, objective 0.140). The
  system was then shut down in an orderly way at 17:40:37 (journal: systemd stopped the unit).
- Its resume checkpoint is intact; the other three arms never started. No long-world result exists.
- This boot the kernel marked the TSC unstable (14.7e9-cycle warp between CPUs; clocksource hpet). PyTorch's
  c10 ApproximateClock asserts on a non-monotonic read at startup (exit 134). It happened once today; a retry
  passed.

**Decision coverage (confirmed from `meta.pt`).**
- The 1,002-root diagnosis panel has 1 death-opportunity root in TRAIN (719) and 0 in TEST (283). Opportunity
  means P(death) varies across the 17 actions over the 4 keys.
- 4 of 1,002 factual futures die within 16 steps.
- Every world comparison since 09-26 measures physical and semantic fidelity, not safe-action choice.

**Finding 1 — the LDAD integration worlds do not predict moves with the scroll copy.**
- `where.py` on the integration worlds, successful moves, interior map tokens (`where_int_*.json`):

| moved, interior tokens | Raw 6k | Raw 18k | LDAD1 6k | LDAD1 18k | LDAD10 6k | LDAD10 18k |
|---|---|---|---|---|---|---|
| weight: self | 0.10 | 0.07 | 0.54 | 0.52 | 0.26 | 0.49 |
| weight: scroll source | 0.87 | 0.83 | 0.19 | 0.13 | 0.49 | 0.22 |
| weight: generate | 0.02 | 0.09 | 0.24 | 0.34 | 0.20 | 0.25 |
| generator alone (x copy) | 2.26 | 6.66 | 1.78 | **0.60** | 4.62 | 1.47 |
| mixture output (x copy) | 0.09 | 0.05 | 0.08 | 0.05 | 0.20 | 0.05 |

- Raw scroll-copies. The LDAD worlds blend the old token with a trained generator: a residual-style route,
  E7's "LDAD tokens make action effects learnable for a head without a copy path".
- Their generator is not starved: LDAD1's standalone move prediction is 0.60x copy, against Raw's 6.66x.
- This is why LDAD's blocked-move rows look "correctly routed" at 6k (target scroll weight 0.019). The copy
  route is barely used at all: even on successful moves the target cell's scroll weight is 0.13-0.19.
- It also explains why λ10's frame-logit AUC falls with training (0.987 → 0.951 → 0.927) while its moves
  improve.
- It replaces the integration note's "why LDAD improves internal routing remains unresolved".

**Finding 2 (E5k) — the time position of the current frame decides blocked moves in the scroll-copy heads.**
- Six-frame training puts predictions at positions 0-4. teval rolls out with a 5-frame window after the first
  step, so the current frame sits at learned time embedding 4.
- Teacher-forced error on TRUE frames, by window w (current frame at position w-1). `w4at4` places 4-frame
  content at position 4 (`posprofile_*.json`):

| world | w1 | w2 | w3 | w4 | w5 | w4at4 | blocked, w4 → w5 |
|---|---|---|---|---|---|---|---|
| corrt 6k | 0.0523 | 0.0519 | 0.0515 | 0.0514 | 0.0645 | 0.0636 | 0.029 → 0.103 |
| corrt 18k | 0.0430 | 0.0426 | 0.0425 | 0.0423 | **0.0586** | 0.0583 | 0.025 → **0.122** |
| corrg 18k | 0.0426 | 0.0423 | 0.0431 | 0.0437 | 0.0656 | 0.0652 | 0.037 → 0.192 |
| fmamba+corrg Raw 18k | 0.0398 | 0.0393 | 0.0427 | 0.0483 | 0.0580 | 0.0579 | 0.098 → 0.142 |
| fmamba+corrg LDAD1 18k | 0.0438 | 0.0431 | 0.0433 | 0.0434 | 0.0447 | 0.0447 | 0.037 → 0.040 |
| fmamba+corrg LDAD10 18k | 0.0554 | 0.0544 | 0.0544 | 0.0544 | 0.0565 | 0.0566 | 0.033 → 0.035 |
| residual 6k (suffix) | 0.1632 | 0.1629 | 0.1630 | 0.1630 | 0.1663 | 0.1661 | 0.056 → 0.058 |
| residual 6k (teacher only) | 0.1587 | 0.1585 | 0.1587 | 0.1590 | 0.1624 | 0.1622 | 0.058 → 0.058 |
| direct 6k | 0.1773 | 0.1773 | 0.1773 | 0.1774 | 0.1796 | 0.1795 | 0.032 → 0.032 |

- Only the heads that decide "did the view scroll" (corrt, corrg, and Raw fmamba+corrg, which scroll-copies)
  degrade, and mostly on blocked moves. The penalty grows with training (corrt 0.103 at 6k, 0.122 at 18k).
- `w4at4` reproduces w5 in every world: it is the position, not the extra frame.
- One true frame predicts as well as four (w1 ≈ w4): these six-frame worlds use essentially only the
  current frame.
- The move decision itself (`blockwin.py`, corrt18k, 1,983 rule-blocked factual move steps, TRUE frames):

| input | blocked moves scrolled | after a DO | not after a DO | mean logit |
|---|---|---|---|---|
| w4 | 0.0% | 0.0% | 0.0% | -10.7 |
| w5 | 15.4% | 31.1% | 0.5% | -6.7 |
| w5dup (content of w4, positions of w5) | 14.8% | 30.0% | 0.3% | -6.7 |
| w5noop (oldest action -> NOOP) | 14.7% | 29.7% | 0.5% | -6.8 |

- Rollouts with a 4-frame window instead of 5, corrt18k (`teval --window 4`, `static.py --window 4`):

| | window 5 (teval default) | window 4 |
|---|---|---|
| teacher-forced error, depths 1 / 4 / 16 (/V) | 0.041 / 0.060 / 0.065 | 0.041 / **0.040 / 0.043** |
| imagined error, depths 1 / 4 / 8 / 16 (/V) | 0.041 / 0.202 / 0.380 / 0.683 | 0.041 / 0.183 / 0.355 / **0.638** |
| static roots: rollouts with any false scroll | 40.8% | **6.1%** |
| static roots: false scrolls per blocked-move step | 33.3% | 2.1% |
| static roots: depth 16 (copy 0.462) | 0.614 | **0.403** |
| generated-fit facts at 16: near tiles / zombie AUC / health R² | 0.722 / 0.651 / 0.639 | 0.770 / 0.617 / 0.630 |

- fmamba+corrg (all roots), window 5 → 4, depth 16: Raw 0.672 → 0.653 (teacher 0.066 → 0.051); LDAD1
  0.615 → 0.629; LDAD10 0.627 → 0.630. The LDAD worlds, which do not scroll-copy (Finding 1), have no
  position penalty.
- This overturns my 09-28 E5j decomposition: nearly all of corrt18k's imagined false scrolls come from the
  slot the current frame occupies, not from imagined-input corruption (1.3% of blocked decisions with window
  4).
- It also confounds the learned-gate audit (MOTION_CARRY.md). Its depth-1 steps use 4-frame windows (blocked
  TNR 1.00); depths ≥ 2 use 5-frame windows. Part of the decline it attributes to generated prefixes is this
  position effect.
- **Cause confirmed: the depth-2 generated suffix.** Position 4 is the only slot the suffix trains with a
  generated last frame. Lane 17 retrained corrt 18k teacher-only: same recipe, seed and budget,
  `--loss teacher`, snapshots every 6k. Paired over 143 walk seeds (`compare_teacher18`):

| corrt | suffix 18k | teacher-only 18k | difference [95% CI] |
|---|---|---|---|
| one step all (x copy) | 0.173 | 0.149 | -0.024 [-0.026, -0.023] |
| moved | 0.180 | 0.142 | -0.038 [-0.040, -0.036] |
| blocked | 0.326 | 0.301 | -0.025 [-0.036, -0.019] |
| imagined depth 4 (/V) | 0.202 | 0.168 | -0.033 [-0.049, -0.019] |
| imagined depth 16 (/V) | 0.683 | 0.622 | -0.061 [-0.092, -0.030] |
| depth 16, both rolled out with window 4 | 0.638 | 0.622 | -0.017 [-0.028, -0.006] |
| position profile, blocked, w4 → w5 | 0.025 → 0.122 | 0.023 → 0.024 | |
| blocked moves scrolled, true 5-frame windows | 15.4% | 0.0% | |
| generated-fit facts at 16: facing / near tiles | 0.883 / 0.722 | 0.940 / 0.762 | |

- At 6k the teacher-only snapshot already wins: one step 0.302 → 0.191 (-0.112 [-0.153, -0.077]), depth 16
  0.810 → 0.760 (-0.050 [-0.076, -0.027]). The schedule is constant after warmup, so a 6k snapshot is
  equivalent to a 6k run.
- The teacher-only 6k profile is flat on blocked moves (0.027 at every position); all transitions carry a
  +8% position-4 bump, like the residual head's +2%.
- So V-JEPA 2-AC's T = 2 rollout loss, as adapted here (generated frame only in slot 4), creates the slot bias
  and is net harmful for corrt at both budgets, depth 16 included.
  **CORRECTED 2026-10-01 (E8 result): the slot bias replicates at seed 8; "net harmful" does not (teacher s8 is
  as bad as suffix on one-step blocked/idle, and depth 16 −0.008 ns). Withdrawn.**
- One seed per arm. The residual 6k pair went the other way at depth 16 (suffix 0.849 vs teacher 0.865*),
  so the effect is head-dependent.
- Consequence for self-fed training: generated frames placed in a fixed slot teach slot-specific decisions.
  Any remedy must expose every position to generated or corrupted inputs.

**Finding 3 — self-feeding is ~90% of depth-16 error in every integration arm.**
- Recursive-minus-teacher at depth 16, TEST roots, 18k: Raw 0.603 of 0.675, LDAD1 0.566 of 0.617, LDAD10
  0.594 of 0.656 (89-92%).
- The best world so far, corrt 18k teacher-only, is 0.622 imagined vs 0.040 teacher-forced at depth 16 (all
  roots): 94% self-feeding.
- The only self-fed training is the depth-2 suffix: at most one generated frame in any training input, while a
  depth-16 rollout's window is entirely generated.
- The length-64 stage keeps that suffix at the end of 64-frame windows: 115,200 generated-suffix terms at its
  dose against 1.44M in the short stage. It extends observed context and cuts self-fed training 12.5x. It
  cannot address long-horizon self-feeding.

**Corrections recorded in place.**
- Same-seed nondeterminism wording.
- carry-smoke figures (the committed JSON reads 0.0).
- E5j false-scroll decomposition (superseded by Finding 2).
- E1c rows for the matched-compute arms (memory rerun, float-level identical otherwise).
- Integration section status.
- Other agents' corrections accepted as correct:
  - readout transfer (v2);
  - identifiability (my "LDAD saturates the ceiling" is retracted);
  - four-key "ceiling" wording;
  - movement labels come from a visible rule;
  - `masked_scan` advances the convolution (its "exact hold" docstring is false; preserved because checkpoints
    pin that source).

---

## 2026-09-30 — Matched 12k/18k outcomes and long-cache preparation

- Raw, LDAD λ=1 and λ=10 `fmamba+corrg` continuations and their 12k and 18k evaluations finished, last evaluation at **15:37 AEST** (`artifacts/eda/levers_logs/lanes.log`). The 18k checkpoints retain source/pool lineage. `20260929_mamba_integration/audit_curve.py` recomputes one-step errors on the same **283 TEST roots / 43 walk seeds** (4,811 labelled actions) and depth-16 errors on the 281 alive TEST roots, with **5,000 paired seed-cluster bootstrap** draws (`audit_curve.json`). Error/copy and `/V` compare within an encoder's latent space; cross-encoder comparisons are descriptive, not control rankings.

  | TEST error ÷ copy | Raw 6k → 12k → 18k | LDAD λ=1 6k → 12k → 18k | LDAD λ=10 6k → 12k → 18k |
  |---|---:|---:|---:|
  | moved (784) | .208 → .202 → **.148** | .166 → .135 → **.126** | .229 → .143 → **.094** |
  | blocked (348) | **5.026 → 1.913 → .696** | .366 → .302 → .295 | .559 → .331 → .284 |
  | all actions (4,811) | .391 → .286 → **.162** | .227 → .192 → .182 | .323 → .221 → .177 |
  | generated depth-16 `/V` | .772 → .720 → .675 | .711 → .651 → .617 | .826 → .722 → .656 |
  | teacher depth-16 `/V` | .125 → .080 → .071 | .063 → .055 → .051 | .111 → .078 → .062 |

  Raw 18k minus 6k blocked ratio is **−4.330 [−9.402, −2.807]**. At 18k Raw blocked is **.696 [.341,1.736]**, so a residual deficit cannot be ruled out. λ1 minus Raw at 18k is −.401 [−1.305,−.126] on blocked actions, but λ1's aggregate ratio is worse by +.020 [.010,.029]. λ10 minus λ1 at 18k moved is −.031 [−.038,−.024]; their aggregate difference is unresolved. Depth-16 generated-minus-teacher `/V` remains **.603 Raw, .566 λ1, .594 λ10**. The training objective contains only a two-step generated suffix, so depth 16 is an out-of-training-depth diagnostic, not an H16 actor result.
- The Raw blocked-move output-routing fault **improves with budget** (`gate_curve.json`): on 1,132 held-out move attempts, action-token moved/blocked AUC rises **.809 → .980 → 1.000**, trained `corrg` frame-logit AUC **.638 → .861 → .987**, and direction-specific target-copy weight on blocked rows falls **.578 → .146 → .050**. Input/target tokens read near 1.000 at 6k. The measured 6k fault is therefore not an immutable inability of this architecture to represent the visible movement rule. Why LDAD changes optimization speed remains unresolved. At 18k generated-fit zombie-near AUC at depth 16 is **.633 Raw, .600 λ1, .610 λ10** (depth 1: .878/.788/.903).
- **Decision-coverage limit:** the current diagnosis panel has **one** one-step death-opportunity root among 719 TRAIN-side roots and **zero** among its 283 TEST roots, where opportunity means actions vary in fatality. It cannot measure within-root safe-action selection or support an actor go. The predeclared fresh all-action seed-65000+ Subrun 3 remains necessary. None of the 18k results is an actor result.
- Length-64 feature-cache preparation exposed two mechanical issues before full collection: the Raw bridge checkpoint needed `read_lewm_bridge`, and its JSON resume contract needed a list-valued shape. Both were corrected; failed zero/eight-row stages were preserved under `raw.failed_*_20260930`. An eight-row smoke succeeded, as did resumption to row 40. The three-arm cache service `lev-mamba-integration-long-pools.service` started at **16:46 AEST** on the sealed **19,789-window** ledger; at 16:51 Raw was active with **6,640/19,789** rows committed. Pool completion and long-world training remain pending.

**18k blocked-move follow-up (2026-09-30):** Repeating the exact frozen wrong-direction-copy intervention on the 348 held-out visible-rule-blocked rows reverses the 6k result: at 18k, removing that copied candidate changes Raw error/copy **0.6945 → 1.0494** (**+51.1% error**); LDAD λ1 **0.2951 → 0.2977** and λ10 **0.2838 → 0.2927** (`blocked_intervention18k.json`). Reconstruction of the frozen head output is exact (max abs 0). Thus the specific wrong-direction-copy mechanism that dominated Raw at 6k has been learned away by 18k; do not carry the 6k explanation forward as the cause of its residual gap.

A separate frozen candidate audit on those same blocked rows (`blocked_candidates18k.json`) locates **97.46% of Raw's residual squared error in map tokens** (HUD 2.54%). Raw's learned map output is **0.7206×** copying the root. A hindsight choice of the best *single normalized* candidate for each tile is **0.2790×** on map tokens and **0.3124×** overall, versus learned output **0.6945×** overall. This shows useful source candidates exist but the learned mixture does not use them optimally on these rows. It is an oracle diagnostic using the true successor, **not** a deployable head or a lower bound on arbitrary mixtures. LDAD λ1's learned mixture is **0.2951×** overall even though its hindsight best single candidate is **0.5711×**: beneficial mixing is material, so this check does not establish a complete causal decomposition or a cross-encoder control ranking. No action-opportunity conclusion follows.

**Length-64 resource and fixed training dose (2026-09-30 17:28 AEST):** All three TRAIN caches completed and published; their 19,789-row ledgers and labels have exactly equal SHA256, and the encoder checkpoint hashes differ. The predeclared warmed full-optimizer resource test selected **batch 16**: B8 median **0.8684 s/step**, 2.115 GB peak allocated; B16 **1.7110 s/step**, 4.125 GB; B24 failed CUDA OOM (B32 untested). B16 has slightly higher teacher-transition throughput (589.1 vs 580.4/s) and satisfies the <=5.0 GiB allocation rule. Fixed long continuation is **3,600 updates** per arm/control, **3,628,800 teacher transitions**, **115,200 generated-suffix terms**, 57,600 common sampler draws and **18,725 unique TRAIN windows**. This exposure matches the short stage's 3.6M teacher transitions approximately; it does **not** match its generated exposure or train depth-16 imagination. `long_resource_steady.json` and the dated `PLAN.md` amendment seal the rule and outcome before fitting.

**Subrun-2 launch (2026-09-30 17:31 AEST):** The source-bound LDAD λ1 full-history long trainer passed a one-step pilot from the exact 18k world+optimizer checkpoint: update 1 objective **0.1943400**, gradient norm **1.4351**, peak allocated **4,121,159,168 bytes**; its atomic optimizer/RNG resume is preserved and will be resumed, not restarted. `lev-mamba-integration-long-train.service` now queues LDAD1 full → LDAD1 reset6 → Raw full → LDAD10 full, all B16/3,600 updates on the sealed shared long ledger, with source/pool/short-checkpoint hashes bound and immutable quarter/half/final snapshots. This is a training launch, not a long-history or actor result; progress and any failures must be read from `artifacts/eda/levers_logs/lanes.log` and the corresponding job logs before interpretation.

---

## 2026-09-30 — Matched Mamba integration at 6k; blocked-move failure localized

- **Run state:** Raw, LDAD λ=1 and λ=10 `fmamba+corrg` worlds and all three 6k readouts finished by 21:51 AEST on 2026-09-29. A length-64 resource smoke finished at 21:52. No long-context trained world, fresh-fork decision screen or actor result exists. Checkpoint, optimizer/RNG, pool and source contracts were verified before continuation. The diagnosis panel has 1,002 roots/143 seeds; 283 roots/43 seeds are held out for the readout. These repeatedly inspected seeds are exploratory, not the fresh Subrun-3 judgment block.
- **Evaluation scope correction:** `teval.py`'s published one-step and rollout aggregates use **all roots**, while semantic probes judge TEST roots. `20260929_mamba_integration/audit6k.py` recomputes physical errors on the held-out roots with 4,000 paired seed-cluster bootstrap draws. Ratios mean error divided by the error of copying the root within each encoder's latent space; comparing ratios or `/V` across different encoders is descriptive, not a control-quality ranking.

  | TEST at 6k | Raw | LDAD λ=1 | LDAD λ=10 |
  |---|---:|---:|---:|
  | moved, 784 actions, × copy | 0.208 [0.190, 0.226] | **0.166 [0.157, 0.174]** | 0.229 [0.220, 0.239] |
  | blocked, 348 actions, × copy | **5.026 [3.164, 11.126]** | 0.366 [0.238, 0.559] | 0.559 [0.420, 0.688] |
  | all, 4,811 actions, × copy | 0.391 [0.330, 0.455] | **0.227 [0.203, 0.249]** | 0.323 [0.297, 0.353] |
  | depth-16 generated `/V`, 281 alive TEST roots | 0.772 | 0.711 | 0.826 |
  | depth-16 teacher `/V`, same roots | 0.125 | 0.063 | 0.111 |

  λ1 minus Raw moved ratio is -0.042 [-0.056, -0.028], blocked -4.660 [-10.642, -2.903], all -0.164 [-0.243, -0.094]; λ1 minus λ10 all is -0.096 [-0.142, -0.059]. Recursive-minus-teacher depth-16 `/V` is 0.647 Raw, 0.648 λ1 and 0.715 λ10 **within each arm**: generated-context error still compounds. Generated-fit depth-1 zombie-near AUC Raw/λ1/λ10 = 0.823/0.805/0.899; depth 16 = 0.591/0.637/0.674. Health R² depth 16 = 0.670/0.665/0.667. The root patch control reads zombie-near at AUC 1.0 for all three on this panel (only 18 positive near-cell labels across 1,132 TEST cells). None of these readouts establishes within-root safe-action choice.
- **Raw blocked-move mechanism:** its absolute blocked error is 0.322 `/V` versus copy's 0.064 `/V` (λ1 0.032 versus 0.088); the 5.03× ratio is not merely a small denominator. It appears in all four move directions (3.75–5.59×); the top 5% of blocked rows account for 16.8% of error. On 1,132 held-out move attempts, Raw target input and backbone target state decode moved/blocked at AUC 1.000 and 0.998, but its action-token state drops to 0.809 and trained `corrg` frame logit to 0.638. λ1's action-token and frame-logit AUCs are 1.000 and 0.999. Raw's direction-specific target-copy weight on blocked moves averages **0.578** versus λ1 **0.019** (λ10 0.016). In `blocked_intervention6k.py`, the frozen forward output reconstructs exactly (max abs 0). Removing just that copy source using the visible-rule blocked label changes Raw blocked error **5.026→1.327× copy**, a **73.6%** reduction; λ1 changes 0.366→0.371. This localizes most of Raw's error to its output routing on these rows, but the intervention uses the visible-rule outcome and is not deployable. The remaining 1.327× error and why LDAD improves internal routing remain unresolved. `gate6k.json`, `blocked_intervention6k.json` and `audit6k.json` preserve the numbers and checkpoint hashes.
- **Literature boundary:** [ITC](https://arxiv.org/html/2605.16457v1) models copy/generation as token correspondence and treats new edge/HUD content separately; our soft local `corrg` head is not its optimal-transport algorithm. [Dreamer 4](https://arxiv.org/html/2509.24527v1) alternates short/long training and warns about fixed start-frame positions; [Po et al.](https://arxiv.org/html/2505.20171v1) evaluate spatial retrieval for a long-context SSM video model. None explains our λ1-versus-Raw contrast or proves six-frame training yields useful 64-frame memory.
- **Next declared check:** the source-bound 6k resumes all read update 6,000 with zero source drift. Raw→λ1→λ10 continuation to 12k and a matching readout started at **08:11 AEST 2026-09-30** (`lane9_short12k.sh`, `lane10_eval12k.sh`; user services `lev-mamba-integration-short12k` and `lev-mamba-integration-eval12k`). The length-64 smoke proved short-prefix equality (max abs 0) and one batch-8 fit (2.09 GB allocated), but its timings are compilation/shape contaminated, so there is no steady throughput estimate. Long training awaits the 18k endpoint and a measured batch/time protocol. Carry transport remains deferred until a long-memory/revisit failure is measured.

**Movement-label provenance correction (2026-09-30):** `20260926_diagnosis/onestep.py::classify` calls `choices.move_table` on the visible root state; the panel does **not** store absolute player positions. Therefore the original “simulator blocked” wording in the registration note and first draft of this entry was wrong. An independent before/after terrain-alignment rule on the held-out key-0 forks agreed on **759/759 classified moved** and **343/343 classified blocked** rows, abstaining on 25 moved and 5 blocked rows (`move_label_crosscheck6k.json`). This supports the visible-rule labels strongly, but is not direct validation against player-position deltas on the 30 ambiguous rows. The intervention and AUC statistics above are conditional on these rule labels; their provenance has been corrected in code and notebook.

The rule is also source-grounded: `craftax_classic/game_logic.py::craftax_step` replaces a sleeping player's action with NOOP, then calls `move_player`; `move_player` uses target bounds, `SOLID_BLOCKS`, `mob_map`, and an explicit lava override before mobs update. `choices.move_table` uses the visible neighbour tile and first three mob channels with the same solid/out-of-bounds and sleep cases; `onestep.classify` explicitly restores lava entries to moved. This explains why the rule should match the game's pre-mob movement decision, but it remains a **derived label**, not a recorded position delta. The two independently aligned views agree wherever terrain makes the shift identifiable.

**Queue amendment (2026-09-30):** The declared 18k endpoint is now staged as `lane11_short18k.sh` followed by `lane12_eval18k.sh`, with user services `lev-mamba-integration-short18k` and `lev-mamba-integration-eval18k`. It waits for all 12k evaluations before consuming GPU, reuses each arm's optimizer/RNG state, refuses existing snapshots, checks the pinned evaluator source hashes and writes completion markers only after every artifact exists. This adds no treatment or changed hyperparameter. Raw's 12k run was observed resuming at update 6,000 and logging update 6,500, objective 0.1678193, grad norm 0.2552, peak allocation 2.55 GB; this is runtime/optimization health, not a performance verdict.

---

## 2026-09-29 — Patch-token Mamba integration (`20260929_mamba_integration/`) — Subrun 2 stopped at update 200 (see the 2026-09-30 audit)

- User clarified the intended treatment: Raw + LDAD λ=1 encoder, patch-token T state, spatial attention plus temporal **Mamba-2** (`fmamba`), `corrg` output, then actor training in imagination. This is not the full-Transformer T backbone. The protocol and stop/diagnostic conditions are in `PLAN.md`; no actor result exists yet.
- Inputs were hashed before the first subrun (`inputs.json`). Raw pool `pool.pt` actual SHA256 matches its manifest: `7662de45ae043ae076d1751c7b34300919ec52f99106546fe39a2ff63ac9b5ac`. λ=1 encoder step-010000 SHA256: `b4c56cff70bcb118a4330d10a6c1e7bf1329ad1e971d908579e2969385ffc7d5`. Raw joint and Raw bridge frozen encoders have 209 identical tensors, max absolute difference **0**.
- Subrun 0 λ=1 per-tile pool completed: **32,647** six-frame windows (24,576 main, 8,071 terminal), SHA256 `410999efe99375a1ecfcda7b3e28a296485f09a866e954871d2866379f7994b0`. Its IDs, actions, reward, alive, health-change and terminal arrays are exactly equal to the Raw source pool (`artifacts/eda/spatial_pool_ldad1_v1/manifest.json`). The matched root-patch fact diagnostic is queued, not yet interpreted.
- Long TRAIN ledger was sealed before treatment results: **19,789** length-64 windows, 12,288 sampled main and all 7,501 eligible terminal endings, 7,755 eligible episodes, 19,739 unique `(episode,start)` windows; SHA256 `cad11d3dac2e90358750a97299f56a2b4ff2eeeeeaaa82f0ef14200ef2a613aa`. Each encoder arm will use this same ledger. A long pool/model has not yet run.
- A strict bitwise six-step resume smoke **failed**: split 3+3 vs uninterrupted 6 max parameter difference `4.7683716e-7`. Diagnosis (`verify_diagnose.json`): split-vs-split repeats also `4.7683716e-7`, uninterrupted-vs-uninterrupted `0`, new-vs-legacy trainer `2.7567148e-7`; sampler RNG states identical. This is a measured numerical floor, not proof of exact reproducibility. The original failed check is preserved. A final-source smoke with a documented `1e-6` mechanical tolerance is running; no 6k world job has started.
- Launcher defect discovered during the failed smoke: under `set -e`, `wait $pid` exited before recording FAILED. `lib.sh` now captures the wait exit status with `wait $pid || code=$?`; a CPU-only deliberate failure returned 1 and wrote the FAILED ledger row. The already-queued root job still waits for the new verified marker.
- Final-source six-step mechanics check passed its amended `1e-6` tolerance (`verify_accept.json`): resumed vs uninterrupted max parameter difference `2.7567148e-7`; new vs legacy trainer `2.7567148e-7`; sampler RNG exactly equal. This is numerical equivalence at the measured GPU floor, **not** bitwise reproducibility. The original stricter failure remains recorded above.
- Matched root patch-token diagnostic on the previously inspected diagnosis TEST split (`root_patch.json`): 283 roots / 43 walk seeds, 575 FIT and 144 validation roots. Raw / λ=1 / λ=10 adjacent-zombie near-cell AUC = **1.000 / 1.000 / 1.000**; there are only **18 positive near-zombie cells of 1,132**, so this panel has little power for hazard-coverage claims. Root health R² = 0.9950 / 0.9974 / 0.9974; λ=1 minus Raw +0.00242, paired seed CI [-0.00065,+0.00588]. Root food R² = 0.9967 / 0.9952 / 0.9981; λ=1 minus Raw -0.00147 [-0.00290,-0.00006]. λ=10 exceeds λ=1 on near-tile class accuracy by +0.00795 [+0.00250,+0.01461]. These near-ceiling root facts show no missing Raw patch zombie-presence fact *on this panel*; they do not establish action-consequence retention.
- Subrun 1 started as `lev-mamba-integration-short6k.service`: Raw, λ=1 and λ=10 use identical `fmamba+corrg` architecture, seed 7, six-frame ledger/sampler, phase optimizer and suffix loss. It stops each arm at update 6,000 with a full optimizer/RNG checkpoint; 18k continuation follows assessment of the matched readouts. The service is monitored and outputs remain source/pool-hash bound.
- Long-run protocol correction made before any long training: `long_world.rollout_loss` teaches 63 factual next frames but only a **depth-2** generated suffix (at frames 61–63). A length-64 scan therefore permits long observed history but does not train H16 imagination; depth-16 is an out-of-distribution diagnostic until a separately declared alias-free generated-depth bridge runs. The λ=1 length control is now a six-frame-reset continuation on identical long windows/targets, plus full-versus-rolling-six evaluation of the same long-trained model. This avoids attributing data-exposure differences to recurrence (`PLAN.md`). [Dreamer 4](https://arxiv.org/html/2509.24527v1) explicitly trains batch length beyond context and alternates short/long batches, but its diffusion/shortcut objective differs; [Long-Context State-Space Video World Models](https://arxiv.org/html/2505.20171v1) reports a block-size-one spatial-retrieval deficit. Neither paper explains our result without local contrasts.

### 2026-09-29 evening — motion-aligned Mamba carry proposal (exploratory, active)

- User proposed transporting **both** Mamba-2 `conv_state` and FP32 `ssm_state` to the new screen slot under the same frame correspondence used by the copy/output route. Code audit: `fmamba` applies independent temporal Mamba scans at **82 fixed slots** (action + 63 map + 18 HUD), six layers; `corrg` adds a scalar move logit to neighbour-copy logits and does **not** output a per-cell flow or transport a recurrent carry. `fcanvas` already approximately aligns input streams in world coordinates using `scroll.estimate`, but its absent-cell `dt=0` leaves the convolution advance (`canvas_diag.json`: held-cell relative RMSE **0.4183**, all-kept max error **7.15e-7**). Thus the new mechanism is related to, but not identical to, the tested canvas variant.
- Factual movement evidence: `scroll.json` on diagnosis futures reports **36.60%** scroll frames; real one-step shift inference matches simulator **99.35%** overall, **97.52%** on moved actions, but generated-frame alignment agreement declines to **83.03%** through depth 16 (`canvas_diag.json`), with missed-scroll rate **43.66%** at depths 9–16. An oracle true-token-shift substitution on frozen fcanvas improved depth-16 error /V from **0.74584 to 0.73529**, paired seed CI on difference **[-0.01697,-0.00498]** (`canvas_oracle.json`): estimator error is real but explains a small fraction of that rollout error.
- Independent visible-terrain registration diagnostic (`20260929_mamba_integration/registration_scope.json`) on the **previously inspected** 1,002-root, 143-seed factual panel: among **16,028** alive transitions, **8,335** move attempts, **5,690** high-confidence scrolls, **2,245** high-confidence blocks, **400** ambiguous. The terrain rule was cross-checked against independently terrain-aligned successors on the existing one-step roots and classified 2,750/2,861 moved and 1,138/1,147 blocked cases, **all classified cases correct**; abstention is explicit and the rule is exploratory, not a sealed estimator. On high-confidence scrolled pairs, overlapping map patch-token MSE is **0.27066 at a fixed screen slot versus 0.04092 after terrain-based one-tile registration**, reduction **84.88%** with seed-cluster CI **[83.66%,85.96%]**. This establishes a large input-stream identity mismatch, not yet a world-error or actor gain. Only **0.62%** of overlapping map cells change zombie occupancy; those pairs' aligned-token error averages **0.2716** versus **0.0396** on zombie-stable pairs, showing that global terrain motion does not register independently moving mobs.
- Pure carry-mechanism CPU smoke passed (`carry_transport_smoke.json`): action/HUD slots held fixed; matched map slots copied to the correct location; new-border conv and SSM state zero; gradients finite; zero-shift four-step `FunctionalMamba2` matches a single scan to max **1.19e-7** in output, **0** conv and **1.86e-9** SSM final-state difference (the committed `carry_transport_smoke.json` is a later rerun, 19:47, that reports **0.0** for all three and for the full layer). Each token/layer carry is **68,608 bytes** in bf16-conv/FP32-SSM mode, or **33.76 MB per sample** over 82 slots × 6 layers, before training activations. This makes resource smoke mandatory before matched training.
- A frozen-weight `fcanvas` versus exact carry-transport equivalence diagnostic is staged after the 6k eval and long smoke (`lane6_transport_equivalence.sh`). It will determine whether a full new training arm is substantively distinct from fcanvas on this setting. No result yet; no claim that moving the carry solves control.
- Primary-source context: [TrajGRU](https://arxiv.org/html/1706.03458v2) learns location-varying recurrent links; [Nilsson & Sminchisescu](https://arxiv.org/html/1612.08871v2) warp propagated hidden estimates and gate unreliable flow; [BasicVSR++](https://arxiv.org/html/2104.13371v1) aligns propagated features; [MGMVFI](https://arxiv.org/html/2608.22861v1) reorders Mamba inputs along motion and explicitly treats unreliable correspondence; [Yang et al.](https://arxiv.org/html/2506.05997v2) find spatial-registration deficits for ordinary RNN/SSM memory. These motivate the mechanism, but no paper tests this exact Mamba-2 `(conv,ssm)` transport in Craftax. MADiff ([arXiv:2409.02638](https://arxiv.org/html/2409.02638v2)) modulates a selective scan with egomotion rather than permuting carries.
- Queue recovery: Raw 6k finished at objective **0.1887997** (snapshot SHA256 `bdf50eb4…`); all user services were externally stopped at **15:56:24** after λ=1 saved update 500. The queue resumed from optimizer/RNG checkpoint at 19:14; no restart from zero. Reused evaluation caches were sampled at root/context/future/all-action successor positions against all three pinned encoders (`cache_audit.json`): Raw max half-token difference **0.0009766**, λ1 and λ10 max **0.0019531**, mean absolute differences below **6.2e-7**, exact half-token agreement **99.42–99.59%**. The strict bitwise audit failed from CPU/GPU rounding, then a documented max-abs **0.002** cache tolerance passed. These are sampled sentinel checks, not a byte-level proof of every cached token. Eval requires the passing audit report.

- Follow-up `20260929_mamba_integration/registration_hazard.json`, same inspected factual panel: on **5,690** high-confidence scrolls, central 3×3 patch-token MSE fixed-slot **0.27363** vs terrain-aligned **0.13041**, reduction **52.34%** [**48.93%, 55.45%**] by seed cluster, smaller than the **84.88%** full-map reduction. Visible zombie occupancy changed in **813/51,210 (1.59%)** central cells and **544/5,690 (9.56%)** scroll transitions. This is a player-local residual, not proof zombies alone cause it. The frozen fcanvas/carry comparator was strengthened to sample 15 each no-scroll, scroll-no-reentry and scroll-reentry from the 1,002 existing roots; its historical model-source SHA is checked by reversing only the later `ldad1` pool registration. GPU evaluation remains queued.
- Learned-correspondence audit: old `corrt` target-tile decision logit had moved/blocked AUC **0.99756** on **1,132** held-out move attempts (`20260927_levers/probe_corrt_raw_suffix_s7.json`), versus old `corrg` shared frame logit **0.79304** (`corrg_probe.json`). AUC does not establish a threshold or generated-prefix reliability. Because `corrt` emits that logit after the final layer at frame t, feeding it to the t+1 carry requires frame-outer/layer-inner scheduling or an early side predictor; current carry prototype uses state-pair shift estimation and does not claim to reuse `corrt`. A learned-gate variant would need a matched `corrt` no-transport control.
- `learned_gate_rollout.py` is queued as `lev-mamba-learned-gate.service` after the transport resource smoke. Frozen historical `corrt`/`fcanvas`; threshold fit on TRAIN-seed true one-step forks only, then TEST true forks and generated-prefix factual-action rollouts (depth 1, 2–4, 5–8, 9–16) against high-confidence visible-terrain movement labels. Compare its move decision to the state-pair shift estimator on the same generated trajectories. This directly tests whether the highly decodable learned gate remains usable in imagination; no result yet.
- First-Mamba-input check (`20260929_mamba_integration/mixer_registration.json`): exact historical Raw `corrt/fmamba` source hash verified; **704** high-confidence factual scroll pairs, **139** seeds. After spatial attention and normalization, map MSE fixed-screen **0.25180** vs terrain-aligned **0.08364**, reduction **66.78%** [**64.76%, 68.70%**] by seed. Center 3×3 is **0.25312→0.17576**, reduction **30.56%** [**25.59%, 34.92%**]. Raw patch tokens on those same pairs reduce **86.01%** map / **53.26%** center. The recurrence-input misregistration is real but much smaller at the player; these are factual features, not world/control gains.
- Prior `20260927_levers/canvas_hold_oracle.json` was rechecked: replacing fcanvas absent-cell convolution advance with a full state hold changed frozen depth-16 /V error **0.745844→0.745757**, difference **-0.000088** with seed CI **[-0.000877,+0.000521]**. This makes that known mismatch unlikely to explain the prior fcanvas rollout deficit. Exact bounded carry can still differ at off-screen reentry; frozen equivalence and resource tests remain queued.
- `20260929_mamba_integration/scroll_hazard.json`: frame-pair shift estimator on the existing one-step simulator panel agrees **98.23%** over **4,008** move attempts; **97.52%** over **2,861** moved, **100%** over **1,147** blocked. On **84** zombie-near roots it agrees **97.62%** over **336** move attempts, including **96.08%** of **204** moved. This is factual correspondence; generated-prefix reliability remains the queued decisive check.
- New-border scope for proposed bounded carry shift, computed from the same high-confidence factual scroll labels: **5,690** scrolls expose **46,126** fresh border map cells, **12.87%** of map slots on scrolled frames and **4.57%** of map slots across all **16,028** alive transitions. Zero carry is Mamba’s existing initial state; off-screen retention would be a separate long-memory design.
- First-layer stage decomposition (`20260929_mamba_integration/positional_registration.json`), same **704** pairs and exact historical Raw `corrt/fmamba` checkpoint source: alignment reduces MSE **85.19%** after learned patch projection, **82.12%** after screen position, **81.21%** after time position, and **66.78%** at actual Mamba input after spatial attention/normalization. The **14.4-point** reduction across the latter stage exceeds the **3.1-point** screen-position change. This measures feature geometry, not performance causality; the carry encodes contextualized features rather than pure terrain identity.
- Carry-transport diagnostic conclusion (2026-09-29 evening): `transport_equivalence.json` used exact historical `corrt/fcanvas` weights/source on 45 existing factual six-frame windows, 15 per no-scroll/scroll-no-reentry/scroll-reentry stratum (candidate counts **308/592/102**). Exact bounded carry versus fcanvas mean relative output RMSE **1.33e-7 / 3.34e-7 / 9.66e-4**; reentry max abs **0.1923**. Prior fcanvas is effectively the same short-window mechanism except rare reentry/off-screen policy. No new world training was run.
- Resource correction: preserve the compilation-contaminated `transport_resource_first.json`; use `transport_resource_steady.json` (cold plus three warmed, full optimizer steps on synthetic six-frame B40). Fixed fmamba **0.7207 s/step, 2.281 GB peak allocated**; explicit transport **3.5459 s/step, 3.274 GB**; **4.92x** slower, **1.44x** memory, fits 6 GB. Prior `corrt/fcanvas` trained 6k in ~83 min vs prior `corrt/fmamba` ~72 min.
- `learned_gate_rollout.json` and hashed `learned_gate_rollout_rows.pt`: TRAIN-only threshold **-1.5254**, TEST true one-step n=**1,132**, AUC **0.9995**, accuracy **99.20%**. Generated-prefix TEST factual actions, high-confidence visible-terrain movement labels n=**2,258**: learned gate **78.65%** vs generated-frame shift detection **68.16%**, paired seed gain **+10.50 pp [7.19,13.86]**. Depth 9–16 n=**1,161**: **73.56% vs 57.62%**, gain **+15.93 pp [10.80,20.92]**; gate blocked specificity **45.19%**, AUC **0.713**, moved prior **70.46%**. The learned cue is better but still misroutes many blocked moves in deep imagination; no carry-training/actor gain is established.
- Decision: **no explicit bounded-carry 6k–18k retrain now**. The short-window mechanism is nearly identical to the prior `fcanvas` arm while its straightforward step implementation is 4.92x slower. Reconsider a distinct long-context/reentry design only if the ongoing main fmamba integration localizes a cross-scroll memory bottleneck.
- Live-run handling: matched λ=1 short6k service was paused only after its atomic **update-4,000** world/optimizer/RNG checkpoint was read back; GPU motion equivalence, resource and learned-gate diagnostics completed in that interval. The unchanged `lane3_short6k.sh` was relaunched as `lev-mamba-integration-short6k-resume2.service` and logged `resume ... update: 4000` at ~20:07 AEST. λ=1→λ=10→6k evaluations/long smoke queue remains active; no treatment result was used to alter training. The three now-obsolete queued motion services were stopped to avoid duplicate GPU work.

---

## 2026-09-29 — Readout-transfer correction (existing levers results remain historical)

The levers evaluator originally fitted every fact ridge on TRUE successor tokens and applied it to imagined
tokens. That conflated generated-state information with readout transfer. Corrected teval.py now records a
versioned report with the historical true-fit readout, the same-capacity readout fitted on each world's
GENERATED TRAIN-seed factual trajectories, and a separate all-17-action generated-fit one-step readout.
Validation remains on a disjoint fifth of TRAIN seeds; judgment remains on TEST seeds. Legacy JSON and per-root
files are preserved; version-2 reports use a __readout_v2 suffix when the old name already exists.

On the 18k corrt world, the corrected run
(20260927_levers/evals/corrt_raw_suffix_s7_u18000__readout_v2.json) reproduces the old physical metrics:
one-step all 0.173 x copy, depth-16 imagined 0.683 /V. Adjacent-zombie AUC on imagined tokens is:

| readout fit | one-step moved, all actions | factual rollout depth 1 | depth 8 | depth 16 |
|---|---:|---:|---:|---:|
| true states (historical transfer) | 0.799 | 0.842 | 0.581 | 0.530 |
| generated states (matched) | 0.799 | 0.889 | 0.649 | 0.605 |

The independent matched control (20260926_mechanism_audit/generated_fit.py/json) found a depth-16
generated-minus-true gain of +0.076, 95% walk-seed-cluster interval [+0.015,+0.141] across 43 seeds and
281 alive TEST roots. Its depth-16 generated-fit AUC was 0.606; the v2 evaluator gives 0.605 from minor GPU
numerical variation. At depth 16, all-cell zombie AUC rises 0.499 -> 0.651, health R² 0.590 -> 0.639.
This is a resolved readout-transfer penalty on an already-inspected diagnostic block, **not** proof of
missing information or action control. The physical error 0.683 /V recursive versus 0.065 /V teacher-forced
and the ground-truth substitutions independently establish rollout drift. Older fact-score comparisons in
this notebook remain explicitly true-fit-transfer scores until each arm has a v2 report.

## 2026-09-27..28 — Levers campaign (`20260927_levers/`) — closed 2026-09-30 (direct-head seed 8 and the 36k corrt curve were deferred and never run)

Questions from the user:
- (b) Can training the Mamba at its deployment length fix it, and is its memory used?
- Does discretization help?
- TC + T?
- Delta/copy variants?

| id | question | script | key parameters | status / result |
|---|---|---|---|---|
| E1 | training length L | `context_length.py` | frozen Raw latents; fresh canonical Mamba; L = 4/16/64, B = 128/26/6 (~384 transitions/update); joint optimizer, 10k updates, no SIGReg | done (see below) |
| E1b | continue L4 at L16/L64 (Stuffed Mamba's remedy); L16 at 3x budget | `context_cont.py` | E1 recipe; continuation 5,000 updates, fresh AdamW; L16x3 30,000 updates | done |
| E1c | what the memory helps predict | `memory_use.py` | 256 held-out 128-frame windows; benefit by transition type | done |
| E1d | L64 at 24 windows/update (length or diversity?) | `context_cont.py` arm L64b24 | 10,000 updates, 1,512 transitions/update (DRAMA: 2,048) | done: diversity, not length |
| E1e | matched compute for E1d | `context_cont.py` arms L4b512, L16b100 | same 1,500-1,536 transitions/update | done |
| E1f | canonical shape (4-frame world, then long continuation, as the H2 bridge) with E1d's diversity | `context_cont.py` arm L4to64b24 | continue L4 at L=64, 24 windows/update, 5,000 updates | done |
| E2 | discretization | `codebook.py`, `tworld.py --head categorical`, `teval.py --snap` | k-means K = 1024/4096 on LN tokens; categorical CE, teacher-forced; snap = nearest code after each imagined step | done; contextual-token categorical arm fails (see E2) |
| E3 | TC + T | `tc_pool.py`, `tc_equiv.py`, `tworld.py --pool tc` | identical windows re-encoded by TC; representational tests; residual/direct T on TC vs Raw tokens | seed-7 arms and seed-8 residual done (see E3c); direct seed 8 deferred |
| E4 | Delta-JEPA LDAD in joint | `ldad_joint.py`, `ldad_eval.py` | canonical loop + CE(D(z_{t+1}-z_t), a_t), MLP 192-256-17; paired init verified | lambda-10 Raw/TC/no-SIGReg and Raw lambda 1 done and rescored |
| E5 | copy / Delta variants on T | `tworld.py --head {direct,residual,gated,corr}` | spatial.World backbone; 6,000 updates, batch 40, AdamW 1e-4 wd 0.01, 1,000 warmup; L1 teacher + depth-2 suffix | seed-7 heads done; corrt 18k done; 36k learning curve deferred |

**Source checks, 2026-09-28 (what each running arm is, and is not, relative to its paper):**
- **E4 is LeWM + LDAD, not Delta-JEPA as published.** Checked in the PDF (`third_party/papers/2606.31232v1-deltajepa.pdf`):
  - Eq. 7 is L = L_pred + lambda L_action, "no ... distribution-matching regularizers". There is no SIGReg; LDAD
    is the anti-collapse term, and lambda = 0 "nearly collapses".
  - E4 keeps SIGReg 0.09 and adds LDAD.
  - Their N = 5 action queries decode the 5 actions of one latent step (le-wm `frameskip: 5` in all four of their
    environments, `config/train/data/*.yaml`). Craftax has one action per step, so single-step decoding is the
    faithful analogue; our MLP decoder for one discrete action is a stated reduction of their 3-layer Transformer.
  - **E4c** (lane 8, `ldad_joint.py --no-sigreg`): Delta-JEPA as published, Raw, lambda 10, paired batches.
    `ldad_eval.py` now also reports z's spectrum (effective rank, spread), backfilled for every run: collapse is
    the question without SIGReg.
- **E2 is not Dedieu et al.'s discretization** (arXiv 2502.01591, HTML read 2026-09-28).
  - Their Nearest-Neighbour Tokenizer codes each RAW 7x7 pixel patch independently: K = 4096, threshold 0.75,
    codes frozen once created. Targets are stationary and non-contextual.
  - E2 quantizes contextual ViT tokens, whose codes flip on 14.6% of unchanged cells (`codebook.json`).
  - `catdiag.py` (lane 7) measures how much of the categorical world's loss is those flips.
- **E6 Lalt's 1:1 ratio is ours.** Dreamer 4 (p. 14): "many short batches and occasional long batches, and
  finetune the model on only long batches afterwards"; long batches (256) exceed the context (192). No ratio is
  given, and neither vendored reimplementation has alternation. Lalt tests the regime hypothesis; it is not a
  Dreamer 4 replication. E1f (L4 then long only) is the finetune half.
- **E6 ceiling (`bdiag.json`)**, closed-form ridge, held-out R^2 of dz:
  - 3 frames + 3 actions 0.101; + frames 3-15 back 0.114; + mean of 16-63 back 0.113.
  - Linear predictability gains only +0.013 from longer history, while E1's measured memory benefit is +18-29%:
    what the Mamba uses from memory is mostly nonlinear. The arms that decide E6 (L4chop, Lalt) are queued.

**Four-key simulator reference for imagined facts (`ceiling.py`, `ceiling.json`), 2026-09-28.**
- The futures roll the same factual actions from the FULL root state under 4 other keys. Predicting sample 0's
  facts from samples 1-4 by their sample mean or mode gives an empirical full-state reference. Four samples do
  not define a proven optimum or upper bound; the old JSON field named ceiling is retained for provenance.
- Rows and labels are exactly teval's (test-seed roots alive at depth k).

| depth 16 | four-key reference | one simulator key | copy root | corrt 6k imagined, true-fit |
|---|---|---|---|---|
| tile near player (acc) | 0.947 | 0.939 | 0.625 | 0.710 |
| zombie, all cells (AUC) | 0.788 | 0.706 | 0.689 | 0.562 |
| zombie, 4 adjacent cells (AUC) | 0.802 | 0.700 | 0.684 | 0.500 |
| health (R^2) | 0.872 | 0.808 | 0.588 | 0.585 |
| food (R^2) | 0.975 | 0.966 | 0.880 | 0.881 |
| facing (acc) | **1.000** | 1.000 | 0.327 | **0.310** |

- These numbers compare a direct simulator fact estimator with a true-fit probe transferred to generated tokens.
  The gap cannot be assigned wholly to world error. The generated-fit control below measures part of the
  transfer penalty; token-space error and ground-truth substitution separately demonstrate rollout error.
- Facing is deterministic given the actions (ceiling 1.000 at every depth), yet corrt reads it at copy level
  (0.661 at depth 1; 0.18 on one-step moves). `where.py` localizes this.
- Craftax renderer (`craftax_classic/renderer.py`, read 2026-09-28):
  - the map view is a slice centred on the player (it scrolls);
  - the player sprite (texture = direction, or asleep) is alpha-blended at the fixed centre over the terrain
    under the player;
  - the 2 HUD rows never scroll.
  - corrt adds ONE frame-level move logit to the neighbour logits of every token, the player tile and the HUD
    included.

**E6 result: why (b) does not beat (a) at matched compute (`bdiag.py`, `bdiag.json`, vs `context_cont.json`).**
- Error x copy under each world's own 3-frame window (its short-context competence):

| positions | (a) L4b512 | L4chop | (b) L64b24 | gap from correlated data | gap from long-sequence training |
|---|---|---|---|---|---|
| 4-7 | 0.523 | 0.676 | 0.850 | +0.153 (47%) | +0.174 (53%) |
| 16-31 | 0.586 | 0.754 | 0.936 | +0.168 (48%) | +0.182 (52%) |
| 64-127 | 0.692 | 0.916 | 1.179 | +0.223 (46%) | +0.263 (54%) |
| 128-255 | 0.715 | 0.954 | 1.226 | +0.239 (47%) | +0.272 (53%) |

- L4chop is L64b24's exact sampled segments cut into 21 consecutive 4-frame windows each (same data, short
  training). So:
  - 46-48% of (b)'s short-context deficit comes from seeing only 24 episodes per update instead of 512
    independent windows;
  - 52-54% comes from training on 64-frame sequences, on identical data.
- (b)'s memory benefit (+24-29%) almost exactly cancels that deficit, hence the tie.
- **Lalt** (L4 x 512 and L64 x 24 alternating 1:1):
  - window 0.624 / 0.696 / 0.842 / 0.877 (better short competence than (b)), but memory benefit only +3-6%;
  - full history 0.605 / 0.661 / 0.794 / 0.829, the same as L4to64b24 (0.596 / 0.646 / 0.777 / 0.824);
  - imagination after 48 frames 0.405, vs L4to64b24 0.400 and L4b512 windowed 0.406.
- Three recipes, one ceiling (~0.40 /V at depth 16). At this compute, on data where exact revisits are under 1%
  of transitions, memory buys no imagination gain for the z-Mamba world.
- Linear ceiling for context: longer history adds only +0.013 R^2 of dz.

**E6c: the literature's length-generalization fix, on our (a) world (`statepass.py`, `statepass.json`).**
- Source: Buitrago Ruiz & Gu 2025, "Understanding and Improving Length Generalization in Recurrent Models"
  (arXiv 2507.02782; PDF read, `third_party/papers/2507.02782-length-generalization-recurrent.pdf`, sha256
  bc87fb04c91185d236ffe865c4a2789551b5c03f5bfbc737715e42a4dde8a790).
  - "Unexplored states hypothesis": models trained on short contexts never visit the states reachable later.
  - State Passing (s4.4) is 100 post-training steps at lr / 10, with the initial SSM state = a final state of the
    previous batch, zeroed with p = 0.1. It fixes Mamba-2 2k -> 128k without hurting in-context performance.
    TBTT (s4.5) is about as good.
- Ours: E1e's L4b512 world, 500 post-training updates, 512 x 4-frame windows, lr 5e-6. Error x copy, full
  history / own 3-frame window:

| arm | 4-7 | 16-31 | 64-127 | 128-255 | imagination 16 (recurrent / window) |
|---|---|---|---|---|---|
| L4b512 (base) | 1.74 / 0.52 | 4.68 / 0.59 | 9.14 / 0.69 | 11.35 / 0.72 | 2.076 / 0.406 |
| post-training, zero state (control) | 1.73 / 0.52 | 4.69 / 0.58 | 9.18 / 0.69 | 11.41 / 0.71 | 2.092 / 0.416 |
| State Passing | 1.70 / 0.77 | 3.12 / 0.86 | 4.79 / 1.07 | 5.62 / 1.12 | 1.839 / 0.685 |
| TBTT | 0.98 / 1.59 | 0.93 / 1.81 | 1.94 / 2.39 | 2.42 / 2.60 | 1.002 / 1.277 |
| (b) L4to64b24 | 0.60 / 0.72 | 0.65 / 0.78 | 0.78 / 0.97 | 0.82 / 1.03 | 0.400 / 0.650 |

- **At this budget it does not transfer.** State Passing halves the long-context blow-up (11.35 -> 5.62x copy)
  and costs short-context accuracy (0.52 -> 0.77). TBTT reaches 2.42x but wrecks short context and zero-state
  starts (2.76x at positions 1-3). Neither approaches (a)-windowed or (b) (~0.40). The control rules out the
  post-training itself.
- **E6d: why (`statenorm.py`, `statenorm.json`).** Their mechanism is that post-trained states stop drifting
  past the training length (their Fig. 4). Mean SSM state norm relative to position 3, over 256 DEV/FINAL
  128-frame windows, full-history recurrence:

| world | pos 4 | 8 | 16 | 32 | 64 | 127 | error x copy at 127 |
|---|---|---|---|---|---|---|---|
| L4b512 (base) | 1.32 | 2.14 | 2.82 | 3.55 | 4.31 | 4.96 | 8.12 |
| + State Passing 500 | 1.31 | 2.08 | 2.70 | 3.33 | 3.97 | 4.46 | 4.21 |
| + TBTT 500 | 1.32 | 2.12 | 2.80 | 3.48 | 4.17 | 4.72 | 1.81 |
| (b) L4to64b24 | 1.31 | 2.25 | 2.97 | 3.45 | 3.88 | 4.50 | **0.70** |
| L64b24 | 1.35 | 1.94 | 2.29 | 2.53 | 2.70 | 2.86 | 0.74 |

  1. The mechanism did not take effect: after State Passing the state still grows 4.5x from position 3 to 127
     (4.96 -> 4.46). The error gain is partial adaptation, not stationarity.
  2. Growth itself is not the failure: (b) grows the same 4.5x and predicts well at 127 (0.70x copy), because it
     trained on those states.
  3. Scale of the unexplored region: they post-train models trained on 2k tokens; ours trains on 3 steps, where
     the state is at 1/4.96 of its position-127 norm. Layers 2 and 3 grow most (9.4x, 7.2x at 127). Covering
     that takes training on long-context states, which is what (b) does.
  - The hypothesis holds on our data; the cheap remedy does not reach a gap this large. **For d4mj: the
    invariant (C = trained context) stands; a recurrence usable past 4 frames requires long-context training
    (option b), not a post-training patch.**

**E2 result and diagnosis (`evals/categorical_raw_teacher_s7_K4096.json`, `catdiag.py`, `catdiag.json`).**
- The categorical world (K = 4096 codes over contextual Raw tokens, teacher-forced CE, 6k) is worse than the
  residual world on every one-step class: all 0.840 x copy, idle 1.99, moved 1.04; depth 16 imagined 1.018 /V
  vs residual 0.849.
- It fails for three measured reasons:
  1. **Target noise.** 80.7% of code changes are contextual flips: the cell's tile and mobs are unchanged but its
     code changes. That is 250,133 flips vs 59,714 content changes, 23% of all cells. The categorical world
     predicts 17.9% of flips. In its own currency it is below the continuous residual world read through the
     same codebook: code accuracy 0.664 vs 0.767 (teacher-forced control), and 0.873 vs 0.967 on cells whose
     code does not change.
  2. **Quantization floor** (the snapped TRUE next frame), x copy: idle 1.60, blocked 1.25, interact 0.69,
     moved 0.077, all 0.293; 0.045 of variance at every depth. It explains most of idle's 1.99 and blocked's
     1.54, not moves.
  3. **No copy path.** On moves it gets 1.2% of content cells right (residual 5.3%, corrt 31.2%). The categorical
     head is a direct head, the same failure as the continuous direct head (0.971 x copy on moves).
- **So E2 as run does not answer "does discretization help".** It combined contextual codes (Dedieu et al. use
  stationary, non-contextual pixel-patch codes) with a copy-less head. ITC is discrete tokens WITH a copy path.
- **Rollout-state discretization** (`teval.py --snap`): every imagined frame snapped to its nearest K = 4096
  code before being fed back. Imagined error / V:

| world | one step all (x copy) | depth 1 | 4 | 8 | 16 | health R^2 at 16 |
|---|---|---|---|---|---|---|
| residual, suffix | 0.596 | 0.168 | 0.425 | 0.605 | 0.849 | 0.555 |
| same, snapped | 0.839 | 0.206 | 0.473 | 0.691 | 1.013 | 0.475 |
| residual, teacher-forced | 0.544 | 0.159 | 0.421 | 0.607 | 0.865 | 0.625 |
| same, snapped | 0.788 | 0.197 | 0.468 | 0.686 | 1.015 | 0.473 |
| direct, suffix | 0.902 | 0.179 | 0.452 | 0.667 | 0.969 | 0.109 |
| same, snapped | 1.133 | 0.215 | 0.498 | 0.734 | 1.058 | **0.515** |
| categorical (trained on codes) | 0.840 | 0.207 | 0.487 | 0.702 | 1.018 | 0.498 |

  - Snapping hurts every continuous world at every depth, and does not damp inherited error: the residual
    world's per-step gain goes 0.982 -> 0.993.
  - Trained or post hoc, discretization over these codes ends ~0.15 /V worse than the continuous residual world
    at depth 16.
  - The one gain is health readability for the direct head (0.109 -> 0.515): snapping projects its drifting HUD
    tokens back onto valid codes. It repairs a head that corrupts the HUD; it does not stabilize dynamics.
- Paired intervals (`compare.json`, `compare_disc`):
  - categorical vs its teacher-forced residual control: worse on every statistic, all resolved (depth 16
    +0.154 [+0.135, +0.171]).
  - residual, suffix vs teacher-only: teacher-only is better one step ahead (idle -0.278*, blocked -0.489*,
    all -0.053*); the depth-2 suffix buys depth 16 (0.849 vs 0.865*).
  - direct head, TC vs Raw tokens: all -0.270*, almost entirely sleep onset (0.750 -> 0.082*); moves -0.021*;
    other classes and depths 4-16 unresolved.
- **E2 verdict:** discretizing CONTEXTUAL tokens does not help, at training or at rollout. The cause is measured:
  81% of code changes are context flips, and each snap adds the 0.045 /V quantization floor. A fair test of
  discretization needs non-contextual codes (Dedieu et al.'s NNT on pixel patches, which is not a JEPA state) or
  discrete tokens with a copy path (ITC). Not pursued further on JEPA tokens.

**E5g: 6,000 updates under-trains the per-tile worlds (`compare.json`, corrg 6k vs 18k, same seed and recipe).**

| | corrg 6k | corrg 18k | corrt 6k |
|---|---|---|---|
| moved (x copy) | 0.203 | 0.176* | 0.196 |
| blocked | 4.612 | **0.374*** | 0.913 |
| idle | 0.840 | 0.459* | 0.846 |
| all | 0.350 | **0.178*** | 0.302 |
| imagined, depth 16 (/V) | 0.839 | **0.703*** | 0.810 |

- Every difference is resolved, and all far exceed the seed-noise floor (all 0.380 vs 0.329 across seeds).
- **corrt at 18k** (`evals/corrt_raw_suffix_s7_u18000.json`), best world so far:
  - all 0.173, moved 0.180, blocked 0.326, idle 0.454, sleep 0.025;
  - imagined 0.041 / 0.202 / 0.380 / **0.683** at depths 1 / 4 / 8 / 16;
  - facing 0.940 at depth 1, 0.868 at 16.
  - corrt and corrg converge at 18k (corrg 0.178 / 0.703): the target-tile gate bought speed, not a different
    endpoint.
  - Historical depth-16 true-fit-transfer facts: tiles 0.72, all-cell zombie AUC 0.50, health R² 0.59.
    The four-key simulator reference is 0.95 / 0.79 / 0.87, but it is not a proved ceiling or a matched
    decoder. See the generated-fit correction below.
  - The logged training objective (single batches) flattens after ~6k while held-out accuracy improved 6k -> 18k,
    so it cannot say whether 18k still under-trains. E5i (lane 15, after stage 2): corrt to 36k with held-out
    snapshots every 6k.
- Plain attention learns the move routing with 3x the updates; corrt's target-tile read only got there faster.
- **Every head comparison so far was made at 6k.** Rankings may change with budget. corrt 18k and the best fix
  at 18k are queued (lane 10); stage-2 backbones need the same check.
- Same-seed runs are not bit-reproducible. **Corrected 2026-09-30** (an earlier line here said they "match
  exactly to update 2,000"): corrg 6k vs 18k logged objectives already differ in the 6th digit at update 500,
  the 4th at 1,000, and by 12% at 3,000 (0.2847 vs 0.2533); corrt 6k vs 18k likewise. GPU kernels are
  nondeterministic from the first updates.

**18k head choice, generated-fit rescore (2026-09-29).** Existing Raw corrg18k and corrt18k
checkpoints reproduced all 103 shared historical numeric fields exactly under `teval.py` v2.
At depth 16, generated-fit adjacent-zombie AUC is .598 for corrg versus .605 for corrt;
all-cell zombie AUC .660 versus .651, health R² .603 versus .639, and facing accuracy
.875 versus .883. Physical error remains .703 versus .683 /V, respectively. No paired
semantic interval is available. Thus corrg is a simpler routing head, but it has no
measured performance lead over corrt at the adequate 18k budget; choosing it for an LDAD
integration is a design choice, not a demonstrated improvement.

**E5f: where corrt's error lives (`where.py`, `where_corrt_raw_suffix_s7.json`), one step, futures roots.**
- **Refuted suspect:** the shared move logit does NOT force copying at the player tile or HUD. On moves the
  player tile keeps itself (w_self 0.957), and the HUD too (0.999).
- **The generator is starved.** Its mixture weight is 0.000-0.062 in every group and class except sleep onset.
  Its standalone output, LN(proj(h)), in x copy:
  - sleep onset (w_generate 0.73): 0.19-0.28;
  - moves: 3.4-5.8 on the map, 256 on the HUD;
  - blocked: 12.6-60; idle: 54-315; interact: 4.7-853.
  - It learns only where it is chosen, and is chosen only where it learned.
- **Entering row/column on moves: 57.4% of all moved error**, x copy 0.908.
  - Its weight goes 0.656 to the scroll source, which is off-grid zero padding, 0.278 to self and 0.018 to the
    generator.
  - Other border cells: x copy 0.076.
  - **CORRECTED 2026-09-28:** the head layer-normalizes its mixture, so the zero-pad weight only RESCALES. The
    effective content is LN(self + ~0.07-0.12 x generator). Self at an entering cell is the root token at that
    screen position, which after the scroll is the in-view neighbour's content: a replicate pad.
    - Measured: zero fill would cost 3.66x copy error at entering cells.
    - The new cell's tile class equals its in-view neighbour's in 75.0% of 22,951 entering cells.
    - So entering-cell error is mostly partial observability: a deterministic world cannot know the other 25%
      without map memory or stochastic generation. The substitution result (-34% at depth 16) measures the
      price of that, not a fixable head bug.
- **HUD never updated:** x copy 1.000 in every class (w_self 1.000). It is 34% of interact error and 36% of
  sleep-onset error. This is why health and food sit at copy level.
- **Rollout:** interior error grows from 0.029 of its variance (k1) to 0.81 (k16), and its share of error from
  23% to 42%. Consistent with entering-cell errors scrolling inward; not yet shown.
- Residual head for contrast: no localized failure. Its error is spread in proportion to copy error
  (interior 0.906, edge 0.921 x copy on moves).
- **ITC does both things we lack** (verified in the PDF, `2605.16457v1`):
  - "leaves the transformer and its training loss unchanged": token predictions are trained on every token, and
    copying is decided at decoding;
  - Craftax: "the optimal transport output [is applied] to the central region of the screen ... the
    transformer's predictions for the screen edges and inventory regions".
- **Fix arms (lane 10):** `--gen-loss` (the generator's own teacher-forced L1 on all tokens), `--regions itc`
  (ring and HUD from the generator), both; at 6k, then both and plain corrt at 18k.

**E5h: what causes corrt's depth-16 error (`substitute.py`, `static.py`, `teval.py --hard`), 2026-09-28.**
- **Ground-truth substitution in imagination.** After each imagined step one component is replaced by the true
  tokens; the error is read before the replacement. Imagined / V at depth 16:

| substituted | total | player | near | interior | edge | HUD |
|---|---|---|---|---|---|---|
| none (= teval) | 0.810 | 1.107 | 0.878 | 0.808 | 0.823 | 0.699 |
| entering row/column | **0.536** | 1.101 | 0.759 | **0.568** | 0.426 | 0.699 |
| HUD | 0.713 | 1.076 | 0.846 | 0.769 | 0.780 | 0.067 |
| player tile | 0.801 | 0.251 | 0.863 | 0.807 | 0.823 | 0.700 |
| all three | **0.434** | 0.236 | 0.709 | 0.530 | 0.393 | 0.066 |

  - Entering-cell errors are causal: they scroll inward (interior -30%, total -34%).
  - HUD and player-tile errors are self-contained.
  - The three localized defects are 46% of depth-16 error.
- **Static rollouts** (the 98 roots whose TRUE view never scrolls in 16 steps; copy 0.462 /V):
  - corrt 0.657 /V (1.42x copy), corrg 18k 1.53x, residual 0.388 (0.84x).
  - corrt imagines a scroll that never happens in 32.7% of these rollouts (4-6% per step from depth 2, 0% at
    depth 1): with a false scroll 0.969 /V, without 0.506 /V.
  - The 17 rollouts with no move attempt at all: **0.044 /V**. The 81 with (blocked) move attempts: 0.786.
  - So corrt handles truly idle futures well. Its static drift is blocked moves that it judges correctly on
    true frames (one-step blocked 0.913 x copy) but wrongly on its own imagined frames.
  - Measured mechanism: 387 imagined steps with a (blocked) move attempt; 22.7% falsely scroll (0% at depth 1,
    on true inputs). False-scroll rate by the input's squared error at the target tile:
    - below median (<= 8.3): 17.5%; median to p75: 15.6%;
    - p75 to p90: 29.3%; above p90 (> 203): **56.4%**;
    - mean target error 111 on false-scroll steps vs 39 on correct ones.
  - A corrupted target tile triples the false-scroll rate. But even with a nearly clean target tile, 16-18% of
    blocked moves falsely scroll on imagined inputs. The decision also degrades from something else in the
    imagined history (OPEN).
  - Lighting is not the cause: at a below-median light change corrt reads 0.604 vs copy 0.384. Residual tracks
    brightness drift (0.406 vs copy 0.624 at above-median change); the copy heads cannot, their generator being
    starved.
- **ITC's binarized decoding does not help an under-trained soft head** (`corrt_raw_suffix_s7__hard`):
  - one step all 0.302 -> 0.467, idle 0.846 -> 0.997; depth 16 0.810 -> 1.330; static roots 1.42 -> 1.69x copy.
  - The small generator admixture HELPS: idle one-step beats pure copying (0.846 vs 0.997). The "soft blending
    leaks garbage" hypothesis is refuted.
  - Argmax fills entering cells with the zero pad (depth-16 edge tile accuracy 0.696 -> 0.392).
  - ITC's hard decoding presupposes a generator trained on every token (lane 10's `--gen-loss`).

**E7: per-tile worlds on LDAD lambda-10 tokens (`spatial_pool_ldad10_v1`: the same windows re-encoded by the Raw
+ LDAD 10 joint encoder; `compare_ldad_t`; seed 7, 6k, suffix loss).**

| LDAD minus Raw | residual head | corrt head |
|---|---|---|
| moved (x copy) | **-0.442*** (0.912 -> 0.469) | +0.046* (0.196 -> 0.242) |
| blocked | -0.949* | -0.398* |
| interact | -0.034* | -0.012 (ns) |
| sleep onset | +0.115* | -0.047* |
| idle | +0.012 (ns) | +0.078* |
| all | -0.122* | +0.031* |
| imagined, depth 1 / 4 / 8 / 16 (/V) | +0.002 / -0.021 / -0.039* / -0.059* | +0.054* / +0.107* / +0.108* / +0.009 |

- Historical imagined facts, depth 1 / 4 / 16 (true-fitted probes transferred to generated states, fitted
  separately per token space; generated-fit control pending for these arms):

| fact | residual Raw | residual LDAD | corrt Raw | corrt LDAD |
|---|---|---|---|---|
| facing | 0.66 / 0.44 / 0.31 | **0.92 / 0.94 / 0.82** | 0.66 / 0.44 / 0.31 | **0.94 / 0.94 / 0.91** |
| zombie adjacent | 0.78 / 0.83 / 0.56 | 0.84 / 0.80 / 0.49 | 0.80 / 0.78 / 0.50 | **0.90** / 0.80 / 0.55 |
| health R^2 | 0.98 / 0.90 / 0.56 | 0.98 / 0.90 / 0.54 | 0.98 / 0.90 / 0.59 | 0.98 / 0.93 / 0.62 |
| food R^2 | 1.00 / 0.95 / 0.86 | 1.00 / 0.94 / 0.83 | 1.00 / 0.95 / 0.88 | 1.00 / 0.94 / 0.78 |
| tiles near player | 0.87 / 0.79 / 0.67 | 0.91 / 0.81 / 0.66 | 0.96 / 0.87 / 0.71 | 0.97 / 0.87 / 0.66 |

- **LDAD tokens make action effects learnable for a head without a copy path.** The residual head's move error
  halves (0.912 -> 0.469; seed spread on moves is 0.007).
- Imagination keeps facing, the purely action-determined fact, at 0.82-0.94 through depth 16 (Raw 0.31), in both
  heads.
- For corrt, which already scrolls, LDAD tokens do not help moves (+0.046*), and shallow imagination is worse in
  /V. Caveat: /V here compares different token spaces.
- Food and far tiles are slightly worse: the same trade-off LDAD showed in z (E4).
- One seed: blocked, idle and sleep differences are not established (E3c: blocked/idle swing 0.3-0.5 across
  seeds).

**E7 generated-fit correction (2026-09-29).** `teval.py` rescored the existing Raw corrt6k,
LDAD10 corrt6k and LDAD10 residual6k checkpoints with identical generated-fit probe capacity.
Historical physical and true-fit fields reproduced exactly (maximum absolute delta 0 for all
three), despite the current `tworld.py` source hash differing from the training snapshots.
The v2 files preserve the old JSON. On the diagnosis TEST roots:

| generated-fit readout | Raw corrt6k | LDAD10 corrt6k | LDAD10 residual6k |
|---|---:|---:|---:|
| adjacent zombie AUC, depth 1 | .857 | .924 | .858 |
| adjacent zombie AUC, depth 16 | .557 | .641 | .564 |
| all-cell zombie AUC, depth 16 | .619 | .732 | .769 |
| health R², depth 16 | .636 | .692 | .648 |
| facing accuracy, depth 16 | .320 | .929 | .943 |
| moved-action adjacent zombie AUC, one step | .679 | .865 | .752 |

The old true-fit transfer readout undercounted several LDAD10+T facts (for example corrt
adjacent zombie .545 true-fit versus .641 generated-fit at depth 16). LDAD10+T has a measured
semantic readout gain even though cross-token-space physical error/V is a poor treatment
ranking. The between-arm semantic differences have no paired interval yet, and neither a
within-root safe-action choice nor an actor test exists. This is **λ=10 + T + corrt**, not the
proposed **λ=1 + T + corrg**; it establishes an interaction worth testing, not additivity.

**E3c: TC + T with a second training seed (`compare_tc_s8`; residual head, suffix loss, seeds 7 and 8).**

| TC minus Raw | seed 7 | seed 8 |
|---|---|---|
| moved (x copy) | +0.024* | +0.030* |
| interact | +0.059* | +0.035* |
| sleep onset | -0.009* | -0.010* |
| blocked | -0.794* | +0.131 (ns) |
| idle | -0.113 (ns) | +0.380* |
| all | -0.055* | +0.037* |
| imagined, depth 1 (/V) | +0.016* | +0.035* |
| imagined, depth 4 | +0.016* | +0.026* |
| imagined, depth 16 | +0.020 (ns) | +0.017 (ns) |

- Within one arm, blocked and idle swing by 0.3-0.5 between training seeds (Raw blocked 1.335 -> 0.789*, TC
  blocked 0.542 -> 0.920*). Those statistics cannot rank arms from one seed.
- **Verdict: TC tokens give the per-tile world no reliable advantage.** Replicated at both seeds, TC is slightly
  worse on moves, interactions and imagination at depths 1-4 (+0.016 to +0.035 /V); the only replicated gain is
  trivial (sleep onset -0.01). Consistent with E3a: same facts, different geometry.

**E5f result 1: `--gen-loss` at 6k (`evals/corrt_raw_suffix_s7_gl.json`, `where_corrt_raw_suffix_s7_gl.json`).**

| one step (x copy) | corrt | corrt + gen loss | corrg 18k |
|---|---|---|---|
| moved | 0.196 | **0.748** | 0.176 |
| blocked | 0.913 | 0.425 | 0.374 |
| idle | 0.846 | 0.504 | 0.459 |
| sleep onset | 0.190 | 0.052 | 0.040 |
| all | 0.302 | 0.462 | 0.178 |
| imagined depth 16 (/V) | 0.810 | 0.939 | 0.703 |

- **The starvation diagnosis is confirmed.** The generator's standalone error (x copy):
  - moves 3.4-5.8 -> 0.83-0.98; blocked 12.6-60 -> 0.42-0.72;
  - HUD 256 -> 1.4 (moves) and 159 -> 0.75 (blocked).
  - The head uses it: blocked interior 0.933 -> 0.380, blocked HUD 1.000 -> 0.492, idle player tile
    0.999 -> 0.232 (facing updates).
- **But moves regress because the head stops scrolling, not because the generator wins.** On moves the interior's
  scroll-source weight falls 0.931 -> 0.204 and self-copy rises 0.047 -> 0.61 (generate only 0.09). Interior
  x copy 0.064 -> 0.714.
  - At 6k, the extra objective on the shared backbone weakens the move decision: the routing that needed 18k
    updates without it (E5g).
  - In ITC the decision (optimal transport) and the generator are not trained through one shared mixture.
- corrg 18k learned facing without any generator loss (depth 1 0.936, depth 16 0.861, vs corrt 6k
  0.661 / 0.310): budget matters as much as design.
- **`--regions itc` alone is harmful** (`corrt_raw_suffix_s7_itc`): one step all 0.696, moved 0.809, sleep 0.440;
  depth 16 0.993 /V; health R^2 at depth 16 **-3.40**, food 0.41. The ring and HUD (46 tokens) come from a
  generator trained only through those tokens: a direct head there (direct is 0.90x copy), and the HUD drifts.
  The regions rule presupposes ITC's trained generator.
- **Both switches** (`corrt_raw_suffix_s7_gl_itc`): all 0.662, moved 0.839, blocked 0.891, idle 0.837, sleep 0.343;
  depth 16 0.927 /V; health R^2 at 16 -0.42, food 0.41.
- **At 6k no ITC-derived change beats plain corrt** (all 0.302, depth 16 0.810). The generator loss cures
  starvation but the head stops scrolling. The regions hand 46 tokens to a generator that is at best copy-level
  on moves (0.83-0.98x). ITC's generator samples plausible DISCRETE tokens for new content; a continuous L1
  generator regresses to the conditional mean, which is all it can give for never-seen cells.
- **Where the 6k fix arms lose moves** (`where_*`, `corrg_probe.py <world>` -> `probe_*.json`):
  - All three stop scrolling: interior scroll-source weight on moves 0.93 -> 0.19-0.24, self 0.05 -> ~0.6.
    That includes regions-only, which has no generator loss: any path that trains the generator does it.
  - Two explanations refuted:
    - the backbone still carries passability at the target tile (AUC 1.00 in every arm);
    - the head's move decision still ranks moved vs blocked (AUC 0.977-0.989, corrt 0.998), and its magnitude is
      not smaller: mean decision logit on moved steps is corrt 3.57, gen loss 1.80, regions 4.45, both 4.57
      (corrg 18k 1.24).
  - So the per-tile selection logits keep self high on moves once the generator carries gradient.
  - The 18k runs (corrt, both switches) decide whether this is another budget effect, as corrg's routing was.
- **18k verdict (`compare_itc18`):**
  - Both switches at 18k: all 0.320, moved 0.476, depth 16 0.803; health / food R^2 at 16 0.45 / 0.61; facing
    at 16 0.278.
  - Budget helps them a lot (6k -> 18k: all 0.662 -> 0.320*, depth 16 0.927 -> 0.803*).
  - At matched 18k they stay clearly worse than plain corrt: all +0.147*, moved +0.296*, depth 16 +0.120*.
  - corrt 18k vs corrg 18k: all -0.005*, depth 16 -0.020* (one seed).
  - **ITC's training/decoding adaptation is dropped for continuous JEPA tokens.** ITC's generator gains from
    sampling discrete, plausible tokens; a continuous L1 generator regresses to the conditional mean, so what it
    generates (new cells, HUD) stays blurry. The head for the architecture is plain corrt (or corrg) at an
    adequate budget.
- **corrt 18k re-localized (E5j: `where_/static_/substitute_corrt_raw_suffix_s7_u18000.json`).**
  - Solved by budget:
    - the player tile is now generated (weight 0.51 on moves; x copy 0.934 -> 0.192), so facing updates;
    - blocked / idle / HUD errors fall by more than half (blocked interior 0.933 -> 0.302, blocked HUD
      1.000 -> 0.426, idle HUD 1.000 -> 0.582).
  - Not solved:
    1. **Entering cells:** x copy 0.903, now 62% of moved error. Partial observability: the class equals the
       in-view neighbour's in 75% of cells.
    2. **Interactions:** near tiles 0.992, inventory 0.995 x copy. Mining and placing outcomes are unlearned.
    3. **Blocked-move false scrolls in imagination:** worse at 18k.
  - Substitution at depth 16 (/V):

| substituted | none | entering | HUD | player | all three |
|---|---|---|---|---|---|
| total | 0.683 | 0.425 (-38%) | 0.583 (-15%) | 0.685 | 0.320 |

  - Static rollouts (98 roots; copy 0.462): false scroll in 40.8% of rollouts (6k 32.7%); 33.3% of blocked-move
    steps (6k 22.7%). Depth 16: 0.934 with a false scroll, **0.392 without (beats copy)**, 0.040 with no move
    attempt.
  - Mechanism, first false scroll only:
    - hazard 17.6% per blocked-move step (31.8% at depths 2-4, ~10% after 8);
    - the head's decision logit on blocked steps is -5.19 on the TRUE window and -4.36 on the imagined window
      (imagined inputs shift it up +0.8);
    - on the steps that falsely scrolled it reads +0.88 imagined vs -0.86 true: borderline blocked moves;
    - 11.0% of blocked first steps are positive even on true windows.
    - So first false scrolls = ~11 points of blocked moves misjudged on real frames + ~6-7 points flipped by the
      imagined-input shift. Which blocked moves are borderline: next.
    - **Superseded 2026-09-30 (E5k, audit section):** those "true windows" were 5-frame windows, which put the
      current frame at time position 4. With 4-frame windows the same world misjudges 0.0% of blocked moves on
      true frames and 1.3% on imagined ones; the 11 points are a time-position-4 bias, not borderline moves.
- **Stage-2 engineering (16:26-16:40):** per-token SSM states (40 x 82 sequences x 4 x 64 x 64 fp32, ~215 MB per
  tensor per layer) OOM'd fmamba even at chunk 64. Fixed by per-layer gradient checkpointing for the factored
  backbones (same math).
  - Measured per update (batch 40): peak GB full 2.53, fattn 0.87, fmamba 2.35, fcanvas 2.02, fscan 0.89;
    steady s/update fattn 0.33, fscan 0.38, fmamba 0.70, fcanvas 0.80.
  - teval evaluates per-token Mamba arms at 4 roots per batch (state size); results are unchanged.

**Incident 4 (2026-09-28 09:05-09:54, self-inflicted):** I appended two paper pins to
`third_party/PAPERS.lock`. It is hashed into every LeWM checkpoint's source identity (`d4mj/sources.py`
"references"), so canonical loads failed and killed `ldad_raw_lam10_nosig` at start. Reverted with
`git checkout`; loads verified; lane 8 relaunched. No other job loaded a canonical bundle in that window.
- Pins kept here instead:
  - `third_party/papers/2606.31232v1-deltajepa.pdf`, sha256 94e394fd9cbffaffb370fdc5f4bb3e2a2833ca41966e02efe800d5afff30371e;
  - `third_party/papers/2605.16457v1-itc.pdf`, sha256 37e15d2ea600d8f71b271b5f980a1e76e7af3b0383f70edc9994c06269a5b70e.

**Stage 2 prerequisites (2026-09-28).**
- **Scroll estimator** (`scroll.py`, `scroll.json`): the shift minimizing token difference over the shared map
  cells, against the simulator's labels:
  - moved 0.975 (2.4% read as no scroll), blocked 1.000, non-moves 0.997 (0.28% false scroll);
  - on the factual futures: 36.6% of transitions scroll, 70.3% of move actions, and the direction matches the
    action in 99.93%.
- **Backbones** (`tworld.py --backbone`, verified by `stage2_checks.py` / `stage2_checks.json`, all exact):
  - `masked_scan` with all steps kept equals `FunctionalMamba2.scan` (max diff 0.0). With absent-frame
    masks it matches the **SSM-only** zero-delta reference (4.8e-7 at scale 2.3); this does not test holding
    Mamba's causal convolution. The 2026-09-29 canvas diagnosis below found that the convolution does advance.
  - with no scroll, fcanvas equals fmamba (0.0);
  - synthetic scrolling views: every canvas cell receives one world token (0.0);
  - every backbone is causal (0.0 on frames 0-3 when frames 4-5 change).
  - Parameters: full 4.98M, fattn 6.56M, fmamba / fcanvas / fscan 6.38M. A single 6k factored arm
    cannot settle the architecture class, especially against a full model trained for 18k updates;
    compare training time, matched data, seeds and control-relevant facts.
  - First version's full canvas (17 x 19 cells) cost ~900 MB per Mamba state tensor per layer at batch 40. Fixed
    by scanning only cells in view at least once, compacted per window.
- **Arms (lane 9):** corrt head, Raw tokens, suffix loss, seed 7, 6,000 updates; backbones fmamba, fcanvas,
  fattn, fscan; then teval and paired comparisons.
- First fmamba attempt: CUDA OOM (2026-09-28 16:26). The Mamba-2 Triton scan allocates sequences x chunks x
  chunk_size^2. With the canonical chunk_size 256 and 3,280 six-step sequences that is ~860 MB. The factored
  arms now use chunk_size 64: kernel tiling only (SSD is exact for any chunking). stage2_checks re-run, all
  exact.

**Stage 2, matched 6k worlds and scroll diagnostics (2026-09-29; exploratory TEST block).**
`lane9.sh` completed after fcanvas's 6k checkpoint was found complete but its evaluation interrupted. The
6k worlds use the same Raw tokens, corrt head, suffix loss and seed. Physical metrics and paired intervals
use all 1,002 diagnosis roots (143 walk seeds), which were not world-training data; the probe facts below
use the disjoint 283 TEST roots (43 walk seeds). Lower physical error is better. Paired intervals cluster
by walk seed (`compare.json`).

| world | training seconds | one-step all x copy | moved x copy | blocked x copy | recursive depth-16 /V |
|---|---:|---:|---:|---:|---:|
| full 6k | 1,289 | 0.302 | 0.196 | 0.913 | 0.810 |
| factored attention 6k | 2,336 | 0.243 | 0.193 | 0.905 | 0.740 |
| per-token Mamba 6k | 4,317 | 0.245 | 0.191 | 0.978 | 0.745 |
| scroll canvas 6k | 4,984 | 0.243 | 0.195 | 0.919 | 0.739 |
| raster-scan Mamba 6k | 2,317 | 0.192 | 0.182 | 0.425 | 0.795 |
| full 18k | 3,451 total | **0.173** | **0.180** | **0.326** | **0.683** |

Canvas minus factored Mamba at 6k is -0.0015 [-0.0026,-0.0005] on one-step all,
+0.0032 [+0.0025,+0.0039] on moved, -0.0586 [-0.1244,-0.0241] on blocked, and
-0.0058 [-0.0187,+0.0083] at depth 16. The depth-16 gain is unresolved. Full 18k beats canvas 6k
at depth 16 by 0.0554 [0.0406,0.0714] and took less training time; this is a compute comparison,
not a matched-update architecture ceiling. Against factored attention, canvas differs by only
-0.0009 /V at depth 16, interval [-0.017,+0.016]. Raster-scan Mamba beats attention one step
(0.192 vs 0.243 x copy, paired difference -0.051 [-0.094,-0.013]) but loses at depth 16
(0.795 vs 0.740 /V, +0.055 [+0.035,+0.073]). Teacher-forced depth-16 error is better for
raster scan (0.063 vs 0.069 /V), confirming its gap is in recursive self-feeding, not factual
one-step fit. The crossover occurs between depths 4 and 8. Full 18k beats attention 6k at
depth 16 by 0.056 [+0.038,+0.074], though attention uses only two thirds of its wall time.
All architecture claims remain one world seed on one inspected diagnostic block; equal-time full
12k versus attention 6k has not been measured.

**Raster-scan recursion localization (`static_*.json`, `where_*.json`, `substitute_*.json`,
`fscan_hud_feedback.json`, `fscan_hud_curve.json`, `fscan_{hud_,}perturb.json`; post-hoc).**
On 98 factual no-scroll roots, fscan's 16-step false-scroll rate is 0.306 versus fattn's 0.337,
so false scrolling does not explain the extra scan error. At depth 16 the scan-minus-attention
error gap is +0.0551 /V. Error *lands* mostly on interior (+0.0232 /V) and edge map tokens
(+0.0278 /V), not on HUD tokens themselves (+0.0014 /V); location of error is not necessarily
its feedback source. Replacing only newly entering map cells with their true tokens helps both
similarly (fscan 0.795 -> 0.511 /V; fattn 0.740 -> 0.461 /V), leaving a +0.050 gap.
Replacing only the 18 HUD tokens with true tokens **after each prediction, before feedback**
helps fscan 0.795 -> 0.619 /V and fattn 0.740 -> 0.647 /V, reversing their ordering. The
paired walk-seed difference in HUD benefit is -0.0832 /V, 95% bootstrap interval
[-0.1024,-0.0645] on 1,002 roots/143 walk seeds. It grows from -0.0027 /V at depth 2
(interval spans zero) to -0.0134 [-0.0252,-0.0040] at depth 4, -0.0303
[-0.0444,-0.0179] at depth 8, and -0.0832 by depth 16. With the *same* isolated
first-step HUD perturbation fed to each frozen world on 283 TEST roots, fscan's excess
error relative to fattn is +0.0055 /V [-0.0031,+0.0201] using fattn's perturbation, or
+0.0066 [-0.0002,+0.0208] using fscan's: neither resolves at depth 2. Full-frame matched
perturbations also do not resolve. The apparent 2.6-2.9x versus ~1.0x depth-2 output-displacement
response has a broad walk-seed paired interval for its *difference* that includes zero
(`fscan_hud_perturb_pair.json`), so it is not a resolved instantaneous sensitivity result.
Thus the late scan disadvantage disappears when both worlds receive true HUD feedback.
This establishes a stronger dependence on that feedback in these frozen rollouts; it does not
prove that native HUD prediction error is the only cause, identify a particular Mamba channel,
or show how much a retrained model would recover. True HUD tokens also supply outcome information
that neither imagined world has on its own.
Both worlds are frozen, use identical actions and true targets, and the intervention is an
oracle; no actor result follows. The architecture scans T x 82 tokens in frame order
(`tworld.py` `h.flatten(1,2)`), placing same-position tokens 82 scan steps apart. That is a
structural difference consistent with the long-context issue described by
[Po et al., 2025](https://arxiv.org/html/2505.20171v1), but this paper is context, not an
empirical explanation of the HUD effect here.

The canvas shift estimator agrees with true encoded-frame shifts on 99.3% of first imagined steps but only
76.4% at depths 9-16, with 43.7% of true scrolls missed late (`canvas_diag.json`). A frozen-world
intervention replaces only its inferred shifts with those estimated from true frames. Recursive depth-16
error falls from 0.74584 to 0.73529 /V, difference -0.01055 [-0.01697,-0.00498] over 43 walk seeds
(`canvas_oracle.json`, the 283 TEST roots only; its native 0.74584 therefore differs from the
all-root 0.73877 in the table). This shows shift errors cause some rollout error, but account for only ~1.4% of
canvas's total depth-16 error under this intervention. The reference shift estimator itself is 99.35%
accurate against simulator steps; this is a near-oracle, not a deployable control.

`masked_scan` also advances Mamba's causal convolution on missing frames: against a scan that removes those
steps, output relative RMSE is 0.418, while all-kept steps agree to 7.2e-7 (`canvas_diag.json`). This is a
real implementation defect, but only 2,467 / 285,138 (0.865%) final-frame map cells in the 5-frame TEST
windows had left view and reentered; 308 / 4,526 windows (6.8%) had any (`canvas_reentry.json`). Its
quantitative contribution in training remains unmeasured. A frozen-checkpoint full-block-hold intervention
(`canvas_hold_oracle.py/json`) removes absent steps from both convolution and SSM, with the same actions and
self-fed tokens. Its all-kept path agrees exactly with the original, including under bfloat16 autocast.
At depth 16 it changes 0.745844 -> 0.745757 /V, difference **-0.000088** with walk-seed paired interval
**[-0.000877,+0.000521]**. Thus correcting inference alone has no resolved useful effect on this block;
a retrained model could differ, but is not justified on this evidence. The historical `tworld.py`
module header says "exact hold" for fcanvas; this is false for the full Mamba block. It is preserved
byte-for-byte because every Stage-2 checkpoint pins that source hash; this notebook is the correction.

A post-hoc context split shows canvas's depth-16 advantage concentrated in 346 roots with two or three
recent scrolls: -0.0181 [-0.0349,-0.0021] /V versus factored Mamba. The 430 no-scroll and 226
one-scroll strata are unresolved (`canvas_context.json`). The shift near-oracle's benefit is largest in the
one-scroll subset (-0.0222 [-0.0430,-0.0058]), so shift-estimation error alone does not explain the
scroll-rich canvas advantage (`canvas_oracle_strata.json`). These strata were selected after viewing
results and need independent confirmation if they become a decision rule. Spatial memory with egomotion is a
reasonable architecture precedent (Gupta et al., [CMP, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Gupta_Cognitive_Mapping_and_CVPR_2017_paper.html)),
but that paper does not validate this five-frame Craftax canvas. The [upstream Mamba-2 implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
keeps causal-convolution and SSM state separately; a zero SSM delta alone cannot mean a whole-block hold.

The corrected v2 readouts (`evals/*fmamba__readout_v2.json`, `evals/*fcanvas__readout_v2.json`)
change the semantic reading. At depth 16, adjacent-zombie AUC is 0.538 true-fit / **0.595 generated-fit**
for fmamba, versus 0.547 / **0.574** for canvas. Facing is 0.313 / **0.548** versus 0.285 / **0.384**.
The old true-fit decoder mildly favoured canvas on adjacent zombies; the matched generated-fit decoder
reverses that order. These fact differences have no paired arm interval yet. Physical comparisons above
are unaffected by probe fitting. Corrected depth-16 generated-fit adjacent-zombie AUC is 0.592 for
factored attention and 0.589 for raster scan; neither has a paired semantic interval. Lane 9 and
Raw LDAD lambda 1 lane 4b have completed; the λ=1 result is recorded below. In the earlier 2k
dose screen, root-z zombie AUC was 0.616 (Raw), 0.627 (λ=0.1), 0.850 (λ=1), 0.934 (λ=10).
Direct seed-8 and the full 36k curve remain deferred for a decision after this comparison.

**Remaining material runs audit (2026-09-29).** `lanes.log` has no `LANE1_DONE` or
`LANE15_DONE`, and no `direct_raw_suffix_s8.pt` or `corrt_raw_suffix_s7_u36000.pt` exists.
The direct-head second seed and the 36k corrt learning curve were scripted (`lane1b.sh`,
`lane15.sh`) but not run. All E1/E6 context and State Passing arms, Stage-2 6k backbones,
ITC/copy controls, and Raw λ=1 10k are complete. The 6k factorized backbone comparison
has no matched 18k follow-up or second world seed. No λ=1 patch-token pool, λ=1 + T +
corrg world, λ=10 + T + corrg world, generated-state decision head for these lever worlds,
or LeWM actor result exists. `TWorld`'s default `full` backbone is a 6-layer block-causal Transformer,
but Stage 2 also implemented `fmamba`: spatial attention within each frame followed by canonical Mamba-2
over time for each token position. Thus λ=1 + patch-token T + `corrg` + `fmamba` would test **Mamba
dynamics**, as intended. It has not been trained at long context: the current per-tile pool and time-position
table are six frames, and the E1/E6 long-context checkpoints use global `z` rather than patch tokens, so
those checkpoints cannot be combined with `fmamba` as weights. The unfinished direct-head seed 8 and
Transformer 36k curve do not answer whether this long patch-token Mamba combination works. A matched
long-context per-tile Mamba run, with actor evaluation, remains the decision-relevant unrun work.

**E1 so far (`context_length.json`), DEV/FINAL 256-frame windows:**

| world | 4-7 | 8-15 | 16-31 | 32-63 | 64-127 | 128-255 |
|---|---|---|---|---|---|---|
| L4, full history (x copy) | 1.51 | 3.24 | 3.78 | 4.92 | 7.45 | 9.27 |
| L4, 3-frame window (x copy) | 0.61 | 0.88 | 0.69 | 0.73 | 0.82 | 0.86 |
| L16, full history (x copy) | 0.76 | 1.06 | 0.88 | 0.96 | 1.27 | 1.69 |
| L16 memory benefit | +11% | +10% | +10% | +6% | -6% | -35% |
| L64, full history (x copy) | 1.20 | 1.57 | 1.30 | 1.36 | 1.68 | 1.82 |
| L64 memory benefit | +18% | +18% | +20% | +20% | +20% | +19% |

- Memory benefit = 1 - err(full history) / err(3-frame window).
- Imagination after a 48-frame context, 16 steps (/ V): L4 window 0.48, L16 recurrent 0.68, L64 recurrent 0.98,
  copy 0.55.
- Reading: training length decides whether the recurrence is usable past it.
  - L4: collapses beyond its window.
  - L16: memory helps up to about 4x its length, then fails.
  - L64: memory helps (+18-20%) at every position to 255.
- But at equal transitions per update, the longer the training window, the less accurate the world. L64 is
  1.2-1.8x copy, against L4-with-window 0.61-0.86x. It sees 6 episodes per update vs 128.
- Revisits (next view closer to a frame 4+ back than to the current one) are 0.9% of transitions.
- E1b and E1c test continuation and what the memory is used for.

**E1b so far (`context_cont.json`), continuation of the L4 world:**

| arm | 4-7 | 16-31 | 64-127 | 128-255 | memory benefit 64-127 | imagine 16 (/V) |
|---|---|---|---|---|---|---|
| L4to16, full history (x copy) | 0.66 | 0.77 | 1.43 | 2.08 | -47% | 0.74 |
| L4to64, full history (x copy) | 1.05 | 1.20 | 1.58 | 1.68 | +7% | 0.91 |

| L16x3 (fresh, 30k updates), full history | 0.58 | 0.65 | 0.96 | 1.32 | -11% | 0.54 |

- Continuing destroys the L4 world's short-window quality: its 3-frame-window error rises from 0.61-0.86 to
  0.71-1.79.
- L16 at 3x budget matches or beats L4-with-window inside 4-63 (0.58-0.86 vs 0.61-0.88 x copy). Its memory helps
  +7-8% to position 31 and +3% to 63, then fails beyond 64. L16's E1 gap was largely under-training.

**E1d, L64 with 24 windows per update (`context_cont.json`):**
- Full history, x copy: 0.65 / 0.91 / 0.69 / 0.73 / 0.84 / 0.87 at positions 4-7 / 8-15 / 16-31 / 32-63 /
  64-127 / 128-255. Never worse than copying, out to 4x its training length.
- Memory benefit +24-29% at every position; by type +22% moved, +37% blocked, +42% sleep onset,
  +32% exact revisits.
- 16-step imagination after 48 frames: 0.447 (/V). The best Mamba world so far (L4 windowed 0.48, copy 0.55);
  its own 3-frame window gives 0.726, so the memory is what helps.
- E1's L64 gap was diversity (6 windows per update), not length. It saw 4x the transitions per update of
  L4/L16, so E1e runs the matched-compute controls.

**E1e / E1f, matched compute (~1,500 transitions per update):**
- Error x copy: full history / own 3-frame window, at positions 4-7, 16-31, 64-127, 128-255.
- Imagination: 16 steps after a 48-frame context (/V); copy = 0.553.

| arm | 4-7 | 16-31 | 64-127 | 128-255 | memory 16-31 / 128-255 | imagine 16 (full / window3) |
|---|---|---|---|---|---|---|
| L4b512 | 1.74 / **0.52** | 4.68 / **0.59** | 9.14 / **0.69** | 11.35 / **0.72** | -699% / -1487% | 2.076 / 0.406 |
| L16b100 | 0.56 / 0.63 | 0.63 / 0.70 | 0.91 / 0.85 | 1.32 / 0.89 | +11% / -48% | 0.426 / 0.524 |
| L64b24 | 0.65 / 0.85 | 0.69 / 0.94 | 0.84 / 1.18 | 0.87 / 1.23 | +26% / +29% | 0.447 / 0.726 |
| L4to64b24 | 0.60 / 0.72 | 0.65 / 0.78 | 0.78 / 0.97 | 0.82 / 1.03 | +18% / +20% | **0.400** / 0.650 |

- At matched compute the windowed L4 world is the most accurate one step ahead at every position; E1d's win was
  partly 4x data.
- For imagination, (b) ties (a): L4to64b24 full 0.400 vs L4b512 window 0.406. (b) keeps a memory worth +18-20%
  over its own window.
- The best (b) is the canonical shape: a 4-frame joint world continued at L=64 with 24 windows per update. The
  H2 bridge continues too, but with 16 windows per update, heads and 2,000 updates, and its world ended at
  1.4-2.6x copy.
- Recall is 0.19% of transitions, so next-frame accuracy cannot show memory's value for long-horizon play.

**E1c (`memory_use.json`)**, 256 held-out 128-frame windows, positions >= 16. Memory benefit by transition type:

| world | moved | blocked | sleep onset | exact revisit | other | all |
|---|---|---|---|---|---|---|
| L4 | -511% | -1274% | -180% | -1114% | -828% | -641% |
| L16 | +2% | -7% | +15% | -7% | -1% | +1% |
| L16x3 | -3% | -15% | +13% | -9% | -7% | -5% |
| L64 | +19% | +24% | +42% | +11% | +21% | +21% |
| L4to64 | +9% | +9% | +13% | -3% | +8% | +9% |
| L4b512 (E1e) | -748% | -1947% | -329% | -1696% | -1210% | -942% |
| L16b100 (E1e) | +1% | -8% | +12% | -0% | -3% | -1% |
| L64b24 (E1d) | +22% | +37% | +42% | +32% | +28% | +26% |
| L4to64b24 (E1f) | +16% | +27% | +24% | +20% | +20% | +18% |

- The last four rows come from lane 2's 09-28 rerun of `memory_use.py`; its values for the earlier arms differ
  from the table above only at float precision (5th significant digit).

- Exact revisits: the next map view pixel-identical to a frame 4+ back and to none of the last 3. They are
  53 of 28,672 transitions (0.19%). Recall of content that left the window is almost absent from this data.
- Where memory helps (L64), it helps every transition type about equally.

**Precedents for (a) vs (b), verified in source:**
- DRAMA (our Mamba base, `third_party/Drama/config_files/configure.yaml`):
  - trains its Mamba world on BatchLength 128 x BatchSize 16 (2,048 transitions per update);
  - imagines 16 steps from an 8-frame context (ImagineContextLength 8);
  - acts from a bounded 16-frame deque (RealityContextLength 16, `train.py:149`, `eval.py:56`).
  - So every deployed position (<= 24) lies inside its training length.
- Dreamer 4: batch lengths 32/128 alternating, context 96-192 frames.
- le-wm: trains 4 frames, deploys a 3-frame window.
- All three keep deployment inside the trained range. We trained 4 frames and deployed an unbounded recurrence.

**E4a action identifiability (`identifiability.py`, exact for its constructed panel):**
- Futures roots, all 17 actions under one key. Bayes accuracy of a frame-pair decoder on this panel:
  0.339 under uniform actions, 0.583 under a fixed GLOBAL action prior copied from the interface pool.
- 44% of transitions have a uniquely identified action.
- Dominant collision: NOOP / DO / SLEEP / failed PLACE and MAKE; blocked moves when already facing that way.
- **Correction:** LDAD's 0.57 training accuracy at update 1,500 and later 0.607 plateau are on a different
  state/action distribution with its logged policy. The 0.583 synthetic-panel score is not their ceiling;
  equality or saturation cannot be inferred from these numbers.

**E5 / E3b, per-tile worlds, seed 7 (`evals/*.json`, `compare.json`).**
- Setup: futures roots; paired bootstrap over 143 walk seeds (root-sampling uncertainty only; second
  training seeds queued).
- Each state's errors are relative to its own copy baseline (x copy) or variance (/V).

| world | one-step all | moved | blocked | sleep | idle | 16-step imagined (/V) | health R^2 at k16 |
|---|---|---|---|---|---|---|---|
| direct, Raw tokens | 0.902 | 0.971 | 0.978 | 0.750 | 1.066 | 0.969 | 0.11 |
| residual, Raw tokens | **0.596** | 0.912 | 1.335 | 0.049 | 0.805 | **0.849** | 0.56 |
| residual, TC tokens | **0.541** | 0.936 | **0.542** | 0.040 | 0.693 | 0.869 | 0.60 |

- Residual vs direct: better on everything except blocked moves (+0.36*); 16-step -0.120*.
- TC vs Raw (residual), seed 7: one-step -0.055*, blocked -0.79*; moved +0.024* and interact +0.059* (TC worse);
  16-step +0.020, not resolved.
  - **CORRECTED 2026-09-28 by the seed-8 replicate (E3c below):** the blocked (-0.79*) and "all" (-0.055*)
    advantages are training-seed noise; the sign flips at seed 8.
- Moves remain almost unlearned by the direct/residual/gated heads: 0.91-0.97x copy, against a local-linear
  ceiling of 0.72 captured.

**E5, copy variants, seed 7 (Raw tokens; `compare.json`):**

| head | moved | blocked | idle | all | gen 4 / 8 / 16 (/V) |
|---|---|---|---|---|---|
| residual | 0.912 | 1.335 | 0.805 | 0.596 | 0.425 / 0.605 / 0.849 |
| gated | 0.910 | 0.933 | **0.483** | 0.548 | 0.418 / 0.598 / 0.845 |
| corr (neighbour copy) | **0.203** | 6.121 | 0.848 | **0.380** | **0.307 / 0.505** / 0.825 |

- corr vs residual:
  - moved -0.709 [-0.721, -0.699]*, all -0.216*;
  - gen 4 -0.118*, gen 8 -0.100*, gen 16 -0.024 (not resolved);
  - blocked +4.79*.
- corr learns WHAT a move does (the scroll) but not WHETHER it happens. Imagined moved/blocked change 1.15x vs
  8.0x true; every blocked move in a rollout injects a full wrong scroll, so the gain fades by depth 16.
- The decision is one global bit set by the tile in front of the player, but corr decides per tile.
- E5b `corrg` = corr + a frame-level move logit from the action token, zero-init (verified identical to corr at
  init).
  - vs corr: blocked 6.12 -> 4.61 [-2.64, -0.94]*; all 0.380 -> 0.350*; gen 4 -0.018*, gen 16 +0.014*.
  - vs residual: gen 16 -0.010 (not resolved).
- E5c (`corrg_diag.json`), read out of the model, moves only:
  - the corrg frame logit separates moved/blocked at AUC 0.78 (training windows 0.77): means 2.19 vs -0.18;
  - neighbour copy mass on blocked moves is still 0.36;
  - the decision is under-learned on training data too.
- E5d (`corrg_probe.json`), probes on the frozen corrg backbone, moved vs blocked AUC:
  - input token of the target tile 1.00; backbone output AT the target tile 1.00;
  - player tile 0.86; action token 0.88; the gate itself 0.79.
  - The backbone knows passability at the target tile but does not route it to the global decision (an
    action-keyed lookup of the tile at player + direction).
- **E5e result, `corrt`** (the gate reads the backbone output AT the target tile). Routing was the problem:

| head | moved | blocked | all | gen 1 / 4 / 8 / 16 (/V) |
|---|---|---|---|---|
| residual | 0.912 | 1.335 | 0.596 | 0.168 / 0.425 / 0.605 / 0.849 |
| corr | 0.203 | 6.121 | 0.380 | 0.096 / 0.307 / 0.505 / 0.825 |
| corrg | 0.203 | 4.612 | 0.350 | 0.086 / 0.289 / 0.495 / 0.839 |
| **corrt** | **0.196** | **0.913** | **0.302** | **0.054 / 0.229 / 0.432 / 0.810** |

  - corrt vs corr: blocked -5.21*, all -0.078*, gen 1 / 4 / 8 all resolved; gen 16 -0.015 (not resolved).
  - corrt vs residual: every depth resolved (gen 16 -0.039*).
  - Imagined moved/blocked change 70.7x (true 8.0; corr 1.15x): it now scrolls when the move succeeds and holds
    the view when it is blocked.
  - Costs: sleep onset 0.19 vs 0.05*; 16-step error still 0.81 of variance (compounding remains).
  - The bias assumes the player sits at the centre of an egocentric view (Craftax-specific). corrg at 3x updates
    (lane 3d) tests whether plain attention learns the same routing.
- **Training-seed noise floor** (corr seed 7 vs seed 8, same recipe):
  - moved 0.203 vs 0.197; blocked 6.12 vs 5.82 (not resolved);
  - idle 0.848 vs 0.518*, all 0.380 vs 0.329*, gen 16 0.825 vs 0.799*.
  - The bootstrap covers root sampling only, so one-seed differences of this size (idle, all) are NOT
    established. Gated's idle 0.48 falls inside this. Effects far beyond it stand:
    - corr on moves (-0.71 vs seed spread 0.006);
    - ~~TC on blocked moves~~ RETRACTED 2026-09-28: blocked moves swing 0.3-0.5 between training seeds within
      one arm (E3c).
- **corr on TC tokens** vs Raw: moved +0.091*, blocked -2.54*, all +0.030*, gen 16 -0.016 (not resolved).
  Same trade-off as under the residual head (blocked better, moves worse), in both heads.
- E5e (lane 3d), which of the two is it:
  - `corrt` adds a gate read directly at the target tile (player token 31 + direction, moves only), zero-init,
    verified identical to corr at init;
  - `corrg` at 3x updates (18,000).

**E4, Raw LDAD lambda 10 training curve** (`levers_ldad_v1/raw_lam10/metrics.jsonl`, paired with canonical Raw):
- At updates 9,501-10,000: prediction MSE 0.097 vs 0.018 (5.3x); SIGReg 2.37 vs 1.25.
- Gradient norm 29 vs 0.93: clipped at 1.0 throughout, so LDAD sets the update direction.
- LDAD training accuracy plateaus at 0.607. E4a's 0.583 refers to different roots under a fixed global
  prior, so no identifiability ceiling for this training accuracy has been established.
- Whether the MSE rise is only larger latent steps: the copy-normalized evaluation (`ldad_eval.py`) decides.

**Incident 3 (15:04):** PC shut down. `/tmp` was wiped, so scratchpad logs are lost; all artifacts survived.
- Lost: the corr arm, E1e, lane D.
- Relaunched as admission-controlled lanes (`lib.sh`):
  - a job starts only when free GPU memory >= its declared need + 256 MiB;
  - CUDA-OOM is retried; any other failure stops the lane;
  - jobs are idempotent.
- Logs: `artifacts/eda/levers_logs/` (`lanes.log` = the timeline).
- Lanes: 1 = per-tile arms + evals + second seeds (`lane1.sh`); 2 = LDAD screens + LDAD evals + E1e/E1f + E1c
  (`lane2.sh`); 3 = early eval of the gated/corr arms (`lane3.sh`).

**Incident 2 (14:48-14:58):** too many concurrent GPU jobs.
- The lambda screens, the screen evaluation and one context run died of CUDA OOM, and the lane script moved on
  silently.
- Fix: lane D (`lane_d.sh`) runs all non-lane-A GPU jobs serially after lane C4, retries CUDA-OOM failures (6 x 3 min)
  and stops on any other failure.

**E4 results, Delta-JEPA (LDAD).**
- Encoder content (`ldad_eval.json`, 10k; `ldad_screens.json`, 2k paired with canonical step-2000):

| 10k updates | canonical Raw | Raw + LDAD lambda 10 |
|---|---|---|
| z: zombie adjacent (AUC) | 0.606 | **0.961** |
| z: passability, 4 dirs (AUC) | 0.588 | **0.855** |
| z: health (R^2) | 0.122 | **0.728** |
| z: d' of one added zombie | 0.51 | **3.38** |
| z: cos(adjacent, far zombie change) | 1.000 | **0.940** |
| action from dz (accuracy) | 0.35 | 0.67 |

- 2k screens, z zombie adjacent / passability: canonical 0.616 / 0.600; lambda 0.1 0.627 / 0.624; lambda 1
  0.850 / 0.791.

**E4c/E4d (2026-09-28): TC + LDAD lambda 10 at 10k, and Delta-JEPA as published (no SIGReg).**
- z on natural root frames (`ldad_eval.json`):

| 10k updates | raw | raw + LDAD 10 | tc | tc + LDAD 10 | raw + LDAD 10, no SIGReg |
|---|---|---|---|---|---|
| effective rank | 15.7 | 11.7 | **2.5** | 5.9 | **5.8** |
| total variance | 159 | 213 | 988 | 1028 | 368 |
| zombie adjacent (AUC) | 0.606 | 0.961 | 0.654 | 0.979 | 0.912 |
| passability, 4 dirs (AUC) | 0.588 | 0.855 | 0.590 | 0.868 | 0.837 |
| health (R^2) | 0.122 | 0.728 | 0.065 | 0.500 | 0.650 |
| cow in view (AUC) | 0.612 | 0.626 | 0.777 | 0.630 | 0.602 |
| action from dz (acc) | 0.353 | 0.669 | 0.473 | 0.667 | 0.676 |

- Imagined facts (`ldad_facts.json`; probes fitted on each run's TRUE z, read on its imagined z, 3-frame window),
  depth 1 / 4 / 16:

| run | zombie adjacent | passability (up) | health R^2 | food R^2 | moved vs blocked | health drop |
|---|---|---|---|---|---|---|
| raw | 0.63 / 0.66 / 0.55 | 0.53 / 0.50 / 0.54 | 0.08 / 0.07 / 0.06 | 0.75 / 0.76 / 0.69 | 0.713 | 0.556 |
| tc | 0.62 / 0.66 / 0.55 | 0.56 / 0.53 / 0.53 | 0.11 / 0.10 / 0.01 | 0.69 / 0.69 / 0.61 | 0.686 | 0.614 |
| raw + LDAD 10 | **0.91 / 0.82** / 0.60 | **0.79** / 0.65 / 0.56 | **0.43 / 0.38 / 0.13** | 0.54 / 0.49 / 0.38 | 0.859 | **0.884** |
| tc + LDAD 10 | 0.75 / 0.64 / 0.60 | 0.78 / **0.68 / 0.64** | 0.24 / 0.27 / 0.01 | 0.34 / 0.35 / 0.21 | **0.878** | 0.744 |
| raw + LDAD 10, no SIGReg | 0.85 / 0.68 / 0.54 | 0.78 / 0.62 / 0.57 | 0.27 / 0.20 / 0.08 | 0.40 / 0.38 / 0.16 | 0.784 | 0.722 |

- TC's z is dimensionally collapsed on natural frames (effective rank 2.5, 6x the variance). Its SIGReg acts on
  temporally centred residuals, leaving the per-window mean unconstrained. LDAD raises it to 5.9.
- **TC + LDAD is not better than Raw + LDAD.** It keeps passability better in imagination (0.64 vs 0.56 at depth
  16) but loses zombie (0.75 vs 0.91 at depth 1), health (0.24 vs 0.43), food and health drop (0.744 vs 0.884).
- **Delta-JEPA as published partially collapses on Craftax.** Effective rank halves (11.7 -> 5.8). Encoder facts
  drop slightly (zombie 0.912 vs 0.961, health 0.65 vs 0.73); action decoding is unchanged (0.676). Imagined
  facts drop clearly (zombie 0.68 vs 0.82 at depth 4, health 0.27 vs 0.43, health drop 0.722 vs 0.884). LDAD
  alone prevents full collapse, as the paper says, but SIGReg does measurable work here. **Raw + SIGReg +
  LDAD 10 has the strongest measured true-state encoder facts among these arms; this does not establish the best rollout or actor.**
- Raw lambda 1 at 10k completed on 2026-09-29 (lane 4b); see the corrected readout section below.

**E4b common currency** (`ldad_facts.py`). Probes fitted on TRUE z per run, read on each world's IMAGINED z
(3-frame window). Imagined at depth 1 / 4 / 16:

| fact | canonical Raw | Raw + LDAD lambda 10 |
|---|---|---|
| zombie adjacent | 0.63 / 0.66 / 0.54 | **0.91 / 0.82** / 0.60 |
| pass up | 0.53 / 0.50 / 0.54 | **0.79 / 0.65** / 0.56 |
| health (R^2) | 0.08 / 0.07 / 0.06 | **0.42 / 0.37 / 0.13** |
| food (R^2) | **0.75 / 0.76 / 0.69** | 0.54 / 0.49 / 0.38 |
| energy (R^2) | **0.71 / 0.67 / 0.62** | 0.47 / 0.37 / 0.26 |
| one-step moved vs blocked from imagined z(a) | 0.71 | **0.86** |
| one-step health drop from imagined z(a) | 0.56 | **0.88** |

- On the historical true-fit transfer readout, LDAD raises several one-step decision-fact scores (zombie, passability, health drop). This is not a safe-action or actor result.
- Costs:
  - slow HUD facts (food, energy) degrade;
  - by depth 16 imagined zombie / passability fall to or below copying the root (0.60 vs 0.66; pass left
    0.51 vs 0.68);
  - prediction MSE 5.3x, SIGReg 1.9x.
- 2k screens cannot judge imagination (the world lags the encoder: raw lambda 1 true zombie 0.96, imagined 0.62).
- TC + LDAD lambda 10 at 2k restores passability in TC's z (true 0.84-0.91 vs plain TC 0.60-0.70).
- Historical status updated: TC lambda 10 and Raw lambda 1 at 10k have since completed; see the corrected readout section below.

**E4b readout-transfer correction (`ldad_facts_v2.json`, 2026-09-29).** The historical E4/E4b
numbers above are true-fit-to-generated transfer. The version-2 rescore fits the same ridge family on
generated TRAIN-seed states for each arm and judges TEST seeds. This changes some comparisons, including
negative shifts; a generated-fit score is an operational decoder score, not information proof.

| arm | zombie adjacency generated-fit depth 1 / 4 / 16 | health R² generated-fit depth 1 / 16 | generated-fit one-step moved/blocked AUC | generated-fit health-drop AUC |
|---|---|---|---:|---:|
| Raw | 0.616 / 0.657 / 0.577 | -0.010 / 0.009 | 0.717 | 0.561 |
| TC | 0.629 / 0.653 / 0.593 | 0.097 / 0.036 | 0.711 | 0.575 |
| Raw + LDAD 10 | **0.875 / 0.850 / 0.629** | **0.278 / 0.108** | 0.869 | **0.899** |
| TC + LDAD 10 | 0.753 / 0.677 / **0.640** | 0.124 / 0.001 | **0.910** | 0.711 |
| Raw + LDAD 10, no SIGReg | 0.793 / 0.740 / 0.591 | 0.121 / 0.039 | 0.867 | 0.813 |

For Raw + LDAD 10, true-fit versus generated-fit zombie AUC is 0.911 versus 0.875 at one step,
0.817 versus 0.850 at depth 4, and 0.599 versus 0.629 at depth 16. Health R² instead falls
0.425 -> 0.278 at one step. The transfer effect has no universal sign. LDAD's short-horizon gain
remains descriptive; at depth 16 the generated-fit zombie margin over Raw is only 0.052 and health
R² only 0.099 higher. No paired uncertainty for between-arm fact differences was computed here, so
these differences are leads, not resolved treatment effects. The world is not yet demonstrated to
select actions or roll out well from these z states. The original `ldad_facts.json` is preserved.

**E4b, Raw + LDAD λ=1 at 10k (2026-09-29; `lane4b.sh`).** Training finished at update 10,000
(4,278.3 s); `ldad_eval_full2` and `ldad_facts2` both finished, and the checkpoint declares
`variant=raw`, `lam=1.0`, `no_sigreg=False`. λ=10's older checkpoint declares `lam=10.0`.
The two 10k runs use the matched Raw recipe, but their latent geometries differ. The report
row is in `ldad_eval.json`; the generated-fit readout row is in `ldad_facts_v2.json`.
Readouts use the previously inspected diagnosis TEST set (283 roots from 43 walk seeds) and
same-capacity ridges fitted on generated TRAIN-seed states; this is not a fresh sealed gate.

| measure | Raw | Raw + λ=1 | Raw + λ=10 |
|---|---:|---:|---:|
| root z: adjacent zombie AUC | 0.606 | 0.818 | 0.961 |
| root z: 4-direction passability AUC | 0.588 | 0.783 | 0.855 |
| root z: health R² | 0.122 | 0.466 | 0.728 |
| effective rank of z | 15.7 | 17.4 | 11.7 |
| generated-fit adjacent zombie AUC, depth 1 / 4 / 16 | .616 / .657 / .577 | .719 / .717 / .598 | .875 / .850 / .629 |
| generated-fit health R², depth 1 / 16 | -.010 / .009 | .157 / .035 | .278 / .108 |
| generated-fit one-step moved/blocked AUC | .717 | .850 | .869 |
| generated-fit one-step health-drop AUC | .561 | .665 | .899 |
| generated-fit food R², depth 1 / 16 | .758 / .720 | .773 / .648 | .617 / .556 |
| generated-fit energy R², depth 1 / 16 | .675 / .628 | .674 / .554 | .479 / .411 |

λ=1 does retain materially more root-side zombie, passability and health information than Raw,
and preserves more food/energy information than λ=10; λ=10 still has stronger safety-related
readouts, especially one-step health drop (.899 versus .665). Both lose a large part of
root-side adjacent-zombie signal during imagination: at depth 1, λ=1 generated .719 versus
copy-root .825 and real successor .904; λ=10 generated .875 versus copy-root .933 and real
successor .992. At depth 16, λ=1 generated .598 versus copy-root .640; λ=10 generated .629
versus copy-root .661. Thus λ=1 does not close the transition-retention defect. The physical
metrics within each latent space are λ=1 one-step error/copy .154 and depth-16 error/V .521,
versus λ=10 .237 and .615; the changing geometry makes these *cross-arm* scalar ratios
unsuitable as a quality ranking. The AUC/R² contrasts are descriptive on the already inspected
TEST block; no paired between-arm interval or all-action safe-choice result exists here.
Neither dose is ready for a canonical or actor claim. No additional lane is running after
`LANE4_DONE`; the next expensive run should be selected only after a decision on these tradeoffs.

**LDAD gradient localization (`ldad_grad.py/json`, 2026-09-29).** On the same two TRAIN
batches of 128, with the canonical prediction + 0.09 SIGReg and the checkpoint's action
CE differentiated separately through the encoder, the weighted CE gradient norm divided
by the prediction-plus-SIGReg gradient norm is:

| checkpoint | batch 1 | batch 2 | cosine(total encoder gradient, CE gradient), range |
|---|---:|---:|---:|
| common initialization, λ=1 | .022 | .014 | -0.142 to -0.030 |
| λ=1, update 2k | 1.725 | 1.092 | .697 to .860 |
| λ=10, update 2k | 6.461 | 6.039 | .986 to .989 |
| λ=1, update 10k | 3.445 | 3.031 | .946 to .965 |
| λ=10, update 10k | 7.721 | 18.242 | .992 to .999 |

The LDAD CE is computed from **encoder** z differences; there is no direct CE gradient to
world parameters in `ldad_joint.py`. At 10k both doses are action-gradient-dominated on
these batches, substantially more so at λ=10. Over updates 8,001–10,000 the logged
prediction loss averaged .0408 (λ=1) versus .0998 (λ=10), while action accuracy averaged
.6055 versus .6078. Logged pre-clip global gradient norm averaged 3.21 versus 29.78,
with the declared clip at 1.0. This identifies the objective pressure that differs: λ=10 spends
more encoder gradient on action discrimination with nearly no training-accuracy gain and
higher latent prediction loss. It does not identify why that pressure preferentially
retains zombies or prove that the resulting optimizer *update* has the same direction:
AdamW preconditions gradients after global norm clipping, and this diagnostic uses two
sampled batches at three checkpoint stages, not a full trajectory of parameter updates.
The Delta-JEPA paper's own Push-T sweep also reports degradation for excessively high λ,
but its continuous-control objective omits SIGReg and is not a Craftax threshold.

**Incident (2026-09-27 12:10):** the machine hung and rebooted with 5 jobs running (no OOM-killer record; no swap).
- `ldad_eval.py`'s probes loaded four full judgement blocks (~13 GB of rows) twice.
- Fix: one lean loader (`tc_encoder.blocks()`: visible state + last frame only).
- Since then every job runs in a systemd unit with MemoryHigh/MemoryMax, in at most 3 lanes:
  - A: `queue_tworlds.sh`, 9/12 GB.
  - B: `lane_b.sh`, 15/17 GB.
  - C: E1b + E1c, 6/8 GB.
- Measured process (anonymous) memory: 1.3-2.0 GB per lane; the rest is reclaimable corpus page cache.
- Lost and restarted: the residual T arm (2,500/6,000 updates) and the Raw LDAD run (487/10,000).

**E3a (`tc_equiv.json`):** the tokens carry the same facts but are NOT linearly equivalent.
- Linear R^2: raw -> tc 0.83, tc -> raw 0.89; on the player token 0.48 / 0.70. CKA 0.63.
- Facts: tile 0.990 / 0.989, zombie near 1.00 / 1.00, health 0.999 / 0.997.
- Linearly learnable change: blocked -2.09 vs -1.07, idle -0.21 vs +0.03.

**E2 codebooks (`codebook.json`):**
- Raw K = 4096: relative error 0.033; tile facts 0.96-0.98 (continuous 0.97-0.99); health R^2 0.94 (1.00).
- Code flips on tiles where nothing changed: 14.7% for Raw tokens, 27.6% for TC tokens (contextual ViT tokens).

---

## 2026-09-27 — Diagnosis part 2 (`20260926_diagnosis/TRANSITION.md`, commits 6a744366, a9ee14d1)

- **The one-step failure is on each world's own TRAIN data** (`insample.json`); held-out is within 0.05.
  - All transitions, x copy: H2 1.34, Z 0.61, sZ 1.17, U 0.57, W 0.97, T 0.89.
- **Per state** (`learnable.json`, `oracle.json`):
  - z: a move's change is not a function of z; ridge captures 0.08.
  - u: the worlds sit at u's linear ceiling (0.63); u lacks passability (AUC 0.69).
  - T: the information is present (local-linear 0.72, rigid-shift oracle 0.83), but the trained world captures
    0.08.
- **T/sZ cannot beat copying because the output has no copy path** (`parameterization.json`). Held-out L1:
  - T: direct 0.141, copy 0.122, residual 0.112.
  - sZ: direct 0.097, copy 0.077, residual 0.069.
- **Canonical H2: bridge broke it; the Mamba does not extrapolate its 4-frame training** (`bnmode.json`,
  `context.json`, `memory.json`).
  - x copy: joint 0.66, bridge 2.01. BatchNorm mode is ruled out.
  - The Mamba state grows 3.2-4.4x past the training length.
- **LeWM's own rule repairs the joint world** (`sliding.json`): le-wm rollout uses `emb[:, -3:]`.
  - Joint world, 16-step imagined error / V: 2.57 -> 0.53 (copy 0.93); one step 0.048 -> 0.025.
- **Recursion compounds because the latent is a near-integrator** (`compound.json`, `persistence.json`):
  - 81-92% of depth-16 error is inherited, passed on at 0.82-0.97 per direction.
  - True autocorrelation 0.92-0.99 (Lambert et al. 2022).
  - Fresh errors are correlated (cos 0.48-0.72); growth exponent 1.18-1.28.
  - Error sits 1.5-22x in low-variance directions, and HUD facts end up below copying.
- **TC encoder** (`tc_encoder.json`):
  - z within-window variance 1.6% -> 44.8%; mobs up to 5x more detectable.
  - Static terrain nearly dropped (d' 0.1); passability unchanged (0.59).
- **Correction (E3a above):** "TC and Raw patch tokens are identical" was too strong. Same facts, different
  geometry.
- **Correction:** head-only whitening on the frozen U world gives +0.01 SLEEP-death AUC, of W's 0.09
  (`headonly.json`). The LeJEPA-Lemma-1 account of W is not supported.

## 2026-09-26 — Diagnosis part 1 (`20260926_diagnosis/DIAGNOSIS.md`, commit 8e95f745)

- **The zombie decision is a visible rule**: move into a free tile. A rule on visible tiles scores 0.977-0.982;
  the hidden cooldown barely matters (`decision.json`).
- **True-successor substitution**: U 0.975-0.981, W 0.990-0.999 against 0.49-0.64 imagined. For U and W the
  whole loss is imagination (`choices.json`).
- **z is position-blind**: cosine 1.000 for an adjacent vs a far zombie. Training erased zombie presence
  (AUC 0.947 -> 0.528) (`twins.json`).
- **Heads underrate SLEEP**: only 3.2% of corpus death frames render asleep. H2 picks SLEEP on 74-79% of
  zombie roots (`sleep.json`, `followups.json`).
- **W > U** is about 79% not choosing SLEEP (`followups.json`).
- Other agent's audit (`20260926_mechanism_audit/FINDINGS.md`):
  - Lava-entry mislabel corrected.
  - CLS position blindness already exists at initialization.
  - Positional embeddings are ~8% of the patch-embedding norm.

## 2026-09-21..26 — Readout ladder (`20260921_readout_ladder/`, WHY.md, INTERFACE.md)

- Sealed 59k: the whitened patch-state Mamba world's own head beats DOWN (+0.083*) and actions_only (+0.056*),
  and beats DOWN on zombie roots (+0.059*). Positive in 7/7 blocks, resolved on zombie roots in 2/7.
- Factorial (post hoc 55k-58k): only mob-in-state + isotropy works (W 0.54-0.64 vs Z/ZW/U).
- Canonical SIGReg z is not eigen-isotropic: spread 6,220x, effective rank 44/192.
- H2 depth alias: P(dead) 0.73 at depth 2, 0.05 at the gate position.
- Superseded by the diagnosis above:
  - "tail learned last" was retracted (ridge oracle).
  - The isotropy/Lemma-1 mechanism for W is not supported (`headonly.json`).
