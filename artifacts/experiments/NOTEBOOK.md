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

## 2026-10-08 (live) — WHY HEALTH FAILS: link-by-link diagnosis (lead; user mandate: exact cause before any ablation)

**Decisions recorded:** Mamba (fmamba, L16) is permanently canonical. E20 services stopped and disabled 08:20.

**Setup.** `health_chain.py` fixes a subset of the E20 factual TRAIN pool (seed 20261008), split fit / held by episode:
- all 1,005 fresh ordinary hits;
- 5,000 other ordinary hits (living dh ≤ −2);
- 8,000 unchanged;
- 1,000 deaths.

Frozen worlds are teacher-forced on their trained context (L16: 15 frames). `health_chain_read.py` / `health_refit.py` read
health with teval's HUD ridge (drop cut 1.5). Controls: the true next frame catches 5,988/6,005 hits; copy catches 0.

**Facts, canonical Mamba M16 s7 (held episodes unless stated):**
1. *The target is easy.* The true token-63 change for a given health transition is nearly constant: cosine to the group
   mean 0.98-0.99, relative residual 0.14-0.20. "Copy + group-mean change" is read as the right drop in 100% of held cases.
2. *Upstream dynamics are mostly right on fresh hits* (all 1,005): emitted scroll correct 99.0%; zombie drawn beside the
   player 82% (true 99.8%). Yet the damage is drawn 8/1,005 (0.8%). Not mainly an upstream failure.
3. *The information is in h.* Hit vs unchanged probes on h (63, 31, 4 neighbours), MLP: AUC 0.958; fresh 0.941; fresh vs
   hard negatives 0.884 (raw-input probes 0.905 / 0.873 / 0.713).
4. *A frozen-h head can depict hits*, but only when its training class mix is not the natural one:

| token-63 head on frozen h (held) | hits / 1,211 | fresh / 207 | false / 1,728 |
|---|---|---|---|
| emitted (trained corrt) | 154 | 4 | 127 |
| dedicated linear, h63, TRAIN hit rate (1.4%) | 83 | 1 | 13 |
| dedicated MLP, h63, 1.4% | 81 | 0 | 11 |
| copy + residual, h63, 1.4% | 17 | 0 | 6 |
| linear, 6 positions, 1.4% | 81 | 6 | 16 |
| linear, h63, subset mix (43% hits) | 723 | 61 | 124 |
| linear, 6 positions, balanced | 899 | 140 | 124 |

5. *The world's own generator alone* (token 63 := generated) draws on 54% of hits but also 51% of unchanged. The router
   suppresses it, and copies the HUD token below health (weight 0.28) on stationary zombie-beside states, which yields
   spurious drops (42% of those no-hit states vs 17% of stationary hits).

**Reading so far.**
- At the training class rate, every head type (corrt, linear, MLP, residual) copies. The corr / corrg head is not the
  bottleneck.
- L1 returns the per-dimension conditional median, so a deterministic head draws damage only when P(hit | features) > 0.5.
  At a 1.4% base rate that needs a likelihood ratio above ~70:1, which this h reaches for only a few percent of hits.
- Open: is the evidence weak in h (representation), or intrinsically uncertain given the visible history? Fresh hits are
  99% determined by the visible history, so a sharp representation would draw them. Next: h at every cell within 2 and the
  action token (the fresh zombie sits 2 away before the move); prior-corrected posteriors per stratum; all worlds.

### Diagnosis 2-5 (09:00-10:00): two blockers, one root

**Corrections to the reading above (each measured below):**
- "The corr / corrg head is not the bottleneck" holds only for the *decision rule* on a frozen h. The trained corrt head has a
  blocker of its own (B2).
- "The information is in h" (AUC 0.958) is a ranking statistic. At the training base rate what decides drawing is the calibrated
  posterior: h63 gives P(hit) > 0.5 for only 24% of held hits (B1).
- E20's "fresh hits are 99% hits" holds in its capped pool only. Population-weighted, the fresh stratum's rate is 0.58: the
  8 fresh-flagged no-hits in the pool are detector misfires (zombie scores 0.30-0.48 vs the hits' median 0.74; 3 have action 6).

**Is a hit predictable from the visible history at all?** (`health_strata.py`; data only; TRAIN mix at t ≥ 14 restored by
population weights: fresh hits ×1, other hits ×7.01, unchanged ×361.4.) `adj_post` = a zombie beside the player's post-move
cell in frame 14 (true scroll); `since` = transitions since the last health drop.

| stratum | population hit rate | share of all hits |
|---|---|---|
| fresh (E20 flag) | 0.58 | 2.8% |
| adj_post, since = 6 | **0.87** | 14.0% |
| adj_post, since 1..5 | **0.008** | 5.7% |
| adj_post, since ≥ 7 | 0.63 | 12.0% |
| adj_post, no drop in the window | 0.67 | 43.4% |
| not adj_post | 0.003 | 22.2% |

The zombie cooldown is crisp in the data: given adjacency, the hit rate by `since` is 0.018 / 0.009 / 0.007 / 0.007 / 0.007 at
1-5 and 0.87 at 6. About 72% of hits sit in strata whose rate exceeds 0.5, so a deterministic median head *could* draw them.

**B1 — information: h63 lacks the cooldown phase and the approach adjacency.** (`health_evidence.py`: balanced-batch MLP
probes, early stop on an inner episode split, weighted Platt calibration to the population mix; drawable = P > 0.5 on held hits.)

| features (M16 s7, held) | drawable hits /1,211 | fresh /207 | since6 /146 | false /1,728 |
|---|---|---|---|---|
| V: visible statistics (zombie score beside post-move cell, since, adj history, health) | 742 (61%) | 172 | 146 | 1 |
| V without `since` | 364 | 126 | 9 | 3 |
| h63 (the corrt head's only input for token 63) | 286 (24%) | 4 | 21 | 5 |
| h_ext (h at 63, 31, 12 cells within 2, action token) | 255 | 22 | 27 | 7 |
| h63 + oracle `since` | 480 | 6 | 88 | 3 |
| h63 + oracle `since`, adj_post, adj_hist | **768 (63%)** | 136 | 124 | **0** |
| the trained world's own emission | 154 (13%) | 4 | 20 | **126** |

- Substitution: supplying the two ingredients lifts h63 from 24% to 63% drawable at zero false drops, above V.
- Positive control (the same probe, same data): the cooldown phase is recoverable from the TRUE token-63 history the world sees
  (frames 0..14) at AUC 0.9993 (since = 6) / 0.9973 (1..5) / 0.9999 (≥ 7). From M16's h63: 0.888 / 0.909 (1..4) / 0.789.
  The world sees the phase and does not keep it at token 63.
- Approach adjacency (adj_post on scroll steps with a zombie within 2): h63 AUC 0.778, h_ext 0.881; stationary 0.997.
- Probe validity: PCA-reduced h_ext loses the evidence (32 PCs draw 1 hit); adding oracle ingredients to h_ext under-performs
  h63 + oracle (data-limited 3,840-d probe). Drawable figures from h are lower bounds; the ingredient AUCs and the positive
  control are the precise statements.

**B2 — expression: the trained corrt generator cannot produce token 63.** corrt's token-63 output is a softmax mixture of the
input token (copy), four spatial neighbours and `proj(h63)` (one linear map shared by all 81 tokens, trained only through its
mixture weight). Its LayerNormed candidate vs the true next token, held, per dimension L1 (copy: 0.486 on hits, 0.036 on unchanged):

| world | generator L1 hits / unchanged | generator beats copy, share of hits | gen weight on hits | projection on true change / orthogonal energy |
|---|---|---|---|---|
| M16 s7 / s8 | 0.917 / 0.947; 0.925 / 0.960 | 2.0%; 0.1% | 0.12; 0.14 | 0.53 / 3.48; 0.56 / 3.44 |
| A16 s7 / s8 | 0.908 / 0.944; 0.873 / 0.892 | 1.3%; 0.5% | 0.09; 0.13 | |
| M6 s7 at 6k, 12k, 18k, 24k, 30k, 36k | 1.030, 1.001, 0.984, 0.975, 0.971, 0.952 | ≤ 0.2% | | |
| A6 s7 at 6k, 36k, 50k, 100k | 1.075, 0.931, 0.896, 0.863 | ≤ 1.0% | | |
| E19 C (health dose λ 1, 6k from M6 36k) | **0.659** / 0.886 | **29.1%** | 0.23 | 0.63 / 1.65 |

- A generator L1 near 0.9-1.0 on unit-variance tokens is close to unrelated content (independent tokens: ~1.13). It carries
  half of the health change plus 3.3-4.2× its energy in unwanted change, so committing to it costs more than copying on
  98-100% of hits, *whatever h63 knows*. The world can only blend: partial depictions read as 154 hits and 126 false drops.
- h63 does hold the token's content: a dedicated linear map from the same frozen h63 reaches L1 0.22-0.25 on hits
  (`health_refit.py`); the world's own shared proj on h63 gives 0.93 / 0.95.
- Fidelity responds to objective allocation, not to training length: natural dose 36k → 100k moves it 0.931 → 0.863;
  E19 C's dose moves it 0.952 → 0.659 in 6k updates.
- The corrt move logit is not involved (M16 s7 / s8, A16 s7, M6 s7): token 63's up / left / right weights are ≤ 0.001 on
  scroll and stationary steps alike, and its down-neighbour (token 72) weight is 0.05-0.08 on both. On fresh hits the head is
  most copy-confident (self 0.91, gen 0.09).

**Root: the objective allocates almost nothing to health.** (`health_gradient.py`: M16 on its own training batches, the exact
replay of tworld.train's sampler, seed-11 order, 20 batches = 12,000 transitions, 341 hits; teacher objective err.mean().)

| M16 | token 63, share of objective | token 63 on hits, share of objective | hits' share of backbone gradient norm | cosine with full gradient |
|---|---|---|---|---|
| s7 | 0.80% | 0.27% | 0.16% | 0.096 |
| s8 | 0.83% | 0.28% | 0.42% | 0.054 |

- Both blockers are what an objective that puts 0.3% of its mass on the event predicts: the backbone keeps no phase at token 63,
  and the shared generator, which sees token 63 through a mixture weight of 0.05 (unchanged) to 0.24 (hits), never learns it.
- The same mechanism explains E16 (Delta-IRIS port). Its posterior sees the TRUE next frame, yet before VQ it encodes ordinary hits
  at AUC 0.634 / 0.701, and its code at 0.548 / 0.642. Deaths, about 3-4× larger token changes and about 3× more frequent per pass,
  are encoded and drawn (generate weight 0.51-0.89, error 245 → 16.5). E16 trained the posterior through uniform latent L1. Delta-IRIS's own
  tokenizer loss is 1.0·L2 + 0.1·L1 + 0.01·worst-pixel L2 (vmicheli/delta-iris `tokenizer.py`), which concentrates gradient on
  rare large local errors such as a heart icon. That was dropped in the port.
- E20 B could not have moved the decision, by design: it oversampled damage (10/40 per batch) but importance-weighted it back
  to its natural loss mass (class mass 0.0129825), so the L1 median and the per-hit incentive were unchanged in expectation.
- E19 C's dose did move B2 (above), but in 6-frame windows: the phase (since = 6) lies outside the window, so B1 could not
  be met for most hits. Every fresh hit is a scroll step; its generator still lost to copy there (0/51 scrolling hits caught).

### Diagnosis 6 (10:00-10:30): across worlds, training length, positive controls

`health_evidence.py main` (`evidence_main.json`), 14 worlds, same probes / calibration / held episodes:

| world | drawable from h63 | + oracle since | + since, adj | `since` 1..4 in h63 | since = 6 in h63 | emitted hits / false |
|---|---|---|---|---|---|---|
| M16 s7 / s8 | 286 / 281 | 480 / 353 | 768 / 747 | 0.909 / 0.898 | 0.888 / 0.878 | 154 / 126; 208 / 151 |
| A16 s7 / s8 | 128 / 182 | 286 / 386 | 639 / 752 | 0.847 / 0.844 | 0.869 / 0.865 | 36 / 122; 125 / 121 |
| E20 B s7 (damage oversampled, natural mass) | 395 | 496 | 695 | 0.952 | **0.950** | 87 / 13 |
| M6 s7 at 6k, 12k, 18k, 24k, 30k, 36k | 0, 142, 186, 200, 245, 164 | | 805, 693, 801, 770, 790, 747 | 0.915, 0.896, 0.889, 0.897, 0.900, 0.904 | (outside window) | |
| A6 s7 at 36k, 50k, 100k | 135, 91, 91 | | 711, 627, 725 | 0.848, 0.849, 0.848 | (outside window) | |

Positive control (true token-63 history over the world's own window): `since` 1..4 0.997-0.999, since = 6 0.999.

- **Training length does not grow the phase.** Under the natural objective, `since` 1..4 in h63 is flat from 6k to 36k (Mamba) and
  from 36k to 100k (attention), against 0.997 available in the input. Drawable-from-h63 stops rising after ~30k. The user's
  "6k too short / grokking late" hypothesis is refuted for the phase; extra *exposure* to damage (E20 B) moved it (0.950).
- **The phase is computed, then discarded** (M16 s7; slot 63's residual stream at the output position, same probe, held AUC):

| slot 63 after | since = 6 | since 1..4 | since ≥ 7 |
|---|---|---|---|
| input embedding (current frame only) | 0.844 | 0.830 | 0.772 |
| layer 0 / 1 | 0.859 / 0.863 | 0.928 / 0.940 | 0.785 / 0.802 |
| layer 2 / 3 (peak) | 0.918 / 0.916 | 0.968 / 0.961 | 0.832 / 0.833 |
| layer 4 / 5 (output) | 0.911 / 0.887 | 0.943 / 0.911 | 0.814 / 0.790 |

  The middle layers build part of the phase from slot 63's history. The last two layers, which feed the copy-dominated head,
  lose a third of it. That is what a missing incentive predicts, not an inability of the per-slot Mamba to compute it.
- **The ingredients are sufficient in every world.** h63 + oracle phase + adjacency makes 627-805 of 1,211 hits drawable at
  ~0 false in all 14 worlds, including M6 at 6k.
- **Correction (approach adjacency).** The adjacency deficit claimed in B1 is NOT established. Positive control on approach
  moves (scroll steps, zombie within 2): raw frame-14 cells + action 0.524, + the true scroll 0.786, M16's h63 0.778. The subset
  is too small for this three-way conjunction; h63 matches the raw + true-scroll control. Only the phase deficit is established.
  The world's map draws the zombie beside the post-move player in 82% of fresh hits.
- **Allocation decomposition** (an identity, not a check): token-63 hit elements are 0.035% of the training elements; their L1
  is 7.7-8.0× the mean element error; product 0.27-0.28% of the objective. L1's per-element gradient is ±1 whatever the error
  size, and contextual continuous tokens never sit at exactly zero error, so an event's gradient tracks its element share.
- **Matched heads at 6k** (attention, teacher, window 5): residual (no mixture gate) draws 23 / 1,211 hits with 328 false;
  corrt 15 / 245; direct (suffix) 9 / 75. No head type rescues hits under the same objective.
- **The generator loss fixes B2 on its recipe.** `--gen-loss` (the generator's own L1 on every token, ITC-derived, 6k suffix,
  attention): token-63 generator L1 on hits 0.973 → 0.491, unchanged 1.005 → 0.209; beats copy on 0% → 36% of hits. It was
  judged in 09-28 on aggregate one-step error only; its effect on health was never read.
- **Existing checkpoints that bear on the blockers** (`evidence_more.json`; 6-frame worlds read at window 5):

| world | drawable from h63 | emitted hits / 1,211 | false / 1,728 | reading |
|---|---|---|---|---|
| corrt suffix 6k / + gen-loss | 0 / 0 | 21 / 12 | 250 / 282 | B2 fixed alone, no B1 evidence yet: nothing drawn |
| corrt suffix 18k / + gen-loss + ITC regions | 15 / 6 | 80 / 19 | 235 / 189 | |
| E19 B (no dose) / C (health dose λ 1), 6k from M6 36k | 264 / **416** | 170 / **376** | 118 / 133 | a health-token dose moves both blockers |
| M16 at 500 / 6,000 L16 updates | 160 / 286 | 32 / 154 | 30 / 126 | since = 6 in h63: 0.868 → 0.888 |
| E20 A (endpoint-only continuation of M16) | 212 | 98 | **2** | since = 6 0.937; class-balanced endpoints remove false drops |

  A health-specific dose (E19 C) draws 31% of held hits at window 5. The question for E21 is whether a *generic* allocation
  does the same.
- **Token-change events are bimodal for HUD tokens** (400 random TRAIN windows, frames 8..55): health-63 change norms cluster
  at 0-2 (drift) and 3-12 (real changes) with an empty gap at 2-3 (4 and 11 of 18,800 in the 2.0-2.5 / 2.5-3.0 bins). At a
  change-norm threshold 2.5, events cover health 5.3%, all HUD 3.2%, map 28%.

### Literature step (10:00-10:40), on the diagnosed problem

Problem as diagnosed: a rare (1.4%) state change on one token of 81, under a uniform per-element regression objective;
(B1) the representation does not keep the hidden variable that predicts it (the cooldown phase), and (B2) the copy-gated
generator never learns to produce the token.

- **EAWM** (arXiv 2601.19336, ICLR 2026; code MarquisDarwin/EAWM @ 49eebbe, read):
  - Mechanism: an event head on the latent (the decoder's input) predicts per-element change events with focal loss; its
    gradients flow into the world model. The "no event predictor" ablation costs ~0.4 mean HNS (Atari).
  - Craftax (EASimulus, `config/world_model/craftax.yaml`, `world_model.py`):
    - modalities: map = token_2d (event = category changed, balance factor 0.25), stats = vector (tendency up/flat/down),
      direction = token;
    - focal α 0.15, γ 4;
    - per-row sparsity weight GES(p) = 1/log(0.1 + p + √(1+p²)), ≈ 10 for sparse rows, ≈ 1.1 at p = 1;
    - modality weight 0.1;
    - total loss = obs + reward + end + events, unit weights.
  - Our E14a "EAWM" was a ×1.7 token reweighting on a frozen backbone (DO / place). It tested neither component.
- **Delta-IRIS** (vmicheli/delta-iris `tokenizer.py`, read): decoder loss 1.0·L2 + 0.1·L1 + 0.01·worst-pixel L2. E16 replaced
  it with uniform latent L1.
- **ITC** (2605.16457, appendix B.2): HUD and screen edges are generated by the transformer with their own loss, never through
  a copy competition. That is B2's remedy, as `--gen-loss` confirms.
- **Dedieu et al.** (2502.01591): CE on non-contextual NNT patch codes, nothing health-specific. Under CE, confidently correct
  unchanged codes give ~0 gradient. Under latent L1 on contextual tokens, every element keeps a ±1 sign gradient.
- **HarmonyDream** (2310.00344): balances reward vs observation loss scales. A per-token harmonizer would barely move token 63
  (its mean error is already 0.65× the average); the imbalance is *within* the token.
- **Imbalanced regression / dense regression**:
  - Yang et al. 2021 (DIR), Ren et al. 2022 (Balanced MSE): re-balancing shifts the predictor's prior, so it hallucinates at
    the true rate. E14 measured exactly that for positives-only doses.
  - Shrinkage / regression-focal losses (Lu et al. 2018) down-weight easy elements; E14's hard-mining dose failed on our
    tokens (93% of the tail is entering / static / HUD).
- Vendored papers checked:
  - Delta-JEPA (2606.31232) addresses encoder collapse; MoP-JEPA (2607.05238) addresses multimodal futures.
  - Hansen & Wang (2606.27326) find that hallucination concentrates in low-coverage state-action regions, and fix it with
    coverage-aware sampling at training time. This is related to E20 B's exposure effect on the phase (0.950), which was
    obtained despite importance weights restoring the natural loss mass.

**Generic remedies with a primary source, one per blocker:**
- B1: EAWM's event head. It is a matched-negative classification loss, so the L1 median that decides drawing is unchanged,
  while the representation is pushed to predict *when* each token changes.
- B2: the generator's own loss (`--gen-loss`, ITC).

### E21 — predeclared before any code (10:40): event head (B1) × generator loss (B2), 2 × 2 on M16

**Arms.** Each continues M16 s7 (canonical; L16, b40, rawlong, teacher, seed 7, batch order 11, fresh AdamW with warm-up) for
6,000 updates:

| arm | flags |
|---|---|
| C0 | none (control: the continuation itself) |
| G | `--gen-loss` |
| E | `--event` |
| GE | both |

Snapshots at 2k / 4k show trends, so a null cannot hide a slow trend.

**`--event` (port of EASimulus's event loss to per-tile tokens):**
- Event label: token i changes from t to t+1 by L2 norm > 2.5. The threshold is the measured valley of the bimodal HUD
  distribution; the analog of "the category changed".
- Head: per token, shared across tokens, reading h_i, the corrt decoder's own input for token i. This is EAWM's principle
  "the event predictor reads the decoder's latent". Shape Linear(256, 4) → LayerNorm → SiLU → Linear(4, 1), EASimulus's
  per-element hidden width outeventdim = 4.
- Loss: focal α 0.15, γ 4; GES per row and modality (map: 63 tokens, balance 0.25; HUD: 18 tokens, balance 1.0); modality
  weight 0.1; added to the teacher L1 with weight 1.
- Deviation: HUD events are binary change events, not up/flat/down tendencies; our HUD is tokens, not values.

**`--gen-loss` fix:** it scored `gen[:, :S.W-1]` (6-frame era). At L16 it must score all L-1 targets. The fix is a no-op for
every past 6-frame run (S.W = 6).

**Readings** (s7; health_chain subset, held episodes; window 15):
1. *B2 fixed*: G and GE token-63 generator beats copy on ≥ 30% of held hits (M16 2%).
2. *B1 fixed*: E and GE, since = 6 AUC in h63 ≥ 0.95 (M16 0.888, input 0.999), and drawable-from-h63 ≥ 2× C0's.
3. *Health drawn*: GE emitted ≥ 30% of held hits (363 / 1,211) with ≤ 2% false (35 / 1,728). Control C0 is reported alongside.
4. *Cost*: teval one-step all / depth-16 vs C0, paired. The margin is 0.005 one-step, as in E19. A failed margin is reported,
   not hidden.

**Interpretation rules, fixed now:**
- If B1 and B2 both move and health is drawn, the cause is confirmed causally. A from-scratch run (s7 + s8) follows overnight
  to remove the continuation caveat.
- If an intermediate does not move, `health_gradient.py` measures whether the arm actually changed the allocation, before any
  conclusion. If the allocation moved and the ingredient did not, the next test is from scratch, not "the method fails".

**E21 v1 aborted (16:44 → 17:21), caught by an in-flight check at 2k updates of GE:**
- B2 was already fixed (token-63 generator beats copy on 78% of held hits; L1 0.471 vs copy 0.486).
- The event head's token-63 logit was constant: AUC 0.505, p = 0.245 for hits and unchanged alike.
- Cause, a porting defect: v1 shared ONE 4-unit head across all 81 slots, so the dense map events own it. EASimulus's
  MultiMotDecoder gives every event element its own output weights: Linear(all tokens, current ⊕ previous-detached → 4·n) →
  LayerNorm(4·n) → SiLU → Linear(4·n → n).
- Fix: `tworld.SlotEvent`. Per slot, Linear([h_i,t ; sg(h_i,t−1)] → 4), one LayerNorm over the 81 × 4 units, SiLU, per-slot
  Linear(4 → 1). The aborted files are kept in `levers_tworlds_v1/e21_v1_aborted/`.

**Pre-flight** (`event_preflight.py`, `event_preflight.json`; frozen M16, head-only training on M16's own batches):

| head, loss, steps | token-63 AUC hit vs unchanged (held subset) | fresh vs unchanged |
|---|---|---|
| v1 shared, E21 focal, 600 | 0.478 | 0.533 |
| SlotEvent, E21 focal, 600 / 3,000 | 0.606 / 0.740 | 0.322 / 0.483 |
| SlotEvent, plain BCE, 600 | 0.508 | 0.455 |
| per-slot linear, BCE / focal, 600 | 0.621 / 0.647 | 0.393 / 0.398 |

- What a token-63 event is (600 random TRAIN windows, 2,357 events = 6.2% of transitions):

| event type | share of token-63 events |
|---|---|
| health drop > 1.5 (hits, deaths) | 36% |
| health −1 | 10% |
| recovery +1 | 30% |
| health unchanged, almost all at action 6 (sleep darkens the whole HUD; the other HUD tokens change 6.9 vs 0.5) | 25% |

  The generic label is meaningful, but hits are a minority of it, and the head learns them slowly.
- Allocation at EAWM's published coefficients (SlotEvent after 600 head steps; 10 other training batches, 166 hits): the event
  term pushes h at slot 63 on hits with **0.38×** the teacher L1's gradient.
- Consequence, recorded before relaunch:
  - E (and E's half of GE) is a faithful test of EAWM *at its published coefficients*. The measured dose predicts a small
    effect on B1. A weak E result would mean "dose 0.38×", not "event prediction cannot install the phase".
  - A dose-matched event arm is the follow-up if E is weak.
  - G is the clean test of B2's mechanism. With the generator usable, emitted hits should approach what h63 can support
    (24% drawable).

**E21 v2 launched** with SlotEvent; order G, C0, GE, E.


## 2026-10-08 — Audit of the 10-04..10-08 runs after handover (Claude): verified, corrected, completed

**Method.** Every headline number below was re-derived from raw rows, logs or per-root files, not from summaries:
- sighting / carry rows;
- check_delta, E16 and recall / damage logs;
- the four E17 H16 stores' per-root safe arrays;
- fixed-clock rows;
- all 12 E19 per-root health files;
- E18 recall logs;
- E20 endpoint JSONs.

All reproduce exactly, except where noted. Tracked code edits were diffed: numerics are unchanged apart from two tied-rank AUC
fixes. Everything is committed in stages 1-8 (ae0868ec..689262c1). Resume journals, `.pt` rows and EDA outputs stay on disk,
outside git. E20's JSON reports are copied to `20260927_levers/evals/e20_v1/`.

**Predeclared readings, final ledger.**

| experiment | reading | verdict | numbers |
|---|---|---|---|
| E17 stage 1 | m6_onestep | TRUE | M6 − A6 −0.002 / −0.005, both resolved (moved −0.004 / −0.011) |
| | m6_depth16 | FALSE | s7 −0.011 [−0.022, +0.001]; s8 −0.018 resolved |
| | m6_hits | FALSE | 0.050 vs 0.030; 0.036 vs 0.050 |
| | m6_h16_traj | FALSE | trajectory M6 − A6 −0.0010 / −0.0045 (0.6441 vs 0.6451; 0.6417 vs 0.6462), same 1,139-root panel |
| E17 stage 2 | long_hits | FALSE | window-15 teacher catch ≤ 0.136 (needs ≥ 0.3) |
| | long_fresh | FALSE | 0/31, 1/31 |
| | mamba_long_edge | FALSE | hits +0.071 / +0.089 pass; H16 trajectory +0.0053 [−0.004, +0.016], +0.0031 [−0.006, +0.012] |
| | long_recall_same | TRUE | Mamba same-slot age 6-15 capture 0.616 / 0.643 vs attention 0.431 / 0.357 |
| | long_recall_used | TRUE | window 15 − 5: +0.247 / +0.279 |
| | moved_unsolved | FALSE as declared | mis-specified: ignored the no-memory baseline (0.38-0.43 at window 5); moved-slot memory gain is +0.01..0.04 in every world, i.e. none |
| check_delta | all three | FALSE | mean-Δ AUC 0.564 / 0.532; disagreement 0.642 / 0.651 |
| E16 | health readings | FALSE | posterior 0/305, 0/32, 0/112 (full s7) |
| E18 | c6_moved_recall | FALSE at s7 | +0.010 (needs +0.15) |
| | c6_same_recall | FALSE at s7 | −0.528 (needs ≥ −0.05) |
| | c6_unseen | TRUE at s7 | −2.8% |
| E19 | learnable | FALSE in all 4 sets | best C catch 0.24 with 1.4-3.1% false; fresh 0/31 in all 12 arms |
| | unshortcut | FALSE | attention s8 C passes the ratio spuriously (same catches with no history) |
| | no_cost | FALSE | B → C one-step +0.007..0.012, depth 16 +0.017..0.046 |
| E20 (seed 7 pilot) | every adoption criterion | FALSE | see below; seed 8 held |

- E18 needs both seeds on the two failed readings, so seed 8 (parked at 24k) cannot change their verdict. E18's c6_unseen is
  undetermined at s8.

**Gaps found and filled.**
- lane73 never reached its predeclared teval / compare tail: the 10-06 20:55 reboot stopped it after the H16 job, which another
  script later resumed. lane95 ran exactly those commands today. M16 s7's teval reports already existed (produced 10-07 by
  E20's pipeline), so lane95 skipped them.
- E17 completed comparators (compare.py, 143 seed clusters, window 15):

| contrast | s7 one-step all | s7 depth 16 | s8 one-step all | s8 depth 16 |
|---|---|---|---|---|
| A6 → A16 | −0.005 * | −0.012 [−0.025, +0.001] | −0.003 * | −0.022 * |
| M6 → M16 | −0.004 * | −0.017 * | −0.004 * | −0.025 * |
| **A16 → M16** | −0.001 * | **−0.016 [−0.027, −0.005]** | −0.005 * (moved −0.012) | **−0.021 [−0.034, −0.007]** |
| window 5 → 15, A16 | 0 | −0.001 | 0 | −0.005 [−0.011, +0.001] |
| window 5 → 15, M16 | 0 | −0.004 * | 0 | −0.001 |

  - At L16, Mamba's imagined rollouts beat attention's at both seeds. The gain over the parents comes from L16 training, not
    from reading a longer window at evaluation.
  - One-step is window-independent because teval predicts it from the 4 root frames.
  - None of this reaches H16 decisions (mamba_long_edge FALSE).

**New findings from this audit.**
1. *The zombie cooldown is visible in the training data and position-locks ordinary hits too.*
   - In the 64-frame long pool's death windows, living ≥ 2 hits by distance to the death frame are 2,578 at 6, against
     150-190 at distances 1-5. Further peaks: 932 at 12, 519 at 18.
   - In ordinary windows, gaps between consecutive hits peak at 6 (574, against 50-160 at the other gaps 1-15).
   - So E17's end-aligned windows put ordinary hits at target positions 9 and 3 (23,266 and 9,351, against ~3-5k elsewhere),
     not only deaths at 15. This is the data mechanism behind the long worlds' time-row dependence of drawn damage
     (row 14 → 4 transplant: 26 → 12, 44 → 13).
2. *Our "two seeds" replicate initialization only.* tworld seeds the batch order with a fixed generator (11), so every seed sees
   the identical batch sequence. Fcanvas s7 / s8 per-update losses nearly coincide: 0.0370 / 0.0369 at 24k. This holds for
   E14-E19 alike; replication over data order has never been tested.
3. *E19 B never tested full de-alignment.*
   - Row 4 exists only in unsplit windows, so P(death | row 4) stays 26.6%, exactly as in A. The census confirms
     2.997% → 26.68% from row 0 to row 4.
   - The rejected crop-and-pad would have kept it near 5.3-6.7% at the cost of 10.6% of targets (my arithmetic, not run).
   - "B is not a repair" therefore says nothing about removing the death-position cue. E20's class-balanced batches did remove
     it: the parent's 108 window-15 false drops fell to 3-4.
4. *E18 failed for a read-path defect, not because world alignment fails.*
   - The corrt head predicts slot (r, c) of frame t+1 from time t's output at (r, c). In fcanvas that output belongs to the
     world cell currently there, so the entering cell's remembered stream is never read (0 / 2,018 direct matches).
   - The per-slot alignment behind Mamba's same-slot recall is also lost: 0.335 vs 0.864. Resetting canvas's SSM changes
     nothing. A world-aligned memory needs an explicit next-view read (Neural Map / MapNet style).
5. *E20 C is Goodhart through a frozen linear probe.*
   - The generated token-63 change has ~13% of the true change's norm (projection 0.03; the token stays ~95% copy).
   - It is aligned with the readers' health direction (cos −0.31). The loss reader weights token 63 more (norm 5.57 vs 3.18;
     reader cosine 0.56), so it reads a full drop: 266/279 hits. The independent reader sees 6/279 at 1.5 and 267/279 only at
     0.5, with 573 false drops.
   - Also: B's +17 catches are over-drops (true 7 → 5, predicted ~0.56). All E20 arms lose same-slot recall (−0.13 to −0.14)
     and add one-step cost (+0.008-0.012): endpoint-only supervision (40 targets/update against 600) erodes the world.
6. *At ages 2-5 the conv buffer carries more than the SSM.* Clearing conv leaves donor pull 0.075 / 0.034; clearing the SSM
   leaves 0.137 / 0.098 (intact 0.249 / 0.257). At ages 6-15 the SSM carries it at both seeds: resetting the SSM gives
   0.0004 / 0.0006 history gain.

**Corrections.**
- Other agent's record:
  - "512 same-slot cases from 168 episodes": 171 distinct pool rows.
  - Attention s7's moved-slot sighting effect, +0.84, was omitted.
  - The E17 teval / compare gap was not noted.
- Mine (accepted):
  - E17 was evaluated at window 16, where the output is untrained; window 15 is correct.
  - My window 5 vs 1 memory contrast also changed scan length and time row; the sighting swap and fixed-clock controls
    supersede it with the same conclusion.
  - E16 trains through latent L1, not MSE.
  - check_context's aggregate miss is not "no history use".
  - moved_unsolved ignored the generation baseline.

**State.**
- E20's driver (`d4mj-e20.service`) and post service are enabled at login. After this morning's boot they re-verified and
  re-exported seed-7 stages: no new results. The driver remains active with seed 8 held.
- No other research process is running.
- Fcanvas s8 is parked at 24k (state archived).


## 2026-10-08 — E20 seed7 endpoints and frozen CPU diagnostics

All three seed7 arms completed their original6,000 updates. Parent and A/B/C
window5/15 endpoints and paired comparisons are complete; the additional window4
endpoint pass remains in progress. Seed8 stays held for review. Numerical sources
and completed endpoint receipts match their recorded hashes. Reused diagnosis
roots, one world seed; no actor or sealed promotion result.

**B expanded factual support but did not repair health.** Independent HUD reader,
teacher successor, window15; strict drop cut1.5 health units:

| endpoint | A | B | paired B−A,95% interval |
|---|---:|---:|---|
| ordinary >=2 hits caught |16/279|27/279|+0.03943[+0.00749,+0.07252]|
| fresh hits caught |0/31|0/31|0|
| ordinary change MAE,health units |2.06469|2.17689|+0.11220[+0.03104,+0.20419]|

The catch-rate gain is not accurate damage prediction: B adds17 catches and
loses6, with all17 new cases true7→5 transitions,17 distinct roots/episode seeds.
Median B successor health0.56459;13/17 below1. These cases already had a zombie
in their history and are **not** the fresh stratum. Four selected frozen CPU
decompositions: A's health generation weight0.000253–0.01740, B's0.40114–0.64822;
B's normalized generated candidate reads health0.809/0.542/0.032/0.191 against
true5, alongside substantial neighbour mixing. CPU/CUDA health difference up to
0.0997 in this subset; endpoint statistics use the saved CUDA outputs. This
localizes those selected over-drops to altered mixing and inaccurate candidates;
it is not a population-wide learning-cause proof. Window5 teacher B catches18/279
versus A6/279, but fresh remains0/31. Window15 self-fed B catches2/174, A1/171;
common-alignment paired gain0.00637[0,+0.02069], fresh0/23 in both.

**Preservation conditions fail.** Paired preselected sample0, window15:
B same-slot age6..15 capture0.50430 versus parent0.63438 over2,088 cells;
difference−0.13008[−0.18633,−0.07653], failing the−0.05 retention margin.
B one-step added error/copy0.007649[0.006391,0.009089], above the0.005 ceiling;
depth16 added error/V0.030255[0.018314,0.044286]. A and C also lose long-age
same-slot recall. These parent contrasts combine the shared E20 layout/data/
optimizer/budget changes; they do not isolate expanded B support as the cause.

**Short checkpoint diagnostics, no optimization:** fixed factual TRAIN panel,
eight ordinary nonfresh and eight detector-estimated fresh events, all true−2.
At context4, B fresh mean predicted changes at parent/1500/3000/4500/6000:
−0.0241/−0.0094/−0.0181/−0.0079/−0.0085 health units,0/8 strict catches at each.
B context15 at6000 also catches0/8 fresh cases. This is a small post-hoc panel,
not representative convergence evidence or a scratch comparison. CPU reference
recurrence versus four stored CUDA predictions: health-readout max difference
0.00136; checkpoint diagnostics took30.11s and22.43s. B's teacher retains ordinary
damage class-loss mass1.29825% despite25% sampled endpoints. No arm tests the
revised recipe from initialization; 6k sufficiency remains unestablished.

**C remains a diagnostic of scalar supervision, not an adopted method.** At
window15 its loss reader catches266/279 ordinary and31/31 fresh drops; independent
HUD reader6/279 and0/31. Both readers catch essentially all real successors with
zero unchanged false drops. Fresh medians: loss-reader predicted delta−1.931,
independent HUD−1.046. Affine decomposition puts the discrepancy in token63:
real independent contribution−1.978, generated−1.071; other HUD contributions
are small. Fixed TRAIN C already shows this two-reader separation at1500 updates.
The scalar objective constrains one affine projection; its success does not imply
faithful health-token geometry. This is consistent with the scalar-projection
ambiguity analyzed in [VaGraM,sec3](https://arxiv.org/html/2204.01464v1#S3), whose
value-weighted state loss differs from E20. No stochastic-averaging or warm-start
causal explanation is established by these measurements.

Evidence: EDA `levers_e20_v1/endpoints/contrasts_e20_C_s7_fmamba_fromM16_w{5,15}.json`,
`analysis_{B_new_catches,B_mae_contrasts,BA_generator_gate_cpu,B_checkpoint_curve_cpu,
checkpoint_curve_cpu,dual_reader_w15,health_geometry_w15}.json`;
scripts `20260927_levers/e20_{checkpoint_curve_cpu,b_checkpoint_curve_cpu}.py`.
All process logs stay in `artifacts/eda/levers_logs/`; notebook updated by agent.

**Historical fresh-hit readability scope, rechecked:** E19's local health hidden
feature h63 plus current health token/action was probed, with a ridge trained on
all ordinary damage versus unchanged, not exclusively fresh hits. Its Mamba B
fresh AUCs: seed7 0.62920[0.42420,0.78722], seed8
0.70155[0.53379,0.89269]; HUD-input/action control
0.67959[0.42350,0.88638]. Each uses9 fresh positives/1,376 scrolling unchanged
controls on43 TEST seeds. These are standalone intervals, not a paired test of
hidden versus input. Overall hidden damage AUC~0.86 is **not** fresh-hit evidence.
No full-hidden/carry, expressive fresh-specific comparison is established here,
and E20 B's hidden fresh readability has not been measured. E19 B randomized
boundaries/contexts; E20 B expands damage support. Sources:
`20260927_levers/e19_trace_readout.py`, `evals/e19_B_trace_readout.json`,
`evals/e19_B_s8_fmamba_from36000__e19_trace_readout.json`. E14a/b strict map-
consequence AP results concern mining/placement, not health.

## 2026-10-07 — E20 factual damage support and scalar consequence objective

Predeclaration: [E20.md](20260927_levers/E20.md), endpoint implementation:
[E20_ENDPOINTS.md](20260927_levers/E20_ENDPOINTS.md). Each arm continues its own
E17 M16 fmamba/corrt seed7/8 parent for6,000 updates; frozen Raw patch encoder,
same world architecture. Targets have14 preceding factual frames, with4..15
observed frames selected in balanced12-update cycles. Every batch40 contains
2 death/10 living >=2 damage/26 unchanged/2 other targets. Predict only the final
factual successor. Teacher importance weights retain E19's class-loss mass
0.0532475/0.0129825/0.912445/0.021325 at every context length.

|arm|ordinary damage source|objective|
|---|---|---|
|A|old E19 TRAIN support, deduplicated with source history recovered|latent L1|
|B|all eligible factual TRAIN ordinary damage|latent L1|
|C|exactly B's targets, batches and contexts|B + frozen health-token63 scalar delta L1, lambda1|

Death/unchanged/other targets are identical in A/B/C. B−A isolates the expanded
ordinary-damage distribution; C−B isolates the scalar objective. Parent→A combines
extra training, history/layout repair and endpoint-only supervision; it does not
isolate the terminal alias. No fork outcome trains a world or the loss reader.

**Full source census and label correction:** all8,325 TRAIN episodes and3,119,763
frames match exact renderer health templates. At outgoing t>=14 there are8,071
deaths,36,068 living >=2 damage transitions,2,891,130 unchanged and59,619 other.
The reward-only health decomposition was ambiguous on exactly two terminal
transitions: reward approximately0.1 was decoded as+1 but the HUD shows9→0.
IDs:`support-v2:20270731:13:14` and `support-v2:20270731:42:15`.
The terminal-aware correction makes every TRAIN reward-derived delta equal the
exact HUD difference. Living ordinary-damage counts do not change. Historical
reward-only census and aborted pool allocation are preserved under EDA; legacy
checkpoint/source files were not edited. E20 uses the corrected labels.

**Measured support admission:** union pool71,918 distinct factual endpoints.
A ordinary support1,961 events; B/C36,068. Under the same TRAIN-fit token hazard
detector and true-pair motion audit, A has54 fresh-damage events/54 episodes;
B/C1,005/934, passing the declared200-event/100-episode minimum. These are
detector-estimated fresh cases, not simulator annotations. The actual6,000-update
ledger covers1,961/29,179/29,179 distinct ordinary targets in A/B/C. B/C ledger
and context bytes match; A/B nonordinary targets match. At each of12 context
lengths the class counts are exactly1,000/5,000/13,000/1,000. This removes class
prevalence versus endpoint-position association; no broader causal claim is made.
Fresh-target draws in that fixed full ledger: A1,664 presentations of54 distinct
events/54 episodes; B/C1,704 presentations of820 distinct events/774 episodes.
Thus fresh presentations are close while independent factual support expands.
This ledger calculation is not a count of completed optimizer updates. Evidence:
EDA `levers_e20_v1/actual_support_draws.json`.

**Reader and execution admission:** factual TRAIN health-token63 ridge, episode
holdout,16,384 fit frames/8,192 selection frames, lambda0.0001. Held health MAE
0.0720653 units; ordinary-hit catch7,489/7,490, false drops0/5,261. Reader validity
passed. Encoder-cache32-window sentinels: maximum token difference0.0009765625
against tolerance0.002. Actual CUDA full-batch/context15 loss and gradients finite.
B/C loss identity and future-input perturbation differences0. Serialized4-update
versus2+restart+2 proof: model/RNG differences0 for all arms; optimizer maxima
1.36e−11/1.09e−11/2.91e−11. The proof consumes fresh temporary worlds, not the
scientific continuation states.

Independent diagnostic real-state controls also completed without world calls:
historical HUD reader MAE0.02535, strict catch278/279; factual loss reader
MAE0.04921, catch279/279. Both catch31/31 fresh hits and have0/12,623 false
drops. Copy controls catch0/279. A rank mismatch in the new dual-reader wrapper
was caught by this CPU control before any endpoint inference and corrected;
failed log and old source-pin/preflight records are retained. World training,
loss and source contracts were unchanged.

**Endpoint contract:** strict1.5-health-unit cut plus separate0.5 sensitivity;
ordinary/fresh/unchanged/cooldown-negative/recovery/starvation strata; independent
historical HUD reader and frozen factual loss reader. Hash-checked teval states
are reused on CPU at windows5/15; window4 gets a compact dual-reader pass. Paired
memory uncertainty uses the preselected sample0 trajectories:2,088 same-slot
age6..15 cells,2,949 moved-slot age6..15 and4,237 same-slot age2..5 across143 seed
clusters. The original all-five-key recall metric is also measured. Retention
requires own-parent difference lower95%>−0.05; one-step cost upper95%<=0.005.
Neither a health gain nor this reused diagnostic panel authorizes an actor.

**Gate-data recovery, not a gate pass:** balanced factual addresses recovered
with4 observed frames and one recorded successor,128 alive/128 dead from128
episodes in each of TRAIN and DEV; split IDs disjoint, no FINAL frames inspected.
Canonical continuation previously had767 alive/1 dead. Numeric health/inventory
retention integration and missing simulator-verified predicate coverage remain
unfinished. The canonical gate was not changed or rerun.

Execution: persistent `d4mj-e20.service`, serialized GPU training/evaluation;
`d4mj-e20-post.service` completes paired parent/readout/memory/cost comparisons.
Full scientific states every500 updates, atomic chunked data/evaluation stores,
source/input hashes checked on resume. A seed7 reached the committed1,000-update
state after admission (560.08 s training session, peak2.596 GB);
other arms/seeds and scientific endpoints are pending. No health-repair result yet.
Scheduling amendment before endpoint judgement: [E20_REVIEW.md](20260927_levers/E20_REVIEW.md).
Seed7 A/B completed their original6,000 updates; C continues its original budget.
Seed8 is held pending user review; seed7 endpoints/parent comparisons remain
queued. Only orchestration sources and approved-seed schedule changed; original
pins/sources preserved in EDA. The current C trainer is not interrupted; a
hash-verified completion handoff replaces the stopped coordinator afterward.
Original two-seed adoption criteria remain unmet during this hold. Lineage:
36k short +6k long parent; new6k endpoint phase240k target presentations, versus
3.6M in the previous6k long phase. This is an adaptation pilot, not a scratch
training test or demonstrated convergence budget.
Sources/evidence:`20260927_levers/e20_{data,labels,prepare,train,verify,endpoints,post}.py`,
EDA `levers_e20_v1/{source_census_corrected,pool,contract_check,health_reader,mechanics,gate_coverage}.json`.
Process logs only `artifacts/eda/levers_logs/e20_*.log`; experiment processes never
write this notebook.

## 2026-10-07 — E17 completed H16 and separating diagnostics

**Original protocol:** four frozen L16-trained worlds (attention/Mamba, seeds7/8),
window15, four observed root frames then self-fed states and recorded continuation
acts. Hazard readout:768 pooled hidden features/step, three head seeds,4000 updates;
FIT training, DEV-A selection, reused DEV-B judgement (1139 opportunity roots,
149 episode clusters). Metrics average each head's own choice. Reports:
`20260927_levers/evals/h16traj/*rawlong*__w15.json`.

| world | trajectory safe | snapshot safe | trajectory−snapshot,95% interval |
|---|---:|---:|---|
|attention7|0.651952|0.631475|+0.020477[+0.007597,+0.033346]|
|Mamba7|0.657265|0.622622|+0.034643[+0.021866,+0.048251]|
|attention8|0.643894|0.623555|+0.020339[+0.009120,+0.031942]|
|Mamba8|0.647013|0.620546|+0.026467[+0.013280,+0.039739]|

References: DOWN0.616083, uniform0.571427, one real future/action with held-out-key
scoring0.679366. Trajectory−snapshot passes both seeds/backbones; one-real-future−0.01
threshold0.669366 fails all. E17's Mamba−attention>=0.02 also fails:
+0.005314[−0.004406,+0.015445]/+0.003119[−0.005676,+0.012384]. Mamba−DOWN resolves:
+0.041182[+0.020717,+0.064258]/+0.030930[+0.009944,+0.054074]. Health endpoints fail;
same-slot memory passes (October6 endpoint record). Parent→long additionally changes
budget/data/optimizer; this is neither a pure context contrast nor actor performance.

**Short-parent seed8 comparison completed:** resumed report
`20261005_recovery/h16_results/corrt_raw_teacher_s8_fmamba_u36000.json`:
window5 trajectory0.641736, snapshot0.615727. `e17_h16_parent_compare.py` verifies
identical1139 DEV-B opportunity rows/outcomes and numerical sources;
long−short trajectory+0.005277[−0.005532,+0.017497], snapshot+0.004820
[−0.008980,+0.017812]. Neither contrast resolves. This changes training/context
jointly and is not a memory-only effect. Evidence: `evals/e17_h16_parent_compare.json`.

**Frozen-reader CPU controls:** `e17_h16_final.py`, `e17_h16_numeric_control.py`,
`e17_h16_action_strata.py` → matching `evals/*.json`. All four original decisions
reproduce exactly. Stable log-survival scoring changes zero Mamba decisions and no
safe score for any world. True32-key hazards reconstruct P16 within1e-6; true full16
safe0.762593 versus first8 0.667091. Learned late action-risk SD0.034783/0.057436,
versus true0.139729; correlation0.275702/0.346305. Full16−first8 gain unresolved in
all four worlds. Mamba SLEEP choices2.25%/3.31%; restricting scores to movement
improves safe+0.006109[+0.002005,+0.010700]/+0.004600[+0.000525,+0.009003].

**Reader optimization diagnosis:** `e17_h16_generalization.py` and
`e17_h16_choice_error.py` → corresponding reports. Selected→final4000 weights,
world/features fixed, original decisions reproduced. Seed7 P16 Brier improves
0.126193→0.092199 while safe falls0.657265→0.634850
(−0.022415[−0.035034,−0.010106]); seed8 safe0.647013→0.637273, unresolved decline.
True-gap-weighted action-pair errors rise0.170571→0.221229/0.188022→0.207206;
optimal32-key choices fall0.591455→0.532924/0.578285→0.544337. Chosen-action error
relative to root mean becomes more optimistic at both seeds; this is action-order
failure despite improved probability errors, not likelihood overfitting alone.
Earlier seed7-only controls and snapshot-objective test are retained in
`evals/e17_h16_s7_*`, `evals/e17_h16_m7_head_objective.json`.

**Matched target intervention, completed:** `e17_h16_target_intervention.py` →
`evals/e17_h16_target_intervention.json`. Same frozen caches/normalization,
Hazard(768,16), initializations, three head seeds,4000 updates, paired32-root×17-action
batches and selection. Conditional uses original hazard BCE; cumulative fits H16
likelihood; rank/rank8 fit the same H16 within-root ranking with16/8 terms.
Grouped conditional differs in batching from the original evaluator.

| reader objective | Mamba7 safe | Mamba8 safe |
|---|---:|---:|
|conditional|0.659012|0.648403|
|cumulative|0.617629|0.617226|
|rank16|0.661710|0.663795|
|rank8|0.666758|0.665103|

Cumulative−conditional−0.041383[−0.057717,−0.026270]/
−0.031177[−0.047815,−0.015629]. Rank16−conditional+0.002698[−0.006443,+0.011462]/
+0.015392[+0.005945,+0.024881]. Rank16−rank8 unresolved at both seeds.
`e17_h16_target_errors.py`: cumulative raises weighted pair misordering
0.167430→0.236266/0.188043→0.244008. Seed7 root-mean MSE improves0.115593→0.094612,
but action-contrast MSE0.022107→0.022273; selected relative optimism worsens at
both seeds. No replicated objective repair or late-feature gain. Ranking energies
are not calibrated probabilities. CPU optimizer-resume proof: `evals/e17_h16_target_smoke.json`.

**Exact TRAIN exposure:** `e17_train_exposure.py` → `evals/e17_train_exposure.json`.
Reconstruct6000×40 draws and match all four saved end-RNG states. Among3,600,000
targets,63,414 deaths (7500 distinct episode transitions) occur exclusively at
position15; living>=2 damage85,338 times (12,748 distinct transitions) across all15
positions. This verifies E17's anticipated terminal alignment and excludes absent
ordinary damage; it does not identify the historical causal contribution of either.

**Health-clock and isolated time-input interventions, completed:**
`e17_health_clock_b16.py`, `e17_health_time_swap.py`, `e17_health_time_true.py`;
reports of the same names under `evals`. Same337 living drops,31 fresh hits,
512 seeded unchanged controls (851 distinct root-transition cases including overlaps),
frozen worlds7/8, exact original real-fit HUD ridge. Real-successor control337/337,
31/31,0/512 false. Original0.5-unit detection cut;1.5-unit cuts reported separately.
Original teacher29/337 and46/337, fresh0/31 and1/31 reproduce exactly.
Repeated-current controls preserve current pixels/final action, set past NOOP:

| inference context | damage drawn, seed7/8 (337) | false drops, seed7/8 (512) |
|---|---|---|
|true history|29/46|5/10|
|repeat current, matched length|18/44|7/10|
|repeat current, length5|0/0|0/0|
|repeat current, length9|6/0|0/0|
|repeat current, length15|30/91|12/26|

On repeated length15, changing ONLY the current time embedding row14→row4 reduces
drawn damage30→14/91→9. Reverse transplant at length5 draws6/0, so the late
embedding alone is not sufficient. On real length15 histories, same single-input
change reduces catches26→12/44→13 of131 actual drops; mean drawn health loss
0.880993→0.377300/1.412231→0.255215. Paired health-change effects
+0.503693[+0.259203,+0.805772]/+1.157016[+0.819676,+1.542660],100 episode clusters.
Pixels, actions, earlier recurrent updates, scan length and other weights are fixed.
Below length15, catches3/206 and2/206;89.7%/95.7% of true-history catches occur at
length15. These controls identify causal reliance on the explicit time input and
its interaction with scan/history; they do not establish a trained fix or prove
that terminal alignment accounts for every error. Repeated-history lesions are OOD.
Raw rows and paired intervals: `evals/e17_health_clock_stats.json` (v2 labels rows
as transitions/distinct roots; legacy root-label report retained, metrics unchanged).

**Numerical guard:** initial batch4 trial failed exact reproduction28 versus29;
preserved at `evals/resume/e17_health_clock`. `e17_health_batch_parity.py` locates
one threshold crossing: root456/step12, current5.952761, predicted5.491476 atbatch4
versus5.427490 atbatch16. Mean absolute difference0.001344, maximum0.233705.
Original batch16/tail10 dimensions restore both seeds' original counts without
changing thresholds. Full forward dependencies and167 source/input/checkpoint hash checks verified
by `e17_health_verify.py`; wrappers verify it again before resume. All tests have
atomic input/source-bound chunks or optimizer checkpoints. The true-history
time transplant also completed an actual cache-resume verification: no new inference
chunks, intact result retained and wrapper exit success. Logs only under
`artifacts/eda/levers_logs`; evaluators never edit NOTEBOOK.

**Primary-source comparison after diagnosis:** [Geirhos et al.,§4/§6](https://arxiv.org/pdf/2004.07780)
distinguish shortcut opportunities from learned cue reliance and recommend controlled
cue changes; the TRAIN census and time transplant address those separately.
[Lambert et al.](https://proceedings.mlr.press/v120/lambert20a.html) document likelihood/control
objective mismatch; our numerical action-order diagnosis and target intervention
are the project evidence. [Nnet-survival](https://arxiv.org/abs/1805.00917) supports conditional-hazard
likelihood/product survival; our true-hazard/numerical controls verify the implementation.
[DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842) combines likelihood/ranking,
but ranks patients/events, not this project's within-root actions. None establishes
our proposed sampler/objective repair.

## 2026-10-06 — H16 evaluator resume and position-balance feasibility

`20261005_recovery/h16_resume_oct06.sh` resumed the declared four frozen E17
worlds at window15 and the queued short Mamba8 reader. Original FIT/DEV splits,
three head seeds, labels and budgets retained. Resume verified71 sources,
2 checkpoints and97 committed chunks against the saved CUDA contract:
`03a8b3e104ebe9ef7d754ec5b9a79144af0db844d191cfb514202b0a45e2b41c`.
This is diagnostic H16 evaluation, not canonical bridge or actor training.

`evals/e19_position_balance_feasibility.json`: CPU reweighting of the saved
1,200,000-target Mamba7 E19 TRAIN ledger by
`P_TRAIN(class)/P_TRAIN(class|row)` preserves contexts, total row loss mass and
global class mass while equalizing marginal class prevalence across rows.
Weights0.199557–1.776447; death prevalence5.32475% and ordinary>=2 damage1.29825%
at every row; numerical spread1.11e-16; Kish effective count1,176,714.
This is algebraic feasibility, not a trained repair or conditional independence.
A cannot support this correction at early rows, where it has zero death targets.

## 2026-10-06 evening — completed E19 matrix and E17 endpoint review (lead)

Attention seed8 A/B/C and booked mechanism/evaluation reports completed18:41 AEDT;
all four E19 sets now exist. E17 true-history recall at windows15/5, aligned
self-fed recall at15 and damage at15/5 completed18:44. The18:48 saved runtime
snapshot showed H16 running, but **live verification at21:05 AEDT supersedes it**:
systemd's journal records the lead stopped18:48:39, and the machine subsequently
booted20:55:12. Lead/replica services are inactive and no research GPU process is
running. H16 is interrupted, not complete. Per this turn's instruction, no new
long-running job or long resume was started. Historical status JSON is retained.

Short CPU follow-up declared before execution: apply the existing, unchanged
`e19_objective_alignment.py --sets full:8` to the final attention seed. Hold its
saved seven routing candidates fixed; reproduce the latent-versus-recorded-health
oracle comparison and paired B→C prediction cost with the same cohort and rules.
Future outcomes select a diagnostic oracle, not an inference input. Source/input-
bound atomic resume; output in EDA `levers_logs`. Separately read/reconcile the
completed E17 cached records, fixed-clock history control and recurrent-state
lesions. These checks use saved measurements and do not fit or forward a world.
**E19 matrix is complete, and remains a health non-repair.** Independently
recomputed primary counts from all12 saved `__e19_health_per_root.pt` files;
all counts match their reports. No new world training/GPU forward in this review.

| set | A hits/false | B hits/false | C hits/false | fresh hits in every arm |
|---|---:|---:|---:|---:|
|Mamba7|14/115|24/122|62/297|0/31|
|Mamba8|18/84|13/164|67/388|0/31|
|attention7|47/240|29/192|15/268|0/31|
|attention8|15/110|12/117|53/174|0/31|

Same279 ordinary living >=2 drops /12,623 unchanged transitions /31 fresh hits;
four trained sets on **one reused diagnostic panel**, not four independent
judgement datasets. Attention8 C−B catch +0.146953[+0.105080,+0.190840], false
rate +0.004516[+0.002204,+0.007047]. Attention7's negative catch contrast does not
replicate; the two Mamba gains do. C still fails `learnable` at every endpoint.
Attention8 alone passes the declared `unshortcut` ratio (0.702128), but that flag
is not evidence of learned damage mechanics: w4_at0 catches51/279, w5_at0 53,
current-frame-only at row0 54, repeated-current-frame five-step scan47. Shifting
the SAME four frames from rows0–3 to1–4 changes catches51→53 and false drops113→179.
Source/digests/count proof: `evals/e19_completed_matrix_review.json`.

**Latest attention seed narrows the composition finding.** B→C forced-generator
ordinary catches109→155, full-token63 L1/copy1.661→1.474, mean generation weight
0.06680→0.14416 and actual catches12→53. Both candidate quality and composition
improve here, unlike attention7's falling generation weight. On fresh encounters
C still has generator L1/copy1.929, mean generation weight0.09432 and0/31 output
catches. Removing the shared HUD motion gate changes that to0/31; forcing health
generation gives7/31 but4,774/12,623 false drops. True63 substitution gives31/31.
Exact mixture reconstruction max0.0. Generated-fit output63 damage AUC0.873586
[0.833242,0.914465] overall; fresh subset has only9 TEST positives and its output
AUC interval includes0.5. Do not call this absence of information.
Sources: attention8 `__e19_router_diagnosis.json`, `__e19_trace_readout.json`.

**Short CPU objective test replicated at attention8:** unchanged seven saved
routing offsets, lowest latent-L1 chooses0/31 fresh depictions, recorded-health
oracle7/31 (all reachable in this family). Cheapest depicting candidate's median
relative latent-L1 premium98.859%. Thus the objective prefers non-depiction even
where this frozen family can depict the hit;24 cases are unreachable only within
the audited family. Oracle future labels do not enter inference or a training run.
New paired B→C costs (143 seed clusters,2,000 resamples): one-step error/copy
+0.011583[+0.009088,+0.014471]; alive-at-depth16 error/V
+0.046369[+0.033958,+0.059465]. Point statistics match historical `compare.stats`.
The fourth C arm also incurs prediction cost. Report:
`evals/e19_C_s8_full_from36000__objective_alignment_v2.json`; console measurements
in `artifacts/eda/levers_logs/e19_objective_alignment_full_s8.log`.

Frozen TRAIN C attention8 extra-loss share: death65.20%, ordinary9.79%, unchanged
19.33%. Total measured output-weight ordinary-gradient dot objective−1.35716e−5,
but generator/proj contribution+2.35920e−5 and selection contribution−3.71637e−5.
This supports local endpoint selection conflict at both attention seeds. It is
not universal over Mamba seeds, nor a reconstruction of historical AdamW/backbone
updates. The learning cause of poor fresh generator fitting remains unresolved.

**E17 older-terrain recall is a replicated capability, with limited registration.**
Reparsed all completed logs; repeated/completed exports agree, and recall_capture
independently matches (neighbour_error−world_error)/(neighbour_error−sighting_error).
TRUE-history cohort:5,010 trajectories (1,002 roots×5 simulator keys), all models
share10,509 same-slot and14,377 moved-slot recallable cells with sighting ages6–15.

| capture on age6–15 terrain | Mamba7 | Mamba8 | attention7 | attention8 |
|---|---:|---:|---:|---:|
|same slot, window15|0.615924|0.643393|0.430657|0.357305|
|same slot, window5|0.369022|0.364637|0.356120|0.369334|
|moved slot, window15|0.405302|0.440272|0.426823|0.417918|
|moved slot, window5|0.396245|0.430596|0.384315|0.413640|

Mamba same-slot window15−5 gains+0.246902/+0.278756; moved-slot gains only
+0.009057/+0.009676. Stage2 declared `long_recall_same` and `long_recall_used`
conditions hold at both seeds. **`moved_unsolved` as literally declared (all moved
captures<=0.35) is false**, not an automatic failure of registration diagnosis:
absolute moved capture can be supplied without the old sighting. The script's
historical `readings:{}` does not implement these E17 conditions; the reviewed
criteria are explicitly reconstructed in `evals/e17_completed_endpoint_review.json`.

Window length also changes time rows/scan length. Causal older-history evidence
comes from the already-completed fixed-clock intervention: same inputs' latest5
frames/actions/length/time rows retained while older visual frames are replaced.
On200 roots/103 seed clusters Mamba's same-slot history gain is0.363581
[0.165949,0.556088] and0.335796[0.149464,0.541272]. Resetting SSM every step reduces
those gains to0.000437/0.000567; current-step recurrence still runs. Moved-slot
history gains−0.005436[−0.098493,+0.081546] and+0.049434[−0.014862,+0.110202]
are unresolved. This establishes useful frozen SSM memory and no resolved
older-history benefit after relocation on that panel, not an information ceiling.
The seed7 dependence on older convolution taps does not generalize to seed8;
both have SSM dependence. These older component reports are context for the newly
exported endpoints, not newly run lesions or a trained memory-transport repair.

**Health remains broken after long-context training.** Half-unit sensitivity
(`check_damage.py`, not E19's1.5-unit cut): Mamba window15 teacher catches29/337
(s7) and46/337(s8), versus5/337 each at window5. Their aligned self-fed rollouts
catch0/240 and0/241. Attention window15 catches5/337 and16/337, self-fed0/243 and
0/238. Real-successor decoder detects337/337 with0 unchanged false drops.
Fresh >=2-hit stratum at window15: Mamba0/31 and1/31; attention0/31 at both seeds.
Thus the positive older-static-terrain memory does not repair health mechanics.
Counts differ from E19 because this lenient check includes all living one-unit
and larger drops and earlier positions; do not pool the two denominators.

Aligned self-fed older same-slot capture is0.265160/0.277738 for Mamba, compared
with attention0.123668/−0.057215. These are model-specific surviving/aligned
subsets (Mamba1,480/1,496 cells, attention1,452/1,425), so differences from the
true-history table or between worlds are **descriptive, not paired retention
contrasts**. Camera-coordinate groups derive from the token scroll estimator.
Reports/logs: EDA `e17_recall_{w15,w5,imagined_w15}.log`,
`e17_damage_{w15,w5}.log`, and source/hash-bound completed frozen-eval journals.

**H16 resume integrity verified without restarting:** first world attention7
FIT features has388/6,393 roots,97 committed batches. All97 chunk hashes,71
recorded numeric-source hashes and2 checkpoint hashes verify. No DEV feature
cache or head checkpoint exists yet, hence no H16 decision verdict has landed.
`evals/e17_h16_interrupted_integrity.json`; live status correction:
`20261005_recovery/review_runtime_oct06_evening.json`. The20:55 reboot does not
explain the earlier18:48 service-stop request; no actor/stop-cause attribution.

**Primary literature and next decision:** [VaGraM's paper/authors](https://www.pair.toronto.edu/blog/2022/vagram-voelcker/)
addresses errors cheap for state prediction but costly for control; it also warns
that optimizing only a learned value on unsupported states can produce invalid
predictions. This supports testing factual consequence alignment while retaining
state-validity constraints; E19's whole-token mask is not VaGraM, and scalar
oracle selection is not a demonstrated trained repair. [TrajGRU](https://arxiv.org/abs/1706.03458)
learns location-dependent recurrent connections; [Neural Map](https://arxiv.org/abs/1702.08360)
provides spatial memory addressing/reads. Both support the specific registration
question, not another unchanged canvas run. Preserve Mamba's established same-slot
capability when testing any transported-carry/read intervention. Health candidate
quality/objective alignment and cross-position memory access remain separate
mechanisms; no evidence that one architecture change fixes both. No C promotion,
new long training, or actor-success claim. H16 can be resumed from its verified
prefix in a later authorized run.

## 2026-10-06 — E19 attention control and frozen objective diagnosis (lead)

Attention seed7 A/B/C and booked mechanism diagnostics have completed; seed8 is
still training. No health arm is promoted. Before running an additional CPU
diagnostic, declare a fixed-candidate objective comparison: reuse the saved
seven generate-logit offsets at both Mamba C seeds and attention C seed7, plus
the already-saved195-member Mamba families. Hold candidate outputs/decoder
fixed; compare latent L1 to recorded-health-change absolute error/9 and fixed
combined coefficients0,.01,.03,.1,.3,1,3,10. Record reachable damage, the latent
loss premium of the cheapest depicting candidate, and paired B→C rollout costs.
Future recorded health is an explicitly labelled selection oracle, not an
inference input or proposed trained head. This isolates endpoint objective
preference from candidate reachability, not the historical cause of learning.
Script:`20260927_levers/e19_objective_alignment.py`; hash-bound atomic arm journals
in the existing `evals/resume`, console measurements in EDA `levers_logs`.
The script never writes this notebook. Detailed completed findings follow after
the measurements have been checked.

Additional CPU census declared before execution: reconstruct local predictor
rows for **all1.2M actual targets** in each completed E19 set. Compare conditional
death frequencies by row, not just the marginal uniformity of terminal depths.
B/C map original target k to k before their cut, otherwise k-cut; every target
must occur exactly once and A/B/C ledgers must match. Count deaths, ordinary
living hits and mask-selected examples. This determines whether random boundaries
actually remove label-position correlation; it cannot prove the model uses a
remaining correlation or that balancing rows would repair prediction.
Script:`20260927_levers/e19_depth_census.py`; immutable per-set journals and EDA log.

Final bounded TRAIN check declared before execution: reuse the incoming census's
exact15 sampled hit and25 unchanged episode/step events. Score completed C worlds
(Mamba7/8, attention7) with their original observed prefix and every actually
sampled boundary, weighting by the recorded multiplicities. Real successor is
the same fixed decoder's positive control; forced health-token generation
separates candidate from router. No future frame/label enters world inputs.
This distinguishes failure on own TRAIN examples from new-root generalization
alone; it does not isolate optimizer history versus sparse unique support.
Script:`20260927_levers/e19_train_fresh.py`, GPU allocator16%, batch4, atomic
per-length/batch resume. No new model training or experiment directory.

**Completed verdict (18:04 AEDT): E19 is not a health repair.** Mamba seeds7/8
and attention seed7 have all A/B/C endpoint and booked mechanism reports.
Attention seed8 remains running, so this is three completed sets, not a completed
four-set claim. A is unchanged6k continuation, B is target-preserving random
boundaries/resets, C is B plus context-selected token63 latent-L1 dose1.
All three completed C arms fail `learnable` and `unshortcut` as declared.
The old reports' historical `no_cost: Pending` is preserved; the new paired cost
measurements reject no-cost. No arm is promoted into the canonical recipe.

| completed set | A hits/false drops | B hits/false drops | C hits/false drops | fresh hits, every arm |
|---|---:|---:|---:|---:|
|Mamba7|14/115|24/122|62/297|0/31|
|Mamba8|18/84|13/164|67/388|0/31|
|attention7|47/240|29/192|15/268|0/31|

Denominators279 ordinary>=2 living drops and12,623 unchanged transitions.
These are the same inspected diagnostic roots, not independent datasets.
For attention7 B→C, catch falls−0.050179[−0.084881,−0.017794] and false rate
rises+0.006021[+0.001284,+0.011309]. Thus C's sensitivity gain replicates over
the two Mamba seeds but does **not** generalize to the completed attention
control. In attention, the generator improves (forced catches151→205, relative
health-token L1 to copy1.759→1.462), while its actual mean generation weight
falls0.08710→0.05964 and final catches29→15. This locates that endpoint reversal
in composition; it does not explain the entire optimizer trajectory.

**Full census confirms the predeclared residual label/position cue.**
Exact census of all1.2M actually sampled targets, identical ledgers/layouts in
all completed sets; every factual target reconstructed exactly once:

| local predictor input row | B/C targets | deaths | P(death at that row) |
|---|---:|---:|---:|
|0|432,239|12,956|2.9974%|
|1|336,120|12,658|3.7659%|
|2|240,000|12,861|5.3588%|
|3|143,880|12,678|8.8115%|
|4|47,761|12,744|26.6829%|

Death counts are nearly uniform; total target counts are not. Conditional death
frequency therefore differs8.90195× from row0 to4. Within the selected health
mask, the corresponding death proportions are14.50% versus82.25%. A had all
63,897 deaths at row4. A row-only TRAIN prior has death AUC0.922497 in A and
0.667647 in B/C. This is an exploitable supervision correlation, not a measured
neural-head AUC. This is **not a newly discovered implementation defect**:
`E19.md` and the Oct5 prelaunch entry already recorded expected row frequencies
2.96%,3.80%,5.33%,8.88%,26.63% and explicitly did not claim alias-free supervision.
The complete actual-ledger census verifies that limitation and adds the row-only
AUC. **B is boundary/context randomization, not alias-free supervision, and its
failure cannot rule out a genuinely balanced layout.**
It also truncates context: this census does not isolate position from history.
Existing attention7 C fixed-four-frame tests draw6 hits/45 false drops on rows0–3,
versus14/252 on rows1–4. The learned endpoint still depends on time-table position.
Reports:`evals/e19_{fmamba_s7,fmamba_s8,full_s7}__depth_census.json` and
`evals/e19_depth_only_prior.json`. The complete6.247GB pool SHA256 was independently
recomputed and matched all three training contracts.

**Frozen output mechanism replicated across backbones.** For C fresh hits,
generation weights are0.07399(M7),0.08665(M8),0.09168(T7). The standalone generator
is1.980×,1.831×,1.946× worse than copying in true-target token63 L1. Removing the
global HUD movement gate still draws0/31. Forcing generation draws10/31,8/31,
15/31 fresh hits but also4,537/12,623,7,680/12,623,9,805/12,623 false drops on
unchanged states. Therefore 'the gate blocks an otherwise correct generator'
is not supported. The candidate itself is poorly calibrated.

Relative to the actual real-successor token change, generator parallel component
medians are0.620,0.713,0.620; orthogonal squared error/true-change power medians
are3.696,3.102,3.538. This is geometry of the **full contextual health token**,
not a pure health direction, nor proof a particular noise process caused training.
It quantifies why a partially health-readable candidate can still be a worse
whole-token prediction than the copy.

**New fixed-candidate objective intervention, CPU:** hold every saved candidate
and the decoder fixed, change only the per-case selection criterion. On the
195-member Mamba families (193 logits plus two endpoints), latent-L1 optimum
draws0/31 fresh hits at both seeds; a recorded-health-error oracle chooses10/31
and8/31. Those are all reachable cases in the respective audited families.
The cheapest depicting candidate has median relative latent-L1 premium53.62%
and45.85% among reachable fresh cases. Attention7's seven-offset family reaches
15/31; latent-L1 chooses0, health-error oracle14, with79.28% median premium.
This is a causal endpoint objective-selection contrast, not a learned/inference
repair: future recorded health chooses the oracle, and the remaining cases are
unreachable only **within these frozen families**, not an information ceiling.
Increasing C's full-vector token63 weight is not the same as supervising scalar
health; it scales correct and off-target components together.

Whole-token weighting was already substantial: selected coordinates receive
270–892× the teacher coefficient. Yet only5.24% of selected targets are ordinary
damage;68.71% unchanged and23.77% deaths. On the30 frozen actual TRAIN batches,
deaths take58.9–64.3% of C's extra loss, ordinary hits10.7–12.1%. These describe
actual allocations, not a sufficient causal proof of starvation. The gradient
conflict is still not universal: ordinary dot total-objective is−9.46e−5(M7),
+1.03e−3(M8),−1.03e−4(T7), on frozen output weights. Historical AdamW/backbone
updates were not measured by these derivatives.

**Own-TRAIN check completed:** the detector/estimated-motion census's exact15
hit events have111 recorded presentations, all mask-selected; matched25 unchanged
events have201 presentations. Mamba C7/C8 draw0/111 hits with the original full
observed prefix and0/111 with their actual sampled boundaries. The same decoder
reads true successors111/111 and has0/201 false drops. Attention C7 draws16/111
with original prefixes,3/111 under the sampled boundaries. Original-prefix
Mamba health changes average−0.00156/−0.02448 versus actual decoded−2.03788.
Forced generators catch42/111,43/111,69/111 but hallucinate93/201,67/201,85/201
matched unchanged drops. Thus failure is not solely new-root generalization or
solely missing history. Sparse unique support can still affect learning; this
test does not identify the full historical reason for poor generator fitting.
Reports:`evals/e19_C_s{7,8}_fmamba_from36000__train_fresh.json` and attention7
equivalent; atomic case journals remain. Sandbox CUDA was unavailable on the
first attempt; a bounded external systemd job completed all three checkpoints.
Unchanged warm resume completed18:09:03 AEDT and preserved all three exported
reports byte-for-byte (`evals/e19_train_fresh_resume_proof.json`); completed case
journals were reused rather than rerunning world forwards.

**Prediction cost:** new paired143-seed-cluster bootstrap(2,000 draws), historical
alive-at-depth16 estimand, C−B error/V increases:
M7+0.039984[+0.024630,+0.056994], M8+0.016712[+0.002064,+0.030313],
attention7+0.028852[+0.011478,+0.045533]. Point statistics independently match
`compare.stats` to1e−6. The first objective-diagnostic v1 also computed all-root
depth16 costs, a different cohort; retained v1 reports are not substituted for
alive-conditional historical metrics. Corrected v2 explicitly records both
estimands. Candidate-selection findings are unchanged between versions.
Evidence:`evals/*__objective_alignment_v2.json`; console logs in EDA. Warm
unchanged relaunch of both CPU diagnostics preserved all six exported reports
byte-for-byte (`evals/e19_cpu_diagnosis_resume_proof.json`).

**Literature check and decision:**
- [ITC](https://arxiv.org/html/2605.16457v1), appendixB.2, generates inventory/HUD
  through its categorical prediction path rather than transport. Our earlier
  forced-generator/region adaptation failed; the current thousands of generator
  false drops explain why simply repeating that switch is not a repair.
- [Segmentation Dreamer](https://arxiv.org/html/2410.09972v1), sections4.1–4.3 and
  limitations, supports targeting control-relevant content and warns that even
  task-relevant appearance reconstruction can encode irrelevant variations.
  A contextual latent token mask is not equivalent to its pixel masks.
- [MuDreamer](https://arxiv.org/html/2405.15083v1) learns reward/continuation/value
  and action prediction; [NE-Dreamer](https://corl-team.github.io/nedreamer/) retains
  reward/continuation beside next-embedding alignment (Barlow Twins in that
  source, not merely a cosine loss). These justify testing factual semantic
  targets beside latent prediction; they do not prove our proposed repair.
- [Shortcut learning](https://arxiv.org/abs/2004.07780) is the relevant general
  problem for the residual row cue. Softmax expert competition papers have
  different models/assumptions and do not establish our learning mechanism.

Keep the booked attention8 control and E17 endpoints running. Do not add C to
the integrated recipe, force-generate HUD, or launch another architecture as a
health repair. Any next loss/supervision intervention must verify **conditional**
row/class balance, preserve/control history, and distinguish direct generator
quality from whole-token versus scalar-consequence objectives on matched factual
data. Position balancing and semantic loss are separate factors; changing both
at once would repeat the old attribution confound. We now have concrete endpoint
mechanisms and a design defect, not a complete optimizer-history explanation.

## 2026-10-06 afternoon — E19 replication and canvas endpoint diagnosis (lead)

**Canvas scheduling correction,16:39 AEDT:** the user challenged the unchanged
seed8 replication. It indeed uses the same implementation as s7; neither missing
direct next-view read nor off-screen convolution advance was repaired. Parked
seed8 at its decoded **24,000/36,000 full state**, with finite weights, optimizer,
order and CPU/CUDA RNG present. Immutable archive:
`20261005_recovery/recovery_inputs/canvas_s8_u24000_parked.state.pt`, SHA256
`d02f8ac94ae376ab2f54b0e217c947e0fd88a1750659e39cc3b3878f77cdecae`.
`20260927_levers/e18_seed8_hold.json` blocks automatic unchanged resume.
Remaining12,000 updates are held, not completed or declared a negative second
seed. Lane74 now scores only completed s7 canvas/Mamba endpoints and passes on
to the already-booked attention controls/E17 H16 queue. Numerical trainer/model
sources and all failed evidence are unchanged. This is a scheduling decision,
not a predeclared statistical early-stop verdict.

At the completed s7 canvas36k endpoint, all2,048 held windows are computed in
CPU FP32: recall capture0.331725, same-slot0.335330, moved-slot0.311509. Resetting
SSM every step changes recallable squared error by−0.090593
[−0.318831,+0.139504], unresolved. The matched slot-Mamba part is still computing;
the608-window interim paired contrast is explicitly provisional, not a full
cohort result. The lack of direct read is a source/address fact; these endpoint
numbers alone do not establish its causal share of the whole deficit.

**Final-checkpoint component diagnosis declared before execution,16:42:** repeat
the existing16-window access, carry-contribution and correct/zero/permuted
last-query-residual substitutions on **s7's completed36k weights**. Reuse the
same deterministic cohort and FP32 equations; oracle future coordinates select
addresses only, never future features. This separates whether extra training
learned a useful held SSM/read and whether supplying the missing frozen residual
rescues prediction. Preserve the12k results. Run isolated source-bound copies
in the existing recovery_inputs folder, changing only checkpoint/tag/scope paths;
no active source or trainer modifications. Convolution-only hold is already a
known non-repair on the earlier6k rollout (+/− interval includes zero), so it must
not be presented as the established explanation of canvas36k's deficit.

**Completed36k component result,16:45 AEDT:** the same2,018 entering-cell
addresses still have0 direct matches. On840 layer/cell queries the SSM state is
nonzero and held exactly, but the off-screen output is exactly0; all-kept CPU
parity max2.38e−7. Restored old-query reads have meanRMS0.559325. Their pre-gate
SSM energy share is only1.846% across layers and1.401% in the final layer;
zero-SSM old-query output cosine0.998364. This energy decomposition is not a
semantic information ceiling. Off-screen conv buffers still advance/erase.

Unlike the12k intervention, the completed36k generator **does benefit** from
correct remembered residuals: normalized generator MSE0.382472→0.297505,
difference−0.084967[−0.159302,−0.028356], a22.2% decrease. Zeroing instead worsens
generator MSE to0.505510; permuting gives0.383625 (unresolved versus baseline).
But routed output MSE0.207283→0.201540 has unresolved difference−0.005743
[−0.053862,+0.023563]. Generate weight0.260806→0.265591. Direct old-token copy
MSE0.057157 shows available past content; it is not a deployable oracle policy.
Untreated outputs are exactly unchanged. Thus missing direct access has a
measured component effect at final training, while this residual substitution
alone does **not** repair routed prediction. This is16 inspected windows/140
cells, one checkpoint, oracle addresses and frozen heads—not a trained causal
repair or an actor result. Reports:`evals/e18_{access,carry,reroute}_s7_at36000.json`.

Primary-source check: official Mamba2 implements distinct convolution and SSM
state updates (`state-spaces/mamba`, `modules/mamba2.py`); dt masking alone holds
the SSM, not the convolution. Neural Map separates spatially addressed writes
from global/context reads (Parisotto/Salakhutdinov, arXiv1702.08360, equations2–6
and sections3.1–3.2). This supports auditing the read path independently of
registration, but proves neither a particular Craftax repair nor causality of
our full canvas deficit. No further canvas training is booked from this lead.

**Full matched endpoint completed,16:45 AEDT:** CPU FP32, all2,048 held windows,
identical cell counts and baselines, paired window-bootstrap2,000 draws:

| recall capture | canvas s7 36k | slot-Mamba s7 36k | canvas−Mamba95% interval |
|---|---:|---:|---:|
|all2,018 recallable cells|0.331725|0.778389|−0.446664[−0.515513,−0.373928]|
|1,615 same-slot cells|0.335330|0.863475|−0.528146[−0.593691,−0.456649]|
|403 moved-slot cells|0.311509|0.301172|+0.010336[−0.105560,+0.128580]|

Canvas loses the existing same-slot capability without a resolved moved-slot
benefit. It has slightly lower unseen-cell error41.535824 versus42.728435;
paired difference−1.192612[−1.770694,−0.598067]. Thus do not call canvas universally
worse or replace this registration test with an overall forecasting/actor claim.
Resetting SSM each step increases slot-Mamba recallable error+4.164600
[+2.160867,+6.435003] and same-slot error+5.331637[+2.888094,+7.987166], while
canvas's corresponding changes remain unresolved. Same clocks/convolution/
actions/current-step update are retained. Source/input/weights-bound report:
`evals/e18_s7_at36000__endpoint_cpu.json`. Raw per-window rows and contracts remain
in `evals/resume/e18_s7_at36000__endpoint_cpu/`. These are inspected held TRAIN-pool
windows, one trained seed; seed8 has no36k verdict and is deliberately parked.
The completed result strengthens the budget hold; it does not prove that the
missing direct read is the only cause, nor that convolution hold alone repairs it.

At16:18 AEDT, E19 fmamba seed8 A/B/C and all booked endpoint/mechanism readings
are complete. Canvas seed7 completed36,000 at13:40; seed8 is actually training
(last logged21,500/36,000). Remaining attention controls and E17 H16 readings
stay booked. Automatic process logs remain in EDA; this entry is manual.

**Completed TRAIN-coverage and conditioning follow-ups:** ABC ledgers/boundaries
are byte-identical across both seeds; every extra-health context mask is reproduced
with0 disagreements. Across240,000 sampled windows/1.2M targets, ordinary damage
appears15,579 times;12,755 are context-selected. Of243,378 selected targets,
ordinary damage is5.24%, unchanged68.71%, deaths23.77%. `unique_targets` in the
first census means **window-target pairs**, not deduplicated physical events.

The corrected gate-equivalent fresh-arrival census (detector/estimated-motion
labels, original positions3/4 where four-frame history exists) finds only
**15 distinct episode/step ordinary-damage events**, drawn111 times, all selected
for health dose. E19's randomized boundary leaves all four frames available on
only28 of those111 draws. The matched fresh unchanged population is25 distinct
events,201 draws,33 selected. These labels use TRAIN-fitted zombie detection and
true-pair token displacement, not full simulator truth. This is direct sparse/
context-truncation **support evidence**, not a proof that it causes0/31 evaluation
hits. It narrows the prior 'adequate hazard exposure' claim: aggregate ordinary
hits do not establish coverage of this moving fresh-arrival condition.
Reports:`evals/e19_fmamba__{exposure_census,incoming_exposure}.json`.

On frozen C traces, fresh generated-health change is−0.9455(s7)/−0.8071(s8),
with generation weights0.07399/0.08665; all actual fresh predictions remain0/31.
Stationary damage generation weights0.18415/0.21082 and catches62/228,67/228.
Forced generation draws false damage on4,537/12,623 unchanged cases(s7) and
7,680/12,623(s8). Physical health9 stationary drops are caught53/65 and59/65,
whereas health7 stationary drops are caught1/51 and0/51. These are specific
candidate/routing/health-stratum measurements, not evidence for a universal
health template or a seed-general gradient-conflict mechanism. Router-trace s8
has387 actual unchanged false drops versus388 in the primary evaluator; preserve
both batch/precision protocols, rather than silently equating their counts.
Report:`evals/e19_fmamba__health_conditioning.json`.

Queue hold was exercised with real checkpoint-hash validation and training/eval
commands replaced by dry-run records: s8 skipped, only completed s7 pool/futures
commands emitted. Both CPU endpoint/component services finished successfully.
The remaining lead/replica services are active; no canvas s8 process is running.
Recovery launch retains the hold and can resume unfinished bound diagnostics.

**E19 replica changes the causal account:** primary living k>=3 reading uses
the same279 ordinary drops,12,623 unchanged transitions and31 fresh drops:

| seed | A hits / false drops | B hits / false drops | C hits / false drops |
|---|---:|---:|---:|
|7|14 /115|24 /122|62 /297|
|8|18 /84|13 /164|67 /388|

All arms at both seeds draw0/31 fresh drops. At s8, B−A hit gain is−0.017921
[−0.046518,+0.007693], while false rate rises+0.006338[+0.003131,+0.009981].
C−B hit gain is+0.193548[+0.151846,+0.237157]; false rate rises+0.017745
[+0.014258,+0.021249]. Boundary randomization alone is not a replicated
ordinary-hit repair; dose improves sensitivity with a replicated false-positive
cost. These are reused diagnosis roots and paired143-seed-cluster intervals,
not two independent judgement datasets or an actor result.

**Direct B→C prediction-cost check, CPU:** paired same1,002 roots/143 seed
clusters,2,000 draws. At s7 C raises depth16 error/V by+0.039984
[+0.024083,+0.057117]; s8 by+0.016712[+0.003517,+0.029959]. One-step all-action
error/copy-error rises by+0.007670[+0.006538,+0.008846] at s7 and
+0.007069[+0.005588,+0.008638] at s8. These are direct B→C contrasts, not the older parent→C rows.
Thus the health sensitivity gain comes with replicated prediction-cost increases;
`no_cost` is not supported. Original per-arm reports' historical 'Pending' field
is preserved. Bound source/input hashes and all metrics:
`evals/e19_fmamba_B_C_prediction_cost.json`; raw console output is in EDA logs.

**Narrowing the morning gradient lead:** frozen C's combined ordinary-gradient
dot objective is−9.46368e−5 at s7 but+1.03314e−3 at s8. Selection weights are
−1.12613e−4 versus+1.00475e−3. S8's ordinary/unchanged/death contributions to
the selection dot are+9.06315e−4,+3.57643e−5,−8.83157e−5. Thus the s7
unchanged-versus-damage endpoint conflict is NOT a seed-general explanation
of fresh-hit failure. It remains valid for that endpoint; no evidence is erased.
This is measured output-weight local GD geometry, not historical AdamW updates.
Reports:`evals/e19_s{7,8}_fmamba__gradient_budget.json`.

**CPU diagnostics declared before running,16:22:**
1. Run frozen canvas36k s7 versus matched slot-Mamba36k s7 on all original2,048
   held windows, with explicit FP32 CPU masked equations. Repeat with SSM reset
   at each step, retaining convolution/history/current-step updates. Preserve
   per-window same/moved/unseen errors and paired window-bootstrap intervals.
   All-kept CPU parity binds the reference equations; this separates endpoint
   SSM use and registration benefit, not an information ceiling or trained repair.
   Repeat unchanged once s8's36k checkpoint exists; source/checkpoint/pool hashes
   and atomic batches prevent mixed partial evidence. No world recipe changes.
2. Census all actually sampled E19 TRAIN targets, rather than30 batches: quantify
   ordinary damage/death/unchanged exposure by scroll, health value and the exact
   target-derived context mask. Rebuild current/next adjacency from the same
   frozen TRAIN-fitted detector. Compare selected/unselected and first-encounter
   support. Detector-derived groups are labelled; no simulator-truth claim.
3. On saved B/C traces at both seeds, condition candidate accuracy, router weight
   and read health on current health and camera motion. Contrast damage versus
   unchanged at matched strata. Determine whether the generator depicts a
   context-specific damage change or a common health template. A distribution
   resemblance is descriptive; causal attribution requires interventions.

**Separating router-family test declared,16:29:** the saved seven-logit sweeps
give0/31 fresh hits even when a per-case oracle selects the lowest true-target
health-token L1, at both C seeds; some other sweep settings depict damage on
10/31 and8/31 respectively. Extend the exact same candidate family to193 offsets
[-12,+12], plus copy/generator endpoints. One frozen forward supplies each case's
actual candidates; all subsequent sweeps are CPU. All279 ordinary hits and the
same512 unchanged controls, both C seeds. Verify every historical seven-offset
loss/health value and actual-mixture reconstruction. Report per-case best latent
L1, reachable health decrease, false drops and fresh/scroll/health strata.
This tests a specified one-dimensional frozen routing family, not a global
information ceiling; future targets are an explicitly labelled diagnostic oracle.
No model training/weight changes. GPU allocator capped16%, admission1600MiB;
source/input-bound batches resume. Original GPU training continues.

**Exposure alignment follow-up declared,16:34:** the completed census's first
encounter is CURRENT adjacency after no earlier adjacency in the original window.
The gate's `fresh` instead asks whether a PRE-move zombie is beside the POST-move
player, with no current/prior3-frame adjacency or prior3-transition damage.
These are different conditions; do not compare their counts as matching support.
Compute the gate-equivalent four-frame-history condition on original TRAIN
windows, using the same frozen zombie detector and token-estimated displacement.
Restrict targets to original positions3/4, where all four frames exist; separately
count whether E19's actual boundary leaves four frames available. Count actual
draws, window-target pairs and deduplicated `(episode_id, start+target)` events.
This is detector/estimated-motion coverage, not simulator-truth attribution.
Old census is preserved, and its `unique_targets` means window-target pairs.

**Dense router result,16:33:** both seeds completed193 offsets plus two endpoints
on791 cases (279 ordinary hits +512 unchanged controls). Maximum historical
seven-offset loss difference2.38e−7; health difference1.91e−6; reconstruction
9.54e−7. Fresh: latent-L1 oracle depicts0/31 at both seeds; ANY audited routing
setting can depict10/31 at s7 and8/31 at s8. Scroll: oracle0/51; any setting15/51
and13/51. Overall ordinary oracle79/279 and98/279, versus actual62 and67. This
separates poor candidates/reachability within this specific family from an
objective that chooses non-depiction even when depiction is possible. It is not
a global convex-hull optimum or information ceiling, and causal training history
remains unmeasured. Evidence:`evals/e19_C_s{7,8}_fmamba_from36000__e19_router_oracle.json`.

## 2026-10-06 — Reboot recovery, registration priority and mechanism replication (lead)

At07:42 AEDT no research process or recovered systemd unit was running. The
Oct5 lane stopped during E19 fmamba seed8 A: its last committed full state is
5,000/6,000 (Oct5 16:59), not the earlier500-update snapshot. No additional world
completed overnight. Canvas s7 remains13,000/36,000; s8 remains unstarted. All
four E17 L16 training endpoints are complete, including their damage/recall
readings and the fixed-clock control recorded above.

Recovery revalidated224 source/parent pins, three datasets and full canvas/M16
states (`20261005_recovery/restart_checks.json`,07:46:59 AEDT). Relaunched the
lead with MemoryHigh20G/Max24G and the runtime monitor. Training/evaluation
contracts remain unchanged: model/optimizer/samplers/CPU and CUDA RNG, original
row/split ledgers, immutable input hashes and atomic progress are preserved.
Old completed jobs first check/reuse their committed results; an active service
in startup/admission is not a new training update.

**Scheduling change, not a treatment change:** finish E19 fmamba seed8 A/B/C and
its endpoints, then bring E18 canvas36k s7/s8 ahead of the remaining E19 attention
controls. Both attention A/B/C seeds remain booked; original E17 H16 trajectory
heads/teval and M6 seed8 trajectory heads follow. Same-slot older-history use is
now measured beyond the time-row confound, making coordinate registration the
strongest unresolved memory intervention. C's extra health dose is a failed
seed7 repair, so it is not combined with canvas or promoted. E16 seed8 remains
held: its seed7 posterior already failed ordinary-hit fidelity after corrected
evaluation; repeating that whole architecture takes lower priority than the
separating registration and objective/head checks.

**Bounded diagnosis queue, declared before execution:** `lane86.sh`, service
`d4mj-oct06-diagnostic-replicas` (MemoryHigh8G/Max10G), first reads already-saved
canvas/fmamba s7 12k snapshots with the existing pool recall metric. This is a
descriptive learning-curve point, not a36k verdict or an early stopping rule.
Then it waits for each new B/C pair (fmamba s8, attention s7/s8) and repeats the
frozen router substitutions, local loss/logit derivatives,30 preselected actual
TRAIN batches with shared-head gradient vectors, and distribution-matched local
readability. Replica CLI plumbing was added to the existing diagnostic helpers;
no trainer/model mathematics changed and historical seed7 reports remain intact.
TRAIN checks explicitly assert B/C sampled-window and boundary ledgers equal.
GPU diagnostic allocators remain capped at16%; admission is serialized. Each
diagnostic retains checkpoint/data/source-bound batch journals and atomic results.

Purpose: test whether candidate fidelity, unchanged-versus-damage gradient
conflict and the motion-conditioned failure recur across training seeds and
backbones. These are endpoint head measurements, not a proof of the entire
optimization history. No new loss, lambda sweep, fork-supervised world or actor
claim is scheduled from the single-seed lead. Numerical failure diagnosis comes
before another literature-backed repair. Automatic experiment logs/summaries
remain in `artifacts/eda/levers_logs/`; this entry was written by the lead.

**Recovery export defect fixed,08:01 AEDT:** completed teval reuse first stopped
at `atomic_json(..., immutable=True)`. Its digest sorted integer horizon keys
before export and string keys after JSON reload, rejecting numerically identical
evidence. `20261005_recovery/teval_export.py` normalizes only JSON exports and
validates completed reuse against original source/checkpoint/runtime/input tensor
hashes, cached results and raw rows. All three seed7 report numerical differences
are0 and all raw rows exact. A focused check confirms repeated export leaves
bytes unchanged and changing an actual number still raises. Trainer/Store sources
and their checkpoint contracts are intact; original reports were not overwritten.
Main E19/E17 teval calls use this compatibility entrypoint. Recovered seed8 A
then actually resumed at5,000 and logged5,100; GPU100% at08:02. Completed E17
cheap readings are reused rather than re-exported unnecessarily.

**New descriptive canvas12k reading:** on the same2,048 held pool windows,
canvas versus slot Mamba capture is0.213211 versus0.268920 on1,615 same-slot
cells;0.281442 versus0.269807 on403 moved-slot cells. Unseen-cell squared error
47.6676 versus47.1440. This is not the36k endpoint, not a sealed block and has
no paired uncertainty. It shows no large early registration advantage; preserve
the declared36k/two-seed contrast. Raw log:`EDA/levers_logs/e18_recall_s7_at12000.log`.

**CPU access diagnosis declared and run,08:04–08:06 AEDT:** `e18_access.py`
counts the actual recurrent addresses on all2,048 held windows, then examines the
first16 with reentry under frozen12k canvas weights. Test whether the next entering
cell's remembered stream is gathered at prediction time; measure off-screen SSM
state retention, its zero-input gated output, and reading the same state with its
last observed query. The actual world runs with explicit FP32 CPU masked equations
checked against FunctionalMamba2's all-kept reference backend (max error<=1e-5).
This diagnoses direct state access, not a full-model information ceiling: spatial
attention/action/HUD streams could still provide indirect retrieval. Future
coordinates select diagnostic addresses only, never next-frame feature inputs.
All input/source hashes and per-window batches resume atomically. No training
architecture or scheduled budget changes; logs stay in EDA.

**CPU access result,08:06:** all2,018 recallable entering cells (1,615 same-slot,
403 moved-slot) are outside the current view and none matches the current cell's
gathered temporal-stream address. On140 cell events across six layers (840
layer/cell checks), stored SSM states are nonzero and held exactly (max change0),
but unobserved outputs are exactly0. Reading the held state with its last observed
query yields nonzero output on840/840 (mean RMS0.60979), reproducing the last
observed output exactly (max difference0). CPU all-kept reference max error1.19e-7.
Evidence:`evals/e18_access_s7_at12000.json` and hash-bound address/batch journals.
This identifies a missing **direct next-view read**: input registration alone does
not give the generator the off-screen cell's state. It does not show that
indirect retrieval is impossible or explain every canvas prediction error.

**Separating CPU follow-up declared before running:** keep these same16 windows
and all weights fixed. At the final temporal block, substitute the entering
output cell's temporal residual with the remembered cell's last-query read;
compare unchanged residual, zero residual and cyclically permuted remembered
reads. Inspect actual output and normalized generator errors separately, with
paired window-bootstrap intervals, and require non-treated outputs unchanged.
True next-view coordinates supply an oracle ADDRESS only, never true future
features. This is a causal component intervention on inspected windows, not a
deployable repair, not a fresh block, not evidence that an untrained readout is
optimal. An old-token-copy control locates the available recall benefit. Code:
`e18_reroute.py`; CPU FP32, source/input-bound per-batch resume. Original36k
training stays booked; no new architecture is trained from this lead.

**Reroute result,08:16 AEDT:** on140 entering cells in these16 windows, output
MSE is0.191480 unchanged,0.207736 correct remembered residual,0.191607 zero and
0.202316 permuted. Correct-read minus baseline is+0.016256[+0.002767,+0.032184]
under the declared window bootstrap. Generator MSE is0.558381 versus0.589950
(+0.031569[−0.015048,+0.085778]); generator weight only moves0.1491→0.1567.
Every untreated output is exactly unchanged. Copying the old observed token has
MSE0.057157, so past content is useful, but this frozen residual substitution
does not make it useful to the trained decoder. Neither inaccessible state nor
nonzero stored state alone explains the whole error. This is a16-window early
checkpoint component test, not an endpoint verdict. Evidence:
`evals/e18_reroute_s7_at12000.json`, with all per-cell errors preserved.

**Carry-contribution CPU check declared before running,08:20:** an old-query
read includes stored C/x/z and Mamba's direct D*x term; its nonzero output is
not evidence that the held SSM carries terrain semantics. `e18_carry.py` uses
the same16 windows/frozen12k model, separates full/zero-SSM/zero-D reads, reports
pre-normalization energy and output differences, and compares the off-screen
convolution buffer with its last observed buffer. The SSM is held by dt=0, but
the convolution still processes zero canvas inputs while off-screen. Check
against the prior CPU equations (max error<=1e-6); per-batch hash-bound resume.
No new world/head training or semantic information ceiling inferred. Logs in EDA.

**Carry result,08:23:**840 layer/cell reads, CPU equation parity error0. SSM
contributes mean1.2006% of (SSM energy + D*x energy) before gating/normalization;
removing SSM leaves cosine0.998424 and mean squared relative output change0.003156.
At the final block the SSM share is0.5731%, cosine0.999504. Thus the earlier
nonzero restored read was largely stored old-query/direct-skip information,
not demonstrated long-lived SSM terrain recall. Conv buffers still advance:
mean nonzero fraction40%, range0–75%, with off-screen gaps1–4. SSM-only dt masking
does not preserve the complete `(conv_state,ssm_state)` carry. This is a measured
component distinction, not proof that small SSM signals are semantically useless.
Evidence:`evals/e18_carry_s7_at12000.json`; per-query traces retained. No canvas
full-budget cancellation or memory-architecture promotion follows from this.

**Long-Mamba CPU mechanism test declared,08:25:** the fixed-clock experiment
proved older visual history helps, not that the SSM supplies that benefit. Six
four-tap temporal convolutions alone have a19-frame receptive field. On the
first64 roots of the original preselected200-root cohort, run both frozen L16
Mamba seeds7/8 under intact, reset-SSM-at-every-step, newest-conv-tap-only and
combined lesions. Preserve the current-step SSM update, newest conv coefficient,
biases, actions and time rows. Within each mode repeat the same older-visual
ablation; report paired seed-cluster differences in history gain on same/moved
cells aged6–15. CPU FP32, no GPU allocation, no new world training or capacity
ceiling claim. `e17_recurrence.py`, per-root source/input/weights-bound resume.
One-frame checks give max difference0 for all three lesions and restore every
model tensor exactly. Logs:`EDA/levers_logs/e17_recurrence_cpu.log`. The64-root
selection is bounded diagnosis, not a new sealed promotion panel.

**Long-Mamba component result,08:29:** both seeds complete, same64 roots/53
seed clusters,117 same-slot and162 moved-slot recallable cells aged6–15. Same-slot
older-visual-history gain (paired original versus fixed-clock visual ablation):

| frozen mode | Mamba s7 gain | Mamba s8 gain |
|---|---:|---:|
| intact |0.200451[0.009875,0.375462]|0.106067[0.005248,0.531627]|
| reset SSM before each step |0.000143[−0.000073,0.000730]|0.000775[−0.000028,0.006473]|
| newest conv tap only |0.001676[−0.000006,0.014201]|0.174590[0.001777,0.335621]|
| reset both |−3.02e−11|−3.85e−9|

Resetting SSM removes a resolved amount of history benefit at both seeds:
−0.200307[−0.375378,−0.009686] and−0.105292[−0.526545,−0.005096]. Thus the
older-history advantage in these frozen weights is not just stacked convolutions.
Older conv taps are additionally necessary at s7; s8 retains a positive SSM
history benefit without them (the change versus intact is unresolved). Do not
generalize the s7 convolution dependence to both seeds. Intact moved-slot gains
are−0.075594[−0.281255,0.037693] and0.031414[−0.131299,0.146506], both unresolved.
Combined-lesion visual-history effects are numerical zero. These FP32 CPU results
are paired within this64-root subset; do not substitute them for the earlier200-
root bf16 readings. Reports:`evals/*L16b40_from36000__e17_recurrence_cpu.json`.

**Additional CPU gradient decomposition, post hoc,08:30:**
`e19_gradient_budget.py` consumes the existing30 exact TRAIN batch gradients;
aggregation reproduces every saved gradient norm within1e−7. For a small negative-
gradient step on the measured `proj.weight`+`choose.weight`, ordinary-damage loss
changes to first order as−epsilon*(ordinary gradient dot objective gradient).
The dot is+2.39185e−6 in B (downhill) but−9.46368e−5 in C (uphill). In C,
projection weights contribute+1.79766e−5; selection weights contribute−1.12613e−4.
For selection weights, contributions are death+1.29224e−4, own ordinary+6.21531e−5,
unchanged−2.59101e−4, teacher−4.97166e−5, other+4.82731e−6. Removing unchanged's
contribution flips selection's dot to+1.46488e−4. This local conflict is concentrated
in routing, not a negative ordinary projection update or death opposing ordinary
damage.17/24 actual batches with ordinary damage have an uphill selection dot.
This is frozen endpoint, measured-weights, first-order **gradient-descent** geometry;
it does not reconstruct AdamW moment/clip/decay updates, unmeasured biases/backbone
gradients or the complete optimization history. No new loss-weight recommendation
is inferred from its zero-crossing, since C's false positives already worsened.
Evidence:`evals/e19_s7_fmamba__gradient_budget.json`, source-bound raw gradients.
The same CPU postprocess is booked after each forthcoming B/C gradient replica.

**Full-cohort CPU continuation declared,08:35:** differing conv dependence and
wide s8 intervals justify extending the same mechanism contrast to all original
200 fixed-clock roots. `e17_recurrence_full.py` imports the unchanged lesion/
statistics functions, verifies64-root prefix inputs, weights and all sources,
and reproduces one nonempty cached root before reusing64 records per seed.
Compute only the remaining136 per seed; new hash-bound per-root journals preserve
the64-root reports. No model training/GPU allocation/new seed block; this resolves
the existing component uncertainty rather than adding an architectural arm.
Logs:`EDA/levers_logs/e17_recurrence_full_cpu.log`.

**Full-cohort result,08:45:**200 roots/103 seed clusters,459 same-slot and589
moved-slot cells, both cold prefix reproduction errors0. Same-slot history gains:

| mode | s7 | s8 |
|---|---:|---:|
| intact |0.363581[0.165949,0.556088]|0.335796[0.149464,0.541272]|
| reset SSM each step |0.000437[−0.000225,0.001809]|0.000567[0.000095,0.001326]|
| newest conv tap only |0.003398[0.000130,0.009305]|0.316337[0.153132,0.495545]|
| reset both |4.79e−9|−1.04e−9|

Reset-SSM versus intact changes are−0.363143[−0.555952,−0.165652] and−0.335229
[−0.540685,−0.149075]. Conv-history removal changes−0.360183[−0.550845,−0.165430]
at s7 but−0.019459[−0.094954,0.055119] at s8. This confirms useful SSM history
at both seeds and a seed-specific additional convolution dependence. It does not
identify whether convolution supplies input content, SSM writes or read queries.
Intact moved-slot history gains remain unresolved:−0.005436[−0.098493,0.081546]
and0.049434[−0.014862,0.110202]. These are input-history/component interventions
on known diagnosis roots, not a new task-control result. Reports:
`evals/*L16b40_from36000__e17_recurrence_full_cpu.json`; all per-root evidence kept.

**Separating conv-channel follow-up declared,08:47:** `e17_conv_channels.py`
removes older taps separately on convolution outputs x, B and C (input/direct
skip, SSM write selector, SSM read selector), preserving newest taps and biases.
Same200-root/103-seed cohort and both frozen worlds, original-versus-fixed-clock
history comparisons and paired intervals. The interventions act across all six
layers, so their downstream interactions remain; do not label them independent
information channels. Reuse intact rows only after exact inputs/source/weights
and cold reproduction checks. No encoder/world/head training or GPU allocation.
This addresses the measured s7/s8 convolution difference, rather than proposing
another architecture. CPU journals and logs in EDA, original reports unchanged.

**Conv-channel result,08:59 AEDT:** both seeds completed the full200 roots/103
seed clusters,459 same-slot and589 moved-slot cells. Intact cold-cache reproduction
error is0 at both seeds. Same-slot older-history gains after removing older taps:

| mode | s7 gain | s8 gain |
|---|---:|---:|
| intact |0.363581|0.335796|
| x input/direct-skip taps removed |0.015059|0.324926|
| B write-selector taps removed |0.093322|0.368382|
| C read-selector taps removed |0.231337|0.287874|

Paired changes versus intact, s7: x−0.348522[−0.545966,−0.151697],
B−0.270259[−0.446134,−0.088327], C−0.132244[−0.237224,−0.041892].
At s8: x−0.010870[−0.070890,0.051192], B+0.032586[0.001338,0.060255],
C−0.047921[−0.090706,−0.008209]. Thus s7's conv dependence involves input,
write and read branches, strongest in x; it is not merely an old read-query
shortcut. S8 retains its SSM history benefit without older x/B taps, with a
smaller resolved C dependence. The B improvement is a post-hoc component result,
not a recommendation to remove B convolution. All interventions cross six
interacting layers; these effects are not additive semantic information shares.
Moved-slot history gains remain unresolved in every mode. Evidence:
`evals/*L16b40_from36000__e17_conv_channels_cpu.json`, all root records retained.

**E19 loss-mask interpretation:** the30 sampled C TRAIN batches contain293
selected death,47 ordinary-damage,876 unchanged and29 other health targets.
The hazard-context mask is therefore not event balancing: unchanged targets
outnumber ordinary damage18.6:1. Together with the measured routing dot budget,
this specifies the local opposition rather than attributing it to deaths.
It still does not reconstruct historical AdamW updates or prove that reweighting
will fix the false-positive/fidelity trade-off. Replicas remain booked before
selecting a repair.

**Primary literature checked after the access measurement:** Neural Map
([paper](https://arxiv.org/pdf/1702.08360),§3.1–3.2/§3.5) separates write, global
read and query-conditioned memory read, including key/value and localized read
variants. MapNet's [author code](https://github.com/jotaf98/mapnet/blob/master/mapnet.py)
explicitly feeds its entire registered map to localization, rather than merely
preserving inaccessible recurrent states. These are relevant access mechanisms,
not proofs of Craftax world prediction or of a Mamba readout repair. The SRU
navigation paper's spatial-transformation task separates remembered category
accuracy from coordinate accuracy ([source](https://arxiv.org/html/2506.05997v1),§4.4);
our direct address defect is a separate, now measured implementation limitation.
The [official Mamba2 implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
updates convolution and SSM caches separately (`step`), and its output adds
`D*x` to the SSM read. This confirms the component accounting, not a diagnosis
of our learned terrain semantics. Our own carries, lesions and model errors
provide that experiment-specific evidence.
The [PCGrad paper](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf),§2.2,
separates gradient conflict, magnitude imbalance and curvature; conflict alone
does not imply a stalled joint optimum. Our exact routing dot budget provides a
local measured mechanism. Curvature/AdamW trajectory and a successful intervention
are still unmeasured. PCGrad's multi-task results do not establish that protecting
our ordinary-hit subset at the expense of unchanged frames would be a good repair.
Revisited the user's [TrajGRU paper](https://arxiv.org/pdf/1706.03458),§5.2/Table3:
its balanced MSE/MAE weights actual target rain intensity (1,2,5,10,30), rather
than a surrounding hazard-context flag. At rain>=30mm/h, offline ConvGRU CSI
is0.0712 without balanced loss,0.1776 with it; balanced TrajGRU is0.1856.
That paper separately measures objective balancing and recurrent alignment.
Our C treatment did not reproduce event balancing, so its failed health repair
does not refute that principle. Different task, pixel objective and metrics:
the paper's gains are not evidence that any proposed Craftax weight will work.

**Runtime handoff,09:01 AEDT:** E19 fmamba s8 A finished6,000 at08:08. B is
actually training (last logged4,100/6,000; last atomic checkpoint4,000), then C
and the endpoints precede canvas36k s7/s8. The lead, runtime monitor and frozen
diagnostic-replica queue are active. Today's bounded CPU mechanism jobs are
complete; replica queue waits for complete B/C checkpoints and checks hashes
before running. No automatic process writes this notebook. Detailed live
updates remain in `../20261005_recovery/live_status.json` and EDA logs; this
paragraph is a timestamped observation, not a promise that later jobs finished.

## 2026-10-05 — E19 seed7 endpoints and measured generator/router mechanism (lead)

Reviewed after all three fmamba arms completed6,000 updates. A/B/C sampled-row and
boundary ledgers are byte-identical (240,000 windows,1.2M original targets per arm).
Each saw63,897 actual deaths; A scores them only at row4, B/C counts by row are
[12,956,12,658,12,861,12,678,12,744]. Source/data/parent hashes and failed evidence
remain preserved. These are the reused diagnosis futures, **not sealed new seeds**.

Primary health reading: sample0,k>=3,living successors;13,022 transitions,
279 drops>=2,12,623 unchanged,96 recovery. A true-fitted HUD reader detects278/279
real drops with0 false; the primary imagined-drop cut is1.5 units (separate from
check_damage's0.5 sensitivity). Reports and raw rows:
`20260927_levers/evals/{parent,e19_[ABC]_s7_fmamba_from36000}__e19_health.json`.

| arm | drops drawn /279 | false drops /12,623 | fresh drops /31 |
|---|---:|---:|---:|
| 36k parent |17 (6.09%)|98 (0.776%)|0|
| A: unchanged continuation |14 (5.02%)|115 (0.911%)|0|
| B: target-preserving boundaries |24 (8.60%)|122 (0.966%)|0|
| C: B + health dose1 |62 (22.22%)|297 (2.353%)|0|

Paired143-seed-cluster bootstrap,2,000 draws: B−A hit catch+0.03584
[+0.00707,+0.06954]; C−B+0.13620[+0.08393,+0.18750], with false rate
+0.01386[+0.00993,+0.01830]. C fails the declared learnable thresholds (>=30%
catch,>=50% fresh catch,<=2% false). Its repeated-frame w5 control draws only4/279
hits versus62 with real history: the improvement uses history, but it is not a
repair. TEST subset:17/87 hits,118/3,546 false,0/9 fresh.

**Cost is measurable:** versus the own36k parent, C's one-step error rises
0.123788→0.131317 (+0.007530[+0.006209,+0.008903]); H16 error/V rises
0.545125→0.582792 (+0.037668[+0.021174,+0.054015]). It fails the declared
<=0.005 one-step noninferiority margin. A/B aggregate cost differences are unresolved.
`20260927_levers/compare.json` retains all three paired comparisons. At the separate
0.5 sensitivity cut, C catches79/337 teacher-forced damage cases but only5/221
self-fed aligned damage cases; each world's alignment changes the denominator,
so this is not a paired teacher→imagination causal contrast.

### What the frozen interventions establish

`e19_diagnose.py`, `e19_gradients.py`, `e19_dose_audit.py`,
`e19_train_diagnose.py` and `e19_trace_readout.py` live in the existing campaign.
Each retains source/input/checkpoint-bound atomic resume evidence. No experiment
process writes this notebook; automatic logs/summaries use `artifacts/eda/levers_logs/`.

1. **The failure is motion-conditioned.** Exact recorded map-tile displacements
   identify51 scrolling and228 stationary ordinary hits; C catches0/51 and62/228.
   All31 fresh hits scroll. This is actual view motion, not merely a move action.
   Removing only the shared movement-logit contribution on HUD still catches0/51;
   forbidding neighbour copies also catches0/51. The suspected global camera gate
   does **not** explain the missing health updates.
2. **Generation is not a safe bypass.** On fresh hits C's health63 generate weight
   averages0.07399 and self weight0.92576. Forcing normalized generator63 draws10/31
   fresh hits, but4,537/12,623 false drops (35.94%); generator63 L1 on those fresh
   targets is1.980× self-copy error, and it beats the best copy on0/31. Replacing
   only health63 with its true successor draws31/31,0 false at the primary cut:
   the reader can see a correctly expressed health change in that token.
3. **The local objective explains why opening this gate is costly.** Actual mixture
   reconstruction max7.15e-7 (C). Sweeping only its generate logit on all31 fresh
   hits from0→+4→+8 raises mean health63 L1 0.47632→0.79085→0.94419 while catch rises
   0→7→10. Autograd locally favours opening on20/31 and closing on11/31: the gate
   is not universally driven closed. At epsilon0.01, finite differences agree on
   all791 C derivative signs (median absolute difference1.56e-6; max0.00700 at L1
   kinks). B has10/791 sign discrepancies at that finite step; no global derivative
   parity claim is made. Raw gradients and sweeps are retained in `__e19_local_gradients*`.
4. **The candidate contains signal plus larger unwanted change.** On fresh hits,
   normalized generator change has median projection0.620× the true health-token
   change, but orthogonal residual energy3.696× true change energy (stationary1.291×).
   This explains the observed fidelity tradeoff numerically; it is not a claim
   about the optimum of a newly trained generator or about information absence.
5. **Exposure and objective allocation are different measurements.** Exact6k ledger
   sees15,579 ordinary living>=2 drops,12,755 selected by the health mask;57,845
   selected terminal deaths and167,229 selected unchanged targets. For copying the
   true health token, deaths supply79.58% and ordinary damage8.74% of the extra
   loss. That copying baseline is NOT a trained-model gradient measurement.
   The independent frozen TRAIN check uses30 fixed recorded batches (updates
   0,200,...,5800;1,200 windows). C's actual masked loss share is60.46% death,
   10.68% ordinary damage,22.49% unchanged,6.37% other. Shared projection/selection
   gradient norms: ordinary0.008778,unchanged0.054089,death0.044246. Cosines:
   ordinary vs unchanged−0.52690; ordinary vs death+0.31365; health vs teacher
   +0.48731. Reconstruction max9.54e-7. This identifies an **endpoint output-head
   gradient conflict with unchanged-health examples**, not an encoder/backbone
   conflict, not the complete learning history, and not proof that death competes
   with ordinary damage (its measured cosine is positive).
6. **Decoder transfer still matters.** Distribution-matched ridge classifiers
   (FIT/validation seeds fixed; TEST43 seeds,87 ordinary drops vs3,546 unchanged)
   on current health63 + generated63 + action read C at AUC0.88874
   [0.85632,0.92072], B0.77365[0.72032,0.82089]. C's local hidden63 is0.86053.
   Thus0/31 drawn fresh hits does not mean damage information is absent. Fresh
   TEST has only9 positives: C generated63 AUC0.70995[0.54495,0.85922] against
   1,376 scrolling unchanged controls, and the HUD-input-only control is0.67959
   [0.42350,0.88638]. No resolved gain over that control or information ceiling is
   claimed. This repeats the M03 lesson: faithful successor geometry and probe
   readability are separate targets. `e19_[BC]_trace_readout.json` retains scope.

**Literature recheck after these measurements:**
[ITC](https://arxiv.org/html/2605.16457v1), appendixB.2, generates inventory and
screen edges separately from central copying. Its discrete generator/decoding
differs from our jointly trained continuous soft mixture. Earlier E5f already
tested generator loss/HUD generation: at18k, both switches still worsened
one-step error by0.147 and H16 by0.120 versus matched corrt. Do not silently repeat
that failed repair or equate ITC's mechanism with our implementation.
[CGSReg](https://arxiv.org/html/2607.15142v1) reweights image-space MSE and explicitly
limits manually chosen concept masks; E19 uses latent L1 and a zombie-context mask,
so its Pong result does not establish a repair here.
[VaGraM](https://arxiv.org/abs/2204.01464) targets value-sensitive directions, which
is closer to the fidelity/importance mismatch than increasing every coordinate
of a selected token; no critic/value-gradient intervention has run in E19.
[Rudy & Sapsis](https://arxiv.org/pdf/2112.00825), sec2.2, show rare-output weighting
can increase false positives and add a false-positive-sensitive term. That is a
matching warning, not proof that their fluid/MSE remedy transfers to latent L1.
[Delta-IRIS](https://arxiv.org/html/2406.19320v1) uses image reconstruction including
max-pixel loss; our uniform latent loss is a different target.

**E17 health endpoints complete:** all four L16-trained worlds, window15:
attention s7/s8 teacher catch5/337,16/337; fmamba s7/s8 29/337,46/337 at0.5 cut.
All four self-fed catches0 (aligned damage denominators243,238,240,241 respectively).
Long history helps some teacher readings but does not repair health forecasting;
recovery and net−1 change-class accuracy also remain0. Recall endpoints are still being collected;
do not infer the complete two-seed long-memory verdict from partial logs.

**Queue at16:12 AEDT:** the lead remains active on E17 window15 recall, followed
by window5 and imagined recall. E19 fmamba seed8 and both attention controls remain
booked before the original E18 canvas36k continuations; canvas s7 remains13k and
s8 is unstarted. Completed mechanism diagnostics did not alter these treatments.
Main resource limit is now MemoryHigh20G/Max24G (persisted in recovery launch),
with bounded diagnostic lanes capped separately. Source/hash-bound resume is retained.
The failed C recipe is not promoted; finish the matched contrasts and distinguish
event-calibrated consequence learning from a latent-fidelity repair before adding
another full architecture. No actor improvement is established.

**E17 fixed-clock control declared before execution (16:22 AEDT):** window15
teacher recall completed. On same-slot cells last seen6–15 steps ago, fmamba
capture is0.615924/0.643393 (s7/s8), attention0.430657/0.357305; these clear the
declared `long_recall_same` point thresholds. Moved-slot capture is0.405302/0.440272
for fmamba and0.426823/0.417918 for attention; the older `moved_unsolved<=0.35`
reading fails in all four arms, although Mamba has no consistent advantage there.
Window5 evaluations continue. Source inspection caught a confound in w15→w5:
`TWorld.inputs` adds `self.time[:t]`, so truncation changes the output time row as
well as removing visual history. It cannot alone establish memory use.
`e17_clock_control.py` now runs on200 roots selected with RNG20261005, sample0,
all four worlds: preserve window length, actions, learned time rows and latest5
frames, replacing only older frames with the oldest retained frame. Score paired
same/moved-slot ages6–15 with2,000 seed-cluster intervals and per-root resume.
This is an intentionally inconsistent visual-history ablation, not an input
information ceiling. No world is retrained; the original E17/E18/E19 lanes continue.
Its raw logs go to EDA and results remain in the existing levers campaign.

**E17 completed and clock control resolved,16:28 AEDT:** window15→window5 same-slot
ages6–15 capture: fmamba s7 0.615924→0.369022; s8 0.643393→0.364637. Both clear
the declared>=0.20 context contrast, but that contrast retains the time-row confound.
The added fixed-clock intervention also finished, on the same200 roots/103 seeds
and459 same-slot,589 moved-slot cells. Original→older-visual-history-ablated capture:

| fixed-clock arm | same slot, original→ablated | history gain,95% seed-cluster CI | moved-slot gain,95% CI |
|---|---:|---:|---:|
| fmamba s7 |0.673509→0.310275|+0.363234[+0.169424,+0.557507]|−0.005305[−0.105591,+0.073193]|
| fmamba s8 |0.671266→0.334793|+0.336473[+0.144386,+0.548151]|+0.049222[−0.017567,+0.114403]|
| attention s7 |0.460175→0.309855|+0.150320[+0.026237,+0.352233]|+0.034048[−0.019931,+0.114090]|
| attention s8 |0.370638→0.381185|−0.010547[−0.040241,+0.001887]|−0.003315[−0.011572,+0.003787]|

Paired Mamba−attention history-benefit contrasts on these exact cells are
+0.212914[+0.079474,+0.363394] at s7 and+0.347021[+0.153328,+0.556499] at s8.
Moved-slot contrasts are unresolved at both seeds. This identifies useful older
visual evidence for same-screen-slot recall beyond a clock-only explanation;
the inconsistent-prefix ablation still does not isolate every possible history
mechanism. Capture is an error-normalized recall statistic, **not tile accuracy**
or actor return. Evidence: `evals/*__e17_fixed_clock{,_rows}.*`,
`e17_fixed_clock_contrasts.json`; raw rows retained. No world weights changed.

Imagined window15 recall also completed. Same-slot ages6–15 capture: fmamba
s7/s8 0.265160/0.277738 versus attention0.123668/−0.057215. These are descriptive
comparisons on each world's still-camera-aligned subset (Mamba1,480/1,496 cells,
attention1,452/1,425), **not paired causal effects**, and are substantially below
teacher-forced recall. Novel terrain is worse than neighbour copying in all four
imagined worlds (gain−0.35 to−0.40). Long-context recall is real on observed history;
long imagination and health remain separate unresolved capabilities.

Queue now actually trains E19 fmamba seed8 A; B/C and matched attention controls
follow, then unchanged E18 canvas. Main/service monitor remains active; completed
bounded diagnostic lanes are stopped. Clock and TRAIN-loss diagnoses have recovery
entries and hash-bound per-root/per-batch journals. Notebook was updated by the
lead after verification; automatic experiment output remains in EDA logs.

**Live verification,16:31 AEDT:** E19 fmamba seed8 A saved its full500/6,000
checkpoint (objective0.04601904,gradient0.05984034,peak2.260GB). Lead service is
active/running; GPU98%. B/C and attention controls are pending; E18 has not advanced
past its preserved13k checkpoint. Logs reflect actual resumed work, not old rows.

## 2026-10-05 — Lead queue: E19 implemented and numerically validated before launch

E19 implementation is now `20260927_levers/e19.py`, within the existing campaign. No new experiment directory.
Predeclaration/contrasts/scope: `20260927_levers/E19.md`; priority coordinator: `20260927_levers/lane81.sh`.
A=unchanged continuation, B=target-preserving random boundaries, C=exactly B + health dose1;
fmamba first, matched full-attention control, both seeds7/8, own36k parents,6000 updates, batch40, same rows/RNG.
Every five original transitions/window remains scored; no crop-and-pad target loss. Both terminal and nonterminal
windows use the same boundary distribution. This changes context length/reset as well as position.

**Important measured/design limit:** randomized terminal row counts do not flatten terminal prevalence. Expected
scored targets/window by row are[1.8,1.4,1.0,0.6,0.2], giving expected TRAIN death rates
[2.96%,3.80%,5.33%,8.88%,26.63%]. The method broadens death exposure; it is **not claimed alias-free**.
Position/repeated-frame checks are necessary to measure remaining dependence. The primary 1.5-unit draw cut is
explicitly separate from `check_damage.py`'s0.5-unit sensitivity. Paired seed-cluster intervals/raw rows retained.

**Actual CUDA, not CPU-only:** old/control loss delta0, gradient max2.91e-11; B/C base-loss delta0;
all three arms four uninterrupted updates vs two+serialized restart+two: parameter delta0, AdamW max7.28e-12,
all sampler/boundary/CPU/CUDA RNG identical. Source/recipe mismatch and corrupt payload rejected. Synthetic
copy control confirms all action/target/mask pairs and denominator exactly retained (base delta0, extra term delta4.77e-7).
Evidence `e19_verify_{cpu,cuda}.json`, `e19_target_ledger_proof.json` in levers. Full batch40 finite CUDA updates:
fmamba A peak2.260GB (first autotune), B/C0.731GB; attention A1.585GB, B/C1.625GB.
The earlier concurrent smoke OOM was a256MiB kernel-benchmark allocation while M16 occupied3.05GiB;
no research treatment failed and M16 survived. Do not use cold first-update timing as steady throughput.

**What landed:** M16 seed7 complete6000, final objective0.05307013. A16 both seeds already complete.
M16 seed8 reached3000, objective0.04806252; full model/optimizer/order/CPU/CUDA RNG state verified before pausing
for exclusive batch40 validation (`e19_queue_boundary.json`). Source/data/parent identities unchanged. No new E17
scientific endpoint is inferred from these losses. Canvas seed7 remains13000/36000, seed8 unstarted.

**Priority:** corrected E19 Mamba seed7 A now first; resume M16 seed8 from3000 next, then E19 B/C and its
matched per-position/damage/cost diagnostics. E17's short health/recall checks follow before seed8/control replicas;
E18 remains booked unchanged; expensive H16 heads come after mechanism/training lanes. One serialized coordinator
replaces competing waiting services. No old failed/partial results are overwritten. Full E19 states every500 updates,
atomic source/data/parent/runtime-bound resume, including exact sampled row/split ledgers. `e19_eval.py` journals
root batches; `e19_note.py` appends completed measurements to
`artifacts/eda/levers_logs/e19_<backbone>_s<seed>_summary.log`. Notebook updates are written by agents after reviewing
the evidence; experiment processes never write here. This corrects the initially proposed automatic notebook writer.
Primary-source sanity check remains
[IRIS sampling/masks](https://github.com/eloialonso/iris/blob/main/src/dataset.py) and
[VaGraM objective mismatch](https://arxiv.org/abs/2204.01464); neither proves our mechanism nor prescribes our mask.

**Launch verified, 12:00 AEDT:** `d4mj-oct05-lead.service` is actually training E19 Mamba seed7 A, not waiting.
At500/6000 it saved a full hash-checked state: objective0.04547052, gradient0.05720623, peak2.260GB. All100,000
original transition targets seen; the20,000 sampled-window and boundary ledgers independently reproduce seed11/19
exactly, including saved generator states; model/AdamW tensors finite. Control's5,323 sampled deaths remain row4
by design. Actual A/B/C treatment comparison is still pending. Evidence `20260927_levers/e19_launch.json`.
Live monitor now tracks the single lead queue; old competing waiting lanes are stopped. `20261005_recovery/launch.sh`
resumes the unified queue after another reboot; each E19 arm self-resumes its immutable contract/checkpoint.
Evaluator correctness controls on the actual13,022 diagnostic transitions pass: copying-health detects0/279 true
>=2 drops; injected exact labels detect279/279 with0/12,623 false drops; paired cluster contrast1.000[1.000,1.000].
These controls validate counting/bootstrap, not any world's performance (`e19_eval_mechanics.json`).

**Latest saved E19 checkpoint, 12:02 AEDT:** control A **1,000/6,000**, objective0.04520756, gradient0.07853370;
full state hash and finite weights verified,200,000 original transition targets scored. B/C outcomes remain pending
(`20260927_levers/e19_live.json`).

## 2026-10-05 — Reboot recovery (reviewer; source/data verification before relaunch)

At 09:42 AEDT (22:42 UTC October 4), reboot inspection found no research Python processes and no loaded research services.
GPU 474 MiB desktop / 5,375 MiB free. Final A16 s7/s8 worlds remain complete at 6,000 updates. Canvas s7's actual full
state is **13,000/36,000**, including optimizer, batch-order generator and CPU/CUDA RNG; s8 has not started. M16 s7 has
no full-run state (historical log reached 1,000); the separate 500-update smoke must not be substituted. Restart it from
its own M6 36k parent. M16 s8 has not started. No newly completed scientific report was found after the preceding audit.

Recovery plan and immutable input hashes: `20261005_recovery/PLAN.md`, `inputs.json`. The original canvas state is
archived separately before its working state advances. Actual Raw pool and Raw-long label/token SHA256 checks are
required against their historical manifests. Numerical trainer source is unchanged; its only prior amendment changes
checkpoint cadence to 1,000 for booked lanes, preserving loss, optimizer and sampler. Historical source pins are retained.

**Actual CUDA resume check, before relaunch:** from canvas update13,000, four uninterrupted updates versus two+two
with optimizer/RNG restoration: parameter max absolute difference **0.0**, optimizer tensor difference
**9.094947e-12**, batch-order / CPU / CUDA RNG all exactly equal. This passes the previously declared mechanical
tolerance 1e-6; it is not a promise of long-run bitwise equality. Evidence: `20261005_recovery/gpu_resume.json`.

The legacy M6 s7 H16 report remains valid with its stated readout scope. The interrupted s8 FIT feature file has no
batch journal and no completed DEV/head/report; it is preserved, never trusted by file size. Its evaluation uses the
new resumable evaluator in a separate result directory after the booked training lanes. E17/E18 retain their declared
treatments and admission rules. Detailed status is recorded in `20261005_recovery/live_status.json` and append-only
`status_events.jsonl`; an active waiting service is distinguished from a computing job.

**Priority remains separating mechanisms.** E17 asks whether trained longer context changes recall; E18 tests mature
world-coordinate registration. Neither alone fixes health reconstruction. Corrected E16 s7's true-future posterior
still draws 0/305 ordinary damage, 0/32 starvation and 0/112 recovery; another prior replica cannot separate that
bottleneck. E19 remains a design until its unchanged continuation / de-alignment-only / additional-dose contrasts are
implemented and verified; do not conflate layout and dose or import the scratch implementation, not recovered in this
review, as independently verified code.

**New data-only diagnosis, exact TRAIN subset (30,599 windows):** all **8,149 actual deaths** occur at predictor row4,
none at rows0–3. The old78% statistic concerns ≥2 health drops, not death labels. Ordinary ≥2 damage with a living
successor is2,000/144,846 (1.3808%). Rebuilding the stored health-context mask from the exact TRAIN-seed ridge yields
**0 disagreements/163,235 labels**; it covers1,636/2,000 ordinary ≥2 drops (81.8%). Its masked ordinary-hit rate is
6.9135%, rather than a universal hit probability;364 drops are outside it. Exact counts/source pins:
`20261005_recovery/health_mask.json`, `health_design.json`.

**E19 design correction before launch:** terminal crop/right-pad discards earlier targets. Enumerating offsets gives
valid targets152,995→136,853 (−10.5507%), ordinary damage2,000→1,636.6 (−18.1700%), death prevalence5.3263%→5.9546%.
Thus it confounds boundary randomization with exposure/normalization. A candidate instead predicts both retained prefix
and terminal suffix, preserving every action/target pair and the loss denominator. CPU checks on real A6: no-split
loss/gradient difference0, future-padding difference2.47e-6; all offsets preserve five pairs. It still changes context
lengths/resets, so it tests a boundary/layout intervention, not positional embeddings alone. Actual Mamba layout check
and matched branch trainer remain pending; **no E19 training launched**. The hp1 whole-pool selected-coordinate
multiplier is~400.18×; weighted health-drop prevalence24.19% (including deaths) vs5.38% is a loss measure, not a
predicted hit rate. `20261005_recovery/FINDINGS.md` records controls, limits and checked primary literature.

**Live recovery, 10:03 AEDT:** M16s7 genuinely restarted, logged1,000/6,000 and wrote a73MB full state; s8 is queued.
CUDA allocated peak2.872GB; actual device usage~3,599MiB leaves~2,251MiB, below canvas's2,556MiB admission. E18 is
explicitly serialized behind E17 including evaluations, preventing an intra-evaluator allocation increase colliding
with canvas after head-fitting releases memory. Only its waiting shell was replaced. Canvas resumes unchanged from13k,
then s8 from scratch. M6s8 H16 follows in a separate resumable directory. M16 fresh-start losses do not reproduce the
discarded run (500:0.06193946 vs0.06049695); no cause of this difference is claimed. Current inputs/parents are pinned.

**Verified recovery point, 10:15 AEDT:** M16 s7 is now **2,000/6,000**, objective 0.04926093, gradient norm 0.04280990,
actual GPU utilization 100%. Full state decoded: finite weights, 16-row time table, optimizer/order/CPU/CUDA RNG present;
archived at `20261005_recovery/recovery_inputs/m16_s7_u2000.state.pt`. Pinned numerical sources remain unchanged.
The initial verification expected the earlier 1k update but the live run had advanced to 2k; dynamic-state validation
resolved that stale assertion, not a training failure (`m16_checkpoint_verified.json`). E18 and M6 s8 readout are waiting
in their scheduled services. Subsequent reboot entry point: `20261005_recovery/launch.sh`, which refuses duplicates and
rechecks immutable source/parent/data pins and decodable working/final states. Detailed new counts and the primary-
literature check are in `20261005_recovery/FINDINGS.md`.

**Final live check, 10:22 AEDT (23:22 UTC October 4):** M16 s7 logged **2,500/6,000** (objective 0.06221406), latest
saved full state **2,000**. GPU **100%**, 3,601 MiB used / 2,248 MiB free. E17 is computing; E18 and M6 s8 H16 services
are waiting in the explicit serial queue. Source/parent and all dataset bytes were rechecked by the future-reboot guard
(224 static pins, three datasets; both current full states decoded). Syntax/whitespace checks pass. No new full scientific
endpoint was claimed from this recovery; the new results are the verified health-count/mask/layout diagnostics above.

## 2026-10-04 late evening — Pending-run rationale, causal scope and evaluator resume repair (reviewer)

Full ten-claim audit, separating evidence and priorities: `20261004_resume_audit/PRIORITIES.md`. No running experiment
was stopped or given a different training treatment. E17 is a long-context-use comparison; E18 is a coordinate-registration
repair test. Both are justified, but neither repairs the ordinary-health reconstruction bottleneck by itself.

**New M6 seed-7 H16 trajectory result:** DEV-B 1,139 opportunity roots; trajectory 0.644105, snapshot 0.624918,
paired difference +0.019187 [0.005559,0.033441]. Same-panel prior 0.616083, one-real-future 0.679366. It roughly matches
attention's 0.645/0.646 trajectory scores; do not claim an architecture advantage before the matched seed-8 comparison.
Evidence: `20260927_levers/evals/h16traj/corrt_raw_teacher_s7_fmamba_u36000.json`. This is the previously opened DEV
panel, not a new sealed actor result; the continuation AUC was produced by the old tie-rank implementation.

**Corrections/limits for the bigger-picture claims:**
- Sparse-consequence objective allocation has intervention evidence, but is not the sole explanation of all failures.
  At 100k s7 catches 0.991 of strict consequences, s8 0.877 and still zero tables. The strict labels omit tree mining.
- Real-HUD injection recovers 94–97% of the H16 **real-fitted decoder transfer gap**, not 94–97% of an actor's attainable
  improvement. It replaces all 18 HUD tokens and supplies the true future outcome. Health fails while other HUD
  reconstruction can improve substantially (corrected E16's aggregate HUD error reduction 79.2%).
- Position/scan dependence is demonstrated; end-aligned sampling's causal contribution still needs a de-alignment-only
  retrain. The older “rawlong deaths sit at position15” wording conflates target frame with predictor input row: at L16
  terminal frame15 is predicted from row14; row15 has no teacher target. The earlier w16→w15 evaluation correction stands.
- E16's ordinary-hit bottleneck exists even with true-future posterior conditioning. More prior samples cannot fix
  that bottleneck. Rarity, magnitude, terminal position and objective weighting have not yet been causally separated.
- An unseen current cell may have been observed earlier: memory should resolve those cases before stochastic generation.
  Local-predictor agreement is not a universal information ceiling. Our latent L1 optimum is a coordinatewise median,
  not an MSE conditional mean. The causal contribution of class geometry to false scrolls remains unisolated.
- E17 short→long changes more than context length (ledger/time table/targets per update/additional training/fresh optimizer).
  E18's historical fcanvas holds
  SSM state on absent frames but advances convolution state; both carries contribute to measured recall. Neither finite-
  window evaluator proves an advantage for persistent streaming Mamba carries in an actor.
- Keep E19 separated: unchanged continuation, de-alignment alone, then the same de-aligned batches plus health-context
  weighting. Positive-only weighting already produces 24–30% false consequences versus 2–3% with attempt negatives.
  Matching negatives are necessary but future-dependent masks still need conditional calibration checks. No E19 run
  was launched in this review. Current literature rationale and the proposal's superseded claims are recorded in PRIORITIES.

**Resumability repaired for future launches:** previously `check_h16_traj.py` overwrote partial feature files, held
optimizer/RNG/best states only in memory, deleted features and reran completed worlds. It now journals hash-checked
complete root batches, saves head optimizer/model/best/RNG each 200 updates, persists E16 draw/generator/end states,
retains raw score/decision rows and skips complete worlds. Sources, checkpoints, input/label contents, window and
runtime are bound; corrupted storage fails; changed contracts cannot reuse old partials. Atomic publication and writer
locks protect evidence. Legacy unbound reports are preserved; `--result-dir` permits a separate re-evaluation.
Chunked standardization removes the whole-matrix float32 allocation without changing its arithmetic. Continuation
AUC now averages tied ranks (constant 0.5); primary safe-choice metrics are unchanged.
Long-window H16 attention also uses four roots per forward (68 action branches), matching the Mamba batch, rather than
16 roots / 272 branches under the lane's 3 GiB admission. This is a conservative evaluation-memory cap, not a measured
GPU peak or a training change. Short-window attention retains its historical batch16.

The other queued evaluators (`check_damage.py`, `check_recall.py`, `teval.py`) now checkpoint root-batch outputs or recall
accumulators and skip completed worlds under the same source/data/checkpoint binding. Numerical forward/metric contracts
are unchanged. CPU original-versus-amended and interruption checks pass for damage, pool recall, future recall, imagined
recall and teval, with zero forward calls on completed reuse. H16 feature/head/optimizer differences are exactly 0,
RNG/best states match, and negative corruption/contract/layout controls reject. Evidence:
`20261004_resume_audit/verify_h16_resume.json`, `verify_queued_eval_resume.json`; original evaluator sources archived nearby.
CUDA bitwise equivalence was not tested: the GPU stayed with the running research jobs.

**Existing live H16 process is still legacy and non-resumable.** Source edits cannot retrofit code already loaded in
PID11404. It was deliberately not interrupted; its new seed-7 primary result is retained, seed8 is still running.
New cache namespaces avoid its cleanup glob. New E17/E18 evaluation subprocesses will load the amended code. Training
already saves full states every 1000 updates for these lanes. At 22:31 Sydney (11:31 UTC), E18 s7 last logged 11000/36000;
E18 s8 has not started. E17 A16 s7/s8 are complete; M16 is still waiting for GPU admission (its old 1000-update log is
the interrupted run). Live status continues in `20261004_resume_audit/live_status.json` and `status_events.jsonl`.
At 22:51 Sydney (11:51 UTC), a read-only checkpoint inspection confirms fcanvas s7 full state **12000/36000** and both
A16 full states **6000/6000**; M16 s7 still has no full-run state. The M6 seed-8 H16 result remains unpublished. No jobs
were stopped during this review.

## 2026-10-04 — Independent resume audit (reviewer)

User explicitly lifted the October 2 read-only restriction and authorized notebook/memory writes and experiments.
Protocol and diagnostic plan: `20261004_resume_audit/PLAN.md`. Existing negative evidence and other agent's changes preserved.

**New protocol defect, caught before the pending E17 evaluations:** `tworld.rollout_losses` supervises positions 0..L−2,
and `teval.step` reads the final input position. With a 16-frame teacher window, prediction from **15 input frames** is
supervised; prediction from 16 is not. A CPU backward pass on the completed A16 s7 checkpoint gives exactly zero gradient
for time row 15; row 0..14 gradient norms range 0.000507–0.001388. Lane73's damage, H16-trajectory, teval and comparison
commands have been corrected from window 16 to **15**, including output tags. Training is unchanged. The recall commands
already used 15 correctly. No final stage-2 window-16 reports had been produced, so no result is being silently replaced.

**Corrections to the causal scope of the entries below (historical numbers retained):**
- E16's posterior is trained through **uniform latent L1 plus commitment**, not uniform MSE: `dworld.train_a` calls
  `tworld.rollout_losses(..., 'teacher')`, whose error is `.abs()`. The pre-VQ/code/decoder AUCs localize weak ordinary-hit
  encoding and output suppression. Rarity, magnitude and terminal position differ together between main and terminal
  windows; their individual causal contributions have not been isolated.
- "Neither 36k world uses its history" was too broad. The aggregate check_context reading fails its declared threshold,
  while the later terrain-recall measurements show a substantial history-dependent contrast. Health drawing is strongly
  position/scan-length dependent, as demonstrated by the same-frame repetition and time-row interventions.
- The damage-rule "ceiling" numbers (0.621 at 5 frames, 0.857 at 6) are in-sample plug-in results for one coarse set of
  history features. They are **not upper bounds** on arbitrary predictors or pixel histories. Rule cases also use the
  measured next-frame view shift and exclude fatal transitions. Do not use 0.62 as an architectural pass ceiling.
- "Per-slot Mamba can carry at most ~61%" is not a capacity bound: its within-frame attention can route information
  across slots. The measured 39.1% is the share of this terrain-reentry oracle's gain assigned to moved-slot cases.
  The memory incentive also clips negative sighting-copy gains to zero, so it is an optimistic selective-copy statistic.
- Window-5 versus window-1 recall changes time position and scan length, not only visible history. The new frozen-checkpoint
  sighting intervention holds those fixed and edits only historical token contents, with an unrelated-token control.
- Root-aware readout gains establish that context helps this head family. They also add root features and double the
  concatenated input width; they do not isolate one zombie cue or prove equivalence from an unresolved difference.
- E19 combines death de-alignment with health weighting. As designed it can test their combined effect, but cannot
  identify the separate cause. Keep a de-alignment-only arm before attributing a gain to the dose. Its true-next-frame
  zombie mask covers 80.3% of main −2 events, not all damage; the remaining causes are not yet attributed exhaustively.

**Interrupted-run reconciliation, before restart:** A16 s7/s8 completed and each full state records update 6000. M16 s7
logged update 1000 but saved no full-run state; its separate 500-update smoke is not a resume checkpoint for that job.
M16 s8, E18 fcanvas s7/s8, M6 H16 trajectory heads and check_delta are unfinished. No research process was running when
checked; GPU compute was idle apart from the desktop. Pending E19 remains unapplied. Restarting M16 s7 must use its
declared 36k parent; completed A16 weights and prior reports are retained. Live statuses/results are recorded in the audit
folder and updated here as they land.

**Primary-literature correction:** Delta-IRIS §2.2 uses a mixture of L1, L2 and max-pixel reconstruction losses, not our
E16's uniform latent L1. Its Crafter health examples support the method's possibility, not adequacy of our loss/encoder.
Mamba's input-dependent Delta is a retention/overwrite mechanism; uncertainty must be measured (pending check_delta).

**Fixed-length sighting intervention completed:** 512 same-slot cases from 168 episodes and 236 moved-slot cases from
35 episodes, existing held-out main windows. Same current frame, actions, window length and time positions in all arms;
only historical target sightings or equally many unrelated map tokens replaced with held-out donor tokens. Same-slot
target error, original → target-sighting swap → unrelated swap:

| world, 36k | original | target swap | unrelated swap | target − unrelated, episode-clustered 95% interval |
|---|---:|---:|---:|---:|
| attention s7 | 37.133 | 52.965 | 36.962 | +16.003 [10.651, 21.487] |
| Mamba s7 | 16.740 | 125.825 | 16.623 | +109.202 [92.734, 126.301] |
| attention s8 | 50.370 | 50.332 | 50.291 | +0.041 [−0.029, 0.131] |
| Mamba s8 | 20.053 | 101.108 | 19.979 | +81.129 [66.567, 96.026] |

Mamba's predicted token is pulled towards the donor (projected displacement 0.278 / 0.204), unlike unrelated swaps.
Moved-slot target-minus-unrelated changes remain unresolved in both Mamba seeds: −0.268 [−0.635,0.011],
−0.121 [−1.015,0.523]. This establishes actual use of historical cell content in the same-slot case, independently of
scan-length/time-position confounds. It does not identify convolution versus SSM storage or establish actor utility.
Edited histories are exploratory interventions. Evidence and checkpoint/pool hashes: `20261004_resume_audit/sighting_results.json`,
raw paired predictions in `sighting_rows.pt`, episode intervals in `sighting_episode_intervals.json`.

**Interrupted check_delta completed on GPU:** 1,009,764 alive map-cell predictions per seed. Mean Delta error AUC
0.564267 / 0.532289; final-layer Delta 0.585018 / 0.600444; input-change 0.471142 / 0.471620; disagreement between worlds
0.641758 / 0.651290. Mean Delta within input-change quintiles 0.563717 / 0.531722. All three declared readings FALSE.
This rejects mean Delta as the proposed useful error gate in these worlds, not every possible learned uncertainty head.
Before the run, fixed tie handling in `check_delta.auc`: constant scores now give 0.5 (old arbitrary ranks could give 1.0).
Direct positive-negative pairwise checks give exactly 0.5, 0.75 and 1.0. No prior completed report was overwritten.
Evidence: `20261004_resume_audit/delta_results.json`, full log `artifacts/eda/levers_logs/oct04_check_delta.log`.

**Second pending evaluator defect fixed:** teval treated `rawlong` as a new encoder and would fail dictionary lookup.
Its pool manifest pins Raw bridge; Raw bridge and Raw joint encoders match in all 209 tensors. `build_cache('rawlong')`
now aliases Raw, with all five returned tensors verified exactly equal. No training/checkpoint change is involved.

**Resume order:** complete M6 H16 trajectory heads at the corrected batch 4, then restart corrected lane73 (E17) and
lane74 (E18) under GPU admission locks and systemd memory limits. In parallel, finish the amended full E16 seed-7
evaluation (all roots, eight samples, posterior conditioning on every frame) rather than relying only on its CPU subset.
E16 seed-8 training and E19 stay held. E18 still tests the historical fcanvas implementation, whose absent-frame hold
applies to the SSM but not the convolution (see the September 29 correction); it is not a test of every carry-transport
design. Local DRAMA replay uses count-dependent sampling probabilities, not strictly uniform starts; the relevant
precedent is avoiding deterministic terminal end-alignment.

**Carry-path intervention completed:** 64 same-slot cases from 55 episodes and 64 moved-slot cases from 30 episodes,
both Mamba seeds, CPU float32 reference scan. Splitting a scan just before its last input reproduces the unsplit path
to max absolute error 1.67e−6 / 3.76e−6. On same-slot cases, donor pull (s7 / s8): intact 0.248871 / 0.256607;
target SSM cleared 0.136614 / 0.097742; target convolution cleared 0.074911 / 0.034068; both cleared 0.001349 / 0.001000;
unrelated slot's two carries cleared 0.248634 / 0.256864. Attenuation from clearing each target component resolves in
both seeds; clearing an unrelated slot does not. This locates the measured short-history content path in **both**
convolution and SSM carry. It is not an additive attribution or evidence of long-range actor utility.
Evidence: `20261004_resume_audit/carry_results.json`, paired rows and attenuation intervals beside it.

**Full amended E16 s7 evaluation completed (19:23:54 Sydney):** 1002 roots, 8 prior samples. Posterior health readout
accuracy remains **0/305** for ≥2 damage, **0/32** for −1, **0/112** for recovery; unchanged accuracy 0.9995 on 15387
transitions. Prior drawn-hit rates fresh 0.0000, already-beside/no-recent-hit 0.0003, recent-hit 0.0003 versus empirical
0.9728 / 0.4802 / 0.0280. Health-drawing and calibration readings FALSE. Delta reduces aggregate decoder error by
79.2% on HUD and 92.8% on other map cells, which does not rescue health drawing; the channel carries much deterministic
map information. The reconstruction/readout failure remains when the posterior sees truth, so more prior samples cannot
remove that particular defect. This is not a claim that hidden features contain no damage information.
Evidence: `20261004_resume_audit/e16_fixed_result.json`, full log `artifacts/eda/levers_logs/oct04_e16_fixed_s7.log`.

**Resume-state protection:** E17/E18 now pass `--state-every 1000`; default remains 6000 for historical recipes. The new
argument changes only full-state serialization cadence, recorded in checkpoint args. Forward, teacher loss and gradients
are unchanged (CPU comparison against the archived original training source: maximum loss/gradient differences 0;
serialization does not change CPU RNG). Old source preserved in `20261004_resume_audit/TRAIN_SOURCE_BEFORE_RESUME.py`;
verification in `checkpoint_cadence_check.json`. Completed checkpoints are untouched. Because the M6 H16 feature pass
is substantial, E17/E18 units were admitted early under the same GPU lock rather than waiting idle for all its heads.
The coordinator checks unit existence to prevent duplicate trainers. Its memory limit was raised to 16 GiB solely for
the existing full-matrix CPU feature standardization; individual training units remain at 12 GiB. E19 remains held.

**Live status, 19:47 Sydney:** E18 fcanvas s7 is actively training, latest logged update 500/36000 (objective 0.137890,
peak torch allocation 2.128 GB). M6 H16 trajectory evaluation is still generating features; there is no completed M6
trajectory report. E17's unit is active **waiting for GPU admission**, not computing a new M16 update; its older log's
1000 updates belong to the interrupted run. Completed A16 checkpoints remain 2/4 of the E17 arms. Timestamped status is
maintained every 30 seconds in `20261004_resume_audit/live_status.json`, with append-only `status_events.jsonl`; this
monitor records runtime/output presence, does not declare scientific passes, and launches no experiments. Full audit:
`20261004_resume_audit/FINDINGS.md`. E19 and E16 seed-8 training remain held.

## 2026-10-04 — Mamba diagnosis: health drops are a position artifact in both backbones; memory incentive measured; E16 correction

**Correction (retracts my 10-04 midday claim "E16's 0/305 health reading is an evaluation artifact").**
- check_e16's first s7 run set Delta on the last window frame only; training conditions every frame. Fixed (f6823df0).
- The fix changes nothing: a CPU rerun on 32 roots gives posterior −2 accuracy 0/10. On 12 true hits (64 roots), a
  training-style call (posterior over the full window) leaves the health token at copy error (e.g. 69.5 → 69.5).
- My pool diagnostic ("generate weight 0.678 on hits, error 258.9 → 13.1") mixed 1,024 end-aligned death windows with
  1,024 ordinary ones. Split by window type (stage-A s7 decoder, posterior Delta on every frame):

| windows | dh | n | last-transition share | generate weight | token-63 error copy → world |
|---|---|---|---|---|---|
| held main | −2 | 106 | 0.29 | 0.090 | 75.4 → 66.1 |
| held main | ±1 | 32 / 171 | | 0.05-0.08 | 66.8 → 57.9 / 56.0 → 54.9 |
| terminal | −2 | 384 | 0.82 | 0.507 | 245.0 → 16.5 |
| terminal | ≤ −3 | 447 | 0.98 | 0.890 | 291.7 → 3.8 |

- Where ordinary-hit information is lost (`dbg`-style trace on held / 6,144 training main windows, dh = −2 vs 0):
  - posterior encoder output before VQ (region 2, which holds token 63): probe AUC 0.634 / 0.701, though it sees the true
    next frame;
  - quantized code: AUC 0.548 / 0.642; I(code; hit) ≤ 0.028 of 0.085 bits per region; hit codes also cover 50-79% of non-hits;
  - decoder h at token 63: AUC 0.772 / 0.808 (0.653 / 0.758 with Delta zeroed); output copies (generate weight 0.09).
- Reading: e16_health_drawable FALSE stands. The posterior is trained through the decoder's uniform latent L1 plus commitment. Deaths (large
  token change, ~6,500 per pass over the pool) get encoded; ordinary hits (~2,200, error 75 each) do not. This is E14's
  rarity diagnosis again, now with the answer fed to the model.

**check_context (lane61; teacher-forced, true frames, window w = 1..5):** aggregate error gain misses the declared history-use threshold (not evidence of no history use; see recall below).
- Map error w1 → w5: attention −3.6%, Mamba −4.5% (w4 −6.0%). HUD error rises 15% at w5 in both.
- Drops are drawn only at w5: catch 3.2% / 6.1%, false 1.1% / 0.8%, fresh 0 / 0.
- uses_history FALSE (both), mamba_uses_more FALSE.

**Pool structure (raw, 6-frame windows):** 78% of the training health drops ≤ −2 sit at time position 4.
- Main windows by position: [276, 258, 300, 298, 339]. End-aligned terminal windows: [185, 195, 167, 158, 6172].
- Terminal −1: 1,897 of 2,071 at position 4. Recoveries at position 4: 363 vs ~640 elsewhere.

**check_position (lanes 64, 66; the same windows with the time table re-indexed, weights unchanged):**

| condition | attention drawn / catch / false | Mamba drawn / catch / false |
|---|---|---|
| 4 frames, rows 0-3 | 0 / 0 / 0 | 0 / 0 / 0 |
| same 4 frames, rows 1-4 | 1.34% / 3.2% / 1.29% | 0.01% / 0.4% / 0 |
| 5 frames, rows 0-4 (evaluation convention) | 1.18% / 3.2% / 1.14% | 0.93% / 6.1% / 0.82% |
| 1 frame, row 0 / row 4 | 0 / 0.86% (all false) | 0 / 0 |
| last frame × 5, noop actions (no history) | 1.37% / 3.6% / 1.32% | 1.25% / 4.7% / 1.18% |
| last 4 frames, first doubled | 1.19% / 2.9% / 1.15% | 0.94% / 6.1% / 0.82% |

- Readings:
  - attention: position_shortcut TRUE, row4_alone TRUE, scan_shortcut TRUE, fifth_frame_used FALSE;
  - Mamba: position_shortcut FALSE, row4_alone FALSE, scan_shortcut TRUE, fifth_frame_used FALSE.
- Exact mechanism: both worlds learned "health drops at the death position" from the end-aligned terminal windows.
  Attention reads it from time-table row 4; Mamba from its fifth scan step (a scan from a zero state counts its steps; it
  ignores the table). Both condition on the current frame only: with no history at all they draw as many drops.
- At positions 0-3 neither draws a single drop, so ordinary hits are not learned at any position. Every evaluation step after
  warm-up (≤ 5-frame windows) sits at the death position.
- This explains 10-03's "drops drawn LESS often with a hit than without one" (E14f).
- The H16 health gap is therefore a training-signal problem (rarity + position-locked deaths), not a context-length or backbone
  problem. Mamba's slightly better discrimination at w5 (catch / false 7.4 vs 2.8) is real for this seed, but it is not
  history use (no-history catch 4.7%).

**check_memory (lane65, data only; 1,500 64-frame windows, 93,942 transitions; scroll share 35.1%):** how much does perfect
memory of terrain re-entering the view pay?

| lookback L | entering cells seen within L | sighting error (memoryless neighbour 57.0) | share of copy-world error removed | per transition |
|---|---|---|---|---|
| 2 | 3.6% | 8.9 | 0.7% | 5.9 |
| 5 | 11.2% | 11.9 | 2.1% | 17.7 |
| 8 | 16.0% | 14.7 | 2.9% | 24.5 |
| 15 | 22.2% | 21.7 | 3.9% | 32.3 |
| 31 | 28.7% | 35.3 | 4.7% | 39.3 |
| 63 | 31.4% | 43.6 | 5.0% | 41.8 |

- Readings: memory_grows_with_L FALSE (22.2% vs the 22.4% threshold), memory_incentive_16 TRUE (3.9% ≥ 2%).
- Long memory has a real but small map payoff, about +1.8 points of copy error from L = 5 to 15. The other dependency, the hit
  cooldown, needs 6 frames (ceiling 62.1% at 5, 85.7% at 6), but hits are not learned at all (above).

**E17 stage 2 amendment 1 (2026-10-04, before any stage-2 run; the lane62 resource smoke still runs):**
- Launch is held until check_recall (lane67) reports.
- Reason 1: long_hits and long_fresh are expected FALSE for a reason unrelated to memory. Hits are not learned inside 5
  frames, where the information is present, and the rawlong pool end-aligns deaths the same way (they would sit at position 15).
- Reason 2: the map content memory can supply is terrain re-entering after a scroll. fmamba scans view slots, not world
  cells, so its state is misaligned with what re-enters. check_recall measures this per backbone (36k full vs fmamba; 6k
  full / fattn / fmamba / fcanvas / fscan).
- If fmamba cannot recall across scrolls, the fair Mamba arm for a memory test is world-aligned (fcanvas), not fmamba.

**check_recall (lanes 67-69): Mamba learns cross-scroll recall about twice as fast as attention, replicated at both seeds.**
Held-out 6-frame pool windows, teacher-forced. Entering cells: 2,018 recallable (seen earlier in the window), 25,967 unseen.
Recall capture = (neighbour − world) / (neighbour − sighting).

| capture | 6k | 12k | 18k | 24k | 30k | 36k | 50k | 100k |
|---|---|---|---|---|---|---|---|---|
| s7 attention (full) | 0.171 | 0.184 | 0.239 | 0.303 | 0.319 | 0.472 | 0.739 | 0.825 |
| s7 Mamba (fmamba) | 0.191 | 0.269 | 0.331 | 0.450 | 0.685 | 0.778 | | |
| s8 attention | 0.168 | 0.184 | 0.201 | 0.301 | 0.300 | 0.283 | 0.304 | 0.304 |
| s8 Mamba | 0.218 | 0.273 | 0.321 | 0.467 | 0.612 | (training) | | |

- Unseen entering cells tie at every snapshot (36k: 41.9 vs 42.7). The 6k suffix arms (full / fattn / fmamba / fcanvas /
  fscan) are all 0.18-0.21.
- Readings:
  - recall_gap_36k FALSE and canvas_recalls_6k FALSE: my misalignment hypothesis was wrong in direction;
  - mamba_recall_edge_s8 TRUE (+0.312 at 30k); mamba_recall_edge_both TRUE (every snapshot from 18k, both seeds);
  - attention_catches_up TRUE at s7 (0.739 at 50k, 0.825 at 100k), FALSE at s8 (0.30 from 24k to 100k).
- Why it never showed in aggregate metrics: the recallable-cell difference (36k s7: 38.3 vs 22.1 on 2,018 cells over 10,240
  transitions) is ~3.2 per transition, about 1.6% of the map error. That matches E17 stage 1's only resolved difference,
  moved −0.004 (scrolls are where recall happens).
- Exact mechanism (v2 split; slot_bias TRUE at both seeds):

| capture | same view slot (1,615) | moved slot (403) | age 2 (957) | age 3-5 (1,061) |
|---|---|---|---|---|
| s7 36k attention / Mamba | 0.508 / 0.864 | 0.274 / 0.299 | 0.570 / 0.922 | 0.324 / 0.558 |
| s8 30k attention / Mamba | 0.300 / 0.671 | 0.295 / 0.279 | 0.299 / 0.715 | 0.300 / 0.454 |
| s7 attention 100k | 0.913 | 0.330 | 0.935 | 0.656 |

  - A cell leaving through an edge and returning through it with no perpendicular move re-enters its old view slot. There, a
    per-slot recurrence is aligned: "what was in this slot k steps ago" is its default pathway. Attention must learn the
    same lookup and learns it slowly (s8: not by 100k).
  - Cells re-entering another slot are recalled by NO backbone (0.27-0.33, attention 100k included).

**check_memory v2 (moved-slot share of recallable cells, long pool):** L2 0%, L3 10.5%, L5 23.5%, L8 33.1%, L15 42.4%,
L31 49.5%, L63 52.2%. Share of the perfect-memory gain at L15: 39.1%. moved_dominates_16 TRUE.
- At L = 16, per-slot Mamba can carry at most ~61% of what memory supplies (same-slot re-entries, ages 6-15). The rest needs
  a world-aligned state: fcanvas, or the 2026-09-29 carry transport. That proposal was deferred "unless the fmamba integration
  localizes a cross-scroll memory bottleneck". It is now localized.
- Literature read for this:
  - DRAMA's replay (`third_party/Drama/replay_buffer.py`) and DreamerV3's (`embodied/core/replay.py`) both sample windows
    uniformly over a continuous stream, so terminations fall at every position.
  - EMERALD (arXiv 2507.04075) uses relative positions, T = 64, and cached keys / values across batches.
  - Po et al. (arXiv 2505.20171) scan spatial blocks over time: the state follows screen positions, as in fmamba. Their
    full-context transformer still edges their SSM on Memory Maze retrieval (SSIM 0.914 vs 0.898).

- How this sits with the recall literature: Zoology (Arora et al., arXiv 2312.04927) finds attention far ahead of
  gated-convolution / SSM models on associative recall (a 70M attention model beats a 1.4B gated convolution on MQAR).
  - Our edge is positional, not associative: same view slot at a fixed lag, which a per-slot recurrence holds by construction.
  - The moved-slot case is a position lookup through the camera offset, closer to associative recall. No backbone has
    learned it yet.

**E17 resource smoke (lane62):** attention at L = 16, 40 windows per update: 0.468 s / update, peak 4.20 GB (500 updates,
objective 0.081). Mamba: below.

**E17 stage 2 amendment 2 (declared in check_recall's docstring, 3ac7e494, before any stage-2 run):** memory readings on the DEV
futures' true 20-frame trajectories (check_recall --futures, window 15 vs 5):
- long_recall_same: M16's same-slot capture at ages 6-15 ≥ 0.5 and ≥ A16's + 0.10, at both seeds;
- long_recall_used: M16's same_6_15 capture at window 15 minus at window 5 ≥ 0.2, at both seeds;
- moved_unsolved (descriptive): moved_6_15 capture ≤ 0.35 for every world.

Same tool, now (lanes 70-71): the parents on DEV futures, teacher-forced (futures_replicates) and self-fed on aligned steps
(imagined_recall_edge). These test whether the pool result holds out of the training ledger and carries into imagination.

**E18 PREDECLARED (2026-10-04; lane72, starts after the lane62 smoke): world-aligned Mamba.**
- Arm: fcanvas (tworld docstring: fmamba's Mamba-2 time scan over world-aligned canvas cells placed by scroll.estimate's
  offsets) on the exact A6 / M6 recipe: corrt, teacher, 6-frame windows, 36k, seeds 7 and 8, snapshots every 6k.
- Why: 42% of L15 recall (39% of its gain) re-enters another view slot, which no backbone recalls (0.27-0.33).
  - The 6k fcanvas arm (lane 9) was judged before recall is learned at all: every backbone is 0.18-0.21 at 6k, and recall
    emerges after 18k.
- Sources:
  - allocentric memory registered by correlation: MapNet (Henriques & Vedaldi, CVPR 2018);
  - memory shifted by ego-motion: Neural Map (Parisotto & Salakhutdinov, ICLR 2018), FIERY (Hu et al. 2021), SRU (Yang et
    al., arXiv 2506.05997);
  - ours on 09-29: explicit carry transport ≈ fcanvas on short windows (relative RMSE 1e-7 except re-entry) at 4.9× the cost.
- Readings (check_recall pool split vs the same-seed fmamba 36k; two-seed rule):
  - c6_moved_recall: moved-slot capture ≥ fmamba's + 0.15;
  - c6_same_recall: same-slot capture ≥ fmamba's − 0.05;
  - c6_unseen: unseen entering-cell error within 5% of fmamba's.

**E19 DESIGNED (not launched; the GPU is booked by E17 / E18 until ~2026-10-05 morning): make hits learnable, then ask memory.**
- Causal chain it acts on (all measured today):
  - deaths are end-aligned, so 78% of training drops sit at position 4 and both backbones learn "drops at the death position";
  - ordinary hits (1.4% of transitions at positions 0-3, 1 token in 81) are never drawn at any position;
  - E16's posterior, which sees the answer, does not encode them either.
- Two components, each from a source:
  - (a) Death de-alignment. IRIS (`eloialonso/iris` src/dataset.py: start uniform in the episode, right padding with a
    loss mask) and DreamerV3 / DRAMA (uniform windows over a continuous stream) put terminations at every position. Here: the
    death transition at a uniform row r ∈ {0..4}, frames after it padded and masked out of the loss (causal, so padding
    cannot leak).
  - (b) Selective dose on the health token (63) over a matching-negatives context: E14's CGSReg form (`--weight maskL`; a
    selective dose needs the context where the change could happen), with the decision-weighting rationale of VaGraM
    (Voelcker et al., ICLR 2022, arXiv 2204.01464). Context = a zombie in a cell beside the player in the TRUE next frame.
    - It contains every zombie hit (an attacking zombie stays put; check_zombie_cue) and the cooldown negatives.
    - Labelled by teval's Probes.zombie on pool tokens. Validated today on 4,811 TEST frames: AUC 1.000; at threshold 0.3
      precision 1.000, recall 0.996; 10.0% of frames flagged.
- Arms and readings (to be fixed in the launch commit before any run):
  - attention and Mamba, continued from the 36k parents, seeds 7 and 8;
  - learnable: teacher hit catch ≥ 0.3 at window 5 (rule ceiling 0.62) and fresh ≥ 0.5, with false drops ≤ 0.02;
  - unshortcut: check_position w4_at0's drawn rate ≥ half of w5_at0's;
  - no_cost: onestep_all ≤ +0.005 and consequences unchanged;
  - then the memory question on hits: beside_no_hit catch (Bayes 0.48, needs the cooldown), Mamba vs attention, w5 vs w16.
- Implementation ready, CPU-tested, NOT yet applied: a patched tworld.py in the session scratchpad (`--dealign`, `--weight hpL`,
  hp_labels(); parser guard).
  - Default recipe identical: equal loss, gradient max difference 0.0. The transform places the death transition at row r
    with correct actions and keep mask.
  - It will be applied after fcanvas s8 has started, so no E17 / E18 checkpoint carries a changed script hash.
- Labels built: `artifacts/eda/hpctx_labels_v1.pt` (zombie beside in the true next frame, per pool transition). On the pool:

| pool transitions | n | zombie beside |
|---|---|---|
| main, dh = −2 | 1,346 | 0.803 |
| main, dh = −1 | 417 | 0.197 |
| main, dh = 0 | 119,051 | 0.062 |
| main, dh ≥ +1 | 1,941 | 0.092 |
| terminal, last transition, dh ≤ −2 | 6,172 | 0.955 |

  - P(hit | beside) 0.1225 vs 0.0023 not beside (53×). The dose covers 0.24% of tokens, so hp1 ≈ ×417 per token (E14's mask1
    was ×304).
  - The ~20% of −2 hits without a zombie beside are probably skeleton arrows (2 damage, at range). Not yet checked; zombies first.

**Queue order (after the E17 resource smoke):** E17 stage 2 (A16 / M16), E18 (fcanvas) alongside as memory allows, then E16 s8
(lane59, relaunched by hand; lane63 stopped 13:20 so it does not jump the queue).
- Mamba smoke (lane62): 1.436 s / update, peak 2.87 GB at 40 windows. The predeclared 40 windows per update stand for both arms.
- lane72 started fcanvas s7 at 13:43, before E17 had a lane. Stopped at 13:44, before any state was saved.
- lane73: E17 stage 2. A16 s7, s8 first (4.2 GB, cannot share the GPU), then M16 s7, s8, then the predeclared evaluations:
  - check_recall --futures at windows 15 and 5, and --imagined;
  - check_damage at windows 16 and 5;
  - check_h16_traj --window 16;
  - teval at windows 16 and 5;
  - compare: parent vs long, A16 vs M16, and w5 vs w16 per world.
- lane74: E18, unchanged, starting after A16 s8 so it runs beside the Mamba runs.
- DEV-futures replication (lane70, teacher-forced, window 5): futures_replicates TRUE (computed from the ordered lines; the
  name bug is fixed in 472d20f4).

| capture | same 2-5 | moved 2-5 | same 6-15 | moved 6-15 (outside the window = no-memory baseline) |
|---|---|---|---|---|
| attention s7 36k | 0.430 | 0.447 | 0.349 | 0.422 |
| Mamba s7 36k | 0.716 | 0.371 | 0.366 | 0.376 |
| attention s8 30k | 0.242 | 0.361 | 0.336 | 0.368 |
| Mamba s8 30k | 0.561 | 0.364 | 0.353 | 0.406 |
| attention s7 100k | 0.857 | 0.349 | 0.339 | 0.391 |

  - Capture is not zero without memory: cells beyond the window score 0.34-0.42 from generation alone.
  - Against that baseline, Mamba recalls same-slot cells at both seeds. Attention does so weakly at 36k / 30k and strongly
    by 100k (s7). Moved-slot cells stay at the baseline for every world.
- In imagination (lane71; self-fed, sample 0's actions, aligned steps only, 1,002 DEV roots), imagined_recall_edge TRUE:

| imagined capture | same 2-5 | moved 2-5 | same 6-15 | moved 6-15 |
|---|---|---|---|---|
| attention / Mamba s7 36k | 0.171 / 0.353 | 0.280 / 0.195 | −0.138 / −0.060 | −0.014 / 0.033 |
| attention / Mamba s8 30k | 0.020 / 0.231 | 0.174 / 0.227 | −0.246 / −0.075 | −0.047 / −0.066 |
| attention / Mamba s8 36k | 0.026 / 0.346 | 0.177 / 0.193 | −0.186 / −0.005 | −0.063 / −0.021 |

  - Unseen entering cells tie: imagined error 69.6 / 70.9 (s7), 72.3 / 72.6 (s8 36k), against ~37 teacher-forced.
  - Caveat: the no-memory baselines differ in imagination too (same_6_15). The difference against that baseline still
    favours Mamba (+0.10 / +0.04 / +0.14), but less. The rigorous contrast is v5 (lane75): the same cells in the same world,
    with the sighting in the window (w5) and without it (w1).
- **Memory contrast (lane75; DEV futures, teacher-forced; computed from the ordered w5 / w1 lines): memory_contrast TRUE.**

| same-slot, ages 2-5 | w5 | w1 (no history) | memory gain |
|---|---|---|---|
| attention s7 36k | 0.430 | 0.294 | +0.136 |
| Mamba s7 36k | 0.716 | 0.258 | **+0.458** |
| attention s8 30k | 0.242 | 0.252 | −0.010 |
| Mamba s8 30k | 0.561 | 0.245 | **+0.315** |
| attention s7 100k | 0.857 | 0.267 | +0.589 |

  - Every world has the same no-history baseline (0.25-0.29), so the gain is memory use alone.
  - Moved-slot gains are ≈ 0 for every world (−0.03 to +0.05).
  - At matched budget, Mamba uses its window for same-slot recall 3.4× more than attention (s7). Attention s8 uses none. Attention
    needs ~3× the updates (s7 100k) to match.

**E17 stage 1, both seeds (M6 s8 finished 13:20; lane52's evaluations):**
- m6_onestep TRUE: onestep_all M6 − A6 is s7 −0.002 [−0.003, −0.001], s8 −0.005 [−0.006, −0.004]. Driven by moved (s7
  −0.004, s8 −0.011, both resolved): the recall mechanism. s8 blocked is +0.012 [+0.006, +0.021] (Mamba worse).
- m6_depth16 FALSE: s7 −0.011 [−0.022, +0.001] (ns); s8 −0.018 [−0.032, −0.003] (resolved, with gen_1 / 4 / 8 all resolved lower).
- m6_hits FALSE: teacher caught s7 0.050 vs 0.030, s8 0.036 vs 0.050. Expected: drops are the death-position artifact in both.
- m6_consequences: held strict caught matches (s7 0.562 vs 0.564, s8 0.559 vs 0.564); hallucinated 0.007-0.015.
- m6_h16_traj: lane77. Lane76 OOMed twice: the Mamba world allocates +1.36 GB over 2.06 GB at check_h16_traj's batch 16 × 17
  branches. Fixed by teval's per-token-SSM batch rule in check_h16_traj (4), check_damage (16) and check_recall --futures
  (16), 3a9681ca. Per-sequence math unchanged. This also covers lane73's E17 stage-2 evaluations.

---

## 2026-10-03 night — interim: Mamba at an equal budget (s7); matched-budget comparators; E16 running

**E17 stage 1, seed 7** (corrt teacher, 6-frame windows, 36k, fmamba backbone vs the attention world; lane60). Mamba matches
attention:
- consequences: held 0.564 vs 0.562; DO 1.000 vs 0.997; every placement type 0.000 in both; hallucinated 0.008 vs 0.007;
- one-step all −0.002 [−0.003, −0.001] (moved −0.004, other classes unchanged); depth 16 −0.011 [−0.022, +0.001] (ns);
- health frozen like attention: hits caught 0.050 (attention 0.030), fresh arrivals 0 / 31, beside drawn with / without a hit
  0.082 / 0.156, ±1 never.
- Seed 8 (lane52) is running. m6_depth16 cannot pass (s7 not resolved).
- Expected from the window argument: with the same 6-frame windows both backbones see the same history.

**Matched-budget deterministic comparators** (the 100k run's 54k snapshots; lane55):
- trajectory H16 0.653 / 0.651 (36k: 0.645 / 0.646); trajectory − snapshot +0.033 / +0.031 (resolved);
- continuation head Brier 0.090 / 0.100, AUC vs realized death by 16 0.863 / 0.844.
- Under amendment 2 (E16 from scratch, 36k), E16's matched comparator is the 36k world; the 54k values are reference only.
- 18k more deterministic updates buy +0.005-0.007 of H16 decision value.

**E16 amendment-2 smoke** (from scratch, 3k): Delta gain +70%, 110 codes. Delta use on that early decoder: it removes 91% of the
rest-of-map error, 45% player, 22% HUD, 15% entering cells. Early on the channel carries deterministic map content; the full
runs (lane59) log the Delta gain every 500 updates.

---

## 2026-10-03 evening — E14f: 100k learns the placements, never health; positions plateau

The 50k worlds continued to 100k from their full states (lane51; readings declared in lane51.sh).

**Held strict consequences caught, per action:**

| | DO | stone | furnace | table | all | hallucinated |
|---|---|---|---|---|---|---|
| s7 50k → 54k → 100k | 1.00 → 1.00 → 1.00 | 0.99 → 0.99 → 1.00 | 0.75 → 0.92 → 0.98 | 0.11 → **0.86** → 0.94 | 0.871 → 0.975 → 0.991 | 0.006-0.012 |
| s8 50k → 54k → 100k | 0.99 → 0.98 → 0.99 | 0.94 → 0.98 → 0.99 | 0.19 → **0.79** → 1.00 | 0.000 throughout | 0.795 → 0.845 → 0.877 | 0.005-0.010 |

- The order is identical at both seeds: DO, stone, furnace, table.
- s7's table jump (50k → 54k) happened in a true continuation (full state, no optimizer reset). A warm restart is not needed for
  a stage-like jump.
- s8 has not learned table by 100k. With one run per seed and the 6-12k run-to-run spread from GPU nondeterminism, seed and run
  cannot be separated.

**Health never learned** (check_damage, every 6k snapshot 54k-100k, both seeds):
- hits caught 0.006-0.134;
- fresh arrivals 0 / 31 at every s7 snapshot, at most 1 / 31 at s8;
- drops drawn LESS often with a hit than without one, in the beside case, at every snapshot;
- recoveries and starvation drops 0.000.

**Positions plateau** (subst16, ever position-wrong):
- s7: 0.449 / 0.298 / 0.283 / 0.269 (18k / 36k / 50k / 100k);
- s8: 0.493 / 0.339 / 0.284 / 0.270;
- the scroll decision is drawn right 0.94 from 50k on.

**compare, 100k − 50k:** one-step −0.003 at both seeds (resolved; interact −0.063 / −0.054); depth 16 −0.012 [−0.024, −0.000]
(s7) and +0.009 [−0.001, +0.020] (s8).

**Readings:** table_learned FALSE (s7 only); hit_mode FALSE; fresh_hits FALSE; position_plateau TRUE at both seeds;
budget_continues FALSE.

So budget keeps buying the deterministic consequence modes one by one, and nothing else: no health change at any budget to
100k, and positions and depth-16 error flat after 50k. That is consistent with the E16 rationale. A hit is a coin flip from the
window where it is most common, and copied where it is predictable; a deterministic L1 world does not acquire it with budget.

---

## 2026-10-03 — E16 DESIGN (predeclared before any code): a Delta-IRIS stochastic channel on the per-tile world

**Why (measured):**
- At every depth, the decision value a real-fitted head misses in imagination is the drawn outcome (HUD), not the map
  (check_transfer_subst, check_h16_subst).
- Health is frozen in every deterministic world: hits caught 3-5%, recoveries and starvation 0%.
- Even visibly predictable hits are copied (fresh arrivals 0-1 / 31). Where the 4-frame input gives a coin flip (P 0.48), the L1
  median erases the hit.
- One faithful future per action carries 55% of the H16 margin, 16 samples ~90%. The best reading of our deterministic futures
  carries 38% (check_h16_value, check_h16_traj).

**Source:** Delta-IRIS (Micheli et al., ICML 2024; arXiv 2406.19320 sections 2.2-2.4; code vmicheli/delta-iris f8d4173, read
2026-10-03). On Crafter, its random-Delta-token ablation keeps layout / movement / items / crafting, but mobs and HEALTH
INDICATORS degrade: health is carried by the stochastic channel.

**Architecture** (Delta-IRIS values unless marked DEVIATION):

| component | design |
|---|---|
| posterior encoder E | a CNN over (frame t tokens, action, frame t+1 tokens) on the 9 x 9 token grid (Delta-IRIS: a CNN over (x1, action plane, x2) on pixels). K = 4 Delta-tokens, one per 2 x 2 spatial region (the grid zero-padded to 10 x 10 so regions are equal, 5 x 5: a mechanical change). Vector quantization: codebook 1024 x 64, cosine similarity, EMA updates (0.99), commitment 0.02, revival only on collapse (Crafter config). DEVIATION: inputs are JEPA tokens, not pixels. |
| decoder D | our corrt TWorld, continued from the 36k teacher world (seeds 7 and 8). Each Delta-token's post-quantized vector covers its region's tiles and is added to frame t's input tile embeddings through a zero-initialized projection (Delta-IRIS concatenates it with the latent feature map at aligned positions). At initialization D is exactly the deterministic world. DEVIATION: additive conditioning on tokens instead of channel concatenation on a CNN feature map. |
| dynamics prior G | a 3-layer causal transformer, width 512, 8 heads, blocks of [I-token, action token, 4 Delta-tokens] per step, 21-step sequences. It predicts the 4 Delta-tokens autoregressively (CE) and episode end (CE, weight 1). I-token = per-tile linear 192 → 8, flattened (648) → 512 → LayerNorm (Delta-IRIS: an 8-channel frame CNN flattened to 512). G is trained on the 64-frame Raw TRAIN ledger with Delta codes from frozen E: 21 steps, 26.4% death-ending windows. Its context covers the 6-step cooldown (the 6-frame ceiling is 0.857). Reward head omitted at first: DEVIATION, noted. |
| training | offline and sequential: stage A trains E + D on the 6-frame pool (teacher L1 + commitment); stage B trains G on frozen codes. Delta-IRIS alternates only because its data grows online. |
| imagination | per step, sample 4 Delta-tokens and the end flag from G (temperature 1), then decode with D (5-frame window). M samples per action for H16. |

**Readings, declared now** (two seeds; comparators = the same-seed deterministic 36k world and check_damage_rule's Bayes rates):
- e16_health_drawable: with POSTERIOR Delta (teacher-forced), drawn health-change accuracy >= 0.8 for −2, −1 and +1 at both
  seeds (the decoder can draw the outcome when told).
- e16_hits_calibrated: self-fed sampled rollouts (diagnosis futures, sample-0 actions) draw hits at each visible-history case's
  Bayes rate within ±0.15 (fresh 0.97, already beside 0.48, recent hit 0.03) at both seeds.
- e16_no_false_hits: sampled hit frequency with no zombie beside the post-move player <= 0.01 (Bayes 0.0017) at both seeds.
- e16_h16: H16 expected safe on DEV-B (check_h16_traj protocol) with M = 1 sampled future per action >= the deterministic
  trajectory value + 0.02 (0.645 / 0.646) at both seeds; M = 4 and 16 reported against one_real_future 0.679 and oracle 0.766.
- e16_map_cost: one-step map-token error with prior samples vs the deterministic world, reported (a cost bound of +10%).

AMENDMENT (2026-10-03, before any E16 run): stage A continues the 36k world for 18k updates, so the E16 decoder has 54k. Every
E16 reading is ALSO reported against the matched-budget deterministic world, the 100k continuation's 54k snapshot
(corrt_raw_teacher_s{7,8}_u100000_at54000; check_h16_traj and check_damage run on it). e16_h16 passes only if it holds against
both the declared 36k value and the 54k snapshot's.

AMENDMENT 2 (2026-10-03, after stage-A smokes; no E16 reading had been measured):
- What the smokes showed:
  - crafter revival: codebook collapse, 3-4 codes, 1.7 bits;
  - atari revival: 76 codes at 500 updates, re-collapse to 8 after revival stops at 400;
  - always revival: diversity held (8-11 bits, 6-126 codes per batch).
- A 6k diagnostic (always revival, the 36k world continued) logged the same-batch teacher loss with Delta zeroed. At EVERY
  checkpoint the loss with the posterior's Delta was 0.4-1.5% HIGHER than with Delta zeroed: the decoder takes no information
  from the channel.
- On 2,560 held transitions, each region's code explains 0.8-1.2% of the deterministic decoder's residual variance. Chance for
  the 13-25 codes in use is about 0.5-0.9%. The encoder does not encode what the decoder misses.
- Cause, matching Bowman et al. 2016 (arXiv 1511.06349, sec. 3.1, read): a model "initially learn[s] to ignore z ... Once this
  has happened, the decoder ignores the encoder and little to no gradient signal passes between the two, yielding an undesirable
  stable equilibrium". Continuing a converged deterministic world STARTS stage A in that equilibrium.
- Delta-IRIS avoids it by its actual recipe: tokenizer (E + D) trained jointly from scratch.
- Amended stage A: the corrt decoder from scratch (same seed and init as the deterministic arms), Delta-IRIS's Crafter revival
  setting, 36k updates. This also removes the budget confound: the comparator is the from-scratch 36k deterministic world
  (corrt_raw_teacher_s{7,8}_u36000), at equal updates.
- A from-scratch smoke (3k updates) must show a positive Delta gain (teacher loss with Delta zeroed minus with Delta) before the
  full runs. The readings and their thresholds are unchanged.

Implementation order: E / quantizer / D-conditioning with CPU unit tests (zero-init identity with the deterministic world;
codebook usage); stage A lane; stage B lane; sampling rollouts in check_damage / check_h16_traj (`--sample M`).

---

## 2026-10-03 afternoon — what decisions miss is the outcome; reading trajectories helps but not enough; terrain bound holds

**H16 transfer gap = the HUD** (`check_h16_subst`, deepeval DEV split, real16 heads; h16_hud_bottleneck TRUE):

| world | H16 imagined | + real HUD only (share of gap) | + real map only |
|---|---|---|---|
| s7 18k | 0.597 | 0.688 (0.96) | 0.591 |
| s7 36k | 0.600 | 0.687 (0.95) | 0.589 |
| s8 18k | 0.585 | 0.685 (0.94) | 0.579 |
| s8 36k | 0.600 | 0.688 (0.97) | 0.594 |

- The real reference is 0.691.
- HUD share by depth: H1 0.998-0.999, H4 0.91-0.95, H16 0.94-0.97. The real map never helps at H16.
- Imagined health at depth 16 is 5.9-6.0 against 2.4 in reality.

**Reading the deterministic trajectory** (`check_h16_traj`; per-step hazard head, a Dreamer-style continue head, on the 36k
worlds' imagined trajectories; DEV-B, 1,139 roots):
- trajectory 0.645 / 0.646 vs depth-16 snapshot 0.629 / 0.629: +0.017 [+0.007, +0.026] (s7), +0.017 [+0.005, +0.031] (s8);
  traj_gain TRUE.
- traj_reaches_one_future FALSE: one real future is 0.679, oracle 0.766.
- Share of the uniform-to-oracle margin: prior 23%, snapshot 30%, trajectory 38%, one real future 55%, 16 sampled futures ~90%
  (check_h16_value).
- So a better head recovers part of the value. Futures that never draw damage cap the decision below one faithful sample: the
  measured case for sampled (stochastic) imagination.

**Health is frozen in every attention world, not only hits** (risk suite, `check_damage` on 36k / 50k):
- Accuracy of the drawn health change: no change 0.989-0.991; −2 or worse 0.029-0.049; **−1 (starvation) 0.000**; **+1
  (recovery) 0.000**. n = 15,387 / 305 / 32 / 112.
- Recovery and starvation follow hidden counters (player_recover, read in game_logic), like the zombie cooldown.
- E5f's 6k finding "HUD never updated" still holds for health at 36k and 50k.

**Terrain bound** (`check_enterbound`):
- v1 was INVALID as a ceiling test: frame-wide class one-hots gave 0.645, below copying the adjacent tile (0.714).
- v2, an aligned 28-cell token patch (a superset of the 3-edge input, alignment verified exactly): 0.767 vs 3-edge MLP 0.756 and
  world 0.760; terrain_bound_holds TRUE.
- Entering trees, coal, iron and tables are never predicted (recall 0.000).
- About 23% of entering terrain is not inferable from the visible frame by this predictor. It is a bound from one predictor
  class (history excluded), not a proof.

---

## 2026-10-03 midday — the queue: every untested or assumed item, and how it gets tested

The user asked for: longer training, Mamba at an equal budget, and decision-linked metrics instead of generic map error.

**Risk suite** (the user's list; committed in check_damage / check_h16_traj / check_rootaware; run on every new world):

| metric | where |
|---|---|
| zombie-hit catch | check_damage, teacher-forced and self-fed |
| damage false positives | check_damage |
| health-change accuracy, by true change (<= −2, −1, 0, >= +1) | check_damage |
| history-conditioned hit prediction | check_damage, catch per visible-history case beside that case's Bayes rate (fresh 0.97 / already beside 0.48 / recent hit 0.03) |
| continuation / death prediction | check_h16_traj: per-step hazard head on imagined trajectories (Brier over k, AUC vs realized death by 16) |
| H16 trajectory decision value | check_h16_traj |
| H1 root-aware decision value | check_rootaware |

- Every check takes `--window`, so long-context worlds are judged with the context they were trained on.
- Map error stays as a secondary metric (it drives position failures).
- Reference for hit catch: a deterministic world can draw at most 47.3% of hits from a 4-frame input, 62.1% from 5, 85.7% from
  6 (check_damage_rule).

**Running:**
- E14f (lane51): the 50k worlds continued to 100k from their full states. Readings: table_learned, hit_mode, fresh_hits,
  position_plateau, budget_continues.
- E17 stage 1 (lane52): Mamba-2 factorized backbone (fmamba), corrt teacher, the attention worlds' exact recipe at 36k, seeds 7
  and 8 (0.72 s/update measured). Readings: m6_depth16, m6_onestep, m6_hits, m6_h16_traj, m6_consequences.
- check_h16_traj (lane50): does a hazard head on the deterministic trajectory carry the H16 decision?

**E17 stage 2, memory (designed; finalized after a GPU resource smoke):** does context covering the 6-step cooldown let a
deterministic world draw hits, attention vs Mamba? Design constraints, each from a measurement or a read source:
- The dependency must lie inside the training window:
  - R2I (arXiv 2403.04253, app. O): models trained on 64-step sequences "fail to perform" when the dependency exceeds 64;
    performance rises with batch length 64 → 1024.
  - Our E1: a recurrence is usable only within (up to ~4x) its training length.
  - DRAMA, Dreamer 4 and le-wm all deploy inside their trained range.
  - Our 6-frame windows give at most 4 transitions of history; the cooldown needs 6.
- Batch diversity confounds length (E1: L64 at 6 windows / update 1.2-1.8x copy, at 24 windows 0.65-0.87x). Windows per update
  are matched, not only transitions.
- Continuing a short-window world beats a long-window world from scratch for imagination (E1f: 0.400 vs 0.406 windowed, memory
  +18-20%).
  - Learned absolute time positions are extended by copying (Longformer, arXiv 2004.05150, sec. 5: copy init 1.957 BPC vs
    random 10.299 before training; 1.705 after 65K updates).
  - tworld `--init` implements it. CPU check: the first 6 frames reproduce the 6-frame world to 7e-7; late frames L1 0.390 vs
    0.043 before any long training.
- Models stay on nearby frames unless the loss gives a reason (Po et al., arXiv 2505.20171, sec. 4.2: "trapped in local minima,
  failing to capture long-term dependencies"). Hits are 2% of transitions, so a predeclared reading tests whether a longer
  window alone changes the catch.
- State Passing (Buitrago Ruiz & Gu, arXiv 2507.02782) did not transfer at our budget (E6c); long-context training was needed.
- 2026-10-04 additions (read in the text):
  - S4WM (Deng et al., NeurIPS 2023, arXiv 2307.02064, sec. 2 and 5.4): "Transformers can be better at capturing local
    (short-range) information", and the transformer world model is best "where the context phase is short", S4WM when it is
    longer.
  - DRAMA (arXiv 2410.08893, Table 2): Mamba-2 15.6 ± 2.6 vs Transformer 24.7 ± 7.4 error on 8-frame sequences; at 64 frames the
    transformer runs out of memory. Its Atari100k score roughly matches STORM's.
  - Every per-tile Mamba comparison we ran gave both backbones the same <= 5-frame context: the regime where these sources
    report no SSM advantage.
  - Before any long-window run, check_context (lane61) measures how much of that short history each 36k world uses.
- E17 stage 2 PREDECLARED (2026-10-04; launched after a GPU resource smoke, which may only change windows-per-update for
  memory, recorded before the run):
  - Arms: A16 and M16, each continued at L = 16 from its own 36k teacher world (corrt_raw_teacher_s{7,8}_u36000 /
    ..._fmamba_u36000; tworld --init, time positions copy-tiled), on 16-frame windows of the 64-frame Raw TRAIN ledger (26.4%
    end-aligned death windows), 40 windows per update (the 6-frame recipe's window count), 6,000 updates, seeds 7 and 8.
  - Comparators:
    - the parent 36k worlds;
    - each continued world evaluated at a 6-frame and a 16-frame window: what the longer context itself buys;
    - for attention, a matched-update 6-frame continuation (the 50k runs' 42k snapshots, 36k + 6k).
  - Evaluation with the trained context: check_damage --window 16, check_h16_traj --window 16, teval --window 16, plus the
    6-frame-window evaluation of the same worlds (what the longer window itself buys).
  - Readings:
    - long_hits: teacher-forced hit catch with window 16 >= 0.3 at both seeds for either backbone (the 6-frame ceiling 0.857;
      current worlds 0.03-0.10);
    - long_fresh: fresh-arrival hits caught >= 0.5;
    - mamba_long_edge: (M16 - A16) on hits caught and on H16 trajectory value, both >= +0.02, at both seeds.
- Arms: A6 → A16 and M6 → M16, each continued at L = 16 from its own 36k world with matched windows per update. The `rawlong`
  path (64-frame Raw TRAIN ledger, 26.4% end-aligned death windows) is implemented and CPU-tested.

**Untested or assumed items, and their tests:**

| item | status | test |
|---|---|---|
| a stochastic channel draws hits at the right rate | untested | E16 (Delta-IRIS design), only if check_h16_traj says sampling is required |
| memory helps hits | untested | E17 stage 2 |
| ~24% of entering terrain is unknowable | assumed (local-MLP baseline only) | a stronger predictor (full view + history) on held-out entering cells |
| encoder geometry → false scrolls | correlational | needs a retrain; lowest priority |
| why furnace is learned before table | unexplained; count (85 vs 103) and token change (separation from grass 3.49 vs 4.29) both favour TABLE. Preconditions (game_logic.place_block, read 2026-10-03): table needs wood >= 2; furnace stone > 0 (no nearby-table requirement in Classic); stone stone > 0. Stone and furnace share one precondition, table needs a different count threshold: consistent with the order, untested | inventory-labelled placement attempts (the pool has none); table-only dose (Saxe: time ∝ 1/strength) |
| the stone jump at 42k-48k in both 50k runs | confounded with the warm restart at 36k | the 100k continuations have no restart; a continuous 0 → 50k control if needed |

---

## 2026-10-03 late morning — item 1 explained (zombie cue), hits copied, the 50k extensions

**Item 1, the 36k gen1 regression next to zombies, is explained** (`check_zombie_cue`, `check_rootaware`):
- Craftax: an attacking zombie stays put; one that does not attack moves, 75% of the time toward the player.
- Correlation with P(death1) across the 17 branches (zombie roots):

  | successor | "zombie beside the player" | "zombie stayed in place" |
  |---|---|---|
  | real | 0.066 | 0.615 |
  | s7 18k / 36k | 0.465 / 0.437 | 0.429 / 0.669 |
  | s8 18k / 36k | 0.481 / 0.398 | 0.526 / 0.680 |

- The 18k worlds' successors carried a shortcut reality lacks: the zombie drawn beside the player mainly in the deadly branches.
- The 36k worlds draw stay / chase like reality, which puts the death information in a ROOT-relative cue.
- dpanel's gen head reads each branch's successor alone.
- With the successor concatenated to the root (aligned by the world's own imagined view shift; 8 head seeds), mean safe next to
  zombies:

  | world | successor only | root-aware |
  |---|---|---|
  | s7 18k / 36k | 0.908 / 0.878 | 0.952 / 0.949 |
  | s8 18k / 36k | 0.908 / 0.873 | 0.957 / 0.949 |
  | real | | 0.999 |

- Root-aware 36k − 18k: −0.004 [−0.014, +0.006] (s7), −0.008 [−0.019, +0.002] (s8), against successor-only −0.030 / −0.035
  (resolved). rootaware_regression_gone TRUE; rootaware_gain_36k TRUE.
- Consequence: every successor-only decision panel so far UNDERSTATES what imagination carries (0.95 next to zombies, not
  0.87-0.91). Decision heads and critics must see history.

**Hits are copied, not only median-erased** (`check_damage`, both budgets and the 50k snapshots):
- damage_copied TRUE: teacher-forced, 1.8-5.0% of real hits drawn (18k / 36k), 3.3-10.1% at 42k-50k; self-fed 0-0.9%.
- damage_small FALSE: a hit is a large token change (HUD squared change 74.9 vs 2.1 for ordinary frames, 36×).
- Fresh arrivals (hit 97% predictable from the window) are drawn 0 / 0 / 0 / 1 of 31.
- In the beside-without-hit case, drops are drawn MORE often without a hit (0.08-0.27) than with one (0.02-0.19): the drawn drops
  do not track hits.
- So two failures stack: the rarity pathology of the DO consequences (copied even when predictable), and the L1 median where
  the window gives a coin flip.

**E14e, the 50k extensions** (lane45; warm restart at 36k; s8 resumed from its own 42k state after the power-off):
- Mode by mode, held strict consequences caught:

  | | DO | place stone | place furnace | place table |
  |---|---|---|---|---|
  | s7 42k / 48k / 50k | 1.00 / 1.00 / 1.00 | 0.000 / **0.992** / 0.992 | 0.000 / 0.106 / **0.745** | 0 / 0 / 0.106 |
  | s8 42k / 48k / 50k | 0.99 / 0.98 / 0.99 | 0.000 / **0.893** / 0.939 | 0 / 0 / 0.191 | 0 / 0 / 0 |

- placement_learned TRUE at both seeds. The order is identical at both seeds: DO, stone, furnace, table. Each type jumps from 0
  within one snapshot interval (Saxe et al.'s stage-like learning). Both stone jumps fall in the same 42k-48k interval; both runs
  began from a warm restart at 36k (noted, not attributed).
- Furnace (85 training consequences) was learned before table (103): the order does not follow count or attempt success rate.
- budget_continues FALSE: 50k − 36k one-step −0.009 / −0.004 (resolved; interact −0.252 / −0.085); depth 16 −0.012 [−0.027,
  +0.003] (s7, ns) and −0.035 [−0.056, −0.017] (s8).
- Ever position-wrong (s7): 0.449 (18k) → 0.298 (36k) → 0.283 (50k).
- dpanel 50k − 36k (s7, opened blocks): gen1 +0.010 / zombie +0.017; transfer1 zombie −0.014; transfer2 +0.013 (head-seed
  variance not in these intervals).
- No hit mode is learned by 50k.

---

## 2026-10-03 morning — the damage is a rule, not chance (correction); what the H16 decision needs

**Operations.** The machine was powered off at 22:01:48 on 10-02 (orderly systemd power-off). Lost: s8 36k→50k at update 47,500,
deepeval_36k_s8 mid-imagination, check_damage (never admitted). The s8 extension resumed from its own full state at 42,000 (AdamW
included; lane45 now prefers that state over the warm 36k one). deepeval_36k_s8 and check_damage were relaunched.

**CORRECTION of 2026-10-02 "damage is mostly aleatoric".** That figure (mean P 0.35-0.51 across 5 futures) compared the 5 futures at
the same STEP, after their states had diverged through random zombie moves. From one state the hit repeats:
- one step from the root, 17 actions x 4 keys: 389 of 403 dropping (root, action) pairs drop in 4/4 keys, 14 in 2/4;
- futures step 1 (shared root): 20 of 21 roots drop in 5/5.

**The hit is a source-exact rule with a hidden timer** (`check_damage_rule`; craftax_classic game_logic read 2026-10-03).
- Mechanics: a zombie hits (2; 7 asleep) iff its PRE-move position is next to the player's POST-move position and its hidden
  attack_cooldown <= 0; the cooldown resets to 5 on a hit and falls by 1 every other step. Arrows hit for 2; a hidden counter costs
  1 health about every 16 steps without food / drink / energy; lava kills.
- On 79,763 futures transitions the zombie rule predicts 0.920 of drops >= 2. The misses are arrows (70% have a skeleton within 4
  cells) and the "false" hits are hits netted against a same-step +1 recovery (48 of 63). rule_exact FALSE only for those reasons.
- P(hit | zombie beside the post-move player, last hit j transitions ago): j = 1-5: 0.024-0.049; j = 6: 0.978 (the 6-step cycle).
- What the world's input resolves:

  | visible history | P(hit), zombie beside the player after the move |
  |---|---|
  | 4-frame window (3 transitions), no hit seen | **0.512** (n 2,263) |
  | ... no zombie beside the player in those frames (fresh arrival) | 0.973 (n 147) |
  | ... a zombie already beside it | 0.480 (n 2,116) |
  | ... a hit seen in the window | 0.028 (n 2,141) |
  | 6 frames (5 transitions), no hit seen | 0.893 (n 1,135) |
  | no zombie beside the post-move player | 0.0017 |

- damage_visible FALSE. In the commonest case the 4-frame input gives a coin flip; a deterministic L1 world outputs the median and
  draws no hit. Six frames would make most hits predictable. The per-tile worlds use essentially only the current frame (09-30
  audit: w1 ≈ w4), where P(hit | beside) is 0.277.
- `check_damage` (relaunched) now also reports the teacher-forced catch per visible-history case (added before its run).

**The H16 panel measures a real decision** (`check_h16_signal`, FIT + DEV rows; the sealed judge is not read):
- within-root covariance of key 0's outcome with the other 31 keys' P: 98.5% of the across-action variation of P(dead by 16) is a
  real first-action effect (binomial noise 6%); observed oracle 0.745 vs 0.655 if the action had no effect.
- My conjecture that the H16 oracle is mostly sampling noise is refuted.

**What the H16 decision needs** (`check_h16_value`, 7,920 FIT + DEV roots with opportunity at 16):

| the chooser knows | expected safe at 16 |
|---|---|
| nothing (uniform) | 0.573 |
| the fixed action prior (DEV) | 0.613 |
| P(dead by 1) / P(dead by 4) exactly | 0.579 / 0.601 |
| **one real future per action** (survives or not; hindsight) | **0.663** |
| M sampled futures from a perfect stochastic world, M = 1 / 2 / 4 / 8 / 16 / 32 | 0.653 / 0.674 / 0.696 / 0.714 / 0.727 / 0.735 |
| P from 31 keys (oracle, unbiased) | 0.730 |

- Perfect short-horizon knowledge is nearly worthless at 16. On the 66% of roots with no opportunity by step 4 it equals uniform.
- One faithful future carries 57% of the margin (one_sample_insufficient FALSE); 16 futures carry 90% (M90 = 16).
- Our worlds' sealed gen16 (0.593-0.602, uniform 0.544, prior 0.586, real16 one draw 0.652) is far from the one-faithful-future
  value. Their futures draw no hits, so every imagined future survives.
- `check_h16_subst` (lane46, DEV) tests whether the H16 transfer gap is the drawn outcome (HUD) as at H1.

**Literature (primary text read):**
- Dedieu et al. 2025 (arXiv 2502.01591, the Craftax-Classic TWM):
  - T_WM = 20-frame windows, imagination 20 steps, burn-in 5;
  - next-state tokens SAMPLED (Q_{t+1} ∼ p_Θ), and rollouts show "feasible hallucinations ... such as spawning mobs and losing
    health";
  - separate reward and termination heads, sampled in imagination (MinAtar: ×10 CE weight "strongly penalizes inaccurate
    predictions of terminal states");
  - the health reward was dropped for a binary achievement reward.
- DreamerV3 (Hafner et al., arXiv 2301.04104, eq. 1, 5):
  - stochastic categorical representations, ẑt ∼ pϕ(ẑt | ht);
  - a continue predictor trained by logistic regression;
  - the actor learns from λ-returns over imagined trajectories with c_t, imagination horizon 15.
- MoP-JEPA (arXiv 2607.05238, Prop. 1): a regression-trained JEPA predictor converges to the conditional mean, "a point between the
  true next states that corresponds to no state at all"; independently trained heads collapse to the same mean, and hard assignment
  is what makes heads enumerate modes.
- Our L1 world is the per-dimension median version of the same collapse.
- Stochastic MuZero (Antonoglou et al., ICLR 2022) is relevant but its text could not be read (OpenReview bot wall), so it is not
  cited for any claim.

**E15 complete: sealed H16 of the budget worlds** (lane42; judge block 62,000-62,399; readings declared in lane42.sh):

| | s7 18k | s7 36k | s8 18k | s8 36k |
|---|---|---|---|---|
| fidelity err/copy at 1 / 4 / 16 | 0.204 / 0.529 / 0.673 | 0.184 / 0.522 / 0.602 | 0.205 / 0.503 / 0.707 | 0.187 / 0.530 / 0.620 |
| gen1 / gen4 / gen16 | 0.900 / 0.675 / 0.597 | 0.879 / 0.666 / 0.597 | 0.890 / 0.672 / 0.593 | 0.882 / 0.670 / 0.599 |
| transfer1 / transfer16 | 0.531 / 0.569 | 0.573 / 0.574 | 0.570 / 0.565 | 0.571 / 0.580 |
| gen16 − prior16 | +0.011* | +0.012* | +0.008 ns | +0.013* |
| transfer16 − prior16 | −0.017* | −0.012* | −0.021* | −0.006 ns |

- Readings:
  - h16_carried TRUE (small);
  - h16_usable FALSE;
  - budget_gen16 FALSE: 36k − 18k is +0.0005 [−0.005, +0.006] (s7) and +0.006 [−0.001, +0.012] (s8);
  - budget_transfer16 FALSE: +0.006 ns (s7); +0.015 [+0.008, +0.022] (s8) resolved, but under the 18k world's head-seed spread
    (0.019).
- Paired, H1: gen1 zombie-adjacent −0.019 ns (s7) and −0.039 [−0.061, −0.014] (s8); transfer1 +0.042 / +0.070 zombie (s7), ~0
  (s8).
- The 36k budget made depth 16 more faithful (err/copy −0.07 / −0.09) and moved no H16 decision. Read with check_h16_value: a
  future that never dies cannot carry the H16 decision, however faithful its map.

---

## 2026-10-02 20:30 — item 1 continued, sealed H16 (s7), the missing damage, option A validated

**Item 1, the gen regression next to zombies is REAL** (`check_headseeds`, 8 head seeds per world, two-level bootstrap):

| | overall | zombie-adjacent |
|---|---|---|
| s7 | -0.013 [-0.030, +0.003] (ns) | -0.030 [-0.052, -0.008] |
| s8 | +0.000 (ns) | -0.035 [-0.060, -0.013] |

- zombie_regression_real TRUE, within_head_noise FALSE.
- Not explained by the imagined states:
  - zombie AUC up;
  - branch alignment up (check_branches);
  - mob blur at depth 1 down (stochdiag: 1.14 / 1.12 → 0.94 / 0.93), lower excess. The "crisp wrong zombies" candidate is
    REFUTED.
- Not explained by the imagined health either (`check_hud_danger`): see the next item. Cause open.

**Imagination never draws the damage** (`check_hud_danger`, dpanel judge, step 1):

| | per-root correlation, -health vs P(death1), zombie / other | health spread across the 17 branches |
|---|---|---|
| real successors | 0.907 / 0.926 | 0.05-0.10 |
| every world, 18k and 36k | -0.07 to +0.11 | 0.010-0.013 |

health_signal_lost FALSE: the signal was never present.

**The whole H1 transfer gap is the HUD** (`check_transfer_subst`; dpanel's own real1 heads on imagined step-1 states):

| world | imagined | + real HUD only | + real 3x3 near the player |
|---|---|---|---|
| s7 18k | 0.643 | **0.999** | 0.666 |
| s7 36k | 0.686 | **0.999** | 0.702 |
| s8 18k | 0.660 | **0.998** | 0.678 |
| s8 36k | 0.686 | **0.999** | 0.704 |

- Swapping ONLY the 18 HUD tokens closes 99.8-99.9% of the gap at all four worlds; the near-player cells close 5-6%.
  hud_bottleneck TRUE.
- Caveat: real1 heads are hindsight references, so the real HUD carries the outcome. What this proves: everything a
  real-fitted head misses in imagination at H1 is the HUD consequence (the damage).

~~**Damage is mostly aleatoric**~~ **CORRECTED 2026-10-03** (see that entry): these numbers compare futures whose states had already
diverged; from one state the hit repeats (389 / 403), and it is a rule with a hidden 6-step cooldown, a coin flip from the 4-frame
input in the commonest case. Original text (diagnosis futures, 5 sampled futures under identical actions, simulator health):
- damage at a step in at least 1 of 5 futures: 945 of 15,836 alive steps;
- how many of the 5 take it: 1 / 2 / 3 / 4 / 5 = 584 / 174 / 81 / 45 / 61;
- mean P(damage | some future takes it) 0.35; given the factual sample takes it, 0.51, and all 5 in only 18%;
- size: mostly 2 health points (zombie hits).
- A deterministic L1 world draws the conditional median, "no damage" whenever P < 0.5. The risk is removed from
  imagination by construction. `check_damage` measures the catch rates.
- PlaNet (Hafner et al., ICML 2019, read): purely deterministic transitions prevent "capturing multiple futures and make it
  easy for the planner to exploit inaccuracies"; "the stochastic component is even more important – the agent does not
  learn without it". Option 3 (stochastic / generative) is therefore DECISION-relevant here, not only a diversity concern.

**Sealed H16, seed 7** (E15; judge block 62,000-62,399; 18k → 36k):

| | 18k | 36k |
|---|---|---|
| fidelity err/copy at depths 1 / 4 / 16 | 0.204 / 0.529 / 0.673 | 0.184 / 0.522 / 0.602 |
| gen16 | 0.597 | 0.597 |
| gen16 - prior16 | +0.011 | +0.012 (carried, small) |
| transfer16 | 0.569 | 0.574 |
| transfer16 - prior16 | -0.017 | -0.012 (not usable) |

- More faithful at depth 16, yet the decision value is unchanged: objective mismatch at the goal horizon.
- The s8 pair and the predeclared paired contrasts are pending.

**Option A validated** (lane44; warm restart of s7's 24k snapshot to 30k vs the true 30k snapshot):
- consfit 0.5625 vs 0.5607;
- onestep_all 0.131 vs 0.131 (every class within ±0.002);
- gen_16 0.642 vs 0.658;
- warm_ok TRUE: weights-only worlds can be extended.

---

## 2026-10-02 19:00 — item 1 (branches), item 9 (decision step), literature on nondeterminism and objective mismatch

**Item 1:**
- `check_branches`: near zombies, the 36k worlds' imagined branches are BETTER aligned with the real ones (0.928 → 0.939 s7,
  0.927 → 0.939 s8), with lower error (36.5 → 32.2, 37.7 → 32.4) and similar spread. branch_alignment_lost and
  branch_spread_lost are FALSE.
- dpanel's head-seed spreads (gen1 overall 0.008-0.045 across 3 seeds) are as large as the effect, and its interval omits
  head-fit variance. `check_headseeds` (8 head seeds, two-level bootstrap) is running.

**Item 9** (`check_decision_step`, both seeds, 18k and 36k):

| | s7 18k | s7 36k | s8 18k | s8 36k |
|---|---|---|---|---|
| false scrolls: target drawn passable, truly blocking | 0.90 | 0.87 | 0.93 | 0.85 |
| ... of those, target revealed after the root | 0.50 | 0.46 | 0.14 | 0.32 |
| missed scrolls repaired by the rest of the map | 0.36 | 0.65 | 0.67 | 0.66 |

- Readings: false_passable_drawn TRUE x4; false_on_revealed TRUE only for s7 18k.
- Most false scrolls erase obstacles the world could see:
  - at 36k, tables and furnaces are drawn as grass (about 27-34% of false scrolls; the placement mode is unlearned);
  - in s8 18k, 120 stone targets are drawn as PATH (the mined result);
  - trees are drawn as grass or path.
- Missed scrolls come from the whole-map content: ring 0.11-0.22, player <= 0.04, HUD <= 0.005.
- So unseen-terrain generation (option 3) can address at most the revealed part of false scrolls, about 15-20% of first wrong
  decisions at 36k.
- `check_erasure` (s7 18k / s7 36k / s8 18k / s8 36k), how the obstacle was lost:

  | | s7 18k | s7 36k | s8 18k | s8 36k |
  |---|---|---|---|---|
  | entering content misdrawn from its first appearance | 26 | 42 | 31 | 38 |
  | **erased after being drawn blocking** | 6 | 30 | 175 | 56 |
  | ... erasing step = DO on the faced cell | 0% | 73% | 73% | 86% |
  | placed never drawn | 12 | 16 | 14 | 16 |

  - In the worlds that learned DO, a DO on a faced obstacle that truly stays is drawn as removing it. Learning the DO
    consequence brought hallucinated removals.
  - CORRECTION (craftax_classic game_logic, read 2026-10-02): mining a tree ALWAYS turns it into grass. My "trees stay" was
    wrong. Stone becomes path only with a wood pickaxe.
  - `check_dohalluc`: DO on an obstacle that truly stays, drawn removed, teacher-forced / self-fed:

    | | removed, teacher / self-fed | true consequences caught |
    |---|---|---|
    | s7 18k | 0.011 / 0.048 | 0.000 |
    | s7 36k | 0.025 / 0.067 | 0.965 |
    | s8 18k | 0.403 / 0.417 | 0.828 |
    | s8 36k | 0.136 / 0.142 | 0.981 |

    - selffed_amplifies TRUE for s7, FALSE for s8: the s8 errors exist one step from true inputs.
    - learned_do_halluc FALSE at both seeds.
    - Edge-of-world cells are drawn removed 0.2-0.5 in every world.
  - `check_stone_pickaxe`: the data follow the rule exactly (no pickaxe → stays 962 / 962; pickaxe → mined 344 / 344).

    | | drawn mined, no pickaxe | drawn mined, with pickaxe |
    |---|---|---|
    | s7 36k | 0.005 | 0.904 |
    | s8 18k | 0.601 | 0.724 |
    | s8 36k | 0.199 | 0.954 |

    The stone-erasing runs ignore or partly ignore the inventory condition, which lives in the HUD tokens far from the
    faced cell. Whether a run learns that conjunction varies run to run.
  - Why ignoring the pickaxe is cheap in training (headfit_labels on the training pool):
    - DO facing stone succeeds in 52% of the pool's cases (n 7,162), but in 26% on the diagnosis futures (344 / 1,306).
    - Without the pickaxe cue, "mined" is the L1-favoured output in training (P > 0.5), and it hallucinates under the
      futures' distribution (a policy shift: the BC policy hits stone without a pickaxe far more often than the expert data).

**LABEL LIMITATION (found 2026-10-02 19:15; affects every "strict consequence" number since E13):**
- In the pool, DO on a tree turns it into grass (probe class after: grass 3,164 / 3,240), but the squared token change at
  the faced cell is median 49 (q90 54). The strict cut is > 120, the 99th percentile of probe-unchanged transitions.
- So consfit / headfit / check_allprobe / check_toperr / the "pos" and "event" counts EXCLUDE tree mining entirely. Their
  "consequences" are mining stone / coal / iron / diamond (stone: median 147) plus placements.
- The mask dose arms dosed the faced tile of every attempt, trees included.
- Tree removal is measured on the futures with simulator labels (check_dohalluc): caught 0.000 (s7 18k), 1.000 (s7 36k),
  0.886 (s8 18k), 0.998 (s8 36k).
- The encoder puts tree and grass close (squared distance about 49), though one blocks movement and the other does not. A
  blur toward grass costs little L1 for a decision-critical error. Tested below.
- Encoder token geometry (frozen Raw encoder, diagnosis futures, simulator labels, mob cells excluded), separation = squared
  distance between class means / mean within-class spread:

  | pair | distance | separation |
  |---|---|---|
  | **tree - grass** | 39.6 | **1.29** |
  | **stone - path** | 106.1 | **1.20** |
  | coal - stone | 103.6 | 1.17 |
  | table - grass | | 4.29 |
  | furnace - grass | | 3.49 |
  | sand - grass | | 3.80 |
  | lava - grass | | 5.84 |
  | water - grass | | 6.76 |

  - The two worst-separated solid / passable pairs are exactly the classes whose consequence token change is small, and
    the two largest groups of truly blocking targets drawn passable before false scrolls after placed objects (stone 13-120,
    trees 20-43).
  - Encoder geometry → small L1 cost for decision-critical errors → false scrolls: the last link is an inference
    consistent with the counts, not a causal test (that would substitute only these classes' tokens).
  - Missed scrolls are repaired more by the true REVEALED region (0.30 / 0.58 / 0.57 / 0.60) than by the observable one
    (0.18 / 0.17 / 0.31 / 0.19).
  - Content the world never saw drives most missed scrolls and 32-44% of the passable-drawn false scrolls at 36k. The
    "15-20%" bound above counted false scrolls only and is superseded.

**Literature (primary sources read):**
- Summers & Dinneen, ICML 2021:
  - cuDNN nondeterminism alone gives the same model diversity as different initialization seeds (Table 1: accuracy SD 0.22
    vs 0.23%, disagreement 10.5 vs 10.7%);
  - a one-bit weight change diverges within epochs (0.18 → 2.33 → 10.42% accuracy gap) and ends as different as any other
    source (Table 3);
  - "instability occurs as soon as a single hidden layer was added".
  - This matches check_determinism: same-seed runs are independent draws.
- Lambert et al., L4DC 2020: "objective mismatch". The likelihood of one-step predictions "is not always correlated with
  control performance". It names our observation (fidelity up, gen decisions not) but does not explain our instance.

---

## 2026-10-02 18:45 — diagnostics of the unexplained (lane41 + checks), and the budget worlds' full readings

**Budget (36k, plain recipe), both seeds vs the same-seed 18k teacher:**

| reading | s7 | s8 | two-seed |
|---|---|---|---|
| held consequences caught | 0.563 | 0.559 | |
| b_cost (onestep_all) | 0.149 → 0.126 | 0.152 → 0.130 | **pass** (every class better) |
| b_depth16 (gen_16) | -0.067 | -0.101 | **pass** |
| b_position (ever position-wrong) | 0.449 → 0.298 (-34%) | 0.493 → 0.339 (-31%) | **pass** |
| b_decision: gen1 overall | -0.012 (ns) | -0.001 (ns) | **fail** |
| gen1, zombie-adjacent | -0.034 | -0.036 | **worse at both seeds** |
| transfer1 | +0.044 | +0.027 | **better at both seeds** |
| transfer2 | +0.024 | +0.014 | **better at both seeds** |
| revealed terrain | still monotone (tree x0.017) | still monotone (tree x0.016) | |
| aligned share at depth 16 | 0.413 → 0.498 | 0.313 → 0.463 | |

**Learning is mode by mode.** The 0.56 plateau is exactly the DO consequences:
- DO caught 0.997 / 0.990;
- place stone / table / furnace caught 0.000 at both seeds;
- 315 / 560 = 0.5625;
- the dose (E14c) learns the placements too (1.0 / 1.0 / 0.94).

**D1, precision across snapshots** (`check_allprobe`): pre-transition h linear AP 0.09-0.21. At the transition, AP and the catch
jump together inside one 6k interval (s7 24k: AP 0.166 → 0.368, caught 0.002 → 0.432; s8 30k: AP 0.216 → 0.515, caught
0.002 → 0.557).
- Readings: s7 simultaneous; s8 representation first, marginally (AP 0.216 at 24k, 2.45x its 6k value, before its catch).
- After the transition, h's MLP AP is 0.74-0.75, above the raw local input's 0.667.
- Empirical relation in the uniform worlds: caught ≈ recall at precision 0.5 of a linear probe on h.
  - 36k: 0.56 vs 0.53 / 0.56; 30k: 0.56 vs 0.49 / 0.55; s7 24k: 0.43 vs 0.38.
  - About 0 below 0.15 recall.
  - Mechanism not tested.
- Literature: Saxe, McClelland & Ganguli (PNAS 2019, verified):
  - each mode of a deep network is learned in a sigmoidal transition, time O(1/s_alpha) up to a log factor;
  - the transition can be arbitrarily sharp from small initial weights;
  - the representation and the readout grow together.
  - Consistent with the measured coupled jump and with DO (the strongest consequence mode) being learned first; its
    predictions (placement later; time ~ 1/dose) are untested.

**D3, dosed backbones**:

| | h linear AP | h MLP AP | caught |
|---|---|---|---|
| E14c s7 / s8 | 0.234 / 0.254 | 0.50 / 0.49 | 0.99 / 0.95 |
| E14d | 0.313 | | 1.00 |
| linear | 0.363 | | 0.98 |

- end_to_end_precise FALSE, refit_paradox_holds FALSE. The dosed worlds catch without a precise h, and a uniform head on
  E14c's h catches nothing, as the precision predicts.

**Item 4, where the dose's damage comes from:**
- `check_facing`: the dosed worlds stop turning the agent.

  | | turns drawn right, moved / blocked |
  |---|---|
  | baseline | 0.985 / 0.950 |
  | mask1 + skip | 0.014 / 0.002 |
  | mask0.1 + skip | 0.163 / 0.118 |
  | linear | 0.004 / 0.002 |
  | 36k | 0.998 / 0.956 |

- `check_facing_h`: h at the player token still encodes the next facing in every world (linear 0.95-0.98, baseline 0.981):
  backbone_lost_turn FALSE.
  - The heads stop generating there on turns: generate weight 0.000 (mask1 + skip), 0.004 (linear), 0.238 (mask0.1)
    vs 0.540 (baseline).
- `check_gencand`, the generate candidate's L1 on turns (copy 0.403):

  | | generate candidate L1 | |
  |---|---|---|
  | baseline | 0.393 | |
  | mask1 + skip | **0.718** | gen_degraded |
  | linear | **1.030** | gen_degraded |
  | mask0.1 + skip | 0.383 | not degraded |
  | 36k | 0.177 | |

  - At lambda 1, the shared generate path is pulled toward consequence content and the gate rationally copies the player.
  - At lambda 0.1, the candidate is fine and the GATE copies (self 0.657): cause open.
- `check_e14c_grad` on E14d: the mask term's backbone gradient at lambda 0.1 is 0.13x the uniform term's. **Gradient
  dominance is retracted as the mechanism** of the damage (it was a lambda = 1 correlate).
- `e14d_cost`: no false scrolls. The player token on moved / blocked steps is +142% / +136% even at x31. The linear arm's
  interact excess is the 4 near-player tiles (0.028 → 0.173).

**D2, oracle substitution on the 36k world** (s7, ever position-wrong):
- the consequence oracle still gives -25% (18k: -37%): consequence_gain_shrinks FALSE;
- the entering-cell oracle gives -51% (18k: -24%). Entering terrain is now the largest remaining trigger of position
  failures against the true future.
- The oracle uses the TRUE terrain, which no world can know; it bounds perfect knowledge, not generation.

**Determinism** (`check_determinism`): plain GPU training differs from update 20 (max |diff| 4.8e-7, then 2.2e-5 at 300,
about x2 per 48 updates). Under `torch.use_deterministic_algorithms(True)` the pair is bit-identical at 20 and 300 updates.
gpu_nondeterministic TRUE: run-to-run divergence is kernel nondeterminism amplified chaotically.

**Resumable training** (f69d4411): full state every 6k updates and at the end; `--resume`; bit-exact on CPU.

---

## 2026-10-02 18:00 — E14b two seeds: budget learns the consequences; same-seed runs do NOT reproduce

Held strict caught, the plain teacher recipe:

| snapshot | s7, 18k run | s7, 36k run | s8, 18k run | s8, 36k run |
|---|---|---|---|---|
| 6k | | | 0.000 | |
| 12k | | | 0.002 | |
| 18k | 0.002 | 0.002 | **0.271** | **0.002** |
| 24k | | 0.432 | | 0.002 |
| 30k | | 0.561 | | 0.557 |
| 36k | | 0.563 | | 0.559 |

- **budget_fixes TRUE**: held caught >= 0.5 at 36k at both seeds; hallucinated 0.007 / 0.015.
- **reproducible FALSE** (predeclared): the s8 36k run's 18k snapshot catches 0.002, the original 18k run 0.271.
  - s7's check was uninformative: both runs sat at 0.002.
  - Weights already differ at 6k at both seeds: median relative difference 19% (s7) / 19% (s8), growing to 32% by 18k.
  - The args are identical. The script hashes differ, but the default recipe is bit-identical on CPU.
  - check_determinism.py (predeclared) tests GPU kernel nondeterminism.
- Consequences until that test reports:
  - "s8 learns earlier than s7" (E14a) is NOT a property of the seed. The transition time varies run to run, at least
    6-12k updates for the same seed.
  - E14a's facts about the specific trained worlds stand: a fresh head on THAT s8 backbone catches 0.257.
- 36k s7 decision panel (36k - 18k teacher):

  | | gen1 | gen2 | transfer1 | transfer2 |
  |---|---|---|---|---|
  | overall | -0.012 (ns) | 0.000 (ns) | **+0.044** [+0.030, +0.058] | **+0.024** [+0.017, +0.031] |
  | zombie-adjacent | -0.034 (resolved) | | | |

  b_decision fails at s7; transfer improves, the same split as E14c.

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
