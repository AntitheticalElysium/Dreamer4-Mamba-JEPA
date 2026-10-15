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

## 2026-10-15 — TRANSPORT MANDATE (lead, Opus): does stochastic reassignment of retained content beat free synthesis?

User mandate (10-15): Stage 1 independent assessment of Codex's `PROVENANCE_ASSESSMENT.md` (458dd18), then Stage 2 feasibility
controls, then at most ONE preregistered matched comparison. No reconstruction, no simulator labels in training, no H16 /
actor, no architecture search.

### STAGE 1 — independent assessment (`transport_stage1.py`, CPU, evaluation-side layouts only)

**Codex's corrections to my S5-S6 audit, accepted:**
- All-source valid movement did rise 1.5k → 3k (52 → 113 / 896). I reported cows only.
- The objective penalizes shuffles +6.2%, not zero. "Nothing in the loss" was too strong; "weak pressure" stands.
- The two direct runs share initialization and ledgers. Their spread measures numerical divergence, not seed variance, and
  "within spread" was too categorical.
- The optimizer weight-decay mismatch is real.

**Measured (3,000 random windows of the exact training ledger, 39,420 steps):**

1. **Locality: the representation holds local, transferable mob content.**
   - Ridge probe from ONE Raw token to its own tile's zombie / cow / skeleton: AUC **1.000 / 1.000 / 1.000**.
   - To "a neighbour holds this mob" when its own tile is empty: AUC **0.73 / 0.74 / 0.78** (base rates 2.0 / 3.1 / 0.8%).
     Neighbour leakage exists but is weak at the linear level. A renderer could still exploit it, so provenance must be tested.
   - **Real single-cow moves (2,946, stationary camera) are nearly additive transfers:** z'_dst ≈ z_dst + (z_src − z'_src)
     with median relative error **5.7%**, and cos(Δz_dst, −Δz_src) = 0.886.
2. **The non-additive tail is the background.**

   | cow moves | n | median error | share > 0.3 |
   |---|---|---|---|
   | same tile class at source and destination | 2,679 | 5.2% | 4.4% |
   | **different tile class** | 267 (9%) | **116%** | 100% |
   | top 10% static noise | 295 | 27.6% | 46.8% |

   - Among moves with error > 0.3: 69% are terrain mismatches, 36% have high static noise, 6% are dark.
   - The token binds the mob to its background. A raw-token move drags the source's terrain along, so a content extractor and
     contextual renderer are justified by measurement, for those 9%.
3. **Coverage of transport (shares of each type's mob changes):**

   | | in-view move | border in/out | interior birth/death | DO kill |
   |---|---|---|---|---|
   | zombie | 79.6% | 15.1% | 3.0% | 2.3% |
   | cow | 72.4% | 22.3% | 3.0% | 2.3% |
   | skeleton | 67.2% | 28.4% | 2.9% | 1.5% |
   | arrow | 58.9% | 14.5% | **26.6%** | 0 |

   - Border crossings sit on unscored edge cells, and kills are deterministic given the action.
   - Inside the scored interior, transport cannot express about 3% of terrestrial mob changes. Arrows are excluded from any
     claim.
   - **The camera moves on 37.6% of steps.**

**Assessment of the proposal:**
- **(a) Frame. Correction needed.** Codex's c_i = C(z_i, h_i) reads the current frame. Then identity T returns the reference
  only on stationary steps (62%). Carriers must be read from the deterministic afterstate (the reference, already in
  next-frame coordinates). That is Stochastic MuZero's afterstate/chance split.
- **(b) The S5/S6 bypass closes only if noise reaches nothing but T.** If the only stochastic input is the assignment, no tile
  has a private creation path. A soft T reintroduces split content (blends), so the operator should be an exact matching in
  the forward pass, used identically in training and evaluation, with a straight-through gradient. That differs from Codex's
  "continuous operator", which reopens the blend failure S6 measured.
- **(c) Residual bypasses:**
  - **The renderer reads context.** D(h_j, ·) sees a belief that knows adjacent mobs, so a moved non-source carrier could be
    rendered as the mob → payload-substitution test (F2).
  - **The reference can already be wrong.** It is jointly valid on only 10/14 strict sources at u0, and transport moves
    whatever the reference holds → report reference validity alongside.
  - **Collapse.** D may ignore its carrier: everything stays.
- **(d) Births, deaths, collisions.**
  - A matching of adjacent swaps excludes collisions and duplication by construction.
  - It cannot express spawns, despawns, border crossings or arrows. Those stay deterministic (reference), so the operator arm
    under-samples them. The scored interior loses ~3%.
  - No stochastic creation path is added, because that reopens S6's bypass.
- **(e) What a matched comparison could establish:**
  - That limiting noise to a joint reassignment improves *learned* movement coherence over per-tile synthesis, for one seed,
    budget and panel.
  - Validity is partly guaranteed by construction, so the binding readings are learned movement frequency, direction and
    alternatives (cow valid move vs real 41%).
  - Remaining confounds: the gradient estimator (straight-through discrete vs continuous), parameter count, run variance, and
    reader artifacts.
- **Literature (checked, mechanism-level):**
  - [ITC](https://arxiv.org/html/2605.16457v3)'s correspondence is a *deterministic* Sinkhorn + greedy decoding step, with no
    learned or sampled transport. Its stochasticity comes only from "wildcard" token generation, and it hands screen edges to
    the plain transformer. It reports per-step accuracy and return gains, but no duplication rates.
  - [Gumbel-Sinkhorn](https://arxiv.org/abs/1802.08665) (Mena et al. 2018) is the precedent for *learned noise-selected
    permutations*.
  - No precedent found trains a noise-selected correspondence with a proper score on continuous predictive features, so the
    question is open, not established.

**Verdict:** credible enough for Stage 2, with the afterstate-frame and hard-matching corrections.

### STAGE 2 / F1a — raw-swap expressibility (`transport_stage2.py rawswap|rawswap3k`; T* from layouts, diagnostic only)

**Setup:**
- 104 bright DEV roots × 32 real keys.
- T* = greedy one-to-one adjacent same-type swaps in the afterstate frame (exact camera offsets).
- Carrier and renderer are identity: ẑ = LN(T*·b), so a destination receives the source's afterstate token exactly.
- Readouts under both frozen readers: the 14 strict sources (205 move keys, 243 stay keys), and all 1,145 interior moves in the
  panel.

| afterstate used | strict move keys (real / bank) | strict stay keys | panel moves, both endpoints (real / bank) | RMS to real at endpoints (median) |
|---|---|---|---|---|
| **true** (aligned real tokens) | **100% / 100%** | 100% / 100% | **99.7% / 99.7%** | 0.060 (static floor 0.009) |
| D parent reference (u0) | 75.6% / **24.9%** | 94.2% / 58.0% | 74.7% / 42.5% | 0.207 |
| paired-3k reference | 62.4% / 63.9% | 92.2% / 91.0% | 72.8% / 73.5% | 0.075 (p90 0.470) |
| reference with T = I | 0 / 0 | as above | 12.6% / 6.8% | 0.355 / 0.475 |

**Readings:**
1. **Transport is fully expressible in the frozen JEPA representation.** Swapping real afterstate tokens gives decodable valid
   moves and stays, with no learned carrier or renderer. The cross-terrain background error (Stage 1) costs token fidelity,
   not reader-level validity: 3 failures in 1,145.
2. **Neither available reference is an afterstate.** Swapped tokens are exact copies, and both readers score each token
   independently, so the failures are in the reference itself:
   - **D parent:** every failed move key is "multiple" (50 / 205 real, 154 / 205 bank). The uncertain mob is ghosted onto its
     neighbours. Under the bank reader, the reference alone "detects" it at 635 of 1,145 eventual destinations and clears only
     98 sources.
   - **Paired-3k reference:** failures are "absent" (77 / 205 real). On some sources the mob is faded out of the reference.
   - Both were trained toward the *post-chance* target, so each absorbed part of the chance as a mean. That is Stochastic
     MuZero's afterstate ≠ E[next state], now measured. Transport cannot remove ghosts on cells it doesn't touch, nor restore
     a faded source.
3. **Consequence for the proposal:** the operator needs a pre-chance afterstate. Whether a trainable reference becomes one when
   the operator owns the chance is the next control (F1b), before any stochastic experiment. Otherwise a Stage 3 negative
   could not be read.

### STAGE 2 / F1b — can a trainable reference become an afterstate? (`transport_stage2.py fit|fiteval`)

**Setup:**
- From the exact D parent: 1,500 updates, batch 16, L16, the parent's own D loss, phase AdamW. Decay groups 127 decayed /
  18 exempt, so the Mamba flags are preserved.
- Identical seeded windows of the training ledger, restricted to the 320,376 windows with layouts.
- **assigned:** output LN(T*·b), with T* from layouts plus the token-estimated camera shift (diagnostic only; 0.53 swaps per
  step, the census rate).
- **identity:** output LN(b), the parent's objective continued for the same budget.
- The launch rule (≥ 90% on all three readings under both readers) was fixed in TRANSPORT_PREREG.md (5aa5f562) before this
  evaluation.

| reference, raw swap on DEV | strict move (real / bank) | strict stay (real / bank) | panel both endpoints (real / bank) | endpoint RMS |
|---|---|---|---|---|
| assigned u500 | 100 / 100% | 100 / 100% | 98.2 / 96.9% | 0.074 |
| **assigned u1500** | **100 / 100%** | **100 / 100%** | **98.4 / 98.9%** | **0.063** (true-swap ceiling 0.060) |
| identity u500 | 77.1 / 19.0% | 93.0 / 49.8% | 78.3 / 35.5% | 0.215 |
| identity u1500 | 100 / **30.7%** | 100 / 66.3% | 81.1 / 45.8% | 0.212 |

**Readings:**
- **Launch rule met: Stage 3 launches.**
- When the reassignment is owned by the operator, here supplied, the trainable reference becomes a clean pre-chance
  afterstate within 500 updates, near the representation's ceiling.
- Trained toward the post-chance target with the same budget, the reference keeps faint neighbour ghosts, the optimum of the
  mean. All bank-reader failures are "multiple" (142 / 205). The real reader's higher threshold hides them.
- So whether the deterministic part absorbs the chance is decided by whether the stochastic operator owns it. This is
  Stochastic MuZero's afterstate decomposition, measured in our continuous JEPA world.
- **Not shown:** that the operator can *learn* the assignment from the energy objective. That is Stage 3.
- **Pre-launch execution checks:**
  - `transport_model_test.py` passes: exact disjoint adjacent swaps; forward = hard permutation; the reference's gradient
    follows the moved content; ~1 move per frame at initialization.
  - Smoke: decay groups 143 / 18 identical by name in both arms; completion parameters 578,752 (free) / 530,693
    (operator); peak memory 1.41 GiB.
  - A smoke-found bug (no save on `--stop-after`) was fixed. Interrupted 2 → 4 vs uninterrupted 4: weights identical,
    optimizer state within 8e-12, ledgers identical.

### STAGE 3 — the matched comparison (TRANSPORT_PREREG.md, 5aa5f562): free synthesis vs joint reassignment

**What ran:**
- Both arms trained in one process for 3,000 updates, with identical windows, Gaussian noise and Gumbel draws (ledger
  `9dadd2560b5bcd02…`).
- Same parent, backbone, trainable reference, noise trunk (initialization asserted equal) and energy objective.
- Decay groups 143 / 18, identical by name.
- Parameters: 6,960,592 free / 6,912,533 operator; completion 578,752 / 530,693.
- Median 0.312 / 0.327 s per update; peak 1.48 GiB with both arms resident.
- Evaluation: 104 DEV roots × 64 samples; 14 strict sources under both frozen readers.

**Primary result at 3,000 updates.** Where the two readers agree, one number is given; otherwise real / bank.

| | free | operator | operator − free, 95% root bootstrap |
|---|---|---|---|
| valid | 63.2 / 53.3% | **93.8%** | +30.6 [19.8, 42.0] / +40.4 [26.6, 55.1] pp |
| all-source valid movement | 8.5 / 4.8% | **29.2%** | +20.8 [12.4, 30.3] / +24.4 [14.7, 34.8] pp |
| cow valid movement (real 41.0%) | 4.5 / 3.5% | **25.0%** | +20.5 [10.7, 31.8] / +21.5 [11.5, 33.0] pp |
| duplicate | 19.0 / 31.5% | 4.91% | −14.1 / −26.6 pp |
| absent | 15.0 / 7.0% | 0% | −15.0 / −7.0 pp |
| dependence (real 45.8pp) | 0.8 / −0.7pp | **31.6pp** | +30.7 / +32.2 pp |
| distinct valid outcomes (real 3.64) | 2.43 / 2.14 | 4.07 | |
| scored joint ES | 0.0257 / 0.0275 | **0.0248 / 0.0256** | |
| scored static corruption | 0.27% | **0.72%** | |

**Gates (both readers):** G1 movement **pass**; G2 coherence **pass** (duplicate 4.91% ≤ 5%); G3 comparison **pass**; **G4
static FAIL** (0.72% > max(0.27%, 0.5%)); G5 joint law **pass**.

**Preregistered verdict: "transport better" is NOT established.** Reserved fresh roots stay sealed; no repair or extension.

**Attribution (frozen diagnostics, `transport_diag.py`):**
1. **Validity:** 13 / 14 operator sources are 100% valid under both readers. The only failure, zombie root 83, comes from
   its *reference*, which holds a ghost ("multiple"); that root produces all the duplicates.
   - Validity is partly constructive: swaps cannot duplicate or erase.
   - It is also partly learned: the operator's reference became a clean afterstate (13 / 14 valid stays under the bank
     reader vs 10 / 14 for the free arm and 6 / 14 at u0), as F1b predicted.
   - Payload substitution: in 278 / 278 cases the destination's mob vanishes when the moved content is replaced. Provenance
     holds.
2. **The movement law is learned, but under-dispersed:**

   | P(move) by content | u0 | u500 | u1500 | u3000 | real |
   |---|---|---|---|---|---|
   | cow | 1.4% | 1.5% | 6.6% | **20.9%** | 36% |
   | skeleton | 1.5% | 1.9% | 8.4% | **38.6%** | 61% |
   | zombie | 1.5% | 1.6% | 3.2% | 6.6% | 43% |
   | no mob | 1.4% | 1.3% | 1.4% | 2.1% | — |

   - Mob moves into solid terrain fell from 21% to 6.9%. 11.9% of strict-source moves go in directions with no real support.
   - Per-source movement tracks reality, mostly below it. Zombies, which chase the player, are barely learned.
   - The straight-through push on source moves still favours moving at 3k (diversity +7.5e-4 vs attraction −5.0e-4,
     8 / 14 roots). The movement curve had not converged.
3. **G4 is caused entirely by terrain swaps.**
   - Scored static corruption: operator samples 0.737%, its own reference alone 0.296%, free samples 0.303%.
   - Swaps per frame at 3k: mob ↔ passable 0.19, mob ↔ solid 0.014, terrain ↔ same terrain 0.94, **terrain ↔ different
     terrain 0.155**.
   - The terrain swaps come from the initialization prior (~1 proposal per frame, β), which is never unlearned. P(move | no
     mob) goes 1.4% → 2.1%.
   - **Why:** at terrain-only tiles the two halves of the energy objective nearly cancel (attraction −4.4e-5 vs diversity
     +3.8e-5 per pair, at 1.5k and at 3k). A rare spurious swap raises both terms by almost the same second-order amount
     under the first-power distance. This is the same per-sample insensitivity S5 found for dependence.
   - The late rise is probably a side effect of the shared proposal weights learning mob moves. That is inference, not
     measured.
4. **The objective itself favours the true law.** Model-free: real-key samples of the true law score ES 0.0365, vs 0.0611 for
   the same keys with moves undone (+68% on 13 / 14 strict roots; +89% on 56 / 62 roots with moves). Attraction alone
   prefers stay on 11 / 14, so the diversity term carries all of it. That is why discrete moves are learned slowly, and
   only where two draws differ.

**Reading:**
- On every coherence and movement measure, joint reassignment of retained content learned far better than per-tile
  synthesis under the identical objective, data, noise and budget. Movement grew from 2% (random) to 29% by learning, not
  by construction.
- The preregistered static gate fails because the same objective does not price spurious terrain reassignment. Its
  per-sample signal is nearly zero there.
- **Remaining confounds:**
  - one training seed;
  - the straight-through estimator exists only in the operator arm;
  - the operator cannot express births, deaths or border crossings;
  - the zombie law is unlearned.
- **Not tested:** recursion, decisions, nuisance, memory.

### TRANSPORT COST MANDATE (user, 10-15 afternoon) — pre-launch audits (TRANSPORT_COST_PREREG.md)

**Principle under test:** preserve what persists, learn what changes, sample only what is genuinely unpredictable. The
transition cost is λ·E[tiles proposing a change]: a general minimum-description-length prior on stochastic change, with
no labels, no terrain rules, and no new parameter.

**1. Lineage is clean** (`transport_lineage.py`, `transport/lineage.json`):
- Stage 3's pinned inputs contain no layouts, F1b code or checkpoints, fan or panel.
- Both Stage 3 arms' u0 weights equal the D parent exactly (145 tensors) and differ from every F1b checkpoint (up to 0.024).
- The D parent is the oracle study's control arm, with identically zero oracle weights.
- All snapshots descend from the single training contract.
- **F1b was strictly an oracle positive control**: evaluated, never loaded for training.
- Disclosure: β ("~1 proposal / frame") was picked after seeing the census rate of 0.5 moves per step.

**2. The cross-terrain contextual-token problem is measured, and a cost cannot fix it** (`transport_cost_diag.py
crossterrain`, exact tiles):
- 5.2% of real in-view mob moves (60 / 1,145) cross terrain classes.
- Swapping the TRUE tokens decodes 100% of those endpoints with the wrong terrain, against 0% for same-terrain moves and 0%
  in real successors.
- The seed-1 operator makes 10.8% of its mob moves across terrain, all with wrong endpoint terrain.
- With an identity carrier, the token moves the mob together with its background. That is a representation limit
  (WHITEBOARD §5: content vs context), not a sampling one.

**3. What the score pays per unit of move probability** (`benefit`, 48 stationary training windows, label-free classes;
seed-1 operator at 500 / 1,500 / 3,000):

| content | value |
|---|---|
| static | +4.5e-5 / +9.8e-5 / +2.9e-5 (window SE 5.7-7.4e-5): indistinguishable from 0, the measured cause of never-unlearned terrain swaps |
| changing, along realized moves | +5.2 to +6.5e-3 |
| changing, other directions at moving steps | +3.5e-3 |
| changing, at staying steps | −1.1 to −2.3e-4 |
| labelled mob tiles, all steps (interpretation) | −4.8e-4 / +2.8e-4 / +2.4e-4 |

- **λ = 1e-4 by a fixed rule**: the smallest one-significant-figure value above the largest mean static value.
- It keeps a positive incentive for content that changes in more than about 1-2% of steps per direction. Cows are at
  ~9-12%.
- The margin over average mob tiles is only 2-3×, so the stay-still risk is real. A preregistered go / no-go probe (matched
  1,000-update continuations with and without the cost) runs before any cost training. Pass: static P(move) −≥ 30% and cow
  P(move) −≤ 15%.

**4. Go / no-go probe: PASS** (`transport_cost_diag.py probe 1e-4`):
- Setup: from seed-1 operator u3000 (model and optimizer restored), two matched 1,000-update continuations (identical fresh
  windows and noise). Measured on the 48 stationary training windows.

  | P(move) | u3000 start | no-cost end | cost end | cost vs twin |
  |---|---|---|---|---|
  | static (label-free) | 0.0131 | 0.0068 | 0.0035 | **−48.5%** (rule: ≥ 30%) |
  | cow | 0.193 | 0.214 | 0.186 | **−13.1%** (rule: ≤ 15%) |
  | zombie | 0.048 | 0.050 | 0.043 | −13% |
  | skeleton | 0.215 | 0.263 | 0.244 | −7.5% |

- Expected changes per frame: about 1.2 without the cost vs 0.85 with it.
- **Caveats:**
  - Continued training alone also halves static proposals: the terrain swaps were being slowly unlearned anyway, and the cost
    roughly doubles that rate.
  - The cost also costs mob movement (−7% to −13%), close to the allowance. The trade-off is real, not free.
- Launching the preregistered experiment: seed set 1 {operator, operator_cost}, seed set 2 {free, operator, operator_cost},
  λ = 1e-4, 3,000 updates (`run_transport_cost.sh`).
- `transport_eval.py` was generalized (directories, arms, `--free`). A regression check reproduces the Stage 3 analysis
  exactly: arms, per-source rows, bootstrap intervals.

### TRANSPORT COST + SECOND SEED — results (TRANSPORT_COST_PREREG.md; mechanical verdict `transport_cost_verdict.py`)

**Runs:**
- Seed set 1: {operator, operator_cost} in one process. The streams are identical to Stage 3: update-1 energy 0.064021 and
  0.8146 swaps / frame, exactly as Stage 3.
- Seed set 2: {free, operator, operator_cost}.
- λ = 1e-4, 3,000 updates. All checkpoints kept under `transport/seed1_cost`, `transport/seed2`.

| at 3,000 updates (real / bank where they differ) | valid | dup | all move | cow move | distinct | dependence | static | swaps / frame | ES |
|---|---|---|---|---|---|---|---|---|---|
| seed 1 free (Stage 3) | 63.2 / 53.4% | 19.0 / 31.5% | 8.5 / 4.8% | 4.5 / 3.5% | 2.43 / 2.14 | ~0 | 0.27% | — | 0.0257 / 0.0275 |
| seed 1 operator (in-process replicate) | 93.8 / 89.4% | 4.5 / 5.7% | 30.4 / 26.9% | 28.1% | 4.07 / 4.00 | 32 / 29pp | 0.80% | 1.67 | 0.0249 / 0.0257 |
| seed 1 operator + cost | 93.8% | 4.6% | 23.3% | **19.5%** | 4.00 | 27pp | **0.61%** | 0.75 | 0.0253 / 0.0260 |
| seed 2 free | 65.3 / 55.8% | 15.0 / 30.4% | 7.0 / 5.0% | 7.8 / 6.6% | 2.86 / 2.50 | 3pp | 0.27% | — | 0.0253 / 0.0282 |
| seed 2 operator | 93.4% | 5.1% | 31.5% | 30.3% | 4.14 | 34pp | 0.68% | 1.15 | 0.0239 / 0.0251 |
| seed 2 operator + cost | 99.0 / 93.9% | 0 / 4.9% | 31.3 / 25.6% | 23.6% | 4.07 / 4.00 | 33 / 31pp | **0.63%** | 0.75 | 0.0257 / 0.0265 |

**Verdict: "not established"** (preregistered, mechanical).
- **R, architecture replication: PASS (seed 2, both readers).** Operator − free: valid +28.1 [16.4, 42.0] / +37.6 [22.7, 55.1]
  pp; all-source valid movement +24.4 [14.1, 35.2] / +26.5 [16.0, 37.6] pp; cow movement +22.5 / +23.6 pp; dependence +31pp.
  - Across Stage 3, the seed-1 replicate and seed 2, the operator lands at 93-94% valid, 29-31% movement, 25-30% cow
    movement.
  - The gain over free synthesis is robust to training variability and to cross-process numerical divergence.
  - Without the cost, the operator still fails G4 everywhere (static 0.68-0.80%), and G2 narrowly on duplicates
    (5.1-5.7%, from the reference-ghost zombie root).
- **C, the cost: FAIL at both seeds.**
  - Seed 1 fails G1 (cow 19.5%), G4 (0.61%) and N.
  - Seed 2 fails G4 (0.63%) and N: cow −6.6pp [−13.7, −1.8], bank movement −5.9pp.
  - It passes coherence (G2) and joint law (G5) at both seeds.

**Mechanism (`transport_diag.py swaps|content` on the new runs):**

| operator → operator + cost | seed 1 | seed 2 |
|---|---|---|
| terrain ↔ different terrain swaps / frame (static damage) | 0.185 → 0.095 (−49%) | 0.127 → 0.093 (−27%) |
| terrain ↔ same terrain swaps / frame | 1.20 → 0.46 (−62%) | 0.73 → 0.43 (−40%) |
| mob ↔ passable swaps / frame (genuine moves) | 0.195 → 0.146 (−25%) | 0.196 → 0.159 (−19%) |
| P(move \| no mob) | 2.9 → 1.1% | 1.7 → 1.0% |
| P(move \| cow), real 36% | 22.7 → 14.9% | 22.8 → 17.7% |
| P(move \| skeleton), real 61% | 42.5 → 32.2% | 41.8 → 32.0% |
| P(move \| zombie), real 43% | 10.3 → 6.7% | 9.8 → 6.7% |

- The cost is selective only ~1.4-2×: damaging terrain swaps fall 27-49%, genuine moves fall 19-25%. That is exactly the
  thin margin the probability-space measurement predicted (static ~0 ± 7e-5 vs mob ~2.4-2.8e-4 per unit probability). A
  uniform price per change cannot separate them, because the score values them too similarly per sample.
- Provenance holds in every run (payload substitution 228-300 / 228-300).
- The static excess over the reference shrinks only ~0.1-0.2pp, while cows move further from the truth.

**Further measured (CorrT calibration, `transport_cost_diag.py corrt`, D parent, 14 strict sources):**
- The deterministic CorrT's neighbour-copy weights rank real moves correctly: r = 0.70 over 56 source-destination pairs;
  keep-the-source weight vs real stay r = 0.89.
- But they are mis-scaled about 13×: mean copy weight 0.0088 vs real move frequency 0.114. The rest of the move mass is
  hidden in a generator blend. Static neighbour-copy mass is 0.0035.
- A mean-trained mixture weight is a regression coefficient, not a probability. CorrT already encodes *where* content can
  go (the ghosts are exactly its small copies), but sampling it directly would under-move about 13×.

### OBJECTIVE INVESTIGATION (user mandate 3, model-free, no training) — `transport_objective.py`

**1. Clean afterstate: oracle control vs unprivileged training** (`afterstate`). Each reference's own deterministic
afterstate, swapped with the TRUE reassignment of every real key (diagnostic):

| reference | strict move (real / bank) | strict stay | panel moves | static corruption |
|---|---|---|---|---|
| true afterstate tokens | 100 / 100% | 100 / 100% | 99.7% | 0.176% (real 0.179%) |
| **F1b oracle-assigned** (privileged) | 100 / 100% | 100 / 100% | 98.4 / 98.9% | 0.266% |
| **unprivileged operators** (5 runs) | 92.7 / 92.7% (one 100 / 77%) | 93.0% | 79.7-84.1% | 0.266-0.298% |
| free arms | 63.9 / 55-68% | 87-91% | 69-75% | 0.266% |
| D parent | 75.6 / 24.9% | 94.2 / 58.0% | 74.7 / 42.9% | 0.266% |

- The fully clean afterstate was shown only by the oracle control.
- Unprivileged reassignment training gets most of the way: 13 / 14 strict sources clean, but panel moves 80-84% vs 98%.
- **With correct reassignment every reference passes the static gate (≤ 0.30%).** The gate excess comes entirely from wrong
  reassignments, and correct cross-terrain moves never touch it (their endpoints are not stable cells). 0% of the gate is
  irreducible under whole-token transport.

**2. Exact model-free scoring rules** (`rules`; real fans, mixture ES = p·A − ½pᵀDp, exact derivatives). Held-out 96 roots,
eval 104 in brackets:

| rule | movement margin (relative) | dependence (shuffle), share of margin | spurious different-terrain penalty at ε = .02, share of genuine-move value |
|---|---|---|---|
| E0 (existing) | 0.55 (0.77) | 0.034 (0.071) | 0.077 (0.106) |
| G (global) | 0.51 (0.73) | 0.040 (0.083) | 0.077 (0.103) |
| CROSS | 0.41 (0.84) | 0.018 (0.031) | 0.091 (0.122) |
| DOM | 0.38 (0.83) | 0.008 (0.015) | 0.091 (0.124) |
| TOK (marginal-only control) | 0.30 (0.82) | **0.000** | 0.091 (0.123) |

- All rules favour the true law over stay (93-96% of roots).
- The spurious penalty is linear in its probability ε (the score is quadratic in mixture weights) and **negative at ε = 0**
  for every rule.
- Locality rescales penalty and movement value together (ratio +18%), and *reduces* dependence detection (also for a break
  confined to each mover's cross).
- Dependence is a small part of the score's value under any rule: marginals alone already favour movement.
- No hybrid weight gives a robust region where false transitions are strongly discouraged.

**3. The learning signal through the actual stochastic head** (`headgrad`; frozen seed-1 and seed-2 operators, 48 stationary
training windows, training-style two-sample estimator, straight-through):
- **Global G:** static ≈ 0 (t ≈ 0); labelled mob +2.5 / +7.0e-4; realized moves +7.8e-3.
- **E0:** static +3 to +5e-5.
- **CROSS / DOM / TOK:** static **+0.9 to +1.4e-4** (t 1.4-1.7) and labelled mob **−4 to −8e-4**. Locality rewards spurious
  static swaps and discourages mob movement, the opposite of the hypothesis.
- **Exact reason:** the operator's two draws are identical outside their swaps. For a one-draw change δ, the score changes by
  ½(|e+δ| − |e|) − ½|δ| ≤ 0 (triangle inequality, e = the afterstate's error). The score never penalizes it unless the
  prediction is unbiased. A global norm only dilutes this; locality concentrates it.

**4. Mechanism confirmed by dose-response** (`headgrad … oraclestatic | hard`). Static-tile push as the afterstate's error at
static tiles goes from 0 (privileged: the real next token substituted) to soft (bias 0.014) to hard-copy (bias 0.028):

| rule | error 0 | soft | hard |
|---|---|---|---|
| G | −8.8e-5 (t −3.8) / −7.5e-5 (t −3.1) | ≈ 0 | +7.2e-5 / +9.3e-5 |
| E0 | −6.3e-5 (−2.9) / −5.5e-5 (−2.4) | +3 to +5e-5 | +1.3 / +1.4e-4 |
| CROSS | −2.9e-5 (−1.6) / −2.7e-5 (−1.3) | +1e-4 | +1.9 / +2.0e-4 |

- Values are seed 1 / seed 2. Genuine-move value is unchanged with error 0.
- **Spurious reassignments are a hedge against deterministic error.** Given an accurate afterstate, the energy score
  penalizes them, so it is not unsuitable in principle.

**5. Where the error comes from** (`biasnoise`, cells with no mob in root or futures):

| | bright eval 104 | darker held-out 96 |
|---|---|---|
| real-future spread (nuisance) | 0.0018 | 0.046 |
| learned afterstates' bias | 0.014-0.026 (operators lowest) | 0.046-0.061 |
| **exact copy of the current token** | **0.048** | 0.122 |
| hard CorrT copies | 0.028-0.038 | — |

- **Persistent content does not have a persistent token.** Contextual drift (~0.05 RMS/step) is systematic and
  predictable, partly modelled by CorrT's soft blend, and makes exact copying worse.
- "Select instead of blend" is therefore falsified model-free.

**6. The chance outcome is recoverable label-free** (`posterior`). Parameter-free local search (gain > 0, greedy one-to-one)
against the observed successor, scored vs T*:

| afterstate searched | recall | precision | false swaps per frame |
|---|---|---|---|
| unprivileged operator afterstates, bright eval | 0.78-0.80 | 0.95-0.99 | ≈ 0 |
| D parent, bright eval | 0.79 | 0.71 | — |
| exact-copy afterstate, bright eval | 0.93 | 0.06 | 3.3 same-terrain (drift again) |
| operators, darker held-out | 0.53-0.58 | 0.24-0.25 | — (night nuisance) |

**Conclusions:**
- Localized scoring is rejected in this target space and model.
- The objective's weakness is conditional: the energy score hedges deterministic error with spurious spread, and the error is
  contextual token drift.
- The proposed next comparison is inferred-chance training (Stochastic MuZero split: point loss through a label-free inferred
  reassignment, likelihood on the chance head), preregistered in `TRANSPORT_INFERRED_PREREG.md` and **not launched**.
- **Representation limits remain:** cross-terrain backgrounds (5.2% of moves, 100% wrong endpoint terrain), contextual drift
  of persistent content, and night nuisance (posterior precision 0.25 in dark views).

## 2026-10-15 — S5–S6 reconciliation and optimizer matching correction

- The preregistered direct/paired3k→6k continuation was held before launch. Both
  checkpoints remain at3k; original negative results and fresh reservation remain.
- S6's signed destination write is a content-creation bypass: at decoded valid
  movement destinations, reader-score contributions are own packet0.413 versus
  source arrival0.070. These are additive readout contributions conditional on
  final normalization, not physical-content or causal-effect percentages.
- Qualification to the S6 curve reading:1.5k→3k also added61 valid movements
  (52→113/896, +6.81pp [2.46,11.16]), alongside absence27→93 and duplication
  162→85. Coordination improvement is unresolved; convergence is not established.
- Qualification to “nothing in the loss”: the actual token objective penalizes
  marginal-preserving endpoint shuffles by0.002270 [0.001341,0.003205], while
  validity falls100%→52.59%. Score sensitivity does not establish optimization
  pressure sufficient to learn coherence.
- **New CPU audit:** saved AdamW groups differ. Direct decays163 parameter tensors
  at0.01; paired decays149 and exempts18 shared Mamba tensors (72 scalars).
  Deepcopy removed direct parameters' `_no_weight_decay` attributes. Its outcome
  effect is unmeasured; architecture-only attribution is therefore incomplete.
  Both use final LR1e−4, betas(0.9,0.999), eps1e−8. Historical states are preserved.
  Evidence: `oracle_v1/paired_event/diagnostics/optimizer_group_audit.json`.
- Full evidence reconciliation and proposed, unimplemented content/operator
  comparison: `20261012_oracle/PROVENANCE_ASSESSMENT.md`. Original S0–S6 audit
  entries and historical recommendations below are retained.

## 2026-10-10 — PROGRAM (predeclared 09:10, before any new code): why does the objective fight us? Diagnosis only

**Mandate (user, 10-10):**
- the purpose this week is to UNDERSTAND what ties the failures together, not to fix them;
- no intervention is tried until the cause is measured and the literature on that cause is swept;
- the GPU and CPU are used continuously.
- The user's frame: a world can fail in its **encoder** (do the tokens hold the needed information, stably?), its
  **dynamics** (does it learn the conditional distribution of the next state, random parts included?), or in
  **imagination** (do self-fed rollouts stay consistent?).
- I add **decisions:** can heads read correct, calibrated signals from imagined states?

**My correction of the record:** D (E21) was chosen from one measurement (the hits' gradient ratio) plus a published loss. Why the
objective fights us, and what it must achieve, were never established. This program does that.

**Working hypothesis (to be TESTED, not assumed):**
- a per-element token-space loss is a currency that mixes three kinds of variation:
  1. variation irreducible from the world's inputs (environment RNG, terrain never seen, hidden timers);
  2. change in the game state;
  3. variation that is not state (lighting, sleep rendering, context bleeding between tiles);
- it allocates gradient by token-distance magnitude, not by learnability or by relevance;
- each past fix moved that allocation for one fact and away from others.

**Predictions, each of which can refute it:**
- **P-A (encoder):** a substantial share of a cell token's variance, and of its frame-to-frame change, is not explained by the
  cell's own visible state.
  - Refuted if own state explains ≥ 90% of both.
- **P-B (dynamics):** ≥ 30% of the world's teacher-forced squared error sits on content irreducible from its inputs (RNG floor +
  entering terrain never seen in the window), and the learnable remainder concentrates on rare state changes.
  - Refuted if the irreducible share is < 10%.
- **P-C (gradient):** the backbone gradient from irreducible content dominates the gradient norm but is inconsistent across
  batches (noise), while the gradient from rare learnable changes is consistent but small.
  - Refuted if rare-change consistency ≤ irreducible consistency, or if the irreducible norm share < 25%.
- **P-D (training dynamics):** across matched L1 (M6) and D-loss (D from scratch, stage 1) snapshots, each loss improves the
  classes it allocates more to, with stalls or regressions elsewhere.
  - Refuted if one loss improves every class at least as fast as the other.

**Part I — Scoreboard, the user's request (breadth of D / E22MS, not health only).**
- **Variants:**
  - M16 (L1), D continued, D from scratch;
  - E22MS: Δ zeroed (its decoder alone), 4 prior samples, and the posterior's codes of the TRUE transitions (matched-code
    reference within the SAME decoder; Codex's lineage point).
- **Data:** the diagnosis futures (1,002 held roots, 143 episodes, 5 real futures per root under the same actions, 17 actions ×
  4 keys from each root; visible and hidden simulator state at every frame).
- **`rollout_export.py` (GPU):** saves every variant's self-fed 16-step rollout, its teacher-forced step at every depth, the 17
  one-step successors and the corrt routing weights.
  - Teval's protocol: 4 context frames, window min(4 + k, 15).
  - E22MS samples use e22_imagine's RNG order, so its summaries must reproduce e22_imagine exactly; the deterministic ones must
    reproduce teval's per-root errors exactly. Either mismatch stops the readouts.
- **`scoreboard.py` (CPU, on the saved rollouts):**
  - teval's errors (one-step classes, depths);
  - the 5-future decomposition (floor / excess, per class; energy score for sampled variants, a proper score against the 5
    real futures);
  - facts by depth for EVERY visible fact (17 tile classes, zombie / cow / skeleton / arrows, health / food / drink / energy,
    facing, sleep, light, 12 inventory items; ridge probes fitted on true TRAIN-seed tokens, read on TEST seeds);
  - a mob census by type and depth vs the 5 real futures;
  - scroll fidelity (missed / false scrolls, first wrong step);
  - terrain persistence on re-entry; HUD dynamics in distribution (event rates vs the real futures, Codex's real-key control);
  - distance to the real-token manifold by depth.
- **Decisions:** `check_h16_traj.py` on D from scratch (`--window 15`) and E22MS (`--e16 --window 15`); M16 exists (trajectory
  0.657; one real future 0.679).
  - The prior's end head has the terminal-position bias (Codex): its readings are reported, not judged.

**Part II — Diagnostics on the hypothesis:**
- **Encoder (P-A), CPU, true tokens + simulator state:**
  - A1 information: probes of every fact from one token / the 3 × 3 neighbourhood / the frame, linear and MLP (the ceiling any
    world can express);
  - A2 target composition: per-cell token variance and frame-to-frame change, explained by own visible state / neighbours' /
    global (light, sleep) / residual (held R², fitted on TRAIN seeds);
  - A3 currency alignment: token distance between cells with the SAME vs DIFFERENT state, per fact type. How well does
    token-space error separate state errors from non-state variation?
- **Dynamics (P-B):**
  - B1 floors: the RNG floor per class and fact from the 5 futures (depth 1) and the 4 keys × 17 actions; epistemic entering
    cells = entering cells whose world position never appeared in the input window (true scroll offsets);
  - B2 error budget of each world's teacher-forced step, per class and fact: RNG floor + unseen terrain + learnable remainder,
    in L1 and in squared-error currency.
- **Gradient and training dynamics (P-C, P-D), GPU, held diagnosis windows:**
  - B3 per-class learning curves (excess over the floor; fact accuracy) for M6 (L1) vs D from scratch stage 1 at 6k / 12k / 18k /
    24k / 30k / 36k;
  - B4 per-class backbone gradient: norm ratio, cross-batch consistency ‖Σg‖ / Σ‖g‖, cross-class cosine, at the same snapshots
    under each world's own loss.
- **Imagination (axis 3):**
  - C1 exposure: per-step error from imagined vs true windows, vs the input's distance from the real manifold;
  - C2 substitution at depth 16 (entering cells, HUD, player, mob cells, the move-target tile) for D from scratch and E22MS.
- **Order:** Part I exports (GPU) and A1-A3 (CPU) start now; B1-B2 and the scoreboard read the exports; B3-B4 follow on the
  GPU; H16 last (longest).
- Every result is recorded against these predictions; the literature sweep follows the measured cause.

## 2026-10-14 — AUDIT (lead, Opus) of everything run 10-12 → 10-14, stage by stage

The user's request: dissect every stage that ran (mostly by the other agent, GPT Sol 6.1), check the raw files rather than
the summaries, run diagnostics where something does not line up, and commit stage by stage. Each sub-entry gives:
- what ran;
- what the raw outputs say;
- what I re-derived or newly measured;
- where the original reading holds, needs qualification, or is wrong.

### AUDIT S0 — 10-12 oracle-innovation test (my design and runs; results never recorded as a stage of their own)

**What ran:**
- four arms, 3,000 updates each, batch 16, E22 long recipe on support-v2 rows (`oracle_world.py`);
- oracle arms receive the TRUE next-frame mob layout per tile through a zero-initialized Linear(4 → 192);
- L1 arms start from M16 and D arms from D scratch;
- evaluation on 200 DEV-split roots (`oracle_eval.py`; `oracle_v1/eval_*.json`).

**Raw results** (cow census / recall / precision / arrival recall / source residue; reader = AllProbes):

| arm | teacher-forced depth 1 | self-fed depths 9-16 (all roots) |
|---|---|---|
| TRUE tokens (reader ceiling) | 1.06 / 0.98 / 0.93 / 1.00 / 0.00 | same |
| control L1 (from M16) | 0.92 / 0.51 / 0.55 / 0.25 / 0.57 | 0.51 / 0.05 / 0.11 / 0.04 / 0.04 |
| **oracle L1** | **0.57 / 0.54 / 0.95 / 0.25 / 0.02** | 0.06 / 0.03 / 0.60 / 0.04 / 0.00 |
| control D (from D scratch) | 1.93 / 0.65 / 0.34 / 0.49 / 0.72 | 1.03 / 0.08 / 0.08 / 0.05 / 0.06 |
| **oracle D** | **1.05 / 0.96 / 0.91 / 0.98 / 0.00** | **1.50 / 0.92 / 0.62 / 0.93 / 0.04** |

**Against the predeclared readings:**
- **Q1 (conditional completion in BOTH oracle arms): FALSE.**
  - The D arm passes every criterion; it sits at the reader's own ceiling.
  - The L1 arm fails: census 0.57, recall 0.54.
- **Recursion (oracle D self-fed: census within 20% and residue ≤ 2% at every depth on aligned steps): FALSE.** Cow census on
  aligned steps is 1.20 at depths 2-4 and 1.49 at 9-16; residue is 0.035 at 9-16. Recall is kept (0.94), so the failure is
  extra detections, not lost mobs.

**New in this audit — why the L1 arm fails** (`audit_s0_arrivals.py`; probe cow score, teacher-forced depths 1-16, 958 arrival /
964 vacated / 746 stay cells):

| cells | TRUE | oracle L1 | control L1 | oracle D |
|---|---|---|---|---|
| arrival (cow now, not before) | 0.85 (98% > threshold) | **0.23 (25%; median ≈ 0)** | 0.20 (23%) | 0.79 (94%) |
| vacated (cow before, not now) | 0.00 (0%) | **0.04 (5%)** | 0.43 (50%) | 0.02 (2%) |

- With the oracle, L1 learned to ERASE moved cows at their source (vacated-cell detections 50% → 5%) but not to DRAW them at
  their destination (arrivals unchanged, 23% → 25%). A cow move becomes a disappearance.
- The training log agrees: zeroing the oracle costs the L1 arm only 6.4% of its loss at update 3,000, against 28% for D.
- That L1 can HOLD arrivals once learned is shown by the other agent's L1 continuation from the oracle-D world (10-12 D-loss
  follow-up: 838 / 870 late recursive cow TP).
- So the L1 failure is in LEARNING to create under L1 from M16 within 3,000 updates. The mechanism is NOT isolated.
  - A consistent hypothesis: L1's bounded per-element gradient gives the ~1 arrival cell per frame the same weight as any
    other element, while arrival needs a relational routing decision (which neighbour the cow came from) that clearing does
    not.
  - The same allocation pattern as P-C.
- **Recursion detail** (`recursion.py`, my 10-12 run, aligned steps): oracle-D false cows per true cow 0.09 → 0.25 → 0.46 →
  0.55 from depth 1 to 9-16.
  - Their share that persists from the world's OWN earlier false detection rises 0 → 0.30 → 0.40 → 0.55.
  - False cows score 0.28-0.35 on the probe vs 0.77-0.88 for true ones: faint, self-sustaining residue.
  - For the control D arm, 87-91% of its late false cows are such persisting false detections.
- My first token-projection (snap) run OOMed at 12:14 (its script is preserved by the other agent as
  `snap_eval_failed_20261012_1214.py`); S1 below is its completed version.

### AUDIT S1 — 10-12 token-projection follow-up (other agent; my snap idea, completed)

**What ran:**
- 200 DEV roots, oracle-D / control-D / TRUE;
- every fed-back map tile replaced by its L2-nearest token among 60,000 real tiles;
- four modes: raw, projection at readout only, the output BEFORE projection under projected feedback, and the projected output.

**Verified from `snap_{TRUE,oracle_deltairis,control_deltairis}.json`** (depths 9-16; all reproduce the NOTEBOOK):

| oracle-D mode | cow false / true | cow recall | zombie recall | skeleton recall |
|---|---|---|---|---|
| raw | 0.569 | 0.924 | 0.842 | 0.507 |
| readout only | 0.244 | 0.746 | 0.599 | 0.201 |
| **output before projection, under projected feedback** | **0.289** | **0.905** | 0.731 | 0.292 |

TRUE tokens projected: cow recall 0.961 → 0.818, zombie 0.902 → 0.643.

**Holds:**
- Cleaner fed-back history halves the next outputs' false cows at nearly unchanged cow recall (the feedback effect is real).
- L2 nearest-token projection is not content-preserving: the chosen wrong token is closer than a semantically correct one in
  64 / 64 sampled losses.

**Added by this audit:**
- **The metric failure is the target-space problem again.** In Raw token space, L2 distance is dominated by context: own
  state explains R² 0.59 of a token, and lighting / position / neighbours most of the rest (encoder_diag, 10-10). So "nearest
  real token" picks the right context with the wrong content. A valid-state projection would need a content-defined metric,
  which these tokens do not provide.
- **Completion is copy-driven even with the oracle.** Projection erases some real zombies from the fed-back history, and
  although the oracle still gives every zombie's position, zombie recall falls 0.84 → 0.73 (skeleton 0.51 → 0.29). The world
  draws a mob mainly by copying it from its history, not from the occupancy signal. S2's decoder decomposition measures the
  same copy dependence from the other side.

### AUDIT S2 — 10-12 D-loss causal follow-up, readout controls, decoder decomposition (other agent)

**What ran:**
- three continuations (D, L1, D without its max term) of the oracle-D world, +3,000 updates each, with identical batches and
  the oracle still supplied;
- evaluation on the 200 DEV roots;
- CPU diagnostics: fresh readout heads, decoder decomposition, model-free conditional optima.

**Verified from `loss_intervention/eval_u*/analysis.json`** (self-fed, depths 9-16, frozen reader; true cows 870):

| update | D: cow FP / TP | L1: cow FP / TP | no-max: cow FP / TP | D zombie FP |
|---|---|---|---|---|
| 0 (shared start) | 495 / 804 | 495 / 804 | 495 / 804 | 270 |
| 500 | 477 / 771 | 387 / 793 | 341 / 786 | 257 |
| 1,500 | 293 / 789 | 235 / 798 | 368 / 821 | 187 |
| 3,000 | **227 / 805** | **161 / 838** | 213 / 828 | 70 |

The same reader on the TRUE future tokens gives 86 FP: that is the instrument's own floor.

**What this audit adds:**
1. **Nothing had converged.** Every arm's recursive false positives were still falling steeply at 3,000 updates. The S0
   recursion failure was at least partly an adaptation budget: the excess over the reader's floor fell from 409 to 75 under
   L1 and to 141 under D.
2. **The frozen reader overstates the residue.** `calibration_u3000_D.json`, 50 TEST episodes, depths 9-16, 215 true cows:

   | reader | TP / FP |
   |---|---|
   | original frozen reader | 205 / 44 |
   | **fresh equal-capacity head fit on REAL tokens only** | **203 / 17** |
   | head fit on generated tokens | 200 / 10 |

   - A head that never saw a generated token cuts false positives by 61%. The original count-matched thresholds fire on faint
     residue.
   - So **given the true outcome and 6,000 total updates, a continuous JEPA world's 16-step recursion is close to coherent
     for mob occupancy**: 17 false cows per 215 true, with recall 0.94.
3. **The residue that remains is carried by the copy path.**
   - Decoder decomposition at 3,000 updates, both-camera-aligned late cells: replacing ONLY the copied candidates with real
     previous tokens cuts cow FP 192 → 82 (D), 133 → 83 (L1).
   - Replacing the generator or the routing alone does nothing.
   - Faint false content in a generated frame is copied into the next one, which matches S0's "self-sustaining" share.
4. **The model-free conditional optimum confirms the whiteboard's regression condition** (`conditional_stat/dopt.py`). Grouping
   real futures by exact next mob occupancy, the full-D optimum gives cows 300 / 302 TP with 10 FP (the real futures: 299 / 302
   and 14 FP). Once the outcome is fixed, a point loss no longer averages incompatible futures.

**Holds:**
- The other agent's reading that D's maximum term is not the main cause (no-max ≈ D by 3,000; its early benefit faded).
- L1 vs D is a tradeoff, not a ranking.

**Answer to the user's question 1 (10-12):** supplying the correct stochastic outcome removes the averaging failure in one
step (S0, at the reader ceiling). With more training and a calibrated reader, it also removes most of the recursive one.
What remains is the copy path propagating faint residue.

### AUDIT S3 — 10-12/13 generated-prefix completion (other agent)

**What ran:**
- two continuations of the S2 D world (+3,000 updates), each with 0.5 teacher loss + 0.5 loss at one extra endpoint;
- the endpoint's context is real (teacher arm) or the world's own detached generated rollout of 1-15 steps (prefix arm);
- both receive the TRUE oracle;
- evaluation on the 200 DEV roots.

**Verified:**
- The other agent's table (late self-fed, frozen reader): cow FP 245 teacher vs 838 prefix.
- With generated-fitted heads: 6 vs 34 FP on 50 TEST episodes.
- Drink / energy HUD gains for the prefix arm.
- Factual frame MSE worse (+12.5%).
- Its cause of the prefix arm's extra false cows was "not isolated".

**New in this audit (`audit_s3_oracle_reliance.py`, `audit_s3_alignment.py`):**
1. **The prefix arm learned to trust its history over the oracle.** Teacher-forced on real inputs with the oracle present, both
   arms are the same (arrival cells: 0.81 vs 0.83 mean cow score). With the oracle ZEROED (which tells the world "no mob
   next"), the share of cells where the arm still draws the cow copied from its history is:

   | cells | shared start | teacher arm | prefix arm |
   |---|---|---|---|
   | stay | 0.29 | 0.17 | **0.40** |
   | vacated | 0.15 | 0.09 | **0.20** |

2. **Not camera misregistration** (my first hypothesis, refuted). Late self-fed false cows per step:
   - prefix 0.56 on camera-aligned steps vs 0.40 on misaligned ones;
   - teacher 0.16 / 0.12.

   The excess is on aligned steps.
3. **It is neighbour-copy spreading.** At late false-cow cells, corrt's routing weights (self, 4 neighbours, generator):
   - prefix 0.50 / **0.37 total neighbours** / 0.14;
   - teacher 0.68 / 0.12 / 0.21.

   68% of the prefix arm's aligned false cows (472 / 690) persist from its own previous detection. The prefix arm pulls mob
   content from adjacent cells into cells the truth says are empty, without clearing the source: duplication.
4. **Mechanism of the training-side cause: a hypothesis, not tested.** In a generated context a mob can sit one cell away
   from where the true oracle says it is next. Copying the neighbour's mob into the oracle-marked cell is then rewarded, and
   that habit produces duplicates at evaluation.

**Reading:** exposure to generated history did not teach robustness. It taught more copying from history, so it amplifies
the copy-path residue S2 localized. The drink / energy gains are real but small; the mob coherence cost dominates.

### AUDIT S4 — 10-13 learned innovation (other agent): the posterior is a mean pool; that is why it learns nothing

**Verified from raw files:**
- **Main study, both arms:**
  - one joint code over all 200 DEV successors, and one code per root over the 32 RNG keys (`content/codes_*_u3000.pt`);
  - all six factors pick the same choice: (6,6,6,6,6,6) for capacity, (2,2,2,2,2,2) for rate;
  - shuffling the current code or using the prior mode changes the output by exactly 0.0;
  - held NLL 0.000245 / 0.0000617 bits.
- **Optimization control, final table exact:** MSE/floor 1.0513 (none), 1.0524 (hard), 1.0472 (soft), 0.0152 (assigned IDs);
  within-root output variance soft 1.05e-5, assigned 0.980.
- **Not recomputed:** the TEST-episode D deltas. They are deterministic-capacity effects of a constant code, so they don't
  bear on the mechanism.

**New in this audit (`audit_s4_channel.py`, 26 control cases, 4 FIT roots):**
1. **The information is there.** A root's futures differ on ~3 tiles: half the within-root target variance sits in 3.35
   tiles per case on average, and one root has 99% on one tile. A flat linear classifier on the posterior's own input
   (per-tile LayerNorm of [belief, current, successor − current], flattened) memorizes all 26 IDs: **100% at 500 updates (E).**
2. **Pooling erases it.** The same linear classifier on the tile MEAN stays at **15-19%** (E).
   - The posterior's attention pool behaves as that mean pool. Attention mass on the tiles where futures differ is 0.0413,
     equal to uniform (3.35/81), in 5 fresh posteriors AND in both trained ones (D).
   - The within-root share of variance is ~1% in the input and in the projected tiles, then **0.02% after pooling** (D).
   - The six factor slots are near copies: pairwise cosine 0.96-0.998. In the trained arms, all 6 slots share one argmax on
     every case (D).
   - The query init (std 0.02) makes the attention logits nearly constant over tiles.
3. **Training never repairs the pool.**
   - The final soft and hard posteriors' logits differ by at most 0.012 / 0.0024 within a root (A).
   - Even under direct cross-entropy at 10× the control's learning rate, the attention posterior reaches 38% at 500 updates
     and **plateaus at 62-73% from 1,500 to 3,000 updates** on 26 fixed cases (B, B3000).
   - The joint unsupervised path gives the posterior far less signal than this, so it never escapes.
4. **The zero-initialized adapter is not the binding cause.**
   - Joint training as the control's soft arm, with the adapter at PyTorch default init (C1), or with that plus unit-std
     code values (C2): MSE/floor at 800 updates is **1.0648 / 1.0661**, on the no-channel / soft curves (1.0624 / 1.0598).
   - Within-root output variance stays 2e-5 throughout.
   - The assigned-ID arm uses the same zero-init adapter and broke below the floor by update 600 (0.959). The decoder side
     was never blocked.

**Reading:**
- The learned-innovation negative is a property of this posterior: a global pool over 81 tiles of a difference that lives
  on ~3 tiles. It is not evidence against learned innovations, and not about hard vs soft codes.
- The other agent's "learned-interface / coadaptation problem, upstream cause unidentified" is now localized: the inverse
  model cannot see where the futures differ.
- This agrees with LPWM's choice of *per-particle* latent actions: innovations inferred locally, not from a global summary.
  That is a design fact for next steps, not a tested fix.

### AUDIT S5 — 10-14 direct stochastic completion (other agent): each tile flips its own coin

**What ran:**
- Two arms from the D parent, both 3,000 updates: point objective vs first-power energy score on the 35 interior tokens.
- Completion is one attention + FF block over belief + noise (4 local dims per tile, 4 shared per frame). It produces a
  generated token and offsets to the routing logits.
- Output = LN(soft mixture of the 5 copy candidates and the generated token).
- Evaluation: 104 bright DEV roots × 64 samples; strict physics on 14 isolated sources (8 cow, 3 zombie, 3 skeleton).

**Verified with independent code (`audit_s5_direct.py`, my own source selection, decoding and event logic), all exact:**
- Energy arm: valid 64.62%, absent 15.18%, multiple 20.20%, source+destination duplicate 15.62%; cow stay 59.77% / move
  8.20%; 3.071 distinct valid outcomes per source; scored joint ES 0.024439.
- Point arm: ES 0.031417.
- Real fan: 100% valid, cow move 41.02%, split ES 0.021161.
- Dependence, against an exact independence baseline (expected validity if the 5 cross cells were independent with the same
  marginals): 2.46pp, vs their 2.73pp from 128 shuffles; real 45.84pp.

**New in this audit:**
1. **Not budget-limited on the binding measures.** Milestones the study did not report (energy arm):

   | update | ES | valid | dependence | cow valid move |
   |---|---|---|---|---|
   | 500 | 0.02567 | 66.2% | 3.9pp | 2.7% |
   | 1,500 | 0.02447 | 61.8% | 0.9pp | 6.5% |
   | 3,000 | 0.02444 | 64.6% | 2.5pp | 8.2% |

   - The distribution score converged by 1,500; coordination never developed. Cow movement is the only measure still slowly
     rising.
   - The training objective is flat from ~1,000 updates, within single-batch noise.
2. **Mechanism: each tile decides from its own noise.** Probe: 14 sources × 256 draws; resample the 4 local noise dims at ONE
   tile, everything else fixed.

   | resample at | own cell's flag flips | other cross cells flip |
   |---|---|---|
   | source | 29.6% | ≤0.06% |
   | one destination | 9.5-16.1% | source 0.2-0.4%, others ≤0.14% |
   | a far tile | — | ~0 |
   | shared noise only | source 12.6% | destinations 4-6%, all together |

   - The shared noise is a common mode: it moves every cell the same way, not the anti-correlation conservation needs.
   - The one attention layer could carry noise between tiles, but the trained model does not use it.
3. **Copy bookkeeping (same probe).**
   - The source's self-copy weight is bimodal: keep has mean 0.65-0.97, sd 0.3-0.46.
   - Destinations almost never copy the source in: summed move weight has mean 0.002-0.15. When the source lets go, nothing
     receives the mob.
   - corr(keep, Σ move) = −0.12, where conservation needs strongly negative.
   - Destination mobs come from the generator path (weight 0.04-0.23 at the cross), driven by the destination's own noise.
   - So "absent" = the source clears and no destination draws; "multiple" = the source keeps and a destination draws. These
     are two independent coin flips.
   - Their own diagnostic is consistent. Real futures with cells shuffled independently give 55% valid / 24% absent / 21%
     multiple / 16% duplicate; the model gives 65 / 15 / 20 / 16. The training metric charges only **+5.6%** for that complete
     loss of dependence (`dependency_sensitivity/result.json`, 104 roots).
4. **A free replicate.**
   - The paired study's "direct" control used the identical initialization and window/noise ledger, yet diverged
     numerically: the update-1 loss differs in the 7th digit, and weights differ by 5.2% relative L2 at 3,000 (cross-process
     GPU nondeterminism).
   - It is therefore an independent run of the same configuration, and it reproduces the mechanism: own-tile flips 12-27%,
     cross-tile ≤0.5%, corr(keep, move) −0.15.
   - It gives run-to-run spread: valid 64.62 vs 62.50%, cow move 8.20 vs 6.45%, dependence 2.46 vs 2.83pp.
   - The studies' bootstrap intervals cover root/episode sampling only, not this training-run spread.

**Reading:**
- The "why" the other agent left open ("does not identify why optimization failed to learn that dependence") is now
  localized. The trained generator makes its stochastic decisions per tile from per-tile noise, and the objective barely
  charges for missing joint dependence.
- Matching marginals through independent per-tile draws is the cheap solution the score accepts. Nothing in the loss pushes
  the model off it.
- That the objective's insensitivity *caused* this choice is consistent with all the measurements but has not been tested by
  intervention.

### AUDIT S6 — 10-14 paired event completion (other agent): moves are created by the destination's own packet

**What ran:**
- Two arms from the D parent, 3,000 updates each: the direct energy generator (retrained) and a paired editor.
- The paired editor gives EVERY tile a packet: a gate, soft routes over (self, 4 neighbours), a removal write
  −g(1−p_self)·R at the tile itself, and arrival writes g·p_j·A at its neighbours. Output = LN(parent reference + writes).
- R and A are free contextual decoders. Same evaluation as S5; a post-hoc failure localization covers hard routes,
  gate masking and pre/post-LN variance.

**Verified with independent code (`audit_s5_direct.py paired`), registered table and milestones exact:**
- At 3,000 updates:

  | | direct | paired |
  |---|---|---|
  | valid | 62.50% | 68.42% |
  | duplicates | 17.19% | 9.49% |
  | cow valid move | 6.45% | 9.57% |
  | ES | 0.024295 | 0.024876 |
  | dependence (exact independence baseline) | 2.85pp | 10.24pp |

  Their shuffle estimates are 2.83 / 10.27pp.
- Paired milestones (500 / 1,500 / 3,000):
  - valid 90.18 / 70.42 / 68.42%, cow move 0 / 6.25 / 9.57%;
  - ES 0.031649 / 0.027167 / 0.024876.
- The entering-cell numbers are internally consistent: 164 low-reference cells, all entering; 81.28% of post-LN variance; 329×.
- **Not re-verified:** the single-packet swap counts, the hard-route table and the second-reader figures.

**New in this audit:**
1. **The gains against run-to-run spread.**
   - The "direct" arm here is the S5 replicate: identical configuration, numerically diverged run. Two runs of the same
     direct configuration give valid 64.62 / 62.50% and cow move 8.20 / 6.45%.
   - Against them, paired is +3.8 / +5.9pp valid and +1.4 / +3.1pp cow move: the size of the replicate spread.
   - The dependence gain, +7.4 / +7.8pp, is robust.
   - The study's intervals cover root sampling only, not training-run spread.
2. **Curves (the 3k question).**
   - The paired arm learns more slowly: its ES at 3k is about where direct was near 1k, and is still falling
     (0.0272 → 0.0249).
   - Duplicates fall (18.1 → 9.5%) and cow movement rises (6.3 → 9.6%).
   - But **dependence plateaued at 1,500** (11.3 → 10.2pp) and **absence rose** (3.0 → 10.4%). The last 1,500 updates turned
     duplicates into absences, not into valid moves.
   - A 3k → 6k continuation would likely finish the ES descent. The coordination measure was no longer moving.
3. **Mechanism of the interior incoherence: exact score attribution** (`audit_s6_paired.py attrib`).
   - Method: 14 sources × 256 draws. LN has no affine, so given each token's own LN scale the real reader's score is exactly
     additive over reference / own packet / the source packet's arrival / other arrivals. Reconstruction is asserted.
   - Results (mean contributions):

     | event / cell | reference | own packet | source arrival | other |
     |---|---|---|---|---|
     | valid move, destination | 0.009 | **0.413** | 0.070 | −0.002 |
     | multiple, destination | 0.035 | **0.436** | 0.077 | −0.003 |
     | absent, best destination | 0.028 | 0.044 | 0.044 | −0.001 |
     | move, source | 0.393 | −0.357 | — | −0.077 |
     | stay, source | 0.861 | −0.030 | — | −0.006 |

   - **Even valid moves are not transfers.** The destination's mob is created by the destination's OWN packet: its
     "removal" write is a free decode, used as creation. The source's arrival, the channel the design meant to carry the
     mob, supplies 14% of the score.
   - Routes carry no direction: argmax matches the decoded move **24.1% (chance 25%)**, entropy 1.41 (uniform over 4 =
     1.39). Arrivals are split ~4 ways and too weak to create anything. Absent draws are the source clearing while no
     destination's own packet fires.
   - Noise probe (`audit_s5_direct.py noise paired`):
     - A destination's own noise flips its flag 10-15%; the source's noise flips destination flags only 1.6-3.3%.
     - Direct, by comparison: ≤0.06%. That coupling is the dependence gain.
     - Tied weights hold exactly (removal mass = Σ arrival mass, corr −1), but the content they carry is not what draws the
       mob.
   - **Reading:** pairing tied one packet's weights, but every tile kept a full-strength per-tile creation path driven by
     its own noise. That is the same escape route as the direct model (S5).
4. **Entering-cell fault: exact cause** (`audit_s6_paired.py entering`). The cause is the parent's routing.

   | entering cells | count | weight on scroll candidate | generator | self | reference RMS |
   |---|---|---|---|---|---|
   | candidate = **zero padding** outside the 9×9 grid | 164 | **0.979** | 0.009 | 0.011 | 0.023 |
   | candidate = a HUD token (rows 7-8) | 72 | 0.0005 | 0.385 | 0.540 | 1.19 |

   - The parent learned to avoid copying HUD tokens but copies the zero padding.
   - Deterministically that is harmless: its final LN discards scale, so the output is just the direction of the small
     gen/self remainder.
   - The paired editor's stochastic edits are added outside the mixture at their own scale, so against a ~0 reference LN
     amplifies them to full scale.
   - The direct model's noise sits inside the mixture and is scaled down with it, so it is not amplified (edge post/pre
     ratio 0.52).
   - This concerns only unscored edge cells, as the other agent said. It is a separate fault from the interior failure in 3
     (they asked for exactly this separation). For recursion it would feed back.

**Reading across S5-S6:**
- Both continuous generators fail coherence for one structural reason. Each tile can create or erase content from its own
  noise, and nothing in the loss (S5: +5.6% for total loss of dependence) makes that path more costly than a coordinated
  transfer.
- The pairing constrained bookkeeping weights, not the content path, so the per-tile path stayed open.

### 10-13 LEARNED-INNOVATION FEASIBILITY — predeclared before implementation/training

**Question:** can a compact latent channel learned only from observed transitions improve conditional completion, and can
an observation-conditioned prior sample the same useful outcomes? This advances the whiteboard's mechanism study beyond
simulator-provided occupancy. It does not assume four-way uncertainty separation will emerge from a posterior/prior.

- **Shared parent:** the existing unconditioned D control (`oracle_v1/worlds/control_deltairis_...u3000_b16.pt`), whose
  oracle projection is identically zero and whose training used no occupancy conditioning. Same frozen Raw tokens,
  corrt/fmamba world, TRAIN 64-frame ledger, 16-frame teacher windows, seeded batches and 3,000 updates at batch 16.
- **Three arms:** no channel; six learned categorical factors with 8 choices each and no rate penalty; identical channel
  with a conditional rate penalty 0.001. The hard channel has an **18-bit maximum per transition**, not an established
  entity interpretation. Cross-attention learns the factors; no object identities, movement masks, tile classes, health
  events, simulator layouts or keys enter the training objective. Terminal metadata only selects factual windows.
- **Mechanism:** posterior sees causal backbone state plus current/next observed tokens. Completion receives only hard
  discrete codes through a zero-initialized adapter, with no continuous future bypass. Prior reads the world's native
  spatial/Mamba features and predicts factors autoregressively; it avoids E22MS's 192→8 per-tile input bottleneck.
  Prior CE learns detached posterior assignments. The regularized arm additionally pays the assignments' conditional
  negative log probability; this is a rate-distortion surrogate, not an ELBO or a measurement of environmental entropy.
- **Separating readings:** identical initial outputs; future-input isolation in the prior; current-code zero/shuffle controls;
  posterior completion versus ancestral prior mode/samples; original and generated-fitted held-episode readouts; actual
  code use and held conditional rate. Evaluate saved milestones 0/500/1,500/3,000, then freeze posterior/world and refit the
  prior if its moving targets limit fitting. A posterior failure calls for capacity/optimization diagnosis; a posterior
  gain with a prior failure calls for prior/conditioning diagnosis. No unconditional second seed or actor launch.
- **Judgment:** existing 200-episode DEV panel is exploratory. Simulator fields are evaluation-only. Initial evaluation
  establishes paired factual/posterior completion and channel dependence; sampling quality must also be checked against
  multiple real outcomes rather than against one realized future. H16 is a separate recursive reading, not inferred from
  one-step success. Atomic model/optimizer/RNG/batch-ledger checkpoints and evaluation batches are required for resume.
- **Limits:** the frozen state still includes nuisance; 15-frame completion history still omits some timers. The channel
  may encode missing history, epistemic content or nuisance. Its actual content must be measured, not named by assumption.
  The unconditioned parent is a continuation reference, not a from-scratch test of the full whiteboard architecture.
- **Literature:** [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) motivates hard chance codes and separate
  afterstate/chance completion; [LPWM §A.4.3](https://arxiv.org/html/2603.04553v1#A1.SS4.SSS3) motivates learned inverse
  transitions and history-conditioned priors. LPWM also uses pixel reconstruction; we borrow the latent-context principle
  without claiming a reproduction. This pilot's conditional rate penalty addresses the previously unregularized
  posterior/code-learning path; whether it allocates capacity to consequential uncertainty remains a measured question.

**Launch and completed controls (10-13):** three matched arms are training to 3,000 updates. Interrupted 2→4 versus
uninterrupted 4 reproduces model and channel weights exactly; maximum optimizer-state difference is 1.09e−11. Sampler,
Torch/CUDA RNG and batch-ledger streams are exact. Mutating the last input frame changes earlier native causal features by
0; zero-initialized channel outputs match the unconditioned world exactly. Hard forward codes are exact one-hots, with a
live completion gradient after the adapter is activated. Proof: `oracle_v1/learned_innovation/validation.json`.
- The evaluation-only CPU fan is complete: **200 DEV roots × 32 independent keys**, at each root's recorded action.
  Every recorded-key replay has pixel maximum difference 0; re-encoding that branch reproduces the panel tokens exactly.
  This fan conditions on full simulator state; it does not measure uncertainty remaining under partial observation.
  Files: `learned_innovation/real_futures/fan.pt` and its per-root atomic records.
- Evaluation fits branch readouts on each arm's **posterior-completed recorded branches**, then freezes them for prior
  samples. Independently sampled branches are never assigned the recorded branch's label during head fitting.

**Early optimization diagnosis, update 500 (exploratory):** the monitored constant-code behavior is confirmed across all
200 DEV roots, not just one training batch. The capacity arm has 2 distinct joint codes; the penalized arm has 1. Swapping
the next frame changes 0.75% versus 0% of factor assignments (initially 35.08% for both).
- At initialization the completion gradient into posterior logits is **exactly 0**, because the adapter's output weight
  is zero, while the β=0.001 rate gradient norm is 4.39e−5 on the two-root control. This is an initialization defect in
  this pilot's learning signal, not a property of continuous JEPA or of the whiteboard hypothesis.
- At update 500, completion-logit gradient norms are 4.02e−7 (capacity) and 1.15e−9 (rate). The penalized arm's rate gradient
  is 7.56e−6, **6,548×** its completion gradient. Random hard codes change the predicted tokens by up to 0.0564/0.0518, so
  the decoder is not literally invariant to all codes; the learned encoder assigns almost constant ones.
- These measurements establish channel disuse and a badly imbalanced initial/early gradient. They do **not** isolate the
  full cause of the unpenalized arm's collapse. It starts using more codes on later monitored TRAIN batches, so the fixed
  comparison is retained through its declared budget. A constant-code negative cannot reject compact innovation.
- This relates to documented latent-channel collapse with strong decoders
  ([He et al., ICLR 2019](https://arxiv.org/abs/1901.05534)); our hard-code rate surrogate is not their VAE ELBO.
  Source/results: `20261012_oracle/innovation_channel_diagnosis.py`,
  `oracle_v1/learned_innovation/channel_diagnosis_u0500.json`. Any repair must separate channel optimization from prior
  fitting and preserve these initial results.

**Separating the observed RNG content (CPU):** `innovation_content.py` evaluates the learned posterior on all 32 real
successors of each identical root history/action, using exported native causal features. CPU/GPU code assignments on the
original branches agree exactly. Mob occupancy varies across keys in **155/200 roots**. At update 1,500 both arms have
**0/200 roots with any code variation across keys**, hence 0 empirical code entropy and 0 measured code/mob mutual
information on this fan. At update 500 the capacity arm has code variation on 1 root, but none of the 155 mob-variable
roots; the rate arm has none. The low prior CE therefore measures a predictable constant target, not learned chance.
K=32 under-resolves rare alternatives; the exact constant-code finding applies to these sampled futures.

**Optimization-control follow-up, predeclared before launch:** use four readout-FIT daytime roots and their distinct
observed successor tokens. The history/action repeats while the successor varies; no readout-TEST episode is trained.
Match four models: no code, learned hard code, learned soft probabilities, and assigned hard codes identifying each
distinct observed future. Same parent, initialization, optimizer and seeded case batches; full D loss on the current
successor. No prior or rate penalty. All models are free to train; no hypothesis about frozen-backbone capacity is added.
- The assigned-code arm tests whether completion can use a known compact identifier. It is a deliberately supplied
  positive control, not a learned innovation. The soft arm deliberately permits continuous future information and tests
  the hard-quantization/gradient path; it is not a deployable 18-bit proposal.
- Report error versus the exact empirical conditional-mean MSE floor, actual code use and the learning curves. Budget
  2,000 updates with an initial stop/review at 500. A failure of the assigned-code control prevents an inverse-encoder
  failure claim; a still-improving curve does not establish a negative. This overfit control provides no generalization,
  sampled-distribution or actor result. Outputs remain under `learned_innovation/optimization_control/`.
- The primary queue tests one-step completion/distribution first; it defers H16 if both current codes are constant or
  shuffling them produces identical output. This is a necessary code-use check, not a new capability gate.

**Completed primary result (10-13): this learning mechanism collapses; the compact-innovation hypothesis is unresolved.**
All three worlds completed 3,000 updates and 720,000 targets each. The independent CPU sampler reproduction matches the
final ledger exactly (`38f088d3...f4696cc30`); 12,784/48,000 sampled windows are terminal (26.63%). The 200-root evaluation
and 32-key content checks are complete.
- Both final posteriors assign **one joint code across all 200 original successors**, and **one code across all 32 RNG
  futures within every root**. All 155 mob-variable roots have constant codes. Empirical code entropy and code/mob MI are
  0 on the fan. Current-code shuffling changes the generated output by exactly 0 for both arms. The prior targets are
  trivial: held assignment NLL is 0.000245/0.0000617 bits for capacity/rate. These are not estimates of game entropy.
- On the 50 readout-TEST episodes, capacity posterior D minus control is **−0.000003 [−0.000602, +0.000552]**; its sampled
  whole-frame energy difference is **+0.000642 [−0.000312, +0.001639]**. Neither resolves. The rate arm improves D by
  **−0.001042 [−0.002188, −0.000070]** and whole-frame energy by **−0.000741 [−0.001417, −0.000082]**. These small exploratory
  prediction gains cannot be credited to learned stochastic innovation: its code is constant, posterior/prior output
  metrics coincide, and the adapter adds deterministic trainable capacity. No actor/decision benefit is established.
- **Measurement control:** deterministic and sampled methods now use the same repeated-root inference batch shape.
  Initial point-loss differences are exactly 0; energy differences are below 6e−18. The earlier unmatched-BF16-batch
  exports (`unmatched_batch_eval_*`) had a spurious initial D difference of 1.54e−5 and are preserved with the exact
  original evaluator source. They are excluded from the result. FP64 accumulation avoids spurious averaging differences.
- H16 was deferred by the predeclared necessary code-use check. A recursive run of a constant-code channel would test
  another deterministic adaptation, not innovation completion. The 26-case, four-FIT-root optimization control is now
  running toward its initial 500-update review; no inverse-encoder/capacity verdict is available before that control.
- **Interpretation:** the deployed prior cannot supply chance that its learned inverse encoder never encodes. The
  initial zero-gradient defect and early rate imbalance are measured, but do not establish the full upstream cause of
  the unpenalized collapse. The small control separates completion with supplied information from hard/soft inverse
  learning; it must succeed before a negative is used against the whiteboard hypothesis.
- Evidence: `learned_innovation/eval_u3000_h01/analysis{,_contract}.json`, `content/*_u3000.json`,
  `recursive_queue_decision.json`, `sampler_reproduction.json`; original/generated-fitted mob heads remain separate.

**Optimization control's 500-update review:** on 26 distinct futures, MSE divided by the exact conditional-mean floor is
1.0733 (no code), 1.0722 (hard posterior), 1.0713 (soft posterior), **1.0618 (supplied identifiers)**. None beats the
deterministic lower bound. The positive control is therefore not established, and a hard/soft encoder failure reading is
premature. Every curve is still improving from 1.5109 at initialization. Reviewed and resumed the same models/optimizers
and sampler to the declared 2,000-update budget; no new seed or architectural change. Source/data/scopes stay fixed.
Proofs: `optimization_control/review_u0500.json`, source-bound resumable `state.pt` and per-milestone outputs.

**Update-800 optimization reading (interim):** the supplied-code model now beats the exact deterministic-mean floor:
MSE/floor **0.75245**, versus no-code **1.06243**, learned-hard **1.06324**, learned-soft **1.05977**. Supplied codes identify
26 futures (at most 5 bits of information per case), rather than providing a complete spatial occupancy map. This
establishes partial future-specific continuous completion on this tiny support, not generalization or coherent H16.
The categorical completion path is therefore capable of using supplied distinctions; the joint learned interfaces have
not yet done so. Because soft learning also lags, blaming hard quantization alone is premature. Continue the fixed-budget
control; distinguish representability from inverse-encoder/decoder coadaptation, rather than declaring a JEPA failure.

**Completed 2,000-update optimization control (10-13): supplied distinctions work; learned hard and soft interfaces fail
on this tiny support.** The unchanged four-arm control finished its declared budget. Recomputed source hashes, contract,
exact target bytes and saved outputs agree. The four FIT roots contain 5/16/3/2 distinct futures, weighted uniformly as
26 training cases; this is deliberately an overfit/optimization diagnostic, not the environment's outcome frequencies.

| arm | final MSE | MSE / exact deterministic conditional-mean floor | own case is nearest real future |
|---|---:|---:|---:|
| no channel | 0.00504717 | 1.05126 | 4/26 |
| learned hard posterior | 0.00505249 | 1.05237 | 4/26 |
| learned soft posterior | 0.00502762 | 1.04719 | 4/26 |
| supplied hard outcome IDs | **0.000073114** | **0.015229** | **26/26** |

- The exact FP64 conditional-mean MSE floor is **0.0048010583**. Supplied IDs reduce error 98.48% below that floor and
  produce 97.98% of the real within-root token variance. The same existing real-fitted reader recovers all **32 skeleton
  and 15 cow occupancies**, with **zero false positives**, on both the real targets and supplied-ID predictions.
  These four roots contain **no zombie or arrow occupancy**: this is no new damage/death result.
- The hard interface has one joint code and exactly zero within-root prediction variance. The soft interface also has
  one argmax code, but argmax counts do not bound its continuous information. Its *actual output variance* is only
  **0.001054%** of the target's within-root variance. Thus removing hard quantization did not recover future-specific
  prediction here. Soft and hard failures localize this implementation's learned-interface/coadaptation problem;
  they do not identify its entire upstream optimization cause.
- The supplied IDs identify cases (at most 5 bits across the 26 IDs). They are not learned, physical innovations or a
  claim about chance entropy. Completion can express these supplied continuous futures; no generalization, calibrated
  sampling, recursive coherence, four-way uncertainty separation or actor benefit follows from this overfit result.
- All negative milestones, final model/optimizer/RNG state, source-bound contracts and prediction snapshots are retained.
  The optimization control is complete. No further posterior-collapse repair or substantial experiment was launched.
- Evidence: `oracle_v1/learned_innovation/optimization_control/{contract,result,completed_analysis}.json`; CPU saved-output
  analysis: `20261012_oracle/innovation_overfit_analysis.py`. No new head was fitted for this final analysis.

### 10-14 PAIRED EVENT COMPLETION — stronger coordination; coherent completion fails

**Question:** does a continuous source-associated event that owns both endpoint edits learn coherent stochastic
movement better than the existing destination-centered direct energy generator? Exactly two arms from the original
D parent; shared encoder/backbone/trunk initialization, TRAIN windows, Gaussian draws, energy objective and optimizer.
Budget:3,000 updates, batch16, L16,720,000 next-state targets/arm. No simulator labels or future-event masks in training.

- Preregistered before implementation in commit`108469a`; executable study and documented pre-main correction in
  `8e0ce47`. Protocol:`20261012_oracle/PAIRED_PREREG.md`; code:`paired_{model,train,eval,analysis,diagnostics,checks}.py`.
  Criteria:`0270895371fa8356764b7243a7d7789864f569be15828dccd61c6a6546c6033a`.
- Shared continuous packet/gate/soft routing drives contextual removal and arrival writes; no independent stochastic
  destination writer, hard assignment or feature-sum conservation. Capacity:direct6,962,134 / paired6,971,350
  parameters; completion580,294 /589,510 (+9,216:0.132% total,1.588% head). Before semantic evaluation/main training,
  the initial narrow decoder was replaced to preserve a full192-rank output map; earlier4-update implementation
  fixtures and their exact sources are archived in `paired_event/preflight_history/v1`.
- Final execution proof: interrupted2→4 vs uninterrupted4 gives model difference0, optimizer7.05e-12, exact RNG/ledger
  streams, causal future/replay difference0 and nonzero gradients from both independent noise draws. Actual B16 peak
  allocation1.590GiB with both models resident. Initialization checkpoints remain byte-identical through resume.
- **Measured reference limitation:** both initialized arms reproduce the same event-off parent reference. On14 clean
  sources, real-reader validity is12/14 and bank-reader validity6/14; the other cases are multiple apparent destinations,
  not absences. Scored static corruption0.2664%. Thus b is not assumed to be a valid afterstate. Final readings separate
  systematic baseline correction from sampled variation and reference-valid/invalid packet interventions.
- Physical validity, valid cow movement and joint event-law improvements are binding alongside scored35 joint occupancy
  ES, original absolute coherence/static/readout caps and alternative coverage. Isolated post-attention packet swaps,
  untied writes, routing concentration,128-shuffle dependence and stable/uncertain variance/mean fidelity are scheduled
  diagnostics. Existing14 sources are exploratory; original fresh reservation/20-cow confirmation rules remain unchanged.
- Scope:local stationary-camera movement coordination. All104 roots retain scroll/static/camera diagnostics. No H16,
  actor, nuisance invariance, long-memory or persistent-terrain claim. Main training/evaluation are resumable; scripts log
  to `artifacts/eda/levers_logs/paired_event_20261014.log`. Evidence:`artifacts/eda/oracle_v1/paired_event/`.
- **Completed endpoint:** both arms3,000 updates. Registered verdict`coherent_completion_not_established`;
  `allows_confirmation=false`. No reserved fresh data opened. Primary-reader source validity560→613/896
  (**62.50→68.42%, +5.92pp [0.33,10.72]**); duplication154→85/896 (**17.19→9.49%**), but absent/multiple
  remain283/896 (**31.58%**). Cow valid movement33→49/512 (**6.45→9.57%**), difference+3.13pp[−1.17,7.23],
  against real41.02%. Second-reader cow movement35→36/512; source-validity improvement unresolved.
  Joint occupancy ES0.024295→0.024876, no improvement; second-reader ES worsens+0.002408[0.000937,0.003968].
  Camera/scored-static/alternative coverage pass; physical coherence, movement, joint law and reader sensitivity fail.
- **Measured dependency benefit:**128 matched marginal-preserving shuffles give validity dependence2.83pp direct vs
  10.27pp paired; gain**+7.45pp[1.45,13.23]**, corroborated by the second reader+8.98pp[3.99,13.90].
  Overall source-count variance0.467/0.461 vs real0; count conservation remains unresolved and reader-sensitive.
- **Single-event intervention:** one post-attention packet changes source clearing103/896 times, joint clear/create27,
  and valid stay→move18 under the primary reader. Only**8/896** moves agree under both readers, all on one cow root;
  either endpoint ablation eliminates them. Real-fan support and eligible terrain match, but generated local features
  remain outside the real split reference. All8 cow event-off references are valid STAY under both readers.
- **Frozen route hardening:** source argmax makes validity613→619/896, increases multiple destinations190→204,
  and lowers second-reader validity568→551. Inference hardening does not repair this frozen model; changed write
  magnitudes prevent isolating soft splitting's upstream contribution. Source/destination edits still fail to construct
  reliable semantic counterparts. Loss/optimization/target-geometry attribution is unisolated.
- **Variance localized numerically:** full81 variance/real fan7.917× paired vs1.705× direct; scored35 only1.369×/1.268×.
  All**164** reference cells with centered RMS<0.1 are newly entering the view under exact recorded camera offsets.
  Their pre-LN variance sum**0.257581→84.813538** after final LN (**329.27×**), accounting for**81.28%** of total
  paired sampled variance. Suppressing unscored-source packets reduces edge variance90.813→17.482 (−80.75%) with
  interior valid-source/cow-movement counts unchanged. LN compresses scored-domain variance(post/pre0.777): entering
  normalization and interior movement incoherence are separate. Entering does not imply never observed in history.
- **Budget/efficiency:** paired training components+12.76%, completion1.881×, +0.132% total parameters; peak1.590GiB.
  Paired validity90.18% at500 updates came with zero cow movement; final joint ES still improves, so convergence and
  universal architectural failure are not established. Dense stochastic computation remains; no adaptive-selectivity claim.
- Final write-up:`20261012_oracle/PAIRED_RESULTS.md`; raw rows/reports under`paired_event/diagnostics/`.
  All29 source/input pins and3,000-update batch/noise ledger verify exactly. Frozen failure diagnostics were declared
  before computation in`PAIRED_FAILURE_PREREG.md`. Negative milestones/checkpoints are retained. One-step exploratory
  evidence only; no new nuisance, long-memory, terrain-persistence, recursive or actor result.
- **CPU semantic curve audit:** last available interval1,500→3,000, paired all-source movement5.80→12.61%
  (+6.81pp[2.46,11.16]); duplication18.08→9.49%(−8.59pp[−12.05,−5.36]), but absence3.01→10.38%
  (+7.37pp[4.13,11.16]). Cow event Brier improves under both readers; primary cow-movement gain unresolved,
  second-reader gain+3.71pp[0.59,6.45]. Temporal dependence change unresolved; total validity does not improve.
  Of61 additional movement draws,36 are skeleton/17 cow/8 zombie, so3 skeleton roots dominate the aggregate gain.
  These are late semantic gains with incomplete coherence, rather than loss-only learning or evidence of convergence.
- **CPU objective sensitivity:** exact current token energy distance,14 strict roots,disjoint16/16 real-future split,
  32 independent five-token shuffle repetitions/root. Each full192-D token marginal is preserved; common whole-frame
  permutation scores bit-identically. Both readers' validity100→52.59%; energy score0.036499→0.038769,
  penalty+0.002270[0.001341,0.003205]. Cow8 penalty+0.002993[0.001743,0.004078]. Full empirical32 sensitivity
  agrees. Objective detects this constructed joint incoherence; scalar gaps do not identify parameter-gradient cause.
  Shuffling includes contextual token changes; finite without-replacement permutations are not iid ensemble draws.
- **CPU normalization separation:** actual saved-output gradients on all46 unscored tokens are exactly0. Scaling
  only unscored synthetic pre-LN inputs by1/1,000 changes scored loss/gradients by0. Entering amplification cannot
  consume gradients through excluded final outputs in teacher-forced training. Earlier packets can still affect scored
  outputs via attention/writes. CPU FP32 reference recomputation puts93.79–99.16%(mean97.85%) copy weight on
  exactly zero-padded candidates at all164 GPU-defined low-reference cells; generation weight mean0.937%.
  This is corroboration, not exact BF16 CUDA replay(low-reference sets agree103/104 roots; max reference difference0.05785).
  Evidence:`diagnostics/{curve_reassessment,objective_sensitivity,normalization_gradients_cpu,reassessment_preservation}.json`;
  raw rows and exact executed sources retained beside them. Assessment:`20261012_oracle/PAIRED_REASSESSMENT.md`.

### 10-14 DIRECT STOCHASTIC COMPLETION — preregistered before optimization

**Question:** can direct Gaussian-noise completion learn physically valid alternative successors, rather than a
future-looking posterior/codebook/prior? This is a controlled one-step feasibility study. It does not establish the
WHITEBOARD four-way uncertainty separation, recursive fidelity or actor benefit.

- Exactly two arms share the frozen Raw encoder, unconditioned D parent, live corrt/fmamba backbone, new spatial
  completion head, initialization, batches, independent noise draws and optimizer: **3,000 updates, batch16, L16**,
  720,000 next-state targets per arm. The sole contrast is first-power point distance versus that same attraction
  minus the energy score's half pairwise distance. Noise influences both routing and generated content.
- TRAIN input-only brightness selection retains **367,619 windows / 6,109 episodes**, with all17 actions. There are
  no simulator labels, future-event masks or terminal oversampling in training. The fixed distance scores35 interior
  tokens, excluding HUD and entering terrain; unscored corruption and variation remain reported diagnostics.
- Frozen real-reader controls pass on all **104×32** exploratory real mob layouts. Strict joint source/destination
  eligibility has only **14 variable roots (8 cow)**, so this panel can establish only an exploratory lead. Samples
  with missing/multiple destinations, duplicates or wrong camera registration stay in the physical denominator.
- Final64-sample evaluation requires resolved joint semantic distribution improvement, valid alternative outcomes,
  physical coherence and paired static/readout fidelity. Frozen criteria ID:
  `fa9a79d435344a3d4840350c5881c10d9220d90be9152bcb79014c05b994f619`.
  Source absence/multiplicity, duplication and camera margins are **0.00669643**; paired static corruption margin
  **0.00328781**; reader disagreement **0.00060969**; valid-alternative coverage minimum **0.88616071**. These are
  operational real-fan/readout resolution allowances, not population confidence bounds.
  The earlier preflight ID `003e76aa608a...` is retained separately; the final source pin enforces the already-declared
  fresh128-root minimum and does not change any numerical criterion.
- A promising final result triggers unchanged evaluation on reserved fresh seeds **2026101400+**: at least128 bright
  roots and20 independent variable clean source roots including20 cows, with512-root/1,536-episode caps. Thresholds,
  readers and checkpoints remain fixed. Insufficient coverage is inconclusive; an incoherent diverse output fails.
- Milestones0/500/1,500 are optimization diagnostics only. Both arms save model, optimizer and all RNG/ledger state;
  interrupted-versus-uninterrupted validation precedes the full budget. Measure backbone/head costs, peak memory,
  marginal sample costs and sampled variation locations. No third arm, larger generator, H16 or actor is scheduled.
- Protocol/code: `20261012_oracle/DIRECT_PREREG.md`, `direct_{model,data,train,eval,coherence,analysis,fresh,checks}.py`.
  Evidence: `oracle_v1/direct_generator/{data,criteria.json,validation.json,training,eval_*}`. Negative results from the
  earlier learned-innovation control above remain unchanged.

  Pre-training numerical validation: both scored noise paths and causal backbone receive nonzero gradients; changing
  the unused future input changes current predictions by **0.0**. Interrupted2→4 and uninterrupted4 runs give model
  weight difference **0.0**, optimizer-state difference **1.0e-11**, identical RNG/ledger streams and peak allocated
  memory **1.40GiB**. These are execution checks, not scientific outcome results.

**Completed result (10-14): `coherent_completion_not_established`.** Both arms finished3,000 updates/720,000 targets;
all input/source/checkpoint pins match, the entire3,000-step batch/noise ledger reproduces exactly, and TRAIN/eval
episode overlap is zero. Joint scored35 mob ES improves **0.031417→0.024439** (paired−0.006978[−0.009680,−0.004326]),
but energy has **317/896** invalid clean-source draws:136 absent,181 multiple, including140 source-retained duplicates.
Both frozen readers agree307 are invalid. Camera and paired static fidelity pass; physical coherence, alternative
coverage and the small reader-disagreement guard fail. The failed result correctly blocked fresh confirmation.

- The energy channel learned semantic alternatives:12/14 sources, mean3.071 valid alternatives versus point0.786
  and real3.643. Cow valid moves remain **8.20% versus41.02% real**; point stays on all512 cow draws. This is partial
  distribution learning, not calibrated coherent movement or a four-way uncertainty separation result.
- **Joint coordination diagnostic (CPU):** independently shuffling destinations preserves exact per-cell marginals.
  Real validity falls100%→54.10%; energy64.62%→61.89%. Learned dependence adds2.73pp[0.39,5.14] of validity, versus
  45.90pp[39.19,51.31] for real futures. Conservation fraction is5.33% (cow9.58%), versus100% real. Coordination is
  weak rather than absent. `direct_generator/diagnostics/joint_shuffle/{analysis,paired_intervals}.json`.
- **Path interventions (GPU, frozen weights):** original forecasts replay exactly. Removing off-source source copies
  changes317 invalid draws to311; hard per-destination choices leave307. Fixing routing at the first draw leaves242,
  fixing generation leaves300; both paths contribute. Shared-only noise over four fixed local references leaves
  37.28% invalid; local-only leaves35.30%. None is a repair or a retrained architecture comparison.
  `diagnostics/{paths,noise_channels}/analysis.json`.
- **Objective diagnostic (CPU):** stable-cell variance is53.09× this real fan, but suppressing it at fixed means makes
  held latent ES worse by+0.001698[0.001194,0.002319]. Attraction improves0.055893→0.051242; lost diversity credit is
  larger,0.027953→0.021604. The current biased predictions receive a local objective benefit from excess spread;
  this does not contradict proper scoring or prove the sole training cause. Static terrain corruption remains
  0.2664% point/0.2724% energy, so contextual variation is not automatically physical corruption.
  `diagnostics/stable_spread/result.json`.
- **Metric dependence sensitivity (CPU, model-free):** per-cell real192-D token permutations preserve feature and
  semantic marginals exactly, but create44.55% invalid source outcomes. On those same14 sources the declared latent
  ES worsens only5.93% (0.037699→0.039936, paired+0.002237[0.001898,0.002566]); cow-only7.08%. Joint decoded ES also
  detects the damage, with6.78% increase. The modest score cost of large coherence damage is measured directly;
  it is not proof of a sole optimization cause. `diagnostics/dependency_sensitivity/result.json`.
- **Interpretation:** direct proper-score learning activated the stochastic path that the prior/posterior study failed
  to activate. Its completed samples still lack joint source/destination consistency. Repeated source copying alone,
  soft-versus-hard destination choice and dropping one noise channel do not explain or fix most failures. Literature
  agrees that a proper population optimum is distinct from finite-model coherent sampling; ITC's joint matching and
  FGN's shared-noise precedent require testing in contextual continuous features. No further training or recursion ran.
  Full results, exact scope and primary sources: `20261012_oracle/DIRECT_RESULTS.md`.

### 10-12 GENERATED-PREFIX COMPLETION — predeclared before training (21:07 AEDT)

**Question:** does exposure to detached generated history improve the oracle-conditioned world's recursive prediction?
User authorized this separating experiment after the D-loss/readout diagnosis. This tests conditional branch tracking;
it does not yet learn stochastic innovations or validate an actor.

- **Initial state:** both arms load exactly `oracle_v1/loss_intervention/D_u3000.pt`, including its OracleCond. Frozen Raw
  encoder/tokens, corrt/fmamba architecture, full D objective, TRAIN support-v2 row pool, optimizer, seed-11 main sampler,
  BF16/IEEE and 3,000 updates at batch 16 are fixed. World and OracleCond remain trainable in both arms.
- **Both arms:** retain the original full-L16 teacher loss and add one matched endpoint loss per row, weighted
  **0.5 teacher + 0.5 endpoint**. The teacher control receives real context at that endpoint; the prefix arm receives its
  own detached generated context. This preserves the ordinary 15-target training signal and matches added supervision.
- **Endpoint exposure:** independent sampler seed 20261012; prefix length p chosen equally from {1,2,4,8,15}; main target
  t chosen uniformly from [30,63], terminal target t=63. Prefix length is independent of target and terminal status.
  Starting context is real frames [t−p−15,t−p); generate p states without gradient, then predict target t with gradient
  from the final 15 frames. Control context is real [t−15,t). Both use actions [t−15,t), next-layout oracle [t−14,t+1),
  target token t and endpoint input position 14. p=15 makes the entire final context generated. These 64-frame slices
  avoid the earlier short-prefix/history confound. No gradients pass through prefix generation.
- **Controls before launch:** frame/action/oracle timeline sentinels; p=0 and truth-substituted-prefix equality; p=15 full
  generated context; first prediction parity with the evaluator; generated tensors detached and final loss gradients live;
  terminal and nonterminal endpoint labels represented at every p; interrupted/resumed optimizer/RNG/sampler restoration.
- **Budget/readings:** save full resumable state every 100 updates and snapshots at 0/500/1,500/3,000. Evaluate the same
  200 DEV episodes through H16 (exploratory reuse), both factual and self-fed. Paired episode comparisons report matched
  generated-fitted mob readouts, original-reader sensitivity, location/source residue, camera alignment, near-player
  terrain, health/food and direct token-error components. Readout FIT/DEV/TEST episodes stay separated as in the preceding
  CPU control. Primary all roots; common alignment is a selected sensitivity. No new binary gate or unconditional seed 2.
- **Interpretation:** a smaller recursive gap without losing factual capability supports generated-context robustness at
  this adaptation budget. A negative while learning is still moving is not proof against continuous completion. Full-screen
  recorded oracle layouts can become inconsistent after a camera/player branch error; diagnose that separately. HUD timer
  memory and unprivileged innovation remain distinct questions. This intervention differs from JEDI's random switching,
  while sharing its predicted/encoded-context training principle ([§3.2](https://arxiv.org/html/2605.13013v1#S3.SS2)).
- **Files:** new `20261012_oracle/prefix_intervention.py` and prefix evaluation/readout helpers; preserve prior source-bound
  scripts. Outputs `oracle_v1/generated_prefix/`, stdout `artifacts/eda/levers_logs/`. The prefix arm costs extra no-gradient
  rollout work; record per-arm time and compute instead of claiming equal wall time. No architecture integration follows.

**Launch/verification (21:17 AEDT):** actual model p=0 outputs, first prefix/evaluator outputs and original D teacher loss
agree exactly (max difference 0). p=15 has 15 no-gradient calls plus one final gradient call; world/oracle gradient norms
0.5543/0.01661 on the two-row check. B16 smoke peak 1.659 GiB. Interrupted 2→4 versus uninterrupted 4 reproduces sampler,
exposure stream, Torch/CUDA RNG and batch ledger; GPU parameter endpoints differ up to 1.58e−6, so no bit-exact training
claim. Serialized model/optimizer restoration is checked exactly. Resume/reference smokes overlapped briefly and are not
used for throughput. Full 3,000-draw CPU preview has terminal fractions 25.65–27.30% across p and predicts final ledger
`b6e6519c7ec90fef26e919558158d133c135dadef8012cf0b8ad7032d7b92a5f`.
- Official run completed a ten-update throughput check, then resumed unchanged as
  `oracle_generated_prefix_20261012.service`; evaluation/readout queue `oracle_generated_prefix_eval_20261012.service`.
  No other research GPU job was active at launch. Both arms will receive 720,000 main targets + 48,000 endpoint targets.
- Proofs: `oracle_v1/generated_prefix_real_checks.json`, `generated_prefix_resume_proof.json`,
  `generated_prefix_sampler_preview.json`, `generated_prefix/eval_validation.json`; helper sources are retained beside
  the trainer. Primary source/inputs contract `5ea495dc9bd534f66c7673a2e0095298e47eace83ed740dc8aa9d7279bc43e04`.
  Initial matched batch (p=4) has equal main teacher loss 0.0296357; endpoint D loss is 0.0279704 on real context versus
  0.154864 on generated context. This is a training-input difficulty check, not an outcome verdict.
- **CPU evaluation-chain validation completed (21:29):** four readout CLIs ran on an isolated old-D prediction fixture,
  with GPU disabled. Prior readings reproduce exactly; 100 identical-arm contrasts and their intervals are exactly zero.
  Resume reuses ten results, leaving all 21 output hashes and timestamps unchanged. Proof:
  `generated_prefix/eval_cli_validation.json`. An initial validation-only provenance-hash comparison failed because arm
  labels correctly differ; preserved in the fixture, with no numerical pipeline defect or source change. The fixture is
  not a result of the new worlds. Training is active with a durable update-400 checkpoint; no final report exists yet.
- **Training completed (10-13):** both arms reached 3,000 updates, each with 720,000 main + 48,000 endpoint targets.
  The final matched batch ledger agrees exactly with the independent CPU preview.
- **Evaluation control:** the identical starting arms produce exactly equal predictions and accessories on all 200 DEV
  roots (`generated_prefix/eval_u0000/analysis.json`). This establishes baseline parity for the treatment comparison.

**Completed result (10-13): partial HUD gains; generated-prefix exposure does not close the recursive gap.**
Primary comparison: 200 reused DEV episodes, all roots, depths 9–16, matched 3,000-update arms. Both receive the true
next-screen zombie/cow/skeleton/arrow layout; the oracle supplies neither intrinsic timers nor attack cooldowns.

| late self-fed reading | teacher control | generated-prefix arm |
|---|---:|---:|
| near-player tile accuracy | 87.52% | 87.00% |
| camera-aligned root-steps / 1,600 | 1,218 | 1,229 |
| health MSE, native hearts | 1.384 | 1.529 |
| food MSE | 0.633 | 0.597 |
| drink MSE | 0.925 | 0.547 |
| energy MSE | 0.807 | 0.570 |
| damage caught / 48 | 28 | 29 |
| false health drops / 1,523 | 28 | 26 |
| cow TP / 870; frozen-reader FP | 819; 245 | 852; 838 |
| zombie TP / 583; frozen-reader FP | 498; 35 | 514; 147 |

- **Resolved benefit:** drink MSE −0.377 [−0.528, −0.238], energy MSE −0.236 [−0.451, −0.039] in paired
  episode-bootstrap intervals. On the common camera-aligned subset (1,138 late root-steps), drink retains its gain
  −0.279 [−0.432, −0.140]; energy −0.099 [−0.313, +0.118] is unresolved. Health, food, damage catch, terrain and late
  camera-alignment contrasts are unresolved. Both arms still catch 0/29 recovery events with the fixed class reader.
- **Objective tradeoff:** recursive frame L1 rises 0.2124 → 0.2376; difference +0.02519 [+0.01386, +0.03714]. Mean
  squared error 0.1677 → 0.1723 is unresolved, while mean frame-maximum squared error falls 9.498 → 7.390;
  difference −2.108 [−2.611, −1.615]. The aggregate D-loss change −0.01393 is composed of **+0.00463 MSE,
  +0.00252 weighted L1, −0.02108 weighted maximum**; it reflects fewer extreme coordinate errors with worse mean
  absolute fidelity. Factual frame MSE also rises 0.01184 → 0.01332 (+12.5%; paired difference +0.00148
  [+0.00116, +0.00183]), so preservation of factual feature fidelity is not established.
- **Readout control:** on the same 50 TEST episodes, prefix cow TP/FP is 212/131 with the original frozen reader,
  207/38 after changing only its threshold on DEV, and 207/34 with a generated-fitted equal-capacity head. Teacher
  generated-fit counts are 200/6. Generated-fit cow F1 is 0.9501 versus 0.9079, difference −0.0422 [−0.0869, +0.0029]
  (unresolved); its prefix precision cost is resolved, −0.1120 [−0.1763, −0.0531], at the DEV-selected operating points.
  The DEV-count sensitivity also retains excess prefix FP (28 versus 6). Generated-fit zombie F1 0.8193 versus 0.7865
  is unresolved. Frozen-reader FP therefore overstates a semantic loss, but refitting does not establish a prefix gain.
- **Spatial information remains readable:** generated-fit cow top-1 is 184/186 teacher versus 183/186 prefix; zombie
  93/94 versus 89/94. Prefix local 3×3 AUC is 0.99674 cow / 0.98757 zombie. These presence-conditioned readings show
  substantial localization under known outcomes; they do not certify calibrated absence, physical state validity or control.

**Post-hoc CPU emission intervention:** ran the existing `decoder_context_shift.py` on both final prediction exports,
with no model inference/training. Reconstruction and frozen-reader equality checks pass. On the prefix arm's own
both-current-and-previous-camera-aligned late subset (1,223 root-steps; 673 true cow cells), 580/666 clean cow FP
(87.1%) have a copied predecessor whose generated features read cow while both the physical source and its true-token
reader are absent. Replacing only copied candidates with true previous tokens changes **FP 690 → 126; TP 663 → 656**.
Paired FP/true delta −0.838 [−1.647, −0.332]. Replacing only the generator gives FP 656; routing only gives FP 723.
The corresponding teacher-arm copy swap is FP 194 → 80 on its own aligned subset. Prefix zombie copy swap is
FP 113 → 52, TP 400 → 403. These are emission-only substitutions: they isolate propagation of false reader responses
through copied generated features, without proving literal duplicated objects or a recursively repaired trajectory.

**Generated-fitted HUD control (10-13, CPU):** same 100 FIT / 50 DEV / 50 TEST whole-episode split; FIT-only
standardization, equal-capacity ridge family, DEV-selected alpha, no depth/position input. TEST late depths 9–16 contain
11 damage and 8 recovery opportunities.
- Health MSE on the 50 TEST episodes: teacher original **1.2445**, generated-fitted **1.4598**; prefix original **1.5790**,
  generated-fitted **1.7038**. Matched heads catch 2/11 teacher versus 4/11 prefix damage events, and 0/8 recovery in both.
  These fits provide no health rescue. They are not information ceilings or a resolved teacher-versus-prefix comparison.
- The fresh head's TRUE-token positive control has health MSE **0.04928** and drink **0.03117**, but food **1.3780** and
  energy **0.29586**. For food, one TEST episode accounts for 55.2% of squared error, and TEST includes food=0 while FIT
  ranges 1–9. The family therefore cannot support a broad claim that generated HUD facts are absent.
- A simpler FIT/DEV-trained affine calibration of the existing four-counter reader lowers teacher health MSE
  **1.2445→1.0719**, and prefix **1.5790→1.3123**. These are descriptive point changes, without a paired uncertainty claim;
  they retain the teacher/prefix ordering and do not establish repaired dynamics.
- Source: `20261012_oracle/hud_readout_refit.py`. Outputs: `generated_prefix/hud_readout_refit/`, including
  `positive_control_calibration.json`; original prediction exports/readers are unchanged.

**Reading:** this matched intervention changes generated-context exposure and reveals a partial benefit. The copy-feedback
failure survives, and the stronger mob result remains the teacher control with generated-state readouts. The cause of the
learned prefix arm's increased copied-feature bias is not isolated by this experiment. No evidence here settles full
backpropagation, from-scratch mixed-context training, learned innovations, persistent timer memory or actor performance.
One training seed; the readout TEST split is separate from fitting but drawn from the repeatedly inspected world panel.
Recorded world/conditioner compute totals are 1,630 s teacher / 2,829 s prefix, so this is matched updates/targets rather
than matched computation. Independent CPU recomputation reproduces the four reported semantic paired intervals.

**Literature:** [JEDI §3.2/§4.4](https://arxiv.org/html/2605.13013v1#S3.SS2) combines mixed predicted/encoded context with
joint latent learning and bounded activations; its switching ablations support that recipe. [Dreamer 4 §3.2](https://arxiv.org/html/2509.24527v1#S3.SS2)
uses denoising with noise-level conditioning and imperfect contexts. Neither establishes that the present frozen-token,
detached-prefix continuation is sufficient. Evidence: `generated_prefix/eval_u3000/{analysis,rows}.json`,
`readout_refit/{paired_u3000,calibration_u3000_*,spatial_u3000_*}.json`, and
`eval_u3000/decoder_context_shift_{teacher,prefix}.json`; stdout remains in `eda/levers_logs/`.

### 10-12 MATCHED D-LOSS FOLLOW-UP — completed; readout controls completed (19:21 AEDT)

**Result: D changes completion tradeoffs; the experiment does not support a D-only explanation of recursive failure.**
`loss_intervention.py` completed the predeclared 3,000 updates per arm from the same oracle-D world/conditioner, with
identical TRAIN batches. Evaluation uses 200 DEV episodes, 15 observed frames, H16, and the frozen real-token readers.
The world receives the true next mob occupancy at every step. These are conditional-completion diagnostics.

Primary population below: all episodes, depths 9–16; 870 true cow cells, 583 zombies, 48 alive-after damage events and
1,523 unchanged-health steps. MSE and event readings use the same fixed health/food readers in every arm.

| matched arm | cow TP / 870 | cow FP | zombie recall | zombie FP | health MSE | food MSE | damage caught / 48 | false health drops / 1,523 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared starting model | 804 | 495 | 0.842 | 270 | 2.051 | 0.742 | 28 | 24 |
| D, +3,000 updates | 805 | 227 | 0.847 | 70 | 1.635 | 0.661 | 32 | 25 |
| L1, +3,000 updates | 838 | 161 | 0.895 | 40 | 1.240 | 0.465 | 29 | 13 |
| D without maximum term, +3,000 | 828 | 213 | 0.873 | 43 | 1.321 | 0.520 | 30 | 24 |

- **D continues to improve:** cow FP falls 54.1% and zombie FP 74.1%, with recall essentially unchanged. Paired
  episode-bootstrap differences in FP/true are −0.308 [−0.458, −0.190] and −0.343 [−0.491, −0.228]. Health MSE improves
  −0.416 [−0.755, −0.129]. These improvements also hold on the common camera-aligned sensitivity population.
- **L1 versus concurrent D:** cow recall +0.0379 [0.0152, 0.0636], zombie recall +0.0480 [0.0171, 0.0870]; both survive
  common alignment. Cow FP reduction is unresolved: −0.0759 [−0.1862, +0.0188] per true cow. Food MSE improves
  −0.196 [−0.337, −0.062], and false health-drop rate −0.00788 [−0.01593, −0.00130]; both survive common alignment.
  Health MAE improves overall, but health MSE and damage-catch differences are unresolved. Terrain and facing differences
  are unresolved. This is a measured tradeoff, not a universal loss winner.
- **Maximum-term contrast:** final cow-FP difference is −0.0161 [−0.1179, +0.0827] per true cow, unresolved. Its large
  apparent FP benefit at update 500 does not persist: D/no-max FP = 477/341 at 500, 293/368 at 1,500, 227/213 at 3,000.
  No-max improves food and drink MSE overall; only drink remains resolved under common alignment. Recall benefits weaken
  on that selected population. Negligible clipping (D 4/3,000, L1 0, no-max 3) does not account for these contrasts.
- **Remaining health limitation:** every arm catches 0/29 recovery steps by the fixed class reader, versus 28/29 on true
  tokens. Late fresh-arrival damage has zero opportunities, so this panel cannot settle that mechanism.
- **Scope:** one pretrained parent and one training seed; a continuation/loss-switch test, not a from-scratch ranking.
  Shared alignment retains 1,143 late root-steps and is a selected sensitivity population. Intervals use 2,000 paired
  episode bootstrap draws. No claim about a learned unprivileged innovation, an actor, or canonical Raw/TC readiness.

**M0–M3 readout control: the counted excess is substantially calibration-sensitive.** `generated_readout_refit.py`,
`readout_calibration.py`, `readout_spatial.py`: 100 FIT / 50 DEV / 50 TEST whole episodes, split seed 20261012. Fit-only
normalization, equal-capacity 192-D linear ridge heads, existing alpha family selected by DEV MSE; no depth/position feature.
DEV-F1 thresholds are primary; DEV-count matching is a separately reported sensitivity. Original readers remain unchanged.
All depths fit the heads; late depths score them. TEST is separate from head fitting but reuses the inspected world panel.

| +3,000 D, late 50-episode TEST | cow TP / 215 | cow FP |
|---|---:|---:|
| original frozen head and threshold | 205 | 44 |
| same frozen weights, DEV-F1 threshold | 200 | 18 |
| generated-fitted equal-capacity head, DEV-F1 | 200 | 10 |
| generated-fitted head, DEV-count sensitivity | 206 | 17 |

- Threshold-only FP/true change is −0.1209 [−0.2591, −0.0259], with recall change −0.0233 [−0.0621, 0]. At the initial
  model, threshold cleanup has a larger recall cost: 206 TP / 89 FP → 175 TP / 33 FP. Neither is a free precision gain.
- **Spatial signal survives:** generated-fitted D top-1 is correct on 182/186 cow-present frames and 94/94 zombie-present
  frames. Local 3×3 AUC is 0.99886 for cows and 0.98999 for zombies. L1 is 184/186 and 93/94; no-max 183/186 and 93/94.
  This excludes an explanation based solely on easy global background cells. Presence-conditioned top-1 and privileged
  top-K are diagnostics, not calibrated absence/count estimation. Generated versus real-fitted zombie AUC advantage is
  unresolved: +0.00298 [−0.00121, +0.00948].
- **Correction:** earlier ghost/duplication language described frozen-reader false positives. Those counts alone do not
  establish literal extra objects, irreversible information loss, or categorical-state necessity. The copied-content
  substitutions below isolate propagation of a false reader response; completed joint-state validity remains unestablished.

**Execution/evidence:** restart at update 2,000; resumed serialized world/conditioner/optimizer state exactly and completed
at 18:58, evaluation at 19:00:16. Resume smoke preserves sampler/RNG/batch order but CUDA endpoints differ up to 1.34e−6;
training is not bit-reproducible. Full numerical/source/input contracts, TRAIN pool verification, atomic checkpoints,
milestone predictions and paired raw rows are under `oracle_v1/loss_intervention/`: `contract.json`, `training.json`,
`resume_proof.json`, `eval_u{0000,0500,1500,3000}/analysis.json`, and `readout_refit/{u,calibration_u,spatial_u}*.json`.
Final shared batch ledger: `6e53c1b824a091ca5d49b7b0788cf76d60e852a4f5bdf0b917140ad3da9c5c87`.
Stdout stays under `artifacts/eda/levers_logs/`. All jobs in this follow-up have completed.

**Literature after diagnosis:** the borrowed D coefficients are reconstruction losses in Delta-IRIS, not a theorem of
physical validity. [JEDI §3.2](https://arxiv.org/html/2605.13013v1#S3.SS2) mixes predicted and encoded next contexts;
[Dreamer 4 §3.2/§3.4](https://arxiv.org/html/2509.24527v1) trains on varying context noise and sequences longer than the
context. These provide precedents for addressing generated-context mismatch, not evidence that it is repaired here.
Historical E8/E12 failures used short rollouts without the present known-outcome control; E12 also documented a context
protocol mismatch. They do not rule out such a controlled intervention. No new architecture or training remedy was launched.

### 10-12 DECODER FEEDBACK DIAGNOSIS (CPU; u0/500/1,500/3,000 exports)

**Measured local contribution: propagation of false frozen-reader responses through copied generated features.**
`decoder_decomposition.py` and
`decoder_context_shift.py` reconstruct the emitted token from the saved six candidate weights, raw generator and previous
frame, then apply the actual terminal LayerNorm. Reconstruction differs ≤1.91e−6 and all frozen mob-reader decisions agree.
- Interpret physical source labels only where **both previous and current generated camera offsets match the recorded
  trajectory**. Strong false predecessor = generated copied candidate reads present, actual source mob flag is absent,
  and the corresponding TRUE copied candidate also reads absent. Actual target must also read absent in the TRUE reader.
- Initial oracle-D, depths 9–16: 1,207 both-aligned root-steps; 340 clean cow-FP cells, 269/340 (79.1%) with a strong false
  ordinary-map copied predecessor (203 at self). Only 62/340 are also FP under teacher forcing at the same target cell.
- **Emission-only substitutions** keep the generated hidden-state prediction and target oracle fixed: replacing only
  copied candidates with real previous-frame tokens reduces cow FP **360 → 147**, TP **628 → 632**; FP/true difference
  −0.320 [−0.457, −0.209]. Replacing only raw generator with its teacher-forced counterpart gives FP 360 → 319,
  TP 628 → 621. Replacing only routing weights with teacher-forced weights gives FP 360 → 399.
- At matched update 1,500, the same both-aligned copied-content substitution gives:

| arm | baseline cow FP | real copied candidates | TF generator only | TF routing only |
|---|---:|---:|---:|---:|
| D | 225 | 98 | 222 | 232 |
| L1 | 171 | 81 | 155 | 245 |
| no-max | 282 | 123 | 259 | 457 |

- At the final 3,000-update milestone, both-aligned baseline → real copied-candidate cow FP is **D 192 → 82,
  L1 133 → 83, no-max 175 → 96**. D FP/true difference is −0.160 [−0.284, −0.072], TP 652 → 656. Substituting only the
  TF generator gives D 192 → 200 FP; only TF routing 192 → 240. These are arm-specific aligned local comparisons;
  the matched-arm primary comparison above uses all roots, with a shared-alignment sensitivity.
- These are **local decoder interventions**, not repaired recursive trajectories. They isolate the contribution of copied
  content at the observed failing emissions; they do not prove its fraction of total control loss or solve the feedback loop.
  Standalone generated candidates are co-adapted mixture components, so their normalized reader scores are descriptive,
  not independent valid-state predictions. Player/HUD source removal clears only 2/463 initial all-root clean cow FPs.
- Decomposition uses source/input-bound resumable per-root CPU caches; small context-substitution controls write atomic
  complete-arm results. Same frozen reader/thresholds. JSON evidence and raw rows are retained under
  `oracle_v1/loss_intervention/eval_u{0000,0500,1500,3000}/decoder_{decomposition,context_shift}_{arm}.json`.

### 10-12 D-LOSS CPU DIAGNOSIS (completed through 15:33 AEDT; matched retrain pending)

**Historical finding: recursive fixed-reader errors occur under both losses. Matched optimization and readout controls are now completed above.**
- `d_existing_compare.py`: fixed TRAIN readers, 200 DEV episodes, 2,000 paired episode-bootstrap draws, original FP16
  saved predictions. Late (9–16) oracle-D cow recall is 0.926 teacher-forced → 0.924 recursive, while false positives / true
  cows rise 0.131 → 0.577 (+0.446 [0.284, 0.657]). Oracle-L1 instead loses recall 0.533 → 0.033
  (−0.500 [−0.542, −0.452]); its small cow census is erasure.
- **Extra decoded zombie occupancy also occurs under L1:** TF → recursive FP/true rises +0.391 [0.243, 0.579] for L1,
  +0.348 [0.227, 0.502] for D. Common camera-aligned sensitivity retains both increases. Terrain accuracy loses
  8.46 points under L1 and 7.69 under D on common-aligned rows. Historical parent weights differ between these arms;
  these comparisons cannot isolate the loss. TRUE projection damage occurs before either world/loss and remains independent.

**Full objective, model-free conditional completion:** `conditional_stat.py`, `conditional_dopt.py`, `conditional_norm.py`.
- Five real futures at depth 1, TEST roots only; group futures sharing exact complete next-mob occupancy. 308 groups
  (2–5 futures each), 269 roots, 1,147 branches. Full D empirical optimum solved as a convex radius problem: 308/308
  validated, 12 tiny independent epigraph checks within 1.73e−10 objective units, two-start gap ≤9.58e−10.
- Conditional full-D optimum: cows 300/302 TP and 10 FP versus real 299/302 TP and 14 FP; zombies 171/179 TP and
  0 FP versus real 172/179 TP and 4 FP. Applying the world's exact per-token FP32 LayerNorm changes neither count.
- **Nonidentical support:** 239 groups have identical tokens; 69 groups vary (60 roots, 247 branches). On these 69,
  D optimum and normalized D optimum give 66/68 cow TP, 0 FP and 54/62 zombie TP, 0 FP; real gives 65/68 cow TP,
  4 FP and 55/62 zombie TP, 4 FP. Variation is night/render/re-encoding in 58 groups and inventory in 11.
  This tests one-step reconstructability under known occupancy, not the population conditional optimum or recursive inputs.
  The normalized candidate is feasible; it is not a solved LayerNorm-constrained optimum.

**Fixed-output incentives:** `d_loss_fixed_outputs.py`, `d_loss_ln_tangent.py`.
- At identical saved oracle-D late predictions, replacing all 502 decoded cow-FP tokens with their actual target tokens
  lowers D loss by 0.002547 and L1 by 0.001937. Both penalize these token errors. Their differing units are not a ranking of losses.
- Cow-FP absolute output-gradient shares: L1 0.388%, D-no-max 0.787%, full D 0.854%. The maximum term alone accounts
  for 47.0% of scalar teacher-forced D loss (34.0% recursive), 9.20%/9.90% of absolute output-gradient mass.
- Among 470 recursive cow-FP cells with a cow-free TRUE-token reader, descent through terminal LayerNorm raises the
  cow score in 25 under D, 16 without max, 17 under MSE, 47 under L1. On 89 teacher-forced FP cells: D 13,
  no-max 1, MSE 0. The max term contributes some locally wrong reader-score directions; it does not explain every false detection.
- Literal autograd checks pass (max differences 5.82e−11 and 2.24e−8). Failed output re-normalization diagnostic preserved;
  corrected tangent uses saved emissions unchanged. These output gradients are not parameter-update measurements.
- Counts above use original FP16 exports (502 cow FP); the later FP32 projection evaluation has 495. Preserve both lineages.

**Source contract:** our coefficients/reductions match Delta-IRIS's tokenizer reconstruction code, but its original decoder
receives quantized future-conditioned codes and reconstructs pixels; our D objective trains deterministic contextual-token
prediction. Its separate world learns code cross-entropy. [Tokenizer](https://github.com/vmicheli/delta-iris/blob/main/src/models/tokenizer/tokenizer.py),
[paper](https://arxiv.org/abs/2406.19320). The original method supplies no guarantee of object conservation here.

### 10-12 D-LOSS CAUSAL FOLLOW-UP — design fixed before training (15:14 AEDT)

**Question:** does D's loss contribute to the remaining oracle-conditioned recursive occupancy errors?
- Existing L1 and D oracle arms differ in pretrained parent; their comparison cannot isolate the loss.
  TRUE-token projection damage is independent of either world/loss and remains a separate measured failure.
- `20261012_oracle/loss_intervention.py`: three arms from the SAME completed oracle-D world and OracleCond checkpoint:
  **D** = mean squared error + 0.1 mean absolute error + 0.01 frame-wise maximum squared element;
  **L1** = mean absolute error; **no_max** = D minus the maximum-element term.
- Train world AND oracle conditioner in every arm, leaving architecture, frozen encoder, exact support-v2 TRAIN rows,
  seed-11 batch/start draws, terminal share, fresh phase AdamW, warmup, BF16/IEEE and 3,000-update budget fixed.
  Batch 16, window 16. Save model snapshots at 0/500/1,500/3,000; full optimizer/RNG/order states every 250 updates.
- Meaningful implementation checks: identical starting world/conditioner weights; D/L1 loss parity against the existing
  implementation; short interrupted/resumed training versus uninterrupted training; source/input hashes and batch ledger.
- Evaluate all snapshots on the existing 200 DEV roots, frozen readers/thresholds: teacher-forced versus H16 recursive
  cow/zombie false-positive occupancy AND recall, terrain/player/HUD fidelity. No token projection. Primary all roots;
  common scroll-aligned rows are sensitivity. Paired episode intervals; reducing counts by erasure is not a repair.
- Separating contrasts: no_max minus D isolates the maximum-element term; L1 minus D tests the full loss switch.
  A reduced excess with preserved completion supports a causal contribution of that loss in this trained world.
  A negative at 3,000 updates is an adaptation-budget result if learning is still moving; it cannot exonerate D for every
  from-scratch recipe. No second training seed or new architecture is implied by this mechanism test.
- CPU complements: exact saved-prediction loss/gradient/correction incentives (`d_loss_fixed_outputs.py`), paired existing
  world comparisons (`d_existing_compare.py`), and model-free means/medians conditioned on identical true future mob
  layouts (`conditional_stat.py`). These distinguish semantic erasure, objective incentives and incomplete conditioning.
- Evidence stays in `artifacts/eda/oracle_v1`; stdout in `artifacts/eda/levers_logs`. Agents record numerical findings here.

### 10-12 PROJECTION FOLLOW-UP — completed (13:29–13:33 AEDT; CPU diagnosis through 13:44)

**Result: projected feedback reduces subsequent cow false positives, but nearest-token projection fails content preservation. Do not integrate it as a repair.**
- Same checkpoints and 60,000-token dictionary as the aborted attempt; 200 DEV episodes per TRUE/oracle-D/control-D arm,
  H16, no retraining. Four modes separate immediate reader filtering from changed subsequent model outputs.
- Primary below: all 200 episodes, depths 9–16 (1,600 root-steps; 870 true cow cells and 583 zombie cells).
  Intervals are paired episode-bootstrap 95% intervals, 2,000 draws. All modes use the same frozen readers and thresholds.

| oracle-D mode | cow false-positive cells / true cows | cow recall | cow census / truth | zombie recall |
|---|---:|---:|---:|---:|
| raw self-fed rollout | 0.569 | 0.924 | 1.493 | 0.842 |
| project only at readout | 0.244 | 0.746 | 0.990 | 0.599 |
| projected history, **output before projection** | 0.289 | 0.905 | 1.193 | 0.731 |
| projected history and output | 0.197 | 0.752 | 0.948 | 0.600 |

- **Feedback effect:** pre-projection cow false positives fall from 495 to 251 (49.3%), a ratio difference
  −0.280 [−0.494, −0.120]. Cow recall changes −0.020 [−0.053, +0.010]; this is not a noninferiority result.
  Zombie recall falls −0.111 [−0.193, −0.035]; skeleton recall falls −0.215 [−0.321, −0.120].
  Depth-1 raw and feedback-pre outputs are byte-identical in every batch, before projected feedback begins.
- **Readout effect:** projecting the unchanged raw rollout lowers cow false positives by −0.325 [−0.532, −0.171],
  while erasing recall by −0.178 [−0.246, −0.121]. Final projected feedback does not resolve beyond readout-only projection
  for cow false positives (−0.047 [−0.147, +0.044]) or recall (+0.006 [−0.056, +0.066]).
- **Common-aligned sensitivity:** 158 episodes contribute 1,112 late root-steps. The pre-projection cow false-positive
  benefit remains (−0.249 [−0.406, −0.112]), with a small resolved cow recall loss (−0.034 [−0.074, −0.004]);
  zombie recall falls −0.145 [−0.230, −0.075]. Primary evidence uses all roots, not arm-specific selected masks.

**TRUE positive control: projection damages genuine tokens.**

| late real-token recall | original | projected | paired change, 95% interval |
|---|---:|---:|---|
| cow | 836/870 = 0.961 | 712/870 = 0.818 | −0.143 [−0.201, −0.088] |
| zombie | 526/583 = 0.902 | 375/583 = 0.643 | −0.259 [−0.352, −0.172] |

- Exact dictionary-source diagnosis distinguishes content replacement from reader errors: 155/158 projected cow
  false negatives and 208/208 zombie false negatives select bank entries whose original physical source lacked that mob.
  Actual source-content preservation is 711/870 cows (81.72%) and 372/583 zombies (63.81%).
- All 201,600 projected TRUE tokens match bank bytes; all 60,000 bank entries match their recorded FIT-cache coordinates;
  duplicate token bytes have no terrain/mob label ambiguity.
- **Support versus metric:** suitable nonplayer bank entries with the same terrain and all four mob flags exist for
  every semantic loss across H16 (300 cow losses, 369 zombie losses). Same-slot suitable entries exist for 260/300 and
  223/369 respectively. On a fixed sample of 64 losses per type, the selected wrong token is closer in L2 than the nearest
  suitable token in 64/64 cases for both types; median correct/chosen distance ratio is 1.228 (cow), 1.499 (zombie).
  Total dictionary absence is excluded for these losses. This diagnoses this L2 selector; it does not isolate context,
  position, light or sampling density as the sole cause of the metric mismatch.
- Player-source contamination is a separate limited failure: 71/198,400 nonplayer queries retrieve player-slot entries,
  accounting for only 5/300 cow and 14/369 zombie semantic losses. Player facing-reader accuracy itself drops
  0.896 → 0.822; 110/3,200 player queries retrieve nonplayer entries. HUD tokens remain byte-identical under projection.
  Across all TRUE future frames, terrain-reader accuracy drops 0.960 → 0.945 (near-player 0.970 → 0.947).
- **Unconditioned control:** late cow recall is 0.079 raw, 0.033 under projected history before projection,
  and 0.020 after projection. Projection does not recover the missing stochastic outcome.

**Interpretation:** altered fed-back tokens causally reduce later oracle-D cow false positives under the fixed reader,
but preservation fails on real inputs and on other mobs. Per-token membership in a real dictionary does not ensure a
valid assembled frame. This result supports feedback sensitivity; it establishes neither categorical-state necessity
nor a learned stochastic innovation. The prior oracle supplies full future occupancy, so its positive completion result
is conditional on that privileged information. No actor/control result follows from this test.

**Execution and evidence:**
- Blocked exhaustive FP64 search completed with 0.659 GiB peak GPU allocation; original OOM script/log preserved.
  Reference-distance/chunk/duplicate/idempotence/HUD checks and helper syntax checks pass.
- One-batch interruption/reload check passed. Full re-execution resumed 75 batches, recomputed zero, and left all seven
  result hashes unchanged (`oracle_v1/snap_resume_proof.json`). No experiment job remains running.
- Re-evaluated raw oracle-D census is 1.493 versus the earlier rounded 1.500; it is not a byte-exact reproduction of
  that earlier execution. Every intervention contrast above uses the matched baseline from this run.
- Source: `20261012_oracle/snap_eval.py`, `snap_analysis.py`, `snap_visible.py`, `snap_bank_diagnose.py`,
  `snap_bank_support.py`, `run_snap_followup.sh`. Evidence: `oracle_v1/stats_{TRUE,oracle_deltairis,control_deltairis}.json`,
  `snap_analysis.json`, `snap_bank_diagnose.{json,pt}`, `snap_bank_support.{json,pt}`, saved root batches and exact replay.
  Stdout: `artifacts/eda/levers_logs/oracle_snap_followup_20261012.log`.

### 10-12 PROJECTION FOLLOW-UP (13:27; post hoc, comparisons fixed before the resumed run)

**Question:** does real-token projection improve subsequent predictions, or merely clean the fixed reader's output?
- Preserve the original 12:14 CUDA-OOM log and script (`oracle_v1/snap_eval_failed_20261012_1214.py`).
- Same 200 DEV roots, oracle/control D checkpoints, 15-frame context, H16, and 60,000 real map tokens sampled with seed
  20261012 from the diagnosis corpus's FIT seeds. No training or threshold changes. This dictionary is diagnostic FIT data,
  not the world's original training-window pool.
- `20261012_oracle/snap_eval.py`: exhaustive nearest-neighbour L2, query/bank blocks 256/8,192 and FP64 distances;
  selected bank values remain FP32. All 63 map tiles (including player slot) project; HUD values are unchanged.
- Modes: raw rollout; projection at readout only on the same raw rollout; **pre-projection outputs under projected feedback**;
  projected outputs actually fed back. TRUE raw/projected is the semantics-preservation control.
- Primary: all roots, depths 9–16, false-positive cow cells / true cows with recall and census; separating contrast is
  feedback-pre minus raw. Readout minus raw isolates immediate filtering. Shared scroll-aligned subset is sensitivity only.
- Same frozen readers/thresholds; retain per-root/depth event counts and paired episode-bootstrap intervals (2,000 draws).
  No passing claim from reducing false positives by erasing true mobs. Bank membership is not joint-frame validity.
- Exact CPU replay adds terrain/HUD/facing/sleep ground truth for the same 200 roots: mob layouts match at every root;
  9 sentinel frames match pixels exactly. `snap_visible.py`, `oracle_v1/snap_visible.pt`.
- Checks: blocked search vs direct exhaustive distances, duplicate tokens, projection idempotence, byte-identical HUD,
  exported readers vs originals, depth-1 pre-projection identity, saved/reloaded predictions equal. Atomic per-root-batch
  checkpoint files bind source/input/bank/reader/checkpoint hashes and resume without recomputing completed batches.
- CPU semantic/paired analysis is `snap_analysis.py`; experiment stdout goes to `artifacts/eda/levers_logs`, never this file.

### 10-12 ORACLE-INNOVATION TEST — predeclared (before any code): can a continuous JEPA world complete a transition coherently when the stochastic outcome is GIVEN?

**User's questions (10-12):**
1. Does providing the correct stochastic outcome eliminate the continuous-model failures we attributed to averaging?
2. Can the innovation be learned without privileged labels?
3. If completion succeeds but recursion fails, what is the minimum structural property that preserves valid transitions?

**Literature read first:**
- Stochastic MuZero (afterstate + categorical chance code; posterior chance encoder + learned prior σ; value-equivalent).
- JEDI (whole-latent diffusion, reconstruction-free).
- Causal-JEPA (arXiv 2602.11389): object-level masking on a frozen slot encoder; reconstruction-free; deterministic predictor;
  near-deterministic tasks.
- **LPWM (arXiv 2603.04553, ICLR 2026 oral):**
  - a per-particle latent action (Gaussian), inferred by a latent inverse-dynamics head from consecutive frames, with a learned
    latent-policy prior and KL;
  - patch-anchored particles that keep their identity;
  - reconstruction-trained;
  - states that "a global action vector cannot naturally disentangle these local dynamics", which matches our factorization
    result.

**Q1 design (oracle completion):**
- **World:** our canonical L16 corrt / fmamba per-tile world on frozen Raw tokens, through the existing per-tile conditioning hook
  (`TWorld.delta`, as E22).
- **Oracle:** at input position t, each map tile gets the TRUE mob layout of frame t+1 at that tile, through a zero-initialized
  Linear(4 → 192). Channels: zombie / cow / skeleton / arrow, from exact replay of the training episodes; target-slot
  coordinates.
- **Arms**, 2 losses × {control, oracle}; continuations of the existing worlds; matched sampler (E22's long sampler, support-v2
  rows only, expert rows excluded in all arms), batch order seed 11, 16 windows, 3,000 updates:
  - L1: from M16;
  - D loss: from D scratch.
- **Evaluation panel:** 200 roots from support-v2 DEV-split episodes (never trained on). Frames t−15..t+16 from the stored
  observations (exact); oracle layouts by replay.
  - Decoding: scoreboard.AllProbes (fitted on true TRAIN tokens of the diagnosis panel).
  - Teacher-forced depth 1 (window 15 true frames) and self-fed 16 steps (true layouts supplied each step).
- **Metrics per depth:**
  - decoded mob census / true (cows, zombies);
  - mob-cell precision / recall vs the true layout;
  - **ghost rate:** decoded cow where the input frame had a cow and the truth now has none (source residue);
  - **duplicate share** (cowdiag classes vs the input frame).
- **Reading Q1:** conditional completion succeeds if both oracle arms at depth 1 have:
  - cow and zombie census within 10% of the truth;
  - ghost rate ≤ 2% of true cows;
  - mob recall ≥ 0.9.

  The controls show their losses' signatures (erasure under L1, superposition under D).
- **Reading recursion:** recursive coherence succeeds if the oracle arms, self-fed, have the census within 20% and ghost rate
  ≤ 2% at every depth on scroll-aligned steps.
- **Next:** only if Q1 succeeds, Q2 (a learned per-tile innovation, no labels) and the recursion analysis follow.

### 10-12 FEASIBILITY RESULT: history resolves the counters; the fresh randomness is ~1.4 bits per visible mob and factorizes; a coherent mode-world carries 75-88% of the decision value and 16 samples add a resolved +0.025

`20261012_feasibility/feasibility.py` (q1-q3), `breakdown.py` and `factorize.py` (both POST HOC). Data in
`artifacts/eda/feasibility_v1/`. CPU only, about 30 min of compute in total.

**Checks:**
- 240 roots, one per support-v2 episode, from 240 different shards; replayed renders equal the stored frames (max pixel
  difference 0).
- My counter equations reproduce the simulator exactly (max error 0.0) after two bugs in my code, both caught by these checks
  and fixed:
  - eating on a threshold-crossing step;
  - eating while the counter is already 0.
- The tracker never eliminates the true value (0 violations).

**Q1, history** (fraction of roots whose next-16-step VISIBLE HUD schedule is determined; recover assumes damage events are
observable):

| counter | L = 1 | 4 | 15 (our window) | 32 | 64 | full prefix |
|---|---|---|---|---|---|---|
| hunger | 0.06 | 0.17 | 0.52 | 0.98 | **1.00** | 1.00 |
| thirst | 0.08 | 0.20 | 0.68 | 0.99 | **1.00** | 1.00 |
| fatigue | 0.03 | 0.11 | 0.47 | 0.79 | **1.00** | 1.00 |
| recover | 0.43 | 0.47 | 0.61 | 0.74 | 0.78 | **1.00** |

- The predeclared reading (≥ 90% at L ≤ 64 for every counter) holds for 3 of 4.
- Recover needs the episode: its resets at full health are invisible, so the phase must be counted from the last visible
  event. That is still memory, not randomness.
- At our 15-frame window, 47-68% of schedules are determined. This is the measured size of the timer failure.

**Q2, one-step fresh randomness** (from the full simulator state; 256 keys × 17 actions; visible symbolic observation):
- Miller-Madow entropy: median 1.68 bits, p90 4.04, p99 6.72.
- K = 64 gives a median of 1.67 (stable); the tail is under-resolved.
- Good-Turing missing mass 1.7%.
- 10% of root-actions are deterministic.
- ≤ 8 outcomes cover 95% in 75% of root-actions.
- **The predeclared "compact" reading is FALSE** (p90 4.04 > 4; coverage 75% < 90%).
- Full world (off-screen mobs included): median ≥ 7.0 bits; saturated, so a lower bound.
- **Post hoc, where it lives:** entropy grows with the visible mobs: 0 mobs → 0.33 bits, 1 → 1.67, 2 → 2.61, 3 → 3.68, 4-6 →
  5.5-6.7; about 1.37 bits per visible mob. It barely depends on the action (noop 1.84, move 1.87, DO 2.01).
- **Post hoc, factorization** (74 roots with ≥ 2 visible mobs, 1,258 root-actions):
  - the joint entropy of the visible mobs' next (position, alive) is 3.25 bits vs 3.38 for the sum of the marginals;
  - total correlation 0.13 bits (3.8% of the sum; p90 0.39);
  - spawns and new arrows add 0.14 bits.
- **The innovation is compact per entity and near-independent across entities.** Frame-level coverage fails through
  combinatorics, not through a large innovation.

**Q3, decision value** (horizon 16, recorded continuation; independent 32-key selection / judgment banks; 77 opportunity roots
of 240; root bootstrap):

| | uniform | coherent mode-world | 1 sample | 4 | 16 | 32 |
|---|---|---|---|---|---|---|
| expected safe at 16 (opportunity roots) | 0.790 | 0.867 | 0.830 | 0.878 | 0.893 | 0.893 |
| native 16-step return (all roots) | 0.160 | 0.447 | 0.420 | 0.458 | 0.483 | 0.486 |

- S16 − mode **+0.025 [+0.009, +0.045]**: the predeclared reading "sampling matters beyond coherence" holds. Return:
  +0.036 [+0.024, +0.050].
- The mode-world captures **75%** (survival) and **88%** (return) of the 32-sample margin over uniform.
- **One sample is WORSE than the mode-world:** −0.038 [−0.071, −0.013] safe; −0.027 return.
- S32 ≈ S16 (+0.001); S4 already beats the mode-world.

**Reading against the hypothesis:**
1. **History:** confirmed, quantitatively. The counters are memory problems; 64 frames determine three of them and recover needs
   the episode.
2. **Compact innovation:** confirmed **per entity** (~1.4 bits per visible mob, near-independent), refuted as a frame-level
   joint code. This points the chance variable at entities (mobs). Coherence then means entity identity and correspondence
   (ITC's per-creature assignment fits this).
3. **Decisions:** coherence is first-order (75-88% of the value); integrating 4-16 samples is second-order but resolved.
   - One sample per decision is worse than a coherent deterministic future. This matches E22MS (one sample 0.630 < D 0.670).
   - Caveat: an actor trained on many imagined trajectories integrates over samples across rollouts, so the one-sample
     penalty applies to per-decision selection.

**Limits:**
- Forks condition on the full simulator state. Q1 covers the counters, not off-screen mobs.
- The mode-world is defined on the visible observation (its mode is "cows stay, zombies chase").
- The continuation is open-loop (recorded).
- Roots come from the support-v2 policy mix.
- Plug-in entropies at K = 256 (tails under-resolved; the joint is biased low, so independence is if anything understated).

### 10-12 FEASIBILITY STUDY — predeclared (before any code): is the whiteboard's hypothesis promising?

**Hypothesis (WHITEBOARD §8, after Codex's corrections):** persistent history resolves the hidden-counter uncertainty; the
remaining fresh randomness per step is compact; and integrating over it matters for decisions beyond a coherent
deterministic future.

**Scope:** a feasibility version of Codex's proposed matched experiment (the user judged its execution over-engineered).
- ~240 roots, one per support-v2 episode, from different shards; root t ∈ [64, len − 17].
- Exact replay from stored actions and keys (`artifacts/eda/replay.py`); simulator only, no learned model.

**Checks:**
- replayed renders equal the stored frames;
- my counter equations reproduce the simulator's counters at every replayed step;
- the true counter value is never eliminated by the tracker.

**Q1, history.**
- Method: a source-derived candidate tracker for hunger / thirst / fatigue / recover, started L steps before the root (L = 1,
  4, 15, 32, 64, full prefix).
- Treated as observable: HUD values, the sleep flag, eat / drink events (DO on a visible cow / plant / water), damage events.
- Invisible: threshold resets while the stat is clipped.
- Reports: fraction of roots whose counter value is identified, and whose next-16-step HUD event schedule is determined.
- **Reading:** memory suffices if ≥ 90% of schedules are determined at L ≤ 64 for every counter.

**Q2, one-step innovation.**
- Method: from each root state, 256 keys × 17 actions, one step. Joint outcomes are hashed on the symbolic visible observation
  (no renderer noise) and on the full world (all mobs incl. off-screen, map, inventory, counters).
- Reports: Miller-Madow entropy (with K = 64 vs 256 as a bias check), Good-Turing missing mass, outcomes needed for 95 / 99%
  coverage.
- **Reading:** compact if the median visible joint entropy is ≤ 2 bits, the 90th percentile ≤ 4 bits, and ≤ 8 outcomes cover
  95% in ≥ 90% of root-actions.

**Q3, decision value.**
- Method: horizon 16, recorded continuation (as H16), the same roots, two independent 32-key banks (selection / judgment),
  common random numbers across actions within a bank.
- Compared estimators:
  - uniform;
  - a coherent deterministic **mode-world**: at each step take the most frequent next visible observation of 64 forks and
    continue from that state;
  - 1 / 4 / 16 sampled futures;
  - 32 sampled futures (high-precision reference).
- Measures: expected safe at 16 (judged on the judgment bank) on opportunity roots, plus native 16-step return.
- **Reading:** sampling matters beyond coherence if S = 16 − mode-world is resolved > 0 (paired root bootstrap).

### 10-11 WHITEBOARD (21:50): the formalization lives in `artifacts/experiments/WHITEBOARD.md` (no runs, no plan)

What it adds to the 20:30 review, each item verified at source:
- **Ni et al. (arXiv 2401.08898).** A sufficient state needs RP + ZP: the next-latent DISTRIBUTION is predictable. Predicting
  the expected next latent (EZP) is the weak version; this is the third party's V(E[z′]) ≠ E[V(z′)]. Our worlds are below EZP
  (coordinatewise median / mean).
  - Their guidance: noisy → self-predictive, sparse reward → observation-predictive, deterministic → ℓ2. Craftax is all three.
- **Stochastic MuZero (Antonoglou et al., ICLR 2022)** = the formal "innovation, not state" model: a deterministic afterstate, a
  categorical chance code (one-hot codebook, no decoder), b′ = g(afterstate, c).
  - Craftax's own `craftax_step` has exactly this order: player action / move (deterministic but a 10% sapling draw) → mob moves
    + spawns (chance) → plants / counters / health (deterministic) → a fresh `state_rng` for the renderer only.
  - If c carries all of the randomness, g is deterministic and a point loss on g is correct: the axis is
    sufficient-chance + coherent g, not categorical vs continuous.
- **The chance is small:** ≲ 1.3 bits per step (source movement rules × 0.77 visible mobs per frame; an upper bound), with
  1.71 cells per step differing across keys (20:30).
- **Our own evidence for the four-row taxonomy (nuisance / forgotten history / aleatoric / epistemic):**
  - E22FJ's unstructured stochastic channel spent 92% of its gain on never-seen edge terrain (random zombie moves 0-3%);
  - only the moved-content mask (structure, not weight) got the aleatoric part (86%).
- **Timers are deterministic counters with observable resets** (source): pure memory, no randomness.
- **Memory addressing, from our record (E18 / fcanvas):** world-aligned writes failed through the screen-aligned READ (entering
  cell read 0 / 2,018). Neural Map separates addressed writes and reads.
  - In imagination the address is the world's own scroll decision, the one discrete instability (chaos test).
- **Reconstruction:** sided with the user (no reconstruction). The obligation is named: retain fast, sparse-reward content via
  locality and consequence heads in the loss. Whether that suffices is open and checkable with existing probes.
- **Papers the third party cited, checked:**
  - JEDI is reconstruction-free and end-to-end (Atari100k + Craftium, 38 A100-h per run);
  - Flow-JEPA's tasks are near-deterministic (a robustness gain);
  - VJEPA's experiment is a toy;
  - Fast-LeWM is deterministic.

### 10-11 REVIEW (20:30): Codex's and a third party's critiques of the 19:30 synthesis, checked against sources

**Corrections accepted (verified):**
- **Dreamer-CDP's 6.3 vs 16.2 removes the prior / posterior KL alignment (Ldyn/rep), NOT the categorical latent** (paper l.229).
  My 19:30 reading was wrong. Its 12.7 (no reward gradient) stands.
- **"Categorical targets were the load-bearing missing part" overclaims.** What our evidence requires is a distributional
  transition whose samples are valid, coherent states. Per-tile categorical CE is one route (Dedieu, ITC on our exact
  frames); continuous latent diffusion / flow is another, already published reconstruction-free:
  - JEDI (2605.13013): end-to-end JEPA latent diffusion world model, Atari100k, 38 A100-h per run;
  - Flow-JEPA (2608.29029); VJEPA (2601.14354).
- **Independent per-tile sampling breaks coherence.** ITC (2605.16457), on Craftax-classic, built on Dedieu's per-tile tokens:
  - parallel independent readouts duplicate and erase creatures;
  - a one-to-one copy-or-generate assignment lifts return 67.4 → 72.5% and creature-step accuracy 33.8 → 38.4%.
  - For an unobstructed cow (Craftax: 50% stay, 4 × 12.5% moves), independent sampling from PERFECT marginals gives the wrong
    cow count 57% of the time (P(0) 0.29, P(≥2) 0.28).
- **Per-tile CE bounds each tile's logit gradient, not its share of parameter updates;** "~4% of the gradient" withdrawn.
- **E22MS's ghost copies:** the cause ("continuous decoding of what the codes leave unresolved") is labelled
  "Inference (not measured)" at its source; 19:30 promoted it. Measured: source residue + recursive duplication with true codes.
- **Dedieu's 21.2 vs 67.4:** our statoracle gives a compatible mechanism; it does not isolate their gap (their continuous arm
  predicts pixels, and its training differs).

**New measurement (third party's question B: how large is the truly random part of a transition?).** 5 futures from the
same state, visible simulator state, all alive:

| depth | cells whose state differs (of 63) | terrain class differs | mob differs | HUD values differ | inventory differs |
|---|---|---|---|---|---|
| 1 | **1.71 (2.7%)** | 0.00 | 1.71 | 0.1% | 4.2% |
| 2 | 2.48 | 0.22 | 2.28 | 1.4% | 8.8% |
| 4 | 3.98 | 1.12 | 2.97 | 3.5% | 15.0% |

- Per step, the world's randomness is ~1.7 cells, all mobs.
- Terrain is deterministic (never-seen terrain is epistemic: fixed but unknown).
- The HUD is deterministic given the hidden counters.
- The third party's questions A (memory vs randomness for timers) and C (which content decisions read) were already answered
  by the 10-10 timer analysis and the region substitution / check_h16_value.

**Not accepted:** a reconstruction decoder for tokenizer pretraining (Codex) conflicts with the project's reconstruction-free
thesis. In Craftax a stable context-free per-tile vocabulary needs no decoder (Dedieu's NNT is a fixed lookup).

### 10-11 SYNTHESIS (19:30): our results against the literature, and the plan they point to (no new runs)

**The pattern we found on 09-25 and partly dropped:** every Crafter/Craftax world model that works has categorical
per-position targets, and reward / termination heads inside its loss (readout ladder PROPOSAL.md §3). We kept the per-tile
state and dropped the categorical target, because over the ViT's contextual tokens a tile's code flips when other tiles change.
The 10-10 diagnosis shows the dropped part was the load-bearing one.

**Connections:**
1. **Continuous per-tile regression → non-state statistics.**
   - statoracle: the median erases uncertain mobs (cows 0.08 vs 0.52 at depth 16); the mean superimposes them.
   - Dedieu et al. (arXiv 2502.01591, our exact frames): 7×7 continuous per-patch prediction 21.20 vs 67.42 quantized.
   - Agrees; our measurement is the mechanism behind their ablation.
2. **E22MS's faint copies are the same failure one level down.**
   - Its decoder regresses continuous tokens given the codes. What the codes leave unresolved (the posterior gets 86% of
     random moves) is averaged, which leaves a copy at the source (cowdiag: 10% per step with TRUE codes vs 1% real).
   - ~~Dreamer-CDP (2603.07083): reconstruction-free needs the categorical latent (6.3 without vs 16.2).~~ WRONG: the 6.3 arm removes the prior/posterior KL alignment (see 20:30 review).
3. **Allocation.**
   - Per-tile cross-entropy bounds each tile's gradient regardless of token distance.
   - Never-seen entering tiles are 761 of ~17,500 map cells at depth 1 (~4%), vs 56-81% of the D loss's update now.
   - Decision heads in the loss: Dreamer-CDP without the reward gradient 12.7 vs 16.2. Our E21 oracle: a label gradient into
     the world installed the cooldown phase (1.000) within 500-1,000 updates.
4. **Why the codes flipped on 09-25:** encoder_why located the context (position embeddings + global attention; HUD tokens move
   9-12 with identical pixels). Dedieu's codes are per-tile, hence context-free.
   - V-JEPA 2-AC / DINO-WM's contextual patch targets work there because a patch-level objective trained them, in
     near-deterministic tasks; neither holds here.
5. **Correction (to my 10-11 15:2x reply):** sampling IS where the decision headroom is.
   - D (0.670) sits at one real future (0.663 FIT+DEV / 0.679 DEV-B); 16 faithful futures reach 0.727 (check_h16_value).
   - E22MS's samples are not faithful enough.
   - Decisions read the HUD, but the HUD's future depends on the zombie map (true map restores 58.5% of damage, 10-09).

**Nearest prior work (short search):**
- Discrete-JEPA (2506.14373): quantizes a global semantic token; patches stay continuous; not a world model.
- DWMR (2603.01748): a global Boolean state, deterministic fully observable puzzles.
- Dedieu: pixel nearest-neighbour codes (no learned encoder).
- Nothing found that learns per-tile categorical targets without reconstruction for a stochastic world model. A proper sweep is
  needed before claiming novelty.

**Plan (pitched to the user 10-11):** a per-tile categorical JEPA world model.
- A local per-tile encoder (no cross-tile attention), reconstruction-free, SIGReg on its embeddings, a categorical code per tile.
- Per-slot Mamba (state carried through imagination, windows ≥ 32), corrt routing over code logits, cross-entropy.
- Reward / termination / health heads in the loss; sampled codes in imagination.
- First, CPU: Dedieu's NNT on our diagnosis frames as a reference target (codebook size day / night; do codes change only when
  the tile's state changes?).

### 10-11 RESULT (18:35): D continued H16 — the second D-loss world replicates the snapshot-head gain over L1

`check_h16_traj.py --window 15` (lane183, 14:57-17:26). DEV-B: 1,139 opportunity roots, 149 episodes. Paired
episode-clustered 95% intervals; seed 20261010.

| | trajectory head | snapshot head | Brier |
|---|---|---|---|
| D continued | 0.664 | 0.645 | 0.085 |
| D continued − M16 s7 | +0.007 [−0.004, +0.018] | **+0.022 [+0.014, +0.032]** | |
| D continued − M16 s8 | **+0.017 [+0.007, +0.027]** | **+0.025 [+0.012, +0.037]** | |
| D continued − D scratch | −0.006 [−0.016, +0.005] | +0.001 [−0.011, +0.013] | |

- The snapshot-head gain of the D loss over L1 holds in both D worlds against both L1 seeds: +0.021 to +0.025, all resolved.
- On the trajectory head, both D worlds resolve against M16 s8 and not against M16 s7.
- D continued is NOT an independent seed: it continued from M16 s7.

### 10-11 RESULT (15:05): E22MS H16 with sampled futures — averaging 16 samples reaches parity with the deterministic worlds, not more

**Setup:** `check_h16_traj.py --e16 --window 15` (lane171 10-10 10:21-17:09, paused at draw 6; resumed in lane183 10-11
10:24-14:57).
- DEV-B: 1,139 opportunity roots, 149 episodes.
- Heads are fit on one sample's trajectory features to the 32-key P(dead by k | root, first action). DEV-B is re-imagined to 16
  samples per action, and the predicted risk is averaged over the first M samples.
- Per-root safe rows are recomputed from the saved draws: M = 1 reproduces the logged 0.6322 exactly.
- Paired episode-clustered 95% intervals.

| E22MS | expected safe | vs D scratch (0.670) | vs M16 s7 (0.657) | vs M16 s8 (0.647) |
|---|---|---|---|---|
| 1 sample | 0.632 | −0.038 [−0.051, −0.024] | −0.025 [−0.040, −0.010] | −0.015 [−0.030, +0.001] |
| 4 samples | 0.647 | −0.023 [−0.034, −0.011] | −0.010 [−0.024, +0.005] | +0.000 [−0.013, +0.014] |
| 16 samples | **0.663** | −0.007 [−0.018, +0.005] | +0.006 [−0.008, +0.020] | **+0.016 [+0.004, +0.029]** |

- Within E22MS: 16 − 1 = **+0.031 [+0.021, +0.043]**; 16 − 4 = +0.016 [+0.008, +0.025].
- Other readings: trajectory arm (1 sample, 3 head seeds) 0.630; snapshot 0.615. The prior's own end head gets worse with more
  samples (0.570 / 0.551 / 0.536); this is the terminal-position defect, reported, not judged.
- **Continuation:** Brier of P(dead by k) **0.050**, the best of all worlds (D scratch 0.071; M16 0.093-0.127). AUC 0.904.

**Reading:**
1. **Averaging sampled futures pays within the sampler (+0.031, monotone in M)**, as the 10-03 analysis predicted for faithful
   futures.
2. **But one E22MS sample decides worse than every deterministic world,** and 16 samples only reach parity:
   - tied with D scratch and M16 s7, ahead of M16 s8;
   - at 16× the imagination cost.
3. **The proper-score advantage (whole-frame ES −2.2 at depth 1, −13.8 at 16) does not become a decision advantage over the
   best deterministic world.**
4. **Calibration vs choice:** E22MS has the best-calibrated death forecasts ACROSS roots (Brier), but not the best choice AMONG a
   root's 17 actions. The decision depends on within-root discrimination (cf. the probe-objective rule).
5. **Not separated yet:** why its single sample is worse.
   - Candidates: the decoder failures (cowdiag; 5.7× copy error on still cells) or the protocol (heads fit to a ROOT-level
     target on ONE sample's features).
   - Decisions are ~90-97% HUD (region relevance), so the candidate that matters is what its samples do to imagined HEALTH.
     That was not measured for E22MS.

### 10-10 RESULT (20:40, recorded 10-11 09:4x): decision relevance by region, M16 at window 15 — the HUD carries 97-100% of what imagination misses; the map carries none

**Setup:** `check_h16_subst.py --window 15` (lane178; the machine rebooted after 20:44, before the D-scratch arm, which is
relaunched in lane183).
- deepeval's DEV split: 3,096 roots; opportunity roots 640 / 1,309 / 2,590 at k = 1 / 4 / 16.
- Its real-fitted depth-k heads read the imagined state with one region replaced by the real depth-k tokens.

| M16 (w15), expected safe | k = 1 | k = 4 | k = 16 |
|---|---|---|---|
| imagined | 0.570 | 0.678 | 0.591 |
| + real HUD (tokens 63-80) | 0.999 | 0.737 | 0.688 |
| + real 3 × 3 around the player | 0.576 | 0.679 | 0.596 |
| + real map (0-62) | 0.622 | 0.642 | 0.572 |
| + real edge ring / interior | 0.605 / 0.596 | 0.676 / 0.676 | 0.592 / 0.589 |
| real | 0.999 | 0.738 | 0.691 |
| HUD share of the gap / map share | **1.00** / 0.12 | **0.98** / −0.59 | **0.97** / −0.19 |

- Imagined health at depth 16: mean 5.98 vs real 2.42. Danger correlation 0.20 vs real 0.54.
- **This reproduces the 10-03 window-5 result (HUD share 0.94-0.97) on the L16 world at window 15.**
- **For decisions, what imagination must get right is the drawn outcome in the HUD (damage, death), not the map.**
  - The edge ring, which holds the largest proper-score gap (61-71% at depth 1; mostly missing information), has ~0 decision
    relevance.
  - The decision-relevant subset of the problems is what drives health: the L1 stall on HUD changes (allocation); the hidden
    zombie cooldown and the timers outside the window (information); and mob movement only through who becomes adjacent.
- **D-scratch arm (lane183, 10-11 09:35-10:23):**

  | expected safe | k = 1 | k = 4 | k = 16 |
  |---|---|---|---|
  | D scratch, imagined | 0.534 | 0.690 | **0.616** |
  | + real HUD | 0.999 | 0.736 | 0.684 |
  | + real map / edge / interior | 0.618 / 0.592 / 0.577 | 0.656 / 0.692 / 0.692 | 0.619 / 0.618 / 0.616 |
  | HUD share / map share | 1.00 / 0.18 | 0.96 / −0.70 | 0.90 / 0.04 |

  - Imagined health at depth 16: 5.46 (M16 5.98; real 2.42). Danger correlation 0.33 (M16 0.20; real 0.54).
  - **The D loss's decision gain flows through the HUD:**
    - its depth-16 imagined state is 0.616 vs M16's 0.591, so the gap to real shrinks from 0.100 to 0.076 (−24%);
    - the HUD still carries 90% of what remains;
    - its imagined health is lower and better tied to danger.
  - **Even this world imagines 3.0 more health than reality at depth 16.** Drawn damage in imagination is the decision
    bottleneck for both worlds.
  - At k = 1 D scratch's imagined state is worse than M16's (0.534 vs 0.570; 640 roots). The real HUD restores both to 0.999.

### 10-10 RESULT (20:50): H16 decisions, D scratch vs M16 — the D loss's factual gains carry into decisions (single D seed)

**Setup:** `check_h16_traj.py --window 15` (lane170, 17:10-20:44); DEV-B 1,139 opportunity roots, 149 episodes, 3 head seeds.
Paired, episode-clustered 95% intervals (2,000 draws) on the per-root safe rows saved in each resume store.

| world | trajectory | snapshot | traj − snapshot | Brier, P(dead by k) | AUC dead by 16 (key 0) |
|---|---|---|---|---|---|
| D scratch (D loss, seed 7) | **0.670** | **0.644** | +0.026 [+0.015, +0.037] | **0.071** | **0.898** |
| M16 seed 7 (L1) | 0.657 | 0.623 | +0.035 [+0.022, +0.048] | 0.127 | 0.823 |
| M16 seed 8 (L1) | 0.647 | 0.621 | +0.026 [+0.013, +0.040] | 0.093 | 0.867 |

References: prior 0.616, one real future 0.679, oracle31 0.766, uniform 0.571.

| paired difference | trajectory | snapshot |
|---|---|---|
| D scratch − M16 s7 | +0.013 [−0.000, +0.026] | **+0.021 [+0.008, +0.035]** |
| D scratch − M16 s8 | **+0.023 [+0.012, +0.035]** | **+0.024 [+0.011, +0.037]** |
| M16 s7 − M16 s8 (seed noise) | +0.010 [+0.002, +0.019] | +0.002 [−0.008, +0.014] |

- **D scratch decides better than M16.**
  - Resolved on the snapshot head against both L1 seeds, and on the trajectory head against seed 8.
  - Borderline against seed 7's trajectory head (lower bound −0.0004).
  - Its death forecasts are better calibrated (Brier 0.071 vs 0.093-0.127).
  - It closes 86% of the prior → one-real-future gap on the trajectory head (M16: 65% / 49%).
- **The link:** the D loss learns HUD changes (−32% vs −1.5%), own-state changes (−46% vs −26%) and zombies (AUC 0.96 vs 0.82)
  (P-D). Those gains reach the decisions, despite its cow superposition and its 2.5× copy error on still cells.
- **Caveats:**
  - a single D seed;
  - M16's own seed spread is 0.010 (trajectory);
  - D continued (lane170 next) is the second D-loss world, but not an independent seed.
- E22MS (1 sample): 0.630 / 0.615, below both. Its 16-sample readout is pending (lane171b).

### 10-10 RESULT (20:05): E22MS cow inflation, mechanism — the decoder leaves faint copies at the source; faint cows persist and duplicate again

**Setup:** `cowdiag.py`, TEST roots, self-fed rollouts.
- Every decoded cow at step k (probe score > 0.164, calibrated on true tokens) is classified against the same sequence's
  previous frame, after its own scroll:
  - persisted: a cow in the same cell;
  - moved: a cow at a 4-neighbour that vacated;
  - duplicated: a cow at a 4-neighbour that is STILL there;
  - created: nothing nearby;
  - entering.
- Shares, with mean probe score in parentheses:

| variant | depth | cows / frame | persisted | moved | duplicated | created | entering |
|---|---|---|---|---|---|---|---|
| real (future 0) | 1 / 9-16 | 0.44 / 0.49 | 0.54 (0.89) / 0.52 (0.90) | 0.31 / 0.34 | **0.01** / 0.01 | 0.11 / 0.09 | 0.02 / 0.05 |
| E22MS, TRUE codes | 1 / 9-16 | 0.57 / **2.36** | 0.35 (0.82) / 0.58 (**0.40**) | 0.42 / 0.22 | **0.10** (0.47) / 0.10 (0.40) | 0.10 / 0.08 | 0.02 / 0.02 |
| E22MS, prior samples 0-3 | 1 / 9-16 | 0.61-0.66 / 2.93-3.02 | 0.26-0.32 / 0.61-0.62 (0.42-0.43) | 0.32-0.47 / 0.19-0.20 | 0.04-0.13 / 0.11-0.12 | 0.15-0.20 / 0.06 | 0.03-0.05 / 0.02 |
| E22MS, Δ zeroed | 1 / 9-16 | 0.97 / 0.56 | 0.12 (0.33) / 0.77 (0.26) | 0.66 / 0.11 | 0.16 / 0.06 | 0.06 / 0.02 | 0.01 / 0.03 |
| D scratch | 1 / 9-16 | 0.92 / 0.47 | 0.39 (0.56) / 0.79 (0.21) | 0.01 / 0.05 | **0.58** (0.20) / 0.04 | 0.01 / 0.05 | 0.00 / 0.08 |

- **The method is clean on real sequences:** 1% duplicated, cows decode at 0.9.
- **E22MS, with the TRUE posterior codes:**
  - 10% of its cows each step are duplicates: the source was not vacated when the cow moved;
  - its cows are half-strength (0.40-0.47 vs 0.90 real) from depth 2 on;
  - those faint cows PERSIST (58-62% of cows) and are duplicated again, so the count grows: 0.57 → 0.86 → 1.34 → 2.36 per frame
    over depth groups 1 / 2-4 / 5-8 / 9-16.
  - The prior samples behave the same (2.9-3.0 at 9-16). **The inflation is the decoder plus feedback, not the prior's
    codes.**
- **The deterministic decoders show the mean optimum at step 1, then dilution:**
  - D scratch: 58% of its depth-1 cows are "duplicates", i.e. the cow both stays and moves (statoracle's superposition), at
    score 0.20;
  - these ghosts fade toward the threshold (0.21 at 9-16) and the count falls (0.92 → 0.47);
  - the Δ-zeroed decoder does the same (0.97 → 0.56).
- **Reading:**
  - Given a movement code, the E22MS decoder renders the destination but not the vacated source fully.
  - Its outputs are partial cows that do not fade, so self-feeding compounds them.
  - **Inference (not measured):** this is the reconstruction loss's conditional mean over what a quantized Δ code leaves
    unspecified. Codex's open question, which of decoding, correspondence and feedback causes the inflation, now has a
    measured answer: decoding (source residue, 10% per step vs 1% real) + feedback (faint residue persists and re-duplicates).
    The prior is not required.

### 10-10 RESULT (19:50): checks prompted by the user's questions and Codex's revision — corrections to Problem 1 and to the currency figures

**1. Is the contextual re-encoding predictable, and predicted?** (`reenc_predicted.py`, depth 1, true context,
factual future, day only)

Share of the target's own change that each world predicts (1 − world error / copy error):

| | unchanged cells, scroll (change 1.50 / token) | HUD values unchanged (jitter 0.56 / token) | unchanged cells, still (change 0.059) |
|---|---|---|---|
| M16 (L1) | **0.88** | **0.95** | world 0.056 ≈ copy |
| D continued / D scratch | 0.82 / 0.73 | 0.74 / 0.72 | world 0.13 / 0.15 = 2.2-2.5× copy |
| E22MS sample / Δ zeroed | 0.52 / 0.47 | 0.72 / 0.73 | world 0.34 / 0.33 = 5.7× copy |

- **The deterministic context dependence of the tokens is predictable and largely predicted** (M16: 88% of scroll re-encoding,
  95% of HUD jitter).
- **"The target isn't game state" was the wrong headline (the user's objection; Codex agrees).** What the target carries that
  no world can predict is the night static, ≈ 25% of the L2 change mass. Usefulness of the context is unmeasured.
- **New:** the D-loss and E22MS worlds damage cells that do not change at all (still unchanged cells: 2.2-5.7× a copy's
  error).

**2. Where do the wrong facts sit, net of the probes' own errors?** (`wrongfacts_by_cat.py` (+ the same on TRUE tokens); depth 1; TEST roots;
17,546 map cells)
- **Probe baseline:** on the TRUE tokens the probes mark 2.04% of cells wrong (2.02% of unchanged cells, 5.6% of own-state-
  changed).
- **Correction to the 10:34 currency entry:** its "4-6% of cells carry a wrong fact" includes ~2 points of reader error.
  - World-attributable: M16 2.1%, D continued 2.2%, D scratch 3.0%, E22MS Δ zeroed 3.4%.
  - The "62.5% of L1 error on fact-correct cells" is conservative: probe-wrong cells have small world error.
- **World-attributable wrong facts by category:**
  - M16: never-seen entering 45%, own-state changes 41%, unchanged 10%.
  - D scratch: entering 31%, own 26%, unchanged 25%, neighbours 16%.
  - D continued: 42 / 36 / 12 / 8%.
  - E22MS Δ zeroed: unchanged 38%, entering 28%, neighbours 19%, own 12%.
- **Correction to the 19:2x inference** ("D's late wrong facts are largely unseen terrain"): at depth 1 unseen terrain is
  31-45% of world-attributable wrong facts, the largest single bucket but not "largely". Own-state changes are the other big
  one.

**3. Damage and movement determinism** (Codex's verification; recomputed from the one-step data, 17,032 root × action pairs
with all four env keys alive):
- 471 pairs show damage under ≥ 1 key; 457 (97.0%) under all four. Immediate damage is deterministic given the full
  (hidden) state. Codex counted 389 / 403 with another definition.
- Mob layouts differ between keys after ONE step in 51% of pairs. The source matches: cows draw from 8 entries, half of
  them "stay"; zombies chase with p 0.75 when close; spawns are random.

**4. Codex's citations from this NOTEBOOK, verified:**
- E17 recall: same-slot window 5 → 15 is 0.369 → 0.616 / 0.365 → 0.643; moved slot is 0.396 → 0.405 / 0.431 → 0.440.
- Consequences caught: 0.002 → 0.562.
- rollout2 was not an exact port (E12 correction).

### 10-10 RESULT (19:04): P-C under AdamW and with seen / never-seen entering — the allocation survives the optimizer

**Setup:** `gradanat.py --seen --adam` (lane179). L16 worlds, 1,002 windows, 60 batches.
- Each run's saved AdamW state (its weights checked equal to the loaded world's; 133 parameters with state) gives
  P = 1 / (sqrt(v̂) + eps).
- Entering is split by whether the cell's world position was in an earlier window frame.

Projection share of the update, raw → AdamW-preconditioned, under each world's OWN loss:

| | never-seen entering | seen entering | own state changed | no state change | HUD changed | HUD same |
|---|---|---|---|---|---|---|
| M16 / L1 | 0.370 → 0.345 | 0.033 → 0.034 | 0.024 → 0.028 | 0.423 → 0.426 | 0.011 → 0.012 | 0.053 → 0.059 |
| D scratch / D loss | 0.613 → 0.564 | 0.129 → 0.129 | 0.049 → 0.059 | 0.074 → 0.079 | 0.058 → 0.080 | 0.053 → 0.060 |

Mean update over batches, AdamW (what momentum follows):
- M16: never-seen 0.15, unchanged 0.63, own state 0.014, HUD changed 0.008.
- D scratch: never-seen 0.46, seen 0.11, unchanged 0.14, own state 0.060, HUD changed 0.19.

**Reading:**
- Preconditioning moves no share by more than ~5 points. **The allocation (P-C) holds for the actual update.**
- Never-seen terrain takes 35% (L1) / 56% (D) of the update. Seen terrain, which memory could recall, takes 3% / 13%.
- Own-state changes get 3-6%.
- The 17:20 entry's raw stage-1 figure (89% to entering, D loss at 36k) included seen terrain. On the L16 world, the split is
  0.61 never-seen + 0.13 seen.
- **Not interpretable yet:** the cosine of every category's mean update with the run's own saved momentum direction is ≈ 0
  (−0.11 to +0.07). The momentum is an average over the last ~10 TRAINING batches (corpus windows); these are diagnosis windows.
  That is noted, not read as a finding.
- **Stage-1 endpoints (19:16; 63 batches, own loss):**
  - DS1 @ 36k (D loss): never-seen entering 0.839 → **0.813** under AdamW (mean update 0.822; consistency 0.20 vs 0.126 noise);
    seen entering 0.05; own state 0.013; no change 0.05; HUD changed 0.025.
  - M6 @ 36k (L1): never-seen 0.355 → 0.311; no change 0.456 → 0.474; own state 0.048 → 0.052.
  - **This replaces the withdrawn 17:20 sentence, now measured properly:** at the end of D-loss training, 81% of the AdamW
    update's projection is on terrain never seen in the window. Its gradient is barely more consistent across batches than
    noise (0.20 vs 0.126). Own-state changes get 1.3%.

### 10-10 RESULT (18:56): loss currency, GRADIENT side — L1 spends 86% of its averaged update on content whose facts are already right

**Setup:** `gradanat.py --facts` (lane179). The L16 worlds on 283 TEST windows, 60 batches, raw backbone gradients
(`--adam` pending). The objective was reproduced (relative error ≤ 1.5e-7). Categories are the world's OWN decoded fact errors
per batch (currency.py's probes).
- **Projection share:** the per-batch update.
- **Mean-update share:** the share of the update averaged over batches, which is what momentum follows.

| world / loss | map, fact wrong: proj / mean update / consistency | map, facts right | HUD values wrong | HUD values right |
|---|---|---|---|---|
| M16 / L1 (own) | 0.21 / **0.10** / 0.20 | 0.70 / **0.81** / 0.34 | 0.014 / 0.007 / 0.27 | 0.06 / 0.05 / 0.33 |
| M16 / D loss | 0.73 / 0.59 / 0.20 | 0.25 / 0.40 / 0.36 | 0.013 / 0.004 / 0.66 | 0.002 / 0.001 / 0.53 |
| D scratch / D loss (own) | 0.57 / **0.37** / 0.30 | 0.30 / 0.21 / 0.39 | 0.06 / 0.09 / 0.41 | 0.06 / **0.33** / 0.48 |
| D scratch / L1 | 0.10 / 0.04 / 0.34 | 0.73 / 0.69 / 0.43 | 0.025 / 0.031 / 0.37 | 0.14 / 0.23 / 0.66 |

(noise consistency 0.129)

- **Under L1, 81% (map) + 5% (HUD) of the averaged update goes to content whose decoded facts are already right.** Only 10%
  goes to the ~4% of cells with a wrong fact; 0.7% to wrong HUD values.
- **The D loss aims at wrong facts:** 37% + 9% of the averaged update. Yet 54% still goes to right content, a third of it to
  HUD tokens whose values are right (the jitter, `encoder_why` E3).
- **The allocation follows the loss, not the world:** swap the losses and the pattern swaps.
- **The fact-wrong gradient is the least consistent across batches** (0.20-0.30 vs 0.34-0.43). Wrong facts are diverse, and
  their per-batch gradients partly cancel in the average.
- **This closes the currency chain on both sides:**
  - prediction: 62.5% of M16's L1 error is on fact-correct cells;
  - gradient: 86% of L1's averaged update goes to fact-correct content;
  - outcome (P-D): L1 stalls on HUD changes (−1.5% in 30k updates) and on own-state changes (−26% vs −46% under D).
- The AdamW version of the state-change split is in the 19:04 entry; preconditioning moves shares by ≤ ~5 points.
- **Over training** (stage-1 snapshots, 18 TEST batches each; noise consistency 0.24, so coarser), mean-update shares:

  | | map fact wrong | map facts right | HUD values wrong | HUD values right |
  |---|---|---|---|---|
  | M6 (L1) @ 6k → 36k | 0.16 → 0.19 | 0.67 → 0.73 | 0.02 → 0.02 | 0.10 → 0.05 |
  | DS1 (D loss) @ 6k → 36k | 0.06 → **0.70** | 0.66 → 0.28 | 0.02 → 0.01 | 0.26 → 0.01 |

  - Under L1, ~75-80% of the averaged update goes to fact-correct content throughout training.
  - Under the D loss the update moves to wrong facts as the correct content's errors shrink.
  - At 36k, D's 70% on wrong-fact cells coincides with 81% of its AdamW update on never-seen terrain (19:04 entry). That the
    late wrong facts are largely unseen terrain is an INFERENCE: the two splits were not cross-tabulated.

### 10-10 RESULT (18:35; all three worlds 18:51): C3 chaos — the deterministic worlds CONTRACT; depth error is accumulated bias, not input sensitivity

`chaos.py` (lane177; D continued and D scratch still running). The unperturbed rollout equals the saved export bitwise.
Root-frame tokens perturbed (x + σ N(0, I), then layer-normed); self-fed 16 steps; median growth rate of the divergence per
step:

| σ (initial / V) | depth 1 | depth 4 | depth 16: divergence / V | scroll differs at 16 | view offset differs at 16 | tiles differ at 16 | median rate per step |
|---|---|---|---|---|---|---|---|
| 0.001 (2e-6) | 7e-6 | 6.2e-5 | 0.0018 | 0.1% | 0.6% | 0.2% | +0.15 → +0.06 |
| 0.01 (2e-4) | 1.7e-4 | 4.2e-4 | 0.0048 | 0.2% | 1.9% | 0.4% | −0.08 to −0.14 |
| 0.1 (0.020) | 0.016 | 0.013 | 0.021 | 0.8% | 9.4% | 1.2% | −0.08 to −0.15 |

- **Perturbations of 0.01-0.1 shrink.** At depth 16 the divergence is the size of the initial perturbation (0.021 vs 0.020 of
  V), 25-300× smaller than M16's actual depth-16 error (gen16 / V ≈ 0.53).
- **M16's long-horizon error is NOT sensitivity to its input:** it is per-step bias accumulated through self-feeding.
- **The one discrete instability** is the scroll decision: offsets diverge in up to 9.4% of roots by depth 16. This matches
  E11 (imagined error growth = wrong move decisions).
- **Replicated on D continued and D scratch (18:51; both unperturbed rollouts equal their exports):**

  | world (gen16 / V) | σ 0.001: div16 / rate d1 → d16 | σ 0.01: div16 / rate | σ 0.1: div16 / offset differs at 16 / rate |
  |---|---|---|---|
  | M16 (0.528) | 0.0018 / +0.15 → +0.06 | 0.0048 / −0.08 to −0.12 | 0.021 / 9.4% / −0.08 to −0.13 |
  | D continued (0.473) | 0.0013 / +0.42 → +0.13 | 0.0016 / −0.17 to −0.07 | 0.021 / 9.1% / −0.19 to −0.10 |
  | D scratch (0.494) | 0.0036 / +0.30 → +0.12 | 0.0044 / −0.11 to −0.06 | 0.022 / 11.7% / −0.12 to −0.08 |

- **Under both losses**, depth-16 divergence is 0.1-4.6% of the actual depth-16 error. Imagination instability is not the
  deterministic worlds' problem; their long-horizon error is per-step bias, plus scroll-decision flips.

### 10-10 RESULT (18:30): P-D (training dynamics) — not refuted; allocation predicts improvement only for learnable content; each world converges toward its loss's statistic

**Setup:**
- 12 stage-1 snapshot exports (lane173, 17:11-18:13) and their CPU readouts (cpu180: budget, currency, entering_memory,
  scoreboard; 18:14-18:25). `pd_summary.py` reads them.
- M6 (L1) and DS1 (D loss): same seed, data and architecture, from scratch; 6k-36k updates.
- Classes: the budget categories in the one-step setting. Measure: excess over the RNG floor per token.
- Allocation: each run's raw projection share under its own loss, averaged over the six snapshots (gradanat).

| class | M6 (L1): 6k → 36k | DS1 (D): 6k → 36k | allocation L1 / D |
|---|---|---|---|
| never-seen entering | 47.3 → 38.1 (−19%) | 43.5 → 41.5 (−5%; 37.6 at 24k, then RISES) | 0.21 / 0.69 |
| seen entering | 43.3 → 15.3 (−65%) | 36.2 → 11.4 (−69%) | ~0 / ~0 |
| own state changed | 39.4 → 29.1 (−26%) | 25.9 → 14.1 (−46%) | 0.036 / 0.027 |
| player sprite left | 23.0 → 0.0 | 18.5 → 1.1 | 0.016 / 0.009 |
| sleep changed | 28.6 → 1.2 | 6.5 → 2.3 | 0.036 / 0.004 |
| no state change | 0.46 → 0.07 (−84%) | 0.65 → 0.29 (−56%) | 0.59 / 0.12 |
| HUD changed | 5.81 → 5.73 (**−1.5%**) | 4.71 → 3.18 (−32%) | 0.011 / 0.046 |
| HUD same | 0.46 → 0.03 (−94%) | 0.43 → 0.20 (−54%) | 0.060 / 0.092 |
| player token | 4.87 → 0.32 | 2.66 → 0.73 | 0.013 / 0.002 |

(neighbour_changed: excess ≤ 0 in both runs, so improvement is undefined.)

**Against the predeclared rule** (refuted if one loss improves every class at least as fast as the other): **not refuted.**
- M6 is faster on 6 classes, DS1 on 3 (seen entering, own state, HUD changed).
- Rank correlation across classes, allocation difference vs improvement difference: 0.18 over all classes. Post hoc, labelled:
  0.37 without neighbour_changed, 0.41 also without never-seen entering. **Weak.**

**What the curves show instead:**
1. **Allocation buys improvement only where the content is learnable.**
   - The D loss's largest allocation, 69% of its raw projection on never-seen terrain, bought −5%, and regressed after 24k.
   - L1's largest allocation, 59% on unchanged cells, bought −84% of an excess that was already < 0.5.
2. **L1 never learns HUD changes:** −1.5% in 30k updates (5.81 → 5.73), the health failure's root, as a training curve. The D
   loss learns them (−32%), and own-state changes twice as well (14.1 vs 29.1 at 36k).
3. **Same totals, different winners:** one-step L2 per frame 148 (M6) vs 153 (DS1) at 36k.
4. **Each world converges toward its loss's optimum's failure** (decoded cows / the five futures' count):
   - M6 at depth 16: 1.30 → 0.31 → 0.15 → 0.22 → 0.17 → 0.11. Erasure grows with training: the median's signature
     (statoracle).
   - DS1 at depth 1: 0.98 → 0.97 → 0.98 → 1.01 → 1.13 → 1.50. Superposition grows: the mean's signature.
5. **Facts:**
   - DS1 improves zombie AUC (0.89 → 0.96) and depth-16 inventory R² (0.77 → 0.89), where M6 stays flat (0.81 → 0.82;
     0.74 → 0.74).
   - Map fact-wrong share: M6 5.1 → 4.3%; DS1 5.2 → 5.1%.
   - Depth-16 tile accuracy: M6 0.73 → 0.77; DS1 0.69 → 0.75.
6. **M6's 24k snapshot is an unstable point** in every reading (one-step L2 239 vs 170 at 18k; gradanat total consistency
   0.86).
7. **Seed-8 replication of the L1 side (20:02; lane181 exports + cpu182 readouts; no D-loss seed-8 run exists):**

   | L1, 6k → 36k | seed 7 | seed 8 |
   |---|---|---|
   | HUD changed (excess) | 5.81 → 5.73 (−1.5%) | 5.81 → 5.73 (−1.3%) |
   | cows at depth 16 / five futures | 1.30 → 0.11 | 1.05 → 0.13 |
   | own state changed | −26% | −22% |
   | never-seen / seen entering | −19% / −65% | −19% / −60% |
   | zombie AUC d1 / inventory R² d16 | 0.81 → 0.82 / 0.74 → 0.74 | 0.82 → 0.82 / 0.74 → 0.74 |

   - The HUD-change stall and the growing erasure under L1 replicate across init seeds.
   - Seed 7's 24k instability does not occur in seed 8: it is seed-specific.
   - The D-loss side (superposition growing, 0.98 → 1.50) remains single-seed.

### 10-10 RESULT (18:00): proper scores — sampling pays, but it is not where most of the gap is; an energy score on Raw tokens repeats the allocation

Prompted by Codex's proposal: a noise-conditioned generator trained by an energy score on completed successors, replacing the Δ
posterior, codebook and prior. `properscore.py` tests two of its premises on the exports, before anything is trained.

**Corrections accepted from Codex's review (verified):**
- **The per-class energy scores in `scoreboard.py` are not proper.** The mob / entering / static masks are chosen from the
  realized futures (`stochdiag.classes`). Only the HUD and player classes are fixed. The scoreboard's "E22MS better on energy
  score" by class is withdrawn as a proper-score claim; the fixed-region scores below replace it.
- **"40% of the L1 change mass is renderer noise" (Part II-A heading) overstates it.**
  - 40% (L1) / 34% (L2) is the mass of NIGHT cells with NO state change.
  - Of their squared change, the measured static (2 × the identical-state variance per ring) is ~74%: 58 / 76 / 84 / 86% at
    rings 1-4 when the view is still, 10 / 39 / 68 / 77% when it scrolls.
  - The rest is re-encoding and lighting drift.
  - Night static is therefore ≈ 25% of the L2 change mass.
  - The noise check matches visible-state descriptors, not full simulator states. By day the variance at identical
    descriptors is exactly 0 over 2,255 frames, so rendering depends on nothing else by day.
- **P-C measured RAW backbone gradients, not AdamW's update, and its `entering` includes terrain seen earlier in the window.**
  "~90% of every update chases never-seen terrain" is withdrawn.
  - What was measured: under the D loss at 36k, entering cells receive 89% of the raw-gradient projection, with cross-batch
    consistency 0.18 (vs 0.126 for pure noise).
  - `gradanat --adam --seen` (preconditioned by the saved AdamW moments; seen / never-seen split) is queued to measure the
    actual update.

**A. Proper scores** (fair energy score, fixed regions, vs the five real futures; self-fed from the root; paired
episode-clustered 95% intervals, 143 episodes). Whole frame:

| depth | ideal sampler (4 real futures, leave-one-out) | ideal point (mean / median of 4) | E22MS, 4 samples | M16 | D continued | D scratch | E22MS Δ zeroed | copy root |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.36 | 5.60 / 5.56 | 8.67 | 10.85 | 10.77 | 10.85 | 11.32 | 27.2 |
| 2 | 3.99 | 6.60 / 6.42 | 11.95 | 15.46 | 15.09 | 15.33 | 15.78 | 37.5 |
| 4 | 5.19 | 8.58 / 8.17 | 18.09 | 24.21 | 23.08 | 23.21 | 24.26 | 51.2 |
| 8 | 6.74 | 11.12 / 10.55 | 26.33 | 36.60 | 34.59 | 34.47 | 36.66 | 66.8 |
| 16 | 9.72 | 16.08 / 15.13 | 39.48 | 55.03 | 51.78 | 53.32 | 57.42 | 84.2 |

- **E22MS beats every deterministic world at every depth.**
  - Depth 1: E22MS − D scratch −2.19 [−2.39, −2.00]; E22MS − M16 −2.19 [−2.52, −1.86].
  - Depth 16: −13.84 [−15.00, −12.60] and −15.55 [−17.44, −13.64].
  - Its own decoder with Δ zeroed is the worst: −2.66 at 1, −17.94 at 16.
- **Depth 1 decomposition:**
  - E22MS's sampling gain (2.19) is the size of the ideal sampler's gain over the ideal point (2.24 [2.03, 2.43]).
  - E22MS's gap to the ideal sampler (5.30 [4.96, 5.66]) equals the deterministic worlds' gap to the ideal point (5.26-5.29).
  - **Sampling removed the point penalty and left the rest untouched.**
  - Of D scratch's distance to the ideal sampler (7.49), sampling closes 29%. At depth 16, it closes 32% of 43.6.
  - The other ~70% is information (hidden state, unseen terrain), model error and compounding.
- **Regions:**
  - HUD at depth 1: every world is ~10× the ideal (2.0-2.3 vs 0.19), and E22MS ≈ the deterministic worlds (−0.13 vs D scratch).
  - Map at night, depth 1: E22MS 14.57 ≈ the ideal POINT 14.37, vs the ideal sampler 9.04. It does not sample the static as
    reality does.
  - Map by day, depth 1: E22MS 6.50 vs ideal 1.79, ideal point 3.1.

**B. The currency of each loss at the output** (depth 1, ONE observed successor as in training, budget.CATS with seen /
never-seen entering). Share of the SYSTEMATIC gradient: the part that does not depend on which future happened. Totals:
systematic fraction 0.72-0.91, vs 0.2 for pure outcome noise.

| loss, at E22MS's samples | unchanged cells | never-seen entering | seen entering | own-state changes | neighbours | HUD, values changed | HUD, values same (jitter) |
|---|---|---|---|---|---|---|---|
| L1 | 0.670 | 0.037 | 0.004 | 0.007 | 0.023 | 0.040 | 0.193 |
| MSE | 0.212 | 0.631 | 0.019 | 0.043 | 0.036 | 0.032 | 0.015 |
| energy score, whole frame | 0.423 | 0.245 | 0.015 | 0.051 | 0.036 | 0.067 | 0.131 |
| energy score, map + HUD regions | 0.227 | 0.101 | 0.006 | 0.024 | 0.015 | 0.106 | 0.503 |

- The deterministic worlds give the same pattern (M16 / D scratch under the whole-frame energy score: own-state 7.3 / 5.1%).
- For a point forecast the whole-frame energy score's gradient is MSE's direction normalized, so its allocation equals MSE's.
  It differs only through a sampler's repulsion term.
- **Reading:**
  - An energy score on frozen Raw tokens sends 5% of its systematic update to own-state changes and 7% to HUD value changes.
  - The rest goes to unchanged cells (42%), never-seen terrain (25%) and HUD jitter (13%).
  - Splitting the score into fixed regions (Pic et al.) makes HUD jitter half of it, because the HUD's small norm is
    normalized up and most of its error is map-context jitter (`encoder_why` E3).
  - The proposal's premise "a sampler under a proper score beats point forecasts" holds. "That loss would put its pressure on
    the facts" is false in this target space.
- At the ideal sampler (pure outcome noise), the score's gradient is on unchanged cells (0.57) and own-state changes (0.22):
  the stochastic outcomes.
- Not additive with P-C: this is output space, model-free; P-C is the backbone side.

**C. Where the gap to the ideal lives** (`properscore.py --subst`). One fixed region at a time is replaced by the ideal of the
predictor's class: the ideal point for the deterministic worlds, the ideal sampler for E22MS. The whole-frame score is
rescored. Share of the gap removed (not additive; ES is a norm):

| | edge ring (30 cells; where terrain enters) | interior (32) | HUD (18) | player |
|---|---|---|---|---|
| depth 1: M16 / D cont. / D scratch / E22MS | 0.65 / 0.71 / 0.69 / 0.61 | 0.05 / 0.04 / 0.06 / 0.17 | 0.18 / 0.09 / 0.08 / 0.08 | ≤ 0.007 |
| depth 4 | 0.43 / 0.49 / 0.50 / 0.45 | 0.16 / 0.15 / 0.16 / 0.20 | 0.13 / 0.10 / 0.07 / 0.09 | ≤ 0.005 |
| depth 16 | 0.31 / 0.33 / 0.35 / 0.33 | 0.28 / 0.28 / 0.28 / 0.29 | 0.09 / 0.07 / 0.06 / 0.07 | ≤ 0.004 |

- At one step, about two-thirds of every world's gap is on the edge ring, the sampler included.
- **The edge content is mostly epistemic, not aleatoric:** the five futures share the map, so the ideal KNOWS the terrain a
  world never saw. A world can at best sample plausible terrain there.
- E22MS removes the edge gap no better than the deterministic worlds (0.61 vs 0.65-0.71). Its Δ channel does not sample terrain.
- With depth, the interior's share grows to ~28%: earlier errors move inward, plus mobs and misregistration.
- **Open → measured below (`terrainsub.py`):** how much of the edge gap a terrain-SAMPLING world could close.

**D. Can sampling plausible terrain close the edge gap?** (`terrainsub.py`, 17:55-18:10)
- **Setup:**
  - 376 TEST+TRAIN roots whose factual future scrolls at depth 1 (76 at night).
  - The world's prediction on the entering line is replaced by terrain from donor transitions of OTHER episodes: the 17
    one-step successors of every root; 632-762 donors per direction.
  - Scored: whole-frame fair ES vs the five futures. For a point forecast, the donors' mean; for E22MS, one donor per sample.

| entering line replaced by | tile agreement with the true entering tiles | gap closed by day: det. worlds / E22MS | at night: det. / E22MS |
|---|---|---|---|
| the other real futures (ideal; knows the map) | 1.00 | 1 / 1 | 1 / 1 |
| 4 donors nearest in the visible edge line's tokens (a conditional terrain sampler) | 0.64 | −0.29 to −0.42 / −0.10 | −0.90 to −1.08 / −0.02 |
| 4 donors best matching the TRUE entering tiles (control; facts no world has) | 0.88 | +0.13 to +0.22 / **+0.41** | −2.26 to −2.63 / −1.75 |
| 4 random donors (terrain prior) | 0.35 | −1.50 to −1.80 / −1.19 | −3.55 to −4.10 / −2.23 |

- **The worlds' own entering-line predictions already beat plausible terrain sampled from what they can see.** The conditional
  sampler is neutral for E22MS and harmful for the point forecasts.
- **Nearly correct terrain WOULD help a sampler by day** (41% of E22MS's entering gap). That terrain is information the window
  does not contain: the visible edge line raises tile agreement only from 0.35 to 0.64.
- **At night, correct tiles under another frame's light and static are far worse than a blurred prediction:**
  - the token of a cell is not a function of its terrain;
  - this is the target-space finding (encoder_why) again, and it confounds this substitution at night.
- **Limits:**
  - a donor bank of ~700 per direction;
  - the nearest neighbour uses one line of tokens, while a learned sampler could use the whole window and biome structure.
  - The bound is therefore "this naive conditional sampler does not help", not "no sampler can".
- **Reading:**
  - The largest part of the depth-1 gap (the edge ring, 61-71%) is mostly missing information, not a missing stochastic
    output.
  - Memory can recover the seen part: 23% of entering cells in L16 windows; oracle recall 4-10 vs the worlds' 6-26
    (entering_memory, 10:2x).
  - No generator samples the unseen part into correctness.

### 10-10 INCIDENT (11:42-17:10): the lane queue was blocked for 5.5 h by one job

- From 11:42 the GPU ran only `h16_e22ms` (2.9 GB); lanes 170 / 173 / 177 / 178 waited.
- Their declared needs (2,300-3,000 MiB + 256) did not fit beside it: free was 2,497 MiB.
- Those declarations were never measured. The CPU was idle after `encoder_why` finished (11:27).
- 17:10: lane171 stopped (resumable: draw files and feature progress are saved; it was in draw 6 of 16). `lane171b` resumes
  it once 170 / 173 / 177 / 178 are done.
- 17:10 h16_dscratch started; 17:11 sb_export_snapshots started beside it.
- Lesson: before queueing, check the declared needs against the longest running job.

### 10-10 RESULT (17:20): P-C (gradient anatomy) — HOLDS; the update becomes noise from unseen terrain as training proceeds

`gradanat.py`, 63 batches per snapshot (stage-1 worlds, bs 16; noise consistency 1/√63 = 0.126), 60 batches for the L16
worlds (bs 4; 0.129). The objective was reproduced exactly (relative error 0.0) for both losses.

Cells: **projection share** of the update / grad-norm ratio / **cross-batch consistency**, under each world's OWN loss:

| world | entering (77% never seen) | own state changed | no state change | total consistency |
|---|---|---|---|---|
| M6 @ 6k (L1) | 0.09 / 0.23 / 0.35 | 0.04 / 0.19 / 0.58 | 0.56 / 0.69 / 0.43 | 0.39 |
| M6 @ 36k (L1) | 0.37 / 0.55 / 0.19 | 0.05 / 0.24 / 0.58 | 0.46 / 0.62 / 0.48 | 0.33 |
| M16 (L1) | 0.40 / 0.54 / 0.20 | 0.02 / 0.13 / 0.51 | 0.42 / 0.55 / 0.46 | 0.34 |
| DS1 @ 6k (D loss) | 0.29 / 0.49 / 0.27 | 0.04 / 0.18 / 0.51 | 0.29 / 0.45 / 0.46 | 0.37 |
| DS1 @ 36k (D loss) | **0.89 / 0.94 / 0.18** | **0.01** / 0.10 / 0.38 | 0.04 / 0.15 / 0.35 | 0.19 |
| D scratch L16 (D loss) | 0.74 / 0.85 / 0.16 | 0.05 / 0.24 / 0.44 | 0.07 / 0.23 / 0.39 | 0.18 |

**Against the predeclared rule** (refuted if rare-change consistency ≤ irreducible consistency, or the irreducible norm share
< 25%): **not refuted.**
- In every late world, own-state changes are 2-3× more consistent than entering terrain (0.38-0.58 vs 0.16-0.20).
- Entering terrain has 54-94% of the gradient norm.

**Over training:**
- Entering's share of the update rises (L1 0.09 → 0.37; D 0.29 → 0.89) while its consistency falls toward noise (0.35 → 0.19;
  0.27 → 0.18). Its learnable part gets learned; what remains is noise, and it takes over the update.
- The D loss's whole update is close to noise late: total consistency 0.37 → 0.19, vs 0.126 for pure noise.
- Own-state changes stay consistent throughout but receive 1-5% of the direction.

**Conflict:** the cosine between the own-state-change and no-change gradients is −0.15 (DS1 @ 36k, D loss) and −0.18 (M6 @ 36k,
L1). Learning a change pushes against copying the unchanged cells. Entering is ~orthogonal to both (−0.01 to 0.09).

**The allocation is the loss's, not the world's:** scored under the other loss, each world shows that loss's pattern.
- M16 under the D loss: entering 0.88 of the update.
- D scratch under L1: no-change 0.43 and entering 0.28.

**Reading:** P-C holds, with the predicted structure.
- Under L1, about half of every update copies unchanged cells, a consistent pull that includes the night static.
- Under the D loss, by the end, ~90% of every update chases never-seen terrain, and that gradient is near-random. **[WITHDRAWN 18:00: raw gradient, not AdamW's update; `entering` includes seen terrain; consistency 0.18 vs noise 0.126 is not 'random'. What holds: 89% of the RAW-gradient projection, consistency 0.18. MEASURED 19:16 (`--adam --seen`): 81% of the AdamW update on NEVER-seen terrain (5% on seen), consistency 0.20; see the 19:04 entry.]**
- Under both, the facts that change get a few percent.
- **This is the third face of the same cause:** the loss's currency allocates the update by error magnitude, and the largest
  errors are the unlearnable ones.

### 10-10 RESULT (17:20): encoder interventions across encoders, and E22MS decisions (partial)

`encoder_why.py`, all 1,002 roots per encoder. Raw's checks reproduce encoder_diag: S0 R² 0.5926 vs 0.592; S2 0.888 vs
0.888. Final tokens match the caches to ≤ 0.002.

| encoder | own-state R² final (day / night) | best layer | day-scroll re-encoding: real / rigid translation / positions matched | HUD jitter, 18 tokens (still day / scroll day) | night static, ring 4: own pixels / context |
|---|---|---|---|---|---|
| Raw | 0.59 (0.79 / 0.33) | L6 0.69 | 1.56 / 1.54 / 0.75 | 12.0 / 9.1 | 11.3 / 0.13 |
| TC | 0.38 (0.49 / 0.41) | L1 0.67 | 6.40 / 6.10 / 1.82 | 54.1 / 36.0 | 20.7 / 1.85 |
| LDAD λ1 | 0.34 (0.41 / 0.23) | emb 0.62 | 21.5 / 19.5 / 1.50 | 11.6 / 28.6 | 9.9 / 0.18 |
| LDAD λ10 | 0.28 (0.34 / 0.14) | emb 0.61 | 18.4 / 17.5 / 0.67 | 15.3 / 135.3 | 2.4 / 0.08 |

**Raw:**
- Re-encoding on scroll is 99% a rigid-translation effect: half position embeddings, half global context.
- The context part is flat with distance from the entering edge (0.84 → 0.68 → 0.74): global, not local bleeding.
- A single-patch edit puts only 3-11% of its change on other tokens, flat with distance. On a scroll every patch moves, and
  those small global effects add up.
- HUD pixels are identical in 100% of 12,546 unchanged-HUD pairs. Their token change is 0 at the embedding and grows with
  depth to 9-12: entirely attention to the map.
- Night static is 99% the cell's own pixels.

**Every encoder:**
- The patch outputs lose own-state alignment in the deep layers that no loss trains, most of all LDAD's.
- LDAD's inverse-dynamics objective makes its patch tokens position-coded: 92-96% of their ×12-14 larger re-encoding is
  position. LDAD λ10's HUD tokens move 135-153 per scroll step.
- Raw's last layer is the most state-aligned of the four. LDAD's measured gains (move routing, 09-30) did not come through its
  patch tokens as targets.

**E22MS H16 decisions** (window 15; DEV-B 1,139 opportunity roots; 3 head seeds), before the pause:

| reading | trajectory head | snapshot head |
|---|---|---|
| E22MS, 1 sample | 0.630 (0.6317 / 0.6298 / 0.6297) | 0.615 |
| M16 (`__w15.json`) | 0.657 | 0.623 |
| references | prior 0.616, one real future 0.679, uniform 0.571, oracle31 0.766 | |

- One E22MS sample decides worse than M16 (−0.027) and only slightly better than the prior.
- The 16-sample readout (M1 / M4 / M16, averaged P(dead by 16)) is the test of whether sampling pays. It is paused at draw 6
  and not judged.

### 10-10 RESULT (11:40): the loss vs the network, by substitution — the per-element losses' OPTIMA already fail the mob facts, in opposite directions

**Question:** a deterministic world trained by a per-element loss can at best output that loss's optimal statistic of the
next-token distribution:
- the per-element median for L1;
- the per-element mean for squared error.

**Method:** `statoracle.py` replaces the network by the exact statistic of the five real futures of each root (same actions,
different env keys), which is better than any world can do since the futures share the full simulator state.
- It reads the statistic with the same probes every world is judged on (fitted on true TRAIN tokens; 283 TEST roots).
- Check: the real futures decode to the simulator's counts (cows 0.428 vs 0.416 at depth 1; zombies 0.201 vs 0.204).
- The floor check holds: mean5's L2 to the futures = 4/5 of the unbiased variance (50.35 vs 62.94), and median5 has the lower L1
  (188.9 vs 214.4), as each statistic must.

**Mob census, decoded counts per frame** (sim = the simulator's mean over the five futures):

| depth | sim cows | median5 | mean5 | sim zombies | median5 | mean5 | sim skeletons | median5 | mean5 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | 0.286 | 0.979 | 0.204 | 0.191 | 0.290 | 0.105 | 0.071 | 0.201 |
| 4 | 0.463 | 0.120 | 1.548 | 0.223 | 0.173 | 0.424 | 0.100 | 0.035 | 0.307 |
| 16 | 0.521 | **0.079** | **2.094** | 0.306 | 0.127 | 0.790 | 0.125 | 0.049 | 0.393 |

**The mechanism, per cell.** Share of cells decoded "cow present", by how many of the five futures have a cow there (depth 1;
the same at every depth):

| futures with the cow | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| a real future (a sample) | 0.20 | 0.40 | 0.59 | 0.80 | 1.0 |
| median5 (L1's optimum) | **0.0** | **0.0** | 0.95 | 1.0 | 1.0 |
| mean5 (MSE's optimum) | **0.82** | 0.96 | 0.95 | 1.0 | 1.0 |

- **The median is a majority vote per element:** any mob with < 50% chance at a cell is erased. Mob positions spread with
  depth (at depth 16, 612 of 652 cow cells have a cow in only ONE future), so the L1 optimum erases nearly every moving mob.
  The erasure does not depend on the reader: the median token there IS the background token.
- **The mean is a superposition:** a faint copy of the mob at every cell any future puts it. It is not a state, and how many
  mobs it "contains" depends on the reader. Probes calibrated on real states read most faint copies as mobs (×4 cows at 16).
- **Only a sample is calibrated:** the per-cell presence equals the chance (0.2 / 0.4 / 0.6 / 0.8 / 1.0), and mass is
  conserved.

**The worlds reproduce their loss's optimum at depth 1** (decoded census, same probes; depth 1 is teacher-forced on the true
context):

| | zombie | skeleton | cow |
|---|---|---|---|
| simulator | 0.204 | 0.105 | 0.416 |
| median5 (L1 optimum) | 0.191 | 0.071 | 0.286 |
| M16 (L1) | 0.117 | 0.039 | 0.357 |
| mean5 (MSE optimum) | 0.290 | 0.201 | **0.979** |
| D scratch (MSE-dominant) | 0.251 | 0.240 | **0.922** |
| E22MS decoder, Δ zeroed (MSE-trained, no Δ) | 0.209 | 0.187 | **0.972** |
| E22MS, prior sample / true codes | 0.212 / 0.209 | 0.170 / 0.156 | 0.611 / 0.569 |

- M16 loses mobs as the median does.
- The MSE-family decoders multiply cows at depth 1 exactly as the mean does (0.92-0.97 vs 0.98).
- Given Δ codes, the same decoder is closest to calibrated (0.57-0.61).
- Self-fed to depth 16, every deterministic world LOSES mobs: cows M16 0.135, D continued 0.039, D scratch 0.359, Δ zeroed 0.399,
  vs 0.521. Iterating a superposition dilutes it: a faint mob's own next-step mean is fainter.
- The E22MS samples' ×6-8 cow growth at depth 16 is a separate, Δ-channel mechanism, present with true codes too. Not explained
  here.

**Which statistic each world learned** (depth 1, the 1.4% of elements where mean5 and median5 differ by > 0.1): share closer to
the median:
- M16 0.55 (the highest);
- D continued 0.38, D scratch 0.36, Δ zeroed 0.36;
- E22MS samples 0.46-0.47.

The ordering follows the loss: L1 → median-like, squared error → mean-like. The separation is moderate; the census is the strong
evidence.

**Seed-8 replication (20:03):**
- M16 seed 8's depth-1 census: zombies 0.124, skeletons 0.039, cows 0.364 (seed 7: 0.117 / 0.039 / 0.357).
- Depth 16: cows 0.132 (0.135), zombies 0.007 (0.011).
- Closer to the median on 0.551 of the differing elements (seed 7: 0.550).

**HUD:** the statistics draw the timer decrements at the real rate, because the five futures share the hidden counters, which
are deterministic given the root's FULL state. The worlds' missing decrements are the information horizon (Part I scoreboard entry, timers), not
the statistic.

**Reading — the "fighting ourselves" mechanism for anything whose outcome is random:**
- For every fact whose future is uncertain (mob moves, damage, entering content), the optimum of a per-element point loss is not
  a state the game can be in.
- Which non-state it is depends on the loss: L1 erases what is uncertain; squared error superimposes it.
- Switching or mixing the losses moves the failure between these two non-states. It cannot remove it, because the target is a
  distribution and the output is one point.
- This is proven on the optimum itself, with no network involved: no architecture fixes it under these losses.
- Only an objective whose optimum is the distribution (a proper scoring rule: energy score, likelihood, diffusion / flow) makes
  samples calibrated. This is the scoreboard's energy-score result seen from the facts' side.

### 10-10 LITERATURE step 1 (11:00-11:20): the target space — what the recipe we copied assumes

Read after P-A / P-B were measured, on the cause they point to. Primary sources were read; arXiv PDFs are in the session scratchpad.

**Sources:**
- **V-JEPA 2 / 2-AC** (arXiv 2506.09985, `third_party/papers/2506.09985v1.pdf`):
  - its encoder's patch features are TRAINED by a patch-level objective: mask-denoising in representation space, L1 between the
    predictor's outputs and EMA-target patch features;
  - 2-AC then trains an action-conditioned predictor on those FROZEN patch features (L1 teacher forcing + a 2-step rollout loss).
    This is the recipe our world copies;
  - planning: CEM with horizon 1, re-planned at every step (Table 3);
  - §4.3 limitations name autoregressive error accumulation and long horizons;
  - stochastic outcomes are neither modelled nor discussed; the predictor is deterministic.
- **DINO-WM** (arXiv 2411.04983):
  - frozen DINOv2 patch features, teacher forcing, a squared-L2 "latent consistency" loss per frame (Eq. 1);
  - DINOPatch vs DINO CLS success rates 0.98 / 0.96 / 0.92 / 0.90 vs 0.96 / 0.58 / 0.60 / 0.44 (Table 2); ResNet and R3M spatial
    features are worse;
  - environments: PointMaze, Wall, Reach, PushT, Rope, Granular. The word "stochastic" occurs only in a reference title.
- **DINOv2** (arXiv 2304.07193, §4):
  - image-level DINO loss + a PATCH-LEVEL iBOT loss: student masked patch tokens predict the EMA teacher's patch tokens, + KoLeo;
  - Table 3b: the masked-image-modelling term improves patch-level tasks.
- **Ours, LeWM:** the CLS-only objective + SIGReg. "Patch tokens are never regularized by SIGReg and never predicted"
  (`d4mj/lewm.py`, `export` docstring). The last block's patch outputs receive no gradient from any loss.
- **Denoising Vision Transformers** (Yang et al., ECCV 2024, arXiv 2401.02957):
  - ViT outputs ≈ f(x) + g(E_pos) + h(x, E_pos): position-dependent, input-independent artifacts, traced to the position
    embeddings;
  - they "maintain a generally consistent relative position in the images" across video frames. Under a scroll, the content
    moves and the artifact does not: this predicts re-encoding of unchanged content.
- **Vision Transformers Need Registers** (Darcet et al., ICLR 2024, arXiv 2309.16588):
  - CLS-trained ViTs recycle low-information patch tokens to store global information;
  - Fig. 4c: only in models ≥ ViT-Large. Our encoder has width 192, so this is NOT claimed for it.
- **Littwin et al. 2024 (JEPA implicit bias):** JEPA objectives bias the representation they TRAIN toward predictive features,
  away from noisy ones. Our patch tokens are not trained by one.

**Reading:** the recipe we copied, deterministic per-element regression of frozen patch features (V-JEPA 2-AC, DINO-WM), is
demonstrated under three preconditions:
1. the target patch features were themselves trained by a patch-level predictive objective (iBOT, V-JEPA);
2. the next frame is (nearly) determined by the action, since no stochastic outcomes are modelled;
3. a short horizon with closed-loop replanning.

All three fail here, each measured:
1. our patch tokens are a by-product of a CLS objective: own state explains R² 0.59 of a token; 40% of the L1 change mass is
   night static; day scroll re-encodes unchanged cells ×25;
2. the RNG floor is 44% of the one-step L2, and 66-85% of every world's teacher-forced error is undeterminable;
3. the actor trains on 16-step open-loop imagination.

The next two runs test the mechanisms this predicts:
- `encoder_why.py`: are the non-state components position artifacts (DVT), context mixing, or the cell's own pixels?
- `statoracle.py`: is the loss's optimal statistic itself the failure?

### 10-10 RESULT (10:34): loss currency, prediction side — most of the per-element loss sits on cells whose facts are already right

`currency.py`: each world's teacher-forced predictions (all depths, TEST roots, 280,612 map cells), decoded by probes fitted on true
tokens. A cell's facts are correct when its tile class and all four mob flags match the simulator.

| world | cells with a wrong fact | L1 error on fact-correct cells | L2 error on fact-correct cells | mean L2: correct / wrong | AUC (error flags a wrong fact) |
|---|---|---|---|---|---|
| M16 (L1-trained) | 4.2% | **62.5%** | 25.0% | 0.80 / 54.7 | 0.974 |
| D continued | 4.5% | 72.7% | 49.2% | 1.49 / 32.4 | 0.952 |
| D scratch | 5.3% | 73.6% | 50.6% | 1.57 / 27.5 | 0.938 |
| E22MS zeroed | 5.8% | 67.5% | 38.2% | 1.30 / 34.5 | 0.959 |
| E22MS sample 0 | 5.4% | 63.6% | 29.2% | 1.20 / 50.7 | 0.967 |
| E22MS true codes | 3.7% | 78.9% | 57.5% | 1.13 / 21.4 | 0.953 |

- Per cell, the token error DOES flag wrong facts (AUC 0.94-0.97).
- But wrong facts are 4-6% of cells. In the L1 currency M16 trains on, 62.5% of its error mass is on fact-correct cells: night
  static, re-encoding, rendering detail (Part II-A).
- The D-loss worlds have halved their wrong-fact errors (27-32 vs 55), so the rest of their error, half of it even in L2, is on
  fact-correct cells.
- HUD (rounded health / food / drink / energy readouts): 17-20% of teacher-forced steps have a wrong value in every world, while
  41-71% of the HUD-token L2 is on steps whose values are all right. The wrong values are dominated by the timer decrements
  that no L16 world can predict (above).

### 10-10 RESULT (10:29): Part I scoreboard — M16 / D continued / D from scratch / E22MS (the user's breadth request)

Variants exported by `rollout_export.py` (lanes 170 / 171), read by `scoreboard.py`, `budget.py`, `entering_memory.py`; intervals from
`scoreboard_compare.py` (paired, episode-clustered, 2,000 draws). E22MS = 4 prior samples scored as one sampled world.

**1. The currency decides the winner.**
- teval's metric (squared error to the ONE recorded future), self-fed, / V:
  - M16 0.528 at depth 16; D continued 0.473; D scratch 0.494 (both better, resolved);
  - E22MS sampled **0.631 (worse, +0.103 [+0.073, +0.133])**; E22MS with the true codes 0.326.
- The energy score against all five real futures (proper for distributions), depth 16, E22MS sampled vs M16:
  - static cells **−12.1 [−13.9, −10.3]** (33.8 vs 45.9);
  - mob cells **−3.2** [−3.8, −2.6]; entering −1.6 [−2.0, −1.2]; HUD −4.8 [−5.4, −4.2];
  - better at every depth ≥ 4 in every class except the player token.
- The deterministic worlds tie or lose against M16 on mob / entering cells.
- **Our standard metric rewards predicting the mean and penalizes correct sampling.**
- D-loss worlds are worse than M16 at depth 1 on the player token (+0.09 to +0.29) and on static cells (+0.57 to +1.54). That is
  the copy-precision cost the error budget shows.

**2. Facts at depth 16 (imagined; TEST roots):**

| | M16 | D cont. | D scratch | E22MS zeroed | E22MS sampled | E22MS true codes | true tokens |
|---|---|---|---|---|---|---|---|
| tile accuracy | 0.776 | 0.783 | 0.776 | 0.759 | 0.731 | 0.844 | 0.978 |
| zombie / cow / skeleton count (truth 0.34 / 0.46 / 0.14) | 0.01 / 0.14 / 0.01 | 0.01 / 0.04 / 0.00 | 0.08 / 0.36 / 0.23 | 0.14 / 0.40 / 0.35 | **0.56 / 3.65 / 0.60** | 0.46 / 2.98 / 0.64 | 0.33 / 0.46 / 0.15 |
| health R² / bias | 0.59 / +0.43 | 0.58 / +0.31 | 0.56 / +0.67 | 0.59 / +0.59 | 0.57 / +0.16 | 0.63 / +0.09 | 0.999 |
| food R² / bias | 0.88 / +0.41 | **0.58 / −0.92** | 0.88 / +0.09 | 0.84 / −0.06 | 0.83 / +0.06 | 0.85 / +0.02 | 0.999 |
| inventory R² | **0.74** | 0.92 | 0.92 | 0.92 | 0.92 | 0.93 | 0.986 |
| facing / sleep accuracy | 0.90 / 0.95 | 0.89 / 0.95 | 0.93 / 0.98 | 0.93 / 0.98 | 0.93 / 0.98 | 0.93 / 0.98 | 0.95 / 1.0 |

- **Mob mass is not conserved in any world.** The deterministic worlds lose their mobs; the Δ-channel worlds MULTIPLY them, cows
  ×8 when sampled and ×6.5 even with the true codes: decoder-side, as for zombies yesterday, and worse for cows.
- D's loss gives inventory dynamics (R² 0.74 → 0.92).

**3. Timer-driven HUD changes are not drawn by any world** (decrements per imagined step vs the five real futures on the same steps):

| | health | food | drink | energy |
|---|---|---|---|---|
| real futures | 2.10% | 3.10% | 4.11% | 3.04% |
| M16 | 0.04% | 0 | 0 | 0.01% |
| D continued | 0.52% | 0.02% | 0.11% | 0.03% |
| D scratch | 0.23% | 0.10% | 0.23% | 0.66% |
| E22MS sampled | **1.56%** | 0.07% | 0.17% | 0.61% |

- Food / drink / energy fall on hidden 20-30-step timers (hunger > 25, thirst > 20, fatigue > 30). Every world draws them 5-40×
  too rarely.
- The same structure as the health cooldown: a hidden phase that a median / mean-seeking world rounds to "no change".
- Only health approaches the real rate, and only with sampling.
- **Why, exactly** (`craftax_classic/game_logic.py` 1256-1325, read; the diagnosis futures' hidden counters):
  - per step awake, the hunger / thirst / fatigue / recover timers grow by 1 (by 0.5 asleep, fatigue −1), and fire at 25 / 20 /
    30 / ±25-15;
  - eating, drinking and killing a cow reset hunger / thirst.
  - Across all 8,183 intrinsic decrements of the diagnosis futures (food 2,465, drink 3,276, energy 2,442), the timer just before
    the decrement is food 24.5-25, drink 19.5-20, energy 30 = **100% ≥ 15**.
  - So the reset that sets each phase is ≥ 20 steps back, ALWAYS outside an L16 world's 15-frame input window.
  - `teval.step` re-runs the world on its last ≤ 15 frames each imagined step (no state carried), so this is the world's
    information horizon.
- **For every L16 world these decrements are unpredictable from its inputs, whatever the loss:** epistemic, not allocation. A
  deterministic world can never draw them (P < 0.5 everywhere). A sampling world can only match the base rate (≈ 1 / period).
  Reaching them needs a memory horizon ≥ 31 steps.
- Contrast with the zombie cooldown (period 6 < 15): inside the window, learnable; its failure was allocation, fixed by D.


**4. Scroll fidelity** (depth 16):
- missed scrolls on true moves: 36% M16, 32% D scratch, 24% E22MS sampled, 17% with the true codes;
- false scrolls 5-8%; view aligned with the truth: 54-59%, 68% with the true codes.
- Wrong move decisions remain the main source of misalignment (E11b, 10-01).

**5. Entering terrain** (teacher-forced, `entering_memory.py`):
- 23% of entering cells were in view earlier in the window; 77% were never seen;
- on seen cells the deterministic worlds' error is 6-26 vs 4-10 for oracle recall of the last observed token (a memory deficit
  of 1.5-2.6×);
- never-seen cells: error 34-37, tile accuracy 78-82% (a deterministic guess of the likely class);
- E22MS sampled draws never-seen terrain with error 44-47 and tile accuracy 72-75%, as a sampler should against one truth;
  with the true codes 15.6 / 86%.

### 10-10 RESULT (10:23): Part II-B (dynamics error budget) — P-B holds on the RNG floor alone; the two losses split the budget in opposite ways

**Validations first:**
- `rollout_export` = teval bitwise for D from scratch (32 roots: gen / tf / one per-root errors identical).
- The scoreboard reproduces teval's M16 errors exactly at depths 1 and 4 and in every one-step class. At depth 16, 0.52809 vs
  0.52778: M16's stored teval file came from an earlier run with different batching. Per root, 9 roots differ at depth 1 (mean
  relative error 7.5e-5) and 986 at depth 16 (mean 0.5%, max 2,107). Self-fed rollouts amplify near-ulp numeric differences into
  different futures: an imagination-stability fact in its own right (diagnostic C3, below).
- The scoreboard reproduces e22_imagine's E22MS zeroed and sample-0 readings exactly (to the last rounded digit).
- `gradanat`'s per-category terms sum to `rollout_losses`' objective exactly (relative error 0 for L1, 6e-8 for D's loss).

**`budget.py`** (each world's ONE-STEP error, split per token by category; the floor = unbiased variance over the real keys from the
SAME full history = what no deterministic world can remove):

| M16 (L1-trained), one-step over all 17 actions × 4 keys | share of its L2 error | share of its L1 error | floor share within it | mean L2 / floor per token |
|---|---|---|---|---|
| unchanged cells, night | 31.4% | **43.3%** | **91%** | 4.11 / 3.73 |
| entering cells, day + night | 29.9% | 9.5% | 0.8% / 23% | 33.4 / 0.27 (day) |
| own state changed, day + night | 22.6% | 11.6% | 23-32% | 47.6 / 10.9 (day) |
| unchanged cells, day | 1.8% | 16.0% | — | 0.059 |
| HUD tokens, values changed | 9.4% | 4.6% | ~1% | 5.6 |
| HUD tokens, values same | 0.4% | 4.7% | — | 0.03 |
| whole frame | 143.7 / frame | 415 / frame | **44.2%** | |

- On the factual first step (5 futures) the floor share is 31.7%; entering cells are 49% of the L2 error there.
- **P-B** (≥ 30% of the teacher-forced squared error irreducible from the inputs): **holds on the RNG floor alone** (44% / 32%),
  before the epistemic part of entering terrain is counted.
- Entering-cell error is NOT random (floor 0.8% by day: the map is the same under every key). It is unseen or unlearned terrain.
  Seen-before vs never-seen is split next (C4).
- Of M16's excess over the floor: entering cells 52%, HUD changes 17%, own-state changes 16%, the rest spread.
- **Combined (budget rerun with entering cells split by whether their world position was in view in a context frame):**
  - the share of each world's error that its inputs cannot determine = the RNG floor + the excess on NEVER-SEEN entering cells;
  - the latter is an upper bound on its irreducible part: on those cells the worlds match a local-context predictor's class
    accuracy, 09-29.

  | world | one-step (17 actions) | factual first step |
  |---|---|---|
  | M16 | 0.706 | 0.783 |
  | D continued | 0.770 | 0.851 |
  | D scratch | 0.742 | 0.830 |
  | E22MS decoder alone | 0.662 | 0.767 |

  - Seen-before entering cells are only 1-3% of the L2 error.
  - In the L1 currency, unchanged cells at night (the static) plus never-seen entering cells are 45-55% of the error mass.
  - **Only ~15-34% of a world's per-element squared error is learnable from its inputs.**

**The two losses split the budget in opposite directions** (one-step, all 17 actions; mean L2 per token):

| | M16 (L1) | E22MS decoder alone (D's loss; Δ zeroed) | floor |
|---|---|---|---|
| own state changed (day) | 47.6 | **15.7** | 10.9 |
| HUD values changed (day) | 5.60 | **2.40** | 0.06 |
| unchanged cells (day) | **0.059** | 0.438 (×7.4) | 0.142 |
| HUD values same (day) | **0.030** | 0.151 (×5) | 0.013 |
| entering (day) | 33.4 | 31.3 | 0.27 |

- L1 (a median, every error counted at its absolute size) buys copy precision on the millions of small-change tokens.
- Squared error buys the rare large changes and gives up precision on unchanged content.
- Neither touches entering terrain. Each objective is good where the other is bad: the "fighting" in exact numbers.

**A sampled world pays twice the floor by construction** (E|X − Y|² = 2σ² for independent draws): E22MS sample 0's own-state error is
27.5, vs 21.8 for an ideal sampler (2 × 10.9). MSE against one realized outcome cannot judge a sampler. The energy score in the
scoreboard is the proper comparison.

**Gradient anatomy, 2-batch CPU test** (M6 at 6k; to be read only from the full GPU run, lane174):
- under L1, unchanged cells carry 46% of the loss and 64% of the gradient direction;
- under D's loss, entering cells carry 55% of the loss and 84% of the direction; own-state changes get −0.8%.

### 10-10 RESULT (10:14): Part II-A (encoder) — P-A holds: the token target is mostly not game state; 40% of its L1 change mass is renderer noise

**[CORRECTED 18:00: 40% (L1) / 34% (L2) is the mass of night cells with no state change. Measured static is ~74% of their squared change, ≈ 25% of the L2 change mass; the rest is re-encoding and lighting drift. See the 18:00 entry.]**

v2 runs: `encoder_diag.json` (Raw, all five futures, 80,765 frames) and `encoder_diag_{raw,ldad1,ldad10,tc}.json` (root + factual
future, 17,030 identical frames per encoder).

**A1 information — present** (Raw; held probes on true tokens):
- HUD health / food / drink / energy R² 0.999 / 0.999 / 0.998 / 0.998; inventory mean R² 0.997;
- tile class from the own token 97.9% (3 × 3: 98.3%); zombie / skeleton / cow / arrow AUC 0.990 / 1.0 / 1.0 / 1.000;
- facing 94.6%, sleep 100%, light R² 0.998.
- The same for the other encoders (tile 96.9-97.7%, zombie AUC 1.0, health 0.998-0.999).
- The encoder does not lose the state. (The "exact after rounding" HUD numbers are wrong: the targets are fp16 values × 9,
  not integers. Disregard them.)

**A2 noise floor — measured directly** (Raw; 2,628 of 15,836 all-alive root-depth cases have IDENTICAL simulator states across the
five futures):
- day: token variance across the five futures exactly 0 for map, HUD and player tokens;
- night: map tokens 0.29 / 1.83 / 4.55 / 5.93 at Chebyshev rings 1 / 2 / 3 / 4 around the player; HUD (18 tokens summed) 0.31;
  player token 0.053.
- The renderer's night static (source above) is the only randomness in rendering. It is radial as coded and reaches the HUD and
  player tokens through attention.

**A2 change composition** (Raw full run; L2 / L1 shares of the frame-to-frame target change):

| cause | day | night |
|---|---|---|
| own state changed (awake + asleep) | 13.5% / 7.6% | 3.6% / 2.4% |
| player sprite left the tile | 9.7% / 4.3% | 2.2% / 1.0% |
| neighbour changed | 1.1% / 1.5% | 2.1% / 2.4% |
| sleep onset / wake | 22.6% / 7.5% | 2.9% / 0.9% |
| **no state change** | 8.4% / **32.1%** | **34.2% / 40.3%** |

- Unchanged cells, mean Σ|Δ|² per cell (Raw):
  - day: still 0.06; scroll 1.4-1.7 (×25: shift re-encoding);
  - night: still 1.0 / 4.8 / 10.8 / 13.8 at rings 1-4 (= 2 × the measured static variance, as expected for independent noise
    in consecutive frames); night scroll 5.6-15.5.
- HUD tokens at unchanged HUD values: about 9-12 per step (18 tokens summed), day or night, still or scroll. Their change when the
  values change: 79-132.

**Encoder comparison** (identical 17,030 frames; scale-free):

| encoder | own-state R² | + lighting R² | MLP R² | L1 share: state change / day no-change / night no-change / sleep | scroll re-encoding ÷ own change | re-encoding explained by state |
|---|---|---|---|---|---|---|
| Raw | **0.592** | 0.888 | 0.865 | 0.19 / 0.32 / 0.40 / 0.09 | 0.028 | 0.13 |
| TC | 0.380 | 0.742 | 0.733 | 0.20 / 0.43 / 0.31 / 0.06 | 0.027 | 0.09 |
| LDAD λ1 | 0.335 | 0.759 | 0.753 | 0.13 / 0.57 / 0.27 / 0.03 | **0.271** | 0.50 |
| LDAD λ10 | 0.277 | 0.536 | 0.485 | 0.12 / 0.62 / 0.23 / 0.02 | **0.350** | 0.31 |

- The MLP R² is on a 5% subsample, lower than the ridge here (undertrained at 17k frames); with all five futures it is 0.902
  for Raw.

**Reading against P-A** (predeclared: refuted if own state explains ≥ 90% of both):
- not refuted, by a wide margin: own state explains 59% of a Raw token's variance and 19% of the L1 change mass (7.6% + 2.4%
  own, plus sprite and neighbours);
- Raw is the MOST state-aligned of the four encoders;
- LDAD's inverse-dynamics objective makes patch tokens re-encode ~10× more on scroll (relative to a state change), which fits
  the 09-30 finding that LDAD worlds do not scroll-copy;
- the information a world needs is in every encoder (A1). What changes is how much of the per-token target is about it.

### 10-10 interim (10:08): the Raw token target is mostly NOT game state — and part of it is renderer noise (Part II-A, P-A)

**Incident (09:28):**
- `encoder_diag` v1 (12.3 GB resident) + a scoreboard test + two exporters exhausted the 31 GB RAM;
- the kernel OOM-killed the diagnostic, and systemd then killed the whole VS Code cgroup, including both export lanes (started
  with setsid inside it).
- Relaunched 10:04 as separate `systemd-run --user` units with memory caps. The exports resume: M16, E22MS zeroed and
  sample 0 were complete; the generator state is restored from the sample-0 meta.
- v1's full-run results (80,765 frames, 1,002 roots) were printed before the kill; they are recorded here. v2 (day/night split,
  noise floor) is rerunning.

**Encoder facts:**
- The Raw encoder is a ViT (12 layers, 3 heads, width 192, patch 7, 9 × 9 patches, absolute positions; d4mj/lewm.py) trained by
  LeWM on the CLS only.
- Its patch tokens are "never regularized by SIGReg and never predicted" (lewm.py export docstring). Nothing makes a patch
  token local, shift-equivariant or stable.

**Static composition** (held R² of a map cell's token, 192 dims pooled; nested one-hot ridge; v1 full run):

| explanatory set | R² |
|---|---|
| own state (tile × mob) | 0.593 |
| + screen position | 0.597 |
| + global (light × sleep, tile × sleep, tile × light) | **0.888** |
| + 4 neighbours' states | 0.892 |
| + diagonal neighbours | 0.893 |
| MLP on all of the above | 0.903 |

- About 10% of a token's variance is not explained by the visible 3 × 3 state plus lighting. Lighting and sleep rendering
  account for 0.29 of R² by themselves.

**Frame-to-frame change of the target, by cause** (consecutive true frames, scroll-compensated, entering cells excluded; L2 =
Σ|Δ|², L1 = Σ|Δ|, the L1 loss's currency):

| cause (exclusive, in this order) | L2 share | L1 share | cells | mean Σ\|Δ\|² |
|---|---|---|---|---|
| own state changed | 17.1% | 10.0% | 74k | 55.8 |
| player sprite left the tile | 11.8% | 5.3% | 28k | 101 |
| a 4-neighbour's state changed | 3.1% | 3.9% | 185k | 3.7-10.2 |
| sleep onset / wake | 25.4% | 8.4% | 34k | 180-186 |
| **no visible state changed** | **42.6%** | **72.4%** | **4.38M** | 2.5 awake, 0.3 asleep |

- In the L1 currency, **72%** of the target's change mass sits on cells whose visible state did not change; all real state
  changes together are 19%.
- Unchanged cells move MORE the farther they are from the player: still steps 0.56 → 1.37 → 2.23 → 2.66 at distance 1 → 4;
  scroll steps 2.4 → 4.0.
- **Scroll re-encoding:** unchanged map cells on scroll steps move by Σ|Δ|² = 3.0; S4 (state, position, lighting, neighbours)
  explains only 11% of it.
- HUD tokens with the same HUD values: 11.3 (still) / 9.1 (scroll), vs 125 / 83 when the values change. Player token with the
  same facing and sleep: 0.13 still vs 10.8 on scroll.

**The cause of the distance gradient, from the source** (`craftax_classic/renderer.py` 408-423; `constants.py` 395):
- at night (daylight < 0.5) every map pixel is blended with `jax.random.uniform(state.state_rng) * 95 + 32`;
- the mask is `2 (0.5 - daylight) x (1 - exp(-0.5 r^2 / 0.5^2))`: zero at the player, near 1 at the edges;
- `state_rng` is redrawn every step (`game_logic.py` 1707).
- So 20.1% of the diagnosis frames carry fresh per-pixel random static. It is not game state, no world can predict it, and
  through the ViT's global attention it can reach every token.
- v2 measures it directly: the five futures' token variance where their full simulator states are identical. By day this is
  exactly 0 in the test, so rendering is deterministic.

**Reading so far (v2 numbers pending; not yet against P-A's thresholds):**
- own visible state explains 0.59 of the static target and real state changes are 19% of the L1 change mass;
- the rest is lighting and sleep rendering (deterministic), shift re-encoding (deterministic, but only 11% linear in state and
  position), and night static (random).
- Every loss we tried spends most of its mass there.

## 2026-10-09 — CPU prior/calibration and real-future controls; E22MP/E22MS endpoint review

**Execution:** CPU only, CUDA hidden, two threads; GPU jobs and numerical training/evaluation sources untouched.
Opened diagnosis panel: 1,002 roots / 143 episode clusters. Evidence and source/input hashes:
`20260927_levers/evals/cpu_review_20261009{,_sources}.json`; scripts `review_prior_{input,fit,calibration}.py`
and `review_real_keys.py`. Runtime logs in `artifacts/eda/levers_logs/cpu_review_*_20261009.log`;
per-root fit/calibration rows in `artifacts/eda/health_chain_v1/d4mj_prior_*_20261009.pt`.

- **E22MP completed:** canonical held code CE/accuracy 20k **2.3072/0.4698** → 60k **3.5717/0.4047**.
  CPU reproduces these to ≈0.00021 CE / 0.00014 accuracy. Confidence increases **0.5461→0.6888**.
  Four-sample hard-interior aligned capture **9.90→11.80%**, localization **48.0→51.8%**;
  declared halfway thresholds 17.5%/59% unmet. The predeclared “CE does not fall ⇒ capacity convergence”
  interpretation is unsupported: held generalization worsens.
- **Calibration intervention:** temperature selected on 20 validation seeds, tested on 43 disjoint seeds.
  Future-only through first terminal target, excluding context/post-terminal frames: 4,528 test transitions.
  20k CE **2.229868**, accuracy **0.485038**, selected T=1;
  60k CE **3.543180→2.587520** with selected T=2, accuracy unchanged at **0.415857**.
  Calibrated 60k−20k CE **+0.357653 [0.306848,0.408980]**, paired episode bootstrap (2,000 draws).
  This separates excessive confidence from residual class-order degradation; sampled-rollout benefits untested.
- **Prior input control:** grouped linear readout of the actual global frame token gives near-cell zombie
  AUC/AP **0.999526/0.981222** at 20k; **0.999948/0.997950** at 60k. Finite exploratory readouts;
  adjacent location survives compression, while sufficiency for motion/cooldown is not established.
- **Real-key event control:** other real futures (keys1–4) versus key0, E22 ordinary-hit/alive masks,
  no alignment conditioning: capture **38.50% [33.95%,43.00%]** overall and
  **24.32% [19.85%,29.14%]** at depths9–16; false **1.282%** overall.
  This is a `caught_all` stochastic reference, not an adjudication of `caught_aligned`.
- **E22MS endpoints landed:** four-sample soft localization/capture/false **57.57%/13.24%/1.29%**;
  hard interior **27.67%/10.41%/1.03%**. Hard decoding's earlier benefit does not transfer.
  Subsequent sampled codes can diverge because changed frames feed the prior; aggregate-only outputs
  cannot isolate the responsible routing branch or supply paired root intervals.
- **Stage-B terminal-position census (CPU):** all **7,501 terminal rows** have their only positive death
  target at position **20**. The **12,288 main rows**, across all **43 allowed window starts**, have zero
  positive death targets. This verifies positional bias in the prior's end-head supervision; learned
  shortcut exploitation has not been measured. `e22_imagine` discards this head, so the finding concerns
  future actor use, not the reported health/localization deficits. Source/protocol:
  `review_prior_terminal_positions.py`, appended evidence in `cpu_review_20261009.json`.
- **Reference lineage:** true-code **70.5%/25.2%** is E22M/D-continued/hard-interior;
  sampled **57.57%/13.24%** is E22MS/D-scratch/soft. No E22MS `postcodes` result exists.
  Their cross-model difference does not quantify the prior's contribution in the current best pipeline.

**Measurement correction:** historical “hit gradient share” is `||g_hit||/||g_all||`, not an additive
optimizer-update allocation. D continued ratio 7.557%, cosine −0.0337 ⇒ signed projection −0.2547%;
scratch D ratio 4.5777%, cosine 0.0104 ⇒ +0.0476%. Adam preconditioning is not measured.
D's health gains stand; these ratios do not isolate the loss components' causal contributions.
Zombie-neighborhood substitution closes 75.4% of the whole teacher gap, retaining a 16.91-point residual.
Detailed protocols, literature and retained leads: [REVIEW_20261009.md](20260927_levers/REVIEW_20261009.md).

## 2026-10-08 (live) — WHY HEALTH FAILS: link-by-link diagnosis (lead; user mandate: exact cause before any ablation)

**Decisions recorded:** Mamba (fmamba, L16) is permanently canonical. E20 services stopped and disabled 08:20.

### Summary at 19:30 (2026-10-09): the whole chain, measured; where it stands

1. **Teacher-forced health.**
   - Under uniform L1, zombie hits get ~0.3% of the objective, so no world draws them.
   - The Delta-IRIS tokenizer loss (D) fixes this generically: row 9 hits 47%, fresh 31, 1 false.
   - **From scratch D is better still** (test 3): hits 53%, fresh 62, drawn within 0.5 51%; phase +0.024 and adjacency
     +0.031 over D continued (resolved); no food cost (imagined food R² at depth 16 0.875 vs M16 0.881); teval better
     than M16; teacher-forced damage caught 81.9%.
   - Not a continuation artifact.
2. **Imagination.**
   - A deterministic world averages the zombie's random moves: D's random-move argmax is 1.1%, its zombie diffuses.
   - Restoring only the true cells near zombies restores 56% of damage; random cells restore 4.5%.
   - Imagination needs SAMPLING.
3. **Stochastic channel (Delta-IRIS posterior + prior on D).**
   - Warm: the channel stays shut (Park & Lee's mechanism).
   - Frozen-then-joint: the channel opens, but its encoder never encodes the zombie's move (probe at chance vs a positive
     control of 0.89; E16 likewise through 36k). Cause: loss share (scroll-entering terrain 49% of D's error, zombie cells
     1.9%).
   - **Causal test** (E22Z, label-derived zombie weighting): P2 1.1% → 82%.
   - **Label-free version** (E22M, moved-content mask whose threshold is the valley of a bimodal ratio): P2 80%.
4. **Sampled imagination.**
   - With the prior, zombies stay localized (E22Z: 87% of roots at depth 8 vs truth 93%, D 0%).
   - Zombie MASS is not conserved: E22Z multiplies (x10 by depth 16), E22M fades (x1/3). Both are decoder-side (they
     persist with the true codes).
   - E22Z's extras are fractional generated zombies (copy mass < 1 everywhere; the "copied twice" reading withdrawn).
   - **Hard (integer) decisions in the interior cure E22M's fade:** census 0.25-0.27 vs true 0.29, hits given adjacency
     0.20 vs 0.18, false 1.6%.
   - ITC's region rule fails again (generated HUD: 43% false).
5. **Where it stands** (label-free E22M + hard interior):
   - with the TRUE stochastic choices: localization 70.5% (reading 1 passes), caught 25% (30% required), false 1.3%,
     hits calibrated;
   - with the prior's samples: 48% / 10%. **The prior is the main remaining gap**; its held code accuracy is 0.470 and
     its curve was still rising at 20k.
   - Running: prior 20k → 60k (E22MP) and the same channel recipe on the from-scratch decoder (E22MS).


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
- **What Delta-IRIS's real loss would allocate** (`health_gradient.py … deltairis`; the same M16 s7 world and 20 training
  batches, scored with 1.0·mean L2 + 0.1·mean L1 + 0.01·per-frame worst-element L2 instead of the teacher L1):

| token-63 component | objective share, L1 → Delta-IRIS loss | backbone gradient-norm share, L1 → Delta-IRIS loss |
|---|---|---|
| hits | 0.27% → 0.78% | 0.16% → **1.85% (×11)** |
| unchanged | 0.30% → 0.18% | 1.37% → 18.4% |
| all of token 63 | 0.80% → 1.69% | 1.5% → 20.8% |

  The squared and worst-element terms put an order of magnitude more of the update on token 63. That covers the hits and also
  the noisy blends the world emits on unchanged frames. The loss the E16 port dropped is the component that addresses the
  diagnosed root. Measured at M16's own errors; indicative of allocation, not a trained result.
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
- **Approach adjacency, resolved at 5× data** (`health_controls.py pool`, `pool_adjacency.json`; all 71,918 pool events,
  M16 s7 h63 at window 15, episode split; supersedes the "not established" correction above, which was data-limited):

| probe target: zombie beside the post-move cell | all | scroll steps | scroll steps, zombie within 2 |
|---|---|---|---|
| raw frame-14 cells (13) + action | 0.9955 | 0.9285 | 0.758 |
| raw cells + action + the TRUE scroll (the move's outcome) | 0.9989 | 0.9749 | **0.935** |
| M16's h63 | 0.9939 | 0.904 | **0.750** |

  - On approach moves, h63 knows no more than the raw input without the move's outcome (0.750 vs 0.758).
  - The world itself predicts the scroll correctly on 99.0% of fresh hits (its corrt move gate reads the action token and the
    target tile), so the outcome exists inside the network; it is not routed to slot 63.
  - This is the exact form of the "scrolling problem": h63 does not combine the zombie's position with whether the player
    actually moved.
  - Caveat: the control uses the true scroll, not the world's predicted one (99% agreement on fresh hits).
- **The phase is computed, then discarded** (M16 s7; slot 63's residual stream at the output position, same probe, held AUC):

| slot 63 after | since = 6 | since 1..4 | since ≥ 7 |
|---|---|---|---|
| input embedding (current frame only) | 0.844 | 0.830 | 0.772 |
| layer 0 / 1 | 0.859 / 0.863 | 0.928 / 0.940 | 0.785 / 0.802 |
| layer 2 / 3 (peak) | 0.918 / 0.916 | 0.968 / 0.961 | 0.832 / 0.833 |
| layer 4 / 5 (output) | 0.911 / 0.887 | 0.943 / 0.911 | 0.814 / 0.790 |

  The middle layers build part of the phase from slot 63's history. The last two layers, which feed the copy-dominated head,
  lose a third of it. That is what a missing incentive predicts, not an inability of the per-slot Mamba to compute it.
- **False drops by context** (held unchanged, emitted drop at the 1.5 cut; **the "same phase failure" reading below is RETRACTED 19:10, see "Correction: false drops are the row-14 death cue"**):

| unchanged events, by context | M16 s7: false / n (rate) | M16 s8 | E20 A (since = 6 in h63 0.937) |
|---|---|---|---|
| all | 127 / 1,728 | 151 / 1,728 | 2 / 1,728 |
| a drop 1..5 transitions back (zombie in cooldown; true hit rate 0.008) | **108 / 327 (0.33)** | **119 / 327 (0.36)** | 2 / 327 |
| no drop in the window | 8 / 1,188 (0.007) | 16 / 1,188 | 0 |
| zombie beside in frame 14 / no zombie within 2 | 83 / 264 (0.31); 34 / 1,360 (0.025) | 98 / 264; 38 / 1,360 | 1; 1 |
| health 9 / health ≤ 3 | 0 / 766; 77 / 303 | 0; 90 / 303 | 0; 1 |

  85% (s7) and 79% (s8) of false drops fall inside the cooldown window. (Retracted reading: "the missing phase explains both
  halves". The contexts are right; the cause is the time row, as shown next.)
- **Correction: false drops are the row-14 death cue, not the phase** (19:10; `health_chain.py world … --window 10` puts the
  same last frames at output row 9 instead of 14):

| world, output row | hits / 1,211 | since = 6 hits / 146 | false / 1,728 | false in cooldown |
|---|---|---|---|---|
| M16 s7, row 14 (window 15) | 154 | 25 | 127 | 108 |
| M16 s7, row 9 (window 10) | 60 | 2 | **0** | 0 |
| G (E21), row 14 | 100 | 10 | 108 | 102 |
| G (E21), row 9 | 64 | 5 | **0** | 0 |

  - The training sampler ends 26.4% of windows on a death at frame 15, which is predicted from row 14. In fights, with recent
    hits, the world learned "row 14 → drop". This is the audit's death-position cue (E20's class-balanced endpoints removed it:
    E20 A, 2 false drops).
  - At an unbiased row both worlds draw ~5% of hits with zero false drops: an honest but almost silent health channel. That
    is B1 (+ B2).
  - **The two blockers at row 9** (`evidence_row9.json`; same probes and calibration as `health_evidence`; positive control at
    window 10: phase 0.9995):

| world, row 9 | generator beats copy | generator L1 on unchanged | since = 6 in h63 | approach adjacency in h63 | drawable from h63 | + oracle phase, adj | emitted hits / false |
|---|---|---|---|---|---|---|---|
| M16 s7 | 3.6% | 0.939 | 0.889 | 0.817 | **334** | 738 | 60 / 0 |
| G | 74.7% | **0.039** (copy 0.036) | 0.937 | **0.648** | **136** | 626 | 64 / 0 |

    - M16's h63 supports 334 drawable hits, but the head emits 60. That gap is B2.
    - G removes B2, but its generator loss (~6× the teacher term) turns h63 toward reconstructing token 63's content. The
      phase improves, approach adjacency falls (0.817 → 0.648), and the evidence drops to 136. G's 64 hits are B1-bound.
    - With B2 fixed, B1 binds.
  - **Consequence for E21:** the predeclared window-15 health readings mix B1 / B2 with the row-14 cue. Every arm is also read
    at row 9 (window 10) and reported beside them. B1 / B2 movement is judged at row 9; the row-14 numbers stay as
    predeclared.
- **The ingredients are sufficient in every world.** h63 + oracle phase + adjacency makes 627-805 of 1,211 hits drawable at
  ~0 false in all 14 worlds, including M6 at 6k.
- **Correction (approach adjacency; itself superseded at 5× data, see "Approach adjacency, resolved" below).** The adjacency deficit claimed in B1 is NOT established on the subset. Positive control on approach
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

- **Exposure.** E20 A / B raised the phase in h63 (0.937 / 0.950) and removed false drops (2 / 1,728). They did it with
  class-balanced endpoint batches importance-weighted back to the natural loss mass: same expected allocation, far lower
  per-batch variance. Under Adam's per-parameter normalization, consistent small gradients win. The world-model literature
  for exposure is Curious Replay (Kauvar et al., ICML 2023, arXiv 2306.15934, PDF read):
  - priority p_i = c·β^{v_i} + (|L_i| + ε)^α, with α = β = 0.7, c = 1e4, ε = 0.01;
  - no importance correction;
  - DreamerV3 on Crafter improves from 14.5 to 19.4%.

  Unlike E20's label-balanced batches it is generic, but it is uncorrected, so it shifts the L1 median: E14's prior-shift
  lesson applies.

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

**In-flight (provisional; not an endpoint reading), G at 2k vs M16, held subset:**
- generator beats copy on 73% of hits (M16: 2%);
- phase in h63 0.906 (M16: 0.888);
- emitted hits 99 (M16: 154), false drops 108 (M16: 126), of which 98 sit in the cooldown window (M16: 107).

B2 removed, B1 unchanged: a usable generator alone draws no more hits, and it still drops health inside the cooldown, as the
diagnosis predicts.

**E21 v2 results, part 1: G and C0** (20:50; `evidence_e21v2_ab.json` via `health_e21_table.py`; row 9 from `__w10` files):

| arm (6k from M16) | generator beats copy | since = 6 / 1..4 in h63 | approach adjacency in h63 | drawable from h63 | emitted hits / false, row 14 | emitted hits / false, row 9 |
|---|---|---|---|---|---|---|
| M16 (parent) | 2.4% | 0.888 / 0.909 | 0.778 | 286 | 154 / 126 | 60 / 0 |
| C0 (control) | 4.1% | 0.897 / 0.898 | 0.761 | 290 | 124 / 107 | 64 / 0 |
| G (+ gen-loss) | **79.7%** | **0.944 / 0.936** | **0.619** | 186 | 100 / 108 | 64 / 0 |

- Reading 1 (B2 fixed, ≥ 30%): **TRUE** for G (79.7%; C0 4.1%).
- Control: the plain continuation does not grow the phase (C0 at 2k / 4k / 6k: 0.886 / 0.892 / 0.897). G's phase gain
  (0.906 / 0.923 / 0.944) is the generator loss.
- The generator loss trades approach adjacency for the phase at h63 (0.761 → 0.619). Drawable evidence falls (290 → 186), and
  at row 9 G draws exactly what C0 draws (64). Fixing B2 alone does not draw health; B1 binds.

**E21 arm D, predeclared 17:50, before any E21 v2 result:** M16 s7 continued 6,000 updates exactly as C0, with the teacher L1
replaced by Delta-IRIS's tokenizer loss in token space: 1.0·mean L2 + 0.1·mean L1 + 0.01·mean over frames of the frame's worst
element L2 (vmicheli/delta-iris `tokenizer.py`).
- Rationale (measured above): the same world and batches give hits 11× the backbone gradient share (0.16% → 1.85%), with no
  label and no new head.
- Risk: L2's optimum is the conditional mean, so drawing a 2-point drop past the 1.5 cut needs P(hit) > 0.75. The phase strata
  (since = 6: 0.87; since 1..5: 0.008) are on the right side of it.
- Readings: the same four as E21. The cost reading is also reported in teval's own currency (MSE / copy), which an L2-trained
  world may favour; one-step L1 is reported beside it.
- Implementation waits until E21's last arm has started, so every E21 arm runs the same `tworld.py`.

### Summary at 03:02 (2026-10-09): the continuation question, and the first generic objective that draws damage

**Q (21:00): if a result is negative, do we know it is not caused by continuing instead of training from scratch?**
- *For the representation (B1): yes, now measured.*
  - Continuing M16 with an objective that asks directly (oracle labels; diagnostic only) installs the cooldown phase (0.888 →
    1.000) and approach adjacency (0.778 → 0.996) at h63 within 500-1,000 updates.
  - Plasticity signatures show no gross loss: 0 dormant units; random-target fitting only 10-20% slower than fresh init.
  - So a continuation arm that fails to install B1 fails because its objective asks too weakly, not because the network is
    locked in.
- *What is not yet answered:* whether the same weak ask behaves differently when present from step 0 (critical periods). That
  is test 3, the from-scratch replications of GE (lane122, ETA ≈ 12:00) and D (lane131, ETA ≈ 21:00). Both are matched to M6 /
  M16, and `tworld.py` with flags off is bit-identical to M6's and M16's sources (`tworld_equivalence.py`).
- *Measured limits of the arms' own doses:*
  - GE's random event head was under-trained (EASimulus trains ~985k updates; our 6k reached token-63 AUC 0.57).
  - E′ (LP-FT) fixed that: it installs approach adjacency, but the phase only transiently.

**D, the Delta-IRIS tokenizer loss (1.0·L2 + 0.1·L1 + 0.01·worst element), continued from M16 for 6k updates:**
- At the unbiased row 9 it draws **569 / 1,211 held zombie hits (47%), 31 fresh, 1 false drop in 1,728** (M16: 60 / 0 / 0).
- It installs B1 without labels (phase 0.952, adjacency 0.964, 790 hits drawable from h63 vs C0's 335) and fixes B2 (generator
  beats copy 85%).
- It depicts zombies far better: near-player zombie AUC 0.78 → 0.96 one-step, 0.53 → 0.61 at depth 16.
- **Limit, measured 03:08:** teacher-forced it catches 73% of real damage on held roots (simulator truth). In its own imagination
  it draws only 6% of aligned hits (every other world: 0). Health in self-fed rollouts, what H16 decisions use, remains mostly
  missing.
  - Localized by substitution at 03:23: feeding back the true HUD restores nothing (4.5% → 5.9%); feeding back the true map
    restores most (→ 58.5%). D's imagined zombies are beside the player at only 8.9% of real damage steps (truth 73.6%).
  - Zombie moves are random 25% of the time (Craftax source). A deterministic world averages them away, so imagination needs
    sampling.
  - Confirmed 09:05-09:17:
    - D's one-step zombie is a blend in proportion to the true move probabilities;
    - self-fed blends diffuse it (peak 0.44 → 0.11, no cell > 0.3 by depth 8);
    - restoring only the cells around true zombies restores 56.4% of damage (true map 58.5%; size-matched random cells:
      4.5%, no effect).
  - Literature (PlaNet, DreamerV2, video prediction) names exactly this failure of deterministic transitions.

**Test 3 for GE (08:47):** from scratch it ends worse than M16 on every ingredient and draws 7.5% of hits at row 9. Its negative is
not a continuation artifact. Measured cause: the generator loss crowds token-63 hits to 0.043% of the gradient.
- teval MSE improves (one-step −0.012, depth 16 −0.052).
- Costs:
  - L1 +20% (fewer large errors, more small ones).
  - The imagined food reading drifts down (R² at depth 16 0.88 → 0.58). Diagnosed exactly: D predicts a fractional decrement
    every step (−0.10 / step) because the intrinsic timers (21-31 steps, verified in the Craftax source) outlast the 16-frame
    window, and a mean-seeking loss averages an event it cannot time.
  - The predeclared window-15 health reading fails on the inherited row-14 death cue (8.6% false; the cue comes from the
    sampler).
- **Next (to predeclare after the from-scratch runs):** faithful Delta-IRIS pairs this loss with a stochastic Δ channel, which
  samples, rather than averages, what the window cannot time. E16 built the channel with the wrong loss; D used the loss without
  the channel.

### Could a negative come from continuing instead of training from scratch? (user question, 21:00; predeclared 21:06, commit b7d58a59)

**Status at 21:06: not known.** Every E21 arm (and E19, E20) continues a trained parent (M16 = 36k 6-frame + 6k L16). No arm has
been trained from scratch. What is measured so far:
- The continuation is not frozen. G moved the phase 0.897 → 0.944 (control flat) and the generator 4% → 80% in 6k; E20 B moved the
  phase to 0.950; E19 C moved the generator 0.952 → 0.659.
- The endpoints are not converged: G's phase still rose at 6k (0.906 / 0.923 / 0.944).
- Our own precedent says continuation can produce a false negative: E16 amendment 2 (10-03). A converged deterministic world
  started Delta-IRIS training in Bowman et al.'s "decoder ignores the latent" equilibrium; the from-scratch recipe was required.

**Literature read (full texts):**
- Ash & Adams 2020 (arXiv 1910.08475): warm-started networks generalize worse at equal training loss. Mechanism: gradient
  imbalance between new and already-fitted data. Shrink-and-perturb (λ = 0.6, noise scale 0.01) matches fresh init.
- Achille et al. 2019 (arXiv 1711.08856, critical periods): connectivity consolidates early; deficits removed late are not fully
  recovered.
- Lyle et al. 2022 (arXiv 2204.09560): capacity loss, measured by target-fitting capacity (Def. 1) and feature rank (Def. 2,
  ε = 0.01). Lyle et al. 2023 (arXiv 2303.01486): plasticity loss tracks loss-landscape curvature, often without saturated
  units; layer normalization mitigates it (our backbone is pre-norm).
- Sokar et al. 2023 (arXiv 2302.12902): τ-dormant units (score = normalized mean |activation|; τ = 0 / 0.025 / 0.1).
- Nikishin et al. 2022 (arXiv 2205.07802): primacy bias, an early-data overfit that resets cure.

**Three measurements, in increasing strength:**
1. *Signatures* (`health_plasticity.py metrics`, lane117): dormant fractions, feature rank, random-target fitting for fresh init,
   M6 at 6k / 18k / 36k, M16, C0, G. Descriptive: they show whether capacity loss is present, not that it caused a negative.
2. *Oracle positive control* (`health_plasticity.py oracle`, lane118): M16 continued 2,000 updates with its own recipe plus
   λ × (CE of the true `since` class + BCE of the true approach adjacency) read from h63.
   - Amended 21:50, before its first run, for the same reason as E′: the two oracle heads are first fitted on frozen h63 (LP-FT;
     60 batches of order 13, 3,000 steps, lr 1e-3).
   - λ is set on the first fine-tuning batch so that the oracle term equals the teacher term (the original 0.015 assumed random
     heads), and logged. Labels validated against
   `strata.pt`: since 99.7%, adjacency 100%. Diagnostic only, never a fix.
   - Readings at 500 / 1,000 / 2,000 updates, health_chain subset, held episodes, window 15, same probes as every other world:
     M16 is plastic for the phase if since = 6 and since 1..4 in h63 are both ≥ 0.97 (input ceiling 0.999); plastic for adjacency
     if adjacency|scroll, zombie ≤ 2 ≥ 0.90 (M16 0.778).
   - If plastic: a continuation arm that fails to install an ingredient failed because its objective asks too weakly, not
     because of lock-in. The from-scratch test (3) still decides whether the *same weak ask* works when present from step 0
     (critical periods).
   - If not plastic: no continuation negative on B1 can be interpreted, and every conclusion waits for (3).
3. *From-scratch replication* (decisive): the selected arm trained through the exact canonical recipe from step 0. Stage 1:
   `tworld.py --head corrt --pool raw --loss teacher --seed 7 --updates 36000 --backbone fmamba --snapshots` + the arm's flags.
   Stage 2: `--pool rawlong --frames 16 --windows 40 --init <stage 1>`, 6,000 updates.
   - Same init seed, batch order (11) and update count as M6 → M16; only the loss differs.
   - Matched controls exist: M6's snapshots (6k … 36k) for stage 1 and M16 for stage 2. ~8.5 h per arm.
   - Readings: E21's four, at row 9 and at row 14, plus stage-1 snapshot curves of generator fidelity, `since` 1..4 and
     adjacency against M6 at the same updates.
   - Interpretation:
     - from scratch passes where the continuation failed → the continuation caused the negative;
     - both fail with the same ingredient values → that negative is not a continuation artifact;
     - from scratch fails differently → reported as such, with the stage-1 curves locating where they diverge.
   - Selection, fixed now: GE (E21's predeclared main arm) runs first. D runs second if its continuation reading on B1 is
     positive or ambiguous. If every continuation arm is negative, GE and D both run (D addresses the diagnosed root most
     directly: 11× hit gradient share).
   - *Structural note, recorded 23:10 before any stage-1 result:* stage 1 uses 6-frame windows, where a cooldown phase of 6 cannot
     be represented (it needs ≥ 6 frames of history). In the from-scratch run, as in M16 and the continuation, since = 6 can only
     be learned in the 6k L16 stage.
     - What "from scratch" changes is the 36k of shaping before that stage: generator, event head, since 1..4, adjacency, all
       read on the stage-1 snapshots against M6's.
     - A from-scratch result on since = 6 is therefore a statement about stage 2 starting from a differently shaped network,
       not about 42k updates of phase learning.

**Test 1 result, signatures** (21:30; `plasticity_metrics.json`; 1,024 subset events):

| world | dormant units, τ = 0.1 (max over 6 layers) | feature rank, slot 63 / all slots (ε = 0.01) | random-target MSE after 1,000 steps |
|---|---|---|---|
| fresh init (seed 7) | 0 | 255 / 255 | 0.102 |
| M6 at 6k / 18k / 36k | 0 / 0 / 0 | 255 / 255 | 0.112 / 0.117 / 0.120 |
| M16 | 0.1% | 255 / 255 | 0.122 |
| C0 / G | 0.1% / 0.3% | 255 / 255 | 0.123 / 0.118 |

- No dormant-unit or rank collapse. Lyle's ε = 0.01 cutoff saturates at our feature scale (singular values 2-14), so feature rank is
  uninformative here, not a pass.
- Target-fitting capacity falls modestly and steadily with training (+10% at 6k, +20% at M16 over fresh), the "modest" loss Lyle
  et al. report for RL checkpoints.
- Reading: no gross loss of plasticity. This cannot exclude a critical-period effect (information routed early is not
  redistributed), which is what tests 2 and 3 address. Also exact: the continuations run at the same constant lr 1e-4 as the
  from-scratch runs (`_phase_lr`: warm-up, then constant), not at a reduced fine-tuning rate.
- Precondition for test 3 verified (`tworld_equivalence.py`, CPU, reference Mamba backend): current `tworld.py` with flags off
  trains exactly as M6's (@60b0c22f) and M16's (@ae0868ec) sources. Parameters, loss and every gradient are bit-identical;
  lane119 re-runs this after the D patch before it starts.

**GE in-flight at 2k (21:38) and a binding caveat on the E / GE arms, recorded before their endpoints:**
- GE @ 2k: generator beats copy 75.7% (G 73%); since = 6 in h63 0.909 (G 0.906, C0 0.886); the event head's own token-63 logit
  ranks held hits vs unchanged at **AUC 0.470**. The event term has not yet acted on token 63.
- Dose, measured against the source:
  - EASimulus on Craftax (`config/base.yaml`, `benchmark/craftax.yaml`) trains its world model, event head included, at AdamW lr
    2e-4, 8 sequences × 20 steps, 100 updates per epoch from epoch 150 of 10,000: ≈ 985k updates, ≈ 160M step-targets.
  - An E21 continuation is 6k updates at lr 1e-4 behind a 1,000-update warm-up, 3.6M step-targets (≈ 2%).
  - The pre-flight already showed SlotEvent learns token 63 slowly even at lr 1e-3 on frozen h (AUC 0.61 at 600 steps, 0.74 at
    3,000).
- Consequence: a negative E or GE *continuation* result on B1 is NOT evidence that EAWM's event prediction fails. It would be an
  under-dosed test (the half-truth to avoid).
  - The event head's own token-63 AUC is reported with every E / GE reading. Only where it is high (≥ 0.85) does a B1 null say
    something about the mechanism.
  - The from-scratch GE run (lane119: 42k updates, 7× the continuation) is the first test that gives the event head a budget;
    its event AUC is read at every snapshot.

**Design change before E ran (21:41; lanes launched 21:42): E → E′ = LP-FT, and the D patch moved between trainings.**
- *Why E as queued would be a half-truth.* Its event head starts random. In GE that head's token-63 AUC was 0.470 at 2k, and the
  pre-flight shows the head learns token 63 slowly even at 10× the world's learning rate.
  - Kumar et al. 2022 (arXiv 2202.10054, read): fine-tuning a pretrained network behind a random head distorts its features
    (proved for two-layer linear networks, measured on 10 shifts). Linear-probe-then-fine-tune (LP-FT) is the remedy.
- *E′:* LP step (`event_preflight.py lp`, lane120) trains SlotEvent alone on frozen M16 with exactly the E21 event term:
  10,000 steps, AdamW lr 1e-3, rawlong batches of order seed 13 (not the fine-tuning order 11), fixed steps, no selection.
  The FT step continues M16 + that head for 6,000 updates with `--event`, everything else as C0.
  - E′ answers the arm's question (does event prediction install the phase into h63?) without the random-head confound.
  - Readings unchanged, plus the head's token-63 AUC at 0 / 2k / 4k / 6k.
- *Provenance bug fixed:* `tworld.py` writes `script_sha256` from the file on disk at save time. lane114 would have patched
  `tworld.py` for D while E was training, stamping E's checkpoints with the patched file's hash.
  - lane121 now applies the patch right after GE's training process exits, before any other training starts.
  - It re-runs `tworld_equivalence.py` on the patched file.
- Lanes 110 / 114 / 115 / 119 were idle waiters (no job started) and were stopped at 21:40. Their readouts now run on
  C0 / G / GE / E′ / D in lane121, and the from-scratch GE replication in lane122, gated on lane121.
- *LP step result* (`event_lp.json`, 21:50): the fitted head's token-63 logit on held subset events.
  - Hit vs unchanged AUC 0.619 / 0.700 / 0.734 / 0.751 / 0.753 at 1k / 3k / 5k / 8k / 10k steps: converged at **0.753**.
  - Fresh hits vs unchanged ≈ 0.50 throughout.
  - So frozen M16 features support this head only to 0.75 on token 63 and not at all on approach hits, consistent with the
    diagnosis (phase weak at h63; approach adjacency not routed there).
  - E′ reading: does fine-tuning with this fitted head push the head's AUC, and the phase / adjacency in h63, above the
    frozen-feature ceiling?

**E21 part 2: GE at 6k, with the noise floor measured** (22:25; `ci_e21_6k.json`, `health_ci.py`: 5 probe seeds; episode-clustered
paired bootstrap over 1,265 held episodes, 2,000 draws):
- GE endpoint, row 14: generator beats copy 81.5%; phase (since = 6) 0.908; event head token-63 AUC 0.570 (2k 0.470, 4k 0.540);
  emitted 73 hits / 37 false.
- Noise floor: probe-seed std 0.001-0.003 for the phase, 0.007-0.015 for adjacency. Bootstrap 95% half-widths ≈ 0.015 (phase),
  ≈ 0.055 (adjacency). GE's snapshot swings (0.909 / 0.932 / 0.908) are real model change, not probe noise.

| paired contrast, 6k | since = 6 in h63 | since 1..4 | approach adjacency |
|---|---|---|---|
| M16 → C0 (continuation only) | +0.008 [−0.002, +0.017] | −0.006 [−0.017, +0.005] | −0.003 [−0.034, +0.028] |
| C0 → G (+ generator loss) | **+0.044 [+0.031, +0.058]** | **+0.034 [+0.020, +0.047]** | **−0.154 [−0.213, −0.097]** |
| C0 → GE | +0.011 [−0.001, +0.024] | +0.011 [−0.003, +0.026] | **−0.081 [−0.142, −0.019]** |
| G → GE (+ event head) | **−0.033 [−0.042, −0.024]** | **−0.022 [−0.033, −0.013]** | **+0.074 [+0.023, +0.128]** |

- The plain continuation moves nothing (all three unresolved).
- The generator loss trades approach adjacency for the phase, both resolved.
- The barely-trained event head pulls G back toward C0 on both, also resolved. Even an under-trained event term reshapes h63.
  Its direction (more adjacency, less phase) is not what B1 needs: B1 needs both up.
- Under the binding caveat (event AUC 0.570 < 0.85), GE says nothing about whether a competent event head installs the phase.
  E′ (LP-FT, head at 0.753 from the start) and the from-scratch GE are the tests.

**EAWM's second component, read in the code (22:55; MarquisDarwin/EAWM @ 49eebbe):** the paper's event-aware observation loss
ℒ_o = ℒ′_o + Σ ω·g(α)·[𝟙(event) − 1]·ℒ′_o(o_i).
- *EASimulus* (their Craftax system, `get_next_token_logits_and_labels`): plain per-token CE, **no event-aware reweighting**. Its
  only EAWM component is the event head (what E21 ports).
- *EADream* (pixels; `tools.AttentionMSEDist`, `configsc.yaml`):
  - per-element weight (1 − ω) + ω·event with ω = `attention_weight` 0.5, i.e. an event element counts 2× a non-event one;
  - applied only while fewer than `event_pred_ratio` = 5% of the frame's elements are events (`dizzy: True`, the GES gate).
- *Measured in our data:* 31% of map tokens change per frame (τ = 2.5). The 5% gate would be closed on nearly every moving frame,
  including every fresh hit (all are scroll steps). Even open, ×2 on a 0.27% share is ≈ 0.5%, against Delta-IRIS's measured ×11.
- *So, exactly why EAWM does not fix health here:* its faithful components are a slowly learned event head plus, in its pixel
  variant only, a ≤ 2× event reweighting that our scrolling view gates off. Neither addresses the diagnosed allocation root.
  E14a's "EAWM" (a ×1.7 token reweighting on a frozen backbone) tested neither component.

**Test 2 result: oracle positive control** (23:02; `evidence_oracle.json`; λ set automatically to 0.0613; LP-FT heads):

| oracle continuation of M16 | since = 6 / 1..4 in h63 | approach adjacency in h63 | drawable from h63 (fresh / false) | + oracle ingredients | emitted, row 14 (fresh / false) |
|---|---|---|---|---|---|
| M16 (start) | 0.888 / 0.909 | 0.778 | 286 (4 / 5) | 768 | 154 (4 / 126) |
| @ 500 | 0.997 / 0.999 | 0.967 | 641 (104 / 1) | 771 | 320 (31 / 96) |
| @ 1,000 | 1.000 / 0.999 | 0.994 | 758 (145 / 1) | 789 | 252 (30 / 58) |
| @ 2,000 | **1.000 / 1.000** | **0.996** | **823 (184 / 0)** | 837 | 345 (43 / 84) |

- **Reading TRUE on both counts:** M16 is plastic for the phase (≥ 0.97 within 500 updates) and for approach adjacency (≥ 0.90
  within 500). h63 alone then carries nearly everything the oracle ingredients add (823 vs 837 drawable).
- **Answer to the continuation question for B1:** a continuation is not what blocks the ingredients. M16's backbone installs both
  in 500-1,000 updates when the objective asks directly. A continuation arm that fails to install them fails because of its
  objective. Whether the *same weak ask* behaves differently from step 0 (critical periods) is test 3's question (lane122).
- **The chain, confirmed in order:** with B1 satisfied, B2 binds. 823 hits are drawable from h63, but the copy-gated generator
  (beats copy on 1.5%) lets the head emit 345.
- The cooldown false drops at row 14 fall as the phase is installed (107 → 86 → 33 → 64 across snapshots). The row-14 time cue
  produces them; the phase is what lets the world veto them. This refines the 19:10 correction; row 9 (lane124) is the clean read.
- **Sufficiency test, queued (lane125, 23:05):** oracle (B1) + generator loss (B2) together from M16, 2,000 updates, read at
  rows 14 and 9. If B1 + B2 are jointly sufficient, this world draws most held hits at row 9 with near-zero false drops. That
  closes the causal chain. The remaining problem would then be one thing only: a label-free objective that installs B1.
- **Test 2 at the honest row** (23:17; `evidence_oracle_row9.json`; window 10 = output row 9, no time cue):

| world, row 9 | since = 6 / adjacency in h63 | drawable from h63 (fresh / false) | emitted hits (fresh / false) | generator beats copy |
|---|---|---|---|---|
| M16 | 0.889 / 0.817 | 334 (4 / 7) | 60 (0 / 0) | 3.6% |
| oracle @ 500 | 0.999 / 0.973 | 601 (110 / 1) | 66 (0 / 11) | 3.9% |
| oracle @ 2,000 | **1.000 / 0.997** | **848 (184 / 2)** | **70 (2 / 3)** | 3.1% |

  - At row 9, B1 is fully satisfied (70% of hits drawable, 89% of fresh hits), yet the world emits 70. The row-14 figure of 345
    was the time cue now gated by the phase, not general health drawing.
  - Exact statement: **with B1 satisfied, every remaining missed hit is in the 848 → 70 gap, and that gap is B2.** The
    copy-gated generator can produce a dropped health token only where deaths trained it (row 14 of end-aligned death windows).
  - The sufficiency test (lane125: oracle + generator loss) now decides whether B1 + B2 are jointly sufficient at row 9.
- Intervals (`ci_oracle.json`, paired episode bootstrap, row 14):
  - M16 → oracle @ 500: phase +0.105 [+0.088, +0.124], since 1..4 +0.086 [+0.071, +0.103], adjacency +0.205 [+0.157, +0.259];
  - 500 → 2,000: +0.003 / +0.001 / +0.027, each resolved but small.
  - The backbone acquires both ingredients almost entirely within the first 500 updates.

**E′ (LP-FT event head) in flight** (lane126; row 14):

| E′ | event head token-63 AUC | since = 6 in h63 | generator beats copy | emitted hits (fresh / false, cooldown) |
|---|---|---|---|---|
| start (LP) | 0.753 | 0.888 (M16) | 2.4% | 154 (4 / 126, 107) |
| 2k | 0.763 | 0.888 | 3.6% | 121 (0 / 88, 77) |
| 4k | 0.777 | **0.930** | 4.6% | 183 (**14** / 139, 97) |

- With a competent head from the start, the event term begins installing the phase between 2k and 4k: +0.04, against the
  oracle's +0.11 within 500 updates.
- Measured differences between the two asks: the event term's gradient into h63 on hits is 0.38× the teacher's, while the
  oracle term equals the teacher term. The event target is 1 bit ("token 63 changes next step"); the oracle's is the 16-way phase.
- **E′ endpoint** (6k; 00:35; `ci_evlp.json`):
  - event head AUC 0.776; generator beats copy 5.9%;
  - row 14: 127 hits (2 fresh) / 109 false; **row 9: 79 hits (1 fresh) / 0 false** (M16 60, C0 64, G 64, oracle 70).

| paired contrast | since = 6 in h63 | since 1..4 | approach adjacency |
|---|---|---|---|
| C0 → E′ @ 4k | **+0.031 [+0.021, +0.042]** | **+0.028 [+0.018, +0.039]** | **+0.128 [+0.094, +0.165]** |
| C0 → E′ @ 6k | +0.009 [−0.003, +0.021] | +0.005 [−0.005, +0.015] | **+0.089 [+0.053, +0.127]** |
| GE → E′ (random vs fitted head, ± generator loss) | −0.002 [−0.015, +0.010] | −0.007 [−0.020, +0.007] | **+0.170 [+0.116, +0.230]** |
| E′ 4k → 6k | −0.022 [−0.032, −0.012] | −0.023 [−0.033, −0.014] | −0.040 [−0.068, −0.011] |

- **Reading:** with a fitted head from the start, EAWM's event objective installs approach adjacency in h63 (C0 0.763 →
  0.85-0.89, resolved). That is the "scrolling problem": predicting a token-63 change requires knowing a zombie will be
  beside the post-move player.
- Its effect on the phase is small and transient (+0.031 at 4k, gone at 6k). At EAWM's dose (gradient 0.38× teacher on hits) it
  is a partial, real B1 mechanism, not a complete one. The oracle installs both in 500 updates.
- Health at row 9 does not follow (79), because B2 binds: E′ has no generator loss.
- This E′ result is **not** a continuation artifact in the sense of test 2: the same continuation installs both ingredients when
  asked strongly. What it measures is the strength of EAWM's ask.

**D (Delta-IRIS tokenizer loss) in flight at 2k** (00:58; `health_row9.py`):

| world | row | hits / 1,211 | fresh / 207 | false / 1,728 (cooldown) | generator beats copy |
|---|---|---|---|---|---|
| M16 | 9 | 60 | 0 | 0 | 3.6% |
| D @ 2k | 9 | **421 (34.8%)** | 1 | **2 (1)** | **71.3%** |
| D @ 2k | 14 | 570 | 4 | 169 (133) | 64.2% |

- At the unbiased row, after 2k updates, D passes the predeclared health reading (≥ 30% of held hits, ≤ 2% false).
- The change is the loss shape only: 1.0·L2 + 0.1·L1 + 0.01·worst element; no label, no new head.
- It fixes B2 by itself: the generator beats copy on 71% of hits with no generator loss, because under L2 the large token-63
  errors carry a large gradient.
- Not yet known:
  - approach adjacency (fresh hits 1/207 suggest it is not installed);
  - the B1 ingredient AUCs;
  - stability at 4k / 6k;
  - the world-quality cost of an L2 objective (teval after the endpoint, judged in both MSE and L1 currencies as predeclared);
  - the from-scratch question for D.
- Precision of drawn hits against the exact next health (`health_row9.py`, row 9):

| world | hits drawn | mean \|pred − true\| on drawn | within 0.5 | over / under by > 1 | unchanged \|err\| |
|---|---|---|---|---|---|
| M16 | 60 | 3.97 | 2% | 0 / 59 | 0.543 |
| oracle | 70 | 3.60 | 6% | 1 / 61 | 0.553 |
| E′ | 79 | 2.41 | 9% | 8 / 59 | 0.542 |
| D @ 2k | **421** | **1.27** | **22%** | 164 / 83 | 0.577 |

  - The few "hits" other worlds draw at row 9 are partial depictions of large drops, under-drawn by > 1 unit.
  - D's are more numerous and much closer, but not yet precise. Mean predicted drop 3.3 vs true 2.75; the reader's own error on
    true frames is ≈ 0.4-0.55.
  - Unchanged-frame error is equal across worlds, so D adds no noise there.
  - So D learns *when* a hit lands, and roughly its size.
- **D @ 4k:**
  - row 9: 478 hits (39.5%), fresh 2, false 2 / 1,728; generator beats copy 82%; drawn |err| 1.19 (27% within 0.5);
  - row 14: phase 0.931, 612 hits, fresh 16, false 152 (123 in cooldown; the row-14 cue persists).
  - Rising from 2k on every health reading; approach hits are still not drawn at row 9.
- **D endpoint (6k; 02:27):**

| D | row 9 hits / 1,211 | fresh / 207 | since = 6 / 157 | false / 1,728 | generator beats copy | drawn \|err\| (within 0.5) |
|---|---|---|---|---|---|---|
| 2k | 421 | 1 | 21 | 2 | 71% | 1.27 (22%) |
| 4k | 478 | 2 | 36 | 2 | 82% | 1.19 (27%) |
| 6k | **569 (47.0%)** | **31** | **63** | **1** | 85% | **1.02 (34%)** |

  - Row 14 at 6k: phase 0.946 in h63; 711 hits, 69 fresh; 149 false (126 in cooldown: the inherited row-14 cue).
  - GE at row 9, for contrast: 30 hits, 0 false, all under-drawn large drops, despite its generator beating copy on 79%.
  - D is the first generic objective (a loss-shape change, no labels, no new head) to draw zombie damage at the unbiased row.
    It passes the predeclared health reading by a wide margin, with every reading still rising at 6k.
  - Pending before any "fix" claim:
    - cost (teval vs C0, both currencies);
    - B1 ingredient intervals;
    - the simulator-truth damage panel;
    - **the continuation question applied to D itself:** D from scratch, the predeclared second from-scratch run.

**Sufficiency test result** (02:34; lane125; oracle B1 + generator loss B2 from M16, 2,000 updates, λ auto):

| world, row 9 | hits / 1,211 | fresh / 207 | false | generator beats copy | drawn \|err\| |
|---|---|---|---|---|---|
| oracle (B1 only) | 70 | 2 | 3 | 3% | 3.60 |
| G (B2 only, 6k) | 64 | 0 | 0 | 75% | 3.22 |
| **oracle + generator loss (2k)** | **114** | 22 | 6 | 59% | 2.50 |
| D (6k, no labels) | 569 | 31 | 1 | 85% | 1.02 |

- Row 14: phase 1.000; 346 hits, 89 fresh, 45 false.
- **Predeclared reading ("draws most held hits at row 9"): NOT met (9.4%).** Qualified honestly: B2 was only partly satisfied in
  2,000 updates (59% vs G's 80% at 6k), so this is a partially dosed test, not a clean negative on joint sufficiency.
- What it does establish: under the L1 objective, an h63 that carries both ingredients plus a half-trained generator draws 114.
  D, with no labels, draws 569. D's success is therefore not "B1 + B2 under L1".
  - The loss shape is itself a factor. Under L2 the head's optimum moves from the conditional median (draw only if
    P(hit) > 0.5, at full size) toward the conditional mean.
  - The worst-element term sends gradient to exactly the rare large errors.
- Next, deciding: drawable vs emitted at row 9 per world (`evidence_e21v2_row9.json`, `evidence_sufficiency.json`). If D emits
  close to what its h63 supports, D fixed expression; if its drawable also rose, it fixed information too.

**E21 world cost, teval** (02:40; `compare.json`; window 15; paired walk-seed bootstrap; MSE / copy and imagined error / V):

| vs C0 | one-step, all | moved | idle | depth 1 | depth 4 | depth 16 |
|---|---|---|---|---|---|---|
| M16 → C0 (control drift) | −0.000 | −0.001 | +0.030 * | −0.000 | −0.002 | −0.003 |
| G | −0.000 | −0.000 | −0.029 * | +0.000 | +0.003 * | **+0.045 [+0.033, +0.058]** |
| GE | −0.001 | +0.000 | −0.032 * | +0.000 | +0.002 | **+0.033 [+0.021, +0.046]** |
| E′ | −0.001 * | −0.001 | −0.013 * | −0.000 | −0.001 | +0.004 |
| **D** | **−0.012 [−0.014, −0.009]** | −0.003 * | −0.042 * | −0.003 * | −0.016 * | **−0.052 [−0.076, −0.028]** |

- The generator loss costs imagination (depth 16 +0.03 to +0.05). The fitted event head is neutral.
- D *improves* one-step and depth-16 error in teval's currency. As predeclared, teval scores squared error, which an
  L2-trained world may be favoured on. D's L1 cost on the same held roots and teacher-forced path (`health_l1cost.py`,
  lane132) is required before calling it costless.

**E21 world cost in L1 and in decision-relevant facts** (02:50; `l1cost.json`, teval JSONs):
- L1 on teval's held roots, teacher-forced, window 15 (mean |err| per element; C0 0.0328):

| vs C0 | all tokens | map tokens |
|---|---|---|
| G / GE | +0.0025 / +0.0025 (+7.7%) | +0.0023 / +0.0022 |
| E′ | −0.00003 (unresolved) | −0.00001 (unresolved) |
| **D** | **+0.0067 [+0.0064, +0.0071] (+20%)** | **+0.0077 [+0.0073, +0.0081]** |

- The L1 / L2 trade, measured: D has fewer large errors (better squared error) and larger small errors everywhere (worse L1).
- What can be *read* from the frames (teval's true-fit probes):

| world | one-step moved: tile near / zombie-near AUC / health R² / food R² | imagined d4: tile near / zombie-near / food R² | imagined d16: tile near / zombie-near / health R² / food R² / facing |
|---|---|---|---|
| C0 | 0.993 / 0.779 / 0.980 / 0.998 | 0.957 / 0.829 / 0.951 | 0.822 / 0.532 / 0.582 / 0.880 / 0.915 |
| G | 0.994 / 0.780 / 0.978 / 0.997 | 0.954 / 0.804 / 0.947 | 0.812 / 0.550 / 0.619 / 0.859 / 0.915 |
| E′ | 0.992 / 0.774 / 0.980 / 0.998 | 0.959 / 0.810 / 0.950 | 0.823 / 0.528 / 0.593 / 0.877 / 0.907 |
| **D** | 0.993 / **0.961** / 0.984 / 0.987 | **0.973 / 0.955** / 0.929 | 0.829 / **0.609** / 0.582 / **0.582** / 0.893 |

- D's extra L1 error leaves terrain intact (tile accuracy equal or better at every depth). It depicts zombies far better: mobs are
  rare large changes, the same class as health, and the L1 median was blurring them away too.
- **D's real cost: the food token degrades in long imagination** (food R² at depth 16 0.880 → 0.582; depth 4 0.951 → 0.929), and
  facing slips (0.915 → 0.893).
- D is therefore not adoptable as-is. The food cost has to be diagnosed exactly (the L2 term? the worst-element term? food's slow
  hunger dynamics under a mean-seeking loss?) before any decision; it is the next item after the from-scratch runs.

**E21 row-9 evidence, all arms** (02:53; `evidence_e21v2_row9.json`; same probes / calibration / held episodes):

| world, row 9 | since = 6 / since 1..4 in h63 | approach adjacency | drawable from h63 (fresh) | emitted (fresh / false) | generator beats copy |
|---|---|---|---|---|---|
| M16 | 0.889 / 0.914 | 0.817 | 334 (4) | 60 (0 / 0) | 3.6% |
| C0 | 0.896 / 0.900 | 0.767 | 335 (3) | 64 (0 / 0) | 5.2% |
| G | 0.937 / 0.924 | 0.648 | 136 (0) | 64 (0 / 0) | 74.7% |
| GE | 0.898 / 0.906 | 0.710 | 80 (1) | 30 (0 / 0) | 78.6% |
| E′ | 0.911 / 0.916 | 0.821 | 367 (13) | 79 (1 / 0) | 6.1% |
| oracle (labels; diagnostic) | 1.000 / 1.000 | 0.997 | 848 (184) | 70 (2 / 3) | 3.1% |
| **D** | **0.952 / 0.969** | **0.964** | **790 (122)** | **569 (31 / 1)** | **85.4%** |

**E21 D, predeclared readings:**

| reading | result |
|---|---|
| (1) B2 fixed, ≥ 30% | **TRUE** (85%) |
| (2) B1: since = 6 ≥ 0.95 and drawable ≥ 2× C0 | **TRUE at row 9** (0.952; 790 = 2.4× C0's 335); at row 14 since = 6 is 0.946, just under |
| (3) health drawn at window 15 (row 14), as predeclared | **FAILS**: 711 hits but 149 false (8.6% > 2%), the inherited row-14 death cue |
| (3) the same reading at row 9 | **PASSES** (569 = 47%; 1 false) |
| (4) cost | teval MSE: one-step −0.012, depth 16 −0.052 (better). L1 +20%. Imagined food R² at depth 16 0.880 → 0.582 (worse). |

- **The exact answer the diagnosis asked for:** a generic, label-free change (Delta-IRIS's loss shape) fixes both blockers at once.
  - Information (B1): it brings the phase and approach adjacency close to the oracle's level.
  - Expression (B2): its generator emits 72% of what h63 supports.
  - Mechanism, measured before the run: 11× the hit gradient share, plus the mean-seeking decision rule.
- Open before any adoption:
  - food cost (diagnosis running, lane133);
  - row-14 cue (inherited from the sampler, removable as E20 showed);
  - seed 8;
  - the continuation question for D itself (lane131).

**D's food cost, diagnosed exactly** (03:00; `health_food.py`, `food_e21.json`; teval's held roots; window 15; HUD reader ×9):

| world | food: teacher-forced change on steps where true food is unchanged | on true decrements (true −0.845) | self-fed food bias, depth 1 / 4 / 8 / 16 |
|---|---|---|---|
| C0 | −0.003 / step | −0.005 (copies, never draws them) | +0.005 / +0.14 / +0.20 / +0.40 (missed decrements) |
| G | +0.002 / step | +0.001 | +0.01 / +0.17 / +0.25 / +0.47 |
| **D** | **−0.100 / step** | −0.046 | **−0.10 / −0.24 / −0.51 / −0.88** (compounding drift) |

- D predicts a fractional decrement at every step and self-feeding compounds it. The same per-step drift appears on HUD columns 2
  and 3 (−0.079, −0.062) and on health's unchanged steps (−0.087).
- Health *hits* are drawn at the right size (predicted −1.266 vs true −1.246 on changes), because D installed the phase that
  makes them predictable.
- Source, verified (Craftax-Classic `game_logic.py` L1256-1290): food −1 when hunger > 25 (hunger +1 per step, +0.5 asleep);
  drink −1 when thirst > 20; energy −1 when fatigue > 30.
  - Periods 21-31 exceed the 15 transitions a 16-frame window shows, so the timer phase is often invisible.
  - From the world's view the decrement's timing is then genuinely uncertain. A mean-seeking loss draws the expectation as a
    fraction every step. (The measured −0.10 / step is larger than a uniform-ignorance mean of ≈ 1/26; recorded, not explained.)
- **Exact link to the literature and to E16:** Delta-IRIS pairs its L2 tokenizer loss with a *stochastic* Δ channel sampled from
  a learned prior, so uncertain changes are sampled at imagination time, not averaged.
  - E16 had the Δ channel with our L1 loss: it failed on allocation (the posterior saw the answer, hits AUC 0.63-0.70).
  - D has the loss without the channel: it fixes allocation and pays in averaging drift on unobservable timers.
  - **Faithful Delta-IRIS (both) is the literature-consistent next arm.** E16's channel code (`dworld.py`) exists; it needs
    D's loss. Predeclaration follows after the from-scratch runs report.

**Simulator-truth damage panel** (03:08; `check_damage.py --window 15`, held diagnosis futures, 0.5 cut; `e21v2_damage_w15.log`):

| world | teacher caught / 337 | false rate | fresh / 31 | beside, no hit in window: caught / drawn without hit | −2 class accuracy | self-fed caught (aligned) |
|---|---|---|---|---|---|---|
| M16 | 0.086 | 0.75% | 0 | 0.127 / 0.134 | 0.085 | 0 / 240 |
| C0 | 0.050 | 0.47% | 0 | 0.077 / 0.103 | 0.052 | 0 / 243 |
| G | 0.027 | 0.31% | 0 | 0.041 / 0.076 | 0.029 | 0 / 242 |
| GE | 0.018 | 0.10% | 0 | 0.027 / 0.004 | 0.013 | 0 / 239 |
| E′ | 0.033 | 0.49% | 0 | 0.045 / 0.080 | 0.033 | 0 / 239 |
| **D** | **0.733** | 4.06% | **0.613** | 0.909 / 0.536 | **0.475** | **0.060 (15 / 250)** |

Bayes hit rates per stratum: fresh 0.973, beside without a hit 0.480, hit in window 0.028.
- Teacher-forced on held roots, D catches 73% of real damage, including 19 of 31 fresh approach hits.
  - In the genuine coin-flip stratum (zombie beside, no recent hit; Bayes 0.48) it draws drops on 54% of no-hit cases: the
    expected behaviour of a mean-seeking world on an event it cannot time.
  - This panel's window spans rows up to 14, so part of the 4% false comes from the inherited row-14 cue.
- **In its own imagination, D draws 6% of aligned hits (15 / 250); every other world draws 0.** For H16 decisions this is the
  reading that matters. Health in self-fed rollouts remains mostly missing.
  - The same mechanism as the food drift is the leading candidate: a deterministic mean-seeking world averages what it cannot
    time, and imagination compounds it.
  - Not yet diagnosed: whether imagined hits fail because the imagined zombie / phase state diverges, or because the head
    averages them away.

**D's imagination gap, localized by substitution** (03:23; `health_imagination.py`, `imagination_*.json`; check_damage's data,
reader, masks and alignment rule):

| D: frame fed back at each step | caught, all 337 damage steps (aligned) | zombie drawn beside the player on damage steps | caught where a zombie is beside | false |
|---|---|---|---|---|
| imagined (= self-fed) | 4.5% (6.0%) | **8.9%** | 30% | 0.43% |
| imagined + true HUD (health / phase history) | 5.9% (7.5%) | 8.9% | 30% | 1.43% |
| **imagined + true map** | **58.5%** | **73.6%** | 69% | 2.07% |
| true (= teacher-forced) | 73.3% | 73.6% | 87% | 4.06% |

C0 for reference: 0 / 0 / 3.0% / 5.0% caught (zombie beside 10.4% imagined, 73.6% true).
- The HUD / phase history is **not** what fails in imagination: restoring it moves 4.5% → 5.9%.
- The **map** is: restoring it moves 4.5% → 58.5%. At the moments damage really happens, D's imagination shows a zombie beside the
  player 8.9% of the time, against 73.6% in truth.
- Source, verified (Craftax-Classic `game_logic.py` `update_mobs`, L758-845):
  - a zombie within Manhattan distance 10 moves toward the player with probability 0.75 (axis chosen at random when tied),
    otherwise takes a uniformly random step;
  - at distance 1 with cooldown ≤ 0 it attacks (2 damage, 7 asleep), stays put, and resets its cooldown to 5. That is the 6-step
    period measured in the data (since = 6 → 0.87).
- **Exact chain in imagination:** hits require adjacency → adjacency depends on random zombie steps → a deterministic world draws
  the average of the possible positions → the zombie smears out of the imagined frame → no hit can be drawn.
  - Same root as the food drift (a mean-seeking world on events it cannot time), now in the map.
  - It matches the 10-03 finding that H16 decisions need sampled futures (one faithful future ≈ 57% of the value, 16 samples
    ≈ 90%).
- **Consequence for the fix:** D's loss fixes allocation (the representation and the head learn the mechanism). Imagination
  additionally needs *sampling* of stochastic events. This is the role of Delta-IRIS's stochastic Δ channel, now motivated twice
  (food timers, zombie moves).

**What happens to zombies, measured directly** (09:00; `health_zombie.py`, `zombie_{D,C0,M16}.json`; teval's held roots; true
zombie cells from the simulator's visible state; zombie scores = teval's probe on the predicted frames):
- True moves, steps with one zombie at relative distance 1..6 (n = 2,529): toward the player 37.9%, stay / attack 51.0%,
  other (random) 11.1%.
- One step from true history:

| true move | D: score at true cell / argmax correct | C0 | M16 | probe on the true frame |
|---|---|---|---|---|
| toward (959) | 0.309 / **57.8%** | 0.176 / 30.1% | 0.174 / 30.4% | 0.681 |
| stay (1,290) | 0.484 / 91.1% | 0.581 / 87.9% | 0.570 / 87.1% | 0.653 |
| other (280) | 0.032 / 1.1% | 0.003 / 0.0% | 0.003 / 0.0% | 0.625 |

  - D learned the rule's deterministic part much better than the L1 worlds (argmax on toward moves 58% vs 30%).
  - Random moves are unpredictable for every world, as they must be.
  - Even on toward moves D's one-step zombie is half-intensity (0.31 vs 0.68): a blend of outcomes.
- Imagination, the 130 roots with exactly one zombie (peak score / positive mass / share with any cell > 0.3):

| depth | D | C0 | M16 | truth |
|---|---|---|---|---|
| 1 | 0.44 / 1.27 / 73% | 0.45 / 1.06 / 63% | 0.44 / 1.04 / 62% | 0.65 / 1.34 / 92% |
| 2 | 0.33 / 1.27 / 55% | 0.40 / 1.01 / 55% | 0.38 / 0.97 / 52% | 0.64 / 1.34 / 89% |
| 4 | 0.21 / 1.19 / 20% | 0.30 / 0.89 / 36% | 0.27 / 0.83 / 31% | 0.64 / 1.34 / 92% |
| 8 | 0.13 / 1.12 / **0%** | 0.19 / 0.75 / 20% | 0.17 / 0.68 / 17% | 0.64 / 1.37 / 93% |
| 16 | 0.11 / 1.16 / **0%** | 0.13 / 0.70 / 8% | 0.12 / 0.64 / 8% | 0.65 / 1.44 / 96% |

  - **Two measured ways to lose a zombie.** D (L2-type) *diffuses* it: total mass is conserved (≈ 1.2) while the peak decays
    geometrically, the zombie spread over many cells at low intensity. By depth 8 no cell anywhere reads as a zombie.
  - The L1 worlds *fade* it: peak and mass both decay (mass 1.06 → 0.70).
  - Either way, by depth 4-8 the imagined world holds no localized zombie, so no zombie can arrive beside the player, and no
    hit can be drawn. That is the 8.9% vs 73.6% measured at damage steps.
- **Calibration test** (lane139, 09:05; distance 2..6 with no attack possible, n = 1,211; positive zombie-score share over the
  5 candidate cells):

| next cell | true frequency | D's predicted share | C0's predicted share |
|---|---|---|---|
| toward the player | 62.4% | 47.9% | 38.1% |
| stay | 25.6% | 35.8% | 36.1% |
| other direction | 12.0% | 16.4% | 24.0% |
| zombie mass on the 5 cells (true frame 0.695) | | 0.553 | 0.319 |

  - D is not mode-seeking: a mode-seeking model would put nearly everything on "toward". It spreads its one-step prediction
    across the possible outcomes in roughly the true proportions, with a bias toward staying. That is the expected
    distribution of the next frame, which is what a deterministic mean-seeking model predicts.
  - C0 is flatter and loses mass (fading).
- **The imagination failure, exact:**
  1. zombie moves are stochastic (source);
  2. the one-step prediction is a blend in proportion to the true move probabilities (calibration);
  3. imagination feeds blends back, so the zombie diffuses (D: mass conserved, peak 0.44 → 0.11) or fades (L1 worlds);
  4. no localized zombie is beside the player at real damage steps (8.9% vs 73.6%); restoring the true map restores 58.5%;
  5. hits are not drawn (6%).

- **Localization pinned by a controlled substitution** (09:17; lane140; D):

| D: frame fed back at each step | caught, all 337 damage steps | zombie beside the player on damage steps | false |
|---|---|---|---|
| imagined | 4.5% | 8.9% | 0.43% |
| + true HUD | 5.9% | 8.9% | 1.43% |
| **+ true tokens on cells within 2 of a true zombie** | **56.4%** | **73.6%** | 1.93% |
| **+ true tokens on an equal number of random non-zombie cells (control)** | **4.5%** | 8.9% | 0.41% |
| + true map (all 63 cells) | 58.5% | 73.6% | 2.07% |
| all true (teacher-forced) | 73.3% | 73.6% | 4.06% |

  - Zombie localization alone accounts for 96% of the full map-substitution effect; the size-matched random control has exactly
    zero effect.
  - **The imagination failure is entirely the lost zombie position.** Every link of the chain is measured.

**Literature step on this exact problem** (09:10; full texts read):
- *PlaNet* (Hafner et al. 2019, arXiv 1811.04551, Fig. 2): "Transitions in a recurrent neural network are purely deterministic.
  This prevents the model from capturing multiple futures and makes it easy for the planner to exploit inaccuracies."
  - The RSSM splits the state into stochastic and deterministic parts "allowing the model to robustly learn to predict multiple
    futures".
  - Fig. 4: the purely deterministic GRU model underperforms the RSSM.
- *DreamerV2* (arXiv 2010.02193, ablation §): categorical latents beat Gaussian latents on 42 / 55 Atari tasks. First hypothesis:
  a Gaussian prior "cannot match a mixture of Gaussian posteriors, which could make it difficult to predict multi-modal changes
  between one image and the next".
- Cited video prediction (Babaeizadeh et al. 2017 SV2P; Denton & Fergus 2018 SVG): stochasticity "to capture multiple futures".
- *Dreamer 4* (read 03:35) and *Delta-IRIS* (read 10-03) both sample: flow / shortcut forcing in continuous token space; discrete
  stochastic Δ tokens with an autoregressive prior.
- **Our per-tile world is a purely deterministic regressor.** Tonight's measurements are the failure these papers describe,
  quantified on our data.
- The remedy family is unambiguous: a stochastic transition that *samples* the next frame. Combined with the allocation finding
  (D), the requirement for the fix is:
  1. a loss that gives rare large changes their gradient (D's L2-type allocation; Dreamer 4's x-space squared loss);
  2. a stochastic component sampled at imagination time.

### E22 — predeclared (committed 09:18:46, 2026-10-09; labelled "09:35" in error), before any code: faithful Delta-IRIS on the canonical world (D's loss + stochastic Δ)

**Why this design** (diagnosis + literature + our own measured constraints):
- Diagnosis: D fixes allocation (B1, B2, router: 7.56% gradient share; row-9 hits 47%). Its imagination fails because a
  deterministic transition blends the stochastic zombie move (blend shares ≈ the true move probabilities) and the blends diffuse
  the zombie. Restoring only the zombie neighbourhoods restores 56.4% of damage.
- Literature: stochastic transitions that *sample* (PlaNet RSSM, DreamerV2 categoricals, Delta-IRIS Δ tokens, Dreamer 4 flow).
- Own constraints:
  - per-token independent sampling destroys known content (E14, 10-02), so the sample must be a few coherent codes per frame,
    not per-tile noise;
  - Delta-IRIS's Δ tokens are exactly that (4 discrete codes per transition, one per 5×5 region), and `dworld.py` implements
    them (E16, validated: zero-init conditioning reproduces the deterministic world exactly; revival "always" needed, others
    collapsed to 3-11 codes);
  - E16 failed for allocation (our L1). D supplies the loss.

**Design:**
- *Stage A* (`dworld.py` ported to the canonical recipe; every other part as E16):
  - decoder = D (the E21 D world, continued); posterior E on (frame t, action, frame t+1); 4 codes per transition;
    revival always; D's loss + commitment 0.02;
  - L16 rawlong windows, batch order 11, 6,000 updates, M16 / D recipe otherwise.
- *Stage B:* E16's prior on frozen stage-A codes (3 layers, width 512, 21-step rawlong windows, 20,000 updates, Delta-IRIS
  crafter.yaml values).
- *Imagination:* at each step the prior samples the transition's 4 codes from the imagined history; the decoder conditions on
  them. Past transitions use the codes sampled (or, teacher-forced, the posterior's).

**Bowman risk check first** (E16 amendment 2: under L1 a continued deterministic world ignored Δ). A stage-A smoke of 2,000 updates
with batch 16 (fits beside the running from-scratch D) must show, before any full run:
- (P1) the Δ gain (teacher loss with Δ zeroed minus with Δ, same batch) > 0 and rising;
- (P2) with the posterior's Δ, which sees the true next frame, the one-step zombie on *random* moves becomes localized: score
  at the true cell ≥ 0.3 (D without Δ 0.032), argmax correct ≥ 50% (D 1.1%).

P2 tests the diagnosed mechanism directly: the channel must carry exactly the stochastic choice D cannot predict. If P1 or P2
fails, stage A is not run at full length; the failure is diagnosed first.

**Readings of the full E22** (s7; held roots and the health_chain subset; D as the comparator):
1. Imagination keeps zombies localized: with prior-sampled Δ, one-zombie roots keep a cell > 0.3 at depth 8 in ≥ 50% of
   roots (D 0%, truth 93%), and the peak at depth 8 is ≥ 0.4 (D 0.13).
2. Imagined damage: self-fed caught on aligned true-damage steps ≥ 30% (D 6%), false ≤ 2%, mean over 4 prior samples per root.
3. Teacher-forced health preserved: row 9 hits ≥ 40% of held, false ≤ 1% (D 47% / 0.06%).
4. Food: imagined food bias at depth 16 within ±0.4 (D −0.88, C0 +0.40).
5. Cost: teval one-step / depth 16 vs D, both currencies; reported, with no pass threshold beyond "no resolved loss on tile facts".

**Interpretation rules, fixed now:**
- 1 + 2 pass → the sampled channel fixes the diagnosed imagination failure; a from-scratch E22 then follows (the continuation
  question, as for D).
- 1 passes and 2 fails → zombies stay localized but hits are not drawn; diagnose the next link (adjacency at damage steps,
  phase in imagination).
- 1 fails with P2 passing → the prior does not sample the moves well; diagnose prior accuracy (code-prediction CE, calibration)
  before anything else.

### E22 pre-flight result (lane142, 09:20:41-09:55:16): P1 and P2 FAIL — the continued D decoder ignores Δ

- **Smoke:** 2,000 updates of D continued with E16's channel (revival always), batch 16 L16 windows, D's loss.
- **Logged gain** at 500 / 1,000 / 1,500 / 2,000: −0.00068 / −0.00161 / −0.00059 / −0.00062 (−1.6% / −2.9% / −1.2% / −1.3%).
  Codes per batch: 72 / 11 / 89 / 54.
  - This logged gain is biased against Δ: it compares the pre-step loss with Δ to the post-step loss zeroed, on the
    batch just trained on. E16 amendment 2's "0.4-1.5% higher" was logged the same way.
  - The logged codebook entropy (8.0-10.4 bits) is not a usage measure: revival resets expired codes' frequencies to
    1/1024 without renormalizing. Codes per batch is the usage measure.
- **Clean P1** (`health_zombie.py` `p1_clean_held`, same weights; held 1,002 roots, 16,028 alive one-step teacher-forced
  predictions):
  - Delta-IRIS loss 0.037205 with Δ vs 0.037215 zeroed: gain **+0.000009 (+0.03%)**.
  - MSE 0.012614 vs 0.012615; map MSE 0.014894 vs 0.014891.
  - **P1 FAILS.**
- **P2** (posterior Δ sees the true next frame):

  | true zombie move | n | E22A: score at true cell / argmax correct | D: score / argmax |
  |---|---|---|---|
  | other (random) | 280 | **0.036 / 1.1%** | 0.032 / 1.1% |
  | toward | 959 | 0.333 / 58.7% | 0.309 / 57.8% |
  | stay | 1,290 | 0.502 / 91.1% | 0.484 / 91.1% |

  - Thresholds were 0.3 and 50%. **P2 FAILS.**
  - Calibration is unchanged: toward share 0.483 (D 0.479; truth 0.624).
- **Why, measured:** the Δ projection `cond.weight` (zero-init) after 2,000 Adam updates has Frobenius 0.350, RMS 0.0027 per
  element, max 0.0129.
  - The lr warms up to 1e-4 by update 1,000. A sign-consistent gradient would move each weight up to ~0.15.
  - The largest weight moved 9% of that, the RMS 2%: a random walk. The decoder's gradient on the channel has no
    consistent direction.
- **Plumbing excluded** (CPU, reference backend, D in the L16 world, 2 rawlong windows):
  - zero-init Δ: loss identical; the gradient reaches `cond` (norm 0.0024);
  - with `cond` ~ N(0, 0.02²): Δ changes the loss (0.031585 → 0.031619) and the encoder gradient rises 800×
    (1.4e-5 → 0.011).
  - The channel is wired; the decoder does not use it.
- **Reading:** this is E16 amendment 2 (10-03; L1, 6 frames) reproduced under D's loss at L16. A converged deterministic
  decoder starts stage A in Bowman's equilibrium, and D's loss does not get it out. Per the rules, no full stage A.
- **Time labels:** the predeclaration's "09:35" and lane142's "09:55" are wrong. Actual times: predeclaration committed
  09:18:46 (81d80819), code 09:20:21, lane launched 09:20:41.

**Substitution test of the warm start — predeclared 09:58, before launch (lane143):**
- `e22s` smoke: the same command without `--init`, i.e. the decoder from scratch (TWorld at seed 7). Recipe, loss, windows,
  batch order and updates are identical.
- Readings, on both smoke files:
  - clean P1 and P2 (`health_zombie.py`);
  - channel diagnostics (`e22_channel.py`, held 16-frame windows = 4 context + 12 future frames, 62 batches of 16):
    1. consistency of the `cond` gradient, ‖Σ_b g_b‖ / Σ_b ‖g_b‖ (1 = consistent; ~1/√62 = 0.13 for noise);
    2. the encoder's gradient from the decoder vs from the commitment term;
    3. the share of the zeroed-Δ residual variance explained by each region's code, vs permutation chance (E16's
       measure).
- Rules:
  - scratch clean gain ≥ +5% and `cond` RMS ≥ 10× warm's (0.0027) → the warm start is the cause, as in E16 amendment 2.
    E22 then trains as Delta-IRIS does, posterior and decoder jointly from scratch, on the canonical two-stage recipe
    (6-frame 36k + L16 6k), matched to D from scratch (lane131).
  - scratch gain < 1% → the warm start is not the cause; the recipe (loss, L16, quantizer) is diagnosed before any run.
  - in between → report and diagnose.
- P2 on the 2k scratch decoder is reported but not a gate (the decoder is immature).

**Substitution result (lane143-146, 09:58-10:35): the warm start shuts the channel; the 09:58 rule came out "in between" on its proxy**
- Measured on held windows with the same weights (`e22_channel.py`: 1,002 held 16-frame windows, 126 batches; noise-level
  consistency 0.089; `health_zombie.py` one-step for P1 / P2):

| | warm u0 | scratch u0 | warm u2000 | scratch u2000 |
|---|---|---|---|---|
| clean gain (zeroed − with Δ), e22_channel | 0 | 0 | **−0.0000046 (−0.01%)** | **+0.0551 (+46.0%)** |
| clean gain, health_zombie one-step | — | — | +0.000009 (+0.03%) | +0.0558 (+45.9%) |
| Δ size: ‖Δ‖ / ‖tile embedding‖ | 0 | 0 | **0.59%** | **16.1%** |
| cond.weight RMS (zero-init) | 0 | 0 | 0.0027 | 0.0168 (6.1×) |
| decoder's cond gradient: consistency | **0.146** | **0.380** | 0.326 | 0.623 |
| encoder gradient from the decoder: norm / consistency | 0 (zero-init) | 0 | 2.1e-4 / 0.29 | 0.0116 / 0.70 |
| codes used per region (held) | 288-515 | 288-515 | **17-25** | 86-105 |
| residual variance explained by the region's code (chance) | 4.1-5.8% (1.9-3.4%) | 4.3-6.5% (same) | 0.3-0.7% (0.1-0.16%) | 3.7-5.4% (0.6-0.7%) |

- **Rule outcome, as written:** the scratch clean gain (+46%) passes ≥ +5%; the cond RMS ratio (6.1×) fails ≥ 10×. So the
  outcome is "in between → report and diagnose".
- **Diagnosis:** the RMS criterion was a proxy for the size of the channel. Δ's size against the tile embedding it is added
  to is 27× (0.59% vs 16.1%). That measure was chosen after the rule; it is reported as a diagnosis, not as a pass.
- **What the warm start does (exact):**
  - at update 0 the decoder's gradient on the channel is barely above noise (consistency 0.146 vs 0.089; scratch 0.380);
  - over 2,000 updates the warm encoder DISCARDS what its random initial codes carried: codes per region fall from
    288-515 to 17-25, and the residual explained above chance falls from ~2% to ~0.3%;
  - the scratch encoder keeps 86-105 informative codes, and its decoder takes 46% of its loss from Δ.
  - Changing only the decoder's initialization flips the channel from unused to used.
- **But the scratch channel at 2k carries deterministic content, not the zombie's choice:**
  - map MSE 0.067 zeroed → 0.027 with Δ;
  - P2 one-step: random moves score 0.009, argmax 0.4%; toward moves argmax 1.8%. The immature decoder predicts "stay"
    (82% share vs 26% true).
  - This is E16's from-scratch pattern (10-04: at 36k Δ removed 92.8% of map error and 79% of HUD error, yet health stayed
    0/305).
- **Correction of my attribution:** E16 amendment 2 and the E22 predeclaration called the warm failure Bowman's
  equilibrium. Bowman et al. 2016 and He et al. 2019 describe collapse driven by the KL term; stage A has no KL term.
  The measured failure is the one Park & Lee (below) describe for pretrained decoders.

**Literature on the diagnosed problem (read 10:00-10:35):**
- **SV2P** (Babaeizadeh et al., ICLR 2018, arXiv 1710.11252, sec. 3.2 and appendix Table 1, read):
  - "naive training usually results in the model ignoring the latent variables and converging to a suboptimal
    deterministic solution";
  - their fix is three phases: (1) generator only, z ~ N(0, I) stacked as a channel on the tiled action, 50k; (2) the
    inference network with KL 0, 50k, which reaches "very low reconstruction error"; (3) KL annealed to β 0.001, 100k.
  - A deterministic warm start can work there, but the read path was live (noise fed in phase 1), not zero.
- **Park & Lee 2021** ("Finetuning Pretrained Transformers into VAEs", arXiv 2108.02446, sec. 4-5.1 and appendix, read):
  - with a pretrained T5 decoder, "posterior collapse was observed in all training schemes without encoder warmup... The
    decoder learns to ignore the initially noisy input signal from the encoder";
  - phase 1 trains with KL 0, and "freezing the decoder and the memory projection layer Wproj while training with an AE
    objective is crucial in learning meaningful encoder outputs" (5 epochs); phase 2 trains jointly (3 epochs).
  - Wproj is a new layer that maps z into the decoder (cross-attention keys and values).
  - This is our case exactly: a converged decoder plus a new, noisy encoder.
- **He et al. 2019** (arXiv 1901.05534): inference lag plus the KL term; aggressive encoder updates. The mechanism goes through
  the KL term, so it is not ours.
- **Delta-IRIS:** the tokenizer is trained jointly from scratch (option A below).

**Decision and predeclaration (10:37, before any E22F code): E22F = Park & Lee phase 1 on D, smoke first**
- **Why E22F before the from-scratch E22 (A):**
  - with D frozen, the channel can only reduce D's own residual. D already predicts deterministic content, so its
    residual is the stochastic part: random zombie moves, hits, and blur. That is exactly what imagination lacks;
  - A's decoder offloads deterministic content into Δ (measured at 2k; E16 at 36k), which the prior must then predict.
  - This deviates from the 09:58 rule's action ("E22 then trains jointly from scratch"). The reason is the literature read
    after that rule plus the scratch P2. A is held, not dropped: it runs if E22F fails, and as E22F's from-scratch
    confirmation.
- **E22F design (Park & Lee phase 1, adapted):**
  - frozen: D (every world parameter) and the projection into D's tile space (`Posterior.cond`) at a new Linear's
    default initialization (`reset_parameters`; measured Δ/tiles at update 0 = 14.8%, the order of the working scratch
    channel's 16.1%);
  - trained: the posterior encoder (conv net, action plane, quantizer pre-projection), the code embedding (quantizer post
    projection) and the EMA codebook;
  - D's loss + commitment, revival always, L16 rawlong, batch order 11.
  - Smoke: 2,000 updates, batch 16 (fits beside D from scratch).
- **Gate (as E22's):**
  - P1: clean held gain > 0;
  - P2: random-move score at the true cell ≥ 0.3 (D 0.032), argmax ≥ 50% (D 1.1%).
  - Also reported: e22_channel diagnostics, Δ size, codes used, and map vs HUD gain.
- **Rules:**
  - P1 + P2 pass → E22F full: phase 1 at D's budget (6,000 updates, batch 40), stage B prior (20k), then E22 readings
    1-5 with a new evaluator that samples Δ from the prior (D as comparator). Phase 2 (joint) is not run unless the full
    phase-1 readings show the frozen decoder is the limit.
  - P1 passes, P2 fails → the channel carries something else; measure what before any run.
  - P1 fails → frozen-read-path warmup does not open the channel on D; run A (E22 from scratch, dworld `--backbone fmamba`
    stage 1 36k + `--long` stage 2 6k, matched to D from scratch).

**E22F smoke result (lane147-148, 10:39-11:14): P1 passes (+0.57%), P2 FAILS — the codes carry D's residual, frozen D cannot read it**
- **Logged gain** (now same-weights, before the step) at 500 / 1,000 / 1,500 / 2,000: −0.3% / −0.6% / +0.01% / +0.13%.
  Codes per batch 69 / 205 / 113 / 154.
- **Clean P1** (held one-step): Delta-IRIS loss 0.036737 → 0.036527, **+0.00021 (+0.57%)**; MSE 0.012865 → 0.012756.
- **P2:** random moves score 0.032 (D 0.032), argmax **1.8%** (D 1.1%); toward 59.0% (D 57.8%); stay 91.0% (D 91.1%).
  **FAILS.**
- **Channel** (e22_channel, held): Δ is 15.2% of the tile embedding.
  - Encoder gradient from the decoder: 0.013 (scratch 0.0116), consistency 0.32 (scratch 0.70).
  - Codes per region: 88-112.
  - D's residual explained by the region's code: **2.3-3.5% vs chance 0.6-0.7%**, near the scratch encoder's 3.7-5.4%
    and unlike warm's 0.3-0.7%.
  - The encoder became informative; the frozen read path does not use it.
- **What each channel carries** (`e22_channel` carries, 1,002 held windows, 15,030 transitions; squared error with the
  posterior's Δ vs zeroed):
  - **D's own error budget (zeroed):** map cells that change (|Δs| > 2.5) are 11% of tokens and hold **68%** of D's
    squared error (11.8 per token vs 0.95 unchanged); HUD 6%; zombie cells 1.9%.

  | error removed by Δ | E22F (frozen D) | warm | scratch (2k) |
  |---|---|---|---|
  | changed map cells | **0.18%** (13% of its gain) | 0 | 37% (27%) |
  | unchanged map cells | 3.3% (**87%** of its gain) | −0.06% | 80% (70%) |
  | true zombie cells | 0.07% | −0.02% | 15% |
  | player / HUD | −0.4% / 0.05% | −0.3% / 0.2% | 46% / 4.6% |

  - E22F's gain is background drift on unchanged cells: the frozen decoder, read through a frozen random projection,
    cannot turn the codes' change information into changed outputs.
  - The scratch decoder at 2k needs Δ even to copy: its unchanged-cell error is 8.25 zeroed (D 0.95). The prior would
    then have to predict everything, which was E16's pathology.
- **Reading:** the three smokes separate the two halves of the channel.
  - warm: the encoder is not informative and the decoder does not read;
  - E22F: the encoder is informative (4× chance) and the frozen decoder cannot read;
  - scratch: both are active, but on deterministic content.
  - This is the situation both sources address with a second, joint phase: SV2P phase 2 (inference network + generator,
    KL 0) and Park & Lee phase 2 (full fine-tuning after the warmup).
  - The E22F rule allowed phase 2 "if the frozen decoder is the limit"; the measurements above show that it is.

**E22FJ predeclared (11:14, before launch): Park & Lee / SV2P phase 2 smoke**
- `dworld --long --init <E22F smoke file>`: world (D) and the whole posterior continue, nothing frozen (cond starts at
  E22F's default init). D's loss, revival always, batch 16, 2,000 updates.
- Same gate: P1 (clean held gain > 0) and P2 (random-move score ≥ 0.3, argmax ≥ 50%). Plus carries and channel
  diagnostics.
- Specific predictions, read in this order:
  1. the warm equilibrium does not re-form (codes per region stay ≥ 50; gain does not return to ≤ 0);
  2. the changed-cell reduction rises above E22F's 0.18%;
  3. P2.
- Rules:
  - P2 passes → E22F/FJ full (phase 1 6k + phase 2 6k at batch 40), prior 20k, readings 1-5 with `e22_imagine.py`.
  - 1 holds and P2 fails → measure the zombie-cell reduction and per-move-type P2 before anything else.
  - 1 fails (channel shuts again) → A (from scratch) is the remaining faithful route.

**E22FJ smoke result (lane149, 11:15-11:48): the joint phase opens the channel (+35%); P2 still FAILS (random moves 1.1%)**
- **Logged gain** (same weights) at 500 / 1,000 / 1,500 / 2,000: +5.0% / +25.8% / +24.3% / +18.5%; codes per batch 272 / 242 /
  268 / 305.
  - The batch order restarts, so these are E22F's batches. On them D's own loss (E22F zeroed = D) is 0.04210 / 0.05462 /
    0.04863 / 0.04780.
  - E22FJ with the posterior's Δ: 0.04065 / 0.04532 / 0.04083 / 0.04125, i.e. 3-17% below D.
  - E22FJ zeroed: 0.04278 / 0.06107 / 0.05393 / 0.05063, i.e. 2-12% above D. The decoder now leans on Δ for part of what it
    predicted alone.
- **Prediction 1 holds:** the warm equilibrium did not re-form.
- **Clean P1** (held one-step): Delta-IRIS loss 0.043075 zeroed → 0.028168 with Δ (+34.6%).
  - MSE 0.013415 → 0.009257, against D's 0.012865: −28% vs D with the posterior.
  - Worst element 2.57 → 1.53.
- **P2:** random moves score 0.041, argmax **1.1%** (D 1.1%); toward 0.340 / 61.1% (D 0.309 / 57.8%); stay 0.484 / 91.1%.
  **FAILS.**
- **Carries** (held, with Δ vs zeroed):
  - changed map cells: **36.2%** removed (13.66 → 8.72; 84% of the gain); unchanged 30.6%;
  - **zombie cells 8.4%** (17.79 → 16.30; 0.46% of the gain); HUD −2.5% (worse);
  - codes per region 297-364, explaining 7.3-9.3% of the residual (chance 2.0-2.4%).
- **Reading:**
  - the channel is open and spent on changing cells, but almost none of it on zombie moves;
  - zombie cells hold ~2.4% of the changed-cell error;
  - per the rule (prediction 1 holds and P2 fails), I measure where the capacity goes and P2 by scroll before anything
    else (lane150).
- **D from scratch at 12k vs M6 at 12k** (lane141, window 5):
  - phase 0.977 vs 0.897: **+0.077** [+0.061, +0.093];
  - adjacency 0.761 vs 0.640: **+0.118** [+0.059, +0.178]; both resolved;
  - emitted hits 311 (fresh 3), false 181.
  - Unlike GE from scratch, D from scratch is ahead of its matched deterministic run on both blockers.

**Where the E22 channels fail, localized (lane150-151 + CPU probe, 11:50-12:09): the posterior ENCODER never encodes the zombie's move**
- **D's error budget** (held, 15,030 transitions, teacher-forced, squared error):
  - scroll-entering edge cells: 4.8% of map tokens, 32.3 per token, **49% of D's error**;
  - changes on no-scroll steps: 25%; other changes on scroll steps: 10%; unchanged cells 25%;
  - zombie cells: **1.9%**.
- **E22FJ's channel** removes 61% of entering-edge error (**92% of its gain**), 7.6% of no-scroll changes, 8.6% of
  zombie cells on no-scroll steps.
- **P2 split by scroll:** random moves, argmax on no-scroll steps (n = 185): E22FJ **0.0%**, D 0.0%; on scroll steps
  (n = 95): 3.2% both.
  - Without scroll there is no edge to compete for the region's code, and still nothing.
  - Capacity competition within a transition is not the cause.
- **Encoder probe** (`e22_zombiecode.py`, CPU):
  - cases: no-scroll one-zombie steps (stay 1,088, toward 393, other 173);
  - probe: grouped-CV logistic regression; the move direction is read from the zombie region's pre-VQ z, its codebook
    vector q, or all four regions' z;
  - accuracy on moving cases only (4 classes):

  | features | toward | random ("other") |
  |---|---|---|
  | **positive control:** true change norm at the 5 candidate cells | 0.916 | **0.890** |
  | relative position (the toward rule's inputs) | 0.814 | 0.0 |
  | E22FJ z / q / all regions | 0.354 / 0.351 / 0.260 | **0.225 / 0.254 / 0.306** |
  | E22F z / q / all regions | 0.308 / 0.316 / 0.229 | 0.237 / 0.225 / 0.260 |
  | scratch z / all regions | 0.351 / 0.328 | 0.133 / 0.225 |
  | warm z / all regions | 0.354 / 0.247 | 0.121 / 0.249 |

  - With stay included (66% of cases) every posterior probe falls back on "stay" (random-move accuracy 0-0.08). That is
    why the moving-only probe and the positive control were added before reading this.
- **Exact statement:**
  - the move is linearly readable from the posterior's own inputs (0.89), yet no stage-A posterior encodes it, before or
    after quantization: it sits at chance, warm, frozen, joint or from scratch;
  - the failure is the encoder, not VQ or the decoder's rendering;
  - the encoder learns what reduces the decoder's loss most: entering terrain is 49% of D's error, while a perfectly
    encoded zombie move could remove at most 1.9%;
  - this is the allocation root of the whole health diagnosis (rare events get a tiny share of the objective), now one
    level down, in the stochastic channel.
- **Open, not measured yet:** whether a longer run would reach zombie moves once the edge gain saturates (the E22FJ
  smoke is 2k updates).

**Training length does not reach the move:** E16's from-scratch posterior (L1, 6-frame transformer world, revival crafter),
probed the same way at its 6k-36k snapshots:

| moving-only probe | 6k | 12k | 18k | 24k | 30k | 36k |
|---|---|---|---|---|---|---|
| random moves, z | 0.214 | 0.237 | 0.260 | 0.277 | 0.237 | 0.301 |
| random moves, q | 0.260 | 0.272 | 0.306 | 0.289 | 0.220 | 0.237 |
| toward moves, z | 0.514 | 0.557 | 0.519 | 0.511 | 0.560 | 0.539 |
| codes in the zombie's region | 111 | 177 | 203 | 223 | 240 | 271 |

- E16's warm 6k posterior: random moves z 0.150, q 0.133.
- In 36k updates of joint training, codebook use grows steadily and the random move stays at chance.

**Literature on the localized problem (read 12:13):**
- OC-STORM (arXiv 2501.16443): pixel MBRL "auto-encoding with an L2 loss ... is dominated by large areas and often fails to
  capture decision-relevant details"; their fix is object-centric (segmentation from few annotations).
- MuDreamer / Make-the-Pertinent-Salient: "object vanishing" of small task objects in Dreamer.
- DIAMOND (arXiv 2405.12399): "compression into a compact discrete representation may ignore visual details that are
  important"; a diffusion world model samples the whole frame instead.
- EAWM (arXiv 2601.19336): event-weighted observation loss.
- Our measured version: the stochastic channel's encoder allocates by loss share (edges 49%, zombie cells 1.9%).

**E22Z predeclared (12:13, before code): is loss share the CAUSE of the encoder ignoring the move? (diagnostic, label-derived, not a fix)**
- **Recipe:** the E22FJ smoke exactly (joint phase from the E22F file, the same 2,000 batches) + λ × the mean L1 over cells
  where teval's zombie probe on the TRUE frame t or t+1 exceeds 0.3 (the move's old and new cells; tworld.rollout_losses'
  mask path).
  - The probe's affine map is exported exactly (zombie_probe_export.py, max deviation 3.4e-7).
  - λ is auto: zombie term = teacher on the first batch, a share comparable to the edges' 49% (as the oracle test, 10-08).
- **Why it is causal:** the decoder alone cannot predict a random move, so the only way to lower the zombie term on random
  moves is to encode the move in Δ.
- **Readings vs E22FJ:**
  - P2 (all and no-scroll);
  - the encoder probe on moving cases (z and q, random moves);
  - carries.
- **Rules:**
  - random-move probe ≥ 0.6 or P2 argmax ≥ 50% → loss share is the cause: the channel carries the move when its share is
    comparable. The fix is then a label-free allocation for the channel (EAWM-type event weighting, per-cell latents, or a
    generative head; chosen after reading).
  - probe at chance (≤ 0.35) and P2 < 10% → even weighted, this channel path does not carry the move; the generative route
    (DIAMOND / Dreamer 4 sampling) is next.
  - between → report and diagnose.

**E22Z result (lane152, 12:15-12:47): loss share IS the cause — weighted, the channel carries the random move (P2 1.1% → 82%)**
- λ auto = 0.204 (first batch: 132 zombie-marked cells of 15,120 map cells, 0.87%).
- **Logged gain** (weighted objective, same weights) at 500 / 1,000 / 1,500 / 2,000: +0.8% / +7.7% / +35.6% / +26.5%; codes
  per batch 189 / 354 / 280 / 315.
- **P2** (held one-step, posterior Δ; D in brackets):

  | true zombie move | n | score at true cell (probe ceiling) | argmax correct |
  |---|---|---|---|
  | **random (other)** | 280 | **0.599** (0.625) | **82.1%** (1.1%) |
  | other, no scroll / scroll | 185 / 95 | 0.603 / 0.593 | 83.8% / 78.9% (0.0% / 3.2%) |
  | toward | 959 | 0.684 (0.681) | 96.2% (57.8%) |
  | stay | 1,290 | 0.653 (0.653) | 95.7% (91.1%) |

  - Predicted mass over the 5 candidate cells is 0.898 vs 0.695 on true frames (some extra mass).
- **Clean P1:** +32.4% (0.041661 → 0.028144); zeroed MSE 0.012935 ≈ D's 0.012865, so the decoder alone is intact.
- **Encoder probe**, moving cases only, random moves vs E22FJ: z **0.486** vs 0.225; q 0.439 vs 0.254; toward z 0.631 vs
  0.354.
  - The linear one-region probe understates what P2 (the decision-level reading) shows.
- **Rule outcome (12:13):** P2 ≥ 50% → loss share is the cause. Given a share comparable to the scroll edges, a 2,000-update
  joint phase encodes the random move AND the decoder renders it at the probe ceiling. E16 never did in 36k unweighted.
- **Carries** (unweighted D loss; E22FJ in brackets):
  - zombie cells **69.9%** removed (8.4%); on no-scroll steps 74.5% (8.6%);
  - scroll-entering edges 52.2% (61%);
  - overall gain +34.5% (+35.8%); codes explain 6.6-10.4% of the residual.
  - The channel had the capacity for both; reweighting moved zombies from 8% to 70% at almost no cost elsewhere. The fix is
    allocation, not channel size.
- E22Z's mask is label-derived: a diagnosis, not the fix.
- **D from scratch at 18k vs M6 at 18k:** phase +0.080 [+0.064, +0.098]; adjacency +0.154 [+0.097, +0.213]; both resolved and
  growing (6k: +0.054 / −0.036; 12k: +0.077 / +0.118).

**E22Z sufficiency test predeclared (12:49, before launch): does SAMPLING the allocation-corrected channel fix imagined health end to end?**
- Stage B: E16's prior (dworld `--stage b`, 20k updates, Delta-IRIS crafter.yaml values) on the frozen E22Z posterior.
- Then `e22_imagine.py` (4 prior samples per root + the zeroed rollout), with E22's predeclared readings:
  1. at depth 8, ≥ 50% of one-zombie roots keep a cell > 0.3 and the peak is ≥ 0.4 (D 0% / 0.13; truth 93%);
  2. imagined damage: caught on aligned true-damage steps ≥ 30%, false ≤ 2% (D's self-fed 4.5-6%).
- Interpretation:
  - 1 + 2 pass → the mechanism (stochastic channel with the right allocation + prior sampling) fixes imagined health; what
    remains is making the allocation label-free;
  - 1 passes, 2 fails → zombies stay localized but hits are not drawn; diagnose phase / adjacency in imagination;
  - 1 fails → the prior does not sample the moves; diagnose prior code accuracy / calibration first.
- The zeroed rollout here is E22Z's own decoder (its decoder trained jointly), reported beside D's.

**Literature for a label-free allocation (read 12:50-12:51):**
- **DyMoDreamer** (arXiv 2509.24804, NeurIPS 2025, sec. 2.1-2.4, read): the closest published mechanism.
  - Backward frame differencing D = 1[‖o_t − o_t−k‖ > ε], ε = 0.001, k = 1, dilated by a convolution to the
    surroundings → masked observation o′ = M(o)·o.
  - A separate dynamic encoder q(d_t | h_t, o′_t) gives 32 × 32 categorical modulators; the decoder reconstructs o_t from
    (h_t, z_t, d_t).
  - +9.5% on Crafter at 1M. Label-free, and it forces a latent's capacity onto changing content via its input.
  - On our data plain differencing marks scroll steps wholesale (entering terrain is the channel's dominant content).
- **EAWM AttentionMSE:** with the measured composition, upweighting all changed elements leaves zombie cells ≈ 4% of the
  event term on no-scroll steps, i.e. ≈ 2% of the objective, no better than now. It is not sufficient here (computed, not run).
- **Dreamer 4** (local PDF, sec. 2-3.2, read): a generative world with no posterior.
  - Shortcut forcing, flow matching + bootstrap, x-prediction, ramp weight 0.9τ + 0.1, K = 4 sampling steps, τ_ctx 0.1.
  - Sampling from the learned conditional instead of an encoder discovering the move. It requires restructuring TWorld
    (each frame predicts its own clean version from its noisy version plus history).
- **Candidate generic version of E22Z (to predeclare after the sufficiency test): moved content.**
  - After scroll compensation, a cell whose new token matches a neighbour's old token, or whose old token reappears in a
    neighbour: mobs translate.
  - Entering terrain has no source and mined tiles change type, so both are excluded.
  - Thresholds from the data's own structure; labels only to validate precision / recall.

**Label-free moved-content mask, feasibility (lane154, `e22_motionmask.py`, 12:53-12:57; labels validate only)**
- Held transitions: 1,009,764 alive map cells, 121,541 changed (scroll-compensated change > EVENT_TAU).
- **The neighbour-match ratio on changed cells is bimodal:**
  - a moved mode at 0.00-0.15 (≈ 20k cells; most changed zombie cells: 1,764 of the zombie changes);
  - a broad mode at 0.6-1.0 (≈ 60k);
  - valley at 0.25-0.35 (≈ 1,000 per bin). So **ρ = 0.3 is set by the data, not by labels**.
- **At ρ = 0.3:**
  - marks 2.4% of map cells; zombie recall 36% of all zombie-occupied cells (staying zombies do not change their cell);
  - 37% of marked cells hold a mob at t or t+1 (mob channel 0 = zombie 9.3%, channel 1 19.6%, channel 2 4.9%, others
    < 1%);
  - holds 9.5% of D's map squared error (zombie cells alone: 3.3%).
- At ρ = 0.5 / 0.8: marked 2.9% / 6.5%, zombie recall 42% / 49%, mob share 34% / 17%.
- **Contaminant:** the player's own centre cell (token 31) matches the moved pattern on every move step. It is known by
  construction and is excluded.

**E22M predeclared (12:57, before code): the generic E22Z — moved content instead of labels**
- **Recipe:** the E22Z smoke exactly (joint phase from the E22F file, the same 2,000 batches, λ auto), with E22Z's
  label-derived zombie mask replaced by the moved-content mask on the TRUE frames t, t+1 of each training window.
  - Ratio < 0.3 from the valley; change > EVENT_TAU; scroll-compensated; token 31 excluded. Token space only, no labels,
    no probe.
- **Readings vs E22Z / E22FJ:** P2 (all, no-scroll), clean P1, encoder probe, carries (zombie cells).
- **Rules:**
  - P2 random-move argmax ≥ 50% and score ≥ 0.3 → the label-free allocation works; then the same sufficiency test as
    E22Z (prior + sampled imagination, readings 1-2), then full scale and from scratch.
  - 10% ≤ P2 < 50% → partial; measure which moves are missed (recall of the mask on those transitions) before anything
    else.
  - P2 < 10% → the mask does not carry the allocation; compare its training-window composition to E22Z's before
    anything else.
- **Measured before the run** (12:59-13:00; CPU, the first 100 training batches of the seed-11 order, 1.51M map cells):
  - E22Z's zombie-probe mask covers 1.27% of map cells; the moved-content mask 2.15%;
  - 40% of the zombie-probe cells lie inside the moved-content mask (the moving zombies; staying ones do not change);
  - 24% of moved-content cells are zombie-probe cells (the rest: other mobs and other moved content).
  - Under the same λ rule, moving zombies get roughly a quarter of the mask term, i.e. ~12% of the objective (E22Z: zombie
    cells ~50%; unweighted: 1.9%).

**E22Z sufficiency result (lane153, 12:49-14:03): reading 1 PASSES — sampled zombies stay localized; reading 2 FAILS (18% caught, 6% false)**
- **Prior** (20k, on the frozen E22Z posterior): latent accuracy 0.593 at 18.5k (E16's 0.615 at 20k).
- **`e22_imagine`** (4 prior samples per root, held futures; D and truth in brackets):

  | reading | E22Z sampled (mean of 4; per sample) | E22Z Δ zeroed | D | truth |
  |---|---|---|---|---|
  | depth 8: roots with a zombie cell > 0.3 | **87.3%** (84.8-92.0%) | 39.3% | 0% | 92.9% |
  | depth 8: zombie peak | **0.609** (0.595-0.624) | 0.298 | 0.13 | 0.642 |
  | depth 16: roots with a cell > 0.3 | 83-90% | 21% | — | 95.9% |
  | caught on aligned true-damage steps | 18.0% (16.7-20.0%) | 7.7% | 4.5-6% | — |
  | false drops on true unchanged steps | 6.3% (6.1-6.5%) | 0.5% | — | — |
  | imagined zombie beside on true damage steps | 47-50% | 22.6% | 8.9% | 73.6% |

- **Readings:**
  - e22_localized TRUE (≥ 50% and ≥ 0.4): the diagnosed imagination failure is fixed by sampling the
    allocation-corrected channel. To depth 16 the zombie stays as sharp as in truth.
  - e22_damage FALSE: caught triples vs D but stays below 30%; false drops are 6.3% (≤ 2% required).
- **Rule:** "1 passes, 2 fails → diagnose the next link".
- **Caveat measured next, not assumed:** false is scored against the TRUE trajectory. A sampled future in which a zombie
  reaches the player and hits is a legitimate outcome of a stochastic world. lane156 reruns with in-distribution
  readings:
  - imagined vs true rate of a zombie beside the player;
  - drawn-hit rate given imagined adjacency vs true hit rate given true adjacency;
  - caught / false split by imagined adjacency.

**E22Z reading 2 diagnosed in distribution (lane156, 14:03-14:16; the v1 readings reproduce exactly)**

| per rollout | sampled (4 samples) | E22Z Δ zeroed | truth |
|---|---|---|---|
| steps with a zombie beside the player | **21.1-22.3%** | 4.2% | **8.7%** |
| hit rate given a zombie beside | **27.6-28.4%** | 10.7% | **18.1%** |
| false drops with an imagined zombie beside (n ≈ 3,100-3,300) | 27-28% | 8.8% (570) | — |
| false drops with no imagined zombie beside (n ≈ 12,100-12,300) | 0.6-0.7% | 0.2% | — |
| caught on true damage with an imagined zombie beside | 34-39% | 27.6% | — |

- **The false drops are not hallucinated health changes:** ~860 of ~945 per sample occur where the world's own imagined
  zombie is beside the player.
- **The sampled world is miscalibrated in two exact ways:**
  1. adjacency 2.5× reality;
  2. hits given adjacency 1.5× reality (the cooldown under-enforced).
- The deterministic decoder errs the other way: zombies fade (4.2%), hits under-drawn (10.7%).
- **Next (lane157, census):** whether adjacency comes from extra zombies (mass inflation; P2 already showed predicted mass
  0.90 vs true 0.70 over the 5 candidate cells) or from faster approach (nearest-distance by depth).

**E22M result (lane155, 14:17-14:48): the LABEL-FREE moved-content allocation passes P2 — random moves 80.4% (E22Z 82.1%, D 1.1%)**
- λ auto = 0.194 (300 moved-content cells in the first batch, 2.0%).
- **Logged gain** (weighted objective) at 500 / 1,000 / 1,500 / 2,000: +0.6% / +13.4% / +29.3% / +22.9%; codes 120 / 251 /
  264 / 417.
- **P2** (held one-step, posterior Δ):

  | true move | n | E22M score / argmax | E22Z argmax | D argmax |
  |---|---|---|---|---|
  | **random (other)** | 280 | 0.534 / **80.4%** | 82.1% | 1.1% |
  | other, no scroll / scroll | 185 / 95 | 0.531 / 82.2%, 0.541 / 76.8% | 83.8% / 78.9% | 0.0% / 3.2% |
  | toward | 959 | 0.637 / 90.1% | 96.2% | 57.8% |
  | stay | 1,290 | 0.581 / 89.5% | 95.7% | 91.1% |

- **Calibration:** predicted mass over the 5 candidate cells 0.731 vs true 0.695 (E22Z 0.898: less inflation).
- **Clean P1:** +39.6%; zeroed MSE 0.01406 vs D's 0.01287 (+9%: the decoder leans on Δ somewhat more).
- **Encoder probe** (moving-only, random moves): z 0.318, q 0.312 (E22Z 0.486 / 0.439, E22FJ 0.225 / 0.254). The linear
  probe again understates the decision-level P2.
- **Carries** (unweighted D loss; E22Z in brackets):
  - zombie cells 65% removed (70%); on no-scroll steps 70% (75%);
  - entering edges 55% (52%); other changes on scroll steps 31% (11%);
  - overall gain +41.4% (+34.5%).
  - Cost: the decoder alone (Δ zeroed) is worse than D on zombie cells (29.7 vs 18.3; it hands zombies to the channel).
    With Δ: 10.3 (−44% vs D).
- **Rule (12:57):** P2 ≥ 50% and score ≥ 0.3 → the label-free allocation works. Next: the same sufficiency test as E22Z
  (prior 20k + `e22_imagine` with the in-distribution readings and the census), then full scale and from scratch.

**E22Z census (lane157, 14:48-15:01): sampled imagination MULTIPLIES zombies — 10× truth by depth 16; adjacency follows**

| cells with zombie score > 0.3 per frame | depth 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| E22Z sampled (4 samples) | 0.27-0.28 | 0.36-0.40 | 0.62-0.66 | 1.19-1.32 | **3.0-3.2** |
| truth | 0.186 | 0.185 | 0.204 | 0.233 | 0.294 |
| E22Z Δ zeroed | 0.198 | 0.193 | 0.177 | 0.155 | 0.152 |

- **Not faster approach:** the nearest zombie's distance at depth 8 is 2.37-2.51 vs true 2.46. The within-1 excess (depth 8:
  21-23% vs 9.8%; depth 16: 42-47% vs 13%) comes from the NUMBER of drawn zombies.
- **Already visible one step ahead, with the posterior's own codes** (P2):
  - with exactly one zombie, E22Z draws 1.0-1.5 cells > 0.3 (other 1.48, toward 1.24, stay 1.0; positive mass 1.4-1.8; mass
    over the 5 candidate cells 0.898 vs true 0.695);
  - a moved zombie is drawn at the new cell without being fully erased at the old one. Compounded over steps, this is the
    10× census.
- **E22M duplicates less in the same one-step test:** 0.86-0.96 cells > 0.3, mass 1.34-1.44, 5-cell mass 0.731. Its
  sufficiency census (lane158) will show whether sampled imagination stays at the true count.

**E22M sufficiency result (lane158, 15:03-16:17): both readings FAIL — sampled zombies FADE (the opposite of E22Z)**
- Prior 20k: code accuracy 0.565, CE 1.78 (E22Z's prior: 0.634, CE 1.36; E22M uses more codes, 417 per batch at the end of
  its smoke).

| reading (4 samples) | E22M sampled | E22M Δ zeroed | E22Z sampled | truth |
|---|---|---|---|---|
| depth 8: roots with a cell > 0.3 / peak | **39.5%** (36-44%) / **0.359** | 3.6% / 0.139 | 87.3% / 0.609 | 92.9% / 0.642 |
| caught on aligned damage / false | **7.6%** / **0.95%** | 2.9% / 0.23% | 18.0% / 6.3% | — |
| steps with a zombie beside | 4.0-4.5% | 1.4% | 21-22% | 8.7% |
| hit rate given a zombie beside | **0.157-0.177** | 0.032 | 0.276-0.284 | **0.181** |
| drawn zombies per frame, depth 1 / 4 / 8 / 16 | 0.15-0.17 / 0.12-0.14 / 0.11-0.12 / 0.09-0.11 | 0.087 → 0.001 | 0.27 → 3.1 | 0.19 / 0.20 / 0.23 / 0.29 |

- **E22M's hit mechanics are calibrated:** given adjacency it draws hits at reality's rate (0.16-0.18 vs 0.18), and false
  drops pass (0.95%).
- **But its zombies fade** to about a third of the true count by depth 16, so adjacency halves and localization and
  caught fall short.
- Its decoder alone erases zombies (zeroed: 0.087 → 0.001). It relies on the channel to KEEP zombies, not only to move
  them (carries: decoder alone 29.7 vs D's 18.3 on zombie cells).
- **E22Z and E22M fail in opposite directions:**
  - E22Z multiplies: a moved zombie is not fully erased at its old cell;
  - E22M fades;
  - neither conserves zombie mass under sampling.
- **Rule:** reading 1 fails → diagnose the prior first.
  - Substitution queued (lane159, `--postcodes`): self-fed frames, but each transition conditioned on the posterior's
    codes of the TRUE transition.
  - If zombies then persist at the true count, the prior's sampling is the cause; if they still fade or multiply, the
    decoder's self-fed behaviour is.

**Prior vs decoder in sampled imagination (lane159, `--postcodes`, 16:17-16:31): the chain WORKS with the true stochastic choices; mass conservation is decoder-side**

| rollout (1 sample each) | depth 8: roots > 0.3 / peak | caught / false | beside / hit given beside | drawn zombies at depth 1 / 4 / 8 / 16 |
|---|---|---|---|---|
| E22Z prior-sampled | 84.8% / 0.595 | 16.9% / 6.1% | 21.2% / 0.276 | 0.28 / 0.65 / 1.27 / 3.20 |
| **E22Z true posterior codes** | **92.9% / 0.647** | **48.9% / 3.3%** | 14.6% / 0.256 | 0.27 / 0.59 / 1.09 / 2.52 |
| E22M prior-sampled | 41.1% / 0.379 | 8.3% / 0.96% | 4.3% / 0.175 | 0.17 / 0.14 / 0.11 / 0.09 |
| E22M true posterior codes | 57.1% / 0.444 | 19.2% / 0.69% | 4.9% / 0.140 | 0.17 / 0.15 / 0.12 / 0.09 |
| truth | 92.9% / 0.642 | — | 8.7% / 0.181 | 0.19 / 0.20 / 0.23 / 0.29 |

- **With the posterior's codes of what really happened** (frames still self-fed), E22Z's imagination matches truth on
  localization and catches **48.9%** of real damage (≥ 30%). The mechanism works end to end when the stochastic choices are
  right.
- **The prior's sampling costs:** caught 48.9% → 16.9%; localization 92.9% → 84.8%. E22M: 57% → 41%, 19% → 8%.
- **Zombie mass is not conserved by either decoder, even with true codes:**
  - E22Z multiplies (2.5 per frame at depth 16 vs 0.29 true → false drops 3.3%);
  - E22M fades (0.09).
  - Both are decoder behaviours under self-feeding, not prior errors.
- **Next, exact:** the ghost left at a moved zombie's OLD cell, one step ahead with the posterior's codes (health_zombie),
  for D, E22Z and E22M.

**Ghost at a moved zombie's old cell, one step ahead (lane160, 16:32-16:49; posterior Δ for E22Z / E22M)**

| moved zombie | D: toward / random | E22Z: toward / random | E22M: toward / random |
|---|---|---|---|
| score at the old cell | 0.110 / 0.313 | 0.078 / 0.110 | 0.009 / 0.018 |
| old cell still > 0.3 | 9.6% / 47.4% | 10.2% / 15.4% | 0.2% / 1.9% |
| cells > 0.3 in the frame (one true zombie) | 0.70 / 0.68 | **1.24 / 1.48** | 0.96 / 0.95 |
| argmax at the true new cell | 57.8% / 1.1% | 96.2% / 82.1% | 90.1% / 80.4% |

- **E22Z:** the old cell is mostly erased, yet 1.24-1.48 zombie cells are drawn for one moved zombie. The old-cell ghost
  explains only 0.10-0.15 of the excess, so most extra zombies are drawn ELSEWHERE.
  - ITC's one-to-one copy constraint (no source copied twice; ITC eq. 3, never ported: our corrt head is an independent
    per-tile softmax) would not stop generated duplicates.
  - Where they are is being measured (lane161: new / old / other candidate cells / elsewhere, true-frame reference).
- **E22M is clean one step ahead** (≈ 1 zombie, no ghost), so its fade is a multi-step phenomenon.
  - Hypothesis to test (lane162): a zombie ENTERING the view has no source cell, so the moved-content mask never marks it
    and E22M's channel never learned to render entries. Exits are then uncompensated and the count declines (truth rises
    ~3% per step).
  - E22Z's mask (zombie at t or t+1 anywhere) covered entries.
- **D on random moves** leaves the zombie at its old cell (47% above 0.3): the mean-seeking "stay".

**Where the extra zombies are drawn (lane161, 16:50-17:07): mostly at the old zombie's OTHER NEIGHBOURS — one source copied into several cells**

| cells > 0.3 for a moved zombie, one step (posterior Δ) | new cell | old cell | other candidates (old cell's other neighbours) | elsewhere |
|---|---|---|---|---|
| truth (probe on the true frame), toward / random | 0.955 / 0.895 | 0 / 0 | 0 / 0 | 0.001 / 0.004 |
| D | 0.483 / 0.000 | 0.096 / 0.474 | 0.109 / 0.229 | 0 / 0 |
| **E22Z** | 0.942 / 0.872 | 0.102 / 0.154 | **0.130 / 0.346** | **0.071 / 0.158** |
| E22M | 0.868 / 0.771 | 0.002 / 0.019 | 0.062 / 0.147 | 0.027 / 0.049 |

- **E22Z's surplus** (0.30 toward / 0.66 random per moved zombie):
  - mostly the old cell's other neighbours, plus the old cell itself, all cells that can COPY the old zombie token
    through the corrt head's neighbour or self candidate;
  - then "elsewhere" (0.07-0.16).
  - The neighbour and old-cell surplus is one source copied into several destinations: exactly what ITC's transport
    constraint (each previous token copied at most once; arXiv 2605.16457 eq. 3, never ported) excludes.
  - "Elsewhere" is not covered by that mechanism.
- **E22M:** surplus 0.09 / 0.22, small; its multi-step fade waits on lane162 (entering zombies).

**Zombies entering the view (lane162, 17:07-17:24; one step, posterior Δ; 70 true entering-zombie cells over 5,865 scroll steps)**

| | D | E22Z | E22M |
|---|---|---|---|
| true entering zombies drawn (> 0.3) / mean score (true-frame probe 0.588) | 0% / 0.016 | 21.4% / 0.171 | **0% / 0.006** |
| false entering zombies per scroll step (true entries: 0.0119) | 0 | **0.0102** | 0.0002 |

- **E22M never draws an entering zombie**, as predicted: its moved-content mask cannot mark content without a source cell.
  - Entries are rare (~0.07 zombies per frame over 16 steps), so this explains at most about a third of E22M's fade (0.09
    vs true 0.29 at depth 16). The rest is zombies already in view lost under self-feeding.
- **E22Z invents entering zombies** at almost the true entry rate (0.0102 vs 0.0119 per scroll step) and catches 21% of the
  real ones: a second, smaller source of its multiplication, beside the neighbour copies.

**Copy mass of the zombie token (lane163, 17:25-17:42): my "one source copied into several cells" reading is REFUTED**

| moved zombie, one step (toward / random) | D | E22Z | E22M |
|---|---|---|---|
| zombie token's total outgoing copy weight (ITC row sum) | 0.81 / 0.80 | 0.83 / 0.84 | 0.86 / 0.84 |
| at the extra zombie cells: copy-from-zombie weight | 0.57 / 0.73 | **0.07 / 0.05** | 0.42 / 0.45 |
| at the extra zombie cells: generate weight | 0.11 / 0.12 | **0.12 / 0.14** | 0.13 / 0.09 |
| extra zombie cells (n) | 190 / 187 | 280 / 175 | 84 / 57 |

- **No world copies the zombie token more than once** (total copy weight < 1). ITC's transport row constraint would not
  bind; ITC's one-to-one decoding would NOT remove E22Z's extra zombies. (The 17:07 reading is withdrawn.)
- **E22Z's extra zombies are fractional generated zombies:** cells that keep ~80% of their own content and mix in ~12-14%
  of a strongly zombie-like generated token. The probe reads the blend as a zombie. The region code locates the move
  ambiguously, and the soft mixture hedges across the candidate cells.
- D's extras are copies (its mean-seeking spread). E22M's are mixed.
- **What this points at:** the SOFT per-cell mixture, which lets a cell hold a fraction of a zombie that self-feeding then
  compounds. ITC's binarized decoding (eq. 4: each cell takes exactly one source, a copy or a generated token) removes
  fractions by construction.
  - Our 09-28 hard-decoding failure was entering cells taking the zero pad. ITC's Craftax rule (edges and HUD from the
    generator) is its answer.

**E22H predeclared (17:43, before code): decoding-only test — do integer (hard) cell decisions conserve zombies in sampled imagination?**
- **Hypothesis, from the measurements above:** the soft per-cell mixture lets a cell hold a FRACTION of a zombie, and
  self-feeding compounds the fractions.
  - E22Z: fractional generated zombies grow into extra zombies (×10 by depth 16).
  - E22M: a moved zombie is re-mixed with neighbouring content each step and dilutes (×1/3).
  - ITC's binarized decoding (arXiv 2605.16457 eq. 4; each cell takes exactly one source, a copy or a generated token)
    allows no fractions: a copy carries the zombie token exactly.
- **What is already known against it (09-28/29):**
  - hard decoding of the soft corrt head filled entering cells with the zero pad;
  - ITC's Craftax rule (ring and HUD from the generator) was harmful with an L1-trained continuous generator (mean
    regression on never-seen cells).
  - The E22 worlds differ: their channel carries 52-61% of the entering-edge error, so the generator at the edges is
    informed by Δ.
- **Arms** (imagination only; no retraining; `tworld.py` untouched while D from scratch trains, since its script hash is
  stamped at save):
  - (a) **itc:** `hard_decode` (per-cell argmax over self / 4 neighbours / generate) + ITC's Craftax regions (map border
    ring and HUD from the generator);
  - (b) **interior:** hard decisions in the interior map cells, the soft head kept on the ring and HUD (two forward passes
    combined per token).
- **Worlds and rollouts:** E22Z and E22M with their priors, 4 samples, e22_imagine's readings (1-2, in-distribution,
  census) vs the soft results above.
- **Rules:**
  - census at depth 16 within 0.5-2× truth (0.147-0.588) AND readings 1-2 pass → the soft mixture is the cause of the mass
    failure, and integer decoding is the generic, label-free fix at imagination.
  - census corrected but a reading fails → diagnose the next link.
  - census not corrected → the soft mixture is not the (whole) cause; diagnose before anything else.

**E22H result (lane164, 17:44-18:51): integer decisions conserve E22M's zombies (census ≈ truth) but worsen E22Z's; ITC's region rule fails again**

| sampled imagination (4 samples) | drawn zombies at depth 1 / 4 / 8 / 16 (truth 0.19 / 0.20 / 0.23 / 0.29) | depth 8: roots > 0.3 / peak | caught / false | beside (truth 8.7%) / hit given beside (truth 0.181) |
|---|---|---|---|---|
| E22Z soft | 0.28 / 0.65 / 1.27 / 3.2 | 87% / 0.61 | 18.0% / 6.3% | 21-22% / 0.28 |
| E22Z itc (hard + ring and HUD from the generator) | 0.36 / 2.1 / 8.0 / **22.8** | 96% / 0.88 | 47% / **43%** | 48-50% / 0.40 |
| E22Z interior (hard interior, soft ring and HUD) | 0.28 / 0.85 / 2.5 / **8.2** | 88% / 0.79 | 21.8% / 8.3% | 25-26% / 0.31-0.32 |
| E22M soft | 0.16 / 0.13 / 0.11 / 0.10 | 39.5% / 0.36 | 7.6% / 0.95% | 4.0-4.5% / 0.16-0.18 |
| E22M itc | 0.18 / 0.42 / 0.71 / 1.71 | 51% / 0.50 | 45% / **43%** | 10% / 0.39-0.42 |
| **E22M interior** | **0.16-0.18 / 0.17-0.19 / 0.17-0.22 / 0.25-0.27** | 48.0% / 0.48 | 9.9% / **1.6%** | 5.1-6.0% / 0.20-0.21 |

- **ITC's Craftax region rule fails here, as on 09-29:** the generated HUD wrecks health (43% false drops) and the
  generated ring floods the view with zombies. Not a fix for continuous tokens.
- **E22M interior: the fade is cured.**
  - The census tracks truth (0.25-0.27 vs 0.29 at depth 16; within the predeclared 0.5-2×);
  - hits given adjacency calibrated (0.20-0.21 vs 0.18); false 1.6% (≤ 2%).
  - Exact integer copies carry a zombie forward; soft re-mixing diluted it.
  - Readings: localization 48.0% (< 50%, just), peak 0.48 (≥ 0.4); caught 9.9% (< 30%). Both readings FALSE.
- **E22Z interior: multiplication WORSE** (8.2 at depth 16 vs soft 3.2). Even its deterministic decoder multiplies under
  integer decisions (zeroed census 0.61 vs soft 0.15).
  - E22Z's extra zombies are not just fractions: integer commitments make them whole.
  - Its label-derived weighting taught it to draw zombies liberally (one-step mass 0.90 vs true 0.70).
- **Rule:**
  - E22M: census corrected, readings fail → diagnose the next link;
  - E22Z: census not corrected → the soft mixture is not its whole cause (it is the label-derived diagnostic arm, not the
    candidate fix).
- **E22M interior's remaining gaps, measured:**
  - zombies beside the player 59-69% of truth; entering zombies never drawn (lane162);
  - per-step imagined damage rate ≈ 0.055 × 0.21 ≈ 1.15% vs true 0.087 × 0.181 ≈ 1.57% (~73%).
  - "Caught on the true trajectory" also has a ceiling for a sampled world: with E22Z's TRUE codes it was 48.9%.

**E22M + hard interior with the TRUE codes (lane165, 18:51-19:01): reading 1 passes and reading 2 nearly does — the prior costs half**

| E22M, hard interior decoding | prior-sampled (sample 0) | true posterior codes | truth |
|---|---|---|---|
| depth 8: roots > 0.3 / peak | 49.1% / 0.51 | **70.5% / 0.64** | 92.9% / 0.64 |
| depth 16: roots > 0.3 | 43.9% | 62.2% | 95.9% |
| caught on aligned damage / false | 11.1% / 1.7% | **25.2% / 1.3%** | — |
| zombies beside / hit given beside | 5.7% / 0.215 | 6.0% / **0.187** | 8.7% / 0.181 |
| drawn zombies at depth 1 / 16 | 0.168 / 0.247 | 0.178 / 0.218 | 0.186 / 0.294 |

- **With the right stochastic choices, the label-free pipeline passes reading 1** (70.5% ≥ 50%, peak 0.64 ≥ 0.4) and is
  close on reading 2: caught 25.2% vs 30%; false 1.3% ≤ 2%; hits given adjacency calibrated (0.187 vs 0.181); census
  conserved.
- **The prior's sampling costs about half:** localization 70.5% → 49%, caught 25% → 11%.
- The remainder (beside 6.0% vs 8.7%) is the channel / decoder, including the entering zombies E22M never draws (lane162).
- **The prior is under-trained by its own curve** (single training batches, noisy): accuracy 0.21 / 0.35 / 0.41 / 0.36 /
  0.48 / 0.45 / 0.47 / 0.56 at 2.5k-20k; CE 4.12 → 1.78 and still falling.
  - The 20k length was E16's adoption of Delta-IRIS crafter.yaml values, not a measured optimum.

**E22MP predeclared (19:02, before launch): is the prior's training length the limit of E22M's sampled imagination?**
- **Run:** E22M's prior continues from its saved 20k state (same run: optimizer, batch order and RNG restored) to 60k.
- **Readings:**
  1. held prior fit (`e22_imagine --prior-held-only`: teacher-forced code CE and accuracy on the held trajectories' true
     transitions), 20k vs 60k;
  2. E22M + hard interior decoding with the 60k prior, 4 samples, vs the 20k prior (localization 48.0%, caught 9.9%) and
     the true-code ceiling (70.5%, 25.2%).
- **Rules:**
  - held CE falls AND imagination moves at least halfway to the ceiling (localization ≥ 59%, caught ≥ 17.5%) → prior
    training length was the limit; then full-scale stage A and the from-scratch E22M.
  - held CE falls but imagination does not move → the prior's fit is not what limits sampling; diagnose (sampling
    temperature / calibration of the sampled moves) before anything else.
  - held CE does not fall → the prior has converged at its capacity; diagnose prior capacity before anything else.

**Test 3 verdict for D: NOT a continuation artifact — D from scratch (36k + 6k, lane131, 09:15-18:39) beats D continued on every reading**

| row 9 (window 10), held | M16 | D continued | **D from scratch** |
|---|---|---|---|
| hits drawn / 1,211 | 60 | 569 | **647** |
| fresh / 207 | 0 | 31 | **62** |
| since 6 / 157 | 2 | 63 | 60 |
| false / 1,728 | 0 | 1 | 5 |
| generator beats copy on hits | 3.6% | 85.4% | **89.5%** |
| drawn health within 0.5 | 2% | 34% | **51%** |

| h63 information (window 15, 5 probe seeds, clustered bootstrap) | M16 | D continued | D from scratch | scratch − continued | scratch − M16 |
|---|---|---|---|---|---|
| phase, since = 6 | 0.892 | 0.947 | **0.971** | +0.024 [+0.016, +0.032] | +0.079 [+0.064, +0.095] |
| phase, since 1-4 | 0.912 | 0.969 | **0.989** | +0.019 [+0.012, +0.028] | +0.076 [+0.061, +0.092] |
| approach adjacency | 0.766 | 0.956 | **0.987** | +0.031 [+0.010, +0.054] | +0.221 [+0.168, +0.277] |

- **Every difference is resolved.** The Delta-IRIS loss installs both blockers from step 0, better than continued; it was
  ahead of M6 at every stage-1 snapshot (6k-30k).
- This answers the 10-08 question for D: the positive E21 result is not caused by continuing a converged world. GE from
  scratch had been worse than its continuation; D is better.
- **Damage panel** (check_damage, simulator truth, 337 damage transitions, window 15):
  - teacher-forced caught **81.9%** (D continued 73.3%), fresh 27/31 (19/31), false drops 1.8%;
  - health change ≤ −2 read correctly 61.3%;
  - self-fed caught 7.2% (D continued 6%).
  - From scratch it draws damage better from true history. Its deterministic imagination still loses it, the
    localization failure the stochastic channel addresses.
- **teval** (/V, D from scratch − M16, seed-clustered CI):
  - one step: all −0.009*, interact −0.522*, moved / blocked / idle n.s., sleep +0.005*;
  - imagination at depth 1 / 4 / 8 / 16: −0.003* / −0.017* / −0.051* / −0.034*.
- **Imagined facts, M16 / D continued / D from scratch:**

  | depth | food R² | health R² | zombie-near AUC |
  |---|---|---|---|
  | 1 | 0.997 / 0.992 / 0.995 | 0.981 / 0.985 / 0.985 | 0.842 / 0.976 / 0.998 |
  | 8 | 0.932 / 0.839 / 0.935 | 0.862 / 0.841 / 0.845 | 0.530 / 0.743 / 0.678 |
  | 16 | 0.881 / **0.582** / **0.875** | 0.592 / 0.582 / 0.557 | 0.511 / 0.609 / 0.614 |

  - **The food cost of D continued was a continuation artifact:** from scratch, imagined food at depth 16 is 0.875 vs M16's
    0.881. The zombie gains are kept.
- **Gradient allocation:** token-63 hits carry 4.58% of the backbone gradient norm (D continued 7.56%, M16 0.16%; 0.20% of
  the objective). The router discriminates: generate weight 0.60 on hits vs 0.10 on unchanged.
- **Overall:** D trained from scratch is the best deterministic world on every health reading, with no food cost and a
  better teval. It is the decoder to build the stochastic channel on.

**E22MS predeclared (19:29, before launch): the label-free channel recipe on the FROM-SCRATCH decoder**
- **Why:** D from scratch is the best deterministic world on every health reading, with no food cost (above). E22F to E22H
  were built on D continued.
- **Recipe, unchanged except the decoder:**
  - E22F-style frozen phase (2,000 updates, batch 16, `--freeze-world`);
  - E22M joint phase (`--motion-weight auto`, 2,000 updates);
  - prior 20k;
  - `e22_imagine` with soft and hard-interior decoding (4 samples), plus P2 one step.
- **Comparator:** E22M on D continued (P2 random 80.4%; hard interior: census 0.25-0.27, localization 48.0%, caught 9.9%,
  false 1.6%).
- **Readings:** P2, census, readings 1-2, in-distribution rates. Reported side by side; the E22MP prior-length result
  applies to both.
- **Rule:** no new threshold. If the from-scratch decoder does better on the predeclared E22 readings, the channel work
  continues on it.

**E22MP result (lane166, 19:02-21:16): the prior OVERFITS — longer training is not the lever**

| E22M prior | 20k | 60k |
|---|---|---|
| training-batch code accuracy | 0.565 | 0.666 (40k) |
| **held** code CE / accuracy (76,152 codes) | 2.307 / 0.470 | **3.572 / 0.405** |
| E22M + hard interior: depth 8 roots > 0.3 / peak | 48.0% / 0.48 | 51.8% / 0.52 |
| caught / false | 9.9% / 1.6% | 11.8% / 1.6% |
| drawn zombies at depth 16 (truth 0.29) | 0.25-0.27 | 0.27-0.33 |

- Rule (19:02): held CE does not fall → the prior's GENERALIZATION is the limit, not its training length.
  - 60k × 32 windows ≈ 1.9M windows from a 19,789-episode pool.
  - The prior's input is a crude per-tile Linear 192 → 8 flattened to 512.
  - Imagination moved ~1/5 of the way to the true-code ceiling (70.5% / 25.2%), not half.

**E22MS result (lane167, 19:30-21:33): the label-free channel on the FROM-SCRATCH decoder is better at every step; hard decoding HURTS it**

| | E22M (D continued) | **E22MS (D from scratch)** | truth |
|---|---|---|---|
| P2 random moves: score / argmax | 0.534 / 80.4% | **0.585 / 86.4%** | — |
| P2 toward / stay argmax | 90.1% / 89.5% | 94.1% / 88.9% | — |
| entering zombies drawn | 0% | 0% | — |
| prior (20k) held accuracy / CE | 0.470 / 2.31 | **0.518 / 2.04** | — |
| soft: depth 8 roots > 0.3 / peak | 39.5% / 0.36 | **57.6% / 0.45** | 92.9% / 0.64 |
| soft: drawn zombies at depth 1 / 8 / 16 | 0.16 / 0.11 / 0.10 | **0.17 / 0.17 / 0.19-0.21** | 0.19 / 0.23 / 0.29 |
| soft: caught / false | 7.6% / 0.95% | 13.2% / 1.3% | — |
| soft: hit rate given adjacency | 0.16-0.18 | 0.175-0.206 | 0.181 |
| hard interior: depth 8 roots / caught | 48.0% / 9.9% | 27.7% / 10.4% | — |

- On the from-scratch decoder, soft decoding already conserves zombies; the fade was a property of the decoder built on
  D continued.
  - Reading 1 passes with SOFT decoding (57.6%, 0.445). Reading 2 fails (13.2% caught; false 1.3% passes).
- **Hard decoding is not a general fix:** it compensated E22M's decoder defect and hurts E22MS (57.6% → 27.7%).

**Where the day ends (21:33):**
- The best pipeline is the label-free channel on D from scratch with soft decoding:
  - one step, random moves 86%;
  - sampled imagination: localization 58% (truth 93%), zombie count stable, hits given adjacency calibrated, false 1.3%;
  - imagined damage caught 13% (30% required).
- **Measured gaps:**
  1. **The prior:** the true-code ceiling (70.5% / 25%, on D continued) vs sampled (48-58% / 10-13%). It overfits; its
     input representation is crude.
  2. **Entering zombies:** never drawn by the label-free mask (rare: ≤ 1/3 of the count dynamics).
  3. **Adjacency:** zombies beside the player ~60-70% of truth.
- **Open evaluation question:** "caught on the true trajectory" has a ceiling for a sampled world. The decision-level test
  (H16 with N sampled futures; actor vs BC) is the real objective and has not been run on these worlds.

**Gradient allocation of the trained E21 arms** (03:25; `health_gradient.py` on each arm's own objective and batches; D under its
Delta-IRIS loss):

| world | objective | token-63 hits: share of objective | token-63 hits: share of backbone gradient norm | other terms' backbone share |
|---|---|---|---|---|
| M16 | 0.0509 | 0.27% | 0.16% | — |
| C0 | 0.0500 | 0.27% | 0.30% | — |
| G | 0.1163 | 0.12% | 0.11% | **generator loss 81%** |
| GE | 0.1182 | 0.14% | 0.56% | generator 79%, event 0.5% |
| E′ | 0.0504 | 0.27% | 0.11% | event term 1.0% (at token-63 hits 0.24%) |
| **D** | 0.0423 | 0.36% | **7.56%** | — |

- D gives token-63 hits 25× C0's share of the backbone gradient (46× M16's): the measured mechanism behind its success,
  confirming the allocation root of the diagnosis.
- G's generator loss takes 81% of the backbone's gradient, so the representation is reshaped for token reconstruction. That
  is the measured cause of G's lost approach adjacency (0.761 → 0.619).
- E′'s event term is 1% of the gradient: its partial effect matches its partial dose.

**Literature step on the newly diagnosed problem** (03:35; a deterministic world averages stochastic events in imagination):
- *Dreamer 4* (arXiv 2509.24527, §2 and §3.2, Eq. 6-8, Table 2; vendored PDF, read): the dynamics model is a *generative* model
  in continuous token space.
  - Shortcut forcing = diffusion forcing + shortcut models, sampled autoregressively with K = 4 steps per frame; past inputs are
    corrupted to τ_ctx = 0.1.
  - Trained in x-space with a squared loss ‖ẑ₁ − z₁‖², ramp weight w(τ) = 0.9τ + 0.1.
  - Ablations (FVD): shortcut model 875 → 329; x-loss 326 → 151; ramp 151 → 102.
- *Delta-IRIS* (2406.19320): discrete stochastic Δ tokens with an autoregressive prior; the decoder trained with L2 + L1 +
  worst-pixel (read 10-03; E16 / D notes above).
- *Reading against tonight's measurements:* both of our measured failures are departures from these sources' dynamics
  objectives.

| failure | our world | the sources |
|---|---|---|
| allocation | per-element L1 (copies rare events: 0.27% of the objective, 0.16% of the gradient) | squared loss (D: 7.56%) |
| imagination | deterministic regression (averages stochastic zombie moves and timers; imagined zombie beside at damage 8.9% vs 73.6%) | sample |

- A flow / shortcut head predicts JEPA tokens, not pixels, so it stays reconstruction-free.
- **Candidate designs, to predeclare after the from-scratch runs report:**
  - (a) Dreamer 4's shortcut-forcing objective as the per-tile world's prediction head (the project's own foundation);
  - (b) faithful Delta-IRIS (E16's Δ channel + D's loss).
  - Either must be judged on self-fed imagination (damage panel, zombie presence at damage steps, food drift), not only
    teacher-forced.

**Test 3 in flight: GE from scratch** (lane122 started 03:26 after the equivalence gate passed again; stage 1 at 0.42 s / update;
lane135 reads each stage-1 snapshot against M6 at the same update, window 5 = output row 4, the 6-frame recipe's death-cue row):

| @ 6k | generator beats copy | since 1..4 in h63 | approach adjacency | event head AUC | emitted hits / false (row 4) |
|---|---|---|---|---|---|
| M6 | 0.1% | 0.916 | 0.708 | — | 9 / 220 |
| GE from scratch | **57.5%** | **0.975** | 0.593 | 0.468 | 12 / 310 |
| paired difference | | **+0.060 [+0.046, +0.073]** | **−0.115 [−0.176, −0.054]** | | |

- The direction matches the continuations (phase up, adjacency down: the generator loss dominating the gradient). The phase
  gain is larger than in any continuation (GE vs C0 +0.011 unresolved; G +0.034).
- The event head is still untrained at 6k from scratch (0.468): its budget problem is not specific to continuation.
- Both worlds' false drops sit at the row-4 death cue of 6-frame training. Stage 2 decides.
- Stage-1 trajectory (05:30; `ci_scratch_ge_at*.json`; window 5 = row 4):

| update | since 1..4: M6 → GE from scratch (paired diff) | adjacency: M6 → GE from scratch (paired diff) | GE: generator beats copy / event AUC | emitted hits / false (M6 → GE) |
|---|---|---|---|---|
| 6k | 0.916 → 0.975 (+0.060 [+0.046, +0.073]) | 0.708 → 0.593 (−0.115 [−0.176, −0.054]) | 57.5% / 0.468 | 9 / 220 → 12 / 310 |
| 12k | 0.902 → 0.964 (+0.062 [+0.045, +0.080]) | 0.646 → 0.513 (−0.133 [−0.201, −0.062]) | 61.0% / 0.524 | 92 / 153 → 49 / 147 |
| 18k | 0.898 → 0.943 (+0.046 [+0.030, +0.062]) | 0.688 → **0.432** (−0.255 [−0.316, −0.194]) | 61.0% / 0.573 | 43 / 113 → 105 / 131 |
| 24k | 0.903 → 0.931 (+0.027 [+0.012, +0.042]) | 0.717 → 0.642 (−0.074 [−0.132, −0.014]) | 78.8% / 0.590 | 100 / 117 → 74 / 105 |
| 30k | 0.904 → 0.927 (+0.023 [+0.009, +0.039]) | 0.788 → 0.631 (**−0.155 [−0.212, −0.097]**) | 68.6% / 0.587 | 96 / 115 → 161 / 143 |

  - From step 0, GE's objective removes approach adjacency from h63 even more than in the continuation (G 0.619, GE 0.680 at
    6k). The held AUC at 18k is below 0.5 (the probe's fit-episode rule inverts on held episodes): noted as anomalous, not
    interpreted.
  - **GE's adjacency loss is not a continuation artifact.** It is the measured property of an objective whose generator term
    takes ~80% of the backbone's gradient.
  - The phase gain is real from scratch too.
  - The event head learns slowly (0.47 → 0.59 by 24-30k, about where it plateaus).
  - At 30k the from-scratch trade (phase +0.023, adjacency −0.155) matches the G continuation's against its control (phase
    +0.044, adjacency −0.154). Trained from step 0 through the canonical recipe, the generator-loss objective converges to the
    same representation trade as the continuation. Stage 2 (L16, row 9 vs M16) is the deciding reading.

**Test 3 verdict for GE (08:47; stage 2 finished 08:31; `ci_scratch_ge_end_w15.json`, `_w10.json`, `health_row9.py`):**

| world | row-9 hits / 1,211 (fresh / false) | generator beats copy | since = 6 / since 1..4 / adjacency in h63, row 9 | event head AUC |
|---|---|---|---|---|
| M16 (from scratch, natural loss, 42k) | 60 (0 / 0) | 3.6% | 0.894 / 0.920 / 0.821 | — |
| GE as a continuation of M16 (48k) | 30 (0 / 0) | 78.6% | 0.899 / 0.910 / 0.725 | 0.570 |
| **GE from scratch (42k)** | **91 (1 / 0)** | 79.9% | **0.862 / 0.875 / 0.699** | 0.672 |

| paired contrast, row 9 | since = 6 | since 1..4 | adjacency |
|---|---|---|---|
| M16 → GE from scratch | −0.032 [−0.046, −0.018] | −0.045 [−0.064, −0.028] | −0.120 [−0.180, −0.061] |
| GE continuation → GE from scratch | −0.036 [−0.049, −0.024] | −0.036 [−0.049, −0.022] | −0.025 [−0.069, +0.018] |

(Row 14: M16 → from scratch since = 6 −0.019, adjacency −0.125; continuation → from scratch since = 6 −0.038; all resolved.)

- **By the predeclared rule, GE's negative is not a continuation artifact.** From step 0 the same objective ends worse than M16 on
  every ingredient and worse than its own continuation on the phase. It draws 7.5% of hits at row 9.
- The stage-1 gain on in-window phase (since 1..4 +0.06 at 6k-12k) did not survive into the L16 world. As recorded beforehand,
  the 6-step phase can only be learned in stage 2.
- **Answer to the 21:00 question, for the arms run tonight:** E21's representation negatives are not caused by continuing.
  - Test 2 shows the continuation installs B1 within 500 updates when asked directly.
  - Test 3 shows the GE objective fails the same way from scratch.
  - What remains open: D's positive result from scratch (lane131, next), and a mechanistic account of *why* from scratch ends
    worse on the phase (gradient allocation of the from-scratch world: lane122's `health_gradient` job).
- **Why GE from scratch ends worse on the phase, measured** (lane122 `health_gradient`, 09:14):

| world | token-63 hits' share of backbone gradient | generator-loss term's share |
|---|---|---|
| M16 | 0.164% | — |
| GE as a continuation | 0.560% | 79% |
| **GE from scratch** | **0.043%** (lowest of any world) | 75% |
| D | 7.56% | — |

  - Trained from step 0, the generator loss crowds the hits' gradient to a quarter of even M16's share, so the phase is learned
    less. The continuation started from M16's phase.
  - Simulator-truth damage panel for GE from scratch: teacher-forced caught 4.2% (C0 5.0%, D 73.3%); fresh 0; self-fed 0.
- GE from scratch, full evidence (`evidence_scratch_ge.json`): drawable from h63 at row 9 is **317** (M16 334), so the
  information is no better than M16's, and the head emits 91 of them.
- **A third measured factor, across all worlds at row 9: emitted / drawable**

| world | drawable from h63 | emitted | emitted / drawable | objective |
|---|---|---|---|---|
| M16 | 334 | 60 | 18% | L1 |
| oracle | 848 | 70 | 8% | L1 + labels |
| G | 136 | 64 | 47% | L1 + generator loss |
| E′ | 367 | 79 | 22% | L1 + event head |
| GE from scratch | 317 | 91 | 29% | L1 + generator + event |
| **D** | **790** | **569** | **72%** | 1.0 L2 + 0.1 L1 + 0.01 worst element |

  - Under every L1-based objective, the world expresses a small fraction of the hits its own h63 supports, even with the
    generator fixed (G, GE). Under D's loss it expresses 72%.
  - So the loss shape matters for *expression*, not only for allocation. The exact mechanism (the router's calibration under L1
    vs L2) was measured next (09:25, `health_express.py`, `express.json`):

| world, row 9 | drawable hits | emitted | generator beats copy on drawable | generate weight: drawable hits vs unchanged | missed: router (good gen, weight < 0.5) | missed: generator (not better than copy) | missed: good gen, weight ≥ 0.5 (partial blend) |
|---|---|---|---|---|---|---|---|
| C0 | 335 | 18.8% | 17.9% | 0.15 vs 0.06 | 1 | 271 | 0 |
| oracle | 848 | 7.0% | 3.5% | 0.09 vs 0.05 | 1 | 787 | 1 |
| G | 136 | 47.1% | 91.2% | 0.60 vs 0.32 | **52** | 12 | 8 |
| GE from scratch | 317 | 28.7% | 94.6% | 0.55 vs 0.43 | **136** | 7 | 83 |
| D | 790 | 64.9% | 94.1% | **0.57 vs 0.13** | 117 | 38 | 122 |

  (Drawable sets come from each world's own calibrated probe, refitted here, hence D's 64.9% vs the 72% from evidence counts.)
  - Without a generator loss (C0, oracle), the misses are the generator (B2).
  - With the generator fixed (G, GE), the misses move to the **router**. The generate weight barely separates hits from unchanged
    frames (0.55-0.60 vs 0.32-0.43): the generator loss makes the generator a near-copy on unchanged frames, so choosing it
    costs nothing there, and on hits the router's signal is the rare-event allocation problem again.
  - D's router separates them (0.57 vs 0.13).
  - **One root, three starved components:** under L1, rare events get almost no gradient. That starves the representation (B1),
    then the generator (B2), then (once the generator is fixed) the router. D's loss allocates to all three.


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
