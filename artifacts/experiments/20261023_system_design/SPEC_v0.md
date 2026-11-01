# System specification v0 — a tiered world model for offline imagination training

Draft 2026-10-23 for review by the user (and Codex). No code exists for this design. Every choice cites the evidence or
the judgement behind it; open choices are marked **OPEN** with the options. Nothing here is validated.

---

## 0. Purpose, commitments, non-goals

**Purpose.** An agent that learns from a fixed offline dataset (Craftax-Classic, 2D, 63x63 frames, 17 actions) by
training its policy inside a learned world model, on one 6 GB GPU. The world model's job is to be a coherent,
grounded playground for the actor: what the agent knows and what imagination uses must agree.

**Commitments (agreed in the 10-23 discussion).**
1. Offline only. Imagination replaces interaction.
2. No full-frame pixel reconstruction. One exception: a fixed-budget fovea whose next-step pixels are predicted.
3. The encoder is trained jointly with the world.
4. Grounded targets: native reward, native termination, fovea pixels.
5. "The model doesn't know" is part of the system (offline).
6. Memory is tiered: a recurrent working state, object records, a location-keyed episodic store with attention over
   (location, content) pairs, and parametric knowledge in the weights.

**Explicit inductive biases (stated plainly, Codex's point).** (a) Locations live in a 2D translational frame and
retrieval attends by relative 2D displacement; the frame is learned from actions and observations, but its geometry is
assumed. (b) Moving things are represented as a small set of object records. (c) Chance is attached to object records
and to never-seen content. Each is a design bet, revisitable.

**Non-goals for v0.** Online data collection; 3D (but §9 keeps the path open); robustness of self-location to real
sensor noise (untestable in Craftax); biologically inspired pruning (later).

---

## 1. Overview

```
 frame o_t, action a_t-1 ──► Perception ──► periphery tokens P_t (81 x d), fovea tokens F_t (k glimpses)
                                │
            ┌───────────────────┼──────────────────────────────────────────┐
            ▼                   ▼                                          ▼
   Self-location belief   Object records R_t (<= K slots)         Episodic store E (real) + overlay O_b (per branch)
   L_t (hypotheses)       beliefs: exists, type, rel. position,    entries: (location key, content value, time,
                          hidden state, last seen, identity         source = real | imagined, confidence)
            │                   │                                          ▲ write          │ read (by location / content)
            └───────────┬───────┴──────────────────────────────────────────┴────────────────┘
                        ▼
              Working state W_t (recurrent core: Mamba over a per-token state; body, counters, recent context)
                        │
        ┌───────────────┼─────────────────┬────────────────────────┐
        ▼               ▼                 ▼                        ▼
   Transition      Grounded heads     Fovea policy           Actor / critic (read-only)
   (afterstate     reward, term.,     where to look next     policy, value
   + chance)       fovea pixels
```

One step has two modes. **Posterior (observed):** the real frame updates every belief and writes to E. **Prior
(imagined):** the transition predicts the next beliefs from the action alone, samples chance, reads E and the branch's
overlay O_b, and writes only to O_b.

---

## 2. Components

### 2.1 Perception
- **Periphery:** a ViT encoder on 7x7 patches -> 81 tokens of width d (start d = 128). Trained jointly (commitment 3).
- **Fovea:** k glimpses per step (start k = 1; EDA: two glimpses cover all one-step consequence cells in 92-96% of
  event steps). A glimpse is a 3x3-cell window (21x21 px); its raw pixels are embedded into k tokens. In Craftax the
  periphery already has full resolution, so the fovea's main role here is the *prediction target* (§4, L2); its input
  role matters for higher-resolution domains.
- **Fovea policy:** a head on W_t outputs a distribution over the 49 window positions (7x7 placements on the 9x9 grid).
  Trained by the value of what it reveals for grounded prediction (§4, L8), never by its own pixel error (dark-room trap).

### 2.2 Working state W_t
- A per-token recurrent state: 81 cell tokens + K record tokens + 1 body token (HUD-derived), width 256, a spatial
  attention block per step and a Mamba recurrence per token across time (the D/PI scaffold that measured the Mamba recall
  edge; R2I shows SSM world models hold long memory).
- Holds: body variables and counters (hunger/thirst/fatigue need ~64 frames; recover needs the episode — a recurrent
  integrator is the right tool), recent context, the faced cell (move success depends on it, S1), interaction state.
- Not responsible for: places last seen more than a few dozen steps ago (that is E), entity identity (that is R).

### 2.3 Object records R_t (entity beliefs, not simulator fields)
- **Capacity K is a cap, not an assumption.** Start K = 8. Evidence: entities alive within 12 cells and seen within 64
  steps are <= 6 in 99.5% of frames and <= 8 in 99.99% — but that count used simulator knowledge of aliveness and
  distance; a learned agent must also hold uncertain and dead entities, so it needs headroom. K is a hyperparameter; the
  overflow rate is logged in the integrated system and K raised if overflow is frequent.
- **Each record is a belief:** existence probability e; type distribution; position belief relative to self (mean and
  spread, in the self-location frame — an object-vector code); a hidden-state latent h (cooldown, health, intent — not
  decoded fields); last-seen time; identity token.
- **Update (posterior).** Visible entity evidence comes from the periphery and fovea tokens. Records and evidence are
  matched by a soft assignment with three options per evidence item: *same as record j*, *new entity*, *nothing*
  (Sinkhorn-style partial matching, as ITC does for tokens). A record that is expected visible (position belief inside
  the view) but unmatched lowers e (disappearance); unmatched evidence spawns a record (birth).
- **Update (prior).** Each record advances with its own dynamics and its own chance code (≈1.4 bits per visible mob,
  near-independent across mobs); its position spread widens while unseen (zombies median 3 cells after <= 64 steps).
- **Uncertain identity.** When matching is ambiguous (7-12% of re-entries had another plausible candidate, by an
  oracle-assisted heuristic), the assignment stays soft in training; in imagination one assignment is sampled per branch
  so each branch is a coherent world.
- **Overflow and eviction.** If more than K records are needed, evict the record with the lowest e x relevance
  (relevance = learned, e.g. distance and threat); before eviction its last belief is written to E as a place trace.
- **Place traces.** On eviction or leaving view, a trace "type T, last seen here at time t, hidden state h" is written
  to E. Traces constrain future matching (prior over where a re-appearing entity came from) but never resurrect an entity
  as a known fact: a trace only raises the prior of a "same entity" hypothesis.

### 2.4 Self-location belief L_t
- **Representation.** A small set of weighted hypotheses (start: 1 primary + up to 3 alternatives), each a learned
  location code in a 2D frame. Codex's point: averaging content retrieved for different hypotheses creates impossible
  scenes, so retrieval is done *per hypothesis*, and imagination samples one hypothesis per branch.
- **Predict.** Move each hypothesis by the action, gated by a learned move-success probability read from the faced cell
  (in Craftax exactly determined by the visible target cell, S1).
- **Correct.** Compare the current view with what E holds around each hypothesis (content -> location retrieval); weights
  are updated by how well each hypothesis predicts the view; hypotheses are re-centred, pruned or spawned (loop closure).
  Aliasing is negligible in Craftax (identical 7x9 view at another position in 0.03% of frames), so this path will be
  lightly exercised.
- **Uncertainty.** Hypothesis weights plus a per-hypothesis spread. Retrieval uses the spread (attention over keys within
  the spread), writes carry the location confidence.
- **No simulator coordinates anywhere.** The code is learned from the observed scroll (a direct self-supervised signal)
  and from the prediction losses that depend on correct addressing (§4).

### 2.5 Episodic store E and per-branch overlay O_b
- **Entries:** location key (relative-displacement encodable), content value (a per-cell content latent of width 64),
  time written, source tag (real | imagined), confidence. Place traces (§2.3) are entries of a second kind.
- **Writes (real).** Every visible cell is written at its location when first seen and whenever its content changes
  (terrain essentially never changes once seen: 0% for gaps <= 256 steps, S3). Entity traces on eviction / exit.
  OPEN: write every step vs only on change / surprise (EM-LLM, Titans) — "on change" is the starting choice because
  re-writing unchanged terrain adds nothing (S3).
- **Reads.** (a) By location: for each cell entering the (real or imagined) view, attend over entries near the cell's
  location under each self-location hypothesis, with a learned relative-displacement bias (2D RoPE-like). (b) By content:
  for relocalisation (§2.4). (c) For the actor: a summary read around the current location.
- **Overlay O_b (imagination).** Copy-on-write per imagined branch: reads check O_b first, then E; imagined writes
  (newly revealed content sampled from the prior; edits caused by imagined actions such as mining or placing) go to O_b
  only, tagged imagined; O_b is discarded with the branch and never merged into E (the source-monitoring principle).
  "Old entry -> trust" is therefore always "old entry unless the branch revised it".
- **Capacity.** Craftax worlds are 64x64: <= 4,096 cell entries per episode plus traces. No pruning in v0 (OPEN, later:
  biologically inspired pruning / consolidation).

### 2.6 Transition (prior) — afterstate + chance
- **Deterministic afterstate** from (W, R, L, action): body and counters, the agent's own move (gated by move success),
  interaction effects on the faced cell, record dynamics before chance.
- **Chance** (explicit, sampled): one chance code per record (moves, attacks, despawn), one for births (spawns), and
  content for never-seen cells entering view. Posterior chance is inferred from the real next frame during training;
  the prior is trained to match it (KL with free bits, Dreamer-style).
- **Entering cells.** If E/O_b holds the cell under the branch's location hypothesis: use the retrieved content (memory
  wins). Otherwise: sample content from a prior conditioned on the visible neighbours (S5: neighbours alone give ~77%)
  and write it to O_b, so the branch stays consistent if it returns ("sample once, persist", WHITEBOARD row 4).

### 2.7 Grounded heads
Reward and termination (from W and R), fovea next-glimpse pixels (from W at the attended window). Trained on posterior
*and* unrolled prior states, so imagined states are readable by construction (MuZero / TD-MPC2 / EfficientZero lesson;
the failure we hit three times).

### 2.8 "The model doesn't know"
An ensemble of M = 4 lightweight chance-prior and reward/termination heads with randomized prior functions (our July
finding: plain shared-trunk heads collapse to identical outputs). Disagreement (mutual information between heads)
separates model ignorance from chance (each head's own entropy). Used by imagination to truncate or penalise branches
(Dynamic Horizons). Known blind spot: confident long-horizon drift (Biased Dreams; our July data: AUC 0.73 -> 0.47 by
steps 12-16) — the research question, not solved by this design.

### 2.9 Actor and critic
Read-only access to W, R, a summary read of E at the current location (the same representations imagination uses,
MERLIN-style), with a gradient stop into the world. Policy prior = behaviour cloning on the expert episodes, read from
the same state. Trained in imagination: horizon 16 (the oracle showed h16 needed), one imagined future per starting
context, lambda-returns, KL/advantage-weighted regularisation to the BC prior (Dreamer 4's recipe).

---

## 3. Execution

**Posterior step (real frame).** encode -> self-location predict & correct -> record matching (same / new / nothing),
births, disappearance -> write changed cells and traces to E -> working-state update -> heads.

**Prior step (imagined).** afterstate from (W, R, L, a) -> sample chance per record, births -> move each location
hypothesis by predicted move success -> for entering cells read O_b then E, else sample from the prior and write to O_b
-> imagined edits to O_b -> heads; the ensemble's disagreement accumulates for horizon control.

---

## 4. Losses (what each teaches)

| | loss | teaches | applied to |
|---|---|---|---|
| L1 | reward (two-hot), termination (BCE) | grounding in native outcomes; relevance | posterior and unrolled prior states |
| L2 | fovea pixels: next-step pixels of the attended window | dense grounding where it matters; intended to counter entity erasure (R63 in our encoder study; SLIM, arXiv 2610.03137: self-prediction alone dropped entities) | posterior and prior |
| L3 | latent consistency: prior latent vs stop-gradient EMA target of the real next latent (cells, records) | dense dynamics signal | unrolled prior, k = 1..H_train |
| L4 | chance KL: posterior chance vs prior chance, free bits | explicit compact chance; imagined samples drawn from the right distribution | per record, births, entering cells |
| L5 | entering-cell content from memory: predict the latent of cells entering view, using reads from E | trains keys and values to be retrievable and *compatible* with the current reader; trains the location code (wrong address -> wrong content) | posterior and prior |
| L6 | record prediction: predict next-step entity evidence (existence, position, type) from R | object persistence, identity, position belief | posterior and prior |
| L7 | scroll / move success: predict whether the view shifts and by how much | self-motion | posterior |
| L8 | fovea policy: value of information — improvement of L1/L6 predictions from what the chosen glimpse reveals | attention to future consequences | posterior |
| L9 | ensemble heads: each head trained on L1/L4 targets with its own random prior | disagreement as a model-ignorance signal | posterior and prior |
| — | actor / critic losses | behaviour | imagination only, gradient-stopped |

OPEN: training unroll length H_train vs imagination horizon 16. Working systems train over 2-6 steps (TD-MPC2 3,
EfficientZero 5, JEPA-WM study: 2 best in simulation, 6 on DROID) while imagining longer; our earlier recursive training
improved distribution drift but worsened decisions (09-04). Start H_train = 4 with the full grounded heads on unrolled
states; measure readability at depths 1..16 in the integrated system.

OPEN: L8's estimator — (a) Gumbel-softmax over glimpse positions with gradients from L1/L6 (differentiable, cheap);
(b) offline counterfactual scoring of a few candidate glimpses per step (exact, costly). Start with (a).

---

## 5. Representation compatibility (Codex's point, resolved on paper)
- **Within a training window** (last 64 frames): writes and reads use the current parameters, gradients flow through
  retrieval (L5 trains compatibility directly).
- **History before the window**: rebuild E from the last 256 raw frames with the current parameters, no gradient
  (burn-in, R2D2). Cost: ~256 frame encodes per sample; acceptable for an 81-token encoder (estimate in §7). Beyond 256
  frames: not rebuilt in training (only 5% of terrain entering a 16-step window was last seen > 256 steps earlier, M2).
- **Context differences** (Codex: same parameters, different context): content values are per-cell latents read from
  the cell's own token after a shallow encoder (local content, less context-dependent); L5 trains the reader on values
  written under varied contexts within the window.
- **At deployment / imagination**: parameters are fixed, so no staleness.
- **Not resolved**: whether values written early in long episodes stay interpretable when rebuilt-memory training is
  capped at 256 frames. Measured in the integrated system (retrieval accuracy vs entry age).

---

## 6. Failure behaviours (specified)
| situation | behaviour |
|---|---|
| more entities than K | evict lowest e x relevance, write a trace to E; log overflow rate |
| entity appears with no matching record | birth (new record); a nearby place trace raises the "returning" prior |
| expected-visible record not seen | existence decays; after threshold, record released and traced |
| ambiguous identity | soft assignment in training; one sampled assignment per imagined branch |
| location ambiguous | multiple hypotheses, per-hypothesis retrieval, one hypothesis sampled per branch; never average content |
| cell never seen | sample from the neighbour-conditioned prior, write to overlay, persist within the branch |
| imagined action edits a remembered cell | overlay write; the branch sees the edit, E does not |
| predicted move wrong in imagination | the address drifts with it (coupled failure, Codex): measured as imagination inconsistency on revisits |
| model ignorance high | ensemble disagreement truncates / penalises the branch |

---

## 7. Compute budget at 6 GB (estimates to verify with a memory profile before training)
- Parameters: encoder ~0.8M (4 blocks, d 128); working state ~3M (256 wide, 4 Mamba blocks + spatial blocks); records,
  location, store reader, heads ~2M; ensemble heads ~0.5M. Total ~6-7M parameters (DRAMA-scale, which runs on 8 GB).
- Training step: batch 16 x (256 burn-in no-grad + 64 window with grad) frames; unroll H_train = 4 from 16 start points.
  Activations dominated by the 64-frame windowed working state: ~16 x 64 x 90 tokens x 256 x (layers ~8 x ~10 saved
  tensors) x 2 bytes ~ 2.4 GB; with gradient checkpointing on the spatial blocks ~1-1.5 GB. Store reads: ~90 queries
  x ~625 local entries per step, negligible.
- Imagination (actor phase): 16 starts x 16 steps, M = 1 branch each, overlays of <= ~200 entries per branch;
  world frozen (no activations kept), actor/critic small.
- These are order-of-magnitude estimates; the first implementation step is a memory profile, not training.

---

## 8. Evaluation (each with a separating control; no toy slices)
1. **Remembered-content fidelity**: re-entering cells predicted vs real, split by last-seen age; control: the same system
   with E disabled at test time (same weights) — isolates what retrieval contributes. Ceiling from S5.
2. **Imagination consistency**: in imagined branches that revisit cells, agreement between first and later imagined
   content (should be exact via O_b); and agreement with reality for cells known in E.
3. **Object persistence**: re-entry prediction of entities (existence, position, type) vs real, by absence length.
4. **Decision value**: actor improvement over its own BC in complete real episodes; the perfect-world ceiling with the
   same actor recipe (+1.4-1.6 achievements today); the September substitution corners (real states + model outcomes;
   model states + real outcomes) separate state errors from outcome errors.
5. **Comparison with a recurrent-only system** (option 1, same losses minus E/R/L): the natural control for "does the
   tiered memory add anything" — the user's call (no-unrequested-baselines rule).

---

## 9. Path to 3D (kept in mind, not built)
Location becomes an SE(3) pose belief (predict from odometry / actions, correct from view retrieval); relative encoding
over pose displacement; "expected visible" uses a view frustum instead of a 7x9 window; place traces and cell entries
become 3D surface or volume entries (VMem's surfels, WorldMem's pose-keyed frames are the closest precedents); the
fovea gains its input role (resolution). Object records are unchanged in kind.

---

## 10. Open decisions for the user
1. Training unroll length H_train (start 4) and how far heads are trained on unrolled states.
2. L8 estimator for the fovea policy (Gumbel-softmax first).
3. Write policy for E (on change first; surprise-gated later).
4. Whether to run the recurrent-only comparison (§8.5).
5. K and the hypothesis count are caps to be tuned by logged overflow / ambiguity in the integrated system.

## Known risks (not resolved by this design)
- Joint training may still erase entity content despite L2/L6 (R63 and SLIM evidence; L2 is our hypothesis, not proven).
- Coupled failure of motion prediction, addressing and dynamics in imagination.
- Confident long-horizon drift invisible to ensemble disagreement.
- The actor recipe itself caps the decision value (perfect-world ceiling +1.4-1.6 achievements, no survival gain).
