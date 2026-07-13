# Step 1 protocol (pre-registered): faithful same-frame I-JEPA representation

Committed BEFORE the first run, per the consensus matrix (re-audit §"Minimal next
experiment matrix", step 1) and the concurrence addition that pass thresholds be
numeric and fixed in advance.

## Objective under test

Same-frame I-JEPA, ported from the pinned source
(`facebookresearch__ijepa @ 52c1ae9`, `src/masks/multiblock.py`,
`src/models/vision_transformer.py::VisionTransformerPredictor`,
`src/train.py::forward_target/forward_context/loss_fn`, config
`in1k_vith14_ep300.yaml`), **isolated from action prediction** (no temporal
model, no task heads, no next-frame term).

Faithful elements:
- 4 target blocks/image, scale 0.15–0.2 each, aspect 0.75–1.5; one context
  region, scale 0.85–1.0, minus target overlap (`allow_overlap=False`);
  per-batch uniform index counts via the official min-trim; `min_keep=4`
  (scaled from 10 at 16×16-grid to our 8×8 grid).
- Context encoder processes ONLY visible tokens (token dropping, not
  mask-token substitution).
- Predictor: fixed 2D-sincos positional embeddings; learned mask token +
  target-position embedding as queries; per-target-block prediction passes
  (context repeated per block, as in the official forward); post-norm +
  projection.
- Targets: EMA encoder on the full frame, **LayerNorm over the feature dim**,
  gathered at target positions; loss = smooth L1 at masked positions only.
- EMA decay 0.996.

Labelled deviations (each with reason):
1. Conv stem instead of ViT patchify → masked-patch pixels leak into visible
   tokens through the receptive field. Direction of risk: makes the pretext
   task *easier* (weaker representation), cannot manufacture a metric pass.
   Fallback if gates fail: CNN-JEPA-style masked convolutions (SparK), to be
   pinned from its official repo before implementing.
2. 2 register tokens ride along in the context pass (needed downstream for
   HUD/inventory); they receive no positional queries and no loss.
3. Predictor depth 2, width d=64, no dim narrowing (theirs: depth 6+, 384 on
   768) — scale-appropriate.
4. Optimizer: AdamW lr 3e-4, wd 1e-4, no warmup/cosine, bf16 autocast — kept
   identical to every previous arm so cross-arm rows remain comparable
   (official uses lr 1e-3 + schedules at batch 2048).
5. No VICReg/SIGReg term in this arm: the experiment tests whether the true
   I-JEPA asymmetry alone prevents collapse at our scale. (The streamwise
   term exists in the hybrid control arm.)

## Arms (identical seeded data; replay RNG `np.random.default_rng(2027)`;
torch seed 101; training frames from the shared cache; probe set = seed-2
collect(400))

- `ijepa300`: objective above, 300 updates, batch 64 frames.
- `hybrid300`: current compact objective (streamwise-VICReg temporal hybrid,
  unmasked, LossConfig defaults, rollout off), 300 updates, batch 4×T16 (its
  native shape) — the incumbent control.
- `untrained`: same init, 0 updates — the reference row for every gate.

## Pre-registered gates (final row vs the SAME-RUN untrained row; measured with
`verification/representation_control.py` corrected functions on the held-out
probe set)

| gate | metric | pass bar |
|---|---|---|
| G1a | `target_observation_variance_fraction` | ≥ untrained − 0.05 |
| G1b | `target_same_stream_unrelated_cosine` | ≥ 0.80 × untrained |
| G2a | `target_stream_effective_rank_mean` | ≥ 0.90 × untrained |
| G2b | `target_patch_pool_covariance_rank` | ≥ 0.90 × untrained |
| G3 | semantic probe test accuracy (sane per train/test majorities) | ≥ untrained − 0.02 |
| G4 | inventory ridge R² (corrected split/standardization) | ≥ untrained − 0.02 |
| G5 | SSL loss, last-100 mean vs first-100 mean | ≤ 0.70 × first (the arm must actually learn; G1–G4 alone are passable by doing nothing) |

Curve logging every 25 updates: `target_observation_variance_fraction`,
`target_stream_effective_rank_mean`, loss. Abort rule: observation variance
fraction < 0.30 at any checkpoint (untrained ≈ 0.43) → stop, record failure.

Decision rule: pass all G1–G5 → step 2 (extend the SAME arm to 500 then 1000
updates; gate = no late reversal of G1–G4 and loss still ≤ its 300-update
value). Fail → CNN-JEPA masked-conv fallback (deviation 1) or official
LeJEPA/SIGReg image-level objective; no other knobs turned first.

Step 3/4 (frozen-encoder temporal + backend attribution) follow the matrix as
written in the re-audit; their thresholds will be pre-registered in their own
protocol file after step 2 fixes the encoder.

## Amendment 1c (pre-registered before the 1c run; after v1/v2 failures)

- v1 (token-dropping I-JEPA): failed G2a/G2b/G3 — conv-stem leak (deviation 1)
  made the task trivial (loss →0.009). As pre-registered, fallback applied.
- v2 (leak-free SparK/CNN-JEPA sparse stem, pinned b7b246c): G3 recovered;
  G2a/G2b still fail — I-JEPA asymmetry alone does not preserve per-stream
  diversity at this scale. Protocol fallback #2 applies.
- **Arm 1c `lejepa`**: identical to v2 plus SIGReg, the published LeJEPA
  composition (pinned rbalestr-lab/lejepa @ c293d29, MINIMAL.md):
  `loss = (1-λ)·L_ijepa + λ·L_SIGReg`, λ = 0.02 (their example default; official
  grid 0.01-0.1). SIGReg ported verbatim (17-knot quadrature on [0,3], 256
  fresh random unit projections, ECF match to standard Gaussian, ×N scaling)
  with two labelled adaptations: (a) sample axis = observations per stream
  (the corrected axis semantics; LeJEPA's axis is the image batch), statistic
  averaged over streams; (b) applied directly to a dense online-encoder pass
  with NO projector — the 2026-07-13 re-audit measured that a projector hides
  encoder collapse (its VICReg-projector arm: flat rank 3.5), and the gates
  measure encoder tokens.
- Gates G1-G5 unchanged. Same data, seeds, budget.

## Amendment 1d (pre-registered before the 1d runs)

1c result: G1a/G1b/G2a/G2b/G3/G5 PASS; G4 FAILS (registers 0.809 vs bar 0.835)
— and supplementary analysis shows the loss of inventory information is real,
not a probe-carrier artifact (HUD-row tokens 0.911→0.804, all-pool 0.907→0.796).
No gate is amended. Observation for the record: every trained arm across both
audits fails G4 (hybrid 0.728; re-audit corrected-VICReg 0.811; lejepa 0.809 vs
untrained 0.855-0.87), so G4-as-constructed may be unpassable for lossy
training at these budgets; this is to be settled with data, not by softening.

**1d grid (all cells reported, no post-hoc selection):**
λ ∈ {0.01, 0.02} × budget ∈ {300, 1000}; the (0.02, 300) cell is the existing
1c run. λ stays inside the official LeJEPA grid. Curve logging gains the
inventory R² every 50 updates to show whether retention recovers with budget.
Gates G1-G5 unchanged. Decision rule:
- any cell passes all gates → it advances to step 2 as-is;
- no cell passes G4 but ≥1 cell passes all others AND its G4 trajectory is
  flat/recovering ≥0.80 → G4 recalibration becomes a CONSENSUS question
  (user + implementation agent), documented with all four cells; no
  unilateral change.

## 1d grid results (all cells; no post-hoc selection)

| cell | G-status | inventory (bar 0.835) | semantic | obs-frac | stream rank | loss first→last |
|---|---|---|---|---|---|---|
| λ=0.02, 300 | all pass except G4 | 0.809 | 0.915 | 0.781 | 4.06 | 0.405→0.272 |
| λ=0.01, 300 | G4 miss by 0.005; G5 miss (0.73 vs 0.70) | 0.830 | 0.906 | 0.789 | 4.23 | 0.294→0.215 |
| λ=0.02, 1000 | G4, G5 | 0.684 | 0.918 | 0.860 | 8.47 | 0.405→0.347 |
| λ=0.01, 1000 | G4, G5 | 0.788 | 0.929 | 0.834 | 7.87 | 0.294→0.302 |

Readings (documented for consensus per the 1d decision rule):
1. The LeJEPA family robustly passes the structural block in every cell
   (G1/G2/G3), with semantic accuracy IMPROVING past untrained at 1000 updates
   (0.918-0.929 vs 0.892) — first arm family in either audit to do so.
2. **G4's pass margin is inside its own measurement noise**: bootstrap over
   30 random 300-frame probe subsets gives untrained 0.801 sd 0.036
   [0.753, 0.842], λ=0.01@300 arm 0.779 sd 0.028 [0.737, 0.820]. The ±0.02
   tolerance and the 0.005 best-arm miss are both smaller than one subset sd.
   The full-set untrained point (0.855) sits at the top of its own subset
   distribution. G4 as constructed is noise-dominated at the decision margin.
   (A real budget trend also exists: λ=0.02@1000 falls to 0.684 — the
   gaussianization/retention trade-off is real, and λ=0.01 mitigates it.)
3. **G5 mismeasures learning under SIGReg at longer budgets**: total loss
   RISES (0.27→0.35 for λ=0.02) because the regularizer makes the pretext
   harder as it succeeds — while semantic, stream rank, and observation
   fraction all improve. G5's intent (exclude do-nothing encoders) is served
   by any of those three improving; its letter is not.

## Consensus question (user + implementation agent; pre-registered branch)

Proposal:
- G4 → paired-with-uncertainty form: paired bootstrap over shared probe
  subsets; pass = (untrained − trained) mean difference ≤ 0.02 with its 90% CI
  reported, or trained ≥ 0.80 absolute.
- G5 → composite learning criterion: any of {loss ↓≥30%, semantic > untrained,
  stream-rank ≥ 1.1× untrained}.
- Independent of recalibration: advance the λ=0.01@300 encoder (checkpoint
  ssl_step1_lejepa001_d300_l001.pt) to matrix step 3 — the frozen-encoder
  prediction gate and Phase E reward-calibration measure directly what G4
  proxies; step 3 does not consume the G4 verdict.

## Amendment 1e (consensus corrections; pre-registered before the 1e rerun)

Per the implementation agent's review (all findings verified and reproduced by
the senior reviewer: encoder positions confirmed at vision_transformer.py:401;
batch-shared block sizes at multiblock.py:128-135; non-rectangular masks 22-41%
across 5 seeds; paired G4 degradation 0.0276 mean, 41.7% of paired subsets
<= 0.02, UCB90 0.068):

Fidelity corrections (implemented, tested):
- Encoder gains fixed 2D-sincos positional embeddings added BEFORE token
  dropping/masking, matching the official encoder; sparse path matches dense.
- Mask sampler: ONE pred-block size and ONE context size per batch (official
  collator); rectangularity + batch-shared-size regression test added.

Instrumentation corrections:
- Prediction and SIGReg components logged separately.
- Held-out pretext bank: fixed mask sets on 128 fixed held-out frames,
  evaluated every 50 updates against the current EMA target.
- SIGReg diagnostic on fixed evaluation projections (pre-drawn A).
- Checkpoints save the full pretrainer, optimizer, config, NumPy/torch RNG
  states, and component histories.

Redesigned gates (replacing G4/G5; G1-G3 unchanged):
- **G4' (episode-blocked paired non-inferiority):** probe stream split into 8
  contiguous 50-frame blocks; 200 block-bootstrap resamples; paired
  degradation (untrained R2 − trained R2) computed per resample on identical
  frames; PASS iff the one-sided 90% upper confidence bound <= 0.02. No
  absolute-R2 escape clause.
- **G5' (component learning):** held-out pretext-bank prediction loss at the
  final evaluation <= 0.70 x its first evaluation. SIGReg component is
  reported as a diagnostic, not gated. (Caveat on record: the EMA target
  moves; G1/G2 guard the degenerate route to a trivially easy bank.)

Rerun: λ=0.01, 300 updates, + untrained control only. No other arms, no other
knobs. Prior 1c/1d results remain on file as evidence from the less faithful
implementation; they do not carry forward.

### 1e abort-rule correction (pre-registered before the retry)

The first 1e run aborted at step 25: sincos positions are position-constant
vectors, which mechanically inflate position variance and halve the UNTRAINED
observation-variance fraction (0.426 pre-positions → 0.202 with positions). The
absolute abort constant (0.30) silently changed meaning under the architecture
correction — the arm was improving (0.202 → 0.281, rising) when it fired.
Correction, preserving the rule's intent (catch encoders LOSING observation
sensitivity relative to their own start): abort iff
`obs_frac < max(0.6 × untrained_obs_frac, untrained_obs_frac − 0.10)`.
All gate thresholds were already untrained-relative; the abort rule now is too.
Per-stream rank/cosine gates are unaffected by the position constant (stream-
mean centering removes it). No other change; rerun λ=0.01@300 + untrained.

### 1e results and pre-registered 1f (final step-1 run)

1e (λ=0.01@300, corrected implementation): G1a/G1b/G2b/G3/G5' PASS
(semantic 0.893→0.923; held-out pretext bank −38%; obs-frac 0.20→0.70).
G2a misses by ~10% (3.32 vs bar 3.69). G4' point estimate improves 3× over the
pre-correction arm (paired degradation 0.0098 vs 0.0276) but UCB90 = 0.066
fails the ≤0.02 rule — the 8-block bootstrap is noise-dominated (blocks are
50-frame chunks of a single seed-2 stream). Component logging shows the total-
loss decline is mostly SIGReg; the held-out bank, not the training component,
carries the learning evidence — vindicating the component-logging requirement.

**1f (pre-registered):** λ=0.01, 1000 updates (1d evidence: stream rank rises
strongly with budget under SIGReg — 4.2→7.9 at λ=0.01), probe enlarged to
3 seeds × 400 frames (seeds 2, 5, 6; 24 blocks of 50) for G4' precision —
an instrument improvement, not a bar change; every gate and threshold
unchanged. Untrained baseline recomputed on the same enlarged probe within the
same run. Pass = all of G1a/G1b/G2a/G2b/G3/G4'/G5' → step-1 PASS; encoder
advances to matrix step 2/3. Any failure → report and stop for consensus.

### 1f result: 6/7 pass; G4' fails decisively; STOP per decision rule

λ=0.01 @ 1000 updates, 3-seed 24-block probe (untrained baselines recomputed on
the same probe):

| gate | result |
|---|---|
| G1a obs-variance fraction | PASS (0.189 → 0.952) |
| G1b same-stream unrelated | PASS (0.038 → 0.871) |
| G2a stream rank | PASS (4.57 → 11.75, 2.6× untrained) |
| G2b pool rank | PASS (1.40 → 9.04) |
| G3 semantic | PASS (0.894 → 0.932) |
| G4' blocked non-inferiority | **FAIL** (paired degradation mean 0.0372, UCB90 0.0749, 24 blocks — no longer instrument noise) |
| G5' held-out pretext bank | PASS (0.381 → 0.216, −43%) |

The budget trade-off reproduces with the corrected implementation and a precise
instrument: at 300 updates retention holds (G4' mean 0.0098) but rank has not
yet grown (G2a fail); at 1000 rank is excellent but HUD inventory erodes
(0.707 → 0.648 on the harder 3-seed probe). Component curves concur: the
training prediction component RISES (0.154 → 0.230) as SIGReg raises pretext
difficulty while held-out prediction improves — the regularizer and retention
pull against each other on low-entropy HUD digit patches.

**Stopped for consensus** (pre-registered rule). Candidate resolutions, in
preference order, NOT enacted:
1. Spatially scoped SIGReg: regularize world-view tokens (rows 0-5) only,
   exempting HUD rows and registers — mechanistically motivated (the HUD is UI
   state, not world appearance; gaussianizing digit patches is precisely what
   destroys counts), one run to test, all gates unchanged.
2. Intermediate budget (~500) accepting partial rank growth — rejected as
   forking-paths budget-tuning unless (1) fails.
3. Advance under explicit G4' waiver with step-3/Phase-E downstream
   arbitration — requires both-agent + user consensus.

## Amendment 1g (pre-registered before the 1g runs)

User challenge accepted: spatially scoped SIGReg risks Crafter-specific
hardcoding, and the FAITHFUL LeJEPA axis must be tested first-class. Verified
before registering: (a) github.com/galilai-group/lejepa and the pinned
rbalestr-lab/lejepa are the SAME repository (identical HEAD c293d29 — org
rename); (b) the paper (2511.08544v2, Eq. "LeJEPA", §5-6) applies SIGReg to
GLOBAL per-view embeddings over the image-batch axis, DINO-style multi-view,
with a projector — never to dense patch tokens; §6.4/Table 4 shows SIGReg
replaces the predictor/EMA as the anti-collapse mechanism.

Pre-run repairs (companion consensus, all landed + tested):
- G4' redesigned: paired degradation per independent probe seed (fixed
  non-overlapping internal splits), bootstrap over seeds; 5 probe seeds
  (2,5,6,7,8) x 400 frames; bar unchanged at 0.02 UCB90.
- Checkpoints now save torch CPU/CUDA RNG, mask-generator state, and the full
  ModelConfig (previously NumPy state only).
- Mask-size helper corrected to the exact official form: one shared uniform
  draw for scale AND aspect, linear aspect interpolation, official clamping.

Arms (λ=0.01, 1000 updates, identical data/seeds; all other gates unchanged):
- **A `lejepa_global` (faithful axis):** SIGReg on projector(pooled dense
  tokens), samples = image batch — the official application point, projector
  shape scaled from MINIMAL.md. Labelled composition: the predictive term
  remains I-JEPA masked prediction (our downstream needs dense tokens; the
  paper's multi-crop view-consistency term is a Crafter-augmentation minefield
  deferred on purpose). Gates never read the projector.
- **B `lejepa_scoped` (mechanistic ablation):** per-stream SIGReg on world
  rows 0-5 only (48 tokens); described as removal of DIRECT SIGReg pressure —
  post-attention world tokens still depend on HUD tokens indirectly.
- Untrained control recomputed in-run on the 5-seed probe.

Advancement rule: an arm advances only on a full G1-G3 + G4' + G5' pass. If
BOTH pass, the faithful-global arm advances (less adaptation). If neither
passes, stop for consensus; the 500-update feasibility probe stays deferred.

## 1g results — STEP 1 PASSED by the faithful-global arm

| gate | A `lejepa_global` (faithful axis) | B `lejepa_scoped` |
|---|---|---|
| G1a obs-variance fraction | PASS (0.184 → 0.576) | PASS (→ 0.934) |
| G1b same-stream unrelated | PASS | PASS |
| G2a stream rank | PASS (4.30 → 5.42) | PASS (→ 13.18) |
| G2b pool rank | PASS (1.40 → 6.00) | PASS (→ 9.12) |
| G3 semantic | PASS (0.893 → 0.910) | PASS (→ 0.930) |
| G4' per-seed non-inferiority | **PASS** (per-seed {+0.031, −0.025, −0.006, −0.013, −0.049}, mean −0.013, UCB90 0.0035) | FAIL (mean 0.021, UCB 0.0295; 3 of 5 seeds > 0.03) |
| G5' held-out pretext bank | **PASS** (0.380 → 0.076, −80%; training prediction component also falls 0.118 → 0.067) | FAIL (−18%; prediction component RISES 0.155 → 0.279) |

Late-curve stability (step-2 condition): obs-frac plateaued, stream rank gently
rising, curve-subset inventory STABLE 0.899→0.910 over the last 250 updates,
bank flat at floor. No late reversal.

Readings:
1. **The user's faithfulness challenge was correct and decisive.** Global
   SIGReg through the projector (the official application point) prevents
   dense-token collapse — the token-level gates pass — WITHOUT gaussianizing
   individual patch tokens, so HUD information survives (G4' passes with the
   trained encoder non-inferior, on average better, than untrained). The
   Crafter-specific scoped variant not only loses on adaptation grounds, it
   fails its own mechanistic promise: indirect pressure through attention still
   erodes HUD (3 of 5 seeds degrade > 0.03), exactly the caveat in the
   companion's conditional endorsement.
2. Unlike every per-stream variant, the global arm's pretext does not fight the
   regularizer: prediction difficulty falls alongside SIGReg.
3. Winner checkpoint: `reviews/artifacts/ssl_step1_lejepa_global_g1000.pt`
   (full state: pretrainer, optimizer, torch/numpy/mask RNG, config,
   component histories).

Per the pre-registered advancement rule, `lejepa_global` @ 1000 updates is the
step-1 encoder. Companion verification of this run is requested before step 3
formally launches; the step-3 protocol (frozen encoder; rollout 0 vs 1;
3 seeds; GRU; paired window-bootstrap copy margin) will be pre-registered in
its own file.

### Post-1g implementation verification and consolidation

Companion verification reproduced the saved G4' mean/UCB, strictly loaded the
winner checkpoint, and confirmed numerical equality between its global SIGReg
path and `MINIMAL.md @ c293d291` under identical random directions. The
operative implementation is therefore global-only: the historical dense-stream
and Crafter-row-scoped modes were removed from executable code, while their
artifacts above remain as negative evidence.

One shadow-only instrumentation mismatch was found: the recorded 1g
`heldout_sigreg_fixedproj` series evaluated dense per-stream tokens, not the
global projector output. It was not a gate and does not affect the step-1
verdict; the gated held-out pretext series and the training SIGReg component are
correct. Future runs evaluate the shadow SIGReg diagnostic on globally pooled,
projected embeddings with fixed random directions and the projector in eval
mode, so held-out evaluation cannot mutate BatchNorm state.
