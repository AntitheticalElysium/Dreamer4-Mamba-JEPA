# Consensus re-audit: reviewer claims, corrections, and stop decision

Date: 2026-07-13 (Australia/Sydney)  
Scope: every report and artifact under `reviews/`, the compact and reference
implementations, pinned primary papers/source repositories, and the actual RTX
3060 Laptop GPU (5794.625 MiB reported by PyTorch).  
Decision: **stop before Phase D efficacy, reliability, or policy training.**

## Executive verdict

The senior review caught real faults and its central architecture diagnosis is
right: the original representation objective collapsed, the original semantic
probe was spatially invalid, the per-token future MLP lacked the cross-token
path needed for view shifts, and teacher-forced training lacked an explicit
bridge to closed-loop imagination.

Several later conclusions do **not** survive independent verification:

1. The VICReg change did not establish non-collapse. It flattened token
   positions into the sample axis, so fixed position codes could satisfy the
   regularizer and produce high rank without depending on the observation.
2. The Phase B/D/rollout arms were not matched seeded experiments.
   `EpisodeReplay.sample()` used the unseeded global NumPy RNG; training and
   held-out windows differed across arms.
3. “Changed tokens” were selected from each model's own latent drift. The
   selected locations and difficulty therefore changed with the encoder.
4. GRU and Mamba-2 trained different encoders. Absolute cosine errors in their
   different target spaces do not establish backend superiority.
5. The claimed 7.5× Mamba-2 recurrent speedup is a cold-kernel timing artifact.
   The GRU arm paid one-time attention compilation while Mamba-2 was already
   warm. Warm, order-reversed measurement makes GRU faster in recurrent
   deployment.
6. The rollout experiment did not rerun representation gates, used a weak
   “any positive mean difference” copy criterion without uncertainty, and
   compared independently learned latent spaces. It supports the bridge as a
   mechanism, not the Mamba-2-default or D1-pass claims.

After repairing the controls, no tested representation/anti-collapse variant
passes non-collapse, semantic/inventory retention, and predictive fidelity
together. That is binding negative evidence. No long run was launched.

## Claim ledger

| reviewer claim | verdict | independent evidence |
|---|---|---|
| Original compact objective collapses | **confirmed** | Unregularized variants and the full objective lose rank; the source comparison also shows the compact objective is neither I-JEPA nor frozen-encoder V-JEPA-2-AC. |
| Original semantic probe is invalid | **confirmed** | `info["semantic"]` is the global map; Crafter renders a transposed local 9×7 world view plus HUD. The repaired crop matches pinned source geometry. |
| Copy-all-token is nearly unbeatable | **confirmed** | Consecutive Crafter frames are mostly unchanged. Binding changed-patch metrics now exclude registers and use fixed raw RGB patches. |
| VICReg fixes collapse at 4k updates | **refuted** | Reproduction: flattened covariance rank 11.9→42.0 while the observation share of variance fell 42.6%→10.8%; same-position unrelated cosine fell 0.124→0.059. A synthetic observation-independent codebook scores flat rank ≈64 with variance loss 0.007. |
| Cross-token predictor is required | **confirmed, source-qualified** | I-JEPA/V-JEPA predictors attend across tokens; the attention predictor materially reduced one-step error. The compact module remains an adaptation (learned positions, action/horizon tokens, separate temporal core), not “JEPA ground truth.” |
| Crafter does not justify mixtures | **provisional** | Typical one-step branch dispersion is small and the synthetic K=2 mechanics work. The environment probe used 75 live states/one seed, compared maxima over eight RNG reseeds, and omitted reward, continuation, achievements, and future hidden-state divergence; rare policy-induced branches remain unmeasured. Deterministic stays default, but the universal premise is not proved. |
| Rollout loss closes the imagination bridge | **mechanism confirmed; efficacy unproved** | V-JEPA-2-AC Eq. 3–4 supports a T=2 final-state bridge. The archived result improves its own metric, but the representation gate was false, windows were unseeded, changed locations were model-defined, and copy wins had no paired CI. |
| Mamba-2 is 7.5× faster recurrently | **refuted** | Warm full imagined step at B=48,S=66: GRU 1.44 ms, Mamba-2 1.91 ms. Isolated temporal step: 0.107 vs 0.857 ms. Results are stable under reversed benchmark order. |
| Mamba-2 is the validated default | **refuted/deferred** | Backend arms learned different target spaces. In the archived rollout result GRU is actually better relative to its own copy drift at k=8/16 (pred/copy 0.865/0.833 vs Mamba-2 0.949/0.912). A fair task comparison requires one frozen shared representation. |
| Reliability must remain shadow-only | **confirmed** | No held-out later-policy calibration exists. Weighting remains disabled. |
| Phase G policy remains blocked | **confirmed** | Phase B is not passed; task-head calibration on imagined states and Phase F have not run. |

## Severity-ranked audit

### Critical

1. **False anti-collapse certificate.** The code and rank diagnostic treated
   `[B,T,S,D]` as `[B*T*S,D]`. Position diversity was counted as sample
   diversity. Actual long-run measurements show the encoder becoming less
   observation-dependent while reported rank rises.
2. **Backend attribution is invalid.** Phase D co-trained different encoders and
   targets per backend, contrary to HANDOFF's same-representation requirement.
   Absolute latent errors and copy distances are not comparable.
3. **Rollout/D1 verdict is invalid as a gate.** The rollout objective changed the
   target representation but representation probes were not rerun. A co-trained
   encoder can reshape the metric/copy baseline. The “pass” required no margin,
   confidence interval, or seed replication.
4. **The representation design remains unresolved.** Corrected pilots find no
   anti-collapse port that preserves information and prediction simultaneously;
   see the controlled matrix below.

### High

5. **Experiments were not reproducibly matched.** Torch was seeded, NumPy replay
   sampling was not. Evaluation batches also differed between the untrained and
   final rows and between arms.
6. **Mamba timing was asymmetrically cold.** In the old Phase D path GRU loaded a
   checkpoint and entered `openloop_eval()` without ever calling the future
   predictor; Mamba-2 had just trained it. A first B=48 imagined step costs
   roughly 164 ms (GRU) or 329 ms (Mamba-2), versus warm 1.4/1.9 ms. Averaging
   that compile only into GRU creates the reported speedup.
7. **Changed-token selection was endogenous.** Top-quartile target-latent copy
   distance chooses different patches for different models and can be gamed by
   the representation. Registers were also mixed with spatial tokens.
8. **VICReg was materially misported.** Primary VICReg uses batches of image
   vectors, a disposable expander, unnormalized MSE invariance, and variance/
   covariance on both branches. The compact port used dense positions as
   samples, no expander, cosine invariance, and only online tokens. The 25:1
   coefficient ratio does not transfer the absolute scale.
9. **The inventory gate was scale- and split-sensitive.** A chronological split,
   unstandardized features, and a penalized intercept made R² depend on encoder
   scale and trajectory drift. Corrected random fixed splits and standardized
   ridge change the baseline from negative/noisy to about 0.87 on the same data.

### Moderate

10. `semantic_probe_sane` compared training accuracy to the **test** majority;
    it could not detect the original held-out failure. It now checks train and
    test against their own majority baselines.
11. `multi_block_mask(ratio=0)` masked 4–36 patches (mean 16.1/64), because the
    first random block overshot `wanted=1`. The full model happened to bypass the
    helper, but `RepresentationControl` did not. Zero now returns an all-false
    mask and the harness bypasses masking explicitly.
12. The 0.6 multi-block mask actually masks a mean 41.9/64 tokens, not exactly
    38.4. More importantly, replacing post-convolution tokens leaks masked-patch
    information through the convolutional receptive field and is not I-JEPA
    token dropping/query prediction.
13. Phase artifacts referenced missing `/tmp/.../*.pt` checkpoints; the objective
    variant/full-rank scripts were not archived; `crafter_branch_latents.py`
    cannot reproduce from the committed tree. Historical JSON remains evidence,
    not a fully reproducible experiment.
14. The Crafter branch probe reports “any of eight branches differs from branch
    zero,” which grows with branch count and is not a transition probability. It
    reseeds hidden RNG states without conditioning on history and omits direct
    reward/termination/achievement divergence.
15. The MoP latent probe pools only register tokens even though the compact
    mixture assigns over all 66 dense tokens; temporally shifted frames are used
    as “unrelated” pairs. Its scale conclusion is suggestive, not binding.

## Primary-source and commit table

All listed clones are clean and their checked-out HEAD matches
`third_party/SOURCES.lock`.

| source | exact commit | audit use |
|---|---|---|
| fmi-basel/Dreamer-CDP | `a851fa3e3d70b624b094ee1810ad4bb602346092` | action/reward timing, observation/imagination bridge, CDP target |
| facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` | EMA/frozen targets, AC predictor, teacher forcing + rollout source |
| facebookresearch/ijepa | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | true masked context/target queries, target LN, predictor positions/attention |
| facebookresearch/vicreg | `4e12602fd495af83efd1631fbe82523e6db092e0` | official image-batch/projector/MSE variance-covariance implementation |
| rbalestr-lab/lejepa | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | SIGReg sample axis and projector/global embedding practice |
| state-spaces/mamba | `f577286d052741c35d39cd43bdc3fad27120f22c` | exact Mamba-2/3 constructors, step methods, cache tensors |
| realwenlongwang/Drama | `a50bd54c34e77d1d13e988a031733a47817098e2` | Mamba world-model/replay/layout baseline |
| danijar/crafter | `e04542a2159f1aad3d4c5ad52e8185717380ee3a` | API, render geometry, semantic map, discount, official score |
| edwhu/dreamer4-jax | `8144b940d801971f12ec5633553b95001e555949` | unofficial Dreamer 4 reproduction |
| nicklashansen/dreamer4 | `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6` | unofficial Dreamer 4 reproduction |
| lucas-maes/le-wm | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | reconstruction-free joint embedding/SIGReg reference |
| NM512/r2dreamer | `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f` | decoder-free baseline |
| corl-team/nedreamer | `11cd3a978b83743f795cbfa81c2e095344912c17` | next-embedding prediction baseline |
| nicklashansen/mmbench2 | `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9` | hallucination benchmark/source caveats |
| leor-c/horizon-imagination | `c79ec5e2450be22711c7d717e49326edf77061f2` | efficient diffusion rollout schedules |

JEDI and MoP-JEPA have no verified official code release in the pinned audit;
their arXiv source archives are retained. Dreamer 4 has no canonical official
implementation in this source set. Paper snapshots and SHA-256 values are in
`third_party/PAPERS.lock`, including the repaired, genuine 62-page AMI v0.9.2
PDF (the previous file was an HTML error page).

### Source conclusions that constrain the architecture

- V-JEPA-2-AC freezes its visual encoder during action-conditioned post-training,
  uses L1, interleaves action/proprioception/frame tokens with block-causal
  attention, and adds a T=2 final-state rollout loss. The compact bridge is an
  explicit adaptation, not a verbatim port.
- VICReg defines samples as `n` image embeddings and applies a projector plus
  unnormalized MSE invariance. Flattening spatial positions is unsupported.
- LeJEPA SIGReg likewise preserves the image-batch axis; it does not justify
  treating time/token positions as independent observations.
- Mamba-3's official cache is four tensors
  `(angle_dt_state fp32, ssm_state fp32, k_state, v_state)` and its source says
  it was only tested on H100. No approximate recurrent state is used.
- Mamba-2's cache is `(conv_state, ssm_state)` and its exact sequence/step path
  passes on this GPU, including the production fp32-cache + bf16-autocast case.
- “On Training in Imagination” assumes deterministic dynamics and global
  Lipschitz/contraction conditions; it supplies diagnostics/bounds, not a safety
  certificate or an uncertainty signal.

## Corrected representation controls and negative matrix

The binding gate now uses per-stream covariance rank, patch/register pooled
rank, fixed-stream variance, observation-vs-position variance, same-stream
unrelated distance, fixed raw-RGB changed patches, semantic retention, and
scale-invariant inventory R². Flat rank is logged but never certifies a pass.

| arm (500 updates unless noted) | key outcome | binding result |
|---|---|---|
| reviewer flattened VICReg, 4000 | flat rank 11.9→42.0; observation variance share 42.6%→10.8% | **false pass / refuted** |
| corrected streamwise raw VICReg 1/.04, 500 | observation diversity retained; prediction still worse than copy; inventory 0.869→0.811 | P1/P3 pass, P2/P4 fail |
| same, 1000 | one-step changed error worsens to 0.131 vs copy 0.060; inventory 0.762 | P2/P4 fail; do not scale |
| disposable VICReg projector | flat rank ≈3.5, stream rank ≈2.6 | P1 fail; projector hides encoder collapse |
| native-scale gamma=0.2 | same-stream unrelated cosine 0.140→0.017; semantic 0.896→0.854 | P1/P3/P4 fail; norm/direction cheating |
| directional normalization 1/.04 | diversity retained but changed prediction 0.203 vs copy 0.068; inventory degrades | P1/P3 pass, P2/P4 fail |
| directional .25/.01 | stream rank ≈2.0 | P1/P2/P4 fail |
| directional .25/.04 | stream rank ≈2.3; semantic degrades | P1/P2/P3/P4 fail |
| masked streamwise control | prediction 0.532 vs copy 0.322; ~305 MiB peak | P2/P4 fail; worse and costlier than unmasked |

The pilot harness evolved while the false certificate was being isolated, so
embedded `criteria` booleans in early `phase_b_v2_*.json` files are not uniform.
Those JSON files are retained as raw provenance; the matrix above reapplies the
final observation-sensitive gate to their logged measurements and does not use
the older booleans as evidence of a pass.

No coefficient or source-shaped variation passed all four gates. Continuing to
tune this hybrid would violate the smallest-discriminating-experiment rule.

## Corrections implemented

These changes are engineering corrections and instrumentation; they do **not**
declare the architecture validated.

- Replay accepts an explicit seeded NumPy generator; tests prove repeatability.
- `mask_ratio=0` really disables masking in both the helper and representation
  harness. Unmasked is the model default; masking remains an ablation.
- Anti-collapse statistics are streamwise across observations. Masked prediction
  paths use a separate dense online pass so random mask patterns cannot satisfy
  the auxiliary.
- Model metrics distinguish flat rank from pooled rank, fixed-stream variance,
  position variance, and observation-variance fraction.
- Changed spatial tokens use per-transition raw RGB patch change and exclude
  register tokens.
- Semantic probe sanity uses train/test majority baselines separately.
- Inventory ridge uses a fixed random split, train standardization, and an
  unpenalized centered intercept.
- Two-step rollout support is in the core model behind `LossConfig.rollout=0`.
  The prefix encoder is detached; gradients cross predictor → temporal →
  predictor. It is tested on GRU and Mamba-2.
- Open-loop evaluation now reports window-level paired bootstrap intervals and
  requires both a positive 95% lower bound and ≥5% relative improvement.
- Recurrent latency warms the exact deployment shape and uses 400 timed CUDA-
  event steps. The invalid Phase D/rollout v1 mains now fail closed with an
  explanation instead of silently regenerating misleading evidence.
- Architecture spec, source lock, paper hash lock, and exact audited dependency
  lock were updated.

## Tests and smoke checks

- `.venv/bin/python -m pytest m3_hjwm_compact/tests -q`: **44 passed**,
  including the CUDA Mamba-2 sequence/step, fp32-cache, mixed-precision, and
  rollout-gradient paths.
- `.venv/bin/python m3_hjwm_compact/smoke_test.py`: **passed**.
- `PYTHONPATH=m3_hjwm .venv/bin/python m3_hjwm/tests/smoke_test.py`:
  **passed** (reference implementation remains separately runnable).
- `python -m compileall -q m3_hjwm_compact reviews/artifacts`: **passed**.
- Added regressions for position-codebook false rank, mask-randomness leakage,
  raw changed patches, Crafter local-view geometry, inventory scale invariance,
  explicit replay RNG, rollout indexing/final target, rollout gradient path,
  rollout loss decrease, Mamba-2 rollout gradients, and fp32 deployment cache
  equivalence.

## Measured GPU memory by phase

Warm steady-state, bf16 autocast, B=4, T=16, 66 streams, d=64;
imagination B=32,H=8. Values are PyTorch peak allocated / peak reserved.

| phase | GRU | Mamba-2 |
|---|---:|---:|
| unmasked world update, rollout off | 189.3 / 226 MiB | 378.0 / 416 MiB |
| unmasked world update, T=2 rollout on | 204.1 / 242 MiB | 424.8 / 522 MiB |
| masked GRU world update, dense anti-collapse auxiliary | 300.6 / 316 MiB | not run (masked already rejected) |
| actor/critic update, H=8 | 38.6 / 68 MiB | 84.7 / 104 MiB |

Memory is not the blocker. The integrated rollout reuses/detaches the existing
prefix graph and is far below the archived scratch runner's 722 MiB Mamba peak.

## Temporal performance re-audit

At training batch B=4,T=16,S=66,d=64:

| metric | GRU | Mamba-2 |
|---|---:|---:|
| sequence inference median | 1.732 ms | 1.348 ms |
| one forward/backward temporal step | 7.425 ms | 9.616 ms |
| recurrent temporal step | 0.107 ms | 0.562 ms |
| bf16 recurrent cache | 0.032 MiB | 2.449 MiB |

At deployment batch B=48,S=66, after exact-shape warmup:

| metric | GRU | Mamba-2 |
|---|---:|---:|
| full imagined step median | 1.44 ms | 1.91 ms |
| isolated temporal step median | 0.107 ms | 0.857 ms |

Mamba-2 can make the parallel sequence path faster at small training batch and
the full measured world update was 30.3 ms vs GRU 34.3 ms (rollout off), but it
is slower recurrently and uses much more state/memory. This is mixed engineering
evidence, not evidence of better learned dynamics.

## Minimal next experiment matrix

Do not add more components to the current hybrid. The next researcher should
choose one representation lineage and keep all downstream modules frozen while
testing it.

| order | smallest discriminating experiment | pass requirement |
|---:|---|---|
| 1 | Reproduce a **true** same-frame I-JEPA-style context/token-dropping + target-query objective, or an official LeJEPA/SIGReg image-level objective, separately from action prediction. Include the untrained and current hybrid controls. 300 updates first. | All observation-sensitive rank/variance metrics above abort thresholds; held-out semantic and inventory no worse than untrained by >0.02. |
| 2 | If step 1 passes, extend only that arm to 500 then 1k updates. | No late reversal; information metrics and raw-patch one-step prediction improve together. |
| 3 | Freeze the one passing encoder/EMA target. Train deterministic temporal predictor with rollout weight 0 vs 1, identical replay index schedule, three seeds, GRU only. | Paired window-bootstrap copy margin lower CI >0 and ≥5% at a predeclared k≤8; one-step/semantic/inventory not regressed. |
| 4 | With that exact frozen representation and predictor protocol, compare GRU vs Mamba-2, same initial non-temporal weights and replay indices. | Task error advantage with CI, not just throughput; report cache/VRAM. |
| 5 | Expand Crafter branching probe to reward, continuation, achievements, health/inventory, multiple seeds, and policy/combat states. | Only enable K=2 if consequential branch mass/separation is reproducible. |
| 6 | Phase E on imagined states from real prefixes, then three-seed replication, then Phase F shadow calibration. | Predeclared calibration/error thresholds. |

Dreamer-CDP must remain a separately runnable baseline throughout; adopting its
RSSM/KL bridge into the core would change the project's identity and attribution.

## Explicit decisions

| decision | status | reason |
|---|---|---|
| Mamba-3 | **NO-GO** | Official H100-only caveat; batch-stride failures; non-finite/inconsistent recurrent path on RTX 3060. |
| Mamba-2 API/backend implementation | **GO as an experimental backend** | Exact cache/step passes and memory fits. |
| Mamba-2 as default / claimed benefit | **NO-GO / DEFER** | Speed claim refuted; task comparison used different latent spaces; Phase B fails. GRU remains default correctness backend. |
| Hard predictor mixture for Crafter | **NO-GO on critical path / DEFER** | Synthetic mechanics pass, real consequential multimodality evidence is insufficient. Deterministic stays default. |
| Rollout bridge implementation | **GO, opt-in only** | Source-backed structural mechanism, unit-tested, affordable. Efficacy is not yet validated; default weight remains zero. |
| Masked temporal hybrid | **NO-GO as default** | Not source-faithful I-JEPA, creates mismatch, worse prediction, higher memory. Keep only as an explicit control. |
| Reliability weighting | **NO-GO** | No held-out later-policy calibration. Shadow logging only. |
| Current representation objective | **NO-GO for scale-up** | No tested arm passes all corrected Phase B gates. |
| Full online policy training | **NO-GO** | Phases B, D attribution, E, and F are not passed. |

This is the stopping point for the present investigation. The negative evidence
is sufficient to revise the architecture; more training on the current hybrid
would reduce attribution rather than increase confidence.
