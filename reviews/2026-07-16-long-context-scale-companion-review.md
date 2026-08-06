# Companion review: Step 4b long-context Mamba scale screen

Date: 2026-07-16

Outcome code commit: `1476dab7e199080f97e91020bdecf83a970880a1`

Pinned monitor-manifest / run commit:
`9cf965cdf76561b402594fcb4b92c18b3c6eb0fe`

Protocol: `reviews/2026-07-16-long-context-scale-protocol.md`

Outcome report: `reviews/artifacts/long_context_scale_screen.json`

## Executive verdict

Running the larger Mamba was worthwhile, but it did **not** pass the registered
quality gate. The result is not “Mamba is bad” and it is not an implementation
failure:

- the width-512/depth-2 Mamba-2 is finite, cache-correct, and extremely cheap
  on the actual GPU;
- it cuts measured steady `B=1,T=128` full training-step time from about 420 ms
  (matched GRU) to 51 ms and cuts actual 4k screen wall time by 4.7x;
- at the final checkpoint it has better four-way suffix retrieval and lower
  absolute matched-target error than the large GRU;
- nevertheless, the registered primary `k=8` counterfactual separation is
  lower, the paired difference has the wrong sign in three of four evaluation
  environment seeds, patch separation is also lower, and neither large core
  improves final separation over its small counterpart.

Therefore:

1. **No** confirmatory replication or shuffled controls are licensed by this
   protocol.
2. **No** architecture-default change is licensed. GRU-64 remains the current
   operational backend under the Step-4 parity tie-break.
3. **Keep Mamba-2 as a research backend**, not a failed idea. It has a real
   long-sequence training-throughput advantage and a post-hoc intermediate-
   horizon signal that deserves a separately preregistered hypothesis if the
   project chooses to pursue it.
4. This experiment does **not** test the broad claim that Mamba handles true
   long-range dependencies better. It tests backend x scale at fixed
   `T=128` in the project's noncanonical pooled topology.

## 1. Was this actually source-like?

Only in the three dimensions deliberately registered before training:
training length 128, hidden width 512, and two Mamba layers. It is not a DRAMA
reproduction.

| property | DRAMA paper | pinned DRAMA source | Step 4b LL-M |
|---|---:|---:|---:|
| sequence length | 128 world-model steps | `BatchLength: 128` | 128 |
| hidden width | 512 | 512 | 512 |
| layers | 2 | 2 | 2 |
| SSM state | 16 | 16 | 64 |
| Mamba-2 head dimension | 128 | omitted; vendored default is 64 | 64 |
| normalization | RMS | RMS/fused wrapper defaults | LayerNorm |
| dropout | 0.1 | 0.1 passed to wrapper | 0 |
| optimizer | LaProp, lr `4e-5`, warmup | same in default config | AdamW, lr `1e-4`, no warmup |
| input | flattened 1024-way categorical latent + action | same | mean-pooled 66x64 continuous tokens with action embedding |
| output path | recurrent state carries model state | same | projected global channel plus dense residual bypass |

Primary-source anchors:

- official Mamba commit:
  `f577286d052741c35d39cd43bdc3fad27120f22c`;
- DRAMA commit: `a50bd54c34e77d1d13e988a031733a47817098e2`;
- official Mamba README calls `d_state=64` or 128 typical for Mamba-2;
- DRAMA `configure.yaml:25-26,51-62,91-95` fixes batch 16, length 128,
  width 512, two layers, `d_state=16`, LaProp, dropout 0.1;
- DRAMA `world_models.py:309-322` passes `d_state` but not `headdim`;
- its vendored `mamba2.py:37-50` therefore uses `headdim=64`, despite the
  paper's Table 6 reporting 128;
- DRAMA's complete wrapper uses RMSNorm, residual-in-FP32, fused add-norm, a
  final norm, dropout, and depth-aware initialization. The compact adapter
  uses direct official Mamba-2 blocks with pre-LayerNorm residuals and a final
  LayerNorm, but not that complete wrapper or initializer.

That source/paper head-dimension discrepancy is in DRAMA itself, not in this
screen. Step 4b matches the pinned implementation's implicit head dimension
and the official repository's typical state size, but deliberately does not
match DRAMA's state size or training recipe.

### What the DRAMA evidence actually establishes

DRAMA's direct GRU comparison is a custom grid-token reconstruction task, not
its Atari world-model ablation. It compares token sequence lengths 208 and
1,664:

| length | Mamba-2 time/error | GRU time/error |
|---:|---:|---:|
| 208 | 25 ms / 15.6% | 75 ms / 21.3% |
| 1,664 | 214 ms / 14.2% | 628 ms / 34.7% +/- 25.4% |

The paper itself cites 1,000+ as “long” and concludes that Drama does not show
a decisive advantage over other Atari100K world models; it explicitly leaves
tasks where longer sequences improve MBRL as future work. Consequently, our
quality parity/failure at temporal length 128 does not contradict the paper.
Our large throughput win is the part most directly aligned with its evidence.

## 2. Protocol and implementation audit

### Passed before outcomes

- Protocol and four-arm design committed before monitor collection.
- Monitor bundle generated after the code commit, repeat-verified, then pinned
  in a separate clean commit before training.
- Bundle hash:
  `7e33efb705c00232fadf76d27c5d28db143766c7a7b7e20853f11a16aca36195`.
- 24 anchors: exactly four day and two night anchors for each of environment
  seeds 111-114.
- Four unique suffixes per anchor, three common-RNG branches per suffix.
- 23/24 anchors pixel-effective; 17/24 task-effective.
- Exact temporal parameter match: LL-G 3,372,004 versus LL-M 3,376,032,
  a 0.119% difference.
- All 84 compact tests passed. New coverage includes actual-width `T=128`
  FP32/BF16 sequence-step equivalence, reset isolation, cache cloning, finite
  BF16 gradients, parameter counting, and registered-gate arithmetic.
- Full-world BF16 forward/backward finite for every arm.
- Initial non-temporal state digest identical across all arms.
- Every sampled training tensor hashed; prefix and continuation stream digests
  are identical across all arms.
- No source/backend fallback: saved classes are `GlobalGRUTemporal`,
  `GlobalMambaTemporal`, `ProjectedGlobalGRUTemporal`, and
  `ProjectedGlobalMambaTemporal`, respectively.

Two pre-outcome harness issues were caught and fixed before collection:

1. the GRU parameter matcher instantiated hundreds of 3.4M-parameter modules;
   it now uses an exact algebraic count;
2. the feasibility profiler initially hashed shared weights after its optimizer
   step; it now hashes initial state, which passes identically.

Neither issue touched outcome training or evaluation.

### Post-run independent validation

- All 16 checkpoint file hashes match the report.
- Every floating state tensor and every one of the 4,000 values in each saved
  loss history is finite.
- Independent re-evaluation of all four final checkpoints reproduces all 96
  rows and all metric values exactly (`max difference = 0`).
- Trained `T=128` cache equivalence on a real replay prefix:
  - LL-M FP32 mean/max absolute difference: `0.000129 / 0.000959`;
  - LL-M BF16-AMP mean/max absolute difference: `0.002581 / 0.015625`;
  - mean cosine discrepancy under BF16: `5.35e-6`;
  - all outputs finite.
- The monitor's `k=8` target is not degenerate: mean between-suffix latent
  distance is 0.05263, the highest horizon value. Branch noise is also highest
  (0.00467), giving the lowest-but-still-large target SNR (11.28).

### Remaining design limitations

These limit inference; they do not invalidate pairing.

1. **One training seed.** The four evaluation seeds estimate environment
   variation conditional on training seed 404, not optimizer/initialization
   variation.
2. **Only 24 monitor anchors.** Retrieval changes in increments of roughly one
   percentage point and endpoint trajectories are visibly volatile.
3. **Batch 1 was not forced by VRAM.** A post-run outcome-blind feasibility
   check showed both large arms can execute `B=16,T=128` below 2.0 GiB reserved.
   B1 kept the smallest equal-exposure screen, but is far from DRAMA's batch 16
   and supplies one correlated episode per update. It likely contributes to
   noisy optimization. A post-hoc B16 rerun is not licensed.
4. **This is not formal long-range memory.** The replay's maximum episode is
   200 transitions (201 observations); temporal length 128 is the longest
   well-supported regime, but not the 1,000+ sequence regime used by DRAMA's
   direct long-range comparator.
5. **Adapter-family, not pure operator, contrast.** Mamba uses pre-norm residual
   blocks (`y + Mamba(LN(y))`); the stacked GRU uses conventional recurrent
   cells followed by norms and no identical block residual. This is a fair
   comparison of the implemented backend families, not a mathematical
   SSM-versus-GRU operator isolation.
6. **Pooled topology remains noncanonical.** Neither large arm tests DRAMA's
   flattened latent state nor a dense jointly mixed temporal topology. The
   dense residual bypass can make good average prediction coexist with weak
   action-specific separation.

## 3. Registered outcome

### Final 4k endpoint

| arm | temporal params | k8 separation | patch separation | tie retrieval | peak reserved | total train wall |
|---|---:|---:|---:|---:|---:|---:|
| LS-G64 | 29,248 | 0.001646 | 0.001630 | 27.60% | 160 MiB | 18.33 min |
| LS-M64 | 34,584 | 0.001230 | 0.001222 | 28.12% | 374 MiB | 5.18 min |
| LL-G | 3,372,004 | 0.001479 | 0.001458 | 23.44% | 206 MiB | 30.45 min |
| LL-M | 3,376,032 | 0.001220 | 0.001237 | 27.08% | 440 MiB | 6.52 min |

Registered arithmetic:

```text
delta_small = LS-M64 - LS-G64 = -0.00041697
delta_large = LL-M   - LL-G   = -0.00025892
interaction = delta_large - delta_small = +0.00015805
```

The positive interaction does **not** mean large Mamba wins. It arises because
small Mamba loses by more than large Mamba. The direct large contrast remains
negative.

Gate conditions:

| condition | result |
|---|---|
| finite and under 5,000 MiB | pass (440 MiB) |
| LL-M separation positive | pass |
| large delta positive in >=3/4 env seeds | **fail: 1/4** |
| interaction positive | pass |
| positive minimum effect (threshold 0.000148) | **fail: delta is negative** |
| not jointly contradicted by retrieval and patch separation | pass (retrieval positive, patch negative) |

Overall: **confirmatory replication not licensed**.

Conditional, environment-clustered uncertainty (one training seed):

- large `k=8` separation delta: -0.000259,
  cluster-bootstrap 95% interval `[-0.000771, +0.000388]`;
- four-environment t interval: `[-0.001398, +0.000880]`;
- retrieval delta: +3.65 points,
  cluster-bootstrap interval `[-1.56, +7.29]` points.

Neither backend superiority direction is established.

### Scale itself did not win at the final endpoint

At 4k:

- LL-G minus LS-G64 separation: `-0.000168`;
- LL-M minus LS-M64 separation: `-0.000010`.

The larger cores fit the training and held-out teacher-forced objectives better,
but do not improve final counterfactual separation. “Mamba only needed more
capacity” is therefore refuted for this pooled B1 screen.

### Rung sensitivity

| step | LS-G64 | LS-M64 | LL-G | LL-M | large M-G |
|---:|---:|---:|---:|---:|---:|
| 500 | -0.000104 | -0.000067 | 0.000794 | 0.000971 | +0.000177 |
| 1,000 | -0.000030 | 0.000005 | -0.000043 | 0.000399 | +0.000442 |
| 2,000 | 0.000401 | 0.000794 | 0.001171 | 0.001264 | +0.000093 |
| 4,000 | 0.001646 | 0.001230 | 0.001479 | 0.001220 | -0.000259 |

This is why the protocol correctly prohibited best-checkpoint selection. Large
Mamba leads at the first three rungs but not at the registered final executed
rung; the small-pair sign also reverses from 2k to 4k.

## 4. The important secondary pattern (exploratory only)

LL-M minus LL-G continuous separation by horizon at 4k:

```text
k1 +0.000020
k2 +0.000139
k3 +0.000178
k4 +0.000353
k5 +0.000536
k6 +0.000600
k7 +0.000189
k8 -0.000259   <- registered primary
```

Post hoc, mean `k=2..7` advantage is `+0.000332`, positive in all four
environment seeds, with a four-seed t interval `[+0.000018, +0.000647]`.
The corresponding backend x scale interaction is `+0.000562`, also positive
in all four environment seeds. It remains hypothesis-generating because:

- the span was selected after seeing the curve;
- eight horizons create multiplicity;
- there is only one training seed;
- the registered hardest endpoint reverses sign.

This pattern should not be erased, but it cannot replace the registered
verdict. If pursued, mean multi-horizon separation must be preregistered on
new monitor and training seeds, with `k=8` retained as a safety constraint.

### Why separation and retrieval disagree

At `k=8`:

| large arm | matched target error | mismatched target error | separation | retrieval |
|---|---:|---:|---:|---:|
| LL-G | 0.059234 | 0.060713 | 0.001479 | 23.44% |
| LL-M | **0.058505** | **0.059725** | 0.001220 | **27.08%** |

LL-M predicts the matched outcome slightly more accurately, but it also moves
closer to the three mismatched outcomes by a larger amount. It is smoother and
less action-discriminative on average at the final horizon. LL-G has greater
mean contrast but worse row-wise ranking. Robust `k=8` separation statistics
still favor LL-G (mean, median, and two-tail-trimmed mean), so this is not only
one extreme outlier.

The primary was designed to test control-relevant action discrimination, so
the lower LL-M separation matters. The lower matched error and higher retrieval
show why “Mamba predicts worse” would nevertheless be an inaccurate summary.

### Strata

LL-M minus LL-G `k=8` separation is negative in every registered aggregate:

- day: about `-0.000266`;
- night: about `-0.000245`;
- pixel-effective: about `-0.000270`;
- task-effective: about `-0.000566`.

The post-hoc `k=2..7` advantage is positive in all four of those aggregates.

## 5. Loss and generalization diagnosis

Mean last-500 training components:

| arm | total | JEPA | rollout | reward |
|---|---:|---:|---:|---:|
| LL-G | 0.16419 | **0.02892** | **0.03438** | 0.09781 |
| LL-M | **0.13775** | 0.02942 | 0.03511 | **0.07015** |

The Mamba optimization is healthy; it attains lower total training loss. The
advantage is mostly the reward term, not the latent prediction terms.

On 80 identically sampled windows from 19 pre-existing held-out episodes:

| arm | total | JEPA | rollout | reward |
|---|---:|---:|---:|---:|
| LL-G | **0.16017** | **0.025940** | 0.032435 | **0.10078** |
| LL-M | 0.18569 | 0.026001 | **0.031880** | 0.12674 |

JEPA and rollout generalization are effectively tied; the Mamba training reward
advantage reverses. These episodes were used by earlier project phases, so this
is a diagnostic, not a fresh confirmatory set. It still refutes “Mamba failed
to train” and warns against using its lower training total as quality evidence.

## 6. Systems result on the RTX 3060 Laptop GPU

Feasibility measurements (`B=1,T=128`, BF16 AMP, FP32 parameters/cache):

| large arm | steady full train step | temporal sequence | B1 H8 imagination | B1 cache |
|---|---:|---:|---:|---:|
| LL-G | 419.8 ms | 35.63 ms | 12.69 ms | 0.004 MiB |
| LL-M | **50.9 ms** | **1.93 ms** | 20.54 ms | 0.535 MiB |

- Core sequence path: Mamba is about 18.5x faster.
- Full steady training step: Mamba is about 8.3x faster.
- Actual auditable 4k runner wall: Mamba is about 4.7x faster after hashing,
  checkpointing, and evaluation overhead.
- Recurrent deployment remains worse: at `B=48,H=8`, LL-M takes 20.72 ms vs
  LL-G 12.76 ms (1.62x slower) and uses 25.69 MiB vs 0.192 MiB cache (134x).
- Both are safely inside 6 GB. Even `B=16,T=128` one-step feasibility reserves
  only 1.91 GiB (LL-G) and 1.96 GiB (LL-M).

This is the clearest positive finding: Mamba earns its place when parallel
long-sequence training throughput matters. It does not currently earn the
online recurrent deployment slot.

## 7. Answers for the researcher agent

### Was there a point to the bigger-Mamba test?

Yes. It ruled out GPU capacity, stale caches, approximate recurrence, tiny-core
capacity, and short training-sequence throughput as explanations for Step 4's
parity. It also exposed a real intermediate-horizon hypothesis that the tiny
Step-4 test could not show.

### Is Mamba performing worse because it is old or incorrectly implemented?

No evidence supports that. Official Mamba-2 is pinned; trained sequence/cache
equivalence passes; gradients and states are finite; training converges; and
Mamba is faster and sometimes more accurate. Mamba-3 remains unusable on this
GPU for the independently verified recurrent non-finiteness/H100-only path, so
“upgrade to Mamba-3” is not a valid fix.

### Is the simple GRU result expected?

Plausible, yes. The cited Mamba work primarily demonstrates scaling and
throughput, not universal predictive dominance over a GRU at temporal length
128 and low data. DRAMA's strongest direct GRU result is a specially constructed
1,664-token memory task. Its own paper disclaims decisive Atari dominance.

### Are we missing training?

Possibly source-style batch diversity, warmup, optimizer, RMS wrapper, dropout,
and depth-aware initialization—but none is an implementation bug, and changing
them only for Mamba would break the controlled test. B1 is the most important
power limitation because B16 fits. A new experiment would need to preregister
the full recipe and matched control; this failed screen does not authorize a
post-hoc rerun.

### Is using Mamba in isolation the issue?

It may be. The current mean-pool -> temporal core -> broadcast adapter is not
used by DRAMA, Dreamer 4, V-JEPA-2-AC, or LeWM. Scaling the core does not remove
the pooled bottleneck or dense bypass. The result therefore weighs against
“just make the pooled Mamba bigger,” not against a source-shaped global state
or jointly mixed dense temporal architecture.

### What should remain the coherent project position?

- Operational Crafter backend now: **GRU-64** (unchanged).
- Mamba-2 correctness/support on 6 GB: **GO**.
- Mamba-2 as a proven quality winner: **NO-GO**.
- Larger pooled Mamba as the next default: **NO-GO**.
- Shuffled controls / fresh 115-130 confirmation for this screen: **NO-GO**
  under the registered conditions; those seeds remain unspent.
- Mamba as a research thesis: **retain, narrow, and reformulate**. The defensible
  thesis is efficient long-sequence state modeling with a possible
  intermediate-horizon benefit, not “Mamba beats GRU in this pooled model.”
- Online policy training and reliability weighting: still **NO-GO**; this
  screen changes neither gate.

The next architectural discussion should decide whether the thesis requires a
genuinely long-dependency task and source-complete Mamba stack, or whether the
project's priority is Crafter control at horizon eight. Those are now visibly
different research questions and should not be conflated in another mutation
of the same pooled adapter.
