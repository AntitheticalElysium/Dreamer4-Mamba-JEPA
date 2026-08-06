# Independent companion audit: Step 4 and the Mamba question

Date: 2026-07-16
Audited run commit: `77d2a2e3237b9283e69ebf9118aa577b17221672`
Artifact commit / current HEAD: `6d04d9fa69df6a388679c8bdb42b2730dbe0db32`
Official Mamba source: `f577286d052741c35d39cd43bdc3fad27120f22c`
(also the live upstream HEAD when re-queried on 2026-07-16)

## Bottom line

I accept the Step-4 **statistical result and operational decision**:

- both model families satisfy all registered causal-validity gates;
- the registered primary backend comparison is parity, not a GRU predictive
  win;
- GRU-64 is the correct operational tie-break for the present recurrent online
  imagination path;
- Step 4 does **not** reject Mamba as a research thesis. It rejects only the
  claim that one tiny, one-layer, pooled-vector Mamba-2 adapter has already
  shown a predictive advantage at `T=16` in this topology.

The user's premise needs one correction: Mamba did not perform worse on the
registered predictive readout. Its pooled retrieval was slightly higher
(29.17% vs 28.82%, +0.35 points), its final dynamics losses were slightly
lower, and its parallel training path was faster. It failed to win because the
three paired seed signs were inconsistent and the uncertainty interval crossed
zero. GRU was then selected by a pre-registered engineering tie-break.

## Independent integrity checks

- Recomputed the primary paired verdict directly from all twelve row files:
  +1.302 / +0.260 / -0.521 points for training seeds 101/202/303; pooled
  +0.347 points, two-level 95% CI [-0.694, +1.432]. This exactly reproduces the
  report.
- Recomputed family summaries and all control means. Both families pass all
  four gates in all three seeds; shuffled controls remain 24.74-25.52%.
- Verified SHA-256 for all 24 retained 8k/16k checkpoints against the report,
  checked their source digests, and checked 192 rows in every final row file:
  no mismatch.
- The report's source digest still equals the current imported-source digest.
  Its run HEAD being `77d2a2e` while the repository is at `6d04d9f` is correct:
  the latter commit only files the completed run artifacts/status.
- Re-ran tests: 75 non-slow passed, 2 slow passed (77/77 total); compileall
  passed. One existing tensor-to-float warning remains. No source file was
  changed by this audit.
- VRAM records are coherent: every GRU arm is 69.8/112 MiB
  allocated/reserved; five of six Mamba arms are 70/112 MiB. The first real
  Mamba arm records a one-time kernel-compilation peak of 312.6/368 MiB. None
  approaches the 6 GB limit.

## Severity-ranked findings

### High: the result must not be narrated as “Mamba lost to a GRU”

The registered conclusion is parity. Mamba's final per-seed retrieval is
30.21/28.78/28.52%; GRU's is 28.91/28.52/29.04%. Mamba also ends with a lower
100-step mean JEPA loss in all three seeds (approximately
0.01672/0.01677/0.01657 vs 0.01705/0.01691/0.01688) and a lower rollout loss in
all three. It is therefore unsupported to diagnose a general Mamba
optimization or dynamics failure from Step 4.

Post-hoc, the registered row matrices contain a useful clue. Continuous
all-token separation (`mean(off diagonal) - mean(diagonal)`) favors Mamba in
all three seeds by +0.000214/+0.000284/+0.000692. A two-level bootstrap gives
+0.000397, CI [+0.000140,+0.000694], including in the pixel- and task-effective
strata. With only three training seeds the seed-level t interval still crosses
zero, and this metric was not registered as the backend decision metric, so it
is a diagnostic rather than a retroactive Mamba win. It does show that the
coarse argmin score is hiding some real continuous improvement.

### High: Step 4 is not a literature-regime test of Mamba

`GlobalMambaTemporal` mean-pools 66 streams to one 64-D token per frame, runs
one Mamba block over only 16 time tokens, projects one global vector, and
broadcasts it behind a dense residual. The temporal module is only 34,584 of
245,083 trainable parameters; the future predictor alone has 112,257.

This is far outside the regimes supporting the cited Mamba advantage:

- Mamba-2's paper benchmarks recall at lengths 256-1024, efficiency from
  hundreds to thousands of tokens, and 125M+ deep language models. Its SSD
  speed crossover discussion is around sequence length 2k, not 16.
- DRAMA trains length-128 sequences with batch 16, a 512-D deterministic
  state, two Mamba layers, and a learned
  `flattened categorical latent + action -> Linear -> RMSNorm -> SiLU` stem,
  followed by a final norm. Its separate grid control calls 208 tokens “short”
  and 1664 “long”. DRAMA itself concludes that it did not demonstrate a
  decisive Atari-100k advantage over other world models.
- The current adapter is explicitly and correctly labelled source-inspired,
  not DRAMA-faithful. Consequently Step 4 licenses only “parity in this compact
  pooled topology at this scale.”

This is the most plausible reason a simple GRU remains competitive: the test
removes most of the long-sequence/deep-stack regime where an SSM is expected to
pay for its extra machinery. It is not evidence that Mamba is intrinsically
unsuitable for world models.

### Medium: two source-level adapter divergences existed, but screens do not
support either as the missing explanation

1. The official complete Mamba stack applies a final normalization after the
   residual stack. The local adapter applies pre-norm residual Mamba blocks but
   sends the unnormalized final residual directly to `out`; the GRU path does
   normalize its hidden output.
2. Official Mamba-2 marks `dt_bias`, `A_log`, and `D` as no-weight-decay. The
   Step-4 runner puts every trainable tensor into one
   `AdamW(weight_decay=0.01)` group.

I screened both under the exact seed-101 Step-4 replay/initialization contract:

| 4k Mamba arm | retrieval | matched separation | JEPA (last 100) |
|---|---:|---:|---:|
| registered adapter | 24.48% | 0.002723 | 0.029014 |
| official no-decay tags | 24.48% | 0.002723 | 0.029012 |
| add final norm | 26.56% | 0.002089 | 0.028271 |
| both | 26.56% | 0.002069 | 0.028268 |

No-decay is empirically negligible here. The final norm gives a noisy discrete
gain but weaker continuous separation. Extending final norm to 8k removes the
apparent gain: 26.04% retrieval and 0.002149 separation versus the registered
8k arm's 27.60% and 0.002119, with virtually identical losses. Neither merits a
larger run on present evidence. The divergences should nevertheless remain
documented rather than described as a full-stack official reproduction.

The adapter also sets `use_mem_eff_path=False`. A direct probe shows why: the
default fused path fails because the optional causal-conv1d forward function is
unavailable (`TypeError: 'NoneType' object is not callable`). The selected
fallback is an official mathematical path and is stable; this is a possible
training-throughput optimization in a separate environment, not a predictive
correctness defect. Recurrent `step()` is unaffected.

### Medium: the primary metric is valid but coarse and underpowered for backend
selection

Four-way retrieval advances in 0.25 increments per anchor. About 15% of final
prediction rows have an exact tie at the minimum because some suffix targets
are observationally identical. The deterministic argmin rule still leaves
shuffled controls at chance and fractional tie credit leaves the backend
verdict at parity, so Step 4 is not invalidated. But the quantization helps
explain why continuous separation can improve in every seed without a stable
argmin win.

For the next backend experiment, continuous symmetric separation should be a
registered co-primary metric and retrieval should use order-invariant
fractional tie credit. The current retrieval result must not be rewritten
post-hoc.

### Low: protocol status prose is now contradictory

`2026-07-14-step4-protocol.md:222-236` interleaves the completed-run paragraph
inside the old smoke sentence, and ends by saying that the run “proceeds” after
the same run is already complete. This is documentation-only, but it should be
cleaned up so the scientific record has one chronological status.

## Direct implementation checks

### Official recurrent state

The local model allocates and updates the exact official Mamba-2 pair
`(conv_state, ssm_state)`. On all three trained checkpoints, comparing a
16-token sequence forward with sixteen official recurrent steps gives:

| seed | FP32 mean abs delta | BF16-autocast mean abs delta |
|---:|---:|---:|
| 101 | 1.74e-5 | 0.001514 |
| 202 | 7.01e-6 | 0.001682 |
| 303 | 1.10e-5 | 0.001573 |

The mean latent magnitude is about 0.89. These are normal kernel/order effects,
not a stale-cache or invented-state failure.

### The temporal path is actually used

On the 48-anchor monitor set at horizon 8, averaged over all three trained
seeds:

| backend / intervention | retrieval | symmetric separation |
|---|---:|---:|
| GRU normal | 28.99% | 0.003372 |
| GRU reset recurrent state every step | 29.51% | 0.003248 |
| GRU zero recurrent broadcast | 25.35% | 0.000039 |
| Mamba normal | 30.21% | 0.003803 |
| Mamba reset recurrent state every step | 30.03% | 0.003291 |
| Mamba zero recurrent broadcast | 26.39% | 0.000474 |

Zeroing the recurrent broadcast collapses nearly all continuous action
separation in both arms. Resetting only the cache is much less damaging because
the autoregressed dense tokens themselves carry history, but the separation
drop is larger for Mamba. Thus the result is not explained by an ignored Mamba
cache.

### Mamba-3 is newer, but is still a no-go on this GPU

The official repository is current and contains Mamba-3; Mamba-2 is therefore
older as an algorithm, but not an accidentally stale package. The official
Mamba-3 recurrent source still says “Only tested on H100”. A fresh RTX 3060
probe at `d_model=64,d_state=32,headdim=16,T=16` produced finite sequence
forwards for B=1 and B=4, but non-finite outputs from repeated official
`step()` in both cases. Replacing a working Mamba-2 backend with Mamba-3 would
therefore violate the recurrent correctness gate.

## Training/budget diagnostics (post-hoc, monitor-only)

These screens do not alter the blind final verdict; seeds 79-94 remain spent.

- A seed-101 LR screen at 4k showed that 2e-4 optimizes the loss faster
  (JEPA 0.02236 vs 0.02901 at 1e-4), but did not coherently improve the causal
  metrics (26.04% retrieval, 0.002577 separation). At 5e-5, retrieval rose
  spuriously to 28.65% while JEPA worsened to 0.04680 and separation halved to
  0.001184. This is evidence for metric noise, not an LR winner.
- Exact checkpoint continuations from 16k to 20k were run for both backends and
  all three seeds. Mean monitor retrieval at 16k was 28.99% GRU vs 30.21%
  Mamba; at 20k it was 29.34% vs 30.21%. Mean matched separation at 16k was
  0.002423 vs 0.003355; at 20k it was 0.003217 vs 0.003560. Mamba remains
  slightly ahead on the means, but seed signs are inconsistent and GRU closes
  much of the separation gap. Extra budget does not expose a hidden Mamba win.

The correct conclusion is that 16k was a registered comparison budget, not a
convergence proof. A new experiment should register learning curves rather than
selecting an endpoint after looking at them.

## Engineering interpretation

The measured Step-4 figures reproduce: Mamba is 2.35x faster on the parallel
`B=4,T=16` temporal sequence and about 30% faster per full training arm, while
GRU is 4.36x faster in isolated recurrent `step()` and has a 76x smaller cache.

An additional full-world benchmark at B=48,H=8 gives:

| path | GRU | Mamba-2 | Mamba slowdown |
|---|---:|---:|---:|
| full `imagine_step`, FP32, ms/step | 1.134 | 1.614 | 1.42x |
| full `imagine_step`, BF16, ms/step | 1.461 | 1.952 | 1.34x |

The full predictor/heads dilute the temporal-core gap but do not reverse it, so
the deployment tie-break remains sound.

Mamba's parallel advantage grows sharply with sequence length in this exact
adapter (B=4):

| T | GRU ms | Mamba ms | Mamba speedup |
|---:|---:|---:|---:|
| 8 | 1.008 | 0.846 | 1.19x |
| 16 | 2.013 | 0.874 | 2.30x |
| 32 | 3.945 | 0.856 | 4.61x |
| 64 | 7.672 | 0.866 | 8.85x |
| 128 | 15.521 | 0.854 | 18.17x |
| 256 | 31.142 | 0.910 | 34.21x |

This is the clearest surviving Mamba thesis: longer parallel history is cheap.
The project has not yet tested whether Crafter prediction or policy learning
benefits from that longer history.

## Recommended consensus and next action

1. **Accept Step 4 and retain GRU-64 operationally** for the next correctness
   and calibration phase. Do not run GRU-72; it was conditional on a registered
   Mamba win, which did not occur.
2. **Keep Mamba-2 separately runnable and keep the thesis open**, phrased as:
   “Does cheap long-context state-space training improve a JEPA world model?”
   Do not phrase it as “Mamba should beat GRU at T=16.”
3. **Do not block Phase E/F calibration work on another backend search.** Full
   online policy training remains gated on those phases, exactly as before.
4. Before permanently demoting the Mamba thesis, pre-register one bounded
   Step-4b on new seeds:
   - first screen the backend-by-context interaction (`T=16` vs `T=64`) in the
     current topology;
   - give both context arms the same observations and effective batch by
     evaluating four contiguous T=16 chunks versus one T=64 sequence before
     each optimizer step, avoiding a fourfold data confound;
   - collect a new long-prefix counterfactual monitor/final bundle;
   - register continuous separation plus fractional-tie retrieval and learning
     curves at fixed rungs;
   - use at least three training seeds only if the one-seed screen shows a
     backend-by-context interaction.
5. If and only if longer history helps Mamba differentially, run a second,
   separately named **DRAMA-shaped** arm: identical learned
   flattened-latent/action stem and output normalization for Mamba and GRU,
   with parameter-matched depth. Do not call it DRAMA-faithful unless the
   stochastic latent, objectives, and training recipe are also reproduced.
6. Do not spend more runs on no-weight-decay or final-norm repairs without new
   evidence. Benchmark the optional causal-conv1d fused path only in an isolated
   environment; it can improve engineering throughput but cannot explain the
   present predictive parity.

This preserves the successful causal-validity result, avoids forcing Mamba into
a regime the literature never promised it would win, and turns the unique
thesis into a falsifiable context-length interaction rather than a preferred
component looking for a favorable metric.
