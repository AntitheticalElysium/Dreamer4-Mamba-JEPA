# Final M0 pre-build audit — fc120c63

**Current-version note:** while this audit was running, a concurrent revision was
committed as `e1b4cb40`. I checked that delta too: see [DELTA_REVIEW.md](DELTA_REVIEW.md).
It resolves F1's intended frame-gradient routing and removes SIGReg from M0, making
F6 historical. Its new chunk-wide BatchNorm introduces a causal-perception problem
that needs closure before P1. F2-F5 and F7-F11 remain relevant. This document below
preserves the audit of the user-requested `fc120c63` rather than silently rewriting
the original target.

**Verdict:** start the data-stream implementation. The architecture is coherent enough
to build. Close the gradient, EDM objective, sampler/cache, and reward-decoding
contracts before the long P1 runs. These are bounded implementation specifications,
not reasons for another architecture research round or new user-facing forks.

Audited HEAD: `fc120c63f004655ecea656d41b89daae142ed7df`. No M0 implementation
exists yet: findings below distinguish specification gaps from measured behaviour of
existing code. This audit did not edit canonical design or eval files. The concurrent
commit is recorded separately. The pre-existing NOTEBOOK modification and other
agents' work were preserved.

```text
frame -> encoder -> percept z_t -----> clean recurrent belief -> later prediction
                         |
                         +-> noisy input -> denoiser_t -> loss_t
                         |                              ^
                         +-> detached target -----------+

Target detachment closes the bottom gradient path. It does not close the noisy-input
path. Diffusion forcing also uses that same noisy input as history for later losses.
```

## Before P1

### F1. Specify the diffusion-forcing gradient graph, not only target detachment

**Status: an unresolved contract; numerical counterexample verified.**

Anchors: `LOSSES.md:16-20,33-36`; `STEP_SPEC.md:22-36,80-82`;
`M0_CODE.md:27-33,51-54`.

LOSSES says denoising gradients train the encoder through past percepts and the belief.
The plan also trains every time position of one diffusion-forcing sequence. A percept
is therefore both its own noisy denoising input and the context of later predictions.
Detaching just the label leaves a same-frame encoder gradient through the noisy input.
Detaching the whole noisy sequence also removes the direct past-input route described
in LOSSES. The clean Mamba conditioning route can still train the encoder.

JEDI Eq. 8 explicitly detaches its future noisy input as well as its clean target;
its past conditioning latents can receive gradients. That separate-input construction
does not directly specify the shared-sequence diffusion-forcing case. Dreamer 4's
separately trained tokenizer does not settle the joint-encoder question either.
[JEDI](https://arxiv.org/html/2605.13013v1),
[Dreamer 4](https://arxiv.org/html/2509.24527).

`check_contracts.py` verifies the distinction on two time positions: label detachment
leaves nonzero own-frame encoder derivatives; detaching all noisy inputs removes the
direct past-input derivative too. This is algebra, not evidence for or against learning.

**Required closure:** declare the chosen graph and test gradients from each loss
separately. A small valid implementation can detach the denoiser's whole noisy
sequence and explicitly train the encoder through clean belief conditioning and the
grounding/regularisation routes. Preserving a separate direct past-context gradient is
another possible graph, but must be implemented explicitly. Do not accidentally claim
both behaviours from a single `.detach()`.

### F2. EDM noise weighting remains open despite the complete-YAML claim

**Status: confirmed omission.**

Anchors: `m0.yaml:62-68`; `LOSSES.md:41,55-57`; `M0_CODE.md:32-33`.

The noise distribution and preconditioning are specified; the loss weighting is not.
LOSSES still explicitly lists it as open. The JEDI/DIAMOND raw-network regression is
equivalent to clean-x regression weighted by

`w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`.

With sigma_data=1, the weight is 250001 at sigma=.002, 101 at .1, and 1.0025 at 20.
Plain unweighted MSE on the preconditioned clean prediction is a different objective.
The equivalence check agrees to `4.55e-13` in float64.
[DIAMOND denoiser](https://raw.githubusercontent.com/eloialonso/diamond/main/src/models/diffusion/denoiser.py).

**Required closure:** use either equivalent form, name it explicitly, and test equality
with fixed inputs. Also specify how the periphery's 16 channels, glimpse's 64 channels,
and 8-dimensional lighting token are reduced into G1: "mean per token" and "mean per
element" need an unambiguous implementation for unequal token widths. The monitor must
use the same declared objective.

### F3. Full-sampler parity needs a stronger contract than D.step == D.forward

**Status: implementation requirement; cannot execute until the denoiser exists.**

Anchors: `m0.yaml:67-68`; `M0_CODE.md:27-33,50-53`; `CURRICULUM.md:35-40,47-50`.

The requested fixed-noise forward parity is necessary but insufficient. Within one
sample, Euler/churn changes the current frame and its noise level. Those intermediate
calls must not append history or advance Mamba. The completed frame is subsequently
history at context sigma=.1, not at the last solver sigma=.002. A KV cache from that
last solver call generally does not represent the required context.

Declare how finished frames are entered into the context cache, including their
conditioning noise and sigma. A straightforward implementation recomputes a completed
frame once in its context role, then commits it. This can add one forward evaluation
per generated frame; include it in the full-model throughput measurement. Whatever
implementation is chosen, compare the **entire cached sampler** against a recomputed
history implementation under identical noise draws, through multiple generated frames,
episode resets, branch duplication, and checkpoint resume.

There is also an upstream porting trap: the currently published DIAMOND sampler starts
with unit noise rather than sigma_max-scaled noise, and passes `sigma`, rather than
`sigma_hat`, to the denoiser after churn. NVIDIA's EDM algorithm uses the scaled initial
noise and the churn-adjusted level. With churn=0 the second distinction disappears;
M0 declares churn=1. Pin the intended EDM equations instead of treating every upstream
line as an unquestioned reference.
[DIAMOND sampler](https://raw.githubusercontent.com/eloialonso/diamond/main/src/models/diffusion/diffusion_sampler.py),
[NVIDIA EDM sampler](https://github.com/NVlabs/edm/blob/main/generate.py#L20-L46).

### F4. Bin likelihood does not define the sampled reward value

**Status: confirmed counterexample to the planned test; sampler not implemented.**

Anchors: `m0.yaml:73`; `M0_CODE.md:35-36,52`; `LOSSES.md:44`;
`saccade/eval/calib.py:32-39`.

The existing evaluator interprets [-.9,1.1] as RAW reward support and then symlogs it.
The zero-reward interval is bin 118, with symlog edges
`[-0.0040126157, 0.0013928188]`. Its centre decodes to reward `-0.0013107567`, not zero.
Thus a perfect mass-on-one-bin head passes the proposed test but can emit a fictitious
cost on every zero-reward step. The discounted 16-step bias is about `-.02051`; the
infinite-survival value shift at gamma=.997 would be `-.43692`. These numbers assume
bin-centre decoding, not an existing M0 bug.

Sampling continuously inside a likelihood bin also leaves behaviour that bin NLL alone
cannot identify. Specify the bin edges, tail handling, and emitted representative or
within-bin distribution. At minimum, deterministic zero must sample zero to a declared
tolerance. Test value-space samples, their means and variances for deterministic and
mixed targets, and agreement between empirical bin frequencies and log_prob.
A zero-centred grid/zero representative can resolve the zero issue without a new
Craftax-specific module. Keep raw support versus transformed support explicit.

## Build-time checks with directly measured evidence

### F5. Carried state and overlapping context must have one owner

Anchors: `M0_CODE.md:10-13,24-26`; `CURRICULUM.md:27-40`.

The next chunk carries eight old frames while Mamba already carries their effect.
Replaying those frames through the incoming state double-counts history. The denoiser
also needs the historical memory tokens associated with those prefix frames, not one
current memory token relabelled as every past state. Make the frame/action/reward
alignment explicit and test contiguous versus chunked execution with the actual
overlapping prefixes. Include convolution carry, SSM carry, attention cache, fixation
coordinates, padding, and episode reset.

The installed Triton kernel supplies an additional concrete trap. Its state-passing
code starts the incoming sequence index at zero. In a CUDA test with nonzero incoming
state:

| Sequence IDs | Output difference from normal carried execution | Final-state difference |
|---|---:|---:|
| all 0 | 0 | 0 |
| all 7 | 31.6868 | 3.28888 |

The all-7 case matched zero incoming state exactly. Arbitrary global episode IDs
therefore must not be passed as local kernel sequence IDs without respecting this
convention. This does not show a bug in unbuilt saccade code; it establishes what the
port tests must catch. See `seq_idx_results.json` and the hashed installed
`ssd_state_passing.py:64-79`. Segmentation must reset convolution history too; the
SSM kernel alone does not do that.

### F6. The regularizer has a material memory cost and a simple equivalent reduction

Anchors: `m0.yaml:79-81`; `d4mj/lewm.py:95-132`; upstream LeWM `module.py:10-36`.

The lambda*N reconciliation matches the pinned statistic's explicit multiplication
by statistical batch size. It is a defensible starting scale, not a guarantee of
equal gradient pressure across different token distributions.

Measured on RTX 3060 Laptop, FP32, 4096 samples EACH of 16- and 64-dimensional tokens,
1024 projections, 17 knots, both regularizer graphs followed by backward:

| Scheduling | Peak allocated memory | Parity |
|---|---:|---|
| direct canonical reduction | 1913.82 MiB | reference |
| eight checkpointed 128-projection reductions, same directions | 224.32 MiB | same weighted loss; max gradient error 2.73e-12 |

The change is scheduling of the same statistic, not fewer projections or a changed
scientific treatment. Its isolated timing was higher; the single measurements are in
the JSON files and are not steady-state M0 throughput estimates. Neither profile
establishes full-model fit, and independent peaks cannot simply be added. Use the
declared 200-update full-recipe profile, including all losses, optimizer state,
accumulation, correct cache commits, and both chunk lengths, before P1.

## Evaluation corrections to pin before reading M0 results

### F7. Three specific reference/control mismatches

1. `TEST_SPEC.md:50-52` still says exact HUD values are available only when fixated.
   The new acuity results contradict that: supervised HUD readability at r=3 is
   0.9998629 for both seeds. Keep the HUD probe, but interpret it beside the current
   percept control. It is not automatically a test of memory for past glimpses.
2. The last-eight-percepts control (`TEST_SPEC.md:56-58`) explicitly lists rewards,
   but omits actions and termination. The recurrent state sees them. Give the short
   history control the same within-window observations/actions, reward/done signals,
   fixations and lighting before interpreting a gain as memory beyond eight frames.
   A finite probe gap still measures accessible readout, not an information theorem.
3. The S3 blocker uses per-kind recall **and precision** (`TEST_SPEC.md:79-84`), but
   the acuity script stores aggregate presence precision and per-kind recall only
   (`eda_periphery_acuity.py:126-129`). The listed .96/.96/.90/.77 are recalls, not
   precision references. Compute and save the per-kind precision references as their
   own metrics before applying that blocker. The corresponding predictions/models
   were not saved by this script, so they cannot be reconstructed from its aggregate
   JSON alone.

The revised acuity-relative blocker is appropriate for separating input degradation
from learned erasure. It does not replace reporting absolute night performance or
testing whether the fovea helps decisions.

### F8. The original acuity rule had no passing candidate

The original saved rule required BOTH the day tolerance and >=.95 night recall for
every kind. No candidate met it, including r=7. Clause (i) alone selects r=3 in both
seeds. The current averaged near-player night recalls are .96463/.95880/.90053/.76674;
the reported rounded table is accurate.

The sensible interpretation is: retain the user-approved r=3 and explicitly record
the amended day-preservation criterion after the original rule proved infeasible.
"The original rule selected r=3 unchanged" would be incorrect. This happened before
M0 training and is transparently recorded in the results; it is not a reason to
rerun the architecture discussion or invalidate the measurements.

## Before P2/P3, rather than before data work

### F9. P2's continuing one-step objectives disagree across documents

`CURRICULUM.md:73-77` says reward/continuation heads train by continuing one-step G3.
But `m0.yaml:104` and `CURRICULUM.md:114-115` continue only G1. G4 detaches sampled
reward/done, so it does not train those heads. Decide explicitly whether G3 continues
on real data, or the heads freeze during P2 while the belief/denoiser change. Do not
assume G1 supplies their likelihood gradients. The former is the direct reconciliation
with the stated contract. This need not block implementing P1.

### F10. Padded patch coverage is unequal

`LOSSES.md:48` claims every token occurs nine times. For 81 zero-padded centred 3x3
windows, actual coverage is 4 at corners, 6 on non-corner edges and 9 in the interior
(verified by unfold/fold). The global score remains, so this does not invalidate the
proper-scoring rationale. Correct the stated weighting, or deliberately normalise
coverage if uniform weighting is intended, before P2 results. Overlapping Euclidean
patch scores are nonlinear, so coverage counts are a diagnostic, not a claim that
each token's loss is exactly multiplied by that count.

### F11. Pin the actor/control recipe before P3

The sourced PMPO alpha=.5, beta=.3, gamma=.997 and reverse prior KL match Dreamer 4;
the formula is `KL(pi_actor || pi_BC)`, not the opposite orientation. Lambda=.95,
255-bin value output, 3-layer/256-unit small MLP and slow rate=.02 match the cited
DreamerV3 settings. These are component precedents, not validation of this offline
combination. [Dreamer 4](https://arxiv.org/html/2509.24527),
[DreamerV3 config](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml).

The BC training dose/batch/schedule, frozen-prior timing, paired initialisation or
warm-start arrangement, and exact actor feature readout are not fully pinned by
`p3.updates` and `bc_data`. Record them before P3; the code plan intentionally defers
P3 implementation. They do not require another user choice if ordinary implementation
defaults are chosen, documented and frozen in advance.

## Source and contract checklist

| Assertion | Assessment |
|---|---|
| 3x3 acuity measurements and honest HUD limitation | verified from saved per-seed results; original selection rule needs amendment wording |
| full-resolution comparison retained; no automatic I1/I2 additions | present in PERCEPTION and TEST_SPEC |
| two temporal layers with four past frames each require eight-frame prefix | consistent with stated architecture; verify full sampler and prefix bookkeeping |
| JEDI LR .3, tanh limit 3, sigma_data 1, three Euler evaluations | sourced; denoiser output-clamping location remains an implementation detail to make explicit |
| churn 1 | JEDI stochastic-experiment choice; M0's declared adaptation, not JEDI's universal default |
| DIAMOND lognormal -.4/1.2, clip .002..20, sampling max5/rho7 | verified in official configuration/code |
| clean Mamba P1, generated Mamba P2; switching removed | declared adaptation; stale "random switching"/"self-conditioning" wording remains in LOSSES:47, CURRICULUM:98, STEP_SPEC:55-57 |
| SIGReg lambda*N | arithmetic follows pinned LeWM statistic; finite sample dependence and gradient balance are not guaranteed invariant |
| world/actor budgets, two world seeds, reserved S3 | explicit and preserved; timings remain estimates |
| all implementation hyperparameters are already specified | too strong: loss weighting, reward value decoding, cache noise lifetime and some module defaults remain to pin |

The action-confounding caveat in DECISIONS remains relevant with reduced acuity:
the expert's symbolic action can reveal facts missing from the pixel history. The
recorded one-step conditional is not automatically an interventional action model.
Retain exogenous-window diagnostics and the real-environment actor test; neither the
one-step loss nor random-play-only long windows settle expert-state counterfactuals.
This is an existing coverage limitation, not a demand for another training campaign.

## Verification and handoff

- `.venv/bin/python -m pytest saccade/tests -q`: **6 passed**, 0.13s.
- `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m pytest d4mj/tests/test_mamba_recurrence.py -q`:
  **3 passed**, 7.44s. Sandbox NVML and SWIG warnings; CUDA checks above ran separately
  with host GPU access and completed successfully.
- CPU contract checks: passed; `contract_results.json` contains exact values and
  hashes of the audited files.
- Actual CUDA SIGReg profile, equivalent-reduction parity, and seq_idx carry check:
  completed; JSON outputs and scripts are in this directory.
- Pinned local upstreams: LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`;
  DreamerV3 `e3f02248693a79dc8b0ebd62c93683888ddaccfe`. Additional installed-kernel
  hashes are in `source_hashes.json`. Web links above identify live primary-source
  pages inspected during this audit; they are not immutable local source pins.

**Handoff:** proceed with data.py and its tests. Incorporate F1-F5 into the actual
module/test contracts, use F6 when measuring memory, and fix F7's evaluator definitions
before freezing the M0 readings. Then run the already-planned smoke/resume/full-profile
checks. Keep P2/P3 details bounded to their own implementation stages. No additional
architecture search or long ablation is justified by this audit.
