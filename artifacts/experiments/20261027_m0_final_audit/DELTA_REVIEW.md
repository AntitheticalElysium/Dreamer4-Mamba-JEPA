# Concurrent revision review — e1b4cb40

During the audit requested for fc120c63, the design changed and was committed as
`e1b4cb40c68b90b5988fba7fd7942f4abf7b37e8`. I did not make those changes. Its exact
diff and the eight changed files are captured in `working_delta.patch` and
`working_snapshot/`, with hashes. This review supersedes the original report's
current-default statements about SIGReg and ambiguous frame-gradient routing.

**Current verdict:** data-stream work can proceed. Before the long P1 run, fix the
normalization's causal contract and close the EDM loss, complete sampler, and reward
decoding specifications from F2-F4 of the original review. This remains a bounded
implementation pass, not an invitation to restart architecture research.

```text
frames [past | future] --> chunk-wide training BatchNorm --> past percept --> Mamba
                 \______________________^                         |
                     future statistics                            v
                                                     predict that future
```

## What the revision resolves

- Denoiser frame inputs and labels are detached; denoising trains the encoder through
  clean past-percept -> Mamba -> memory-token conditioning. This is a coherent choice
  from F1. State this as "frame inputs", not "every denoiser input": memory tokens
  must remain differentiable. LOSSES principle 3 still mentions the superseded direct
  past-frame route and SIGReg.
- The glimpse tokens now have their own pixel-reconstruction objective. This is a
  sensible grounding path: predicting a glimpse from the periphery alone does not
  directly require the glimpse encoder to preserve its own content. It remains a
  learning claim, not proof that every relevant entity survives joint optimization.
- SIGReg is no longer the M0 default. The measured 1914 -> 224 MiB equivalent reduction
  is retained as evidence for the documented comparison, not a new default or a reason
  to add the regularizer back.

## D1. Ordinary chunk-wide BatchNorm violates causal, frame-local perception

**Measured mechanism, before implementation.** The new CURRICULUM specifies
microbatch statistics over roughly 83k tokens; M0_CODE specifies
`BatchNorm1d(affine=False)` after tanh. Those statistics include later frames of the
same chunk. Therefore the percept of time t is a function of the recorded future,
including while it is read by the supposedly clean, causal Mamba conditioning path.
Detaching the denoiser's frame inputs does not remove this route.

The reproducible CPU check changes only future input tokens:

| Check | Measured result |
|---|---:|
| earlier output change, train-mode BatchNorm | max 0.814805 |
| derivative of earlier output w.r.t. future inputs | max 0.220423 |
| earlier output change with fixed-stat eval normalization | 0 |

This is an ordinary BatchNorm property, not an empirical claim about how much M0 would
exploit the leakage. No saccade encoder exists yet. The counterexample is enough to
show that chunk-wide training normalization does not satisfy the declared interface.

It also makes a percept depend on the other episodes in its training microbatch.
The fact that 16x64x81 and 4x256x81 contain the same number of tokens does not make the
statistics equally independent, nor make train-mode chunk perception identical to
one-frame deployment.

**Required closure:** use a causal normalization convention for the percepts that
enter belief/targets, and test it explicitly. Fixed/running statistics held constant
for the relevant forward pass are one small option; any EMA updates must have a
declared timing that does not use a frame's future to encode its past. The exact
engineering choice can be made within the build. Test that replacing future frames,
other episodes, or padding leaves the current percept unchanged, and that streaming
and chunked perception agree under the declared normalization state. Retain the
intended whole-frame lighting-statistics path; the issue here is extra dependence
across time and episodes.

Freezing encoder parameters during P2 must also freeze its normalization buffers and
use the declared fixed-stat forward mode. `requires_grad_(False)` alone does not
freeze BatchNorm running statistics. Save these buffers in checkpoints, and avoid
updating them again on prefix replay or activation-checkpoint recomputation.

## D2. BatchNorm after tanh removes the declared bound

The same check feeds values bounded by +/-3 through non-affine BatchNorm and obtains
an absolute output of 9.949735. Unit channel variance is not an elementwise bound.
Do not assume JEDI's [-3,3] latent range still holds after this normalization, or clamp
denoiser outputs to that range while asking them to match normalized targets that can
exceed it. Specify the domain in which scaling/clamping happens. This needs a unit
test, not a new benchmark.

## D3. REPA-E supports a mechanism, not a collapse-prevention guarantee

The source is relevant, but the new wording overstates what transfers. REPA-E trains
with a pretrained visual-feature alignment target and VAE losses as well as its
stop-gradient and normalization. Its official code applies the normalization in the
diffusion model and uses eval-mode BN for the VAE-alignment update, then train-mode
BN for the detached-latent diffusion update. It is not a causal recurrent observation
encoder with one shared forward convention.
[Official paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Leng_REPA-E_Unlocking_VAE_for_End-to-End_Tuning_of_Latent_Diffusion_Transformers_ICCV_2025_paper.pdf),
[training code](https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py#L333-L380),
[normalization code](https://github.com/End2End-Diffusion/REPA-E/blob/main/models/sit.py#L199-L202).

M0's routing plus TSP/change/reconstruction/outcome grounding is a defensible proposed
counter to collapse. It should be described as the chosen training mechanism whose
entity retention is measured in S1, rather than as something that prevents collapse
by construction. Normalization can also give unit marginal variance without ensuring
semantic content or high joint rank.

## Remaining original findings

F2 (EDM weighting), F3 (complete sampling/cache semantics), F4 (reward value decoding),
F5 (prefix/carry ownership and kernel sequence IDs), F7 (HUD/control/precision
references), F8 (acuity-rule wording), and F9-F11 (P2/P3 contracts) still apply.
The selected 3x3 acuity and fixed training budgets do not need reopening.

Evidence: `batchnorm_results.json` and `check_batchnorm.py`; original results and
commands are in `REVIEW.md`. These are contract checks, not toy learning results.
