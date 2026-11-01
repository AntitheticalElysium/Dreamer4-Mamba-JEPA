# Budgets, S3 blockers and P2 summary: follow-up review

Reviewed against current commit `1367a1d7`, the pasted proposal, current M0 contract and the stand-in timing script. No design or package edits. Two bounded CPU checks are in [proposal_check.py](proposal_check.py); they establish limits of the proposed score and blocker, not a learned-model result.

**Disposition:** keep the proposed frame/update/seed caps as an initial bounded experiment. Strengthen the aggregate bookkeeping blocker and replace the pooled global P2 summary with a full percept-trajectory vector. Declare the discrete-sampling gradient boundary. These are small corrections within M0, not reasons to reopen the architecture.

**Budgets and runtime.** The checkpoint cap, independent world seeds, reserved actor budget and prohibition on extending an unsuccessful run are sound experimental choices. Ten epochs are a chosen cap, not a measured convergence requirement for this model. The real implementation is still absent.

I inspected `/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/e3e54d67-08a5-45eb-a859-a08987e119c0/scratchpad/review/m0_cost.py`. It is explicitly a stand-in. Its P1 path runs a Mamba sequence but not the separate action predict/percept correct steps; imagination and P2 keep the memory vector fixed, and omit the recurrent belief update, reward and termination heads. P1 omits TSP, change-map and SIGReg objectives; P2 uses a stand-in distance rather than the proposed patched/global loss. Thus the timings do not establish full-M0 throughput or peak memory. The proposed real-model benchmark is necessary; repeating the stand-in would not settle this.

Benchmark the actual 16-by-64 update, four accumulated 4-by-256 microbatches, H=16 P2 with two samples plus the continuing one-step loss, and the full actor rollout/update. Include the optimizer step, losses, data transfer and state ownership/reset path. Establish wall time and memory headroom; keep the scientific frame budget fixed if throughput is slower.

Before padding/reset masks, the P1 budget already determines update counts:

| Half | Frames | Frames per optimizer step | Optimizer steps |
|---|---:|---:|---:|
| 16 streams x 64 | 12.8M | 1,024 | 12,500 |
| 16 streams x 256, accumulated | 12.8M | 4,096 | 3,125 |

Throughput determines the time estimate; valid-frame counting determines exact step counts. Checkpoint and learning-rate schedules should use a declared clock: 80% of optimizer steps occur in the first half of frames. Withholding another 5% of train episodes also makes the effective epoch count slightly larger than ten.

Accumulation must mean four groups of four distinct episode streams, each with a complete 256-step gradient path, followed by one optimizer step. Four consecutive 64-step chunks of the same four episodes do not provide the claimed diversity. Normalize by valid frames, clip after accumulation, and detach state at the declared boundary. Ordinary additive losses accumulate cleanly; SIGReg depends on batch statistics, so its microbatch estimator is not automatically identical to a full 16-stream estimator. State which is used rather than assuming equivalence.

Use one fixed train-monitor episode split across all arms. Define the checkpoint-selection metric and its examples/noise/fixations before training; a learned harmonised loss alone is not a fixed scientific yardstick. Interpret three actors per each of two world seeds as nested replications: two independently learned worlds, not six. Report each world's gates and actor/BC comparison, including worlds blocked before S3.

**The aggregate bookkeeping blocker misses complete loss of a kind.** I used all 2,236 recorded exogenous dev windows, retained the real initial frame and removed every arrow from the subsequent frames. These are the same window and terminal conventions as S0, using ideal labels to isolate the blocker.

| Proposed blocker quantity | Real | All future arrows removed |
|---|---:|---:|
| Mean visible count | 1.052633 | 1.008106 |
| Relative count change | — | -4.23%, passes the 30% bound |
| Four-step persistence | 0.936990 | 0.973972 |
| Persistence change | — | +0.036981, passes the -0.10 bound |

The persistence average improves when short-lived arrows stop contributing future start entities to its denominator. This is a limitation of that aggregate, not evidence the corrupted worlds are better. The perception gate cannot exclude a denoiser dropping arrows that its encoder still represents. Evidence: [proposal_bookkeeping.json](proposal_bookkeeping.json).

Keep the blocker count small, but apply bookkeeping checks by entity kind where supported, using the frozen reader on real and generated samples, with sample counts and uncertainty. Recheck the declared corruptions through that reader. Keep precision and per-kind perception results visible too: pooled recall/kind accuracy can conceal rare classes and does not constrain false positives. The 0.90/0.02/30%/0.10 values can serve as declared gross-failure tolerances; passing them does not establish physical fidelity. Include probe-seed variability, not only episode-bootstrap errors.

**The local trajectory patches are a useful proposal; a pooled global term leaves a known gap.** Time-stacking a local patch captures local temporal dependence. It does not identify the joint law of distant parts of the scene. Overlapping patches do not remove that limitation.

Exact example, for mean pooling: in each trajectory select one of two patterns with equal probability and keep it for all 16 steps. Distribution P places ones on either diagonal pair of the four grid corners. Distribution Q places ones on either the top pair or the bottom pair. Every 3x3 space-time patch has the same distribution under P and Q. Both always have two active corners, so all mean-pooled trajectories are identical. Lighting/outcomes are constant and the fixed glimpse is central.

Enumerating independent draws exactly gives a summed patched score of 4 under either forecast, and a pooled global score of 0 under either forecast. A full-grid global score is 2 for the correct forecast and 3.656854 for the wrong one. This is an information limitation, not Monte Carlo variance or a learnability experiment. Other pooling operators require their own contract; an unspecified pooled feature should not be assumed to preserve the full joint distribution. Evidence: [proposal_checks.json](proposal_checks.json).

Use a positive-weight global energy score on the flattened full trajectory (all periphery tokens, lighting, glimpse, reward and termination) alongside the patches. This is approximately 41,472 periphery coordinates at H=16 and D=32; the distance computation adds no network and is small compared with the denoiser. Profile it rather than pooling solely on presumed cost grounds. Keep the fixed encoder, train-only scaling, a standard-deviation floor, beta=1, and independent samples.

Pacchiardi et al. explicitly identify patched scores as non-strictly proper and use a score on the full field to restore strict propriety; magnitude balancing is supported, but pooling the global field is a substantive change. See sections 3.2.2–3.2.3 and E.4.2 of [the paper](https://jmlr.org/papers/volume25/23-0038/23-0038.pdf). Two samples are sufficient for an unbiased score estimate; that is not a claim that its gradients have low variance in this task. [NeuralGCM](https://arxiv.org/html/2311.07222v1) uses two members for CRPS.

**The gradient cutoff still needs a discrete-sampling contract.** Continuous denoising can use pathwise derivatives. Bernoulli termination, mixture-component selection and hard reward-bin sampling do not automatically admit those derivatives. A sampled terminal mask also changes which later steps exist. Simply saying gradients flow through the last two steps leaves these cases unresolved.

The minimal option is to detach the hard reward/termination samples explicitly and train those heads by the continuing one-step likelihood losses. Then describe P2 as a truncated pathwise surrogate through continuous percepts and the belief update; it does not provide the full score gradient for discrete outcomes. If score-function or straight-through estimators are intended instead, specify and check them. There is no need to add a new estimator merely to preserve a stronger theoretical claim.

During implementation, verify the intended gradients reach the denoiser/update and do not reach frozen encoders or detached earlier steps. Shared parameters still change the early rollout on the next update; detachment only removes the early-step causal gradient paths. Fixations shared across targets and samples must be exogenous, rather than selected using the recorded future's consequences.

One additional weighting fact: 49 valid 3x3 patches cover a corner token once and a central token nine times. That imposes a spatial weighting, including on the HUD at the boundary. Declare it as a choice rather than calling it uniform. The full global term helps preserve coverage without introducing another module.
