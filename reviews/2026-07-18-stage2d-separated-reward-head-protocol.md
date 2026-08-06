# Stage-2D protocol: separated reward-head state factorial

Status: **pre-registered before implementation, fitting, or outcome inspection**

Date: 2026-07-18

Authority: the user authorized corrective implementation and the Stage-2C
outcome routes explicitly to a separated task-head diagnostic.

## Question

Stage-2C establishes two facts on the spent DEV tier:

- C-L improves generated latent accuracy at K1/K2/K4/K8 but loses deployed
  reward quality and fork action ranking.
- Adding generated reward gradients through the shared trunk restores reward
  and ranking, but creates false reward and gives back part of the latent gain.

This experiment asks whether the reward readout can be repaired **without
altering the representation**:

1. Is another matched amount of ordinary reward-head fitting sufficient?
2. Does fitting the same reward head on naturally sampled generated K1/K2
   states add value beyond the equal-update teacher-forced control?

This is a local causal control motivated by the Stage-1 equal-update result.
It is not a reproduction of Dreamer, SPR, or V-JEPA 2.

## Fixed base and data

- Base: C-L checkpoint `reviews/artifacts/stage2c_cl_s505.pt`.
- Base checkpoint SHA-256:
  `227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1`.
- Base full-state digest:
  `a0cf4ec132a9e023ecf71fa63d7f1f8e17dd00d6080684f1e7b6962844b8c1c9`.
- Initial reward-head digest:
  `091e08894efc407b7b8d1cd2b4af375adadf340c4f603c53ff3e23a9fa8ac7f3`.
- Initial non-reward digest:
  `c44815c4236b748fb4f95d0f82a14671aa7ddbc462d6cd3c62cf388e5686c6c5`.
- Replay SHA-256:
  `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad`.
- Natural uniform schedule seed: `10_505`.
- Schedule: 3,000 updates x batch 8 = 24,000 sampled windows.
- Schedule SHA-256:
  `d8ed746758296f365282823eba8595751b407d616c96b93e8f8417904126fc4c`.
- The schedule contains 18,195 unique windows. Across all nine aligned labels,
  its reward-event fraction is `.038185`; at the two final labels it is
  `.042938`.

Evaluation reuses only the already-spent Stage-2 DEV tier:

| Bundle | SHA-256 |
|---|---|
| natural seeds 960-975 | `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e` |
| terminal seeds 932-947 | `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf` |
| fork seeds 143-150 | `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8` |

The FINAL tier must not be indexed, deserialized, evaluated, or used for
selection.

## Frozen contract

Only the six tensors in `world.reward` are trainable. The encoder, action
input, temporal backend, future predictor, continuation head, reliability
system, and all buffers remain frozen and must be bit-identical to C-L.

The world stays in evaluation mode while contexts are generated under
`torch.no_grad()`. Only the reward head is placed in training mode. This
prevents accidental stochastic or mutable-buffer differences in the frozen
trunk.

Both arms use:

- GRU seed 505 and the exact C-L reward-head initialization;
- 3,000 AdamW updates, batch 8, learning rate `1e-3`, default AdamW
  hyperparameters, gradient clipping 100, and bf16 context generation;
- the exact same replay windows, ordering, nine reward labels per sample, and
  optimizer-update count;
- natural uniform sampling only;
- no event/terminal pool, class reweighting, continuation loss, latent loss,
  world-model loss, actor/critic loss, or parameter-dependent curriculum.

## Arms

For a ten-observation window, reward `r_t` belongs to transition
`obs_t --a_t--> obs_{t+1}`. Each arm consumes labels `r_0,...,r_8`.

- **D-R — real-state equal-update control:** observe `obs_0,...,obs_9`;
  train the reward head on the nine post-transition contexts after
  `obs_1,...,obs_9`.
- **D-G — generated-state arm:** observe `obs_0,...,obs_7`; use the seven
  post-transition real contexts after `obs_1,...,obs_7`, then imagine with
  `a_7` and `a_8` and use the generated K1/K2 contexts for `r_7,r_8`.

Thus the sole intervention is the state distribution at the last two aligned
head inputs. Labels, sampling, loss, initialization, updates, and trainable
parameters are matched.

## Required executable checks before fitting

1. Synthetic transition IDs prove the nine context/label alignments.
2. D-R and D-G consume exactly the same reward-label tensor and schedule.
3. The first seven post-transition contexts are exactly equal between arms.
4. Only `reward.*` requires gradients and receives optimizer state.
5. A backward pass leaves every non-reward gradient absent.
6. An optimizer step changes the reward digest and leaves the non-reward
   digest bit-identical.
7. Checkpoint loading asserts the C-L config, checkpoint hash, full digest,
   reward digest, and non-reward digest.
8. Evaluation asserts D-R/D-G latent and continuation raw arrays are exactly
   equal to C-L.
9. Paired-analysis tests cover contrast direction, false-reward ceilings,
   ranking gates, and FINAL-manifest non-access.
10. The focused and full CUDA suites plus compileall and `git diff --check`
    pass.

## Evaluation

Reuse A and C-L raw rows from the hash-pinned Stage-2C artifact. Evaluate D-R
and D-G on the identical natural, terminal, and fork targets.

Report:

- reward metrics at K0/K1/K2/K4/K8;
- fork chosen-minus-random, regret, within-anchor correlations, and raw/gated
  zero-suffix predicted returns;
- exact latent/continuation identity checks;
- paired episode-cluster reward contrasts and environment-seed-cluster
  ranking/zero-suffix contrasts;
- head/non-head hashes, schedule/data/script/commit provenance, wall time, and
  peak VRAM.

Primary contrasts:

- `generated_effect = D-G - D-R`;
- `real_extra_fit = D-R - C-L`;
- `generated_vs_latent = D-G - C-L`;
- `real_vs_baseline = D-R - A`;
- `generated_vs_baseline = D-G - A`.

## Gates

This is a one-seed diagnostic on a spent DEV set. No pass can directly license
planner execution or an architectural claim.

### I — isolation invariant

For both D-R and D-G:

- non-reward state digest is exactly the registered C-L digest before and
  after fitting;
- latent and continuation predictions are elementwise identical to C-L;
- only the reward checkpoint differs.

Any failure invalidates the experiment.

### M — generated-state mechanism, D-G versus D-R

- K8 reward AUROC, signed Pearson, and decoded event magnitude improve as
  points;
- at least one of the paired K8 AUROC/Pearson intervals excludes zero in the
  favorable direction;
- ranking advantage/regret do not significantly worsen;
- absolute zero-suffix predicted return does not significantly worsen.

M is reported separately. Its failure does not prevent D-R from being the
better operational head.

### C — reward-head candidate gate, evaluated separately for D-R and D-G

Against C-L:

- K8 event AUROC, average precision, signed Pearson, decoded event magnitude,
  and event MAE all improve as points;
- at least one of the paired K8 AUROC/Pearson intervals excludes zero in the
  favorable direction;
- fork chosen-minus-random improves and regret falls, with both paired
  intervals excluding zero in the favorable direction.

Against A:

- K8 average precision is at least A's point estimate;
- K0 and K1 AUROC/Pearson are not significantly worse;
- ranking advantage and regret are not significantly worse;
- absolute zero-suffix predicted return delta and its paired CI upper bound
  are both `<= +.02`;
- K0 and K1 zero-reward MAE deltas are each `<= +.005` and are not
  significantly worse.

Passing C means only “promising separated reward head on spent DEV.” The
inherited C-L continuation K1 safety failure remains unresolved.

## Outcome-independent routing

- **I fails:** implementation invalid; repair and rerun the same protocol.
- **I passes, neither candidate passes:** stop head adaptation. Diagnose the
  scalar two-hot reward parameterization/calibration before more world-model
  training.
- **D-R passes, D-G fails or M fails:** ordinary extra head fitting is the
  supported intervention; generated-state exposure is not.
- **D-G passes and M passes:** generated-state covariate shift is supported;
  replicate the isolated head intervention on a genuinely fresh evaluation
  tier before any planner gate.
- **Either candidate passes:** continuation must be handled in a separate,
  frozen-trunk calibration step; no full-world objective, Mamba transfer,
  FINAL evaluation, planner execution, actor/critic, or online policy is
  licensed here.

## Outcome (appended after fitting)

Independent outcome record:
`reviews/2026-07-18-stage2d-outcome-and-independent-review.md`.

- I isolation: **PASS exactly**. Both arms retain the registered non-reward
  digest, and latent/continuation raw predictions are elementwise identical
  to C-L at every depth.
- M generated-state mechanism: **FAIL on safety**. D-G improves K8 Pearson
  over D-R by `+.04879 [+.01245, +.12829]`, but increases absolute
  zero-suffix return by `+.01260 [+.00949, +.01681]`.
- D-R candidate: **FAIL**. Aggregate ranking is exactly C-L's and deep reward
  does not improve.
- D-G candidate: **FAIL**. K8 discrimination partially improves, but ranking
  falls below C-L and is significantly worse than A.
- Registered route: **STOP reward-head adaptation on C-L**. The next allowed
  diagnostic is low-capacity, split-safe calibration of frozen C-LR outputs;
  no shared-world update or Mamba transfer.

The FINAL tier was not accessed.
