# D4-lite Stage M1: matched Transformer versus Mamba protocol

Registered: 2026-07-20, before runner implementation or outcome.

## Question

When everything except the dynamics temporal operator is held fixed, is the
official Mamba-2 replacement a viable backend for this D4-style world model?

This is a one-training-seed feasibility discriminator. It cannot establish
population-level architecture superiority or long-context scaling.

## Frozen inputs

- Tokenizer:
  `outputs/d4_mamba_jepa/preflight_t_base_5k/tokenizer_t_base.pt`
- Tokenizer SHA-256:
  `91a210dc8c76fa29793599ced04190438d776a0c1a757b674691272eeb58b22c`
- Training replay:
  `data/replay_40k_v1.pt`
- Training replay SHA-256:
  `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad`
- Diagnostic dev replay:
  `data/heldout_20ep_v1.pt`
- Dev replay SHA-256:
  `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad`

The dev replay is already spent and may only support this feasibility screen.

## Arms

- `T-BASE`: unchanged pinned MMBench2 temporal attention.
- `M-BASE`: only dynamics temporal attention is replaced by the pinned
  official Mamba-2 implementation.

Tokenizer, spatial attention, token layout, action adapter, reward head,
continuation head, objectives, optimizer, batch schedule, evaluation rows, and
planner code remain identical.

## Pairing contract

- Shared initialization seed: 20260721.
- Shared training-noise seed: 20260722.
- Shared replay-schedule seed: 20260723.
- Generate the complete `(episode index, window start)` schedule once and hash
  it before either arm trains.
- Construct both worlds from the same initialization seed.
- Every state tensor common to the two architectures, excluding only dynamics
  temporal-module keys, must be bit-identical before training and produce one
  shared-state digest.
- Reset Torch CPU and CUDA RNGs to the shared training-noise seed before each
  arm. Their final RNG digests must match.

Any pairing failure invalidates the comparison.

## Training

- World updates: 5,000 per arm.
- Batch size: 4.
- Sequence length: 16.
- Optimizer: AdamW, LR `1e-4`, weight decay `1e-2`, betas `(0.9, 0.999)`.
- Gradient clip: 1.0.
- Warmup: 1,000 updates.
- Shortcut self rows: 25%, bootstrap starts at update 10,000. Therefore this
  screen measures the empirical flow phase and does not claim shortcut
  consistency is trained.
- BF16 autocast on the RTX 3060 Laptop GPU.

## Continuation ruling

No class weighting or terminal oversampling is added in M1. The pinned
Dreamer-CDP implementation uses an ordinary binary head and unweighted binary
log loss (`agent.py`, `heads.py`, and `outs.py`, hashes in the source
manifest). Introducing local weighting now would confound the temporal-backend
comparison. The known terminal failure is reported but is not attributed to
either backend in this stage.

## Evaluation

Both arms use identical fixed:

- 16 uniformly sampled dev batches of 32 windows (512 generated rows);
- every eligible reward-event-aligned transition (141 generated rows);
- every terminal-aligned transition (14 generated rows);
- flow noise seeds;
- action-shuffle permutations;
- K=4 generated-state noise seeds.

Report tokenizer/source/core/runner hashes, exact checkpoints, initialization
and schedule digests, loss curves, throughput, peak VRAM, flow error,
correct-versus-wrong action prediction, action-shuffle ratio, generated reward
ranking and false reward, and continuation behavior.

## Preregistered decision

Mamba is a viable backend for the next CDP factorial only if:

1. all source, shared-initialization, schedule, RNG, finite-gradient,
   checkpoint, and evaluation integrity checks pass;
2. M-BASE uniform flow MSE is at most `1.25 * T-BASE`;
3. M-BASE paired action-shuffle ratio is at least `1.05`;
4. M-BASE mean wrong-minus-correct generated latent MSE is positive;
5. M-BASE uniform generated reward event AUROC is no more than `.05` below
   T-BASE;
6. M-BASE uniform zero-target mean absolute generated reward is no more than
   `2.0 * T-BASE`.

No throughput win is required at sequence length 16. The original
long-context-compute hypothesis requires a later matched length sweep.

Pass means: retain Mamba and move to the `BASE` versus `CDP` factorial.
Failure means: stop before CDP and localize whether the cause is recurrent
implementation, optimization, or insufficient scale.

## Post-outcome reproducibility amendment

Added after the first outcome and its repeat, before accepting the result as
commit-quality proof.

The first paired run passed every frozen gate. Its immediate repeat reproduced
the decision and all reported Mamba metrics to within `8e-4`, but the final
Mamba tensors were not bit-identical (`1.80e-4` relative L2 difference).
Transformer tensors, schedules, initialization, and CPU/CUDA RNG digests were
exact. A focused two-repeat kernel audit then became bit-identical only after
enabling PyTorch deterministic algorithms and the official Mamba deterministic
path.

Therefore the final evidence runs additionally require:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `TRITON_CACHE_AUTOTUNING=1`;
- `torch.use_deterministic_algorithms(True)`;
- deterministic cuDNN with benchmarking disabled;
- a byte match between the installed and pinned official
  `mamba_ssm/utils/determinism.py`.

The source hash is recorded in `SOURCE_MANIFEST.md`, the settings are embedded
in every final report and checkpoint, and the runner refuses a CUDA process
that initialized CUDA before installing this contract. No arm, input,
initialization, schedule, loss, optimizer, training budget, metric, threshold,
or gate was changed. The original runs remain negative evidence about the
default CUDA path and are not relabeled as deterministic.
