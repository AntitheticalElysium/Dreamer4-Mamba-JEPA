# Integrating M0–M3 into the existing system

Implemented 2026-09-06 against `162efd1`. The initial implementation created too much parallel infrastructure. Joint training now uses the existing package's data, training, checkpoint, gate and execution infrastructure. The existing Flow/Direct × Attention/Mamba baselines remain active callers of that infrastructure.

## Ownership and boundaries

| Responsibility | Final owner and integration |
|---|---|
| Episodes, validation and sampling | `data.py` owns `Batch`, `JointBatch`, `BridgeBatch`, corpus lineage and all canonical samplers. M4 keeps ragged observed burn-in separate from main loss windows and records episode/start identities. |
| Latent caches | Shared `cache.py` owns identities, family-specific encoding adapters, loading and one verified resumable archive writer. Joint exports require an immutable parent and resume only from hash-valid registered shards under the identical contract. |
| Training | `train.py` owns the original phase trainers plus `train_joint`, `train_bridge` and `train_actor_lewm`. Shared optimizer mechanics retain phase-specific objectives, ownership and schedules. Counterfactual fork supervision is not a canonical callback. |
| Recipe handling | `config.py` owns canonical serialization, hashing, JSON loading and family dispatch. The original flat `Config` defaults remain intact. Separate nested `LeWMConfig` settings prevent invalid combinations of legacy and joint fields. |
| Gates | `gates.py` owns `Gate`, dependency handling, component reports, `preflight`, `ComponentGateError` and `require_joint_gates`. It runs the original six gates and the new family's probes. A failed prerequisite blocks dependent components; the architecture verdict remains unevaluated. |
| Runtime state | `world_api.py` contains concrete legacy and LeWM adapters, constructed through `ModelBundle.create` or `from_models`. No adapter duplicates transition mathematics. `state.py` owns both state types and the spatially correct legacy memory repeat operation. |
| Existing callers | `execution.run_episode` and `imagination.imagine` advance through the adapter. `diagnostics.rollout_predictions` provides a shared observed-prefix/generated-successor path, used by the existing `multistep_error` and `latent_stats`. Existing raw-world call signatures remain accepted. |
| Commands | `python -m d4mj` exposes preflight, resumable paired joint training, export, bridge, actor and real evaluation. `experiments.py` orchestrates shared modules; campaign directories contain no live architecture implementation. |
| Checkpoints | `checkpoint.py` owns distinct legacy, joint, bridge and actor formats. Numbered states are immutable; bridge/actor bind source, cache, parent, gate, modes, gradients, optimizer/RMS and every RNG stream. |

`lewm.py`, `lewm_config.py`, `mamba_recurrence.py` and `lewm_diagnostics.py` remain separate for concrete reasons: the framewise encoder/objective differs from MAE; nested recipes have different validity conditions; functional Mamba carry has different gradient semantics from the legacy source inference cache; and source/objective/numerical probes are model-specific. The existing `representation.py`, `transition.py`, `time_mixer.py`, head/target code and actor-critic mathematics continue to serve the baseline family. Their specialized training losses and diagnostic targets are not forced through a generic signature.

## Preserved contracts and deliberate differences

- Legacy memory includes the current committed frame and carries separate temporal encoder memory. LeWM memory includes only completed `(z, outgoing action)` pairs; its current latent remains unconsumed. After a C-frame prefill, their relative step counts are C and C−1 respectively.
- Legacy Flow retains explicit world RNG, corruption and stochastic commits. Direct draws no world noise. LeWM remains deterministic. Policy and world generators stay independent.
- Legacy `advance` retains S55's detached incoming memory. LeWM retains differentiable conv/SSM carry and accepted latents. Fork/detach/repeat preserve storage ownership; legacy repeats unflatten batch and spatial slots before branching.
- MAE cache identity, bounded temporal encoding and cache schema are preserved. LeWM cache identity still includes projector parameters, BN buffers, preprocessing and imported ViT implementation; it stores unbounded float32 `[T,1,192]` latents. The shared writer validates registered shard bytes before resuming and refuses to overwrite orphan shards.
- Both optimizers honor Mamba's `_no_weight_decay`. Legacy phases continue to decay ordinary vectors/biases; joint training explicitly excludes them. Joint warmup/cosine and legacy warmup schedules remain different. No research hyperparameter, source tolerance, statistical batch or adaptation group changed in this refactor (TC-33).
- Shared execution enables LeWM only from checkpoint capabilities and accepted gates; recipe presence never authorizes it. H2, H16 and the 500-update actor screen are hard boundaries. Renderer/play remain closed. Mechanical rollout support is not evidence of a learned horizon.

## Verification

The retained [evidence](../../../artifacts/experiments/20260905_m0_m3_validation/evidence/integration/) includes:

- **Legacy git-reference parity:** all four Flow/Direct × Attention/Mamba arms on the RTX3060. Existing diagnostic outputs, imagined trajectories, RNG advancement, chunked latent caches/cache identities and optimizer updates match exactly. A deterministic simulated environment produces identical execution actions and episode results. The reference caller functions are loaded directly from git commit `162efd178ef72a8eca05730fe3996358162054ad`; this does not measure live Craftax policy quality.
- **Original joint-trainer parity:** four CPU verification updates using the original optimizer, sampler and update functions, with only moved imports redirected. Encoder/world weights, BN buffers, metrics and sampled windows are exact. The record identifies the compared original source bytes. This small test checks the refactor; it is not a replacement architecture or resource gate.
- **Full architecture GPU gates:** both raw and TC use B128/F4/J1024 BF16, all six 256-wide recurrent layers and the native 63-pixel encoder. All eight component gates pass on the RTX3060 Laptop GPU. Peak allocated/reserved memory remains 952,169,984 / 1,193,279,488 bytes. Only TRAIN windows from the first hash-verified support shard enter the resource probe; it is a technical fixture, not a selected research corpus.
- **Full architecture GPU resume:** four updates versus two plus two resumed updates produce bit-exact weights/BN and exact metrics under the unchanged 10,000-update schedule. The earlier immutable checkpoint retains its SHA256 and remains indexed.
- **Regression tests:** the full CPU suite and targeted CUDA tests cover sampler/cache lineage, v1/v2 rejection and resume, state timing, observation memory, spatial branch order, recurrence gradients, source parity, existing execution and CLI family dispatch. **199 passed, 7 skipped** in the full CPU suite; **21 passed, 4 skipped** in the targeted CUDA invocation. The latter skips are old tests whose fixture explicitly selects CPU. Commands and logs are retained in the evidence README.

Temporary before-refactor legacy traces were unavailable after an environment refresh, so the legacy parity check uses git-recovered reference functions instead. Original joint trainer/data copies survived long enough for the recorded comparison; their source hashes are retained, while temporary training artifacts are not research checkpoints.

## Provenance and remaining scope

The joint source closure now hashes the shared runtime files. Old M0–M3 reports remain historical under `evidence/m0_m3`; fresh reports are under `evidence/integration`. A pre-refactor joint gate/checkpoint cannot silently resume against changed source files. Re-run preflight for the final recipe/data/runtime before a new run. Legacy v1 source and encoder-cache contracts remain unchanged.

M4 recursive training, outcome targets, frozen-world actor and paired real execution now use this shared runtime. Their presence does not establish Craftax retention/control performance: identity-bound G2–G4 evidence still gates every continuation, and a negative component report stops at that component.
