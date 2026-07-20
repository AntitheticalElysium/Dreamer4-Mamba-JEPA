# D4-Mamba-JEPA reboot: implementation and preflight record

Date: 2026-07-20
Workspace starting point: `d1ccfa1`
Status: baseline validated; Stage M1 outcome recorded separately

## Plain-language outcome

There is now a separately runnable architecture, not another modification of
the compact experimental world model.

It reuses the pinned MMBench2 D4-style tokenizer and world-model classes,
official Mamba-2, official Crafter, and the specific Dreamer-CDP predictive
representation mechanism. The local code exposes four explicit arms:
`T-BASE`, `M-BASE`, `T-CDP`, and `M-CDP`.

The first real `T-BASE` result is mixed but useful:

- the baseline learns Crafter pixels and latent dynamics;
- it measurably uses actions;
- generated reward ranking improves;
- a real receding-horizon planner executes reproducibly and beat random in a
  three-seed, 200-step exploratory screen;
- terminal prediction is still broken and reward magnitude is still weak.

This is the first result in this track that reaches actual environment actions.
It is not yet Dreamer 4: there is no actor, value function, PMPO update, or
online replay loop.

## Exact source boundary

Primary pins and byte hashes are in `d4_mamba_jepa/SOURCE_MANIFEST.md`.

Reused without editing:

- MMBench2 `src/model.py` at
  `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`;
- official `Mamba2` at
  `f577286d052741c35d39cd43bdc3fad27120f22c`;
- official Crafter at
  `e04542a2159f1aad3d4c5ad52e8185717380ee3a`.

Referenced, not transplanted wholesale:

- Dreamer-CDP at
  `a851fa3e3d70b624b094ee1810ad4bb602346092`;
- unofficial Dreamer-4 JAX at
  `8144b940d801971f12ec5633553b95001e555949`,
  read-only because no license file was found.

The MMBench2 model file is dynamically loaded under an isolated module name
only after its SHA-256 matches. The installed Mamba-2 Python source must be
byte-identical to the pin. Existing replay and Crafter canonicalization helpers
are also hash-checked.

## What was changed locally, and why

The complete numbered record is `d4_mamba_jepa/DEVIATION_LEDGER.md`. The
important changes are:

1. Scale: 64x64 input, width 64, depth 4, 16 tokenizer latents, Kmax 4. This is
   sized for the 6 GB RTX 3060 and is the main capacity caveat.
2. Actions: MMBench2's continuous action MLP is replaced by a 17-way Crafter
   embedding plus a dedicated `-1` start action. Token position and causal
   convention are unchanged.
3. Continuation: a local multi-token continuation head reads the same
   post-transition agent tokens as reward.
4. Mamba: only dynamics `TimeSelfAttention` modules are replaced by official
   Mamba-2. Spatial attention, tokenizer, MLPs, residual order, heads, and token
   layout are unchanged. `expand=1` gives 15,014 parameters versus 16,644 for
   the replaced temporal-attention block; `expand=2` was rejected as a 67%
   capacity confound.
5. Mamba rollout cache: every candidate branches from a cloned clean prefix
   state because official `step()` mutates its cache.
6. CDP: a predictor consumes clean causal agent state plus the next action and
   predicts a stop-gradient next latent with cosine loss. The encoder gets
   0.3x learning rate; ordinary flow/task losses receive detached latents; a
   frozen decoder supplies a reconstruction anchor. This is CDP-shaped, not a
   faithful Dreamer-CDP system.
7. Planning: categorical random shooting is used as the smallest executed
   planner. It is not Dreamer-4 PMPO.
8. Checkpoints: atomic save, strict state/config/source/implementation hashes,
   optimizer, loss normalizers, Torch CPU/CUDA RNG, and explicit NumPy
   Generator state.

The tokenizer reconstruction, flow loss, reward alignment, and Euler rollout
equations were ported from the pinned MMBench2 executable scripts because those
flat scripts are not safe importable modules. Docstrings identify the exact
source function. This port is a first suspect if parity later fails.

## Mechanical evidence

Before real training:

- source/config/action/continuation/CDP/training/rollout/checkpoint tests pass;
- Mamba full-sequence and recurrent-prefix outputs agree;
- cached and uncached rollouts agree for Transformer and Mamba;
- candidate cache isolation passes;
- all four arms complete finite BF16 CUDA forward/backward updates.

CUDA one-update smoke:

| Arm | Total parameters | Trainable | Peak allocated | Result |
|---|---:|---:|---:|---|
| T-BASE | 914,588 | 417,084 | 36,841,984 B | finite |
| M-BASE | 911,328 | 413,824 | 314,674,688 B | finite |
| T-CDP | 939,612 | 689,268 | 133,153,280 B | finite |
| M-CDP | 936,352 | 686,008 | 370,525,184 B | finite |

The first Mamba calls include kernel compilation and are not throughput
comparisons.

On a deterministic moving-square process, 50 tokenizer and 100 world updates
gave:

- full reconstruction MSE: `.26857 -> .01994`;
- flow loss: `.10488 -> .00915`;
- reward loss: `1.49686 -> .39545`;
- continuation loss: `.59615 -> .00199`;
- peak allocated VRAM: 235,809,792 B.

That establishes learnability and timing alignment on a controlled process,
not Crafter competence.

The same controlled run was repeated with Mamba as the temporal backend:

- full reconstruction MSE: `.26857 -> .01995`;
- flow loss: `.10483 -> .01071`;
- reward loss: `1.49606 -> .34379`;
- continuation loss: `.51858 -> .00387`.

This establishes that the official Mamba path also learns the controlled
action-conditioned process, not merely that it has finite gradients.

## Real-Crafter T-BASE preflight

Artifacts:

- tokenizer:
  `outputs/d4_mamba_jepa/preflight_t_base_5k/tokenizer_t_base.pt`,
  SHA-256
  `91a210dc8c76fa29793599ced04190438d776a0c1a757b674691272eeb58b22c`;
- world:
  `outputs/d4_mamba_jepa/preflight_t_base_5k/world_t_base.pt`,
  SHA-256
  `6d4a2a18ed968ab29b0ef32d02f656284647b50714b25d54abfd90884ed079e4`;
- report:
  `outputs/d4_mamba_jepa/preflight_t_base_5k/report.json`,
  SHA-256
  `05d3dfdb0dd8a90800b7e001e0f2df99fee97a14ce9aa047edbd579a4616c984`.

Data:

- 42,979-transition replay, SHA-256
  `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad`;
- previously used 20-episode dev replay, SHA-256
  `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad`.

This is not a sealed final tier. It uses 500 tokenizer updates and 5,000 world
updates at batch 4, sequence length 16, LR `1e-4`, AdamW, 1,000-step warmup,
and the upstream 25% self-row schedule. It ends before the upstream
10,000-update shortcut-bootstrap start.

Runtime:

- tokenizer: 16.79 seconds;
- world: 205.24 seconds;
- peak allocated VRAM: 262,928,384 B.

Key before/after results on fixed dev rows:

| Measure | Before | After |
|---|---:|---:|
| tokenizer full MSE | .21321 | .01988 |
| uniform flow MSE | .30561 | .00520 |
| action-shuffled / true flow loss | 1.000 | 1.216 |
| correct-action latent MSE | .30601 | .00750 |
| wrong-minus-correct latent MSE | .00000 | +.00039 |
| uniform generated reward event AUROC | .5929 | .7509 |
| uniform generated reward Pearson | -.0155 | .2038 |
| uniform zero-target absolute predicted reward | .00018 | .00154 |

The action result is real but not uniform: correct action wins only 48.6% of
individual uniform rows, while its mean is better. On reward-event-aligned
rows it wins 53.2% and improves mean latent MSE by `.00216`.

The main failure is continuation. On all 14 terminal-aligned generated rows,
mean P(continue) is `.99799` and terminal Brier is `.99601`. The implementation
knows the timing contract, but uniform BCE learns the 99.5% majority class.

Reward also remains incomplete. Event ranking improves, but event-aligned
generated reward MAE remains `.4664`; the model mostly underestimates reward
magnitude.

## Executed control

Protocol:
`reviews/2026-07-20-d4-reboot-executed-preflight-protocol.md`.

Artifact:

- `outputs/d4_mamba_jepa/executed_control_t_base_5k.json`,
  SHA-256
  `3309863c880e5dfac6809ed98b3e45148426abc2d3e6470e6618884e6ab37e8f`.

On fresh seeds 2000-2002, capped at 200 steps:

| Policy | Mean return | Mean achievements | Tiny-sample score |
|---|---:|---:|---:|
| uniform random | .733 | 1.333 | .670 |
| random-shooting planner | 1.733 | 2.667 | .843 |

Paired return differences are `+1.0, 0.0, +2.0`; achievement differences are
`+2, 0, +2`. The planner most often used at most 11.3% of one action, so this
was not a single-action collapse. It found `collect_sapling`, `place_plant`,
and `wake_up`.

The warning is equally concrete: seed 2002 died after 27 planner steps versus
91 random steps. That is compatible with the terminal-head failure. Predicted
horizon return has almost no correlation with immediate reward, and a
state-independent planner control has not yet been run. Three short episodes
cannot establish planner superiority.

The entire executed run was repeated. After removing wall time and throughput,
every substantive field—including actions, returns, achievements, predicted
scores, and summaries—was exactly equal.

## Final validation

- `python -m compileall -q d4_mamba_jepa`: pass.
- New-track tests: 40 passed.
- Repository-wide tests: 229 passed in 77.92 seconds.
- One warning remains in the pre-existing compact BF16 mechanism test; there
  are no test failures.
- `git diff --check`: pass.
- Generated checkpoints and reports remain ignored local evidence rather than
  repository payload. No push was made.

## What this does and does not establish

Established:

- the new track is runnable and learns;
- the Transformer baseline is not silently broken;
- the discrete action path is used;
- imagination produces action-dependent futures;
- the planner reaches and acts in official Crafter;
- source/checkpoint/trajectory provenance is reproducible.

Not established:

- Mamba improves compute or modeling;
- CDP/JEPA improves representation or planning;
- the tiny planner result generalizes;
- the model is Dreamer 4;
- the reward and continuation heads are safe enough for policy learning.

## Current first suspects and next move

1. Fix the baseline continuation objective with one source-backed
   imbalance-aware formulation, held common across all arms. The present
   failure is not attributable to Mamba or JEPA because both are off.
2. Retain aligned generated reward, false-reward, paired action shuffle, and
   executed behavior as diagnostics; do not turn them into a giant conjunctive
   veto.
3. The matched `T-BASE`/`M-BASE` pair has now passed. Its exact evidence and
   limitations are in
   `reviews/2026-07-20-d4-stage-m1-matched-backend-outcome.md`.
4. Next run matched `BASE`/`CDP` arms. CDP must earn its complexity against
   the base objective; it is not assumed to repair reward or termination.
5. Actor/value/PMPO and online replay are a separate final stage. Until then,
   call this a D4-style world model with random-shooting control.

The work from the compact repository was not wasted: its transition indexing,
source pins, replay, canonicalization, checkpoint/RNG discipline,
generated-state reward and continuation readouts, action-ranking instinct, and
false-reward warning were reused. The large protocol stack was not carried
over as a prerequisite to acting; the new track reached an executed planner
after one baseline training run.
