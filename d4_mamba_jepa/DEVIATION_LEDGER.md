# Deviation ledger

Every departure from a pinned primary implementation must be entered here
before training. A change is not allowed to hide behind a generic "D4-Mamba"
label.

Status values:

- `PLANNED`: documented, not implemented.
- `IMPLEMENTED`: code and unit tests exist.
- `SCREENED`: smallest discriminating runtime check passed.
- `REJECTED`: implementation retained only as evidence or removed.

| ID | Status | Local change | Source contract retained | Why required | Isolation / first failure suspect |
|---|---|---|---|---|---|
| D000 | SCREENED | Load MMBench2 `model.py` by exact path and reject digest drift | File contents and class definitions remain untouched | Its executable uses flat imports and is not packaged as a dependency | Source-loader test passes; a digest failure means source drift, not a model failure |
| D001 | SCREENED | Reduce image resolution and model width/depth to a named `D4LiteConfig` | Same tokenizer, block-causal layers, shortcut tokens, and flow head | Upstream training scale requires hardware far beyond 6 GB | Four-arm CUDA smoke, synthetic overfit, and real T-BASE preflight pass; remaining failure may still be under-capacity |
| D002 | SCREENED | Replace the continuous 16-D action MLP with a 17-way discrete embedding | Exactly one action token occupies the same token-layout slot | Crafter exposes 17 categorical actions | Led-to/indexing tests pass; after 5k T-BASE updates, paired action shuffling raises uniform-dev flow loss by 21.6% |
| D003 | SCREENED | Add a continuation MTP head beside the upstream reward MTP head | Head reads the same post-transition agent tokens | Crafter planning must distinguish terminal from continuing futures | Alignment and gradient tests pass, but efficacy fails: the 5k baseline predicts P(continue)=.998 on terminal-aligned generated states |
| D004 | SCREENED | Replace only `TimeSelfAttention` modules in dynamics with official Mamba-2 modules | Spatial attention, MLP, residual ordering, token layout, tokenizer, and heads stay unchanged | Tests the original long-context temporal-compute hypothesis | Source/mechanical checks pass; matched real-Crafter M1 passes all frozen feasibility gates and reproduces bit-exactly under the pinned official deterministic path; there is no short-context speed win |
| D005 | SCREENED | Branch a clean Mamba prefix state across shortcut denoising candidates and commit only the generated latent | Each denoising candidate sees one identical clean prefix | Mamba state is mutable whereas attention KV context is functionally shared | Cached/uncached equivalence and candidate-isolation tests pass; first suspect for any future rollout-only Mamba failure |
| D006 | SCREENED | Add a CDP-shaped cosine predictor from clean causal agent state plus next action to a stop-gradient future latent; update the encoder slowly and retain a frozen-decoder reconstruction anchor | Upstream flow loss remains present, receives detached tokenizer latents, and stays independently reportable | Tests predictive representation learning without letting future-target gradients or a moving decoder obscure attribution | Target-detach, encoder-gradient, decoder-freeze, optimizer-group, and CUDA tests pass; scientific efficacy remains untested |
| D007 | SCREENED | Reuse episode-bounded replay and canonical Crafter wrapping from `m3_hjwm_compact` through adapters | Observation/action/reward/continuation timing is unchanged | These utilities already passed indexing and reproducibility regressions | Adapter/source/replay hashes pass; real-data training and two bit-identical executed trajectories pass |
| D008 | SCREENED | Use categorical random shooting before implementing categorical CEM or PMPO | Upstream imagined reward scoring and first-action receding horizon remain | Produces executed behavior early with fewer new optimizer assumptions | Three-seed executed preflight and exact substantive repeat pass; it remains an evaluation instrument, not a learned Dreamer policy |
| D009 | SCREENED | Add strict atomic checkpoints around the composed model, full config, source digests, optimizer, and every RNG | Model state still loads with `strict=True`; no source checkpoint is rewritten | The upstream executable checkpoint format cannot reconstruct local adapters and prior work exposed partial RNG resume | Strict roundtrip, digest/config/source rejection, Torch/NumPy resume, tokenizer roundtrip, and failed-save preservation tests pass |
| D010 | SCREENED | Port only the tokenizer reconstruction, shortcut-flow, reward-alignment, and rollout equations needed around the unchanged upstream model classes | Operators, led-to convention, RMS normalization, K-grid, and Euler update follow the pinned scripts | MMBench2 is an executable-style project whose flat trainers cannot be safely imported as a library | Line-level provenance is in docstrings; synthetic overfit, paired action shuffle, cached rollout, and real preflight are the first parity checks |

`D010` is an administrative ledger correction made after the first preflight:
the port boundary was already disclosed in `README.md`, `SOURCE_MANIFEST.md`,
and source docstrings before execution, but it did not have its own numbered
row. No code or result was changed when the row was added.

## Transition convention

All local adapters use:

```text
(observation_t, action_t) -> (observation_{t+1}, reward_{t+1}, continue_{t+1})
```

Within the block-causal sequence, the action token stored at state position
`t+1` is `action_t`, the action that led to that state. Position zero receives
the dedicated start action. Reward and continuation head zero predict the
transition outcome that led to their current state position.

## Forbidden silent changes

- No Mamba fallback to GRU or attention.
- No Mamba inside the tokenizer during the initial factorial.
- No removal of reconstruction in the initial CDP arm.
- No reward-bin, event-sampling, calibration, or planner-budget tuning that
  differs between factorial arms.
- No loading Transformer dynamics weights into the Mamba arm with
  `strict=False`.
- No claim of a Dreamer-4 PMPO agent until policy/value imagination is
  implemented and tested separately.

## Current first-suspect order

1. Continuation class imbalance and calibration: implementation alignment
   passes, but the baseline learns the dominant "continue" label.
2. Reward magnitude and generated-state deployment: event ranking improves,
   while event magnitude remains strongly underestimated.
3. Baseline scale and training budget: the 5k screen is before the upstream
   10k shortcut-bootstrap start and far below upstream training scale.
4. The categorical action adapter: it is measurably used, but correct-action
   benefit is small and heterogeneous.
5. CDP gradient routing and representation usefulness. It is mechanically
   screened but disabled in every real-Crafter result recorded so far.
6. Mamba cache/recurrent semantics only if future long-context or planner
   behavior diverges. Matched M1 training passes; Mamba is not an explanation
   for the failures shared with T-BASE.
