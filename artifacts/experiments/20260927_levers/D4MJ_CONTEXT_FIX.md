# Proposed d4mj fix: bounded world context (for review, not implemented)

## The bug
The canonical Mamba world is trained on 4-frame windows from a zero state (`joint.frames = 4`, le-wm's
`history_size 3 + num_preds 1`). It is deployed as an unbounded recurrence. Measured
(TRANSITION.md part 1d/4, levers E1):
- one-step error vs copying goes from 0.80 at positions 1-3 to 2.25 at 64-95;
- the SSM state grows 3.2-4.4x past the training length;
- under le-wm's own 3-frame truncation the same weights reach a 16-step imagined error of 0.53 against 2.57
  unbounded (copy 0.93).

Every precedent keeps deployed positions inside the trained context:
- le-wm: `emb[:, -HS:]`;
- DRAMA: trains on 128-frame sequences and deploys a 16-frame deque;
- Dreamer 4: trains 32/128 frames and deploys a 96-192 frame context.

## The invariant
**C = the trained context: the most completed pairs any training phase ever scans before a prediction.** No
rollout may condition on more than C completed pairs. With fewer, the world runs as a plain recurrence; with
more, every step rescans the last C from a zero state.

| setting | C | what it is |
|---|---|---|
| today (joint only, 4-frame windows) | 3 | le-wm's rule (option a). Measured best one-step accuracy at matched compute; imagination 0.406 |
| after a long-context phase (E1f recipe: continue at 64 frames, 24 windows per update) | 63 | option (b). Imagination 0.400; memory +18-20% over its own window |

## Code (minimum; no new files or functions; signatures unchanged)

1. `lewm_config.py`
   - `DynamicsSettings.context: int = 3`, mirroring the existing `TransformerDynamicsSettings.context = 3`
     (le-wm's value) that the transformer backend already bounds its window with.
   - It is part of the recipe digest, so a recipe states its C.
2. `lewm.py`
   - `LeWMWorld.advance(state, action)` and `LeWMWorld.observe_latent(state, action, z_next)`: when
     `state.step >= context`, rebuild the carry by scanning the last `context` pairs from `start(...)`, then
     advance once.
   - This needs the last `context` latents and actions in the state. `PredictiveState` gains two fields,
     `past_latents [B,k,D]` and `past_actions [B,k]` (k <= context), the same buffers `WindowPredictiveState`
     keeps for the transformer backend.
   - `teacher(z, actions)` refuses `z.shape[1] - 1 > context` unless `state` is None and the call is a
     training scan on a window of length <= context + 1.
3. `world_api.py`
   - `LeWMWorldAdapter.fork / detach_state / repeat_state / state_tensors` carry the two new fields.
   - `prefill` keeps only the last `context` pairs, as `LeWMTransformerWorldAdapter.prefill` already does.
4. `data.py` / `lewm_config.py`, bridge recipe (Phase 2 must train what it deploys):
   - `burn_in = 0`;
   - `sequence` and `sequence_long` <= context + 1 + recursive depth;
   - the `sequence_long = 4 x sequence` and `burn_in = 3 x sequence` validators relaxed accordingly.
   - For option (b) a long-context world phase precedes the heads. It is dynamics-only, on the cached latents
     (E1f: L = 64, 24 windows per update, 5,000 updates, joint optimizer settings).
5. `train.py` `bridge_rollout`: unchanged in signature; it inherits the bounded `advance`.
6. `checkpoint.py`: capabilities record `trained_context`; loading refuses a checkpoint whose recipe context
   differs from its recorded training.
7. Tests (`d4mj/tests/test_m4_core.py`):
   - (i) at step >= context, `advance` equals a fresh scan of the last `context` pairs, bit for bit (fp32);
   - (ii) at step < context it equals the plain recurrence;
   - (iii) fork / repeat / detach preserve the buffers;
   - (iv) mutation: removing the rescan fails (i).

## Consequences
- `train.py`, `lewm.py`, `world_api.py`, `data.py`, `lewm_config.py` and `checkpoint.py` are all in the hashed
  sources closure. Every existing canonical checkpoint stops loading; the lineage restarts from joint training.
  Bundle it with the H2 depth-alias fix (branch `h2-terminal-depth`, 9e42b7a2).
- The frozen-eval proof for m03 needs re-measuring (2 tests fail closed until then).
- Recurrence semantics change for every consumer: imagination, execution, gates, diagnostics. Diagnostics that
  deliberately probe long memory (`lewm_diagnostics.py` lines 160-236, the Mamba-state supplement) must state
  their C.

## Decision needed
1. Adopt the invariant as the fix? (Recommended: it is the measured cause, and it covers both options.)
2. C = 3 now, with (b) as a follow-up recipe? Or build the long-context world phase (C = 63) straight away?
