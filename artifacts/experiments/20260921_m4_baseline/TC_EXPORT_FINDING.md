# TC cannot produce a TC-14 compliant latent cache

**Measured 2026-09-21, both arms, same frozen encoders, same execution settings.**

## What happened

`export` writes the frozen post-joint latent cache the bridge trains on. It verifies
**batch invariance**: a frame encoded alone must match that frame encoded inside a chunk, because
the deployed actor encodes one frame at a time while the cache is built in chunks of 128.

```python
z = encoder(frames)                       # chunk
single = encoder(frames[:, t:t+1])        # alone
torch.testing.assert_close(single, z[:, t:t+1], atol=1e-5, rtol=1e-4)
```

Raw passed and wrote a 2.9 GB cache in 42 minutes. **TC failed in 2 minutes.**

## The measurement

Six DEV episodes, 60 checks per arm, 11,520 coordinates:

| arm | worst absolute (tol 1e-5) | worst relative (tol 1e-4) | coordinates failing `\|Δ\| ≤ atol + rtol·\|z\|` |
|---|---:|---:|---:|
| raw | 2.50e-06 | 4.15e-03 | **0 / 11,520** |
| tc | 3.32e-05 | 2.26e-03 | **14 / 11,520** |

At 0.12% of coordinates failing per check and roughly 60,000 checks in a full export, TC fails
essentially every chunk. This is systematic, not an unlucky frame.

## Why TC and not raw

The reported failure landed on a coordinate with `|z| ≈ 1.7e-3` — near zero. `assert_close` allows
`atol + rtol·|z|`, so on a near-zero coordinate the allowance collapses to `atol` alone and a
~3e-5 absolute perturbation fails.

**TC centers its latents temporally.** Its residual therefore carries near-zero coordinates that
raw's representation does not, and those coordinates are where a fixed absolute tolerance bites.
The non-determinism itself is shared: `cuda_matmul_tf32` is false but **`cudnn_tf32` is true**, so
the ViT's patch-embedding convolution selects different cuDNN algorithms for different batch
shapes. Raw is exposed to the same effect (its worst *relative* error is in fact larger, 4.2e-03)
but has no near-zero coordinates for it to land on.

## Why the tolerance was not loosened

The perturbation is physically negligible: 3e-5 absolute contributes ~9e-10 to a squared error,
against TC's own prediction MSE of 0.327. It would have been easy to widen `atol` and proceed.

That would have been manufacturing a pass. The guard encodes a real requirement — the cache the
bridge trains on must equal what the deployed actor produces — and TC does not satisfy it at the
declared tolerance. Every knob that would legitimately fix it is sealed:

- `cudnn_tf32`, `cudnn_deterministic` are in the checkpoint's `sources.execution` manifest;
  changing them invalidates the joint checkpoints that export must load.
- `runtime.cache_chunk = 1` would make the cache batch-invariant by construction, but
  `cache_chunk` is inside `recipe_digest`, and export refuses a checkpoint whose `recipe_id`
  disagrees.

Both are closure changes belonging to a new sealed recipe, not to a live run.

## Consequence

**Raw proceeds** through bridge, gates, actor and real evaluation. **TC stops here**, and this is
recorded as a result about the TC arm rather than a reason to abandon the run.

The honest scope of the campaign is now narrower than planned: a **one-arm** M4 baseline. The
raw-versus-TC comparison this run was built to make cannot be completed from these checkpoints,
and the joint-phase numbers (raw 0.019 vs TC 0.327 prediction MSE) are what the comparison
delivers instead.

## What a follow-up would need

A new sealed recipe declaring `cache_chunk = 1`, or `cudnn_tf32 = false` in the execution block,
retrained from scratch — because both change the manifest the existing checkpoints are bound to.
Whether TC's near-zero coordinates are also a problem for the bridge's own objective is a separate
question this finding does not answer.
