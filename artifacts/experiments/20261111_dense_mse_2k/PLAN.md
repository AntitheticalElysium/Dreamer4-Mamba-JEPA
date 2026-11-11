# Matched Continuation Extended to 2,000 Updates Total

## Scope & Objective
Extending both the endpoint and dense multi-step arms from their step 384 checkpoints up to 2,000 updates total (2,048,000 frames each).

- **Resumes strictly from step 384:** Loads `latest.pt` containing exact model, optimizer, stream, carry, cache, and generator states.
- **Identical Training Pipeline:**
  - `LAMBDA_MULTI = 0.1`
  - Rebased CUDA segment IDs: `seq[:, split_idx:] - seq[:, split_idx-1:split_idx]` (carry continuity verified at RMS $\sim 0.0039$).
  - Exact replay of saved HUD fixations from `gaze.bin`.
  - Constant learning rate `2e-5 * g["lr_scale"]`.
- **Primary Readout:**
  - 24-pair native hunger test ($K=32$ futures, 100% held-out reader accuracy).
  - Paired contrast difference: dense vs endpoint, dense vs parent, endpoint vs parent.
  - Factual consequence guards on 120 DEV episodes (risky, adjacent, coal, iron).
- **Immutable Assets:** Parent checkpoint and all canonical code remain 100% read-only.
