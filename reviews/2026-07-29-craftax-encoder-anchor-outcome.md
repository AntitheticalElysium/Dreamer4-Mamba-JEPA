# Craftax encoder-anchor outcome

Date: 2026-07-29
Branch: `craftax-clean-baseline`
Launch commit: `06e1ba5`
Checkpoint implementation digest: `faae55a5267f16fb025b678d2084b547c457721590b340c1596e0889e7eb3180`

## Question and design

Does the Craftax JEPA objective erase task state because the randomly
initialized world encoder is optimized at the same `1e-4` learning rate as the
dynamics and predictor?

The completed grid compares three encoder conditions across seeds `20260727`,
`20260728`, and `20260729`:

- full: encoder and the rest of the world at `1e-4`;
- slow: encoder at `6e-6`, the rest at `1e-4`;
- frozen: encoder at `0`, the rest at `1e-4`.

This is a clean mechanism diagnostic: Transformer, pooled predictor, EMA
anti-collapse, JEPA-only loss, `terminal_fraction=0`, 2,500 updates, and the EMA
tau ramp pinned to the real 20,000-update denominator. The slow arm is a 16.7x
local separation relative to this repository's `1e-4` world LR. Dreamer-CDP's
own `6e-6` versus `4e-4` recipe is a different 66.7x ratio.

The representation oracle ran at updates 0, 1,000, and 2,500. Every self-audit
passed. The replay and probe digests were identical in all nine cells:

- replay: `7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b`;
- probe: `bb5c7c703c0125131dcdb56cb24660ad22febf18c236cd6cf5336b8f748d1fdb`.

## Result

Mean update-0 to update-2,500 change across three seeds (sample standard
deviation in parentheses):

| Encoder | Inventory linear R2 | Inventory nonlinear R2 | Vitals linear R2 | Vitals nonlinear R2 | Dev cosine change |
|---|---:|---:|---:|---:|---:|
| full `1e-4` | -0.161 (0.032) | -0.203 (0.028) | -0.194 (0.079) | -0.248 (0.025) | +0.664 (0.058) |
| slow `6e-6` | +0.003 (0.012) | -0.001 (0.007) | -0.023 (0.025) | -0.019 (0.014) | +0.593 (0.068) |
| frozen | 0 by construction | 0 by construction | 0 by construction | 0 by construction | +0.540 (0.071) |

On the 16 targets times three seeds:

- slow beat full on 48/48 nonlinear target-seed changes;
- slow beat full on 45/48 linear target-seed changes;
- full worsened 48/48 nonlinear target-seed values;
- full worsened 44/48 linear target-seed values.

The slow encoder retained about 89% of the full-LR dev-cosine improvement while
removing almost all semantic erosion. The frozen encoder still allowed a large
cosine improvement, establishing that the dynamics and predictor can learn
substantially without moving the encoder.

## Interpretation and limits

This is strong causal evidence for an encoder-timescale problem, not evidence
that the architecture now passes. All 16 continuous oracle targets were still
classified `degraded` at update 2,500 in every slow seed. That is expected from
the degraded random-encoder starting point: slowing optimization preserves the
available random features but cannot create an absolutely sufficient
representation by itself.

The diagnostic also does not yet establish transfer to the failed production
recipe. It removes terminal oversampling and task-head losses and stops at
2,500 updates. No BC, imagination, or executed Craftax score was measured.
Therefore this result must not be recorded as a baseline improvement or oracle
pass.

The nonlinear pixel ceiling has small GPU nondeterminism, so the comparative
claim is based on the paired latent R2 changes, not tiny differences in
ceiling-derived verdict thresholds.

## Decision

Proceed with:

1. a paired 2,500-update full-versus-slow prefix under the actual
   `terminal_fraction=0.5` and all-heads recipe;
2. the complete slow-encoder T-JEPA and M-JEPA pipelines, holding BC and
   imagination fixed because both already freeze the world;
3. representation-oracle and fresh-seed executed-control evaluation against
   the existing full-LR baseline;
4. SIGReg full-versus-slow as a higher-dimensional mechanism ablation.

Do not rerun the complete spatial grid. Spatial context did not address
encoder erosion at full LR. At most, run the narrow slow-LR pooled-versus-spatial
interaction after the primary transfer and full-pipeline jobs.

Artifacts:

- `reviews/artifacts/encoder_anchor/anchor_{full,slow,frozen}_s20260727.json`
- `reviews/artifacts/encoder_anchor/anchor_{full,slow,frozen}_s20260728.json`
- `reviews/artifacts/encoder_anchor/anchor_{full,slow,frozen}_s20260729.json`
