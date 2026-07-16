# Step 4b protocol: backend x scale at fixed long context

Status: **pre-registered before collector execution or outcome-bearing
training**. Feasibility-only random-tensor measurements may precede the screen;
they may change batch size solely if the declared 6 GB safety limit is exceeded.

## Question and scope

Step 4 established predictive parity between one tiny pooled GRU and one tiny
pooled Mamba-2 at training length `T=16`. It did not test the regime supporting
the Mamba literature: longer parallel sequences and deeper/wider state-space
stacks. Step 4b asks:

> At `T=128`, does increasing the temporal core to a DRAMA-scale width/depth
> improve Mamba relative to a parameter-matched GRU more than it improves the
> small cores?

This is a **backend x scale test at fixed long context in the existing pooled
topology**.
It is not a DRAMA reproduction: dense JEPA tokens are still mean-pooled, the
latent is continuous rather than categorical, and the objectives remain the
validated compact objectives. A flattened-latent/action topology is a separate
conditional follow-up, not part of this test.

Context is fixed at 128 in every Step-4b arm. The two small arms determine
whether long-context training alone is enough for the small Mamba to separate
from the small GRU. Comparison with Step 4's `T=16` endpoint is descriptive
only because batch geometry, sampled windows, update count, and monitor seeds
differ. This screen cannot claim a backend x context-length interaction; that
would require matched `T=16` and `T=128` cells in one experiment.

## Source-grounded regime

- Pinned official Mamba: `state-spaces/mamba` commit
  `f577286d052741c35d39cd43bdc3fad27120f22c`.
- Pinned DRAMA: `realwenlongwang/Drama` commit
  `a50bd54c34e77d1d13e988a031733a47817098e2`.
- DRAMA uses training length 128, hidden width 512, and two Mamba layers.
- Official Mamba-2 documents state size 64/128 as typical. This screen uses
  `d_state=64`, `headdim=64`.
- DRAMA itself configures `d_state=16` (and uses its complete wrapper, RMSNorm,
  dropout, flattened categorical latent, and a different optimizer). Thus
  “DRAMA-scale” here means only `T=128`, width 512, and depth 2; the core is a
  deliberately stronger official-Mamba-2 state configuration, not a faithful
  DRAMA hyperparameter reproduction.
- Mamba-3 remains excluded: its official recurrent path is H100-only and is
  non-finite on the RTX 3060 under the required step test.

## Arms

All arms use the same frozen validated encoder, deterministic predictor,
unmasked representation, `frozen_dynamics_recipe()`, replay file, sampled
windows, optimizer (`AdamW`, lr `1e-4`), and shared non-temporal initialization.

| ID | temporal adapter | training context |
|---|---|---:|
| LS-G64 | existing `GlobalGRUTemporal`, hidden 64, depth 1 | 128 |
| LS-M64 | existing `GlobalMambaTemporal`, width 64, depth 1, state 32 | 128 |
| LL-G | projected global GRU, hidden chosen for <=0.5% temporal-parameter match, depth 2 | 128 |
| LL-M | projected global Mamba-2, width 512, depth 2, state 64, head 64 | 128 |

The two large arms share the same external contract:

1. mean-pool `[B,T,S,64] -> [B,T,64]`;
2. learned `64 -> hidden` input projection;
3. two recurrent blocks;
4. final LayerNorm;
5. learned `hidden -> 64` output projection;
6. broadcast and add to the unchanged dense token residual.

The GRU hidden width is selected mechanically before training to match LL-M's
temporal parameter count; it is not tuned on outcomes.

## Data and pairing

- Training seed: 404 for the one-seed screen.
- Replay: pinned `data/replay_40k_v1.pt`; only episodes supporting 128
  observations are eligible. This leaves 240/264 episodes and 40,821/42,979
  transitions.
- Batch: `B=1,T=128` for every arm. This gives every arm the same observations
  per update and makes the recurrent-context difference real rather than
  gradient accumulation over reset 16-step chunks.
- Sampling uses the same explicit NumPy RNG state for every arm; a digest over
  every sampled tensor must match.
- All non-temporal state entries are copied from one reference initialization
  and their digest must match.
- Training rungs: 500, 1,000, 2,000, and conditionally 4,000 updates.
- Continue 2k -> 4k only when all arms are finite and at least one Mamba arm's
  monitor continuous separation improves from 1k to 2k. This is a resource
  gate, not an arm-selection rule.

This 4k screen exposes each arm to 512k observation instances, half the Step-4
per-arm count. It licenses screening only.

## Fresh long-prefix monitor set

- Environment seeds 111-114, which occur in neither training replay nor prior
  final bundles.
- Prefix length 128, suffix length 8.
- Per seed: 4 day and 2 night anchors (24 total).
- Four action suffixes and three equal common-RNG branches per suffix.
- Live environment and forks canonicalized; repeat verification must be
  bit-exact; bundle and manifest hashed before training.
- These are monitor/selection seeds and are spent after this screen.

A confirmatory run, if licensed, must reserve a separate 16-seed final bundle
(provisionally 115-130) before any three-seed training starts.

## Metrics

At each rung evaluate the eight-step open-loop predictions after observing the
full 128-frame prefix.

Primary screen metric:

- symmetric continuous all-token separation at `k=8`:
  `mean(off-diagonal suffix distances) - mean(diagonal distances)`.

Secondary metrics:

- patch-only continuous separation;
- tie-aware four-way retrieval (fractional credit among exact minimum ties);
- legacy deterministic-argmin retrieval for continuity with Step 4;
- per-horizon `k=1..8` curves;
- day/night and pixel/task-effective strata (descriptive at n=24);
- JEPA, rollout, reward, and continuation losses;
- wall time, peak allocated/reserved VRAM, warm sequence and recurrent-step
  latency, cache size, and full-world H=8 imagination latency.

The monitor set is too small for a superiority claim. CIs and per-environment
signs are diagnostic only.

## Registered interaction readout

At the final executed rung define:

```
delta_small = separation(LS-M64) - separation(LS-G64)
delta_large = separation(LL-M)   - separation(LL-G)
interaction = delta_large - delta_small
```

The screen licenses confirmatory replication only if all conditions hold:

1. LL-M is finite and fits below 5,000 MiB peak reserved VRAM;
2. LL-M primary separation is positive;
3. `delta_large > 0` in at least 3 of 4 environment-seed means;
4. `interaction > 0` overall;
5. LL-M exceeds LL-G by at least 10% of `abs(separation(LL-G))` or by
   `0.0005`, whichever is smaller;
6. the direction is not contradicted by both tie-aware retrieval and
   patch-only separation.

If these pass, train same-seed shuffled-action LL-G/LL-M controls before any
three-seed replication. Each large real arm must exceed its backend-matched
control by 1.5 retrieval points or have a positive paired continuous-separation
margin. Failure stops the scale claim.

If the screen fails, the conclusion is not “Mamba is generally bad”; it is:
“Increasing Mamba to width 512/depth 2 at T=128 did not produce a useful
backend-by-scale interaction in the compact pooled Crafter topology.” No larger
or source-shaped run is licensed from this experiment.

## Correctness and resource gates before training

- exact large-arm temporal parameter match within 0.5%;
- sequence/repeated-step equivalence for both large arms in FP32 and BF16;
- reset isolation and cache cloning tests;
- finite forward/backward with `B=1,T=128` under BF16 autocast;
- peak reserved VRAM below 5,000 MiB;
- no source/checkpoint mismatch and no silent backend fallback;
- full compact test suite passing;
- a clean committed source tree for the outcome-bearing run.

No online policy training, reliability weighting, or architecture replacement
is authorized by this screen.
