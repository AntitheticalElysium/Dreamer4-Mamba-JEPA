# D4-lite Stage M1: matched backend outcome

Date: 2026-07-20
Protocol:
`reviews/2026-07-20-d4-stage-m1-matched-backend-protocol.md`
Verdict: **pass as a Mamba feasibility result; not a model-quality pass**

## Plain-language result

The Mamba replacement works.

Under the same tokenizer, non-temporal initialization, 20,000 replay windows,
training-noise stream, optimizer, loss, and fixed evaluation rows, both the
source-pinned Transformer and official Mamba-2 world models trained for 5,000
real Crafter updates. Both learned action-dependent latent dynamics. Mamba
passed every preregistered relative-safety gate and the full pair reproduced
exactly after deterministic execution was enabled.

This does **not** mean the current world model is ready for policy learning.
Both backends still predict almost-certain continuation at true terminals, and
K=4 generated reward event ranking is near chance. It means Mamba is not the
reason for those shared failures and is viable for the next isolated
`BASE`-versus-`CDP` experiment.

## Pairing and source integrity

- Tokenizer SHA-256:
  `91a210dc8c76fa29793599ced04190438d776a0c1a757b674691272eeb58b22c`.
- Training replay SHA-256:
  `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad`.
- Dev replay SHA-256:
  `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad`.
- Frozen replay schedule SHA-256:
  `70b5c323e66c4faccdad94f80c45be6b991bec69f10773fe55fcb0baabd1dc3f`.
- Shared non-temporal initialization: 193 tensors and 881,555 values,
  bit-identical, digest
  `dc7d4b4b02092ae9d99639aef5667ff8b5e9ea8b6596b9406410a734b610e911`.
- Post-training Torch CPU/CUDA RNG digest, both arms:
  `f697f38aeba6bffe3aa2fc7cd8954de0311aeabe0fdcf1cc5eef746c7090d342`.
- Core implementation SHA-256:
  `6241c000fca459825b44bc6482c60f50294dd0635870e0a8cda56ec20c2aec62`.
- Final runner SHA-256:
  `c20b571c6f540f35a9416fc41b600c8243b5c19a84398997a45ba332aed7cade`.
- Pinned and installed official Mamba determinism helper SHA-256:
  `cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d`.

Both checkpoints strictly reconstructed their full config and state at update
5,000 under the pinned source and core hashes.

## Frozen gate

Uniform fixed-dev results:

| Gate quantity | T-BASE | M-BASE | Mamba requirement | Result |
|---|---:|---:|---:|---|
| Flow MSE | .005661 | .005946 | M/T <= 1.25 | 1.050x, pass |
| Action-shuffled / true flow loss | 1.200 | 1.124 | M >= 1.05 | pass |
| Wrong-minus-correct latent MSE | +.000645 | +.000384 | M > 0 | pass |
| Generated reward event AUROC | .5389 | .5132 | M-T >= -.05 | -.0258, pass |
| Zero-target absolute generated reward | .002651 | .004959 | M/T <= 2 | 1.870x, pass |

All five checks pass together. There were no source, schedule,
initialization, RNG, finite-gradient, checkpoint, or evaluation-integrity
failures.

## Evidence that Mamba actually learned

This is more than a finite forward pass:

- uniform latent flow MSE moved from `.30738` before training to `.00595`;
- action shuffling moved from exactly no effect (`1.000x`) to `1.124x` the
  correct-action loss;
- wrong actions became worse than correct actions on average by `.000384`;
- real-encoded-state reward event AUROC moved from `.5940` to `.7882`;
- on reward-event-aligned generated rows, reward Pearson correlation is
  `.4794`.

Separately, a controlled moving-square Mamba overfit run reduced full
reconstruction MSE `.26857 -> .01995`, flow loss `.10483 -> .01071`, reward
loss `1.49606 -> .34379`, and continuation loss `.51858 -> .00387`.

The cautious interpretation is that the official Mamba path, action adapter,
heads, gradients, and real-data trainer are operational. It does not establish
generalization from one training seed.

## Reproducibility audit

The first ordinary-CUDA run and its immediate repeat passed the same gate, but
Mamba's final tensors differed by `1.80e-4` relative L2 despite exact
initialization, replay schedule, and RNG states. Transformer was exact. A
focused two-repeat audit localized this to nondeterministic CUDA arithmetic in
the official Mamba path: enabling PyTorch and official Mamba deterministic
execution made all 30,273 audited values bit-identical.

The protocol records this transparently as a post-outcome reproducibility
amendment. Two new full pairs were then run in fresh processes. The committed
verifier reports:

- scientific reports: exactly equal;
- decoded T-BASE checkpoint payloads: exactly equal;
- decoded M-BASE checkpoint payloads: exactly equal;
- T-BASE world-state digest, both:
  `6e1093ce50f75e1e01edcaab166e9c3674f0273c485f4498725257dff1b22e61`;
- M-BASE world-state digest, both:
  `bda70f4b3fce51ce3caffc4df7064cb8187891d4ddb1779625c192d19d962b89`.

Raw `.pt` file hashes differ because `torch.save`'s zip container is not
byte-stable even when every decoded field and tensor is equal. They are still
recorded:

| Artifact | First SHA-256 | Repeat SHA-256 |
|---|---|---|
| Report | `55b5ca428e4897eb392ac60685be613e1c560ac7769345361cb2d36a5875bf37` | `30e499bdb9eec701204532a61ff4937bc780a7ceee36e12c8cfd26d5f31eaf09` |
| T-BASE checkpoint | `7fd87a6fe5c3b29f8a1b90edf152b1a68545b0fb7afd8833598eb3cd098bdfc1` | `3f6bd2c3cb116baff8f0ba2b6bcaba9a9a77f92e6b0fad32d91a43d1edd52d42` |
| M-BASE checkpoint | `8bdfa83fc1675aa0ad91e1aa20b9c2660b2194a4e834644cebb38189340fb83d` | `20e9768136f7635880a9943b966689a29596d1b1227a56471b00840bdcc5544d` |

The report JSON hashes also differ because they intentionally retain wall
time, throughput, peak VRAM, output paths, and raw checkpoint hashes. The
verifier excludes only those operational fields.

Reproduction command:

```bash
.venv/bin/python -m d4_mamba_jepa.verify_reproduction \
  outputs/d4_mamba_jepa/stage_m1_deterministic \
  outputs/d4_mamba_jepa/stage_m1_deterministic_repeat
```

## Important failures and boundaries

1. **Continuation is still broken.** On all 14 terminal-aligned generated
   rows, mean P(continue) is `.99871` for Transformer and `.99872` for Mamba.
   This is a shared objective/data-imbalance failure, not evidence against
   Mamba.
2. **Generated reward is not yet planner-grade.** Uniform K=4 event AUROC is
   `.539` and `.513`; both are close to chance. Mamba has strong real-state
   reward ranking but loses much of it through imagination.
3. **No short-context compute win exists.** In the first deterministic run,
   Transformer trained at 20.99 updates/s and Mamba at 15.93 updates/s. Mamba
   also used more peak allocated VRAM. Kernel compilation and the short
   sequence make this a feasibility measurement, not the proposed
   long-context benchmark.
4. **This is one training seed on spent diagnostic data.** Passing says the
   backend is viable; it does not establish superiority, population
   robustness, Crafter performance, or long-context scaling.
5. **JEPA/CDP remains scientifically untested on Crafter.** Its code has
   mechanical and controlled-gradient evidence only.

## Decision

Stage M1 passes. Retain both temporal backends.

The next isolated stage is a matched `BASE`-versus-`CDP` factorial. It must
keep the same tokenizer, backend-specific initialization, replay schedule,
task heads, flow objective, evaluation rows, and deterministic execution.
CDP must improve predictive representation or generated deployment without
being credited for unrelated continuation/reward changes. Long-context timing,
planner execution, and actor/value learning remain later, separately named
questions.

## Validation

- New-track suite: 40 passed in 21.34 seconds.
- Repository-wide suite: 229 passed in 77.92 seconds.
- The only warning is the pre-existing compact BF16 tensor-to-scalar warning.
- `compileall`: pass.
- `git diff --check`: pass.
- No generated replay, checkpoint, report, cache, or output directory is
  included in the source commit.
