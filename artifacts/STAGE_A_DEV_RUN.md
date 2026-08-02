# Stage-A DEV run, 2026-08-02

**This is not a Stage-A result.** It is a DEV smoke on `dd60710`/`d49693e` to see
whether the pipeline learns anything and whether the failure modes the register
worries about show up. The preregistered FINAL protocol (S52) was not run: no
BC-prior control, 16 seeds instead of a sealed set, an 800-step cap instead of
Craftax's native 10000, and one training seed per arm. Nothing here may be
reported, and the FINAL seeds remain untouched.

## Setup

One tokenizer for all four arms (S20), 3000 steps. Per arm: 5000 dynamics, 2500
agent, 800 actor. Corpus: 96 archive episodes plus the full support corpus --
344 train / 44 dev episodes, 260,081 transitions, 274 terminals, 76 BC-eligible.
Phase 1A restored from checkpoint on relaunch and reproduced the cache digest
`a62705fcfbace70c` exactly.

## Results

| arm | score | 95% CI | vs random | lower bound > 0 | separation | contraction | outside unit |
|---|---|---|---|---|---|---|---|
| flow-attention | 1.44 | (0.98, 1.64) | +0.27 | no | 0.0063 | 1.445 | 0.506 |
| flow-mamba | 1.79 | (1.05, 2.08) | +0.62 | no | 0.0288 | 1.043 | 0.218 |
| direct-attention | 2.19 | (1.47, 2.54) | +1.02 | yes | 0.1717 | 0.952 | 0.000 |
| direct-mamba | 3.18 | (2.25, 3.53) | +2.01 | yes | 0.2866 | 0.950 | 0.000 |

Random control: score 1.17, 2.06 achievements.

| arm | achievements | length | reward MAE | 1-step latent error | dynamics state | fwd/bwd | FLOPs/step |
|---|---|---|---|---|---|---|---|
| flow-attention | 2.12 | 101 | 0.084 | 0.797 | 1,179,648 | 14.3/s | 2.18e9 |
| flow-mamba | 2.69 | 144 | 0.080 | 0.265 | 860,160 | 13.3/s | 2.15e9 |
| direct-attention | 3.69 | 158 | 0.045 | 0.041 | 1,179,648 | 8.1/s | 4.40e8 |
| direct-mamba | 4.00 | 136 | 0.044 | 0.040 | 860,160 | 7.4/s | 4.34e8 |

## What can and cannot be concluded

The ordering is monotone in both substitutions and they compose, but the
comparison against random is *not* the S52 criterion, which also requires beating
the arm's own frozen BC prior. Two arms clearing a random baseline after 800
actor steps is weak evidence of anything.

**The continuation separation column is not trustworthy.** Every arm's figure
rests on **3 terminal targets** -- the DEV diagnostic drew 24 batches, of which 3
carried a support row, each contributing one lead-0 terminal. The diagnostic
reports `terminal_targets` precisely so this cannot be read as a measurement. It
needs far more DEV batches, or aggregation across leads, before it says anything.

**Direct's low one-step error is the thing to be suspicious of, not pleased
about.** 0.04 against flow's 0.27-0.80, with contraction 0.950, is exactly the
signature S35 predicts for a conditional-mean collapse: under squared loss the
collapsed solution is the one that minimises that number. The
nearest-mode/mode-mean split in `multistep_error` exists to adjudicate this and
was not supplied with successor samples here.

`outside_unit` behaves as declared: Direct's readout is tanh-bounded (S2) and
reports 0.000; flow's is unbounded and half of flow-attention's predictions leave
the unit range.

Cost matches S19's prediction: flow spends 2.2e9 FLOPs per imagined step against
direct's 4.4e8, a 5.0x ratio, and Mamba's dynamics state is 860,160 elements
against attention's saturated 1,179,648.

## Incident

The first launch died on `flow-mamba` with a CUDA OOM inside Mamba's Triton
autotuner, which benchmarks several kernel configs and needed headroom the
resident tokenizer was holding. The tokenizer is now moved to CPU during the
per-arm phases and back for evaluation; peak went from ~5000 MiB to ~1450 MiB.
`flow-attention` was already complete and was skipped on relaunch.
