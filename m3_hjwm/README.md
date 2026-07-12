# M3-HJWM reference implementation

This repository is a clean architectural scaffold for a reconstruction-free,
JEPA-native world model with a Mamba-3 temporal target, efficient multimodal
future prediction, and Dreamer-style imagination training.

It is **not** a claim that the architecture is already benchmark-optimal. It is
the tensor-level implementation contract from which experiments can be run
without redefining state semantics.

## Design decisions

- Dense 8×8 visual tokens, not one globally pooled embedding.
- EMA target encoder and stopped-gradient target features.
- Explicit spatial mixer; Mamba is temporal, not a substitute for 2D geometry.
- Mamba-3 target backend with Mamba-2/GRU fallback behind one interface.
- Default hard-mode mixture future predictor for single-pass multimodality.
- Deterministic predictor retained as a control.
- Rewards and continuation are predicted from the **post-transition** state.
- Raw frames remain in replay.
- Reliability is trained/calibrated as a shadow signal before it can weight RL.
- No Mamba internal (`delta`, hidden norm, etc.) is assumed to be uncertainty.

## Install

```bash
pip install torch numpy pytest
# Optional, for the target backend:
pip install mamba-ssm
# Optional, for the environment:
pip install crafter
```

## Smoke test

```bash
PYTHONPATH=. python tests/smoke_test.py
```

## Batch convention

For a sequence of `T` observations:

```text
obs:       [B,T,C,H,W]
actions:   [B,T-1]
rewards:   [B,T-1]
continues: [B,T-1]
```

`actions[:, t]` causes `rewards[:, t]`, `continues[:, t]`, and the transition
from state `t` to state `t+1`.

## Mamba-3 integration

The official Mamba-3 kernels and recurrent cache API are evolving and may be
hardware/version specific. `temporal.py` isolates that dependency. Sequence
mode is wired when the installed API matches; recurrent `step()` deliberately
raises until pinned to the exact installed official version instead of silently
implementing a wrong cache.

The GRU backend is not the intended research model. It exists so every other
component and every indexing invariant can be tested on any machine.

## Recommended experimental order using the final graph

1. Representation-only: mask prediction, effective rank, spatial/state probes.
2. One-step deterministic predictor control.
3. Hard-mode mixture predictor; verify mode precision and usage.
4. Temporal backend comparison under identical interfaces.
5. Reward/continuation calibration.
6. Shadow reliability calibration against held-out multi-step latent error.
7. Full imagination policy training.
8. Enable confidence weighting only after held-out calibration succeeds.
