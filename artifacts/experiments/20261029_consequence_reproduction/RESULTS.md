# One-step consequence deficit: independent reproduction (2026-10-29)

Claim, competing explanation and control: `reproduce.py` docstring (written before the run). Frozen P1 world seed 1,
checkpoint 10 (sha256 `74eccc43...`); dev split. Production streaming sampler (`model.sample` after `context_pass` on the
8 real context frames), fresh tuples (seed 2029), heads applied to both the generated and the recorded successor.
Episode-bootstrap SE (200 resamples) in parentheses.

| Contexts | Recorded death | Head on recorded successor | Generated: predicted death | Generated: sampled death | Recorded r < -0.05 | Head on recorded | Generated predicted | Generated sampled |
|---|---|---|---|---|---|---|---|---|
| Random, 6,000 histories x 4 futures | 0.367% (0.075) | 0.367% (0.075) | 0.0001% | 0 / 24,000 | 2.15% (0.18) | 2.25% | 0.67% (0.06) | 0.64% |
| Random-behaviour subset, 722 histories | 0.97% (0.36) | 0.97% | 0.0000% | 0 / 2,888 | 3.19% (0.63) | 3.25% | 0.76% (0.20) | 0.76% |
| Health <= 2 and a zombie at L1 distance 1, 3,805 histories x 16 | 12.77% (0.40) | 12.77% | 0.015% (0.005) | 8 / 60,880 | 12.75% (0.40) | 13.70% | 4.14% (0.21) | 4.21% |

Reading:
- The 10-28 finding reproduces with different code and draws, and is larger here (0.0001% vs 0.0034% predicted death on
  random contexts). The heads are calibrated on recorded successors (0.367% vs 0.367%; 12.77% vs 12.77%), so the deficit
  enters before the heads read a real successor: in what is generated, or in how the heads read generated percepts.
- In the risky contexts nearly every recorded negative reward IS a death (12.75% vs 12.77%: with health <= 2 a zombie hit,
  2 damage, kills). Generated futures carry negative reward in 4.1% but death in 0.015%: part of a consequence without
  its rule-implied completion. An observation for the diagnosis, not a conclusion about its mechanism.
- Engineering limit found: the streaming denoiser's temporal attention flattens B x 95 slots; SDPA raised "CUDA error:
  invalid argument" at B = 1,024 frames (97,280 rows) and ran at B = 256. P2 / P3 batches must stay below it (or chunk).
