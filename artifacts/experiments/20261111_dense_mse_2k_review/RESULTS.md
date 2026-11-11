# 2,000-update result

Both continuations and readouts complete. Final CPU audit: all 19 pins match; exact stream/gaze/horizon replay,
Adam counters 2000, both inference exports bound to saved training SHA with native tensors equal.
No new training run was launched for this review.

| Native hunger decrement | Due | Not due |
|---|---:|---:|
| Simulator | 100% | 0% |
| Parent HUD | 6.51% | 6.64% |
| Endpoint 384 | 9.24% | 9.24% |
| Dense 384 | 9.11% | 9.24% |
| Endpoint 2000 | 9.90% | 9.90% |
| Dense 2000 | 9.90% | 9.90% |

Both final models produce 76/768 decrements per member. Every one of the24 histories has exactly the same
paired decrement count in each arm; not cancellation across contexts. Held-out readers qualify at1.0.
Nearest-real-endpoint geometry gives endpoint9.90/9.90%, dense9.77/9.90% (one discrepancy);
this is not a classifier-only null. Zero-width paired bootstrap here reflects identical finite-bank
differences, not proof of equality in the population or across training seeds.

Factual guards, identical DEV histories:

| Forecast | Recorded | Parent | Endpoint 2k | Dense 2k |
|---|---:|---:|---:|---:|
| Risky termination (442 histories) | .1471 | .0812 | .1134 | .0961 |
| Adjacent negative reward (2523) | .1570 | .0841 | .0794 | .0653 |
| Coal unlock on actual unlock histories (48) | 1 | .6175 | .7314 | .6297 |
| Iron unlock on actual unlock histories (44) | 1 | .1645 | .2366 | .2243 |

Both improve risky death and iron against parent on this panel, neither improves adjacent damage/reward.
Endpoint has higher factual guard point readings at2k. Dense's coal gain at384 (.7427) does not persist
at2k. Pair intervals in guards.json compare each arm to parent, not dense to endpoint, and quantify
evaluation-case variation conditional on one trained seed.

Conclusion: dense vs endpoint does not repair native use of hunger-phase evidence within this continuation;
both instead increase the unconditional drop rate. No proof that Mamba lacks all phase information,
that no from-scratch run could work, or that all memory fails. Do not extend this same objective again
on the strength of latent consistency alone; prefer a new objective accountable to native forecasts.
