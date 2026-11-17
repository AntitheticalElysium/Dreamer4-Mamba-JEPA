# Dense-vs-endpoint continuation extended to 2,000 updates: results

Protocol: `PLAN.md`; live audit `../20261111_dense_mse_2k_review/AUDIT.md`. Both arms resume from their own 384-update
checkpoints of `../20261111_dense_mse_short/` and stop at 2,000 updates in total (2,048,000 frames each); loss and
readout code are byte-identical to the short run, constant LR after 384. Raw outputs: `run/SUMMARY.json`,
`run/guards.json`, `PHASE_*.json`.

## Primary: exact-history hunger pairs (24 pairs, K = 32, readers 100% held out)

| Checkpoint | Decrement when due | When not due | Contrast (truth 1) |
| :--- | ---: | ---: | ---: |
| Parent (HUD-Change) | .0651 | .0664 | -.0013 |
| Endpoint 2k | .0990 | .0990 | .0000 |
| Dense 2k | .0990 | .0990 | .0000 |

Dense minus endpoint contrast .0000 (95% [0, 0]); both minus parent +.0013 (95% [0, .0039]). As at 384 updates, the
decrement rate rises equally in both histories: the world draws more drops but does not time them.

## Factual guards (first 120 DEV episodes, random-policy windows)

| Generated | Recorded | Parent | Endpoint 2k | Dense 2k |
| :--- | ---: | ---: | ---: | ---: |
| Risky P(negative reward) | .143 | .090 | .124 | .105 |
| Risky P(termination) | .147 | .081 | .113 | .096 |
| Adjacent P(negative reward) | .157 | .084 | .079 | .065 |
| Coal-collection class, P(unlock) | 1.000 | .618 | .731 | .630 |
| Iron-collection class, P(unlock) | 1.000 | .164 | .237 | .224 |

At 2k the endpoint arm improves on the parent's danger forecast (risky P(negative reward) +.034, 95% [.017, .052])
and on coal and iron unlocks; dense improves less on danger and coal and is lower than the parent on adjacent
damage. Neither reduction teaches the timing distinction at this budget. One training seed; the guards are rates on
fixed histories, not a test of memory.
