# Short dense-vs-endpoint continuation: results (384 updates per arm)

Protocol: `PLAN.md`, `LAUNCH.md`. Two 384-update continuations of P1-HUD-Change checkpoint 10, identical except the
time reduction of the blind-rollout auxiliary: **endpoint** scores the student's last masked step against the
observed teacher belief, **dense** averages every step 1..H (`loss.py`). Corrected fork (rebased segment IDs), parent
belief preserved plus one mask token, weight .1, horizons {2, 4, 8, 16}, same draws and replayed HUD fixations.
393,216 frames per arm. Raw outputs: `run/SUMMARY.json`, `run/guards.json`, `PHASE_*.json`.

## Primary: exact-history hunger pairs (24 pairs, K = 32, readers 100% held out)

| Checkpoint | Decrement when due | When not due | Contrast (truth 1) |
| :--- | ---: | ---: | ---: |
| Parent (HUD-Change) | .0651 | .0664 | -.0013 |
| Endpoint 384 | .0924 | .0924 | .0000 |
| Dense 384 | .0911 | .0924 | -.0013 |

Dense minus endpoint contrast -.0013 (paired episode 95% [-.0039, 0]). Both continuations raise the decrement rate
equally in both histories: more drops, none of them timed.

## Factual guards (first 120 DEV episodes, random-policy windows)

| Generated | Recorded | Parent | Endpoint | Dense |
| :--- | ---: | ---: | ---: | ---: |
| Risky P(negative reward) | .143 | .090 | .064 | .062 |
| Risky P(termination) | .147 | .081 | .060 | .058 |
| Adjacent P(negative reward) | .157 | .084 | .076 | .075 |
| Coal-collection class, P(unlock) | 1.000 | .618 | .702 | .743 |
| Iron-collection class, P(unlock) | 1.000 | .164 | .172 | .168 |

Both short continuations lose part of the parent's danger forecast (endpoint - parent risky P(negative reward)
-.026, 95% [-.035, -.017]) and gain some coal. No timing effect at this budget; a short-continuation null does not
rule out longer or from-scratch learning. One training seed.
