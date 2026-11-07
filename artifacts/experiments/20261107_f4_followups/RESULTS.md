# P1-F4 follow-ups: results against PLAN.md's declared readings

## 1. Danger at K = 8: the trained world, not the sampler (`steps8.py`, `readings/k8/`)

Same checkpoint, populations, seeds, reader protocol and interface classes as F4's readout; only K = 4 -> 8.

| reading | K = 4 | K = 8 | truth / recorded | declared band |
|---|---|---|---|---|
| risky-class death (interface) | 0.036 | 0.043 | 0.128 | world: within 0.02 of K = 4 |
| ready zombie damages (curves) | 0.082 | 0.086 | 0.321 | world: within 0.04 |
| lethal hits (curves) | 0.040 | 0.050 | 0.335 | - |
| sleeping agent hit by a ready zombie | 0.099 | 0.133 | 1.00 | - |
| adjacent-class damage (interface) | 0.054 | 0.056 | 0.159 | - |

Verdict (declared rule): **the trained world** draws danger rarely at any step count; doubling the steps moves it by
<= 0.01-0.03. Unlike F3 (3 -> 16 steps: risky death 0.036 -> 0.088), F4's sampler is not the bottleneck. The gains are
step-invariant: false wakes 0.008-0.024 per stratum, woken at energy 9 1.00, coal / iron 0 -> 1 0.98 / 0.92,
increments unchanged, every interface unlock within 0.015 of K = 4 (sapling 0.06 -> 0.07). Woken beside a ready
zombie 0.54 (damaged 0.13), beside a cooling one 0.18 (truth 0.007): the wake / hit decoupling also persists.
