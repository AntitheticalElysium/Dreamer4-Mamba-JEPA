# P1-F4 follow-ups (user 11-07: "run all 4 one by one; report the results back to me")

Four frozen measurements on P1-F4's selected checkpoint (`../20261106_p1f4/run/ckpt_10.pt`, sha 1b70edcc...), one GPU
job at a time, in this order. Each part's reading is declared here before its code runs. No training.

## 1. Danger at K = 8 (`steps8.py`)

F4 draws danger about 3x too rarely at K = 4 (`../20261106_p1f4/RESULTS.md`): risky-class death 0.036 (recorded
0.128), a ready zombie damages 0.08 (truth 0.32). F3 drew more danger with more sampler steps (risky death 0.036 at 3
steps -> 0.088 at 16).
- Claim: the under-draw is the 4-step sampler. Competing: the trained world draws danger rarely at any step count.
  Separating control: the same checkpoint, populations, seeds and reader protocol at K = 8, the finest grid
  (d = 1 / k_max: the flow-trained step, Dreamer 4's sampler at its smallest step).
- Instruments unchanged: exp 2's determinant curves (all four populations, as in F4's readout, readers refitted by
  the same protocol) and the actor interface (all classes), at K = 8.
- Declared reading: sampler-limited if risky-class death >= 0.07 AND ready-zombie damage >= 0.20; the trained world if
  both are within 0.02 / 0.04 of K = 4 (0.036 / 0.08); otherwise partial. Read alongside (descriptive): whether K = 8
  keeps F4's gains (false wakes <= 0.15, coal / iron 0 -> 1, increments, recursion not measured here) and the sleeping
  agent's hit (0.10 at K = 4, truth 1.00).
