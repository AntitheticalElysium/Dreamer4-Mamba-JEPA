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

## 2. Memory versus cache routes on F4 (`routes_f4.py routes`, then `analyse`)

The user's question: does the dense path supersede Mamba? Exp 3 (`../20261106_determinant_routes/`) answered it for F3
with a 2 x 2 swap on 842 simulator-edited pairs: the denoiser side (the cached frame t - 1 for sleep and energy, the
denoiser's own action input for coal / wood counts) and the memory (the belief after frame t - 1 and the action) each
from member d0 or d1. F3: sleep both routes, the memory dominant (memory sufficiency 0.53 CARRIES); energy no route
(the world never learned it); counts gated by the memory's action (memory NOOP -> no increment whatever the denoiser's
action).
- Unchanged: the bank (`bank.pt`, model-independent), `pair_state` (beliefs over the full prefix, caches through F4's
  own context pass), the 4 conditions with common random numbers (K = 8 futures), `factorial` and its effect classes,
  the shams. Changed: F4's checkpoint, its own sampler (K = 4 shortcut evaluations), readers refitted on F4 by the same
  protocol, F4's S1 reader. Dropped: exp 3's EDM-specific probes (the first-call functional, location-resolved
  availability at sigma 0.1 / 0.67): F4 has neither a sigma ladder nor that first call.
- Declared reading per family: Mamba BYPASSED if memory sufficiency and memory necessity are both NO EFFECT while a
  denoiser-side effect CARRIES; Mamba CARRIES if memory sufficiency CARRIES; otherwise shared / unresolved, as classed.
  My predictions (declared, not tests of architecture): sleep still memory-carried, the both-asleep cell falling from
  0.47 to near F4's natural false-wake rate; energy (now learned) carried by the cache side, where the energy token of
  frame t - 1 is one dense attention step away (F3's belief recovered energy only partially, 0.58-0.73); counts keep
  the memory gate, coal now near wood.
- Caveats carried from exp 3: crossed cells are interventions, not in-distribution states; one world seed. The context
  is F4's own (signal 7 / 8, Dreamer 4's corrupted context), not F3's sigma 0.1.

## 3. Movement with F4's own tokens (`routes_f4.py movement`)

T1's 64 movement pairs (`../20261106_decision_levels/readings/movement_bank.pt`: a move vs NOOP from the same replayed
frame, both successors simulated), the same `pair_state`, K = 16 futures per member through F4's sampler (K = 4).
Read as T1 did, in the evaluated model's own token space: the generated map + lighting vector projected on the axis
from the move successor's tokens (0) to the NOOP successor's (1), moved if < 0.5. No external reader is needed: the
axis is F4's encoding of the two simulated successors.
- Declared reading: movement holds if the move member moves >= 0.95 and the NOOP member <= 0.05 (F3: 1.00 / 0.00);
  otherwise a regression to explain before adoption.

## 4. Dense-attention engineering: profile and FlexAttention benchmark (`flex_bench.py`)

Engineering only (same mathematics; no reading selects architecture). Prototype as a replacement of `Block.dense_time`'s
chunked path inside the benchmark process (no core change before adoption): one `flex_attention` call per dense layer
on the flattened frames, mask = the window rule rebuilt from the existing (index, allowed) window tuple and decision 1
(physical queries read physical keys only), compiled (`dynamic=False`, default mode,
`torch._inductor.config.deterministic = True`); the streaming sampler keeps SDPA.
- Correctness: one real 64-step batch through F4's checkpoint, denoiser outputs and parameter gradients, flex vs SDPA,
  against SDPA's own bf16-vs-fp32 discrepancy as the tolerance scale.
- Determinism: two fresh processes, 4 updates each from the same init with flex -> checkpoints compared bitwise.
- Cost (sec / update, peak GiB, from `train.p1` as F4's checks): F3's recipe at its own split (1 / 4) and at F4's
  (2 / 8): what the smaller microbatches cost; F4 SDPA at 2 / 8 (the run); F4 flex at 2 / 8 and at F3's split 1 / 4;
  64 steps (bootstrap off and on) and 256 steps (on). The projected training hours from them, as checks.py computes.
- Declared reading: flex is adoptable if it matches SDPA within that tolerance, is bitwise reproducible across
  processes, and fits F3's split (peak <= 5.0 GiB on the 6 GB card); the hours are reported either way.
