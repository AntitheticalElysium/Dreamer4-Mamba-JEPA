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
- Determinism: two fresh processes per path (flex, and SDPA as the control), 4 updates of F4's preflight configuration
  each -> checkpoints compared within each pair. (Amended before running: F4's SDPA path is itself not bitwise across
  processes, model 4.7e-18 / optimizer 1.1e-19 in its preflight, so "bitwise" is replaced by "no larger than the SDPA
  pair's own discrepancy, both within checks.py's 1e-11", logged terms bitwise.)
- Cost (sec / update, peak GiB, from `train.p1` as F4's checks): F3's recipe at its own split (1 / 4) and at F4's
  (2 / 8): what the smaller microbatches cost; F4 SDPA at 2 / 8 (the run); F4 flex at 2 / 8 and at F3's split 1 / 4;
  64 steps (bootstrap off and on) and 256 steps (on). The projected training hours from them, as checks.py computes.
- Declared reading: flex is adoptable if it matches SDPA within that tolerance (flex vs SDPA in bf16 no larger than
  SDPA's own bf16 vs fp32 discrepancy, and flex vs SDPA in fp32 below 1e-4 relative: the mask exact on the GPU), its
  cross-process discrepancy is no larger than SDPA's, and it fits F3's split (peak <= 5.0 GiB on the 6 GB card); the
  hours are reported either way.

## 5. Danger, M1: health changes by cause (`danger_m.py m1`; user 11-07 "run those two experiments")

`danger_leads.py` (a4c297c2): F4 under-draws zombie hits even where the inputs determine them, and wakes a sleeping
agent (0.69) without drawing the health drop (0.11). Which part fails: the health token, entity-caused writes, or
zombies alone? Craftax-Classic game_logic.py: an arrow reaching the player costs 2 and wakes it; the hidden recovery
counter gives +1 health about every 26 steps awake (13 asleep) while all needs are met, and -1 about every 16 steps
awake (32 asleep) while a need is 0.
- Strata on dev (situation at t - 1; hostile-free = no zombie / skeleton within 3, no arrow in view): arrow adjacent
  (no zombie adjacent), awake and asleep; need damage (a need at 0, hostile-free), awake and asleep; recovery (needs
  met, health < 9, hostile-free), awake and asleep. Up to 1,000 rows each, K = 4 futures, exp 2's generation and
  readers (curves.generate, refitted by its protocol), F4's checkpoint under the canonical code (df702338).
- Sham (R0): 400 rows of exp 2's zombie population regenerated now equal F4's saved futures bitwise.
- Reading: per stratum, drawn vs true P(health drop) and P(health rise), episode bootstrap 95%.
  Entity-caused writes weak (H1) if the arrow ratio (drawn / true drop) <= 0.5 while need damage and recovery are
  within 0.7-1.4. The health token weak generally if need damage and recovery are both <= 0.5. Zombie-specific if
  arrows >= 0.7. Otherwise reported as mixed.

## 6. Danger, M2: memory versus cache for the hit (`danger_m.py m2build`, `m2run`)

Exp 3's 2 x 2 on hit pairs: a sleeping agent with a ready adjacent zombie at t - 1 (truth: hit, -7, woken) vs the
same simulator state with the adjacent zombies removed at t - 1 (truth: neither). Both successors simulated from the
recorded key and action; pairs where the edited member still loses health or wakes are excluded. pair_state, the 4
conditions with common random numbers (K = 8), shams and factorial as in part 2 (`routes_f4.py`).
- Read per cell: P(health drop) and P(woken). Besides exp 3's classes (against the true contrast), the share of the
  world's own effect each route carries: cache-only (E01 - E11) / (E00 - E11), memory-only (E10 - E11) / (E00 - E11),
  episode bootstrap.
- Reading: the routes differ (H2) if the wake is memory-carried (memory-only share >= 2/3, cache-only <= 1/3) while
  the drop is not (memory-only share <= 1/3, or no drop effect at all: diagonal <= 0.05); the same route if both
  shares point the same way. Caveat as exp 3: the edited history shows a zombie vanishing at t - 1 if it was visible
  before (an intervention).

## 7. Danger, three checks and two data questions (`danger_checks.py`; user 11-07 "run those 3 and report back")

From M1 / M2 (RESULTS 5-6): the health token under-responds to every cause while the same cached zombie wakes the
agent. Three theories (none tested): (T1) a single token whose cause arrives through attention or a hidden timer, not
through the action's conditioning, is under-learned (small loss share); (T2) a copy habit on HUD digits; (T3) F4's
loss weighting starves the noisiest levels, where a token's change-or-not is decided (F3 drew hits 2-3x more).
- **C1, the other HUD timers (F4):** food -1 (hunger, ~26 steps awake), drink -1 (~21), energy -1 awake (~31), energy
  +1 asleep (~11); hostile-free rows, no DO (eating / drinking), awake / asleep as the timer runs. Drink gets its HUD
  reader (cell 65, s_body column 2) by the same protocol. Reading: T1 (single token without an action trigger, not
  health-specific) if these are drawn <= 0.5 of their true rate; health-specific if >= 0.7.
- **C2, M1 on F3** (its own code, commit fa6edd1a; exp 2's 16 EDM steps; its own readers and S1 reader), plus C1's
  strata; sham: exp 2's zombie population regenerated equals exp 2's saved futures. Reading: an F4 regression if F3's
  ratios exceed F4's by >= 0.2 on need damage, recovery and arrows; long-standing if within 0.1.
- **C3, the decision level** (M2's 232 hit pairs, natural inputs: cache and memory of the hit member): the generated
  health token projected on the axis from the no-hit successor's token (0) to the hit successor's (1), and the same
  for the wake (map + lighting tokens): F4's estimate at pure noise (its first sampler evaluation) and its final
  sample; F3's denoised estimate at pure noise of sigma 5 (its sampler's start) and 20, and its 16-step sample.
  K = 8 noise draws per pair. Reading: the decision exists and sampling loses it (T3's side) if F4's pure-noise
  health projection is >= 0.5 while its samples draw <= 0.15; the conditional itself never leans to the hit (T1 / T2)
  if the projection is <= 0.2.
- **D1, data (CPU):** zombie hits in the train labels per (starting health, drop) against F4's drawn success in exp 2's
  zombie futures per starting health: per-value learning (the counts' non-shared flavour) if success follows each
  value's own exposure (Spearman >= 0.7 over cells with >= 30 futures); flat if not.
- **D2, tools (CPU):** a held tool's count + 1 (crafting's 'madeN', drawn 0.00-0.06) and its exposure in the train
  labels per tool and starting count, beside the first tool's (drawn 0.81-0.96). Descriptive.

## 8. Danger readings and adversarial checks (`danger_readings.py`; user 11-07: "run those readings... consider that
this might not be a regression, keep doing adversarial checks"; Codex's danger review taken: see RESULTS 9)

Done on saved data before this plan (CPU, undeclared, reported as such): (a) zombie-hit discrimination (ready minus
cooling) for fresh arrivals is about equal in F3 and F4 (+0.10 / +0.06 vs +0.07 / +0.07, awake / asleep); F3's higher
hit rates come with a higher rate of impossible drops (asleep cooling 0.30, truth 0.005); (b) the HUD reader's
confidence on generated health tokens is ~1 for both (below 0.9 in 0.3-0.4%). So "F4 regressed on hits" is not
supported as a discrimination claim. The timer table (RESULTS 7) is the remaining regression candidate.
- **R1, timer phase (adversarial: learning vs base rate).** Each timer's due time is visible in the history (steps
  since the HUD value last changed the same way; Craftax: food -1 every 26 awake steps, energy +1 every ~11 asleep,
  health +1 every 26 awake steps while needs are met, energy -1 every 31 awake). Strata: food -1 awake, energy +1
  asleep, recovery awake, energy -1 awake (C1's quiet rows); rows binned by steps since the last same-direction change
  (1-5, 6-10, ..., 31+, none in the episode yet), up to 250 rows per bin, the SAME rows for every model: F4 (K = 4), F3
  at 16 EDM steps, F3 at 3 (adversarial: F3's advantage could be its 16-step stochastic sampler). Truth hazard per bin
  from the labels. Reading: a model has learned a timer if its drawn rate rises at the bins where the truth is due
  (due minus early >= half the truth's contrast); a base-rate difference if its profile is flat (< a quarter of it).
  F4 regression on timers if F3 (16) learned a timer that F4 did not; not a regression if F3 is flat at a higher rate;
  F3's sampler if F3 at 3 steps loses what F3 at 16 has.
- **R2, attention of the health token (descriptive; attention mass is not causal).** M2's 232 hit pairs, both members,
  natural inputs, at the decision level (F4's first evaluation from pure noise; F3's at sigma 5): the attention mass of
  the health token's queries and, for comparison, of the lighting token's (the wake carrier) over key groups: the 4
  cells beside the player (current frame; and past frames in F4's dense layers), the query token's own past, the
  memory tokens, the rest; per layer, averaged over heads. Reading: routing to health is weak if the health queries put
  less than a quarter of the lighting queries' mass on the zombie cells; and if the mass moves when the zombie is removed.
- **R3, reliance at each signal level (frozen.py T1's instrument on the health token and the wake).** Same pairs and
  inputs; the generated frame's input set to the hit successor (own) or the no-hit successor (conflict), noised to each
  level (F4 signal 0, 2/8, 4/8, 6/8 at the sampler's step code; F3 sigma 5, 2, 1, 0.5, 0.2): the output's projection on
  the health and wake axes. Reading: where the output starts following its input (own minus conflict >= 0.5), and
  whether any level overrides a no-hit input toward the hit (conflict projection >= 0.5), as the truth (a certain hit)
  would require at low signal.

## 9. Is the timing information in the belief? (`belief_probe.py`; user 11-07: "run that check... find the cause")

R1 / R2 (RESULTS 10): neither world draws hidden timers or zombie readiness. The phase lies 11-31 steps back, beyond
the denoiser's 8-frame reach, so only the memory tokens (the predicted belief bp_t, which generation of frame t reads)
can carry it. Inference failure (the belief lacks it) or use failure (the belief has it, generation ignores it)?
- Targets (dev, simulator truth at t): food -1, drink -1, energy -1 awake, energy +1 asleep, health +1 (recovery,
  needs met), health -1 (a need at 0), on C1's quiet rows; zombie readiness (an adjacent ready zombie, awake adjacent
  rows, exp 3's READY) and the hit itself (awake and asleep adjacent rows). Up to 40k rows per target.
- Features: bp_t (the denoiser's memory input); controls: the frame t - 1's percept embedding + the action (no
  history); the visible-history reference: steps since the HUD value last changed the same way, binned (timers), exp 3's
  observable heuristic (readiness). Linear probes, exp 3's protocol (L2 logistic, 5 episode folds), out-of-fold AP with
  episode bootstrap, and the out-of-fold predicted hazard per since-bin beside the truth (R1's bins).
- Reading per target: the belief HAS it if AP(bp) - prevalence >= half of AP(reference) - prevalence; it LACKS it if
  <= 0.2 of it; in between, partial. Has + flat generation (R1) = use failure; lacks = inference failure. F4 vs F3
  (fa6edd1a's code): F4-specific if F3's belief has a target F4's lacks (AP difference >= 0.1). Linear probes give a
  lower bound on what the belief holds (Codex's 11-06 point: reader power); an absent reading is "not linearly present".

## 10. Does the hit reach the health token inside the denoiser? (`denoiser_probe.py`; user 11-07: "run the probe")

Belief probe (RESULTS 11): the memory input predicts the hit (AP 0.70 awake, 0.81 asleep), F4 draws it at 0.05-0.15.
Between the belief and the drawn digit, three explanations remain (Codex's "objective allocation, output-specific
routing"; R3's sampler lock-in): (a) routing - the cause never reaches the health token's activations; (b) the output
- it reaches them and the output (x-prediction) discards it or moves the digit too little; (c) the sampler - the first
estimate is right and later evaluations lose it. This probe localizes; it does not test why.
- Rows: every adjacent-zombie dev transition (t >= 8), awake (22,502; 3,613 hits) and asleep (1,619; 816 hits); F4
  only (the use failure is F4's question; F3's belief = F4's).
- The production path unchanged: interface.futures through determinant_curves.generate (8 real context frames, the
  recorded fixation, K = 4 sampler evaluations), ONE future per row (curves.K set to 1); the generated health digit by
  exp 2's readers. Captured by forward hooks (outputs untouched; sham: the first batch's readings equal a run without
  them): the health (63) and lighting tokens' residual at the input and after each of the 8 blocks of the first
  evaluation (pure noise); the health token after the last block and its x-prediction at each of the 4 evaluations.
- Targets: the hit (health(t) < health(t - 1), simulator truth). Linear probes as belief_probe.py (L2 logistic, 5
  episode folds, out-of-fold AP, episode bootstrap). Reference: the 4 memory tokens (the denoiser's copy of the belief;
  instrument check: within 0.05 of bp's AP, else the projection loses it). Baseline: the action alone (17-way one-hot:
  what adaLN alone could carry). Negative control: the health token at the first evaluation's input (pure noise),
  expected at the prevalence.
- Share(token, place) = (AP - AP(action)) / (AP(memory tokens) - AP(action)). Lean of the health x-prediction on hit
  rows: (x_hat - copy) . (truth - copy) / |truth - copy|^2 (0 = copies the previous digit's token, 1 = the hit's), beside
  its calibrated value at the first evaluation (the mean out-of-fold memory-token probe probability on hit rows).
- Reading (declared): (a) routing if the health token's share is <= 0.2 at every block of the first evaluation;
  (b) output if the share is >= 0.5 at the last block and its x-prediction either loses it (share <= 0.2) or carries it
  (>= 0.5) with a lean below half its calibrated value; (c) sampler if the first evaluation's lean is >= half its
  calibrated value and the drawn rate on hit rows is far below it. In between: partial, reported as such. The lighting
  token on asleep rows (the wake IS drawn, 0.69) is the positive control for a token the cause reaches; where in the
  blocks the share rises (spatial blocks 1-3 read the memory tokens and the action's adaLN; temporal blocks 4 and 8 the
  cache) is descriptive. Linear probes are a lower bound.

## 11. Where on the belief's path are the timers lost? + the timer data, exactly (`counters.py`, `belief_path_probe.py`; user 11-07)

User: "run the test... be careful - Craftax is relatively simple, don't claim that the model is naturally bad at this
too eagerly... most of the time, the problem was something that we built a certain way, or the data (maybe check it
as well, more precisely?)". Every timer is a belief-inference failure (RESULTS 11, 13). The game's four counters
(game_logic.py 1256-1325: hunger, thirst, fatigue, recover) are deterministic given the visible history.
- **Exact counters (data, CPU).** `counters.py` replays every train and dev episode through the simulator's own step
  (artifacts/eda/replay.py's keys and actions, its scan with the per-step state recorded) and stores hunger, thirst,
  fatigue and recover per state; exactness check: the replayed health, food, drink, energy and sleep equal the labels at
  EVERY state (else stop).
- **Data facts (train split, from the counters and labels; descriptive):** events per pass of each timer (food -1, drink
  -1, energy -1 awake, energy +1 asleep, health +1 recovery, health -1 needs) beside hits and coal increments; resets
  that change no digit (eating / drinking at 9, decrements at 0, energy capped at 9); the share of each timer's events
  whose previous reset falls inside the same training chunk (64 / 256 steps, the gradient's reach) - against episode
  starts; per behaviour-policy epsilon.
- **Probe (frozen F4, dev).** Up to 40k transitions (t >= 8). Features for the step producing frame t: the HUD tokens
  of frame t - 1 (cells 63-66, control: the readers read them perfectly), percept_embed(frame t - 1) (the belief's
  input), b_{t-1} (the belief right after it), bp_t (after the action: the memory's source). Targets at state t - 1:
  the digit values (health, food, drink, energy), the exact counters (hunger, thirst, fatigue, recover), and on asleep
  rows the steps asleep (an integration control: a duration of a visible state). Readers, 5 episode folds: ridge
  (linear, lambda 10 on standardised features; sensitivity at 1000) and an MLP (one hidden layer of 256, the
  nonlinear reader: Codex's point that a failed linear probe is only that reader's failure); out-of-fold R^2 (and
  accuracy of the rounded value for digits).
- Reading (declared): **input** carries a digit if percept_embed's R^2 >= 0.9 x the HUD tokens'; the **state keeps**
  it if b / bp reach >= 0.9 x percept_embed's; a counter is **encoded** if the MLP's R^2 from b or bp >= 0.5, **absent**
  if <= 0.1. Input lacks -> the belief never receives what it must count. Input has, state lacks the current digit ->
  the state does not keep the HUD. Digit kept, counter absent -> receives and keeps but does not integrate (then: the
  steps-asleep control says whether it integrates any duration). Counter encoded -> the representation exists and the
  hazard failure lies in its precision or its use. Not a claim about Mamba in general: a claim about this trained
  system and its data; construction and data causes are checked before any capacity explanation.
