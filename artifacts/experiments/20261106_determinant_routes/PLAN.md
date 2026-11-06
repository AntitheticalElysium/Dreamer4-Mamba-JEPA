# Determinant routes: where does each blind determinant stop? (user 11-06: "the third experiment"; for Codex's review)

Frozen P1-F3 (checkpoint 10), inference only. One question: for each consequence the determinant curves found BLIND
(`../20261106_determinant_curves/RESULTS.md`), where between the frame and the generated consequence is its
determinant lost? Not run; nothing launches before Codex's review and the user's go.

## What is already established (not re-measured here)

- **Perception is not the limit** for any visible determinant: the asleep state and every HUD value are read at 100%
  from F3's coarse tokens plus lighting (determinant curves, 8 readers, all dev frames); adjacent digits sit 6-49x their
  within-value spread apart; the tokens are compositional for digits (the "+1" step points the same way in every count
  slot, cosine 0.85-0.99; a digit reader fitted on wood + stone reads coal at 93%).
- **The heads read what the frame draws** (WORLD-DRAWN for wake, death, damage), so the generated frame is the readout.
- **Routes divide the work** (review of Codex's hostile path, `../20261106_mamba_hostile_path_review/REVIEW.md` 3):
  memory-only evidence draws the HUD consequence (health 0 at half the natural rate) but almost never the entity;
  the cache draws entities. The coal event is in the belief (consequence trace B1, AP 1.0) while the coal slot copies.
- **Small changes shrink at the percept embedding**: Codex's fine zombie signal is 14x weaker than coarse there
  (fusion 4.4x, projection 3.4x), before Mamba (1.8x). The digits' own profile through that stage is unmeasured.
- **Counts** follow each (item, digit) transition's own exposure (Spearman 0.91); **wake** fails one step earlier: the
  world does not hold the asleep state (~50% awake per step at every energy, every action; 0.71 at night, 0.36 by day).

## Claims, competing explanations, separating observations

For a visible determinant d (value d0 vs d1 in the last context frame, t - 1), the generated frame t can receive d
through two routes: the CACHE (frame t - 1's tokens, read by temporal attention at layers 4 and 8 and spread by the
spatial layers) and the MEMORY (the belief after frame t - 1 and a_{t-1}, projected to 4 tokens every spatial layer
reads). Competing loci: (L1) d never reaches the denoiser's inputs at a usable scale (projection / Mamba / memory
projection; or the cache, at the noise the denoiser trained with); (L2) it reaches an input but the output does not
follow it through that route; (L3) it reaches the consequence tokens inside the denoiser but the output mapping does
not act on it (for counts: BINDING, the slot never receives its item's event, vs OPERATION, it receives it and fails
to apply "+1"). Separating observations, all on the same simulator-edited pairs: a 2 x 2 factorial (cache t - 1 from
d0 or d1) x (memory from d0 or d1) on the generated consequence; the paired availability of d at every stage from the
frame token to the generated consequence tokens' hidden states; the working relation (wood increments) as the
positive control for the counts.

For a hidden determinant (zombie cooldown, mob health) the competing loci are (H1) the belief does not infer it from
the history vs (H2) the belief has it and the world does not use it. Separating observation: how well the after-action
belief reads the hidden variable against what the visible history makes inferable.

## A. Visible determinants: simulator-edited pairs (the hostile-path bank method)

Backgrounds: dev episodes, exact replay (`artifacts/eda/replay.py`; the recorded frame t - 1 must equal the replayed
render bitwise), first eligible t in each episode in label order (at most 2 per episode for the count pairs), t >= 8.
One edit to the simulator state at t - 1, re-rendered through Craftax; the successor of each member from the RECORDED
step key, so both share every random draw; truth = the simulator's successors (asserted to differ as the rule says).

| pair | background (state t - 1) | d0 (natural) | d1 (edit) | action at t - 1 | consequence tokens | truth |
|---|---|---|---|---|---|---|
| SLEEP | asleep, energy <= 8, no zombie / skeleton within distance 3, no arrow | asleep | is_sleeping = False | NOOP | map 0-62 + lighting | d0 asleep, d1 awake |
| ENERGY | asleep, energy 8 at t - 2 and t - 1, same exclusions | energy 8 | energy 9 | NOOP | map 0-62 + lighting | d0 asleep, d1 awake |
| COAL | awake, DO, faced coal, wood pickaxe, coal 1-3, no mob on it | DO | NOOP (same frames) | the member's | coal slot (70) | d0 coal n + 1, d1 n |
| WOOD | awake, DO, faced tree, wood 1-3, no mob on it | DO | NOOP (same frames) | the member's | wood slot (68) | d0 wood n + 1, d1 n |

240 SLEEP pairs (120 light >= 0.5, 120 below: the light effect), 200 of each other. WOOD is the control: the same
relation, which F3 performs (0.82 at n 1-3). The action is set to NOOP for SLEEP / ENERGY (Craftax overrides it while
asleep; for the awake member it removes the recorded action's effect). For the counts the ACTION is the edit (DO vs
NOOP on identical frames): a frame edit of the faced block (coal / tree -> path, the first design) left frames t - 8 ..
t - 2 showing the block, so the edit itself looked like a collection; in the CPU/GPU smoke the wood member with a path
faced still drew the increment (first-call functional ~0.1 of the way from n + 1 to n). Actions never appear in
frames, so the action edit is history-consistent. The factorial's denoiser side is then the denoiser's own action
input (adaLN) instead of the cache, and the availability stages that can differ are s4, s5 and s7.

**Context, belief, cache.** Frames t - 8 .. t - 2 and their beliefs as the actor interface reads them (S1 encode, the
fixed random gaze); frame t - 1 re-encoded at its recorded fixation for both members (R1: the natural member's tokens
equal S1 encode's within storage precision). The belief over the full real prefix in one causal call, batch 2 (the two
members share the prefix bitwise, differ only in frame t - 1's percept). Cache: interface's context passes, one context
noise draw per pair shared by both members.

**Factorial.** Four conditions per pair: (cache d0 | d1; for counts the denoiser's action input) x (memory d0 | d1); generated frame t at 16 steps, K = 8 per
condition, the same generator per pair in all four (common random numbers); heads read with that condition's belief.
Readouts: (i) the event by the frozen frame readers (asleep; HUD slot value), recorded successors through the same
readers; (ii) the cascade's first evaluation (the trace's `first_call`, one shared draw per pair) projected between the
pair's two simulated successors over the consequence tokens: f = (x - A).(B - A) / |B - A|^2 (A = d0's successor
tokens, B = d1's): 0 = d0's outcome, 1 = d1's, deterministic.

**Availability profile** (both members, every pair; paired ratio |member difference| / background spread, and
UNPAIRED held-out readability: logistic C = 1, standardised on the fitting folds, 5-fold by episode, the two members of
a pair in the same fold; random-label control = labels flipped per pair at random):
s1 the determinant token(s) of frame t - 1 (SLEEP: player cell, mean map token, lighting; ENERGY: cell 66; COAL / WOOD:
the faced cell); s2 percept embedding; s3 belief after the percept; s4 belief after the action; s5 memory tokens;
s6 frame t - 1's temporal-layer keys / values (layers 4, 8) at the determinant token(s) and at the consequence tokens,
at the inference context level (0.1) and at the training median (0.67: diffusion forcing noises every context frame);
s7 the generated frame's residual stream at the consequence tokens after each of the 8 blocks (first evaluation).

### Declared readings (A)

Route, with Tr the true contrast and N = E(d1, d1) - E(d0, d0), C = E(cache d1, memory d0) - E(d0, d0),
Mm = E(cache d0, memory d1) - E(d0, d0) (events oriented to the truth; pair bootstrap 2,000, seed 1106):
NONE if N's upper 95% bound < Tr / 3; otherwise CACHE if C >= 2N/3 and Mm <= N/3, MEMORY if the reverse, EITHER if both
>= 2N/3, JOINT if both <= N/3, MIXED otherwise. Availability per stage: READABLE >= 0.9 held-out, WEAK 0.6-0.9, ABSENT
< 0.6 (the random-label control must read 0.4-0.6). Locus:
- **NOT AT INPUT** (L1): s5 and s6-at-consequence both below READABLE; the stage named is the first of s2-s6 to fall
  from READABLE (projection, Mamba, memory projection, cache entry), and s6 at 0.67 vs 0.1 says whether training noise
  hides it in the cache. Counts: s4-s5 only (the frames are identical).
- **UNUSED** (L2): s5 or s6 READABLE but the route verdict NONE for that input (C, Mm < Tr / 3).
- **OUTPUT** (L3): s7 READABLE at the consequence tokens' last block yet N < Tr / 3.
- COUNTS: **OPERATION** if the coal slot's s7 is READABLE like wood's while coal's N < wood's N / 3; **BINDING** if
  coal's s7 is WEAK / ABSENT where wood's is READABLE, with coal's s5 READABLE.
- SLEEP asks one more thing, the leak: E_asleep(d0, d0) ~ 0.5 is the known natural rate; which single input carries the
  asleep output (the route verdict on "stays asleep") and whether its strength differs by light.

## B. Hidden determinants: is the hidden variable in the belief?

Rows: the determinant curves' natural dev populations (zombie adjacency, awake; mob hits). Targets: zombie READY (an
adjacent zombie's cooldown <= 0) and KILL (the faced mob's health <= the sword's damage). Readers (logistic, 5-fold
by episode, out-of-fold scores) from the after-action belief b_pred(t) (the memory's only source). History ceiling,
hand-coded from the visible history alone (labels, not the model): READY = no damage taken while that zombie was
adjacent in the previous 5 steps; KILL = initial health (cow 3, zombie 5) minus the visible earlier DO hits on that
mob slot times the sword's damage. BELIEF-LIMITED if the belief's AP < the ceiling's AP - 0.15; AVAILABLE if within
0.05; between otherwise. Descriptive beside it: the world's event rate (the curves' futures) by truth x the belief
reader's decision.

## Smoke record (11-06, before review)

CPU build smoke: 11 pairs, every recorded frame t - 1 replayed bitwise, every natural successor equal to the stored
frame t, every edit changing the simulator's outcome as the rule says (0 excluded). GPU smoke (2-3 pairs per kind):
the pipeline runs end to end (R1 natural tokens within storage precision); verdicts at this n are meaningless and are
not read. Two bugs fixed: readers were fitted under no_grad; the first call received a [1, 1, 1] action. The
faced-block confound above was found here and replaced by the action edit. Part B has no smoke subset: its smoke run
computed the full-row values already (belief AP for READY 0.79 vs history ceiling 0.92; KILL 0.89 vs 0.99; both
BETWEEN by the declared rule). They are recorded here so that nothing is read silently; Codex may amend the ceiling
definitions before they are taken as results.

## What each outcome feeds (no automatic follow-up)

A localisation, per consequence, to one of: projection / Mamba / memory projection / cache entry (L1), a route the
denoiser does not use (L2), the output mapping (L3, with BINDING vs OPERATION for counts), or the belief's inference
(H1) vs use (H2). Each verdict then gets a literature sweep on that mechanism and a design discussion with the user. The
experiment diagnoses the trained world; it does not compare or select architectures.

## Limits

One world seed. Edits make single-variable counterfactuals on real backgrounds; SLEEP / ENERGY's d1 (a woken or
energy-9 frame after asleep history) is natural in kind (agents wake after asleep frames) but at energy <= 8 rare (a
hit). Readability is a linear, finite-sample reader: READABLE shows availability to a linear map; ABSENT is not proof of
erasure (Elazar et al. 2021; Hewitt & Liang 2019 control tasks). The factorial is an interchange intervention at the
two input sites (Heimersheim & Nanda 2024: denoising d1 into d0 measures sufficiency of a route; the reverse cells give
necessity), on in-distribution counterfactual inputs rather than noise (Zhang & Nanda, ICLR 2024).
