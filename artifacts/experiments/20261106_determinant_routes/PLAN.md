# Determinant routes: does changing a blind determinant through either input route change the generated consequence?

User 11-06 ("the third experiment"). First version fe8d8b2f; **amended after Codex's review**
(`../20261106_determinant_review/REVIEW.md`, all points verified against the code and the smoke, all adopted; list at
the end). Frozen P1-F3 checkpoint 10, inference only.

## Questions (bounded, as amended)

For each consequence the determinant curves found BLIND (`../20261106_determinant_curves/RESULTS.md`):
1. Does changing the actual determinant through the denoiser-side input (the cached frame t - 1; for counts the
   denoiser's own action input) or through the memory (the belief after frame t - 1 and a_{t-1}) change the generated
   event, including interactions and wrong-direction changes?
2. Is the determinant recovered by the declared readers from the actual inputs and from location-resolved states,
   with controls?
3. What remains UNRESOLVED rather than forced into a causal label?
The experiment establishes route effects under these swaps and probe recoverability. It does not identify a unique
faulty layer or computation, and it does not compare or select architectures.

## Already established (not re-measured)

Clean coarse tokens read every visible determinant (asleep, every HUD value) at 100% with lighting; this rules out
simple perceptual aliasing, not geometry or use under noise. The heads' probability mass sits on generated-frame events
(Codex: 0.0023% of wake-unlock mass on frames still drawn asleep), so the generated frame is the readout. On Codex's
hostile bank, memory-only coarse evidence draws health-0 frames (~4%) while the cache version draws the zombie: a
route behaviour on that bank, not a law. Codex's matched binary readers: single-state zombie-vs-cow readability falls
98% (fine) -> 81% (embedding) -> 61% (belief) -> 56% (memory) while the counterfactual-referenced readout stays 100%.

## A. Simulator-edited pairs (the hostile-path bank method)

Backgrounds: dev episodes, exact replay (recorded frame t - 1 equals the replayed render bitwise), first eligible t
per (stratum, episode) in label order, t >= 8. One edit; the successors from the RECORDED step key (shared draws); the
natural successor must equal the stored frame t; truth = the simulator's successors (pairs whose successors do not
differ as the rule says are excluded and counted).

| pair | background (state t - 1) | d0 (natural) | d1 (edit) | strata | consequence site | truth |
|---|---|---|---|---|---|---|
| SLEEP | asleep, energy <= 8, no zombie / skeleton within 3, no arrow; action NOOP | asleep | is_sleeping = False | light >= / < 0.5 (120 each) | map tokens + lighting | asleep / awake |
| ENERGY | asleep, energy 8 at t - 2 and t - 1, same exclusions; NOOP | 8 | 9 | (200) | map tokens + lighting | asleep / awake |
| COAL | awake, DO, faced coal, wood pickaxe, no mob on it | DO | NOOP, same frames | starting count 1 / 2 / 3 (67 each) | coal slot | n + 1 / n |
| WOOD | awake, DO, faced tree, no mob on it | DO | NOOP, same frames | starting count 1 / 2 / 3 (67 each) | wood slot | n + 1 / n |

WOOD is the control: the same relation, performed by F3. Counts use the action edit because a frame edit of the faced
block left the unedited history showing it (smoke: the edit itself looked like a collection); actions never appear in
frames. Each member is a plausible state on its own; the CROSSED cells (one route from d0, the other from d1) can be
mutually inconsistent states: they are interventions, not in-distribution inputs.

**Context, belief, cache.** Frames t - 8 .. t - 2 and their beliefs as the actor interface reads them (S1 encode,
fixed random gaze); frame t - 1 re-encoded at its recorded fixation. Belief over the full real prefix in one causal call,
batch 3: the two members and a SAME-INPUT SHAM (natural frame, recorded action = S1 encode's inputs). Cache: the
interface's context passes, one context draw per pair shared by both members, plus a sham cache from S1 encode's
stored tokens. Shams (declared tolerances, ~2x the 27-pair smoke's maxima 0.0197 / 0.0095 / 0.0054; bf16 perception at
different batch shapes): natural tokens vs S1 encode max |diff| < 0.04 (asserted), sham belief vs S1 encode relative
< 0.02, sham cache vs member cache relative < 0.011, a repeated generation bitwise equal every 25th pair (asserted).
A sham beyond tolerance invalidates the run.

**Factorial.** Four cells (denoiser side d0 | d1) x (memory d0 | d1), 16 steps, K = 8 per cell, the same generator
per pair in all cells (common random numbers); heads with that cell's belief. Readouts: (i) the event by the frozen
frame readers (asleep; the slot's value), recorded successors through the same readers; (ii) the cascade's first
evaluation projected between the pair's two simulated successors over the consequence site, f = (x - A).(B - A) /
|B - A|^2 (A = d0's successor, B = d1's), deterministic.

**Availability** (both members; held-out readability of member identity from ONE state, logistic C = 1 standardised on
the fitting folds, 5 folds by episode with both members in a fold; random-label control = labels flipped per pair; the
paired ratio |member difference| / background spread beside it). Location-resolved: s1 the determinant site of frame
t - 1 (tokens + lighting), s2 percept embedding, s3 belief after the percept, s4 belief after the action, s5 memory
tokens, s6 frame t - 1's temporal keys / values (layers 4, 8) at the determinant site and at the consequence site, all
tokens kept, at the inference context level (0.1) and at one controlled stress level (every context frame at 0.67;
NOT the training distribution, where each frame and group draws its own level), s7 the generated frame's residual
stream at every consequence-site token after each of the 8 blocks (first evaluation, cells 00 and 11). Site means are
reported beside, descriptively (averaging can hide information). Raw tensors are saved.

### Declared readings (A)

All four cells with episode-bootstrap 95% intervals (2,000, seed 1106); oriented to the truth (Tr = the true contrast):
the diagonal (11 - 00); sufficiency of each input (putting d1 into the d0 run: 10 - 00 denoiser side, 01 - 00 memory);
necessity (removing d1 from the d1 run: 11 - 01 denoiser side, 11 - 10 memory); the interaction (11 - 10 - 01 + 00).
Each effect: CARRIES (lower bound >= Tr / 3), REVERSED (upper < 0), NO EFFECT (upper < Tr / 3 and lower > -Tr / 3),
otherwise UNRESOLVED; interaction PRESENT if its interval excludes 0. A small diagonal is reported as "no adequate
correct diagonal contrast", never as "route unused": the simple effects are read first. Sleep by light, counts by
starting count (coal vs wood compared within count). Availability: RECOVERED (>= 0.9) / PARTIAL (0.6-0.9) / NOT
RECOVERED (< 0.6) BY THIS PROBE; CONTROL FAILED if the control leaves 0.4-0.6 (then that stage is not read);
IDENTICAL INPUTS where the edit cannot reach (counts: s1-s3, s6). No stage is named a causal loss site; a failed
probe is "not recovered by this probe". For counts, readability of DO at a token is action identity (the action
conditions every denoiser block), not evidence of a computed or bound collection event: descriptive only.

## B. Hidden determinants (descriptive)

Rows: the determinant curves' natural dev rows (adjacent zombie, awake; mob hits). Targets: READY (an adjacent zombie's
cooldown <= 0) and KILL (the faced mob's health <= the sword's damage). Out-of-fold logistic readers (5 folds by
episode) from (1) the after-action belief (the memory's source) and (2) a CURRENT-PERCEPT control (frame t - 1's
percept embedding + the action), against (3) OBSERVABLE REFERENCE HEURISTICS from the visible history (not ceilings):
READY = no damage taken while an adjacent zombie was adjacent in the previous 5 steps; KILL = initial health minus
the DO hits on that mob over its contiguous visible span ending now (the scan stops when it leaves the 9 x 7 view).
AP with episode-bootstrap intervals, overall and by stratum (READY: lethal / safe health; KILL: cow / zombie), and the
world's event rate by truth x belief decision. No inference-versus-use verdict. **Disclosure:** the smokes computed
these values on the same rows (first version: belief 0.79 / 0.89 vs heuristics 0.92 / 0.99; amended version adds the
current-percept control: 0.67 / 0.84); they are not fresh confirmatory outcomes.

## What it feeds (no automatic follow-up)

Route effects and recoverability per consequence, with UNRESOLVED where the intervals say so; then a literature sweep
on what they show and a design discussion with the user.

## Amendments after Codex's review (11-06)

1. Locus logic removed (it called energy "NOT AT INPUT" although readable at its own cache site); location-resolved
   determinant AND consequence sites, lighting included, raw tensors saved; failed probes = "not recovered by this probe".
2. BINDING / OPERATION / OUTPUT verdicts removed (DO conditions every block); count readability descriptive.
3. Route verdict replaced by the full factorial (cells, simple effects both ways, interaction, reversed, UNRESOLVED);
   Codex's XOR counterexample (00 = 0, 10 = 1, 01 = 1, 11 = 0) now reads both sufficiencies CARRIES, both
   necessities REVERSED, interaction PRESENT (checked on CPU).
4. Hidden-state "ceiling" renamed observable reference heuristic; KILL restricted to the contiguous visible span (47 of
   3,600 histories had used hits from before an invisible interval); current-percept control added; no
   BELIEF-LIMITED / AVAILABLE verdict.
5. Episode bootstrap (pairs from one episode resampled together); counts selected per starting count 1 / 2 / 3 and
   compared within count; the random-label control gates each stage; same-input shams for belief, cache, generation
   with declared tolerances (the first version compared a NOOP-member belief with a recorded-action belief); reader
   parameters, raw generated periphery / lighting, seeds, source and checkpoint hashes saved; 0.67 labelled a stress
   level; crossed cells labelled interventions; UNRESOLVED allowed everywhere.

## Limits

One world seed; one-step generation from real histories; 16 steps; fixed gaze. Linear finite-sample readers: RECOVERED
shows availability to a linear map from one state; NOT RECOVERED is not erasure (Elazar et al. 2021; Hewitt & Liang
2019). The factorial is an interchange intervention at two input sites (Heimersheim & Nanda 2024: sufficiency vs
necessity, interactions, scope limited to these swaps); simulator-edited counterfactuals rather than noise (Zhang &
Nanda, ICLR 2024).
