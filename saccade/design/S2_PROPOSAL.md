# S2 proposal: what imagination must get right for the actor (11-07, for Codex's review)

User 11-07: build S2 first; "think about what makes an actor successful, this is cardinal: we need a coherent world, the
necessary and sufficient requirements should be tested by S2"; P2's objective is chosen afterwards, from where the
candidate world fails. S2 reads a frozen world (any P1 or P2 candidate: F4 first). It decides whether P3 can run on
it and, if not, which failure must be fixed. It trains nothing and selects no architecture by itself. This replaces
nothing yet: TEST_SPEC's S2 section (counts, persistence, events, revisits, uncertainty) becomes its first two layers.

## 1. What the actor needs

P3 (CURRICULUM 3, LOSSES): PMPO with a frozen BC prior, from real beliefs after real prefixes, H = 16 imagined steps,
lambda-returns with a learned value, survival-weighted continuation; the actor and the value read the corrected belief
b_t of each imagined step. PMPO moves the policy by the SIGN of the imagined advantage. So the actor improves on BC
exactly where imagination ranks actions as reality does, on the states it visits, within 16 steps plus the value's
bootstrap. Model accuracy matters only through those rankings (value equivalence, Grimm et al. NeurIPS 2020;
objective mismatch, Lambert et al. L4DC 2020); the simulation lemma bounds the value error by the per-step model
error times H^2 (Kearns & Singh 2002), so errors that compound over 16 steps matter more than one-step ones.
Our history: the 09-02 actors lost to BC entirely through WAKE_UP (a payoff beyond their 2-step horizon), and the
10-03 H16 panel found risk decisions need sampled futures (one faithful future 57% of the value, 16 samples 90%).

Requirements, each with the failure it prevents:
- **N1 action contingency.** Imagined futures differ across actions where real ones do. A world that ignores the action
  gives every action the same value: no signal, the actor stays BC (or drifts on noise).
- **N2 consequence calibration, both directions.** Reward- and termination-bearing events (unlocks, health changes,
  death, sleep / wake, eating, drinking, crafting, mining, kills) at the right probability given history and action.
  A miss hides a real advantage (F4's danger: hits drawn at 0.1-0.3 of truth would make risky actions look safe); a
  false positive creates a fake one the actor will exploit (a false WAKE_UP, a phantom health gain).
- **N3 coherence over the horizon.** The imagined state stays a valid world for 16 steps: entities persist and move
  legally, counts and HUD stay consistent, termination follows health 0. Consequences at step k are computed from the
  imagined state at k - 1, so an incoherent state makes every later consequence meaningless.
- **N4 stochasticity where reality is stochastic.** Risk must appear as sampled outcomes; a world that collapses to the
  likeliest outcome draws no hits (10-03: deterministic worlds), so every gamble looks safe.
- **N5 the actor's inputs survive imagination.** The belief computed from generated percepts keeps what the actor's
  decisions read (inventory, health and needs, nearby threats, sleep state); otherwise a correct world is misread.

Sufficiency cannot be proven by a static test; only P3 / S3 settle it. The closest frozen test is to measure P3's own
mechanism: do imagined action values rank first actions as true values do, and does choosing actions by imagination
gain or lose real value against BC? That is layer 3.

## 2. Three layers

**L1 coherence (N3; TEST_SPEC S2.1 / S2.2, kept):** free-running rollouts (m = 8) on the 2,236 dev exogenous 16-step
windows, the frozen reader on generated and recorded futures: per-kind visible counts (block > 30% and > 3 SE), 4-step
persistence of zombies / cows / skeletons (block > 0.10 below recorded), legal motion (no diagonal steps; the S0 checks),
HUD-termination consistency (health 0 <=> terminated), revisit consistency by time since observed. The S0 oracle and
corruptions supply references and power, as before.

**L2 consequences (N2, N4, N5), at depth:** the determinant-curve families (exp 2 / M1 / crafting / C1 strata, one
instrument) measured on free-running rollouts at depths 1, 4, 8, 16, conditioned on the IMAGINED situation as the
reader sees it (e.g. P(damage | an adjacent zombie in the imagined frame)), with the real-history depth-1 reading
beside it. For each family: drawn vs true rate (truth from the simulator oracle: the replayed state stepped with fresh
keys under the same actions), misses and false positives separately, and the contrast across actions (N1 at the
event level). N5: the S1.2 belief probes applied to beliefs after k generated steps, against the same probes on real
histories.

**L3 decisions (the sufficiency proxy):** from real start states (dev; random-play AND expert-mixed, so the states a
competent actor visits are covered), for a fixed action set per state (all 17, or the subset BC gives > 1%), m
imagined futures of H = 16 steps under the BC policy after the first action (BC reads the imagined belief), giving
the imagined Q(s, a) as P3 would compute it (rewards, survival weighting, the frozen value at the end when one exists;
without a value, the 16-step return). True Q(s, a): the simulator from the replayed state with fresh keys, the same
first action, BC acting on the real percepts' belief, m' futures. Readings:
- value calibration: imagined vs true Q, per state, bias and error;
- ranking: sign agreement of imagined and true action gaps, weighted by the true gap (does imagination order first
  actions as reality does?), against the null of random ranking and the oracle (simulator vs itself: the noise ceiling);
- exploitation: the action imagination prefers vs BC's action, scored by TRUE Q. Positive: imagination improves on BC
  where P3 would move; negative: its errors are exploitable. Per event family, which errors drive the negative part.

L3 needs no exogenous windows: the truth is re-simulated with fresh keys, so recorded actions never leak the future
(the A8 confound does not arise). This also removes the coverage limit for evaluation.

## 3. Declared readings (before any world is measured; numbers for review)
- L1: TEST_SPEC's blockers unchanged.
- L2: per family with support, block if a reward- or termination-bearing event is drawn < 0.5 or > 2 of its true rate
  at depth 1 or 16, or if its false-positive rate exceeds twice the truth's; everything else reported.
- L3: block if the exploitation gain is negative beyond 3 SE, or ranking agreement is not above the random null beyond
  3 SE; report agreement as a fraction of the oracle's.
- Pass all: P3 runs on that world. Fail: the failing requirement (N1-N5) is the target of the next change (P2 or
  otherwise), chosen after diagnosis, never by training longer.

## 4. Cost (estimates, to measure)
F4 generates ~280 frames / s at K = 4. L1 ~290k frames (~20 min); L2 ~4 families x 4 depths x ~4k futures x depth
(~30 min); L3 500 states x ~6 actions x m = 4 x 16 steps ~190k frames (~15 min) plus the simulator's m' = 8 on CPU.
About 1.5 h per world on the GPU, plus the instruments' build.

## 5. Open for review
1. L3's continuation policy: BC (what P3 starts from) vs the actor itself (not available before P3); BC reading the
   imagined belief is exactly P3's setting at its first update. L3 therefore needs BC trained first (P3's own first
   stage, CURRICULUM's BC recipe, reused unchanged by P3), and the simulator side needs the online loop (render ->
   perception -> belief -> BC), which the actor interface's code mostly has.
2. Value bootstrap: P3's value does not exist before P3; L3 at a pure 16-step return measures the world, not the value.
3. Thresholds in 3 are first proposals; S0-style corruptions should set their power before they bind.
4. Whether L2's depth-k truth (oracle futures given the full state) should condition on the imagined or the real
   situation: imagined is what the actor sees; real isolates the world's per-step error.
5. Glimpse: the actor's fixation is part of its action in P3; S2 uses the random prior (as TEST_SPEC) and reports a
   chooser-gaze replicate where it changes an event family.
