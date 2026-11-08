# S2 proposal v2: what imagination must get right for the actor (11-07; v1 bfafb929 reviewed by Codex)

User 11-07: build S2 first; "think about what makes an actor successful, this is cardinal: we need a coherent world,
the necessary and sufficient requirements should be tested by S2"; P2's objective is chosen afterwards, from where the
candidate world fails. S2 reads a frozen world (F4 first, as the known baseline: this is a post-result harness design,
not a blind preregistered test of F4; the harness is frozen before it judges any later change). It trains nothing and
selects no architecture by itself. Codex's review (`artifacts/experiments/20261107_s2_danger_review/S2_REVIEW.md`):
every point taken below, with one qualification (depth-1 conditionals stay valid, item L2a).

## 1. What the actor needs

P3 (CURRICULUM 3, LOSSES): PMPO with a frozen BC prior, from real beliefs after real prefixes, H = 16 imagined steps,
lambda-returns with a learned value, survival-weighted continuation; the actor and value read the corrected belief of
each imagined step. PMPO moves the policy by the sign of sampled imagined advantages through its critic and
regularisation: imagined action values that order actions as reality does, on the states it visits, are the evidence
that the update points the right way, not a theorem that it does (v1 overstated this; value equivalence, Grimm et al.
NeurIPS 2020, concerns specified policies, value classes and Bellman updates). Errors that compound over 16 steps
matter more than one-step ones (simulation lemma, Kearns & Singh 2002), and timing matters, not only rates: 11-07's
checks found F4 applies energy 9 -> wake correctly but raises energy while asleep at 0.27 of the true rate, so imagined
sleep lasts ~3.7x too long and WAKE_UP's payoff arrives late (the 09-02 actors lost to BC through WAKE_UP).

Requirements and the failure each prevents:
- **N1 action contingency.** Futures differ across actions where real ones do; otherwise no advantage signal.
- **N2 consequence calibration in both directions, with timing.** Reward- and termination-bearing events, and the
  prerequisites that time them (needs, energy, recovery), at the right probability and speed. A miss hides a real
  advantage; a false event creates one the actor will exploit.
- **N3 coherence.** Each generated trajectory obeys its own observable rules (entities persist and move legally,
  inventory bookkeeping, prerequisites, health and termination); joint futures, not only marginals, are right.
- **N4 stochasticity where reality is stochastic.** Risk appears as sampled outcomes (10-03: deterministic worlds draw
  no hits).
- **N5 the actor's inputs survive imagination.** The belief after generated steps keeps what its OWN generated history
  determines (inventory, health and needs, nearby threats, sleep), judged against that history.

No static test proves sufficiency; S3 settles it. S2's closest proxy is the real simulator return of a policy that
chooses by imagination, against BC (L3).

## 2. Three layers

**L1 coherence (N3).** TEST_SPEC S2.1 / S2.2 kept (counts, 4-step persistence, legal motion, revisits), on a
**terminal-inclusive** dev bank (P2_DATA.md's admission contract, applied to DEV: starts admitted whatever the recorded
future's length; a branch that outlives the recorded death continues the precommitted action schedule; each
trajectory absorbs on its own termination; time truncation is not padded as death). The 2,236-window bank (12
terminations, all at step 16: survival-censored, Codex reproduced) is kept only as a labelled legacy comparator.
Termination rule as installed: health <= 0 or lava or the time limit; "health 0 implies done" is checked, the converse
only where the other causes are absent. Camera alignment: generated trajectories are not aligned to the recorded
player path; motion and revisit checks use the generated frame's own coordinates, as S0 defined them.

**L2 consequences (N2, N4, N5).**
- **L2a, depth 1 from real prefixes:** the situation is the real one, so the simulator oracle (replayed state, fresh
  keys, same action) is the truth for the next step: the determinant-curve families (exp 2, M1, C1, crafting) with
  misses and false events separately, the contrast across actions, and conditioning on what the history shows.
- **L2b, depth k > 1 (after branches diverge):** two different measurements, never conditional comparisons across
  branches. (i) Trajectory distributions from a shared real prefix, model vs simulator, under the same exogenous action
  schedule or the same feedback rule (each acting on its own observations): event timing (time to damage, to death,
  to energy 9, to the next need decrement), return and death distributions, joint summaries (the energy score as a
  reported diagnostic), stratified by information at the shared start. (ii) Internal consistency of each generated
  trajectory with the laws observable in it (bookkeeping, observed prerequisites, health and sampled termination).
- **N5:** probes of beliefs after k generated steps scored against targets read from that trajectory's own generated
  observations and rewards; real-percept replay through the same update as the control (P2_READINESS 4's design).

**L3 decisions: a one-step planning stress test** (not PMPO itself). From real dev starts (random-play and
expert-mixed), all 17 physical actions, the action with the highest imagined 16-step value (m model futures, BC
continuing on the imagined belief) is executed in the real simulator, then BC continues on real observations through
the full pipeline (render, perception, belief, BC; its cost measured). Primary reading: the real return of this
imagination-guided policy minus BC's, paired by start and parent episode, with selection and evaluation draws
independent. The privileged oracle ranking (replayed full state) is a labelled diagnostic only: a correct belief can
disagree with a hidden-state-specific ranking (Codex's exact counterexample: 50%), so it is never a blocker. Limits
kept explicit: BC continuation is P3's starting policy, not every later actor; a truncated 16-step return cannot
certify delayed consequences or the lambda-return bootstrap; expert-mixed starts widen coverage, not to every state a
competent actor visits.

## 3. Power and readings (to agree before the harness binds; amends TEST_SPEC's two-blocker rule only once agreed)
- **The harness passes a correct world first:** the simulator oracle (and exact toy models: hidden-state, stochastic
  reward) must pass every reading at the chosen sample budgets; S0-style corruptions (wrong-action predictions,
  delayed and fake consequences, censored admission) must fail. Budgets are set by the effect sizes that matter;
  unresolved differences are reported as unresolved, never as wrong (Codex: 4 model / 8 oracle draws agree 47.8% on an
  exact stochastic model).
- **Outcomes per reading: pass / fail / unresolved / unmeasured**, with parent-episode intervals; absolute error bounds
  beside ratios, and absolute rates for false events (ratios to zero are undefined).
- **A declared hierarchy:** primary = L1's blockers, L3's real-return difference, and a short list of decision-critical
  L2 families with population weights (damage and death given danger, energy / sleep timing, food and drink, the
  first achievements of the tech tree); everything else descriptive, so a rare tool count is not a universal veto.
- Pass: P3 runs on that world. Fail: the failing requirement is the target of the next change, chosen after
  diagnosis (incorrect event probability vs incorrect timing vs inconsistent consequences), never by training longer.

## 4. Cost (to measure in the harness build)
L1 on the terminal-inclusive dev bank ~20-30 min; L2 ~30 min; L3 with all 17 actions at 500 starts, m = 4, H = 16:
544k model frames and 1.088M simulator steps with full-pipeline BC (Codex's count), so likely hours; budgets follow
the power check, not the reverse.

## 5. Open for the user / review
1. **Fixation contract:** CURRICULUM 3 makes fixation part of the actor's action; TEST_SPEC S3 says the SAME controller
   in both arms with fixation learning evaluated later. S2 uses the controller P3 will actually use, on model and
   simulator observations alike; random gaze as a declared extra control. Needs a decision before L3 is built.
2. **BC first:** L3 needs BC (P3's own first stage, CURRICULUM's recipe, reused unchanged by P3).
3. **Primary L2 families and weights** (3, hierarchy): a proposal to agree, not to tune after reading F4.
