# Records (object beliefs): design v3 (2026-10-26)

History: v1 per-record hypotheses + soft assignment (incoherent identities, Codex) -> v2 scene particles of object
files (SWB; coherent but M-fold cost) -> v3: ONE coherent association per step, per-object position distributions,
layered ownership, no dormant state. The user asked whether particles change anything compared with one coherent
hypothesis plus per-object distributions, and whether their cost is necessary. Answer below.

## What particles add, exactly
Per-object position uncertainty is represented equally well by per-object distributions (factored) as long as objects'
uncertainties are independent. Joint particles add only CROSS-object structure: (1) which record is which object
(association / identity), (2) exclusion (two objects cannot share a cell), (3) shared causes. Incoherence in v1 came
from SOFT association, not from factoring: a single HARD association per step is coherent (each record is one object
with its own hidden state); it is just sometimes confidently wrong. So the cost of particles buys representing
association uncertainty, nothing else.

## Is association uncertainty frequent or consequential? (A14b, data; A14 v1 RETRACTED)
A14 v1 kept the true returning entity on its real path while testing alternatives, so it could block them from the
entry cell, and omitted zombies stopping to attack (Codex). A14b (`eda_identity_posterior_v2.py`) removes every
identity in question from the blocker set, adds the attack stop (per-particle cooldown) and records health.
500 replayed episodes, 161,724 steps, perfect observer, the game's mob rules, births not included (lower bound):
- zombies: 2,555 re-entries (gap <= 32), 460 with a reachable alternative; P(true) < 0.9 in 7.2% of re-entries
  (1.1 per 1,000 steps); the MAP association wrong in 2.7%, mostly after longer gaps (gap 1-2: 8 / 143 ambiguous,
  3-8: 27 / 198, 9-32: 35 / 119); alternatives (posterior >= 0.1) differing in health: 10 events (0.06 / 1,000
  steps); in cooldown readiness: 0.
- cows: 4,126 re-entries, 1,188 with an alternative; P(true) < 0.9 in 7.2% (1.8 / 1,000 steps); MAP wrong 2.4%;
  differing in health: 37 events (0.23 / 1,000 steps).
Reading: split identity posteriors are not negligible (7% of re-entries) but rare per step, and consequential splits
are rarer still. One association per step stays a reasonable APPROXIMATION; it needs a revision mechanism (below).

## Brain and other fields
- Humans track positions much better than identities; identities swap when targets come close (Pylyshyn 2004);
  tracking identities competes with tracking locations for resources (Cohen, Pinto, Howe & Horowitz 2011).
- Perception holds one interpretation at a time and switches stochastically between alternatives (multistability as
  sampling over time, Gershman, Vul & Tenenbaum 2012), rather than holding many in parallel.
- Deep object-centric world models keep one slot set (SAVi, SlotFormer, G-SWM, SCALOR, OC-STORM); practical trackers
  keep one association per frame (SORT, DeepSORT, ByteTrack, TrackFormer). Multiple hypotheses are kept only where
  ambiguity is the core difficulty: MHT / PMBM (pruned to a few global hypotheses), RBMCDA (particles over
  associations only, per-target filters analytic; Sarkka et al. 2007), SWB (scene particles; its tasks were built
  around long invisibility and identity). JPDA's soft association coalesces tracks (Blom & Bloem 2000).
Corrections (Codex): particles buy cross-object structure (identity, exclusion, correlated location / existence /
hidden-state uncertainty), not only identity; independent per-record sampling plus collision resolution changes the
joint. One consciously experienced interpretation does not mean one internal hypothesis (Gershman et al. model
sampling through the posterior); the brain argument supports economy, not a verdict.
Conclusion: v3 is an approximation chosen for cost; scene particles stay the documented upper end.

## Design v3
- Records: K exchangeable object files (working number 8): presence flag, appearance descriptor (slow, changeable),
  dynamic state (per-record recurrent state = a slot of the working state; SlotSSM-style), position belief (a small
  hypothesis set, working number, as agreed). Visibility is derived each step, never a lifecycle state.
- Layers: the store is the background; records are foreground over it. Ownership = visibility: each location has one
  visible owner, the front-most record whose footprint covers it, else the background. Occluded content receives no
  evidence and keeps its trace; competition happens only among foreground candidates.
- Association: per step, records compete for foreground evidence (learned compatibility with interactions between
  competing records, as SWB's matching; Gumbel / straight-through hard sample in training), giving ONE coherent
  partition. Hard is not automatically valid: "one object = one record" (no splitting, no absorbing two objects) is
  what the mechanism must learn, and a probe target. Unclaimed foreground evidence goes to an inactive record.
- Revisability (deferred commitment, MHT's N-scan idea): when the association posterior is split (top < 0.9; A14b:
  ~7% of re-entries), the alternative pairing is kept for N steps (working number) with both hidden-state assignments;
  later evidence either confirms or swaps the records' states and histories; after N steps the leader is committed.
  Cost is negligible at ~1 split per 1,000 steps, and it repairs the wrong-history problem a later position fix
  cannot (Codex).
- Correction: amortised (the update is a learned network, as differentiable particle filters): it reweights each
  record's own hypotheses (including non-detection where a record was expected but the background owns the location,
  scaled by the viewing-conditions reliability) and updates owned records from their evidence. Trained end to end by
  the world losses, with calibration checks. No explicit observation likelihood is needed, so the agreed observation
  stage (energy score) stands unchanged.
- Lifecycle: presence on / off; inactivation is a learned transition (dies, leaves) plus evidence; capacity releases
  the longest-invisible, least-supported record and resets its state; releasing is forgetting, so a later appearance
  is a birth. Births = an inactive record activating (location + appearance in its latent), conditioned on context,
  action and other records' sampled events (parents). Option: released records leave store entity traces with age
  and uncertainty (object-trace cells) for returns; they are memory of past observations, not rendered background.
- Presence: the belief keeps each record's presence probability (tentative entities stay uncertain for the actor).
- Imagination (updated 10-27, generator v3): no concrete world is sampled. The generator samples the next PERCEPT from
  the predicted belief and the same update absorbs it (belief-MDP simulation; Kaelbling, Littman & Cassandra 1998).
  Records' predicted motion is the update's own deterministic predict step (propagating each hypothesis
  distribution); the stochastic outcome appears in the generated percept. Superseded wording: "the simulated world is
  one concrete sample ... the actor still reads its belief, updated by the same network from the generated percepts
  (Codex; POMCP's separation of hidden world and observation history). Cost per imagined step: one world + one belief.
- Store writes: only where the background owns the location in the (single) association.

## Checks (evaluation labels only)
Existence / location / kind separately; identity switches and their consequences (hidden state, counts, future
outcomes) rather than simulator identity alone; out-of-view coverage vs A3; first-sight precision (arrows); spurious
record lifetime and false confirmation through imagination; store contamination; drift during interactions.
