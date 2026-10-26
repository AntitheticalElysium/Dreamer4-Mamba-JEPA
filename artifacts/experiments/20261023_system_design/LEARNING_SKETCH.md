# Learning sketch v2 (2026-10-25) — SUPERSEDED by STEP_SPEC.md v0 (2026-10-26) where they differ

A sketch to check coherence, not a spec. Route 1 working proposal: one belief update reads real or generated
perception. v2 resolves the interfaces Codex flagged on v1 (posterior input route, KL wording, footprint, store
exceptions, change target).

## Modules
- Periphery encoder E_p: frame (map + HUD rows alike) -> bounded-capacity continuous patch tokens p, modulated by a
  viewing-conditions latent; cross-patch context OPEN (PERCEPTION.md v2). History-free.
- Fovea encoder E_f: higher-capacity crop at fixation f (anywhere, HUD included) -> glimpse tokens g.
- Belief B = self-location + working state W (Mamba) + K object records with per-object position hypotheses and one
  coherent association per step (RECORDS.md v3; per-record recurrent state = slots of the working state) + store S
  (background layer).
- Update U(B, a, P): (1) discount global components: self-motion (predicted from the action, checked by global
  alignment) and global photometric change (light level, sleep darkening); (2) records: propagate hypotheses, match
  detections EXCLUSIVELY (each detection explains at most one record), correct, discover from unexplained change,
  update existence; (3) store: write traces of seen cells, apply edit events; (4) Mamba step.
- Generator G(B, a, c, eps): renders content from codes, then a final stochastic observation stage (conditions +
  shared per-frame noise eps) gives actual-percept samples P^ = (p^, g^ at f), change-map probabilities, reward,
  termination. Codes c: per-record move / event, births (four sources, below), first-seen terrain, faced-cell
  outcome, self-state events (changes to the agent's own body / inventory, wherever they are displayed).
- Posterior Q(c | B, a, P_next) (training only, local windows); prior p(c | B, a) (local context + record history +
  global context).
- Heads on B: consequences (reward, termination), TSP predictor, fixation policy, actor, critic.

## Structure the model is given
- Records render a LOCAL footprint (one cell in Craftax; a learned small extent in general). This bounds what one
  record paints; it does not prevent identity switches or duplicates, which one hard association per step
  handle (and probes check).
- The store is static except through EDIT EVENTS from three sources: the agent's action at the faced cell (mining,
  placing, eating), entity events (an arrow destroying a table / furnace), and slow autonomous processes on stored
  traces (plants ripen on a timer; traces are time-stamped). Craftax source: these are the only map writes.
- Births are events from four sources: the undetected-entity rate (edges and interior spawns), out-of-view or dormant
  records returning (e.g. Craftax DO revivals), and parent records (skeleton -> arrow); all conditioned on action and
  context. Requirement = predict these observable appearances faithfully; dormant records are one implementation.

## Real step (world training on the corpus)
1. Encode o(t+1) -> p. TSP predicts the glimpse at f(t+1) from (B_t, p) BEFORE reading it; then g = E_f(crop).
2. One-step dynamics: c ~ Q(B_t, a_t, P(t+1)); P^ = G(B_t, a_t, c). Losses:
   - latent prediction: energy score between samples of P^ (eps) and the stop-gradient encoding of P(t+1);
   - KL(Q || prior), applied twice with different stop-gradients: KL(sg Q || prior) trains the prior ("dyn"),
     KL(Q || sg prior) with a small weight trains Q and its inputs toward predictable codes ("rep"). Both are the SAME
     direction; the KL also pays for information about the realised outcome (a calibrated prior on a fair coin still
     has positive KL), so its size is not a leakage measure by itself;
   - change map: cross-entropy against the SOFT label-free target (A9b: detector trained on generically corrupted day
     pairs; soft and observation-derived with measured reliability, not ground truth: ECE 0.005 day / 0.016 night);
   - consequences (reward, termination; the agreed grounded targets, no screen-region-specific heads).
3. B(t+1) = U(B_t, a_t, P(t+1)).
4. Anti-collapse regulariser on p (SIGReg-type, LeJEPA).

## Gradient routes into the encoders (declared)
- E_p receives gradients through (a) the belief path (p enters B, B feeds G and the heads), (b) the posterior's input
  (P(t+1) enters Q; the generator loss and the rep KL flow back through Q) — this is the weak coupling, scaled by the
  rep weight and an explicit gradient scale on Q's input, (c) TSP (p is TSP's input), (d) the regulariser. Never
  through the latent prediction target.
- E_f receives gradients only through its contribution to the belief (and Q's input on glimpse codes).
- No gradient from the fixation policy, the actor or the critic into the world.

## Imagined step (same U)
Actor picks a_t (and f) from B_t (stop-gradient) -> c ~ prior -> P^ (glimpse rendered from the SAME sampled content;
first-seen terrain sampled once and written to the store) -> B(t+1) = U(B_t, a_t, P^) -> reward / termination heads.
World frozen during actor training.

## Rollout training of the world (before the actor)
- Posterior-conditioned unrolls (all data, recorded actions): codes from Q on real futures, U reads its own P^,
  pointwise losses to real targets, truncated gradients. Explains recorded futures; does not make the prior causal.
- Prior-sampled unrolls on exogenous windows: eps = 1.0 episodes (255,660 sixteen-step windows) and short windows
  inside expert-mixed episodes (24,515 four-step, 1,258 eight-step; overlapping): two samples, energy score on grounded
  vectors (change maps, consequences) plus joint checks (counts, identity persistence).
- Confounding diagnostic: prior predictions on random-branch vs expert-branch steps with matched histories AND actions.

## Where entities get supervision
| entity | existence / location | appearance |
|---|---|---|
| moving, visible | change map, latent dynamics, consequences | latent dynamics, TSP if fixated |
| still, visible | latent dynamics; shared per-cell encoder; consequences | weaker unless fixated (acceptable) |
| out of view | hypotheses; re-entry and birth predictions | n/a |
| terrain edit | faced-cell / entity edit events -> store; change map | store revisit prediction |

## Bypasses and what blocks them
- encoder erasure / collapse: regulariser + fixed-target losses (soft change map, TSP pixels, consequences).
- independent change without records: away from edit events and entering cells, only records move.
- a record painting several objects: local footprint; duplicates / swaps: exclusive hard ownership per step; swaps are a probe target.
- codes leaking or ignored: local posterior windows, small alphabets, KL balancing; codes are the only random input.
- future leaking to the actor: Q exists only in training; the actor reads B.
- store keeping entities: revisit prediction penalises stale entities (learning claim).

## Open learning claims (probe targets)
One object per record; still entities kept via shared appearance; store / record separation; night supervision
(A9b: AP 0.61 vs 0.93 reference); prior calibration for rare events; confounding size; record drift during
interactions (DLPWM, 2511.06136, traces unstable control to it).
