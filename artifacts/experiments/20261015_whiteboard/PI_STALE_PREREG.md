# Teaching PI's belief to read stale representations: stale-context augmentation, preregistration

2026-10-15 night. Written before any code for this study exists. A user question; Track B; one change to PI's training input.

## Question

Exact copies keep persistent content but freeze its representation: stale rendering and content-dependent position effects
(R34, R36). In an idealized rollout with true transformations, this costs PI's belief:
- chase toward the player, −30 to −64%;
- consequence content, +25 to +62% error.

Can the belief learn to read persistent *information* regardless of its representation's age, and does that carry into
recursion?

## Evidence that motivated it

- **R34:** stale vs fresh contexts with true transformations, on the same windows. Content-level decisions are intact
  (gate AUC ≥ 0.97, scroll 0.99); relational and consequence readings degrade.
- **R35:** on fresh inputs the belief keeps temporal context across slot changes (health-drop gate AUC 0.97-0.98).
- **R36:** re-deriving an additive position code changes nothing. R15: rendering steps are unpredictable from history.
  So the representation cannot be refreshed, and the belief must be invariant to its age (WHITEBOARD row 1).
- **R33:** PI's one-step chase is near real and decays with recursion depth.

## Arms

Both arms are trained in one process per seed set, on identical batches, from PI u3000 (weights and optimizer state of
`pi_seed<s>/state.pt`). 1,000 more updates (3,000 → 4,000) with the PI objective unchanged; new continuation streams
(windows 20261026 + s, noise 2026102600 + s).
- **piplain:** PI training as is (the control for the extra updates).
- **pistale:** identical, except the BACKBONE INPUT.
  - Per window, m ~ U{0..14}.
  - Context frames 15−m..14 are replaced by the idealized stale reconstruction (`stale_context.py`'s rule), built from the
    current model's own label-free inference on the fresh window: scroll estimate, IC posterior, edit posterior; real tokens
    at changed cells; exact aged copies elsewhere. No gradient.
  - The copy, the posterior, the targets and all losses use the fresh frames, exactly as PI. Only the belief sees stale
    inputs.

## Readings (both seeds; checkpoints u4000)

- **P1 (invariance):** the stale-substitution gap (`stale_context.py`, m ∈ {8, 14}, windows with ≥ 1 scroll, pooled).
  - The relative drop of the chase toward-probability (stale vs fresh) for pistale is ≤ half of piplain's.
  - AND the relative increase of the map-change content error is ≤ half of piplain's.
- **P2 (recursion):** zombie pursuit approach at depth 6-16 (`pursuit.py`, bright; real 0.38) is higher for pistale than
  piplain.
- **H (no harm, recursion vs piplain):**
  - persisting-mob score d16 ≥ piplain − 0.05;
  - seen terrain d16 ≤ piplain + 0.5pp;
  - phantoms ≤ 0.05 at every depth;
  - |cow census d16 − 1| ≤ piplain's + 0.05;
  - PI's one-step conditions (valid ≥ 90%, duplicate / absent ≤ 5%, static ≤ max(free, 0.5%), movement ≥ 15%, cow ≥ 20.5%).

**Verdict:**
- **"Belief invariance learned and it carries into recursion"** if P1, P2 and H hold at both seeds.
- **"Invariance learned, recursion unchanged"** if P1 and H hold and P2 fails.
- **"Not supported"** if P1 fails at either seed.
- Anything else is reported as partial.

**Secondary:**
- decision proxies (zombie-adj, cow-adj, health AUC16 and rates);
- gate AUC and belief cosine in the substitution;
- training losses of both arms.

## Known limits

- The augmentation covers stale representations under true transformations, not the model's own transformation errors.
- One budget (1,000 updates), no tuning. Fresh confirmation stays sealed.
