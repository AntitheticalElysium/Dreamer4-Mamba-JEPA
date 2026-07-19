# Senior verification: Stage-2F reward-operator control — CONCUR

Date: 2026-07-19. Scope: independent verification of the companion's
Stage-2F round (154594a -> b133779 -> 79a5eb5 -> 6ad8f2d -> full runs ->
87a0afa -> 54e4c0d -> 0cc1b36) per the user's cardinal-verification request.

## Checks performed (all pass)

1. COMMIT CHAIN: exact claimed order verified in git history; protocol
   precedes implementation precedes preflight precedes training precedes
   evaluator precedes outcome; working tree clean at 0cc1b36.
2. ARTIFACTS: all 8 SHA-256 hashes match the review's table (preflight,
   both checkpoints, training report/raw, eval report/raw, paired
   analysis). Cosmetic note only: the review table's filenames differ in
   case/wording from the on-disk names (e.g. `stage2f_flz_s505.pt`,
   `stage2f_train_report.json`, `stage2f_analysis.json`); hashes bind them
   unambiguously.
3. STATISTICS REPRODUCED FROM RAW ROWS (my own recomputation, not the
   analysis file): F-LZ-F-R K8 AUROC -0.03977 (point exact; CI equivalent
   under bootstrap-seed variation); F-DZ-F-LZ zero-suffix -0.02903
   [-0.0376,-0.0211]; F-DZ-A +0.01794 [+0.0121,+0.0232]; all four per-arm
   zero-suffix means exact to 5 decimals; latent K2/K4/K8 deltas exact
   with CIs excluding zero. Zero discrepancies.
4. OPERATOR IMPLEMENTATION vs PINNED SOURCE: support construction matches
   Dreamer-CDP heads.py:132-144 odd-bin branch exactly (live-verified:
   endpoints +-4.85165184e8, strict ordering, exact symmetry, exact center
   zero); original-space two-hot interpolates and clips correctly (targets
   sum to 1, exact scalar reconstruction); symmetric decode matches
   outs.py pred() including the mirror-magnitude PAIRING (hand-checked:
   flip(negative)[i] pairs bin -b with positive[i] bin +b, so uniform
   probabilities decode to EXACT zero — live-verified).
5. F-R REFERENCE HONESTY: F-R k8 reward predictions and fork rows are
   byte-identical to the committed Stage-2C C-LR raw block (max diff 0.0)
   — the no-retraining reference is real.
6. BLINDNESS: the trainer imports no evaluation modules; FINAL tier
   untouched.
7. SUITE: 163 passed, 1 pre-existing warning — as claimed.

## Judgment

- The ZERO-INITIALIZATION FACTORIAL is the standout methodological catch:
  a naive two-arm operator swap would have attributed a significant
  -0.04 K8 AUROC initialization effect to the DreamerV3 distribution and
  produced a wrong causal conclusion. The three-arm design was necessary
  and is now proven so by its own data.
- The split verdict is correctly framed: the mechanism gate is labelled
  permissive, adverse ranking/event-MAE points are retained as explicit
  negative evidence, and the operational rejection follows the
  pre-registered conjunction. "DreamerV3/CDP reward-distribution-aligned"
  is the right maximum source claim.
- I find NO implementation errors and NO unsound design decisions.
  Limitations (one seed, third DEV reuse, low-power fork CIs, fixed .10
  coefficient) are all self-disclosed and correctly handled as
  interpretation bounds, not hidden.
- ROUTING CONCURRENCE: F-DZ rejected; categorical-operator/calibration
  search on DEV closed; the next control should be the separately
  preregistered shared-representation reward/action-relevance auxiliary
  (DeepMDP/DBC vs TACO-inspired vs BYOL-AC-motivated, source-verified,
  branching from C-LR with the scalar reward objective unchanged, with a
  vacuous-head control — repairing the H3 vacuity). The specific choice
  among the three awaits tri-party consensus. All NO-GOs stand.
