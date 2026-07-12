# Architecture status: what is validated, what is not, and are the gates rigorous?

Companion to `2026-07-12-senior-review-phase-b.md`. State as of commit `0c23a82`.

Legend — **V**: verified against pinned primary source; **E**: empirically
validated in our own controlled runs; **T**: covered by passing regression tests;
**U**: unvalidated (no evidence either way); **R**: refuted; **F**: fixed after
refutation, fix verified at small scale only.

## 1. Component-by-component status

| # | Component / claim | Status | Evidence |
|---|---|---|---|
| 1 | Transition contract (a_t → s_{t+1} → r_{t+1}, c_{t+1}) | V, T | Dreamer-CDP source `rssm.py`; tests `test_action_t_affects_reward_t_plus_1_but_not_earlier_reward`, leakage tests |
| 2 | Crafter adapter (discount vs truncation, action space, `info` fields) | V, T | crafter source pinned; `test_crafter_continuation_uses_environment_discount` |
| 3 | Replay indexing (`previous_actions`, BOS, episode-bounded slices) | T | Phase A tests; audit |
| 4 | EMA target covers the whole encoder (no trainable target path) | V, T | I-JEPA/V-JEPA practice; `test_target_encoder_is_a_disjoint_frozen_full_copy`. Reference impl `m3_hjwm/` violates this (spatial mixer on target branch outside no_grad) — retired as ground truth |
| 5 | **Representation objective as originally specified (§3)** | **R → F** | All unregularized variants collapse (rank ~12→3 in 300 updates; full objective 12.5→4.2). VICReg variance+covariance terms fix it (12.6→12.7 masked, 12.5→15.4 unmasked, JEPA loss still improving). Fix validated at 300-update scale only — the long-run Phase B must confirm |
| 6 | Masking (masked context vs unmasked) | U (open) | No verified-source precedent for the hybrid; no effect on collapse; unmasked mildly ahead at 300 steps (rank 15.4 vs 12.7, JEPA 0.21 vs 0.25) and removes the train/imagination input mismatch. Decide in the Phase B re-run; `mask_ratio=0` switch added |
| 7 | Encoder yields linearly separable local semantics | E (weak) | Fixed probe: 0.93 vs 0.76 majority — but the *untrained* encoder scores the same. The probe detects information destruction, it cannot certify learning |
| 8 | Registers carry inventory/global state (spec hyp. 7) | U | Inventory probe exists; not yet run on a non-collapsed trained encoder |
| 9 | Mamba-2 recurrent adapter == official semantics | V, T | Pinned `state-spaces/mamba`; sequence/step equivalence, reset isolation, bf16 gradient tests |
| 10 | Mamba-3 on this GPU | R | "Only tested on H100" (mamba3.py:319); dep-pin fragility; batch-stride kernel failures. Decision: Mamba-2 backend, GRU control |
| 11 | Mamba beats GRU *for this task* (spec hyp. 3 core bet) | U | Only throughput measured (seq 1.8× faster, step 5× slower, cache 76× larger). Phase D exists precisely for this |
| 12 | Hard mixture models real transition modes (spec hyp. 4) | U | `mixture_control.py` (synthetic, well-designed, MoP-faithful controls incl. codebook arm) built but **not yet run**; Crafter-level benefit doubly untested — HANDOFF §9 rightly doubts Crafter is stochastic enough |
| 13 | Reward/continuation heads calibrated | U | Timing unit-tested; Phase E calibration not run |
| 14 | Reliability signals predict rollout error (spec hyp. 8) | U | Shadow-only (enforced by test). Prior-project evidence (Δ, u_s flat) argues skepticism; Phase F must show correlation/AUROC on held-out true errors before any weighting |
| 15 | Actor/critic imagination graph isolation, λ-returns | T | `test_actor_critic_update_does_not_backpropagate_into_world_model`; smoke test |
| 16 | Imagination is *useful* for policy learning | U | Phase G; see gate additions below — the old project's fatal gap |

Sources secured this turn: VICReg (`third_party/papers/2105.04906v3-vicreg.pdf`),
LeJEPA/SIGReg (`2511.08544v2-lejepa.pdf`, repo pinned in SOURCES.lock).
LeCun AMI v0.9.2 could **not** be re-archived (OpenReview bot-gate); the previous
file was an HTML error page and was removed — re-download manually.

## 2. Are the tests rigorous?

The 24 unit tests are genuinely contract-pinning (leakage, off-by-one, EMA
coverage, stale caches, backend equivalence, shadow-only reliability,
anti-collapse formula/gradient-path/defaults). Gaps worth closing, none urgent:

- No test that imagination-time inputs match training-time input distribution
  (the open masked-vs-generated mismatch); once the masking decision lands, add
  a distribution-consistency test.
- No seeded end-to-end determinism test (would catch silent nondeterminism in
  future refactors).
- Nothing exercises `resets` mid-sequence for the Mamba path (currently raises
  NotImplementedError by design; test the raise so it can't silently regress).

## 3. Are the HANDOFF gates rigorous? (the user's key question)

Verdict: the gate *structure* (A–G ordering, controls-first, shadow-only, no-go
defaults) is sound and already caught a real failure. The gate *metrics* were the
weak point — Phase B failed in a way that produced three artifact numbers out of
four. Specific problems and required amendments:

1. **No pre-registered pass/fail criteria.** Gates say "report X" without
   thresholds, which invites motivated reading in both directions. Amend: before
   each phase run, write the pass criteria into the phase's report header.
   Proposed Phase B criteria (relative, never absolute):
   - covariance effective rank ≥ untrained baseline throughout training;
   - `improvement_over_copy_changed_tokens` > 0 and growing with budget;
   - all probes sane on the untrained row (`semantic_probe_sane` true);
   - trained encoder ≥ untrained on inventory R² and semantic accuracy.
2. **Baselines must be structural, not optional.** Every metric now gets an
   untrained-encoder row and an unrelated-pair scale ruler (implemented).
   Rationale: copy-latent ≈ 0.01–0.02 and semantic ≈ 0.93 are achievable with
   zero learning; absolute numbers were misleading twice.
3. **Probe validity is itself testable.** A converged probe below train-majority
   = misaligned labels (this exact failure shipped in the harness). The sanity
   flag is now computed; make it a hard gate abort.
4. **Gate the objective you will actually train.** Phase B isolates the
   representation; the spec trains jointly with task heads. Keep the isolation
   diagnostic but add the full-objective rank curve (my
   `full_objective_rank` protocol) as the binding check.
5. **Phase C needs a "does Crafter need it" arm.** Before crediting the mixture,
   measure real multimodality: for matched (context, action) pairs from replay,
   the dispersion of true successors vs the noise floor. If Crafter transitions
   are effectively deterministic at this resolution, the mixture is dead weight
   regardless of how well it passes synthetic controls (HANDOFF §9 already
   suspects this).
6. **Phase D must carry the old project's hard-won gate:** multi-step open-loop
   imagination must beat copy-the-current-latent by a pre-stated margin on
   changed tokens over horizons 1–16 *before* any policy training. The previous
   project trained policies for ~15h inside a world model that never cleared
   this bar. HANDOFF's "held-out one/multi-step latent error" does not name the
   copy baseline — add it explicitly.
7. **Phase F thresholds in advance:** reliability weighting stays off unless
   Spearman ρ and top-decile AUROC on held-out, later-checkpoint rollouts clear
   pre-stated values; report calibration curves either way.
8. **Cross-cutting:** every verification run persists a JSON/markdown artifact
   (now in `reviews/artifacts/`); rank/variance logged during any SSL training
   with an abort threshold; smallest-experiment rule enforced by running the
   50-step pilot version of any gate first.

## 4. Immediate next steps (for consensus)

1. Phase B re-run at a real budget (≥ 3–5k updates, both mask settings, new
   metrics, criteria pre-registered per §3.1). This decides masking and confirms
   the anti-collapse fix at scale.
2. Run `mixture_control.py` (synthetic; cheap) + the Crafter-multimodality
   measurement (§3.5) to settle Phase C's premise.
3. Phase D GRU-vs-Mamba-2 with the copy-fidelity bar (§3.6).
4. Consider SIGReg (LeJEPA) as a controlled alternative to the VICReg terms —
   single hyperparameter, official code pinned — only after Phase B passes with
   VICReg (one change at a time).
