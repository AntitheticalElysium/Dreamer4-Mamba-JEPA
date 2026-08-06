# Exploratory protocol: no-bypass topology + conditioning strength screen

Status: **EXPLORATORY, pre-registered before bundle collection or training.**
Run under the user's standing exploration authorization (2026-07-17, user
absent) — dual consensus is NOT claimed; nothing here changes defaults,
gates, or the operational GRU-64 selection. Outcomes license at most a
registered confirmation on fresh seeds.

## Motivation (from the 4b post-mortem + literature round)

The 4b screen's most decision-relevant negative was backend-neutral: scaling
the POOLED adapter (with its dense residual bypass) improved neither
backend's final-horizon counterfactual separation. Both external reviews and
the ledger flag the pooled bottleneck + bypass as the project's largest
unlabelled-then-labelled divergence from every cited system, and the 4b
review explicitly notes "the result weighs against 'just make the pooled
Mamba bigger', not against a source-shaped global state". The user's
standing skepticism ("testing backends in isolation doesn't reflect scaled
performance") has exactly one cheap testable form at our scale:

- **H-T**: the topology (pooled bottleneck + dense bypass), not the backend,
  limits action-discriminative dynamics. Screen a DRAMA/RSSM-SHAPED
  flattened-latent core where the recurrent state CARRIES the entire context
  (no bypass), with GRU and Mamba-2 inside the identical adapter.
- **H-C**: predictor action-conditioning strength is the first literature-
  backed lever (BYOL-AC; LeWM AdaLN-zero). Screen AdaLN-zero modulation
  added on top of the existing token conditioning.

## Arms (validated step-4 training contract: T=16, B=4, 16k updates, frozen
## encoder, frozen_dynamics_recipe, AdamW 1e-4, clip 100, bf16)

| arm | temporal core | predictor | seeds |
|---|---|---|---|
| X-FLG | FlattenedGRUTemporal (width matched to X-FLM pre-outcome, depth 2) | standard | 505, 606 |
| X-FLM | FlattenedMambaTemporal (256 / depth 2 / d_state 64 / headdim 64) | standard | 505, 606 |
| X-ADA | GlobalGRU-64 (operational) | AdaLNFuturePredictor (LeWM ConditionalBlock module.py:88, zero-init gates) | 505, 606 |
| X-FLG-shuf | as X-FLG | standard | 505 (shuffled actions) |
| X-ADA-shuf | as X-ADA | AdaLN | 505 (shuffled actions) |

Baselines at zero training cost: the twelve committed step-4 16k checkpoints
(M1/M2 real, M3 shuffled) re-evaluated on this screen's fresh bundle — the
identical training contract makes them the topology/conditioning control.

Pairing: per-seed shared reference (global-gru base construction, as in
step 4); FL arms copy every non-temporal state entry; X-ADA copies every
entry outside `future.` (its predictor is structurally different — recorded,
not digest-matched). Replay stream digests must match across arms per seed.
Source-faithfulness labels: FlattenedGRUTemporal / FlattenedMambaTemporal are
"DRAMA/RSSM-shaped, source-inspired" (continuous JEPA tokens, per-token
upstream action embedding, AdamW — all labelled divergences).
AdaLNFuturePredictor is "LeWM-inspired" (modulation added on top of token
conditioning, not replacing it).

## Monitor bundle

Environment seeds 131-134 (fresh: never in replay, monitors, or final sets;
115-130 stay reserved for a potential 4b confirmation). 4 day / 2 night
anchors per seed = 24 anchors, 4 suffixes x 3 common-RNG branches, canonical
collector (live env canonicalized) with verify_repeat, hash-pinned in a
clean commit before training. These seeds are SPENT for selection after this
screen.

## Metrics

`symmetric_eval` (all-token / patch / common-mask changed 4x4 matrices),
retrieval + separation with env-seed-clustered summaries, day/night +
pixel/task-effective strata. n=24 anchors and 2 training seeds: screening
resolution only; no CIs are decision-bearing.

## Pre-registered screening readouts (computed by the runner, not by hand)

- **H-T interesting** iff mean X-FL retrieval_all (4 runs) >= step-4
  global-64 baseline mean on the SAME bundle AND separation_all > 0 in all
  four FL runs AND FL mean >= X-FLG-shuf + 1.5 pts.
- **H-C interesting** iff mean X-ADA retrieval_all >= baseline + 1.0 pt AND
  both seeds individually above baseline AND >= X-ADA-shuf + 1.5 pts.
- The FL-M vs FL-G contrast within the flattened topology is recorded as a
  diagnostic (the "topology unlocks the backend" sub-hypothesis) — two seeds
  cannot decide it.
- Failure modes are informative either way: if the no-bypass state cannot
  even match the bypass topology at equal budget, the bypass is not the
  bottleneck at this scale, and the topology-skepticism answer changes.

## What this screen cannot do

No superiority claims, no default changes, no gate changes, no reuse of
seeds 79-94 or 115-130, no policy/reliability work. Anything "interesting"
goes to the companion + user for a registered three-seed confirmation with
fresh final seeds.

## OUTCOMES (appended 2026-07-17 after the run; registered readout committed
## at cb27d20, extension at 956ee02, corrections per companion audit)

- H-T: PASS mechanically. FL retrieval 29.17/37.50 (G), 30.21/35.42 (M) vs
  pooled baseline 31.94%; ALL FL separations positive and 2-3x every pooled
  baseline (0.00693-0.01257 vs 0.00304-0.00499); FL-G shuffled control
  26.04% / sep 0.000698. Extension seed 707: FL-G 36.46% / 0.01351,
  FL-M 33.33% / 0.01269; FL-M shuffled control 25.00% / 0.00071.
- H-C: FAIL mechanically (AdaLN mean +0.35 pts vs required +1.0; seed 505
  below baseline). Not a general rejection of action modulation.
- LICENSED CONCLUSION (companion correction adopted): "a large full-grid,
  no-bypass JEPA adapter is a promising architecture family with stronger
  action-discriminative open-loop behavior than the pooled controls." NOT
  licensed: "pooling/bypass proven causal" — the arm moved capacity,
  flattening, projections, mixing, and bypass together. Factor isolation:
  reviews/2026-07-17-mechanism-screen-protocol.md.
- Across 3 seeds: FL-G 34.38% / sep 0.01096, FL-M 32.99% / sep 0.01118 —
  backend PARITY again (mixed per-seed signs); Mamba trains ~1.43x faster
  end-to-end and fits ~18-21% lower teacher-forced JEPA but its matched
  open-loop error crosses over around k=3-4 and is worse at k=8 in all
  three seeds (companion diagnostic; cache mismatch refuted) — motivates a
  controlled K=2-vs-K=5 per-step-target test, NOT a blind K increase.
- Strata (post-hoc summaries added to the report JSON): night separation is
  2-5x weaker than day in every FL arm (e.g. 0.0023-0.0056 vs 0.0083-0.0179)
  though still above pooled controls. Tie-aware retrieval (preregistered
  primary for any confirmation): FL mean ~33.6% vs pooled baseline 31.5% —
  direction robust to tie policy.
- Corrections adopted: t(n-1) critical values fixed in seed_level_summary
  (FL-G per-env CIs now include zero at n=4; FL-M extension CIs remain
  positive; no registered decision used these CIs); "RSSM-shaped" and
  "LeWM-faithful" relabelled; xtopo checkpoints + monitor bundles force-
  added for artifact retention; seeds 115-130 REMAIN RESERVED; fresh-seed
  confirmation is ON HOLD pending the mechanism screen.
