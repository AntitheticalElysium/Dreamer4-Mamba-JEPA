# Architecture evidence ledger

Purpose (2026-07-15, adopted from companion recommendation): one row per
component — where it comes from, how we diverge, what evidence we hold, and
the MAXIMUM claim that evidence licenses. Divergence is not the failure mode
this ledger polices; UNLABELLED divergence and claim inflation are. Update
this file in the same commit as any architectural change.

Legend: claims must never exceed the "max licensed claim" column in reviews,
reports, or paper text. "Operational" = selected for use by our own screens;
carries no literature authority.

| Component | Source(s) | Local divergence | Evidence held | Max licensed claim |
|---|---|---|---|---|
| Masked same-frame SSL pretraining | I-JEPA (2301.08243); SparK masked convs | Same-frame (not video); Crafter-safe masking; conv encoder w/ SparK re-zeroing + visible-only norm stats | Step-1 gates 7/7 incl. semantic + inventory retention; strict-load ckpt; companion-verified | "Faithful I-JEPA-style masked pretraining adapted to Crafter" — NOT full I-JEPA (no ViT-scale, no ImageNet) |
| Global projected SIGReg | LeJEPA (2511.08544) | Statistic + projector faithful; but combined with EMA target, stop-grad, learned predictor — all things LeJEPA removes | Step-1 winner over spatially-scoped variant; anti-collapse held under corrected axis diagnostics | "LeJEPA-inspired SIGReg regularizer inside an I-JEPA hybrid" — NOT "we use LeJEPA" |
| Frozen-encoder two-stage contract | V-JEPA-2-AC practice (2506.09985) | Faithful in spirit (freeze after SSL); our encoder is conv, theirs ViT | Fork oracle: frozen latents contain 85-88% headroom over copy; all dynamics gates passed under freeze | "Two-stage frozen-representation system following V-JEPA-2-AC practice"; step-1 certificate VOID if encoder unfreezes |
| Dense tokens + per-token additive action | SPR transition model (2007.05929 Sec 2.3: one-hot action broadcast to every spatial location of a 64x7x7 conv latent); V-JEPA-2-AC action tokens | Ours adds a learned action embedding additively instead of concat+conv | Stage A: strong action RESPONSE, weak action IDENTITY; Stage B: identity signal appears with budget | "Precedented dense spatial action conditioning (SPR-style)"; NOT "sufficient action conditioning" — BYOL-AC says predictor-side conditioning strength is the lever |
| Per-stream independent recurrence (66 GRUs) | none | Matches NO cited system (implementation convenience, caught late) | Consolidation corrected: 2/3 seeds pass all gates; ~1.1pt below global-64 (CI [-0.22,+2.39], p~0.07) | "Viable ablation, operationally outperformed by global-64" — NOT dead, NOT literature-backed |
| Global pooled recurrent memory (pool->GRU/Mamba->broadcast, dense residual bypass) | Loose: DRAMA single-global-vector (mixer_seq_simple.py:188); DreamerV3/CDP RSSM; LeWM global embedding | No cited system mean-pools dense JEPA tokens and broadcasts state back; dense u_t bypasses the recurrence | Consolidation corrected: 3/3 seeds all gates; paired topology diff +1.09pt, CI [-0.22,+2.39], p~0.07 | "Dense instantaneous tokens + shared global recurrent memory; operational selection by screen" — NOT "global memory is superior", NOT "literature-standard" |
| GlobalMambaTemporal (official Mamba-2 core in the same topology) | Mamba-2 (2405.21060), pinned repo f577286; DRAMA precedent for Mamba-as-WM-core | Source-inspired only: DRAMA feeds flattened categorical latent+action through a stem; we feed pooled JEPA tokens | Contract tests (seq/step equiv, reset isolation, drop-in fwd); companion-verified independently; warm figures: step 0.584ms vs GRU 0.118ms, cache 0.891 vs 0.012 MiB, seq 0.882 vs 2.00ms | "A compact Mamba-2 comparator in our topology" — step-4 result will license at most "Mamba-2 helps/matches/hurts IN THIS TOPOLOGY at this scale" |
| Two-step rollout bridge | V-JEPA-2-AC Eq. 3-4 (teacher forcing + T=2 rollout) | Close adaptation; K=2 (theirs T=2; SPR uses K=5) | S3-B' 3/3 seeds both scales; shuffled control shows copy-margin can pass without action semantics | "V-JEPA-2-AC-style rollout bridge, validated for closed-loop drift suppression" — NOT causal-action sufficiency |
| Attention FuturePredictor | I-JEPA ViT predictor; V-JEPA-2-AC causal attention | Two-block single-step spatial predictor; temporal recurrence kept separate (cited systems attend jointly) | Cross-token attention necessity: view shifts move content between positions (probe) | "JEPA-inspired predictor; invariant backed = cross-token attention" |
| Deterministic prediction (no mixture) | Dreamer-CDP (deterministic CDP target); MoP-JEPA (mixtures, deferred) | Mixture backend exists but off | Crafter branch dispersion 0.0075 pooled cosine, far below MoP separation regime; oracle found consequential divergence at many anchors | "Deterministic default is regime-appropriate for random-policy Crafter"; mixtures DEFERRED not rejected (policy shift may change regime) |
| Streamwise VICReg terms | VICReg (2105.04906), axis = observations | Sample axis corrected after false-pass position-codebook bug | Regression tests pin the axis; terms OFF in validated recipe (frozen encoder makes them moot) | "Available diagnostic/regularizer for online-encoder training only" |
| Reward/continuation heads on 2-register pool | DreamerV3 heads | Consume pooled registers only; UNCALIBRATED (Phase E gated) | None yet on temporal path | NO claim; DeepMDP/DBC say reward grounding is required for control-relevance — Phase E is the test |
| Causal fork evaluation (same-anchor 4-way, common RNG, canonical env) | Hallucination-in-WM action-sensitivity diagnostics (2606.27326); own construction | Novel protocol implementation (canonicalized Crafter, bit-exact repeats, common-union masks) | Collector end-to-end deterministic (digest regression); synthetic mask-flip regression; shuffled-trained controls at chance | "A controlled counterfactual action-selection protocol for Crafter" — candidate methodological contribution |
| Control-centric objectives (per-action heads, TACO-style InfoNCE, predictor update-ratio) | BYOL-AC (2406.02035); TACO (2306.13229); Tang (2212.03319) | Not implemented | Literature notes 2026-07-15 | Registered FUTURE arms; anything gate-adjacent must be separately labelled |

## Standing rulings

- 2026-07-14: GlobalGRUTemporal selection = "empirically validated engineering
  choice, not a literature reproduction" (user faithfulness question).
- 2026-07-15: corrected consolidation margin is 3/3 vs 2/3 (not 3/3 vs 1/3);
  "global memory is superior" remains UNLICENSED (CI includes zero).
- 2026-07-15: defaults drift repaired — `LossConfig()` now IS the validated
  recipe; any run intending the rejected joint config must opt in explicitly.
- Two-stage frozen-encoder contract is a REGISTERED DECISION as of
  2026-07-15: resuming encoder updates reopens step-1/inventory/causal gates
  as live gates for the entire run.
