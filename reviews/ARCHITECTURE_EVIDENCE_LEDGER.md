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
| Frozen-encoder two-stage contract | V-JEPA-2-AC practice (2506.09985) | Faithful in spirit (freeze after SSL); our encoder is conv, theirs ViT | Fork oracle: frozen space contains branch-discriminative information; all dynamics gates passed under freeze. NO freeze-vs-finetune comparison exists | "Frozen space contains branch-discriminative signal and enables controlled attribution" — NOT "freezing is performance-optimal", NOT Markov/reward/control sufficiency; step-1 certificate VOID if encoder unfreezes |
| Dense tokens + per-token additive action | SPR transition model (2007.05929 Sec 2.3 + pinned source: one-hot action CONCATENATED at every location of a dense 3x3 conv transition); V-JEPA-2-AC action tokens | Ours adds a learned action embedding additively and keeps a separate attention predictor — SPR precedents dense spatial action conditioning, NOT our additive/action-token implementation | Stage A: strong action RESPONSE, weak action IDENTITY; Stage B: identity signal appears with budget. Current action also enters FuturePredictor as a direct attention token | "Dense spatial action-conditioned transitions have precedent (SPR)"; our exact conditioning is a labelled adaptation. NOT "sufficient action conditioning" — a stronger-conditioning ablation (BYOL-AC-motivated) is a registered candidate, not a diagnosis |
| Per-stream independent recurrence (66 GRUs) | none | Matches NO cited system (implementation convenience, caught late). NOTE: recurrent CACHES are independent, but FuturePredictor attention still mixes positions at every prediction — "no spatial information flow" is FALSE | Consolidation corrected: 2/3 seeds pass all gates; ~1.1pt below global-64 (CI [-0.22,+2.39], p~0.07, crosses zero) | "Uncited but viable ablation, operationally behind global-64 in the present screen; superiority (and harm) unresolved" |
| Global pooled recurrent memory (pool->GRU/Mamba->broadcast, dense residual bypass) | Loose: DRAMA single-global-vector (mixer_seq_simple.py:188); DreamerV3/CDP RSSM; LeWM global embedding | No cited system mean-pools dense JEPA tokens and broadcasts state back; dense u_t bypasses the recurrence | Consolidation corrected: 3/3 seeds all gates; paired topology diff +1.09pt, CI [-0.22,+2.39], p~0.07 | "Dense instantaneous tokens + shared global recurrent memory; operational selection by screen" — NOT "global memory is superior", NOT "literature-standard" |
| GlobalMambaTemporal (official Mamba-2 core in the same topology) | Mamba-2 (2405.21060), pinned repo f577286; DRAMA precedent for Mamba-as-WM-core | Source-inspired only: DRAMA feeds flattened categorical latent+action through a stem; we feed pooled JEPA tokens | Contract tests (seq/step equiv, reset isolation, drop-in fwd); companion-verified independently; warm figures: step 0.584ms vs GRU 0.118ms, cache 0.891 vs 0.012 MiB, seq 0.882 vs 2.00ms | "A compact Mamba-2 comparator in our topology" — step-4 result will license at most "Mamba-2 helps/matches/hurts IN THIS TOPOLOGY at this scale" |
| Two-step rollout bridge | V-JEPA-2-AC Eq. 3-4 (teacher forcing + T=2 rollout) | Close adaptation; K=2 (theirs T=2; SPR uses K=5 with per-step targets + joint RL training) | S3-B' 3/3 seeds both scales: MEASURED effect = better 8-step open-loop changed-patch margin than rollout-off; it did NOT by itself beat copy. Shuffled control shows copy-margin can pass without action semantics | "Rollout training improves the open-loop changed-patch margin over rollout-off (3/3 seeds, both scales)" — NOT "drift suppression validated" in absolute terms, NOT causal-action sufficiency |
| Attention FuturePredictor | I-JEPA ViT predictor; V-JEPA-2-AC causal attention | Two-block single-step spatial predictor; temporal recurrence kept separate (cited systems attend jointly) | Cross-token attention necessity: view shifts move content between positions (probe) | "JEPA-inspired predictor; invariant backed = cross-token attention" |
| Deterministic prediction (no mixture) | Dreamer-CDP (deterministic CDP target); MoP-JEPA (mixtures, deferred) | Mixture backend exists but off | Crafter branch dispersion 0.0075 pooled cosine, far below MoP separation regime; oracle found consequential divergence at many anchors | "Deterministic default is regime-appropriate for random-policy Crafter"; mixtures DEFERRED not rejected (policy shift may change regime) |
| Streamwise VICReg terms | VICReg (2105.04906), axis = observations | Sample axis corrected after false-pass position-codebook bug | Regression tests pin the axis; terms OFF in validated recipe (frozen encoder makes them moot) | "Available diagnostic/regularizer for online-encoder training only" |
| Reward/continuation heads on 2-register pool | DreamerV3 heads | Consume pooled POST-TEMPORAL context; weights 1.0 in every validated dynamics run — they DO backpropagate into the temporal core (companion gradient diagnostic 2026-07-16: temporal-core grad sum 8.41 under reward-only backward; encoder/predictor 0). CORRECTED 2026-07-16 — the previous row ("none yet on temporal path") was FALSE | Gradients verified; predictive quality UNCALIBRATED for imagined deployment | "Reward/continuation already train the temporal core; Phase E = held-out calibration + imagined-rollout validation, NOT first introduction of reward grounding" |
| Causal fork evaluation (same-anchor 4-way, common RNG, canonical env) | Hallucination-in-WM action-sensitivity diagnostics (2606.27326); own construction | Novel protocol implementation (canonicalized Crafter, bit-exact repeats, common-union masks) | Collector end-to-end deterministic (digest regression); synthetic mask-flip regression; shuffled-trained controls statistically consistent with chance | "A controlled counterfactual action-selection protocol for Crafter" — candidate methodological contribution |
| Control-centric objectives (action-conditioning strength, counterfactual InfoNCE, predictor update-ratio) | BYOL-AC (2406.02035); TACO (2306.13229 + pinned source); Tang (2212.03319) | Not implemented. Faithfulness pre-labelled: literal BYOL-AC per-action predictors = ~1.80M extra params (NOT trivial at 240k scale) — the realistic arm is action-modulated conditioning (FiLM/AdaLN) or small per-action heads, "BYOL-AC-motivated"; faithful TACO = batch-matched BxB InfoNCE — same-anchor true-vs-wrong-action negatives are "TACO-inspired counterfactual action ranking", gate-adjacent; Tang's two-timescale result is derived for JOINTLY learned representations — an update-ratio ablation under our frozen encoder is an empirical probe, not a theorem transplant | Literature notes 2026-07-15 + 2026-07-16 corrections; SPR/TACO/DBC sources pinned | Registered FUTURE arms, all post-step-4; source-faithful vs source-inspired variants must be labelled at registration |

| Full-grid / no-bypass temporal family (SPRINT CANDIDATE) | DRAMA-inspired input stem (mixer_seq_simple.py:188 flattened-latent-through-stem invariant only); own construction otherwise | 66x64 tokens -> Linear(4224->256) stem -> 2 recurrent blocks (Mamba-2 d_state=64/headdim=64, or width-261 GRU control) -> LayerNorm -> Linear(256->4224) -> reshape; NO dense bypass; ONE global sequence state, not 66 dense states | Exploratory screen: 6 runs (3 seeds x 2 backends) sep 0.0069-0.0135, all above every pooled baseline (0.0030-0.0050), controls at chance; mechanism screen: no single factor identified (capacity 53%, cache-free full-grid 53%, bypass-restored 34% of the gap; bypass restoration hurts pairwise in both seeds); night 2-5x weaker than day | "Best tested candidate; gain distributed across capacity, mixing, recurrence, and bypass removal" — NOT a confirmed default (fresh-seed confirmation deferred; 2026-07-18 sprint uses it as CANDIDATE with matched GRU control) |
| Mechanism controls (MS-PC / MS-FB / MS-FF) | own construction | pooled+bypass @3.03M; flattened+bypass restored; full-grid explicit-recurrent-CACHE-free control (NOT "no temporal state" — autoregressive token feedback remains; not operator-exact) | reviews/2026-07-17-mechanism-screen-protocol.md OUTCOMES | Screening heuristics only (anchors not fully paired); retained as reusable controls |

## Standing rulings

- 2026-07-16 (companion second audit, adopted): shuffled-control results are
  described as "statistically consistent with chance", never unqualified "at
  chance". The zero-broadcast intervention on trained global checkpoints
  (retrieval falls 1.0-5.7 pts across seeds) shows the models USE the shared
  recurrent channel; it does NOT prove the channel causally improved learning
  (post-hoc ablation = distribution shift; a from-scratch residual-only
  control would be needed for that claim).
- 2026-07-16: the pooled-channel risk statement is "localized HISTORICAL
  information must compress through one pooled 64-dim state; the dense bypass
  preserves current spatial detail but no per-location recurrent memory" —
  action identity does NOT route solely through the pooled vector (previous
  action enters every dense token; current action is a predictor attention
  token).

- 2026-07-14: GlobalGRUTemporal selection = "empirically validated engineering
  choice, not a literature reproduction" (user faithfulness question).
- 2026-07-15: corrected consolidation margin is 3/3 vs 2/3 (not 3/3 vs 1/3);
  "global memory is superior" remains UNLICENSED (CI includes zero).
- 2026-07-15: defaults drift repaired — `LossConfig()` now IS the validated
  recipe; any run intending the rejected joint config must opt in explicitly.
- Two-stage frozen-encoder contract is a REGISTERED DECISION as of
  2026-07-15: resuming encoder updates reopens step-1/inventory/causal gates
  as live gates for the entire run.
