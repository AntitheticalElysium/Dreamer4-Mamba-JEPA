# Validation run results: Phase C premise/mechanics, Phase B long, Phase D backends

All criteria were pre-registered in the protocols before launch
(`reviews/artifacts/phase_b_long.py`, `phase_d_backend.py`). Raw artifacts in
`reviews/artifacts/`. Data: random-policy Crafter, seeds 0+1 train (9.4k
transitions), seed 2 held-out; 4000 updates; single seed per arm (screening
scale, not final claims).

## Phase C — mixture premise and mechanics

- **Premise (Crafter multimodality): NOT SUPPORTED.** Forking live env states
  into 8 reseeded-RNG branches and stepping the same action: 53% of transitions
  diverge somewhere in view, but by only ~2 tiles of 63 (~0.6% of pixels);
  consequential divergence (inventory/health) 3–5%. Creature jitter, not
  discrete branching. (First probe run showed zero divergence everywhere — an
  RNG-aliasing bug in the probe itself; verify every zero.)
- **Mechanics (synthetic, 3 seeds): VALIDATED.** K=2 hard mixture: best-of-K
  0.0035 vs 0.181 deterministic regression-to-mean; precision 1.0; coverage
  0.81; router total-variation 0.15 against true 0.2/0.8 branch probabilities;
  input-agnostic codebook control correctly fails everything. K=4 with the
  balance term costs precision (0.77) when true modes = 2.
- **Decision proposed: deterministic predictor stays the Crafter default; Phase
  C leaves the critical path.** The mixture is validated machinery for a future
  genuinely-branching environment.

## Phase B long run (both mask arms, 4000 updates)

| criterion | masked 0.6 | unmasked |
|---|---|---|
| P1 rank never below untrained | **PASS** (12.5 → 30.1, rising) | **PASS** (12.5 → 42.3, rising) |
| P2 improvement over copy, changed tokens | FAIL (pred 0.242 vs copy 0.019) | FAIL, near-miss (pred 0.047 vs copy 0.028; untrained gap was −1.09) |
| P3 semantic probe not degraded | FAIL (0.892 → 0.838) | FAIL borderline (0.892 → 0.864, tolerance 0.02) |
| P4 inventory R² not degraded | **PASS** (−0.46 → **+0.34**) | PASS (−0.46 → −0.18) |

Readings:

1. **The anti-collapse fix is confirmed at 13× the budget that motivated it** —
   rank grows monotonically after warmup in both arms; variance loss goes to
   ~0 (the hinge is satisfied, not fighting the objective).
2. **Unmasked wins prediction decisively** (JEPA 0.030 vs 0.215 final; P2 gap
   −0.019 vs −0.223) and was pre-registered as the Phase D setting. Masked wins
   inventory retention (P4: +0.34 vs −0.18) — masking forces registers/context
   to carry HUD state. Worth revisiting masking *for the registers' sake* only
   after P2 passes.
3. P3's small semantic-accuracy decline (still ≫ majority, probe sane) says the
   encoder trades a little tile-identity legibility for predictive structure —
   watch, don't panic.
4. Peak VRAM 181 MiB; 4000 updates in 1.6–4.6 min. Budget is not the current
   bottleneck; data scale and architecture are.

## Phase D — temporal backends + copy-fidelity bar (unmasked, matched budget)

| k (changed tokens) | GRU pred/copy | Mamba-2 pred/copy |
|---|---|---|
| 1 | 0.060 / 0.045 | 0.040 / 0.030 |
| 4 | 0.149 / 0.083 | 0.099 / 0.054 |
| 8 | 0.311 / 0.146 | 0.171 / 0.091 |
| 16 | 0.550 / 0.205 | 0.280 / 0.123 |

- **D1 copy-fidelity bar: FAIL for both backends at every k.** Imagination
  error compounds ~2× faster than the world drifts. This is the old project's
  fatal condition, now measured *before* policy training instead of after 15
  hours of it. **No-go on Phase G (policy) stands.**
- **D2 Mamba-2 vs GRU: first genuine positive evidence for the Mamba bet.**
  Lower open-loop error at every horizon (both absolute and relative to its own
  copy baseline: k=16 ratio 2.27 vs 2.69), **7.5× faster recurrent imagination
  step** (2.0 ms vs 14.9 ms at batch 48×66 streams), trains in 1.7 min at
  578 MiB peak. One seed, screening scale — but direction and margin are clear.
  Recommend Mamba-2 as the default backend going forward, GRU retained as the
  control.

## Root-cause hypothesis for the failed fidelity bar (source-grounded, testable)

The compact `FuturePredictor`'s `PredictorBlock` is a **per-token MLP**
(`nn.Linear` over the feature dim only): token (i,j)'s future is computed from
that token plus action/horizon embeddings, with **no cross-token pathway** in
the predictor. I-JEPA's predictor is a full ViT with self-attention across
tokens (`VisionTransformerPredictor`). In Crafter, the dominant transition is a
whole-view shift when the player moves — content moves *between* token
positions, which a per-token MLP cannot express; the only spatial mixing in the
entire prediction path is the encoder's single attention layer. This predicts
precisely the observed failure signature: error concentrated on changed tokens,
prediction unable to beat copy.

**Proposed next lever (needs consensus + implementation agent):** add spatial
self-attention to the predictor (1–2 blocks, I-JEPA-shaped), then re-run Phase
B/D under identical protocol. Secondary levers if needed, in order: more data
(9.4k random transitions is tiny), then a short multi-step rollout loss
(V-JEPA-2-AC recipe). One change at a time.

## Gate status after this run

- Phase C: resolved (premise no, mechanics yes) — off the critical path.
- Phase B: P1/P4 pass, P2/P3 fail → representation objective healthy, predictor
  architecture is the suspect.
- Phase D: D1 fail (blocks Phase G), D2 answered (Mamba-2).
- Phases E/F/G: remain blocked pending D1.
