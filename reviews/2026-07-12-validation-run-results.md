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

---

# Addendum: MoP-JEPA applicability (numbers) and predictor ground-truth alignment

## MoP-JEPA read in full (2607.05238) — why it does not transfer to Crafter

The paper is internally sound: for stochastic branchings, a regression-optimal
single predictor outputs the conditional mean of successor embeddings (Prop. 1),
and hard-assigned heads converge to a quantizer of the transition distribution
(Prop. 3) — *under a well-separated-modes assumption*. Its testbeds are OGBench
teleport mazes, where a successor "mode" is a distant maze cell: mode separation
is on the order of the distance between unrelated states.

Measured in the paper's own currency (pooled L2-normalized latents, cosine
distance, trained Phase B encoder; `reviews/artifacts/crafter_branch_latents_*`):

| quantity (pooled cosine) | value |
|---|---|
| Crafter branch dispersion, same action, 8 RNG branches (mean) | **0.0075** |
| … p90 / max over 60 probe states | 0.011 / 0.277 |
| one-step copy distance (ordinary world drift) | 0.0141 |
| unrelated-state distance (MoP's mode-separation scale) | **0.284** |

Crafter's typical stochastic spread is **2.6%** of the separation scale MoP's
propositions assume, and **half** the deterministic one-step motion the
predictor must model anyway. Hard assignment (argmin over head distances) would
be decided at the 0.0075 scale while training error lives at 0.03–0.05: no
learnable mode structure, except for ~1–2% of transitions (the 0.277 outlier —
consequential combat/health branches exist but are rare). Conclusion: MoP-JEPA
motivates the mixture *interface*, not its use on Crafter. Two port deviations
also noted for the record: the paper mixes over a **pooled** latent (compact
port mixed over 66 dense tokens) and L2-normalizes latents before prediction.

## Predictor brought back to JEPA ground truth

Checked both pinned implementations: I-JEPA's predictor is a ViT
(self-attention across tokens + positional embeddings); V-JEPA-2-AC's
(`ac_predictor.py`) prepends learned action/state tokens to the frame-token
sequence and attends jointly. The compact predictor was a per-token MLP — an
undocumented divergence from every source, and the measured root cause of the
failed copy-fidelity bar.

Change (commit below): `FuturePredictor` now uses self-attention blocks over
[action token, horizon token, context tokens + learned positional embeddings],
V-JEPA-2-AC-shaped conditioning, LayerNorm + projection head (I-JEPA shape).
Deviations from source, justified: depth 2 and no separate predictor width
(model is 64-dim; I-JEPA's narrowing/deepening is a large-scale concern).

Test-first: `test_predictor.py` written before the change; cross-token flow and
position sensitivity failed on the per-token MLP, pass after (28/28 suite).
Test-harness lesson recorded: a constant perturbation sits in LayerNorm's null
space — perturb with random directions when testing information flow.

Phase B/D re-run under the identical pre-registered protocol: in progress.

## Re-run with the attention predictor (identical protocol, commit b9c5e80)

**Phase B (one-step, teacher-forced):** the fix does what the root-cause
analysis predicted, but P2 is still short:

| changed-token gap (copy − pred) | per-token MLP | attention predictor |
|---|---|---|
| masked | −0.223 | **−0.030** (7× closer) |
| unmasked | −0.019 | **−0.0072** (pred 0.032 vs copy 0.025) |

P1 passes (rank 29/43, rising). P3 unchanged-borderline (0.85–0.86 vs 0.89).
P4 flips against unmasked (inventory R² −1.15; masked holds +0.29): with
attention, the unmasked objective invests registers elsewhere. The masked arm
is no longer prediction-crippled (0.080 vs copy 0.050), so the masking decision
is genuinely open again.

**Phase D (multi-step, open-loop): new dominant failure mode exposed.**
D1 still fails for both backends — and the earlier "Mamba-2 wins D2" result
**flips at multi-step**: with the attention predictor, Mamba-2 diverges in
closed loop (changed-token error 0.027 → 0.21 → 0.52 → 0.74 at k=1,2,4,8)
while GRU now degrades gracefully (0.038 → 0.27 at k=8; better than its
previous 0.31, and 0.28 at k=16 vs 0.55 before). Reading: training is purely
teacher-forced; feeding generated tokens back is out-of-distribution, and the
SSM accumulates the drift where GRU's saturating gates bound it. Both D2
verdicts (previous pro-Mamba, current pro-GRU) are single-seed screening
results conditional on the predictor — the backend question is REOPENED, not
answered.

**Where this leaves the critical path.** One-step prediction is nearly at the
copy bar; the binding failure is now closed-loop compounding — exactly what
V-JEPA-2-AC's training recipe addresses with short multi-step rollout losses
(teacher forcing + k-step rollout). Proposed next single change for the
implementation agent: add a 2–4-step closed-loop rollout term to the world
objective (V-JEPA-2-AC recipe, pinned source), re-run this identical protocol,
and only then revisit backend choice and data scale.

## Rollout-loss experiment (option 3): D1 crossed for the first time

Protocol and criteria pre-registered in `reviews/artifacts/rollout_loss_experiment.py`
(V-JEPA-2-AC Eq. 3-4: unweighted `L_forward + L_rollout`, T=2, final-step cosine,
gradient through the predictor→temporal→predictor composition via re-run
parallel scans). Unmasked, 4000 updates, same data/seed as all prior runs.

| open-loop, changed tokens | GRU pred/copy | Mamba-2 pred/copy |
|---|---|---|
| k=4 | 0.112 / 0.114 **← beats copy** | 0.068 / 0.063 |
| k=8 | 0.174 / 0.201 **← beats copy** | 0.116 / 0.122 **← beats copy** |
| k=16 | 0.227 / 0.273 **← beats copy** | 0.165 / 0.181 **← beats copy** |

- **R2 / D1 (beats copy at some k ≤ 8): PASS for both backends** — first time
  any model in this project (old repo included) has beaten the copy baseline in
  open loop. GRU crosses at k=4, Mamba-2 at k=8; both stay better through k=16.
- **Mamba-2: all criteria pass.** R1: 7.6×/6.4× improvement at k=4/8 (0.520→0.068,
  0.740→0.116) — the divergence is fully resolved by the bridge. R3: one-step
  0.0333 vs 0.0319 baseline (+4%).
- **GRU: R1 formally missed** (1.8×/1.5× vs required 2×) and **R3 missed**
  (one-step 0.0486, +52%) — GRU pays for multi-step stability with one-step
  accuracy; Mamba-2 does not.
- **The backend story stabilizes:** without the bridge the SSM amplified
  out-of-distribution feedback (previous run's divergence); with the bridge,
  Mamba-2 beats GRU at every horizon again, plus the standing 7.5× recurrent
  speed advantage. D2 verdict: **Mamba-2 default, conditional on the bridge —
  which is now a validated part of the objective.**
- Cost: +2 parallel-scan passes/update; 3.2-3.4 min per 4k-update run; 722 MiB
  peak. Rollout losses converge (0.54→0.023).

Caveats: single seed, screening scale, random-policy data; k ≤ 2 does not beat
copy (near-unbeatable one-step bar, as analyzed) — the crossover at k=4-8 sits
exactly at the spec's imagination horizon (5-8).

**Consensus package for the implementation-agent handoff:**
1. Implement the rollout term in `m3_hjwm_compact` properly (scratchpad probe
   archived as reference; needs a config flag, tests: gradient-through-
   composition, loss-decreases regression, backend parity).
2. Spec v2 amendments per `2026-07-12-imagination-bridge-analysis.md` §4
   (objective term, backend=mamba2 default, Phase E/F calibrate on imagined
   states, options 1/2 recorded as non-goals).
3. Next gates in order: Phase E (reward/continue calibration on imagined
   states from real prefixes), then 3-seed confirmation of this result, then
   Phase F shadow-reliability calibration. Phase G stays gated on E.
