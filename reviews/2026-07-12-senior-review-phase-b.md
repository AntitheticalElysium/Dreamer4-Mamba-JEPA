# Senior review — implementation-agent turn 1 (source audit → Phase B failure)

Reviewer: oversight agent. Scope: everything the implementation agent did up to and
including the 300-update Phase B representation control. I modified nothing in
`m3_hjwm_compact/`; all probe scripts live in the session scratchpad and their
results are reproduced below.

## Verdict in one paragraph

The agent's engineering work is solid and its claims reproduce (18/18 tests pass,
diffs match its report, the Mamba-3 findings are verified against the pinned
source). Its bottom-line decision — do **not** advance this model to temporal or
policy training — is correct. However, its *diagnosis* of the Phase B failure is
partly wrong, and two of the four headline metrics come from a **buggy or
structurally invalid control**, not from the model. Most importantly, the failure
was predictable from the primary sources before spending the 40-minute run: the
compact model's "JEPA" objective does not match any verified JEPA source and has
no anti-collapse mechanism. The next step must be fixing the objective and the
controls, not abandoning the representation direction.

## What I verified and confirmed (agent was right)

1. **Edits are behavior-scoped and match the report.** Reconstructed the exact
   diff against `m3_hjwm_compact.zip` (base was never committed). Changes:
   explicit backend selection (no silent GRU fallback), true BOS action index,
   differentiable soft balance surrogate (clearly labelled as a paper deviation),
   action+horizon-conditioned router, official Mamba-2/3 `step()` wiring with
   in-place cache clone/detach in `imagine()`, stale-state revision guard,
   `info["discount"]`-based continuation, reliability auxiliaries zero-weighted
   (shadow-only). All defensible; each has a regression test.
2. **Tests reproduce:** `.venv/bin/python -m pytest m3_hjwm_compact/tests/` →
   18 passed. Smoke tests pass.
3. **Mamba-3 status verified in the pinned repo:**
   `mamba_ssm/modules/mamba3.py:319` says "NOTE: Only tested on H100…". The
   headdim/stride failure and the `apache-tvm-ffi` pin the agent found are
   consistent with what I see in the source; Mamba-2 as research backend with a
   GRU control is the right call.
4. **Transition timing** (a_t → context_{t+1} → reward_{t+1}) matches the
   Dreamer-CDP contract; `data.py` `previous_actions` indexing is correct;
   replay slices never cross episode boundaries, so the Mamba sequence path's
   `NotImplementedError` on mid-sequence resets is never hit by `sample()`.

## Where the agent's conclusions do not hold

### F1 (critical): the "JEPA" objective is not JEPA, and the collapse was predictable

Verified against the pinned I-JEPA source
(`third_party/sources/facebookresearch__ijepa/src/train.py`, `forward_target` /
`forward_context`):

- I-JEPA computes the loss **only at masked target positions**
  (`apply_masks(h, masks_pred)`), from a context encoder that sees only context
  patches, with a predictor conditioned on target-position queries.
- I-JEPA **layer-normalizes the target features** before the loss.
- V-JEPA 2-AC (per the agent's own audit) trains the action-conditioned
  predictor on a **frozen** target encoder.

The compact model does none of this. Its only representation-shaping loss in
Phase B is: cosine distance between predictor(masked-online(obs_t), a_t) and
EMA-target(obs_{t+1}), averaged over **all 66 tokens** (registers + unmasked
positions included), with unnormalized targets and a trainable encoder. That is
a temporal BYOL variant with input corruption — a configuration with no
verified-source precedent and no anti-collapse mechanism (no masked-position
restriction, no target LN, no variance/covariance regularization, no frozen
encoder, and — in the Phase B protocol — no task-gradient grounding à la
Dreamer-CDP). Representation collapse is the *expected* optimum of this
objective. Per the project's cardinal rule (read sources before implementing),
this should have been caught in the static audit, before the 40-minute run.

### F2 (critical): the Phase B semantic probe is structurally invalid

Crafter's `info["semantic"]` is `SemanticView.__call__` → a copy of
`world._mat_map`: the **global 64×64 world map**, not the local egocentric view
rendered in the observation (`.venv/.../crafter/engine.py:251`). The control's
`semantic_probe` nearest-downsamples this global map to 8×8 and pairs it with
the 8×8 tokens of the *local view*. The two are spatially unrelated, so "probes
do not beat the majority baseline" is guaranteed by construction and says
nothing about the encoder.

Empirical confirmation (scratchpad `probe_convergence_check.py`, random
untrained encoder, held-out seed 2): the probe **converges** (train CE → 0.68)
yet test accuracy stays at 0.33 vs 0.52 majority from step 100 through step
3000. A converged probe scoring below majority is only possible when labels are
misaligned with features. The inventory probe, by contrast, is structurally fine
(inventory is rendered in the HUD strip of the observation).

### F3 (major): copy-latent and shuffled-target numbers were misread

Random-encoder baseline on the same held-out seed (scratchpad
`random_encoder_probe.py`):

| metric (untrained encoder) | value |
|---|---|
| dense copy-latent cosine (t vs t+1) | **0.022** |
| dense unrelated-pair cosine | **0.091** |
| pooled copy-latent cosine | 0.004 |
| effective rank (singular, matches `effective_rank`) | **45.3 / 64** |
| semantic probe / majority | 0.33 / 0.52 |

Consequences:

- **Copy-latent ≈ 0.01–0.02 is intrinsic to Crafter** (consecutive frames are
  nearly identical), not a property of the trained model. As scored — mean
  cosine over all tokens — the copy baseline is a near-unbeatable bar; a gate
  defined this way will fail almost any one-step predictor. The gate metric
  needs redesign (see recommendations), though the trained predictor's 0.44
  error is still damning: it exceeds even the unrelated-pair distance (0.09),
  i.e., the predictor output lies off the encoder manifold.
- **"Shuffled targets beat true targets" is vacuous under collapse.** With
  effective rank ~3, all targets are nearly collinear; pred-target distances are
  equal up to noise, so the ordering carries no information. It is a symptom of
  collapse, not an independent finding about prediction quality.
- **The one robust Phase B finding is the rank trajectory: 45 → ~3 in 300
  updates.** Training actively destroyed dimensionality. That, plus F1, is the
  whole story; the other three headline metrics are artifacts.

### F4 (moderate): run discipline

- Phase B results were only printed to stdout — nothing persisted to disk. Every
  verification run must write a JSON/markdown artifact.
- The 40-minute run violated "smallest discriminating experiment": logging
  effective rank every ~20 steps would have revealed the collapse within a
  minute or two and allowed an early abort. The 10-step pilot's copy-latent
  anomaly (0.01 vs 0.44) already contained F3's signal and warranted checking a
  random-encoder baseline first.

## Still-open items the agent correctly flagged but has not resolved

- Train/deploy distribution mismatch: temporal model trained on masked online
  tokens, deployed on dense generated target-like tokens (its own
  high-severity finding; unaddressed).
- Mixture (MoP) controls: synthetic branching control announced but not yet run.
  Fine to run, but pointless to gate on until the representation objective is
  fixed — mixture quality is unmeasurable in a collapsed space.

## Recommendations (need consensus before any code change in `m3_hjwm_compact/`)

1. **Fix the representation objective against verified sources** (pick one,
   cite the source in the commit):
   a. True I-JEPA same-frame masked prediction: loss at masked positions only,
      layer-normalized targets, context encoder sees only visible tokens —
      as the primary SSL loss; temporal prediction becomes a separate predictor.
   b. V-JEPA-2-AC-style two-stage: pretrain encoder with (a), then freeze it
      and train the action-conditioned temporal predictor on top.
   c. Dreamer-CDP-style grounding: accept that reward/continuation/value
      gradients shape the encoder and change the Phase B protocol accordingly
      (this contradicts HANDOFF's "representation only" gate — needs an explicit
      HANDOFF amendment, not a silent change).
   The 2026-07-09 milestone in the previous project used real CNN-JEPA-style
   masked SSL and worked; option (a)/(b) is closest to that evidence.
2. **Fix the controls before re-running Phase B:**
   - semantic probe: derive local-view labels (crop `info["semantic"]` around
     `info["player_pos"]` to the rendered view, excluding HUD rows), or drop it
     and rely on the inventory/health probes;
   - report prediction error as improvement-over-copy, and separately on the
     tokens that actually changed between frames;
   - always report the unrelated-pair distance as a scale ruler and the
     effective-rank trajectory during training;
   - sanity-check every probe on an untrained encoder first (it must sit at
     chance/majority, and a converged probe must never be below majority).
3. **Process:** commit the current state immediately (base + agent edits as two
   commits if possible), persist all run outputs, log rank/variance during any
   SSL run with an abort threshold.

## Go/no-go positions (my side of the consensus)

- Mamba-3: **no-go** as research backend (H100-only note verified; fragile deps;
  batch-stride failures) — keep the probe results, use Mamba-2. Agree with agent.
- Advancing current model to Phase C/D/RL: **no-go.** Agree, but for reason F1,
  not for the control's headline metrics.
- Predictor mixture / reliability calibration: **defer** until a non-collapsed
  representation exists.
- Re-run of Phase B after objective+control fixes: **go**, with rank logging and
  a random-encoder baseline column in the report.

---

# Addendum (same day) — spec-level verdict from controlled objective variants

Question from the user: does the fault lie in `ARCHITECTURE_SPEC.md` or only in the
compact implementation? Answer: **the collapse is a spec-level property of §3's
objective, not an implementation artifact — but it is the specific, repairable flaw
the spec itself anticipated, not an invalidation of the design.**

## Experiment (scratchpad `objective_variants.py`, `full_objective_rank.py`)

Identical encoder/predictor/data/optimizer (compact modules, deterministic
predictor, seeds fixed, 300 updates, real Crafter random-policy data, held-out
seed 2). Covariance effective rank of EMA-target tokens (untrained reference ≈ 12):

| variant | source shape | cov rank @300 | outcome |
|---|---|---|---|
| base (spec §3: masked ctx, dense loss, raw targets) | none (hybrid) | 3.3 | collapse |
| + layer-normed targets | I-JEPA practice | 3.5 | collapse |
| masked-position-only loss + LN | I-JEPA-faithful loss | 2.7 | collapse |
| no masking, dense | Dreamer-CDP-shaped (no grounding) | 2.8 | collapse |
| **full objective** (JEPA+reward+continue, temporal model, `M3HJWM.forward`) | spec §7 | **4.2, still falling** | collapse |
| + VICReg variance/covariance reg (probe coefficients) | LeCun line (VICReg/SIGReg) | **16–19** | **no collapse** |

Reading: at this scale and with random-policy Crafter data, EMA asymmetry does not
prevent collapse under *any* loss shape; task-head gradients only slow it (rewards
are too sparse to ground the representation); explicit variance regularization
categorically prevents it. Masking made no difference to collapse and has no
verified-source precedent in this hybrid form — it only creates the train/deploy
input mismatch already flagged.

Caveats: 300 updates, tiny data — collapse *rates* may differ at scale; the VICReg
coefficients were probe choices, not a tuned proposal. But the monotone rank decay
of the full objective versus the flat/rising VICReg curve is a clean discriminating
signal.

## Also found while comparing implementations

- **Reference implementation bug (m3_hjwm/m3_hjwm/world_model.py:158-161):** the
  trainable online `spatial` mixer is applied to the target branch *outside*
  `no_grad`, and the target is never detached in the predictor loss — the "EMA
  target" is not actually stop-gradded end-to-end. The compact refactor silently
  fixed this (EMA duplicates the entire encoder incl. spatial). The refactor is
  therefore not behavior-preserving in both directions; the compact version is the
  more correct one here. Reference also uses action 0 (a real Crafter action) as
  the null previous action; compact's BOS index fixed that.
- **Provenance defect:** `third_party/papers/lecun-ami-v0.9.2.pdf` is an HTML error
  page, not the paper (curl saved an OpenReview error page). Re-download and verify
  with `file`/`pdftotext` before citing.

## Spec verdict, section by section

Sound and kept: §2 timing (verified against Dreamer-CDP source), §5 temporal
contract (amend target backend to Mamba-2 per evidence), §6 mixture-with-controls,
§9 actor/critic, §10 shadow-only reliability, §11 test list, §12 non-claims.

Must change (spec v2 amendments to agree on):

1. **§3: anti-collapse becomes a default, not a contingency.** "Not enabled until
   diagnostics justify them" is now empirically wrong — diagnostics justify them
   in every tested configuration. Adopt a cited mechanism (VICReg/SIGReg terms per
   LeCun line, coefficients from the source paper) and keep rank/variance logging
   with an abort threshold as the gate.
2. **§3: resolve the masking hybrid.** Either full I-JEPA structure (loss at
   masked positions, LN targets — though this alone demonstrably does not prevent
   collapse here) or no masking (Dreamer-CDP shape). Decide by controlled
   comparison *after* anti-collapse is in; the current hybrid has no source and no
   measured benefit.
3. **§7/HANDOFF Phase B: make the gate match the objective.** Phase B currently
   tests the representation in a regime (no task gradients) the spec never runs;
   with the full objective collapsing too this distinction currently doesn't bite,
   but the gate metrics must change regardless: improvement-over-copy on changed
   tokens, rank trajectory, untrained-encoder baseline column, and probes fixed per
   the main review.
