# Chance-forced self-fed training (Track A): preregistration

2026-10-15 evening. Written before any self-fed training. The user framed the research in two tracks:
- **Track A** understands the current CorrT + Mamba + transport platform, as a controlled baseline, not the final
  architecture.
- **Track B** investigates better architectural principles.

This study is Track A.

## Question

Is the recursive fading of the deterministic dynamics primarily a **training** failure (the model never sees its own
predictions) or a **structural** one (soft persistence)?

## Prior evidence, measured before this document

- **Fading is deterministic.** With the chance outcome supplied from the real future every step, fading is unchanged
  (posterior-forced cow census d16 0.36 vs afterstate-only 0.36; NOTEBOOK R14).
- **Fading is largely structural** (`20261015_whiteboard/persistence_variants.py`, inference only). Exact copies (the CorrT
  argmax candidate) keep persisting-mob scores at 0.76-0.83 at depth 16, vs 0.37-0.50 for the soft blend. But CorrT's
  discrete choices raise seen-terrain error 1.3-2.8% → 4.7-7.5% and phantoms 0.03-0.07 → 0.11-0.16.
- **Historical warning:** S3 generated-prefix training made recursive ghosting worse. The conditions differ now: chance is
  inferred from the real future, so a self-fed point loss does not average over outcomes.

## Arms (one process per seed; identical windows and noise streams; matched updates)

- **ic:** the IC v3 model of that seed, continued from its 3,000-update state (weights and optimizer) with the unchanged IC
  objective for 1,500 more updates (3,000 → 4,500).
- **icsf:** the same start and the same IC objective, plus a self-fed term. In each 16-frame window, frames 0-11 are real
  context; frames 12-15 are generated one at a time from the model's own context:
  - the afterstate b (from the current context);
  - the chance outcome inferred by the IC posterior (model prior) against the REAL frame;
  - the generated frame LN(T̂ b) is scored with the D point loss plus the chance cross-entropy, then fed back (detached) as
    the next context frame.
  - The self-fed term is averaged over the 4 steps and added to the IC loss with weight 1.
  - No labels. The chance outcome comes from the label-free posterior.

**Shared:** seeds sets 1 / 2 (each from its own IC v3 checkpoint); continued streams from the saved IC state; phase AdamW
(full learning rate); batch 16; L16.

## Readings (both seeds; bright evaluation roots; 1,500 continuation updates)

**Primary (recursion, `transport_recursion.py` + `mob_fading.py`, 8 samples):**
- **P1:** persisting-mob score at depth 16 is higher for icsf than ic, by ≥ 0.05, at both seeds.
- **P2:** cow census at depth 16 is closer to 1 for icsf than ic, at both seeds.

**One-step non-inferiority (`transport_eval.py`, both readers):** icsf keeps valid ≥ 90%, duplicate and absent ≤ 5%, static
≤ max(free, 0.5%), all-source valid movement ≥ 15%, cow valid movement ≥ 20.5%.

**Secondary:** zombie census, terrain error on seen cells, camera consistency, pursuit, decision proxies (zombie
adjacency, health drops).

**Interpretation rule (fixed now):**
- P1 and P2 at both seeds, with one-step non-inferiority → exposure to its own predictions was a major missing ingredient (a
  training failure the current architecture can learn to correct).
- Otherwise → the structural account stands: the soft persistence mechanism must change (Track B).
- A partial result is reported as partial.

## Limits

- One continuation budget; no tuning of K, weights or budget; no automatic extension.
- Self-fed context is 4 steps, while the readings are at depth 16.
- The continuation also has more total training than IC u3000; the matched control is the continued ic arm.
- Fresh confirmation stays sealed.
