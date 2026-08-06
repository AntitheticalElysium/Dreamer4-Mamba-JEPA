# The imagination bridge: whole-architecture analysis and option decision

Question (user): how does the closed-loop compounding failure affect the wider
architecture (`m3_hjwm/ARCHITECTURE_SPEC.md`), and which of three options do we
take — (1) DreamerV3/4-style KL prior/posterior, (2) Dreamer-CDP, (3) multi-step
rollout loss? Constraint: no technique-stacking; every change must be placed in
the whole data flow.

## 1. The data-flow map: where every distribution mismatch lives

Training-time flow (spec §3–§7, as implemented):

```
o_t ──online enc (mask?)──> X_t ─(+a_{t-1} emb)─> temporal ──> C_t
o_{t+1} ──EMA enc──> Y_{t+1}                         │
                                                     ▼
                              predictor(C_t, a_t) ─> Ŷ_{t+1} ──cosine──> Y_{t+1}
C_{t+1} ──pool──> reward/continue heads              (teacher-forced only)
```

Imagination-time flow (spec §8):

```
Ŷ_{t+1} ─(+a_t emb)─> temporal ──> C_{t+1} ──> predictor(C_{t+1}, a_{t+1}) ─> Ŷ_{t+2} …
                        │
                        └──pool──> reward, continue, actor, critic
```

Every mismatch between the two flows, enumerated (M1–M5):

- **M1 — temporal input:** trained on online-encoder tokens of real frames;
  deployed on predictor outputs (target-like generations). No training term ever
  feeds a generated token into the temporal core.
- **M2 — predictor input:** trained on contexts built from real histories;
  deployed (k ≥ 2) on contexts built from its own generations. No gradient ever
  crosses the predictor→temporal→predictor composition.
- **M3 — task heads:** trained on real-history contexts; consumed on imagined
  contexts (this is where Biased-Dreams-style reward overestimation would
  enter). Phase E must calibrate on imagined states, not only real prefixes.
- **M4 — online vs EMA token space:** the temporal core consumes online-space
  tokens at train but EMA-space predictions at deploy (small once EMA
  converges, real early in training).
- **M5 — actor/critic:** live entirely in imagination; they inherit whatever
  M1–M4 leave broken. The reliability system (spec §10) can only *flag* this,
  not fix it.

Measured consequence (reviews/2026-07-12-validation-run-results.md): one-step
prediction is near the copy bar, but closed-loop error compounds ~2–10× faster
than world drift, catastrophically so for Mamba-2 (0.027 → 0.74 by k=8). The
architecture is not "wrong component by component" — it is missing the one
structural element every working lineage has.

## 2. Every working lineage has an explicit train↔imagination bridge

Verified in pinned sources/papers this session:

| lineage | bridge mechanism | where verified |
|---|---|---|
| DreamerV3 / Dreamer-CDP | KL(dyn): prior is *trained* to match the posterior at every step, so imagination (rolling the prior) stays on the observed-state distribution | `fmi-basel__Dreamer-CDP/dreamerv3/rssm.py:134-141` |
| Dreamer 4 | diffusion/shortcut **forcing**: training inputs are corrupted contexts by construction; x-prediction avoids high-frequency error accumulation; inference *keeps* contexts at τ_ctx = 0.1 noise "to make the model robust to small imperfections in its generations" | paper §Shortcut forcing (2509.24527) |
| V-JEPA 2-AC | explicit rollout loss: `L = L_teacher-forcing + L_rollout`, with `L_rollout = ‖P(a_{1:T}; s_1, z_1) − z_{T+1}‖₁`, T=2, gradient through the autoregressive composition | paper Eq. 3–4 (2506.09985); Fig. 6 |

Our spec §7+§8 has none of these. That is the wider-picture diagnosis: the
previous project's death-by-tweaking happened because fixes were applied to
M1–M5's *symptoms* one at a time; the three lineages each fix the *class* with
one mechanism.

## 3. The three options against the thesis

**Option 1 — DreamerV3-style stochastic latent + KL.** Structurally sound
bridge, but it replaces the JEPA representation contract with a distributional
bottleneck, and the result is "Dreamer with a Mamba core" — which is DRAMA,
already published. Kills the project's identity; also drags in KL balancing,
free bits, unimix. **Rejected as core** (remains a reference baseline).

**Option 2 — Dreamer-CDP.** Recon-free and cosine-based (JEPA-adjacent), but
its bridge is still the retained RSSM stoch/KL machinery — the "baggage" is not
incidental, it *is* their bridge. Adopting it means our world-model core becomes
CDP's, and the dense-token + Mamba + reliability contributions become
decorations on someone else's architecture. **Rejected as core; promoted as the
mandated comparison baseline** (HANDOFF §7 already lists it).

**Option 3 — multi-step rollout loss (+ Dreamer-4-style context robustness in
reserve).** The JEPA-native bridge, verbatim from V-JEPA-2-AC (Eq. 3–4): keep
teacher forcing, add a T=2 autoregressive composition term with gradient through
predictor→temporal→predictor. Directly closes M1 (temporal trains on generated
tokens), M2 (gradient through the composition), M4 (generated inputs are
EMA-space). M3 is addressed at gate level (Phase E calibrates on imagined
states). Cost: ~2 extra parallel-scan passes per update — measured 722 MiB peak,
minutes per run; trivially affordable. Theory: "On Training in Imagination"
(2605.06732) bounds compounding error monotonically in the Lipschitz constants
of the learned dynamics — a rollout loss directly minimizes the k-step error the
bound describes; Lipschitz logging stays a diagnostic (the old project verified
that satisfying the bound alone does not fix a broken model). **Adopted,
pending the pre-registered experiment below.**

Alignment check against project goals: option 3 keeps (a) reconstruction-free
JEPA latents, (b) dense tokens, (c) the Mamba temporal bet testable, (d) the
Dreamer-4-scale-down framing — it is, in fact, the deterministic-latent analog
of Dreamer 4's own forcing idea. Reserve lever from the same source: small
input corruption of temporal inputs at train *and* imagination (τ_ctx analog) if
rollout alone is insufficient — note that 60% masking was accidentally a crude,
overdosed version of this; Dreamer 4 uses 10% noise, not 60% occlusion.

## 4. Spec v2 amendments implied (for consensus)

1. §7 objective gains the bridge term:
   `L = … + λ_TF L_JEPA + λ_roll L_rollout` with `λ_roll = 1`, `T_roll = 2`
   (paper defaults), final-step cosine, gradient through the composition.
2. §8 imagination unchanged — it is already the deployment flow; the point is
   that training now visits it.
3. §5 backend decision reopened: re-judge GRU vs Mamba-2 only *after* the
   bridge exists (both prior D2 verdicts were artifacts of its absence).
4. §10/Phase E-F: reward/continue calibration and reliability training must use
   imagined states from real prefixes, not only real-history states (M3).
5. Explicit non-goal recorded: no KL/stochastic latent (option 1), no RSSM
   adoption (option 2), unless the bridge experiment fails at scale.

## 5. Pre-registered experiment (running; scratchpad `rollout_loss_experiment.py`)

Unmasked arm, 4000 updates, same data/seeds, both backends, T_roll=2:
- R1: closed-loop changed-token error at k=4,8 improves ≥2× vs the no-rollout
  run (GRU 0.203/0.269; Mamba-2 0.520/0.740);
- R2: D1 bar (beats copy at any k ≤ 8) reported either way;
- R3: one-step prediction regresses ≤20% (baseline 0.0319 changed-token).

Results to be appended to `2026-07-12-validation-run-results.md` when complete.
