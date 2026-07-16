# Control-centric action-conditioned self-prediction: literature notes

> **CORRECTIONS (2026-07-16, companion ground-truth check — read these before
> the sections below):**
> 1. **RETRACTED**: "the stack currently trains with NO reward/continuation
>    pressure on the temporal path" (item 4 rationale below) is FALSE.
>    LossConfig defaults reward=continuation=1.0 and both heads consume pooled
>    POST-temporal context; the companion's gradient diagnostic confirms the
>    temporal core receives reward gradients in every validated run. Phase E
>    is therefore held-out CALIBRATION + imagined-rollout validation, not the
>    first introduction of reward grounding. DeepMDP also does not establish
>    that reward grounding is required or sufficient — only that jointly
>    learned reward-aware latents carry control-relevance bounds.
> 2. SPR (source verified at mila-iqia/spr@0b9dd4e): precedents dense spatial
>    action-conditioned CONV transitions with per-step targets and joint RL
>    training. Merely raising our rollout_steps to 5 is NOT a faithful SPR
>    arm; label any such arm "SPR-inspired depth ablation".
> 3. BYOL-AC: literal per-action predictors would add ~1.80M params beside a
>    240k model — not "trivial". The realistic ablation is action-modulated
>    predictor conditioning (FiLM/AdaLN-zero, cf. LeWM) or small per-action
>    heads, labelled "BYOL-AC-motivated". Its spectral results rest on strong
>    idealized assumptions, and our predictor already receives a direct action
>    token — BYOL-AC motivates an ablation; it does not diagnose the current
>    architecture as weakly conditioned.
> 4. Tang et al.: the two-timescale/semi-gradient analysis concerns JOINTLY
>    learned representations; with our frozen encoder an update-ratio ablation
>    is an empirical optimization probe, not a theorem-backed transplant.
> 5. TACO (source verified at FrankZheng2022/TACO@84c38e3): faithful TACO
>    contrasts batch-matched pairs in a BxB InfoNCE. A same-anchor
>    true-vs-wrong-action objective is "TACO-inspired counterfactual action
>    ranking" — gate-adjacent, separately labelled, untouched-set evaluation.
> 6. SPR/TACO/DBC official repositories are pinned under third_party/sources/
>    (see SOURCES.lock) as of 2026-07-16.

Date: 2026-07-15. Context: both external reviews (third-agent transcript,
companion audit) independently identified the same corpus gap — the project's
pinned literature was strong on 2025-26 JEPA/world-model work and missing the
older RL line that studies exactly our blocker (weak use of action identity in
latent dynamics). Six papers acquired, verified against title pages, pinned in
third_party/PAPERS.lock. Sections below record what each actually says (read,
not summarized from memory) and what it licenses for us.

## 1. SPR — Schwarzer et al., ICLR 2021 (2007.05929v4)

What it is: model-free Rainbow + auxiliary K-step action-conditioned latent
self-prediction. Online encoder f_o, EMA target encoder f_m (tau=0.99 without
augmentation, tau=0 with), iterative latent transition model
z_hat_{t+k+1} = h(z_hat_{t+k}, a_{t+k}), cosine-similarity loss on
projected + predicted embeddings (Eq. 4), K=5, truncated at episode
boundaries, loss weight lambda=2 against the RL loss.

Directly relevant details:
- **The transition model h is dense/spatial**: two 64-channel 3x3 conv layers
  applied to the 64x7x7 spatial output of the encoder, with the one-hot
  action broadcast to EVERY spatial location and concatenated to the input of
  the first conv (their Sec. 2.3, following MuZero). This is a real published
  precedent for "dense spatial latent dynamics with per-position action
  conditioning" — i.e., for OUR dense-token + per-token additive action path,
  and against the claim that pooled-global state is the only precedented
  choice at this scale.
- SPR trains representation and dynamics JOINTLY with the RL loss (no frozen
  encoder); BatchNorm in the transition model; no negative samples.
- K=5 vs our rollout_steps=2: multi-step latent rollouts with per-step targets
  is their core mechanism, at K larger than ours.

Licenses for us: (a) dense spatial action-conditioned transition has
precedent; (b) if the causal signal stays weak after step 4, increasing
rollout depth K (2 -> 5) with per-step losses is a literature-backed lever,
cheaper and better-precedented than new architecture.

## 2. BYOL-AC — Khetarpal, Guo et al., 2024 (2406.02035v1)

What it is: theory (ODE analysis) of action-conditional self-prediction.
Three objectives: BYOL-Pi (single predictor, action-marginalized), BYOL-AC
(**one predictor P_a per action**), BYOL-VAR (difference objective). Main
results under idealized assumptions:
- BYOL-Pi converges to top-k eigenvectors of (T^pi)^2 — policy-averaged
  dynamics features (Thm 1).
- BYOL-AC converges to top-k eigenvectors of |A|^-1 sum_a T_a^2 — features of
  the PER-ACTION dynamics (Thm 2).
- Variance relation (Remark 1): E[D_a^2] = (E[D_a])^2 + Var_a(D_a) — BYOL-AC
  additionally selects features whose eigenvalues VARY across actions, i.e.
  action-distinguishing features. Model-free view (Thm 5): Phi fits V, Phi_ac
  fits Q, Phi_var fits the advantage. Empirically BYOL-AC is best overall.

This is the theory-level statement of our Stage-A result: an
action-marginalized (or weakly conditioned) self-predictive objective learns
T^pi-features — "acting vs not acting", persistence, saliency — without
distinguishing WHICH action. The paper's fix is architectural strength of
action conditioning in the predictor: per-action predictors, not a shared
predictor with the action mixed weakly into its input.

Licenses for us: the pre-registered action-discriminative arm (if step-4
parity/weakness persists) should be **strengthened action conditioning in the
predictor** — per-action prediction heads (17 actions is small; parameter
cost trivial at our scale) or action-modulated (FiLM/AdaLN-zero, cf. LeWM)
predictor conditioning, evaluated as a separately labelled arm against the
same causal gates. This is better-grounded than inventing a contrastive loss.

## 3. Understanding Self-Predictive Learning for RL — Tang et al., ICML 2023 (2212.03319v1)

What it is: the theoretical predecessor of BYOL-AC (BYOL-Pi analysis). Two
algorithmic ingredients make naive latent self-prediction avoid collapse
despite the trivial optimum: (1) **two-timescale optimization — the predictor
P must be optimized (much) faster than the representation Phi**; (2)
**semi-gradient** updates (stop-gradient on the target). Learning then
performs spectral decomposition of the transition matrix; orthogonal
initialization preserved along the ODE.

Licenses for us: (a) our stop-gradient/EMA-target discipline is load-bearing,
not incidental; (b) "optimization budget was the lever" has a sharper,
cheaper-to-test refinement: a **predictor/temporal-core update-ratio ablation**
(e.g., 2-4 predictor steps per encoder-free world update, total FLOPs
matched) may recover the 16k-step benefit at lower cost — registered as a
post-step-4 ablation candidate, not a step-4 change.

## 4. TACO — Zheng et al., NeurIPS 2023 (2306.13229v3)

What it is: InfoNCE mutual-information objective between (state
representation + ACTION-SEQUENCE representation) and future state
representation; learns state and action embeddings jointly; theory: the MI
objective suffices to represent the optimal value function. Strong DMC
results online (+40% at 1M steps over SOTA model-free) and offline.

Licenses for us: the contrastive route is the documented alternative if
regression-style prediction keeps under-using actions: a same-anchor
true-vs-wrong-action InfoNCE term is TACO-shaped. Caution: it directly
optimizes our evaluation gate's discrimination structure, so it must live in
a separately labelled arm and be judged on the untouched final set only.
Also: TACO learns an action REPRESENTATION — our 17 discrete actions
currently enter as a learned embedding table, which is already the discrete
analogue; the gap is conditioning strength, not embedding existence.

## 5. DeepMDP — Gelada et al., ICML 2019 (1906.02736v1)

What it is: latent space model trained on two losses — reward prediction and
next-latent-distribution prediction — with guarantees (via MMD/Wasserstein
metrics) that the latent state is a good representation (bounds value-function
quality; connects to bisimulation). Atari: large gains as auxiliary task.

Licenses for us: the missing ingredient it names is **grounding**: our
dynamics stack currently trains with NO reward/continuation pressure on the
temporal path (heads exist but Phase E is gated off). DeepMDP says latent
transition + REWARD prediction jointly is what buys control-relevant
representations; pure observation-side self-prediction is not sufficient for
control. This sequences Phase E (reward/continuation calibration) as the
theory-backed next lever after step 4, before any policy work.

## 6. DBC — Zhang et al., ICLR 2021 (2006.10742v2)

What it is: reconstruction-free bisimulation-metric representation learning —
latent distances trained to equal bisimulation distances (reward + dynamics
Wasserstein recursion), discarding task-irrelevant visual detail.

Licenses for us: mostly a conceptual yardstick — "visually predictive" is not
"control-sufficient". Crafter's view has less distractor content than their
driving tasks, so DBC is not an immediate lever; it becomes relevant at Phase
E calibration (register-state sufficiency for reward/value), where a
bisimulation-style probe is the right kind of check.

## Consolidated implications (ranked)

1. Step 4 proceeds UNCHANGED — nothing above touches the GRU-vs-Mamba
   backend question; changing arms now would unregister the protocol.
2. First post-step-4 lever if signal stays weak: **predictor action-
   conditioning strength** (per-action heads / AdaLN-modulation), per BYOL-AC
   — separately labelled arm, same gates, untouched-seed evaluation.
3. Second lever: **rollout depth K 2->5 with per-step targets** (SPR) and/or
   **predictor update-ratio** (Tang) — cheap, well-precedented optimization-
   side ablations that refine "budget matters".
4. Phase E is theory-motivated, not just roadmap: reward grounding is what
   DeepMDP/DBC say converts predictive features into control-relevant ones.
5. TACO-style contrastive term stays in reserve, clearly labelled as
   gate-adjacent (it optimizes the discrimination the gate measures).
6. SPR's spatial transition model is PRECEDENT for our dense-token action
   path; recorded in the evidence ledger against the "no cited system"
   over-statement (the pooled-global-memory add-on remains our own labelled
   divergence).
