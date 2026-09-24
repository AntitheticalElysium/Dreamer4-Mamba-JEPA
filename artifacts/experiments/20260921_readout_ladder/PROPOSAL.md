# What I propose after A/B/C, and why it answers our exact failure

Written 2026-09-25. Everything cited was read (paper text or code) before being cited; the numbers
from this repo are from the committed evidence files named beside them.

> **Corrected the same day, after review.** Four statements below were wrong or overstated and are fixed
> in place:
> 1. **Shared heads are not new.** H2 already trains its reward and continuation heads on *generated*
>    readouts, with gradients into the world (`d4mj/train.py:_bridge_head_losses`), and those shared heads
>    did not transfer. What is new here is a health-change target read only from the states, and the
>    per-tile state.
> 2. **The 11× compared two different encoders** (Raw H2 tokens against the old TC encoder's `u`). Within
>    the Raw encoder the ratios are 2.2× against z and 4.8× against the pooled grid. They are *squared*
>    distances, not the per-token L1 loss proposed.
> 3. **The one-step opportunity is stay versus move.** A SLEEP chosen now is hit as an awake player
>    (CONFIRM.md correction); the 7-damage sleep penalty arrives a step later.
> 4. **§7 is superseded.** The test that runs is the revised 2×2 in `spatial.py`: H2-style heads in
>    every arm, crossed with the added health head. It is judged on the *trained* head against shortcut
>    controls.

## 1. The problem, stated precisely

The H2 gate needs a head that reads the world's **imagined** successor to choose the action that
does not kill the agent. On Craftax-Classic the deciding case is a zombie next to the player:
staying put costs 2 health and moving away costs nothing, so the one-step opportunity is stay versus
move; at low health that is death. A SLEEP chosen now is hit as an awake player: attacks read the
sleep flag from before the action (CONFIRM.md). The 7-damage sleep penalty arrives a step later. In the frame the agent sees, the consequence is a change to
**one tile**: the health counter, HUD row 7 column 0, token 63 of the encoder's 9×9 grid
(`craftax_classic/renderer.py:498-507`, texture swapped at health 0; measured in §6: 94% of the fatal
direction in token space sits on that tile). It is caused by a local interaction between three things: the player's tile, the zombie's tile and the action.
(The forks are rendered with `auto_reset=False`, so a fatal successor shows health 0, not a reset.)

Two stages of the pipeline lose that consequence, and they are separate failures:

1. **Interface — the state the world receives.** Canonical LeWM gives the world only
   z = projector(CLS). On zombie roots CLS reads at the action prior, while the same encoder's
   81 per-tile tokens carry the hazard: +0.080 over CLS overall and +0.075 on zombie roots, on sealed
   seeds 51,000+ (BOUNDARY.md). No dynamics can restore what its input lacks: I(death; ẑ′) ≤
   I(death; z, a) (data-processing inequality).
2. **Objective — what the world is trained to get right.** Given a patch-derived state that does
   carry it (u, root readout 0.800 on sealed seeds), a Mamba predictor trained with next-state MSE:
   - reproduces 95.5% of each action's effect;
   - reproduces none of the component that decides death. That component is 0.04% of the effect
     energy and lies in u's lowest-variance PCA directions. Along it the prediction is worse than
     copying the root (error ratio 25), and a direction fit on real successors reads generated ones
     at 0.495 (DIAGNOSE.md).

   The imagined successor is therefore *less* informative for the decision than its own input: A
   reads 0.739 against root u's 0.800 (ABC.md, sealed). Adding deaths as targets under MSE made it
   worse (B − A −0.042*). Re-weighting coordinates on identical data helped (C − B +0.070*) but did
   not close the gap (C 0.768).

In one sentence: **the world is trained to be right on average in a geometry where the survival
consequence is a vanishing component, so it spends its capacity on the view scrolling and gets the
health counter wrong** — and in canonical LeWM the geometry does not even contain the cause.
Measured directly (§6): in the u-world's state, getting only the health counter wrong costs 2.2% of a
typical action's effect.

## 2. This is a known failure, with a literature

| Our symptom | The same failure outside this repo |
|---|---|
| MSE allocates by energy, not by decision relevance | *Objective mismatch*: one-step likelihood is not correlated with control (Lambert et al. 2020). VaGraM (ICLR 2022): "a mean squared error would penalize deviations equally by their L2 norm without accounting for the relevance of the dimensions". Its fix weights the error by the value gradient; our error along the fatal direction w *is* that quantity, and it is 25× the average. |
| A small, rare element that decides the outcome is lost under a loss dominated by large elements | OC-STORM (Zhang et al., ICLR 2026): world models trained with pixel reconstruction "often fail to capture small, task-critical objects in complex, dynamic scenes"; directing capacity to objects fixes it on Atari 100k. EMERALD (2025, **Crafter**): DreamerV3's compressed latent "fails to perceive crucial details like diamonds and skeleton arrows". |
| A self-predictive objective alone does not guarantee decision-relevant content | Ni et al. (ICLR 2024): next-latent prediction (ZP) alone "is trivial and can be achieved by employing a constant representation"; it needs reward prediction (RP) as well. Tang et al. (2023): self-predictive learning performs a spectral decomposition of the transition matrix. DeepMDP (2019): reward and transition losses *together* bound value error. |
| CLS keeps slow, predictable content and drops mobs driven by the random number generator | Klindt, LeCun & Balestriero (2026), *When does LeJEPA learn a world model?*: the optimum "extracts the slowest features"; with fewer dimensions than the world has, which subspace is kept is undetermined. LeWM's own paper: "fine-grained rotational information remains difficult to encode in compact latent spaces". DINO-WM: "world models that encode observations as a single latent vector show a significant drop in performance" (PushT 0.90 patch vs 0.44 CLS; Wall 0.96 vs 0.58). *Better Slots, Better Worlds* (arXiv 2608.12078): LeWM's CLS "collapses" under object-level shifts where patch features "degrade only moderately". |
| Predicted change misses how the action matters | Hansen & Wang (2026): *action-marginalized* hallucination, a data-coverage gap; re-sampling beat loss re-weighting. Delta-JEPA (2026): LeWM's predicted displacement barely depends on the action; decoding the action from real latent differences fixes it on 4 control tasks. |
| Deterministic regression under branching futures | MoP-JEPA (2026): the regression-optimal predictor outputs the conditional mean, "a point between the true next states that corresponds to no state at all". DreamerV2: categorical latents fit multimodal, non-smooth changes (objects appearing or disappearing). |

## 3. Who fixed it, with architectures like ours

| system | environment | world state | per-position loss | decision heads in world loss | reconstruction-free | result that isolates the lever |
|---|---|---|---|---|---|---|
| Dedieu et al. 2025 (ICML) | **Craftax-Classic, 63×63, 9×9 tiles of 7×7 — our exact frames** | 81 per-tile tokens | cross-entropy over nearest-neighbour tile codes | reward, termination | no (pixel-patch codes) | per-tile factorisation 43.36 → 58.92 reward; **continuous per-patch regression collapses to 21.20** vs 67.42 |
| EMERALD 2025 | Crafter | spatial 4×4×32 categorical | MaskGIT cross-entropy / KL | reward, continuation | no (L2 reconstruction) | vector → spatial latent at the same architecture: 31.8 → 58.1 |
| Dreamer-CDP 2026 | Crafter | 32×32 categorical + JEPA-style continuous target | KL + continuous prediction | reward, continuation | **yes** | continuous prediction without the categorical KL: 6.3 (vs 16.2); without the reward gradient: 12.7 |
| TD-MPC2 2024 | 104 continuous-control tasks | vector, SimNorm (soft one-hot groups) | JEPA consistency ‖z′ − sg(h(s′))‖² | reward and value (cross-entropy over bins) **on rolled-out latents**; termination head for episodic tasks | **yes** | the closest JEPA relative: heads read the *predicted* latents during training |
| V-JEPA 2-AC 2025 | robot arm | frozen patch tokens | per-token LayerNorm + L1, plus a 2-step rollout loss (`configs/train/vitg16/droid-256px-8f.yaml`) | none (goal-reaching) | **yes** | |
| DINO-WM 2024 | PushT, Wall, … | frozen patch tokens | MSE per patch | none (goal-reaching) | **yes** | patch vs CLS: PushT 0.90 vs 0.44, Reach 0.92 vs 0.60 |
| PSG-JEPA 2026 | robot arm | JEPA latent | forward prediction + grounding of latents and latent *pairs* in physical-state labels | — | **yes** | beats forward-prediction-only JEPA world models |

The pattern: **every world model I found that works on Crafter or Craftax has a categorical target
per position or per group, and reward and termination heads inside the world-model loss. The
strongest also have a spatial state (tile-aligned in Dedieu et al.).** Ours has none of the three. (One caveat from §6: their
categorical targets are over *local* content — pixel patches, or latents shaped by reconstruction; over
our encoder's contextual tokens, codes do not isolate the consequence.) The JEPA world models that
work without decision heads are goal-reaching planners in continuous control, with no rare fatal
events. That is the regime LeWM was designed and evaluated for.

## 4. The proposal, exactly — revised after the premise check (§6)

**A per-tile, decision-grounded LeWM.** Three changes to the world. The encoder, SIGReg and the
reconstruction-free rule stay as they are.

1. **State: the encoder's 81 per-tile output tokens**, one per 7×7 tile, including the 18 HUD tiles.
   LeWM's ViT already computes them (`LeWMEncoder._hidden`); today they are discarded and only
   projector(CLS) reaches the world.
2. **Predictor and loss: per tile.** The predictor attends across the 81 tiles within a step and runs
   over time, as V-JEPA 2-AC's block-causal transformer and Dedieu et al.'s transformer do; Mamba can
   still carry time. The target is each tile's layer-normalized token under L1, V-JEPA 2-AC's shipped
   recipe. §6 measured, in squared distance, what a per-tile state charges for getting only the health
   counter wrong: 0.25 of a typical action effect, with 91% of that on the health tile itself. Against
   the same Raw encoder that is 2.2× z and 4.8× the pooled grid. The 11× over the u→u world's state
   crosses encoders. None of these is a measurement of the L1 loss.
3. **Continuation and health-change heads, inside the world loss.** They are trained on encoded tokens
   and applied to predicted tokens, with gradients into the predictor. This is the reward-prediction
   (RP) condition of Ni et al. and TD-MPC2's heads on rolled-out latents. **H2 already does this for
   reward and continuation** (`_bridge_head_losses`), and its shared heads still do not transfer
   between real and generated states: sharing does *not* make them agree. H2's heads also read the
   predictor's history, which can bypass the generated latent. The new part is a **health-change**
   head that reads only (s_t, s_{t+1}), so its gradient reaches the predictor only through the
   generated state (verified in `spatial.py`'s unit test).

**Dropped:** per-tile categorical targets built on these tokens. Their large cost (1.03) is 89% *other*
tiles' codes flipping, because the ViT's tokens are contextual. Dedieu et al.'s codes are
context-free pixel patches. Categorical targets come back only with local, context-free tile targets,
which this encoder does not produce.

Windows must reach terminal transitions. The TC layout can never show a death (DIAGNOSE.md); B showed
that deaths *under vector MSE* hurt, which is a statement about that loss, not about exposure.

## 5. Defense, failure by failure

| measured failure | mechanism | component | outside evidence the lever works | prediction on our data (falsifiable) |
|---|---|---|---|---|
| CLS lacks the zombie (sealed: z at prior on zombie roots) | a compact global summary keeps slow, predictable content; mobs driven by the random number generator are the opposite | 1. per-tile state | EMERALD +26 pts on Crafter (vector → spatial, same architecture); Dedieu +15.6 on Craftax (factorise into 81 tiles); DINO-WM patch vs CLS | the root's per-tile state carries the hazard. **Already shown**: +0.080 over CLS, +0.075 on zombie roots, sealed |
| the u-world spends capacity on the scroll; getting the consequence wrong costs it almost nothing | pooling folds the health tile into its neighbours; uniform MSE follows the energy | 2. per-tile state and loss | V-JEPA 2-AC, DINO-WM (per-patch targets); Dedieu (per-tile factorisation) | **measured (§6)**: the consequence's cost rises 11× over u (0.022 → 0.25), 91% on the health tile |
| even per tile, the consequence is a quarter of an action effect; imagined states put it where no real-fit head looks (alignment 0.49); heads do not transfer real ↔ generated | next-latent prediction alone does not ask for decision-relevant content; a constant satisfies it | 3. heads on predicted tokens, shared with real ones | Ni et al. (RP + ZP); DeepMDP bound; TD-MPC2; Dreamer-CDP without the reward gradient 12.7 vs 16.2; every working Crafter world model has these heads. Delta-JEPA and PSG-JEPA: supervising latent *differences* reshapes a JEPA's transition geometry without reconstruction. Delta-JEPA supervises the action, which our world already gets (95.5%); what we lack is the consequence | real-fit fatal direction reads generated successors well above 0.5; generated not resolved below the root |
| the TC world never saw a death; deaths under vector MSE hurt | coverage | terminal-reaching windows | Hansen & Wang 2026 (coverage-aware sampling); Curious Replay 14.5 → 19.4 on Crafter | carried by every arm of §7 |

**Why this is not Direct again.** Direct had several tokens and had heads, and it still failed
state-conditioning (terminal-supervision verdict: escape-rich 0.35–0.42 against a 0.999 ceiling), with
terminal tails installing an action-prior shortcut. Two differences:

- **Direct's tokens were slots, not tiles.** The health counter and the zombie were never tokens of
  their own.
- **Whether Direct's root state carried the cause was never shown.** A head can only fall back on the
  action prior when the state lacks the cause. Here, carrying the cause is shown on sealed seeds.

§7 tests this directly: heads on z, which lacks the cause, *should* reproduce the shortcut.

**Why this is not a corruption of LeWM.** Patch-token JEPA world models are the mainstream of the
family: V-JEPA 2-AC (FAIR) and DINO-WM predict patch tokens with a frozen or JEPA encoder and no
decoder. LeWM's single-CLS state is a design for compact, goal-reaching planning in continuous
control, which is the regime its paper evaluates. Ours is a different regime: rare fatal events,
discrete tiles.

## 6. Premise check, run before recommending anything

Nothing trained. Observability roots (EXPLORATORY), canonical Raw H2 encoder, all 17 real successors;
493 fatal and 2,268 damage judgement roots.

**Declared** (`geometry.py`, predeclared `01333be0`; memory-only fixes `b01e5055`, `cdab80f5`):
**`per_tile_premise_fails`.**

| geometry | validity (within-root AUC) | share of effect energy along the mean fatal-minus-surviving direction | where the direction lives |
|---|---|---|---|
| z | 0.923 | 2.7% | |
| grid_pca | 0.950 | 6.4% | |
| u_old | 0.949 | 28.0% | |
| per-tile tokens (raw = layer-normed) | **0.998** | 2.1% | **one tile, the health counter (7,0): 94%** |
| per-tile codes | 0.996 | 0.18% | health counter 82% |

By the rule, the per-tile shares are under 3× grid_pca's, so the premise fails. What the run
establishes regardless:

- **The consequence is one tile**, the health counter: 94% of the direction, 74% for damage.
- **The scroll is the effect**: the HUD holds 7.6% of the per-tile effect energy.

There is a flaw in my measure. In the pooled geometries the mean direction also carries
stay-versus-move energy, because fatal actions are mostly the stay-put ones. It shows as lower
validity, and as u_old's 28% against the 0.04% DIAGNOSE measured along a discriminative direction.
So the pooled baselines are inflated.

**Post hoc, confound-free** (`twin.py`, predeclared `a7d50420`): **`mixed`.** Each fatal successor is
paired with a twin that is identical except for the health tile, copied from a surviving branch.
Every fatal-versus-surviving HUD difference was verified to lie inside that tile. The cost is the
twin distance divided by the mean within-root action effect: what each geometry charges for getting
*only* the consequence wrong.

| geometry | fatal | damage | share of the cost on the health tile |
|---|---|---|---|
| z | 0.113 | 0.036 | |
| grid_pca | 0.052 | 0.010 | |
| u_old (the A/B/C world's state) | **0.022** | 0.010 | |
| per-tile tokens | **0.250** | 0.067 | **91% / 95%** |
| per-tile codes | 1.034 | 0.366 | 11% / 19% |

- **Per-tile tokens charge 11× what u did** (2.2× z, 4.8× the pooled grid), and 91% of it on the right
  tile.
- **It is still a quarter of an action effect.** No geometry makes the consequence dominant, which
  is why heads carry the objective (§4.3).
- **The codes' cost is mostly spurious.** It is 89% other tiles' codes flipping through context, so
  codes are dropped.
- **CLS already charges 5× what u did.** Canonical LeWM's loss is not blind to the consequence; its
  failure is the missing cause (§1.1).

## 7. What would decide it — a 2×2 matched test, to predeclare and review before running

The premise says what the loss *would* weigh. Only training says what a world learns. Two factors,
crossed:

- **factors:** state (z vs per-tile tokens) × decision heads (none vs continuation and health-change
  heads on predicted states);
- **held fixed:** the frozen canonical Raw H2 encoder, one factual pool with terminal-reaching
  4-frame windows and broad hazard-choice coverage, one small block-causal predictor, and the same
  updates for every arm;
- **judged:** once, on a new sealed seed block (53,000+), in the existing harness.

| arm | state | heads | prediction written now |
|---|---|---|---|
| V | z | — | zombie roots at the prior (the cause is absent) |
| V+H | z | ✓ | **the action-prior shortcut**: no zombie gain. If V+H succeeds, the interface story is wrong |
| T | tokens | — | preserves more of the root than the u-world did, but is still resolved below the root |
| T+H | tokens | ✓ | generated not resolved below the root per-tile readout; zombie gain over V+H |

Readouts:

- generated-fitted safe choice against the root per-tile readout (ceiling) and the prior (floor);
- the zombie stratum and escape-rich roots;
- alignment of the real-fit fatal direction to generated states;
- the health tile's prediction error.

It fits in 6 GB: 4 × 81 = 324 tokens, d = 192.

## 8. Risks, and what this does not claim

- **Nothing here is shown to work on our data yet.** §6 measures what a loss would weigh, and the
  declared rule failed. Only the post-hoc measure favours per-tile. §7 is the test.
- **Heads are a loss change.** A/B/C returned `stop_loss_tweaking` and pointed to spatial dynamics or
  uncertainty modelling. The per-tile state is spatial dynamics. Heads add a new supervised signal,
  not a re-weighting, but they are still a change to the loss. That is why they form their own
  factor, not a bundle.
- **The two-step problem is stochastic.** After one step the zombie moves at random, and
  deterministic regression then predicts the mean (MoP-JEPA). Dropping codes leaves this open.
  Remedies are a hard-assigned mixture predictor (MoP-JEPA), flow matching (Flow-JEPA), or local
  categorical tile targets. None is proposed until one-step works.
- **The d4mj code contract.** `LeWMWorld` is vector-only. Everything above lives in
  `artifacts/experiments` until it wins. Porting a per-tile world into `d4mj` is a new module, which
  the contract forbids without an explicit decision from you.

## Sources

- Dedieu et al. 2025, *Improving Transformer World Models for Data-Efficient RL*, arXiv 2502.01591
- EMERALD, *Accurate and Efficient World Modeling with Masked Latent Transformers*, arXiv 2507.04075
- Hauri & Zenke 2026, *Dreamer-CDP*, arXiv 2603.07083 (text in scratchpad)
- Hansen et al. 2024, *TD-MPC2*, arXiv 2310.16828
- Assran et al. 2025, *V-JEPA 2*, `third_party/vjepa2/app/vjepa_droid/train.py`, config `droid-256px-8f.yaml`
- Zhou et al. 2024, *DINO-WM*, arXiv 2411.04983
- Maes et al. 2026, *LeWorldModel*, arXiv 2603.19312
- Klindt, LeCun & Balestriero 2026, *When Does LeJEPA Learn a World Model?*, arXiv 2605.26379
- Ni et al. 2024, *Bridging State and History Representations*, arXiv 2401.08898
- Tang et al. 2023, *Understanding Self-Predictive Learning for RL*, arXiv 2212.03319
- Gelada et al. 2019, *DeepMDP*, arXiv 1906.02736
- Lambert et al. 2020, *Objective Mismatch in Model-based RL*, arXiv 2002.04523
- Voelcker et al. 2022, *VaGraM*, arXiv 2204.01464
- Zhang et al. 2026, *OC-STORM*, arXiv 2501.16443
- Hansen & Wang 2026, *Hallucination in World Models is Predictable and Preventable*, arXiv 2606.27326
- *Delta-JEPA*, arXiv 2606.31232; *PSG-JEPA*, arXiv 2608.06799; *MoP-JEPA*, arXiv 2607.05238;
  *Flow-JEPA*, arXiv 2608.29029;
  *Better Slots, Better Worlds*, arXiv 2608.12078
- Hafner et al. 2021, *DreamerV2*, arXiv 2010.02193; Kauvar et al. 2023, *Curious Replay*, arXiv 2306.15934
