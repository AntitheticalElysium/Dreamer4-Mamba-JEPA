# Whiteboard: what a world model for imagination must get right, and where ours went wrong

Started 2026-10-11. A living document: our measured results, read against the literature, written as concepts rather than
as a plan. Each number points to its NOTEBOOK entry or source. Where something is my inference rather than a measurement,
it says so.

---

## 1. The object: three things we kept blurring

The decomposition is the third party's; the formal conditions are Ni et al. (ICLR 2024, arXiv 2401.08898).

| | what it answers | formal object |
|---|---|---|
| **Belief** | What does the history imply about now (adjacency, cooldown phase, terrain seen)? | b_t = φ(h_t) |
| **Transition** | What may happen next, with what probability? | a kernel P(b_{t+1} \| b_t, a_t) |
| **Decision** | Which action is better? | Q(b, a) = E[ r + γ V(b′) \| b, a ] |

**Sufficiency** (Ni et al. §3):
- **RP:** the expected reward must be predictable from b.
- **ZP:** the *distribution* of the next latent must be predictable from b.
- **EZP:** predicting only the *expected* next latent is the weak version.
- **ZP alone is trivial:** a constant satisfies it, so it must be paired with RP or something richer.

**The third party's Jensen point is the gap between ZP and EZP.** V is nonlinear, so V(E[b′]) ≠ E[V(b′)].

**Our worlds fall short even of EZP.** Per-element L1 / L2 optima are the coordinatewise median / mean, which is not the
expectation of any valid state. statoracle (NOTEBOOK 10-10 11:40):
- the median erases uncertain mobs (cows 0.08 vs 0.52 at depth 16);
- the mean superimposes them;
- each world converges toward its loss's statistic as it trains, in both L1 seeds.

**Which details a state may drop:** bisimulation (Ferns et al.; DBC, arXiv 2006.10742). Two states may share a representation
iff they have the same rewards and the same transition distributions. Renderer static changes neither, so it should go. A
timer phase changes future rewards, so it must stay. Visual prominence and frequency are not the criterion; consequence is.

Ni et al.'s own guidance names our tension. Noisy tasks favour self-predictive representations, sparse-reward tasks favour
observation-predictive ones, and deterministic ℓ2 objectives are for deterministic tasks. **Craftax is noisy, sparse-reward
and stochastic.**

---

## 2. Four sources of unpredictability, each needing a different treatment (the central table)

The taxonomy is the third party's. The Craftax facts come from `craftax_classic/game_logic.py` and the renderer; the sizes
are ours.

| source | in Craftax | size (measured) | correct treatment | reaches decisions? | what our losses did with it |
|---|---|---|---|---|---|
| **1. Nuisance** | night static: every map pixel blended with fresh `state_rng` noise, no game consequence | ≈25% of the token's L2 change mass; per-token variance 0.3-5.9 at night (rings 1-4), exactly 0 by day | **invariance** (discard) | no (rendering only) | 43% of M16's L1 error is on night cells with no state change |
| **2. Forgotten history** | hunger / thirst / fatigue / recover counters, per-zombie attack cooldown | every one of 8,183 intrinsic decrements has its reset ≥ 20 steps back; cycles 21-31 steps; the window is 15 frames and imagination rebuilds the state from ≤ 15 frames every step | **memory** | **yes**: the HUD carries 90-100% of the decision gap; imagined food / drink / energy decrements at 0-22% of the real rate across the deterministic worlds; imagined depth-16 health 5.5-6.0 vs 2.4 | no loss can fix it: the information is not in the input |
| **3. Aleatoric** | cow 50% stay / 4 × 12.5%; zombie within 10 cells 75% chase, else a uniform move; spawns; skeleton arrows | **1.71 cells per step** differ across RNG keys from the same state, all of them mobs; source upper bound ≈ **1.3 bits per step** (cow 2 bits, near zombie ~1 bit) | a **conditional distribution, sampled coherently** | **yes, indirectly**: damage is 97% deterministic given the state, but who becomes adjacent is random; the true map near zombies restores 58.5% of imagined damage; 16 faithful futures lift H16 0.670 → 0.727 | L1 erases (median), MSE superimposes (mean); E22MS draws hits at the real rate but duplicates mobs (10% per step vs 1% real) |
| **4. Epistemic** | terrain never seen: fixed per world, unknown until revealed at the view edge | 77% of entering cells never seen in the window; 61-71% of every world's depth-1 gap to the ideal | **sample once from a prior, then persist** within the branch | ~0 on H16 (edge ring share ~0); for an actor that navigates and gathers, **unmeasured** | the D loss's final AdamW update is 81% here; E22FJ's stochastic channel spent 92% of its gain here (random zombie moves 0-3%) |

Details for each row:
- **Row 2:** the counters are deterministic, and their resets are observable (eating, drinking, sleeping, HUD drops). At food 0
  or full health a reset is invisible, but it is still countable from the last visible one. The timers are a pure memory
  problem, not randomness (source, NOTEBOOK 10-10 timer analysis).
- **Row 3:** "1.71 cells" is NOTEBOOK 10-11 20:30. The bits are my estimate from the movement rules (0.77 visible mobs per
  frame, arrows included), an upper bound that ignores blocking.
- **Row 4:** the 61-71% comes from `properscore.py --subst`. The terrain substitution showed that sampling plausible terrain
  from the visible edge does not close it.

**The overarching issue (the answer to the 10-10 question).** A per-element point loss gives ONE currency to four kinds of
unpredictability that need four different treatments, and it spends that currency by error magnitude (P-C, AdamW-corrected).
Every loss we tried moved capacity between the rows:
- **L1:** unchanged cells and the static.
- **The D loss:** epistemic terrain.
- **An unstructured stochastic channel:** epistemic terrain again (E22FJ: 92%).

**The only time we got row 3 was when we told the channel which content moved** (E22MS's moved-content mask: random-move
reconstruction 1.1% → 86%). That was a structural distinction between rows, not a loss weight. It is the lesson to keep:
**the rows must be separated by structure.**

---

## 3. What a transition must be

**Craftax's own step is an afterstate decomposition** (`craftax_step`):
- the player's crafting / action / placement / movement (deterministic except a 10% sapling draw);
- then mob moves and spawns (the chance);
- then plants, intrinsic counters and health (deterministic);
- then a fresh `state_rng` used only by the renderer (row 1).

**The formal model of this exists: Stochastic MuZero** (Antonoglou et al., ICLR 2022):
- the belief and action give a deterministic **afterstate**;
- a **chance outcome** c ~ σ(· | afterstate) is categorical: a fixed one-hot codebook ("a decoder is not required");
- b′ = g(afterstate, c).

It is reconstruction-free and trained by value / policy / reward. The third party's "generate innovations, not states" is this
idea; neither review cited it.

**The point that reframes "categorical vs continuous":** if c carries *all* the randomness of the realized outcome, then g is
deterministic and a point (regression) loss on g is correct, with no averaging. Averaging appears exactly where c is
insufficient. The right axis is therefore not the output type. It is:
- whether the randomness is carried by an explicit, sufficient chance variable;
- whether g is coherent.

**In Craftax the chance is small** (≲ 1.3 bits per step, on mobs). Everything else is deterministic given memory (row 2),
discardable (row 1), or epistemic (row 4: sampled once, when revealed).

| family | randomness carried by | coherence | where it pays generative compute | evidence |
|---|---|---|---|---|
| continuous diffusion / flow over the whole latent (JEDI 2605.13013, Flow-JEPA 2608.29029, Dreamer 4) | noise in every dimension | in principle | everywhere, including the ~97% of cells whose next state is deterministic (1.71 of 63 differ across keys) | JEDI: reconstruction-free and end-to-end, Atari100k + Craftium, 38 A100-h per run. Flow-JEPA: near-deterministic control tasks, a robustness gain (its own limitations note it does not isolate stochastic training) |
| per-tile categorical, independent readouts (Dedieu 2502.01591) | per tile | **breaks**: perfect marginals for one free cow give the wrong count 57% of the time; ITC (2605.16457) measures the duplication on Craftax and a one-to-one assignment lifts return 67.4 → 72.5% | per tile | pixel nearest-neighbour codes |
| spatial categorical + MaskGIT refinement (EMERALD 2507.04075) | iterative joint decoding | yes (3 refinement steps) | the whole frame | latents trained with reconstruction |
| explicit chance + deterministic completion (Stochastic MuZero; Delta-IRIS; our E22MS) | a chance code | lives in g | sized to the chance | see E22MS below |

**E22MS was the right shape and failed for three measured reasons:**
- its channel absorbed epistemic terrain until masked (E22FJ: 92% of the gain);
- with the mask, its posterior captures 86% of random moves, i.e. c is insufficient. That the leftover is averaged into ghosts
  is an **inference**, not a measurement;
- g has no conservation structure: 10% per step source residue even with the TRUE codes (cowdiag).

**Coherence lives in g.** A cow that moves must leave its source. One-to-one assignment (ITC) is one structural way to get it.

---

## 4. Memory: what it must hold and how it is addressed

**What it must hold:**
- the counters of row 2: a handful of bounded integers with observable resets and cycles ≤ 31 steps;
- a branch's own sampled epistemic terrain (row 4) once revealed;
- the branch's own chance outcomes.

**Addressing, from our own records:**
- **Screen-slot memory:** E17, two seeds. Window 5 → 15 lifts same-slot recall +0.25 / +0.28 and moved-slot recall only
  +0.01. Memory tied to screen slots does not follow content across a scroll.
- **World-aligned memory:** fcanvas / E18 failed through its **read path**. Writes were world-aligned, but the next frame's
  prediction still read the screen slot. The entering cell's remembered stream was read 0 / 2,018 times, and same-slot recall
  fell 0.86 → 0.34.
- **The literature's answer:** Neural Map (Parisotto & Salakhutdinov) separates addressed writes from reads. A world-indexed
  memory needs an explicit read at the next view's address.
- **The 09-29 motion-aligned carry proposal** (transport `conv_state` / `ssm_state` along the correspondence) was nearly
  equivalent to fcanvas on short windows and 4.9× slower; it was not trained.

**In imagination the memory address is the world's own scroll decision,** which is the one discrete instability we
measured. The chaos test: deterministic worlds contract, but 9-12% of offsets flip at σ 0.1, and 24-36% of scrolls are missed
at depth 16. A wrong scroll corrupts both the view and the memory address. *(Connection, not measured.)*

---

## 5. Representation, and the reconstruction question

**The trade-off:**
- **Reconstruction (observation prediction)** guarantees retention, but it must model the nuisance (≈ 25% of the change mass)
  and every texture. EMERALD and Dedieu pay that cost.
- **End-to-end predictive latents:** JEDI beat its separately trained latents (Horizon Imagination) where directly
  comparable, and reports that the predictive latents "change more than compute alone".

**The real obligation of reconstruction-free.** Retain content whose value shows only through rare reward (Ni et al.: sparse
reward → observation prediction). JEPA / SIGReg optima favour slow features (Klindt / LeCun / Balestriero 2605.26379, per our
09-25 notes), and our CLS encoder did drop mobs.

What we measured that retains content without pixels:
- **locality:** the per-tile state kept the zombie (+0.080 over CLS, +0.075 on zombie roots; readout ladder, sealed);
- **consequence gradients in the loss:**
  - E21 oracle: a label gradient installed the cooldown phase (1.000) within 1,000 updates;
  - Dreamer-CDP 16.2 vs 12.7 without the reward gradient;
  - JEDI's reward / termination heads train its encoder.

**Position.** No reconstruction, with the obligation met by named means:
- locality, so each tile's content cannot be traded for global slow features;
- consequence heads in the loss;
- the predictive objective's own bias against unpredictable nuisance (Littwin et al. 2024; VJEPA's noisy-TV toy, which is
  only a toy).

Whether that suffices for fast, sparse-reward content (mobs) is **open**, and checkable on our encoders with our existing
probes.

**Separating content from meaning** (Codex's point; it also answers the user's 10-10 objection):
- a stable, local target for *what is visible in this tile*;
- contextual reasoning (spatial attention + Mamba) for *what it means and what happens next*.

Context belongs in the belief and the dynamics, not in the target. The 09-25 categorical attempt failed partly here: 80.7% of
its code changes were contextual flips (plus quantization error and no copy path).

---

## 6. Decisions

- **Decisions need sampled futures** (check_h16_value): one faithful future 0.663, 16 faithful futures 0.727. D (0.670) ≈ one
  future. E22MS's 16 samples tie D (0.663) at 16× the cost.
- **Calibration is not discrimination.** E22MS has the best Brier score (0.050) but not the best choice among a root's 17
  actions.
- **Two different heads with different supervision:** a branch-outcome head reads the sampled successor and memory; a
  root-risk head reads the belief. The terminal position must not be a cue (our row-14 artifact).
- **H16 sees HUD-driven risk only.** Navigation, resources and crafting go through the map, so the actor against BC is the real
  test.

---

## 7. The two reviews, checked

**Codex**

| claim | verdict |
|---|---|
| Dreamer-CDP's 6.3 removes the KL alignment, not the categorical latent | **right**: I misread it (paper l.229) |
| categorical targets were not shown to be load-bearing | **right** |
| independent per-tile sampling is incoherent (the cow counterexample) | **right**: 57% wrong count with Craftax's real movement distribution; ITC measures it |
| CE bounds logit gradients, not update shares | **right**: my "~4% of the gradient" is withdrawn |
| the ghost-copy cause was labelled inference | **right** |
| separate a stable content target from contextual belief | **agree** |
| memory should follow the evidence; correspondence-carry | agree in principle; our canvas / E18 record says the **read path** is the crux |
| pretrain a tokenizer with a reconstruction decoder | **disagree**: it brings back the nuisance (≈25% of change mass), Ni et al. favour self-prediction for noisy tasks, JEDI's end-to-end latents beat separate ones, and it dissolves the thesis |
| preferred realization: a spatial categorical generator with refinement (EMERALD-style) | **disagree as the default**: it pays generative compute on the whole frame while the chance is ≲ 1.3 bits per step, and EMERALD's latents are reconstruction-trained |

**Third party**

| claim | verdict |
|---|---|
| belief / transition / decision separation; the Jensen mismatch | **right**: formally Ni et al.'s ZP vs EZP |
| four kinds of uncertainty need different treatments | **right**, and our data size each one (table §2); E22FJ is direct evidence |
| timers need memory, not randomness | **right**, confirmed from source: deterministic counters with observable resets |
| "generate innovations, not states" | **right**, and it has a formal precedent they missed: Stochastic MuZero |
| stochastic reconstruction-free JEPA world models exist | **true**: JEDI verified. But Flow-JEPA's tasks are near-deterministic, VJEPA's experiment is a toy, and Fast-LeWM is deterministic. None is tested with rare, decision-critical local chance plus hidden counters plus long imagination |
| investigations A / B / C | mostly answerable from data we have: B measured (1.71 cells per step, all mobs); A answered from source; C answered for H16 (region substitution, check_h16_value), open for an actor |

---

## 8. The research question

> In environments whose dynamics are mostly deterministic *given a sufficient memory*, with sparse local chance (Craftax:
> ≲ 1.3 bits per step) and large nuisance and epistemic variation, can a reconstruction-free world model:
> - **represent the chance explicitly and compactly**, so the deterministic part is predicted (a point loss is then correct)
>   and the chance is sampled coherently;
> - **discard the nuisance;**
> - **remember what history determines;**
> - **sample epistemic content once and persist it,**
>
> making imagination both faithful where decisions need it and cheap enough to train an actor on 6 GB?

**The Dreamer-4-type lever this points at:** generative compute proportional to the entropy of the chance, not the size of the
frame. A continuous diffusion world (JEDI) pays denoising on every dimension every step. Here the random part is about one
mob move.

**Novelty status: unestablished.** Nearest found:
- Stochastic MuZero: value-equivalent, board games, global chance, no state self-prediction, no nuisance / epistemic split;
- Delta-IRIS: reconstruction, Δ tokens;
- JEDI: whole-latent diffusion;
- ITC: coherent assignment over pixel codes;
- Discrete-JEPA 2506.14373: a global VQ token;
- DWMR 2603.01748: a global Boolean state, deterministic puzzles.

Nothing found combines a reconstruction-free self-predictive state, an explicit chance variable sized to the aleatoric part,
structural separation of the four rows, and world-addressed memory. A proper sweep must come before any claim.

---

## 9. Open questions (not a plan)

1. **Separating the rows without labels.** Our only success used a moved-content mask. Candidate structural signals:
   - **persistence:** epistemic content stays once revealed; aleatoric does not;
   - **memory-resolvability:** predictable from a longer history;
   - **consequence:** changes reward / transition (bisimulation);
   - **location:** enters at the view edge vs changes in the interior.
2. **Generality of the chance variable.** Does a sufficient chance variable stay small outside Craftax, and how should its
   granularity be chosen (per object, per region, global)?
3. **Reconstruction-free retention** of fast, sparse-reward content: are locality + consequence heads enough?
4. **Memory addressing in imagination,** where the address is the model's own scroll decision.
5. **Decision cost:** how many sampled futures a decision needs, and whether an agent can get the benefit without paying for
   16 rollouts per action.

Sources: Ni et al. arXiv 2401.08898; Antonoglou et al. ICLR 2022 (Stochastic MuZero; read via the 2048 reimplementation
arXiv 2212.11087); Zhang et al. arXiv 2006.10742 (DBC); Lim et al. arXiv 2605.13013 (JEDI); arXiv 2608.29029 (Flow-JEPA);
Huang arXiv 2601.14354 (VJEPA); Gao & Xu arXiv 2606.26217 (Fast-LeWM); Dedieu et al. arXiv 2502.01591; ITC arXiv 2605.16457;
EMERALD arXiv 2507.04075; Dreamer-CDP arXiv 2603.07083; Discrete-JEPA arXiv 2506.14373; DWMR arXiv 2603.01748; Parisotto &
Salakhutdinov arXiv 1702.08360 (Neural Map).
