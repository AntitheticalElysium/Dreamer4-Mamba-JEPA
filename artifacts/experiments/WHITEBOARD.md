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

**Our worlds sit at EZP at best, and use it as a state.** The per-element L2 optimum IS the expected next embedding, so
MSE-trained worlds approximate EZP; the L1 optimum (the coordinatewise median) is not even that. The failure is treating
E[z′] as a concrete future and feeding it back or applying a nonlinear head to it (corrected 10-12 after Codex's review; the
first version said "below EZP"). statoracle (NOTEBOOK 10-10 11:40):
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
| **2. Forgotten history** | hunger / thirst / fatigue / recover counters, per-zombie attack cooldown | every one of 8,183 intrinsic decrements has its reset ≥ 20 steps back; periods 26 / 21 / 31 steps awake (hunger / thirst / fatigue; hunger and thirst halve their rate asleep, giving 52 / 42), recover crosses at +25 / −15; the window is 15 frames and imagination rebuilds the state from ≤ 15 frames every step | **memory** | **yes**: the HUD carries 90-100% of the decision gap; imagined food / drink / energy decrements at 0-22% of the real rate across the deterministic worlds; imagined depth-16 health 5.5-6.0 vs 2.4 | no loss can fix it: the information is not in the input |
| **3. Aleatoric** | cow 50% stay / 4 × 12.5%; zombie within 10 cells 75% chase, else a uniform move; spawns; skeleton arrows | **1.71 cells per step** differ across RNG keys from the same state, all of them mobs; movement-only estimate ≈ **1.3 bits per step** (cow 2 bits; near zombie 0.99 bits, 1.54 with tied chase axes); omits spawns, off-screen changes, the 10% sapling draw (inventory differs across keys in 4.2% of depth-1 cases), and conditions on the FULL state, not on the agent's history (see §10) | a **conditional distribution, sampled coherently** | **yes, indirectly**: damage is 97% deterministic given the state, but who becomes adjacent is random; the true map near zombies restores 58.5% of imagined damage; 16 faithful futures lift H16 0.670 → 0.727 | L1 erases (median), MSE superimposes (mean); E22MS draws hits at the real rate but duplicates mobs (10% per step vs 1% real) |
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
reconstruction 1.1% → 86%). That mask is computed from the observed transition and applied as a loss WEIGHT (`dworld.py`),
so "structure, not weight" (first version) misstated it. The lesson that survives: **a generic, label-free signal that tells
the rows apart redirected learning where loss reweighting by error magnitude did not.**

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

**In Craftax the visible chance looks small** (1.71 cells per step, all mobs; the bit estimate is movement-only and full-state-conditioned). Everything else is deterministic given memory (row 2),
discardable (row 1), or epistemic (row 4: sampled once, when revealed).

| family | randomness carried by | coherence | where it pays generative compute | evidence |
|---|---|---|---|---|
| continuous diffusion / flow over the whole latent (JEDI 2605.13013, Flow-JEPA 2608.29029, Dreamer 4) | noise in every dimension | in principle | everywhere, including the ~97% of cells whose next state is deterministic (1.71 of 63 differ across keys) | JEDI: reconstruction-free and end-to-end, Atari100k + Craftium, 38 A100-h per run. Flow-JEPA: near-deterministic control tasks, a robustness gain (its own limitations note it does not isolate stochastic training) |
| per-tile categorical, independent readouts (Dedieu 2502.01591) | per tile | **breaks**: perfect marginals for one free cow give the wrong count 57% of the time; ITC (2605.16457) measures the duplication on Craftax and a one-to-one assignment lifts return 67.4 → 72.5% | per tile | pixel nearest-neighbour codes |
| spatial categorical + MaskGIT refinement (EMERALD 2507.04075) | iterative joint decoding | yes (3 refinement steps) | the whole frame | latents trained with reconstruction |
| explicit chance + deterministic completion (Stochastic MuZero; Delta-IRIS; our E22MS) | a chance code | lives in g | sized to the chance | see E22MS below |

**E22MS was the right shape and failed for three measured reasons:**
- its channel absorbed epistemic terrain until masked (E22FJ: 92% of the gain);
- with the mask, its posterior reconstructs 86% of random moves. That shows the complete conditional path fails on the rest;
  it does not show the code is insufficient (decoder capacity or optimization could explain it), and that the leftover is
  averaged into ghosts is an **inference**, not a measurement;
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
- **Reconstruction (observation prediction)** biases toward retention (practical reconstruction can still drop rare details), but it must model the nuisance (≈ 25% of the change mass)
  and every texture. EMERALD and Dedieu pay that cost.
- **End-to-end predictive latents:** JEDI beat its separately trained latents (Horizon Imagination) where directly
  comparable, and reports that the predictive latents "change more than compute alone".

**The real obligation of reconstruction-free.** Retain content whose value shows only through rare reward (Ni et al.: sparse
reward → observation prediction). JEPA / SIGReg optima favour slow features (Klindt / LeCun / Balestriero 2605.26379, per our
09-25 notes), and our CLS encoder did drop mobs.

What we measured that retains content without pixels:
- **locality (a bias, not a guarantee; not isolated as the cause):** the per-tile state exposed the zombie where CLS did not (+0.080, +0.075 on zombie roots; readout ladder, sealed);
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

- **Integrating over futures has decision value** (check_h16_value, 7,920 FIT + DEV roots: one real future 0.663; 16
  simulated Bernoulli draws from the true death probabilities 0.727). That estimator differs from the one that scores D
  (0.670, DEV-B), so the two are NOT directly comparable (corrected 10-12). A learned risk / value function can amortize the
  integration; it need not mean 16 rollouts per action. E22MS's 16 samples tie D on the same estimator (0.663 vs 0.670).
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

---

## 10. Codex's 10-12 review: accepted corrections (applied above) and two refinements

- **Corrections applied:**
  - an MSE world approximates EZP; the failure is using E[z′] as a state;
  - counter periods double asleep;
  - the bit estimate is movement-only and full-state-conditioned;
  - the moved-content mask is a loss weight;
  - 86% does not prove an insufficient code;
  - 0.663 → 0.727 is a different estimator from D's 0.670;
  - reconstruction biases toward retention rather than guaranteeing it;
  - locality was not isolated.
- **The precise regression condition:** Var(z_{t+1} | afterstate, c) = 0. Capturing the simulator's RNG is not enough while the
  model still lacks the cooldown phase or the terrain.
- **Fresh randomness vs state uncertainty:** Var(Y | H, a) = E[Var(Y | S, a) | H, a] + Var(E[Y | S, a] | H, a). Forks from a
  simulator state measure only the first term. History recovery addresses the second.
- **Memory must hold uncertainty, not only facts.** Starting mid-episode, clipped invisible resets can leave several phases
  possible; a branch must keep its own sampled hypothesis and terrain consistently.
- **The four rows are operations on state** (remember, infer, sample, persist, discard), not four independent channels: the
  same zombie carries a remembered cooldown, a random move and action-dependent consequences.
- **The efficiency hypothesis kept:** compute deterministic context once and reuse it while sampling compact joint
  innovations. "Compute proportional to entropy" is stronger and unestablished: one random bit can trigger consequences
  across a large state.

---

## 11. Feasibility study (10-12; NOTEBOOK "10-12 FEASIBILITY RESULT"): the hypothesis, tested in the simulator

- **History resolves the counters.** At our 15-frame window, 47-68% of next-16-step HUD schedules are determined; at 64 frames,
  100% for hunger / thirst / fatigue. Recover needs the full episode (78% at 64): it is invisible at full health, so it is
  counted from the last visible event. Memory, not randomness, as the third party argued.
- **The fresh randomness is compact per entity.** About 1.37 bits per visible mob, near-independent across mobs (total
  correlation 3.8% of the sum of marginals). 0.33 bits with no mob in view. It barely depends on the action. As a frame-level
  joint code it is NOT compact (p90 4 bits, long combinatorial tails). **Chance variables belong to entities.**
- **Coherence is first-order for decisions; sampling is second-order but real.**
  - A coherent deterministic mode-world gets 75% (survival) and 88% (return) of the value of 32 sampled futures.
  - 16 samples add +0.025 [+0.009, +0.045].
  - **One sample is worse than the mode-world** (−0.038), which is the E22MS-vs-D pattern seen in the simulator itself.

---

## 12. 2026-10-14/15 — what the transport, objective and recursion studies measured (NOTEBOOK: TRANSPORT, OBJECTIVE, R1-R12)

**Row 3: chance as reassignment of retained content.** One stochastic operator that only permutes adjacent cells (an
exact matching) beats per-tile synthesis at every seed: valid 93-100% vs 53-65%, cow movement 25-43% vs 4-8%. Free
synthesis explodes under recursion (×3-12 mobs, 4.5 phantoms per frame); transport conserves.

**Who owns the chance decides the afterstate.**
- With the outcome supplied, the deterministic reference becomes a clean pre-chance afterstate (F1b).
- Trained toward the post-chance target, it ghosts (the mean).
- 31% of true mob moves (zombie chase) are predicted by the afterstate itself and need no chance.

**Objective: the energy score hedges deterministic error.** For draws that differ only by a one-cell reassignment, the
two-sample first-power score cannot penalize the change unless the afterstate is exact (triangle inequality):
- dose-response at both seeds;
- local scores make it worse.

**Inferred chance** (Stochastic MuZero's split: point loss given a label-free posterior reassignment, log-score on the
chance head) removes the hedge.
- Spurious terrain swaps fall 100-280×; one-step validity is 100%; movement and dependence reach the real rates.
- **The posterior must be Bayesian under the model's own prior.**
  - Maximum likelihood accepts look-alike swaps (precision 0.25 in a synthetic test).
  - A global prior self-tightens to "nothing moves".
  - The per-pair prior of the chance head works.
- **Remaining failure:** where the posterior is blind (crowded mobs, chase combined with a scroll), the point loss averages,
  leaving ghosts at a few roots and faster mob fading in recursion.
- **Final IC measurements:**
  - The label-free posterior converged (89% of genuine chance moves, ~95% precision, both seeds).
  - Under IC, intact zombies chase at the real rate.
  - Supplying the true chance outcome every step does **not** stop fading: it lives in the deterministic dynamics iterated on
    their own outputs.

**Row 1: nuisance is not a context problem.**
- Persistent content's token drifts because Craftax re-renders it every step (daylight blend, deterministic and global) and
  because position codes change with scrolls.
- The drift is equal in the encoder's purely local layer 0 and in Raw.
- It is not a linear subspace, so no fixed projection or loss metric discards it.
- ~~Deterministic rendering must be *predicted*~~ (corrected 10-15 evening, `render_predictability.py`): the rendering change
  is not predictable from history. The previous step's change explains −120% of the day drift energy (worse than predicting
  no change); linear extrapolation explains −471%. Daylight reaches the 8-bit frames as intermittent quantization steps.
  Deterministic in principle, unpredictable from observation in practice: treat it, like night static, as nuisance requiring
  *invariance*, not prediction.
- Mob content is perfectly local at every layer. Mob-over-terrain occlusion makes 5% of real moves non-transferable at
  every layer.

**Rows 2 / 4: memory is addressed by screen slot.** Revisited terrain is recalled well after ≤ 2 scrolls (10-23% error) and
lost after 3+ (35-41%), at the same age. Terrain seen 8-14 frames earlier is recalled no better than never-seen terrain.
The information is in the window; the address moves.

**Recursion and decisions.** Every model's deterministic dynamics degrade when they read their own frames:
- **Mob tokens attenuate geometrically:** a soft blend is not an idempotent copy.
- **Scroll decisions flip:** wrong imagined content at the faced cell, 37-81% divergence vs 6%.
- **Zombie pursuit decays:** 38% real vs 10-19% by depth 6-16. Faint zombies chase less, and crisp ones also under-chase.

Consequence: in imagination the world stops threatening the player. Zombie adjacency stays flat while reality's rises
0.03 → 0.10, and health drops vanish. Stochastic coherence (row 3) was not the decision bottleneck; recursion-stable
deterministic dynamics and HUD consequences (row 2) are.

**Candidate general mechanisms, each now motivated by a measurement (none yet tested as a whole):**
**Track B, 10-15 evening (NOTEBOOK R17-R20):**
- A world-aligned exact cache reads revisited terrain at 0.999. E18 failed by its read *address*, which a hard scroll
  decision supplies.
- PI's edit gate is EM for the edit prior. At initialization the label-free evidence already says persist (unchanged
  terrain / mobs, prior ~0) and edit where content changes (0.78-0.91).
- **A no-op world passes every recursion persistence reading.** Only movement and decision readings separate dynamics from
  copying (R19).
- IC's imagined mobs under-move because they are faint (R20).

1. **One reassignment operator, three uses:**
   - the deterministic camera translation (action);
   - the stochastic local swaps (chance);
   - **the same translation applied to the recurrent memory**, so that memory moves with content.
2. **Exact, idempotent persistence** instead of soft blends that decay when iterated. Measured 10-15 evening
   (`persistence_variants.py`): exact copies keep persisting-mob scores at 0.76-0.83 at depth 16 vs 0.37-0.50 for the soft
   blend, so fading is structural. But CorrT's argmax is not a trustworthy persistence *decision*: terrain error and
   phantoms rise. The open problem is deciding what persists.
3. ~~**Chance-forced self-fed training**~~ — **tested 10-15 evening (TRANSPORT_SF_PREREG, NOTEBOOK R21), not the fix.**
   The persisting score moved +0.005 / +0.058 against a ~0.4 structural gap, and the census effect flipped sign across
   seeds. The self-fed loss did not fall with training. Exposure cannot teach a soft blend to stop regressing toward the
   mean, because the blend *is* the mean. The persistence mechanism must change (item 2 → Track B's PI world).
4. **A Bayesian posterior under the model's own prior** as the only learned-chance signal: one rule that keeps identity
   unless evidence pays for a change.

## 13. 2026-10-15 night — the persistent-state principle, tested (NOTEBOOK R17-R28)

**The principle under test:** keep a persistent world state, and change it only through explicit, inferred transformations,
each a hard decision that must pay its description cost:
- the camera scroll (from the action);
- chance swaps (the IC posterior under the chance head's own prior);
- deterministic edits (an EM-gated per-cell edit).

**Results:**
- **Track A** (R21): exposing CorrT's soft blend to its own predictions does not fix fading (+0.005 / +0.058 on a ~0.4 gap).
  The blend is the mean.
- **Track B, PI** (R22-R23): **supported at both seeds, as preregistered, and not the copy shortcut** (R19 guards pass).
  - Persisting-mob score 0.37 → 0.71-0.73; zombie census ×7; terrain and phantoms unchanged.
  - Joint ES beats IC and free synthesis under both readers. Per-step vanish rates are at the real level.
  - The label-free EM gate becomes a change detector (AUC 0.93 terrain, 0.98 HUD). It persists 98% of unchanged content and
    edits 84% of changed terrain.
  - Its generated token learns HUD consequences (wins at 95% of real HUD changes, from 52%).
- **What PI does not do:**
  - **Arrivals** (R24): new mobs entering the view are never drawn, 0-0.02 vs real 0.12-0.26 per frame. Its new content is a
    point prediction, the mean.
  - **Pursuit** (R27, corrected by R33): PI's chance head chases close to reality on real contexts (toward 0.25-0.41 vs
    real 0.35-0.45; IC's chance head 0.09-0.18). Its imagined pursuit decays with depth (0.29 → 0.22 vs real 0.38): a
    recursive effect, not fading and not one-step.
  - **Health drops** are seed-inconsistent.

**Two evaluation-only variants, each one change:**
- **World-aligned canvas** (R26): revisited terrain error 14% → 0.1%. Addressed memory works once the read uses the world's own
  next-view address, which was E18's defect. But it replays stale mobs: off-screen persistence must follow the content's own
  volatility.
- **Sampled edit decisions** (R28): ghosts (phantoms ×4-5, vanish ×2.5). Sampling *whether* a cell changes while filling it
  with the *mean* of what it could become is worse than either alone.

**The separation the user asked for, as it now stands:**

| part | treatment | status |
|---|---|---|
| stable content | exact copy by default, under an explicit camera transformation | works (PI) |
| contextual belief | the screen-slot Mamba belief drives the decisions | works for one-step decisions; memory needs world addressing (canvas works for terrain) |
| deterministic evolution | EM-gated hard edits, content = gen | works where predictable (84% of changed terrain, 78% of HUD changes) |
| stochastic reassignment | chance swaps with an inferred posterior | works (92% capture); PI's chance head carries the chase near the real rate one-step (R33), but pursuit decays with depth in recursion |
| stochastic innovation | new content **drawn** from a learned categorical (K 4096 codes) **only where the uncertainty is epistemic** (never-seen entering cells) | **works** (R32): arrivals ~0 → 0.4-0.56× real, no harm, camera +3pp. Sampling everywhere (R29) or sampling decisions with mean content (R28) fails |

**Innovation rule (R28-R32):**
- A change and its content are one random event, and each half must be calibrated where it is applied.
  - Sampling the decision with mean content gives ghosts (R28).
  - Sampling content at deterministic or gate-fired sites gives terrain noise and zombies from nothing (R29, R31).
  - Sampling content at epistemic sites only works (R32).
- **Still open:**
  - interior spawns;
  - mob volatility in memory;
  - arrival rates below reality;
  - pursuit direction;
  - health consequences.

## 14. 2026-10-15 late — persist information, not representation (user question; NOTEBOOK R34-R38)

**User's concern:** a latent need not represent a fixed object. A zombie moving closer changes its significance, so exact
token copying may preserve stale context.

**Measured:**
- **The belief updates context correctly when its inputs are fresh** (R35). The next health drop beside a zombie is predicted
  with AUC 0.97-0.98, best for zombies that just changed slot. Slot-wise Mamba memory does not freeze temporal context
  within the window.
- **Exact copies freeze the representation** (R34). With perfect transformations and only stale encodings:
  - content-level decisions survive (change detection AUC ≥ 0.97, scroll 0.99);
  - the relational reading (chase toward the player, −30 to −64%) and consequence content (+25 to +62% error) degrade;
  - this is the mechanism behind PI's pursuit decay in recursion (R33).
- **What the frozen Raw token binds to content:** global rendering (unpredictable steps, R15), a content-dependent position
  code (R36: not additive), and the mob's background (R37: swaps drag terrain, 0.5-5% of moves).
- **Two repairs inside the current representation fail:**
  - re-deriving an additive position code (R36: no change);
  - learning invariance by stale-input augmentation (R38: the belief hedges, and pursuit gets worse).

**Principle (the user's):** persist the information, not its representation or interpretation.
- In this pipeline that means the persisted state must be content-level and context-free.
- Context (position, rendering, relations, consequences) is recomputed each step by the model from where and when the content
  now is.
- Our frozen contextual ViT tokens cannot serve as that state. The persistent-state principle (PI) is right about *what*
  persists, and wrong to persist the token.
- **Open design question for the user:** the persisted state's representation. Options include a local, position-free
  content encoder, or context-free per-tile codes (Dedieu et al.), with the encoder-side change it implies.

## 15. 2026-10-16 — what remains invariant, what is transformed, what must be recomputed (NOTEBOOK R39-R42; user's guiding question)

Measured with **exact** transformations (diagnosis futures; R41 fixed my construction defect; R42 holds the corrected numbers).

| aspect of persisting content | what the evidence says | treatment |
|---|---|---|
| **identity** (tile class, mob type) | exact copies keep it. PI's label-free gate learns which cells persist, including that the cell the player vacates must be regenerated (gate 0.996, R41) | **invariant:** persist exactly |
| **position** (scroll, chance moves) | exact transport is correct. A stale *position code* in the copied token is nearly harmless (pixel-exact windows: content +3-12%, belief cosine 0.998), and recomputing context through the encoder's upper layers from position-free content codes removes it exactly (R42) | **transformed:** explicit transformation; context recomputed (cheap, exact) |
| **rendering** (global lighting) | the dominant harm. Under lighting events, consequence content error more than doubles and chase drops 16-18% (R42). The content code is linear in pixels, so it carries the light | **must be recomputed, and nothing tested does it:** global affine relight (R39), learned relight MLP (R41), normalization / learned invariance (R39) all fail |
| **relations / consequences** | carried by the *current* frame's representation, not by history (a fresh current frame alone recovers 87-97%; R42). On fresh inputs the belief computes them correctly, including temporal context after slot changes (R35) | **recomputed by the belief** from a current, consistent frame |
| **occluded content** (under the player / mobs) | not in the current observation. PI regenerates the vacated cell. Literal swaps drag mob backgrounds (0.5-5% of moves, never repaired; R37) | memory / uncertainty, not transport |

**Reading:**
- The persistent-state principle is right about identity and position.
- The open problem is narrower than "context": it is the frame-global *rendering* that every copied cell carries. Rendering is
  nuisance for content, but the belief's readouts depend on it being current.
- Candidate treatments, none validated:
  1. encoder-level invariance to lighting, learned during JEPA training;
  2. a belief whose readout renders every cell from one state (LB; the global-RSSM baseline, see §16);
  3. a properly learned renderer conditioned on a carried lighting state.
- **Context-free representation is not a requirement** (user, Codex): content codes are already spatially equivariant
  (camera drift 0.007 vs 0.086) and lose nothing, but they do not remove lighting.

**Addendum (R43, 10-16):**
- The Craftax source makes rendering a deterministic light clock by day (random static at night). Even with the true light
  level, a copied token cannot be re-rendered (8-bit quantization flips), so at the token level rendering is practically
  nuisance.
- **Caution on R42's magnitude:** its substitution mixes freshly-lit changed cells with stale-lit copies in one frame. In PI's
  own imagination nothing is freshly observed: copies carry the root's light, and generated cells are predicted from a
  root-lit context.
- So imagined frames are largely *consistently* lit at an earlier time. R42 measures the harm of *mixed* lighting (real when
  imagination is interleaved with observation). It does not establish that pure imagination suffers it.
- Not yet tested directly. A time-frozen frame (all cells at the root's light) needs a renderer.

**Addendum (R47-R48, 10-16): appearance is a fourth kind of transformation, and it must be shared.**

| aspect | evidence | treatment |
|---|---|---|
| **global appearance** (sleep grey-out; daylight by day) | the source renders both as one per-pixel colour map. On the pixel-linear content code it is an exact affine map. **One 12-parameter colour map shared by every pixel**, fitted label-free, keeps 97-98% of zombies and cows through the sleep onset. A free operator (192 × 192, content code or token) learns the step's conditional mean and erases rare content (zombies 0.51-0.64). FiLM on tokens fails outright | **transformed by a global, low-dimensional, shared operator**: the bulk of the scene determines it, and it cannot encode content-specific change |

- The table now reads:
  - identity **persists**;
  - position and appearance are **transformed** by global operators (one displacement; one colour map);
  - local change is a **gated edit**;
  - relations / consequences are **recomputed** by the belief.
- Night static is random and vignette-shaped: nuisance, not an operator.
- **Decision value (R48):** emulating PI plus this operator at the sleep onset improves DV16 decisions at both seeds (0.355 →
  0.492, 0.478 → 0.522).
- **The catch:** the 12-parameter operator exists only in pixel-linear coordinates (the frozen first layer, inverted exactly).
  On Raw tokens no shared affine map preserves content.
- **Open design choice for the user:**
  1. persist copies as content codes (pixel-equivalent), recontextualize them with the frozen upper layers, and transform
     them with shared operators. This is exact, but the world then holds and edits pixel-equivalent state: a step toward
     reconstruction;
  2. learn an encoder in which global photometric changes act as a shared low-dimensional operator (an equivariance
     objective), staying latent;
  3. leave appearance to regeneration and accept the erasure.
- **R49 adds a measured fourth option:** a latent operator on the frozen Raw tokens, Image-World-Model style (conditioned on 12
  numbers, trained only on synthetic recolourings, θ inferable label-free).
  - It keeps 93-96% of mobs through the sleep onset and gives the same DV16 decision gain as option 1.
  - Terrain is 0.95-0.96, below the declared 0.98.
  - **It is not compositional:** repeated application compounds its error (16 identity applications: zombies 0.26).
  - So it can only serve as a gated, one-shot event transform (sleep onset / wake-up), never as per-step relighting.
  - Option 1 composes exactly.
- **Weight of the choice:** fixing appearance moves PI's decisions only to ~0.49-0.53. The sleep-excluded ceiling is 0.57, and
  the larger lever is the consequence hedge (§16).

## 16. 2026-10-16 — selective persistence vs an ordinary latent belief (user's sharpened question; NOTEBOOK R44)

- A standard reconstruction-free RSSM (R2-Dreamer's, unchanged, on our JEPA tokens; 60k updates, two seeds) **does not encode
  mobs at all**: posterior readout mob recall 0. It loses terrain (21% error at d16) and fails every one-step condition.
- PI wins every preregistered reading, but against this baseline that **does not answer** the question.
- Still open: does selective persistence beat an evolving latent belief **that represents the scene**? The fair test is a
  latent belief on the same spatial scaffold.
- **What is established on the shared scaffold** (D → IC → PI):
  - hard selective persistence + inferred chance doubles mob persistence;
  - it keeps one-step coherence;
  - it improves food / hazard decision proxies.
- Whether that improves *decisions*: DV16, running.

**DV16 (R45): decision value.**
- Imagined-risk action choice over 17 first actions is dominated by a **sleep blind spot**. Sleep is deadly (P16 0.86); PI
  imagines it as safest and picks it at 24-43% of roots, falling below uniform.
- With sleep excluded, every world beats uniform by ~0.03-0.05 and **PI ≈ IC**: persistence does not improve decisions.
- **Binding constraint for useful imagination: the consequences of rare, high-stakes actions** (health / damage). This
  outranks representation staleness, and it is the row-2 / row-3 consequence problem first diagnosed in E14.

**Localized (R46-R48).** The sleep blind spot has three parts:
1. **Onset erasure** (no appearance operator; fixable, see §15 addendum). It is about 45-65% of the decision loss.
2. **Damage magnitude hedged** to ~45-70% of real over 16 steps, awake as well as asleep. The HUD edit draws the conditional
   mean of the consequence.
3. **Early timing:** the onset step's mob moves.

PI's Mamba belief *does* know the mechanism: the sleeper-attack rule and the zombie's hidden cooldown, read from history
(0.93 / 0.82 vs real 0.94 in the cell the behaviour data rarely shows). So memory is not the gap. **The gaps are
representational (appearance) and distributional (magnitude = mean of consequences).**

**The sharpened question on decisions (R50, secondary).** PI vs the D parent (the regenerating Mamba belief on the same
scaffold), paired over DV16's roots, sleep excluded:
- **HD: no difference** (−0.011 [−0.034, +0.009] / −0.001 [−0.023, +0.018]);
- **HL: PI worse** (−0.03, resolved at both seeds);
- **ZA: PI slightly better** (+0.02 / +0.01).

**Selective persistence buys coherence (mob persistence ×2, census ×7), not decision value on this test.** The binding
constraint for useful imagination is consequence fidelity: the magnitude hedge and appearance events. Longer horizons and
actor training remain untested.

**What it costs (R51).** On the escape choice (noop + 4 moves), imagined health loss ranks the moves at within-root AUC:
- D 0.73;
- IC 0.68;
- PI 0.58-0.63 (resolved vs D at both seeds; not sampling noise, not the HUD gate).

Teacher-forced, PI's gate under-calls uncertain attacks (awake recall 0.58 / 0.81 vs D 0.90) and shrinks them; D predicts the
graded expectation.

**The MAP copy/edit decision that makes the scene coherent also discards graded risk.**

| content | treatment the evidence supports |
|---|---|
| scene (terrain, mobs) | exact persistence + transformations (R23, R47) |
| consequences (body state on the HUD) | graded expectation from the belief, or a calibrated sample; never a copy/edit MAP decision |

**Hybrid tested (R52, evaluation only):** PI's scene with D's HUD readout in lockstep does not keep both advantages.
- The damage ranking rises only to 0.62 / 0.66 (D 0.73).
- D's own readout ranks worse on PI's scene than on its own.

So most of the cost sits in the persisted scene's *dynamics* (candidate: weak pursuit, R23), not only in the copy/edit
decision on consequences.

## 17. 2026-10-16 — how persistence, stochastic dynamics and decision-relevant risk coexist (user direction; NOTEBOOK R53-R54)

**User decision.**
- Pixel-equivalent content codes and the one-shot appearance operator are controls only.
- Persistence is established. The question is how it coexists with stochastic dynamics and decision-relevant uncertainty.
- A simpler evolving belief is welcome if it does better.

**Where decision-relevant risk lives (R53).**
- In DV16's escape choice, one real future's first-step damage ranks the moves at 0.71.
- Its later damage ranks them at 0.46: most real damage comes later and is action-independent at this horizon.
- The action-specific part is the immediate consequence. The rest matters only as an expectation over many futures.

**What each world gets right:**

| | scene dynamics (adjacency kept, pursuit, hit rate given adjacency) | immediate consequence (first-step damage ranking) |
|---|---|---|
| D (regenerating belief) | poor: zombies vanish (0.02 at d16), pursuit 0.05, hits 0.05 | **0.711** (graded expectation, anchored on the current token) |
| PI (persistent scene) | **best**: 0.35-0.40, 0.25-0.30, 0.12-0.14 | 0.60-0.62 (the MAP copy/edit gate under-calls) |
| PI's *belief*, read as an expectation (R54b) | (same scene) | **0.70-0.71** |

**Resolution so far:**
- Persistence and decision-relevant risk do not conflict in the scene: PI's scene is the most action-informative.
- They conflict only in how consequences are **produced** (a MAP edit of an observation token) and **read** (a few-sample
  maximum over a realistic, noisy future).
- Read as expectations from the belief, PI's consequences rank the moves +0.08 to +0.11 better than its own rollouts and near
  D's rollout, without imagining a frame.

**Working principle (to be tested in a trained model):**
- persist the scene, with transformations and sampled chance;
- read consequences from the evolving belief as graded expectations or calibrated distributions, never as MAP edits;
- this is the Dreamer pattern (reward / continuation heads on imagined latents), with label-free targets on the observation's
  own consequence channel.

**Open, and now the binding problem:** counterfactual consequences of actions the behaviour policy takes only when they are
safe.
- Sleeping next to a zombie is the case here.
- Heads trained on behaviour data judge sleep safe (confounding). Nonlinear heads recover the interaction only sometimes.
- This is where *epistemic*, decision-relevant uncertainty should act: a world should know it has rarely seen this
  (state, action) pair.

## 18. 2026-10-16 — representation and objective, reassessed together (user request; NOTEBOOK R55; R46-R54)

### Verified facts (code, our measurements, papers)

1. **Our encoder has no patch-level objective.** LeWM (`d4mj/lewm.py`; LeWorldModel 2603.19312) trains only projector(CLS)
   with next-z MSE + SIGReg. The shared embedding and attention weights do receive gradient through the CLS path, but the
   last-layer patch outputs that D, IC and PI use as state *and* targets are never predicted or regularized themselves.
   *(Wording corrected 10-16 after Codex's audit: "no gradient to the patch tokens" was too strong.)*
   - Content is linearly present in them (tiles 0.96, mobs AUC 1.0, health readable; R2). LeVJEPA shows CLS-only training can
     organize patches.
   - **But static decodability is not evidence of predictive learning** (audit control, 10-16): RGB pixels alone decode tiles at
     ~96% and mobs at AUC ~0.997, and an *untrained* ViT matches them.
   - Nothing in our objective asked the patch outputs to be stable, salient or predictive.
2. **The token target is misaligned with what matters (R55):**
   - shares of the target change: ~64% nuisance (rendering / context drift of unchanged content), ~27% never-seen entering
     terrain, ~9% consequences;
   - a health change moves its HUD token *less* than light moves an unchanged tile;
   - discrete codes on these tokens are unstable: 25-46% of persisting tiles change code per step, and HUD codes do not react
     to value changes.
3. **Continuous point prediction cannot represent stochastic outcomes:**
   - the L2 optimum at stochastic mob cells reads as a mob 67% of the time vs 40% in real outcomes (ghosts; R55);
   - MoP-JEPA proves the conditional-mean collapse;
   - on our exact Craftax frames, continuous 7×7 patch targets collapse MBRL (21.2 vs 67.4; Dedieu et al.);
   - Dreamer-CDP needs a categorical latent with KL (6.3 → 16.2) and reward gradients (12.7 → 16.2).
   - PI avoids ghosts with a MAP gate but discards graded risk (R51; its belief has it: R54b).
4. **The theory of our own objective warns against it here.** LeJEPA's optimum is the slowest features; identifiability holds
   only for Gaussian additive-noise worlds and breaks on policy data; with too few dimensions the kept subspace is undetermined
   (Klindt, LeCun, Balestriero 2026). Craftax is discrete, event-driven, behaviour-policy data, much larger than 192 dims.
5. **What decisions need (R53-R54):**
   - the immediate consequence of the action (first-step damage ranks escape moves 0.71);
   - the *expectation* of later consequences. Rollouts from few samples dilute it; graded heads on the belief read it.
6. **Rare counterfactual consequences are a coverage problem.**
   - Sleeping next to a ready zombie: 384 training transitions.
   - It failed in every representation: the CLS world (09-24), PI, and R54's heads.
   - Hansen & Wang 2026: action-marginalized hallucination concentrates in low-coverage state-action regions. No
     representation alone removes it.

### Assessment: the most promising direction

**The world model should learn its own spatial information state**, instead of adopting the frozen encoder's tokens as state
and regression targets:
- **spatial:** per position, on the scaffold;
- **self-predictive:** predictable from the belief and the action, with stop-gradient targets (Ni et al.; Dreamer-CDP);
- **discrete where outcomes branch:** categorical posterior / prior, or hard-assigned mixtures, giving modes *with
  probabilities* rather than means or MAP choices;
- **consequences in the loss:** graded heads for the observation's own consequence channel, plus continuation / reward.

The frozen encoder stays as the *input* (the information is there) and at most as an auxiliary next-embedding target (cosine,
as Dreamer-CDP). *(Superseded the same day: the user made retraining the encoder at the patch level the central candidate.
See §19.)*

**What survives from PI and inferred chance (evidence, not loyalty):**
- the spatial scaffold;
- persistence as the *default mode* of a position's state (exact carry when unchanged: no fading, R23);
- explicit global transformations (scroll; appearance as a global factor, R47);
- stochastic events as discrete choices with inferred posteriors (IC ≈ MoP's hard assignment ≈ Stochastic MuZero's chance);
- consequences read from the belief as expectations (R54b).

**What should not survive:**
- frozen Raw tokens as the persistent state;
- continuous point regression of branching content;
- MAP edits of the consequence channel.

### Proposed experiments S0-S3: superseded, not run

S0 (output parameterization, head-only) and S1 (a learned per-tile state on top of the frozen tokens) were proposed here and
**were not run**. The user redirected (10-16):
- S0 overlaps the existing stochastic-prediction / inferred-chance work;
- S1 does not test how the encoder itself was trained.

The central candidate became **retraining the encoder with a patch-level predictive objective** (§19; NOTEBOOK R56).
S3 (coverage via branched counterfactual data) remains a separate, open lever for the sleep blind spot.

## 19. 2026-10-16 — retraining the encoder at the patch level (user's central candidate; NOTEBOOK R56-R59)

**Question.** Our encoder was trained only through its CLS objective. Does an encoder whose *patch* representations are trained
for action-conditioned prediction give a better predictive world state than frozen Raw?

**Design** (SPATIAL_ENCODER_PREREG.md):
- The new encoder's objective = Raw's (CLS) + a patch path:
  - a spatial predictor with K = 4 hard-assigned heads + router (against the conditional-mean collapse);
  - patch SIGReg or VICReg;
  - optionally photometric invariance and inverse dynamics.
- Same architecture, corpus, batch, schedule and seed as Raw.
- Stage 1 screens at 2k updates. Stage 2 compares 10k encoders against the same predictor trained on frozen Raw, judged on
  decision-relevant predictions (Tier B).

**Evaluation trust (R57).** Codex's audit found, and I verified:
- a probe split that leaked episodes;
- persistence masks that admitted mob changes and the player;
- an analyzer that could not run;
- intervals that ignored episode clustering.

All are corrected in new scripts registered before any Stage 2 result.

**What the corrected numbers say so far:**
- **Static linear probes measure visual decodability, not predictive understanding.** RGB pixels and an untrained ViT decode
  tiles (0.97) and mobs (AUC ≥ 0.999) about as well as Raw; Raw is worse on rare classes and at night.
- **At the common backbone stage, patch training matches or beats Raw on content** (tiles 0.976-0.980, rare-class recall
  0.82-0.84 vs 0.79). It is worse on daytime drift.
  - The learned projector makes the deployed state harder for a fixed linear probe. R59 localizes the loss to its nonlinear
    warp + final compression (conditioning contributes; stale BN statistics do not).
- **The corrected screening selects the full arm** (MoP + SIGReg + photometric invariance + inverse dynamics). The original,
  defective evaluation had selected MoP + SIGReg, which Stage 2 is training. Stage 2 is kept and evaluated correctly; the
  full arm at Stage 2 is a separate decision.
- **Stage 2, both seeds (R58, R60): "mixed"** (MoP + SIGReg vs Raw-predictor), by the preregistered rule.
  - Seed 1: no primary improves.
  - Seed 2: escape AUC 0.550 vs a below-chance 0.459 [+0.046, +0.140], but both guards fail (census explosion; calibration).
  - Attack recall stays at ~1-2% throughout; D's own rollout reaches 0.711 on the same escape test.
- **Open hypothesis (R60, untested):** the patch targets are not detached (as in LeWM's CLS path). The encoder may make its
  deployed tokens predictable by discarding hard-to-predict mob content.

**Three levels must be kept apart** (R59; user, after Codex's BN_AUDIT.md). Tier B reads predicted *deployed* states with
deployed-state readers, so one number mixes all three:

| Level | What it is | Seed-1 evidence |
|---|---|---|
| 1. Backbone | the patch-trained ViT | at least Raw's rare-mob decodability (zombie AP 0.986-0.993); lower night drift. Already reacts to sleep rendering (zombie FP 4.4% on sleeping vs 0.03% on awake cells) |
| 2. Deployed state | projector (Linear → BN → GELU → Linear 2048→192) | zombie AP 0.51-0.62. Lost in the nonlinear warp + compression (hidden state 0.995; same-size PCA 0.873); not stale BN. In sleeping frames the deployed state goes blind at both seeds (R61: zombie AP 0.006 / 0.10 vs backbone 1.00; false positives are seed-dependent; Raw's reader also misfires in sleep, 1.1-1.3%). Evaluation frames come from another policy (legacy BC vs PPO expert): earlier, brighter, less sleep |
| 3. World predictor | 2-frame, 4-layer, K = 4 heads | R60 (both seeds; location, not counts): Spatial worlds' imagined mobs are in the wrong cells (location precision 0-3%; expected states show none) and inflate through recursion (s2 d16: 2.8 zombies vs 0.25). Raw-predictor ≈ a degraded copy, then erases mobs. The s1 cow-census and graded-health edges did not replicate (retracted). Both miss attacks |

**Consequence for interpretation.** The verdict concerns this deployed interface with this predictor. It is not a verdict on
patch-level learning or on the backbone.

**Measured next, not assumed:** whether the full arm's photometric invariance reduces the sleep confound (Amendment 6; both
stages, all arms; registered before any full-arm number).

**Candidate after the comparison:** a matched backbone-state vs projected-state world-prediction control. It isolates level 2
from level 3 and is preferred to a BN → LN swap as the first follow-up.

**Result of the three-encoder comparison** (R62; 2 seeds each, identical corrected protocol):
- **Full vs Raw: "not supported."**
  - Attack recall 0; escape AUC below chance at both seeds (0.462 / 0.460);
  - descriptively worse than MoP on escape at both seeds;
  - better census stability and the best background prediction *in token MSE* (equal in content, R63), but it erases mobs at
    the first step.
- **Sleep (Amendment 6):** the preregistered misfire reading is "does not reduce". Descriptively, the full arm restores cow
  detection in sleeping frames at both seeds (deployed AP 0.73-0.91 vs MoP 0.03-0.05); zombies only partly.
- **Synthesis:**
  - Level 1 is not the bottleneck: every backbone keeps rare mobs; patch training costs temporal stability.
  - Level 2 costs the Spatial arms 9-37 AP points.
  - Level 3 fails for all three encoders, in three different ways: copy then erase / hallucinate then explode / erase.
  - Changing how the encoder is trained changed how the world fails, not whether it fails.
- **Proposed next (user decides):** the matched backbone-state vs projected-state world-prediction control. It separates level 2
  from level 3 directly. Two cheap evaluation-only checks could precede it:
  - the deployed-AP trajectory over saved checkpoints vs the prediction loss;
  - a known-good localization reference.

## 20. 2026-10-17 — why worlds erase mobs, and what still blocks decisions (user question; NOTEBOOK R63)

**The objective, restated.** Coherent, reconstruction-free imagination that **improves decisions**, not latent prediction
accuracy. Two numbers keep us honest:
- a predicted state that copies its input already reaches ~97% of token variance;
- the decisions depend on the ~1-2% of cells where mobs and consequences live.

### What we now know (measured; R60-R63)

1. **Static decodability is not the bottleneck.** Every encoder's backbone keeps rare mobs (AP ≥ 0.97). So do RGB pixels.
2. **Imagined mobs are judged by location, not counts.** Census agreement hid wrong-cell hallucination (R60).
   - Location measures are validated: they reproduce independent implementations exactly.
   - They are threshold-free, tolerant to one-cell displacement, with an oracle head choice, and stratified by camera motion.
   - They hold on held-out training-distribution windows.
3. **Jointly trained Spatial worlds LEARN to erase mobs.** All four runs (MoP, full × 2 seeds) keep next-step mob information at
   copy level for ~1k updates, then destroy it between ~1k and ~4k. Zombie retention at 10k: 0.07-0.26.
   - The same predictor, loss, budget, data and seeds on frozen Raw tokens never does (0.77-1.05).
   - So erasure is **not** a property of the predictor or the MSE objective. It is not caused by low mob energy in the space
     either: Raw's space has less.
4. **Background prediction is equal across encoders in content** (~+0.7 pt tile accuracy over copy; ~24% of camera-shifted cells,
   vs 93% real). The full arm's token-MSE advantage tracks its larger frame-to-frame drift.
5. **Retention is necessary but not sufficient.** The Raw-predictor retains mobs (cows slightly beyond copy) yet predicts no
   attacks (recall ≤ 2%) and ranks escapes at chance (R60).
   - **[R64]** Its retention is *screen copy*: it does not move mobs with the camera or predict arrivals beyond copy.
   Two distinct level-3 failures:
   - **F1, erasure:** learned in joint training;
   - **F2, no consequences:** present without any projector.
6. **Predicted states can carry information off the real-state manifold.** Early joint predictors retained mobs that real-state
   readers could not read. Imagination read by heads trained on real states needs predictions **on** the manifold. The
   Raw-predictor's predictions move onto it as training proceeds; the joint predictors' never do.

### What remains hypothetical

- **The cause of F1.** Three candidates:
  - (a) co-adaptation through non-detached patch targets: Ni et al. 2024, online targets lose rank; Littwin et al. 2024, JEPA's
    bias toward predictable features;
  - (b) the projected space's geometry;
  - (c) the patch-trained backbone.
  The existing runs cannot separate them, because space and predictor co-evolve.
- **The cause of F2.** Either the consequence is never represented in the predicted state (uniform token MSE: consequence cells
  are rare, R55 ~9% of target change, E14), or it is represented off-manifold and unread.
- **That fixing F1 helps decisions.** Raw-like retention alone does not (R60).
- **The full arm's drift explanation** for its token-MSE advantage.

### The next discriminating questions

1. **F1:** given a frozen patch-trained encoder, does a fresh predictor on its **deployed** tokens erase, and one on its
   **backbone** tokens not?
   - The design is `20261016_spatial_encoder/FROZEN_STATE_CONTROL_PROPOSAL.md`: ~5 h GPU, predictor-only; staged.
   - If co-adaptation is implicated, a detached-target joint run follows.
2. **F2:** in a world that retains the mob, is the attack's consequence present in the predicted state at all?
   - A cheap refit check on existing Raw-predictor checkpoints comes first.
   - Only then an objective-level intervention, e.g. consequence-weighted or discriminative prediction, preregistered and after
     a literature sweep on that diagnosis.

F1 decides which state to build on. F2 decides whether any of them can improve decisions. The control answers F1 cheaply; F2 is
the larger blocker for the objective.

## 21. 2026-10-17 — after the audit: mobs at four levels, consequences within roots (NOTEBOOK R64; Codex audit)

**Measurement first.**
- AP is now tie-grouped (a confirmed defect, corrected; small impact; R63's trajectory survives).
- "Retention" is renamed relative next-cell decodability.
- Consequences are judged **within roots, across actions** (a pooled AUC measured root danger).
- Mobs are judged at four levels, because the central objective needs all four.

### What the Stage 2 worlds do, against the objective (§8's five questions; §17's working principle)

| need (§8 / §17) | best Stage 2 world | measured |
|---|---|---|
| 1. decodable state | Raw tokens | the real next frame's mobs refit-decodable at 0.94 / 0.99; joint spaces 0.49-0.91 |
| 2. persistence ("persist the scene, with transformations") | Raw-pred = **screen copy** | static stays ≈ copy; **shifted stays 0.001** (world persistence 0.14 / 0.38); joint worlds lose even static stays |
| 3. stochastic dynamics (future locations) | none | no world beats screen copy on arrivals (zombies chase the player); the sibling reference is 0.51 |
| 4. chance sampled coherently | none | samples miscount objects: Raw 6× exact persistence; joint 20× |
| 5. consequences read for decisions | none | within-root attack AUC ≤ the linear action prior (0.62); Raw s1 0.36, worse than it. The real next frame: 1.00. §17's D world: 0.711 escape ranking |

**Interpretation.**
- The Stage 2 patch predictor (2-frame context, 4 layers, min-over-K MSE + router; heads sampled independently per patch) is a
  **copy machine with blur**, on every input.
- Joint training adds erasure on top. Neither BN mode nor cohort explains it (C-BN negative).
- The earlier Raw-token belief worlds (D, PI; §17) were stronger on both axes that matter: D's first-step consequence ranking
  0.711, PI's pursuit 0.25-0.30.
- So the encoder study has not produced a better world. It has localized two failures:
  - **erasure:** learned under joint training;
  - **no dynamics or consequences:** in the Stage 2 predictor on any input.

### What remains hypothetical
- Why joint training erases. FD (learnability of the final interface) is running.
  - Co-adaptation through target gradients is **not** testable by FD and is not claimed.
- Whether any of this matters once a stronger predictor (D / PI class) reads these states. Untested; no new architecture
  experiment is planned now.

### The next decision (after FD s1 / s2; registered rule, δ = 0.153)
- **FD learnable like Raw:** the final deployed interface is learnable without erasure. The joint history is implicated
  (mechanism open). No FB.
- **Otherwise:** FB (frozen backbone) isolates the representation interface.
- In either case the decision-relevant conclusion stands: no Stage 2 predictor delivers dynamics, coherent samples or
  consequences. The bottleneck for the objective is the world model's dynamics / consequence channel, not the encoder alone.
