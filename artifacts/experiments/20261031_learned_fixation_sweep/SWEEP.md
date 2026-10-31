# P1-F2 literature sweep: a fovea that learns where looking teaches the world (10-31)

User, 10-31: pretrain the fovea by letting it learn where to fixate to improve where the world model is starved of
learning (hoped: the sparse consequence events); later, in P2 / P3, step up to value of information (VoI), keeping
the learning-driven fixation at a reduced rate. Questions: would this address failure mode 3 (the old P1 failures:
rare and stochastic single-cell consequences under-generated); why has the hand-placed fixation not done so; what does
the literature say. No conclusion is drawn here on the design; the open questions and candidate tests are listed for
discussion. Codex's TSP / faithful-loss investigation is deliberately not consulted.

## 1. What our own measurements already say

**What the fovea changes mechanically.** The G1 loss averages over every scalar of the next percept: 81 x 16
periphery + 9 x 64 glimpse + 8 lighting = 1,880 per frame (LOSSES G1 row). The glimpse is 31% of every frame's
next-percept loss, wherever the eye is. A cell under the eye carries 16 + 64 elements instead of 16, about 5x its
weight in that frame, plus TSP's and the reconstruction's pixel grounding. The fixation decides what fills a third of
the predictive target. The chooser is therefore an allocator of the training signal, which is the lever the user
means by "starved of learning" (generation diagnosis H, 10-29: single-cell consequences were 0.31% of G1 loss, their
gradient nearly orthogonal to the total, their drive gone by checkpoint 10). It is also, mechanically, an adaptive
reweighting of the target by attention; the standing "no loss reweighting" rule is noted, the fovea being the agreed
mechanism (DECISIONS 10-25).

**The lever needs a target in which the consequence is visible.** M0.5 (old NLL decoder) put the eye on the counter
and the increments stayed null; the dissection found the digit at 0.3x the within-value radius of the glimpse token
(the extra gradient landed where the digit was not represented). P1-F's glimpse tokens carry it at 1.5-2.0x and its
glimpse learned the increments (matched 0.73 / 0.94). So "where to look" and "what the looked-at token can express"
are coupled; the decoder question Codex is examining bears directly on P1-F2.

**What the hand rule covered, and what came of it.** The rule (map.json, label-derived, Q = 1 in P1-F) targeted five
consequences: drink, wood, sapling, table, zombie hit. Mining, eating, crafting, mobs, and deaths other than beside a
zombie were never preferentially fixated. Under P1-F's own gaze, G1 by class against world 1:
- deterministic, visible: drink up 0.087 vs 0.158, wood up 0.031 vs 0.106, table 0.095 vs 0.132: learned;
- stochastic coin (sapling, p ~ 0.1 per attempt): field better (0.23 vs 0.44) but imagined saplings 0.08 against 0.135
  recorded (world 1 0.05). The 10-29 diagnosis B measured that the production 3-step sampler suppresses minority
  outcomes even with a perfect field (0.15 -> 0.05);
- stochastic with hidden state (zombie hit, cooldown not rendered): 0.21 vs 0.20, unchanged although the rule put
  the eye on the health window in its zombie-adjacent contexts (119 of its 129 contexts).

**What it taught is gated by the exact fixation** (Codex, `../20261031_fovea_shortcut_review/REPORT.md`, frozen
checkpoints, 18,432 generated successors, simulator successors replayed pixel-exactly). With history, action and noise
fixed, P1-F's periphery draws the wood gain on 83.6% of samples with the rule's window, 7.3% with a window one row up
that still contains the whole counter, 0% with a map window (drink 64.6 / 12.0 / 0). Random histories give the same.
NOOP produces no gain (0 of 4,608 verified). World 1 is 9-15% under every window. The key is the exact next window
(its coordinates or the token placement they imply: not separated, Codex), not the counter being in view; the action
still matters.

**Earlier attempts at loss-driven selection** (E14, 10-02, a different architecture): the hardest tokens were 93%
entering terrain, static and HUD (unpredictable or trivial, not consequences); an RHO-style irreducible-loss model
copied consequences too (circularity); positives-only doses hallucinated 24-30%. "A selective dose needs MATCHING
NEGATIVES (the context where a change could happen)." The hand rule had them by construction (context from state t-1,
outcome unseen), and P1-F stopped inventing changes on failed attempts (0.66 -> 0.07).

## 2. The literature

### Learning progress, not error
- **Intelligent Adaptive Curiosity** (Oudeyer, Kaplan, Hafner, IEEE TEC 2007; read pp. 268-272). Reward = the
  smoothed decrease of a region's prediction error (mean over theta = 25 recent errors vs the same tau = 15 steps
  earlier); regions split from experience (variance criterion); action = greedy on expected progress with epsilon
  "typically 0.35". "The robot will not stay in front of white noise for a long time or in unlearnable situations
  because this does not lead to a decrease of its errors." Comparing error rates across DIFFERENT activities in time
  is meaningless; progress must be compared within similar situations (regions). Observed sequence: easy situations
  first, then harder ones, unlearnable ones avoided.
- **Automated curriculum learning** (Graves et al., ICML 2017; ar5iv read). Prediction gain (same sample, before vs
  after the update) was "the most consistent signal" for maximum likelihood; gradient prediction gain ||grad||^2
  "favours tasks whose loss has higher variance" and was "much worse than uniform" on repeat copy; self prediction
  gain re-samples a fresh example from the same task, "an unbiased estimate of expected learning progress" at higher
  variance. "Uniformly sampling from all tasks is a surprisingly strong benchmark." Non-stationary bandit Exp3.S,
  rewards rescaled by reservoir quantiles. Prediction gain costs one extra forward pass.
- **Active World Model Learning with gamma-Progress** (Kim, Sano, De Freitas, Haber, Yamins, ICML 2020; pp. 2-13
  read). Closest precedent: a gaze controller (9 rotations) rewarded by world-model progress. Reward = L(theta_old) -
  L(theta_new) on newly collected data, theta_old an exponential moving average of the weights (gamma = 0.9995), one
  extra model copy; controller a DQN with epsilon-greedy. Beat delta-progress, RND, disagreement, adversarial (error)
  and random; "baselines frequently perform worse than random due to noise fixation". Attention to animate agents
  (learnable but not yet learned) emerged without being built in.
- **Model-based exploration by empirical learning progress** (Lopes, Lang, Toussaint, Oudeyer, NeurIPS 2012; abstract
  level): progress-driven exploration robust where count- or prior-based model certainty is misled. Partially
  reproduced in 2024 (missing protocol details).
- **Absolute / bidirectional progress** (Portelas et al., CoRL 2019, ALP-GMM; Kanitscheider et al. 2021, Minecraft;
  abstract level): progress in either direction counts, so forgetting is detected and revisited; unidirectional
  progress produced forgetting cycles.
- **Generalising progress estimates** (MAGELLAN, Gaven et al., ICML 2025; abstract level): a learned predictor of
  competence and progress transfers between related goals, making progress estimation sample-efficient in large
  spaces, matching expert-defined groupings. Relevant to rare contexts (mining: 12 diamond histories in dev).
- **Error minus a learned noise floor** (Curiosity-Critic, Bhaskara & Wang, arXiv 2604.18701, 2026; partly read):
  reward = current error minus a critic's estimate of the post-update error, near zero on stochastic transitions; grid
  world only, authors flag high-dimensional settings as untested.

### Looking to learn, and learned masking (the same choice in other fields)
- **Learning to Look by Self-Prediction** (Grimes et al., TMLR 2023; abstract level): camera policy rewarded by the
  NEGATIVE prediction error; a "behavioural fovea" emerged. Opposite family to curiosity: look where prediction is
  possible (cf. Active Efficient Coding, Zhao, Rothkopf, Triesch 2012, secondary sources: eye movements rewarded by
  coding efficiency). For our purpose it would steer the eye to what is already easy.
- **SUGARL** (Shang & Ryoo, NeurIPS 2023; abstract level): separate sensory and motor policies; sensory reward =
  how well the observation lets one infer the motor action. **EyeRobot** (arXiv 2506.10968) and **Gaze on the Prize**
  (arXiv 2510.08442): task-driven gaze. These are the VoI family (P2 / P3), not world-model learning.
- **Hard Patches Mining** (Wang et al., CVPR 2023; abstract + supplement snippet): a head predicts per-patch
  reconstruction loss; the masked patches mix predicted-hard ones (fraction alpha_t, growing over training) with
  random ones. **ADIOS** (Shi et al., ICML 2022; abstract level): adversarial masker, semantically meaningful masks.
  Both choose which part of the input becomes the target, as our fixation does, and both keep randomness or a schedule.
- **RHO-LOSS** (Mindermann et al., ICML 2022; abstract level): select points by training loss minus a holdout model's
  irreducible loss: high-loss selection picks noisy points, easy-point curricula add little once learned. Our E14 found
  the circularity risk when the holdout model shares the bias.
- **Curious Replay** (Kauvar et al., ICML 2023; read): replay priority p = c beta^v + (|L| + eps)^alpha (beta 0.7,
  alpha 0.7, c 1e4, eps 0.01), v the replay count. DreamerV3 on Crafter 14.5 -> 19.4; adversarial-only 17.3,
  count-only 16.2: each helps, combined best. Authors argue the white-noise problem is weaker for prioritising a
  supervised fit to a fixed buffer than for exploration. The closest evidence that reallocating a world model's
  training signal by loss helps on our environment family.

### Gated knowledge: policy confounding and how it is diluted
- **Policy confounding** (Suau, Spaan, Oliehoek, RLC 2024; mitigation sections read): "we do not have a complete
  answer"; remedies "revolve around the idea of broadening the distribution of trajectories": large replay buffers,
  exploration (DQN with a final epsilon of 0.1 instead of 0 generalised off its habitual trajectories), domain
  randomisation. Our case is the cleanest instance: the deterministic rule made "eye at (7, c)" a perfect proxy of
  "DO facing tree"; nothing in training ever separated them.
- **View-invariant scene state** (Eslami et al., Science 2018, GQN; summary level): a representation aggregated from
  observations at viewpoints, a generator rendering any QUERY viewpoint from it. The scene state cannot depend on the
  query by construction. Williams (Neural Computation 2025, in READING.md): a fixation is a measurement of ONE latent
  scene. Our denoiser generates periphery and glimpse jointly, conditioned on the next fixation, so nothing
  structural stops the physical prediction from depending on where the eye will be. Overlap in the data (above) and
  structure (here) are the two known routes to consistency; whether overlap suffices for our model is a fact to
  measure.

### Search before sampling (P1 then P2 / P3)
- **Gottlieb & Oudeyer** (Nature Reviews Neuroscience 2018; abstract level) distinguish information SAMPLING (reduce
  uncertainty for a familiar task) from information SEARCH (open-ended); learning progress among the preferences
  shaping both. The user's sequence maps onto it: learning-progress search while the world forms, VoI sampling once
  there are decisions.
- **Infants tailor attention to learning progress** (Poli et al., Science Advances 2020; abstract level): looking
  continued while progress was high, looked away when stimuli did not allow learning (replicated in a reanalysis).
- **Active sensing** (Yang, Lengyel, Wolpert, eLife 2016; abstract level): human fixations more informative than
  random, about 70% of the expected-information-gain optimum. **Friston et al.** (Frontiers 2012): saccades as
  experiments on hypotheses. Support for VoI / information gain in the later stage.

### Generation of minority outcomes (what fixation cannot reach)
- **Minority guidance** (Um, Lee, Ye, ICLR 2024; abstract level): on long-tailed data, Tweedie's formula biases
  denoising against minority samples. Consistent with our 10-29 measurement that the 3-step sampler suppresses
  minority branches under a perfect field. A training-signal allocator does not change this.

## 3. Would a learning-progress fovea address failure mode 3?

Partly, and the parts can be named.
- **Deterministic, visible consequences** (drink, wood, table; also stone and coal mining, eating, crafting outputs,
  none covered by the rule): yes in principle. These are learnable from the crop and showed large gains when looked
  at. A progress chooser would find them while they are being learned and leave them when learned. Caveats: they are
  rare (attempt frames 13%, the right window 1 of 49, so ~0.3% of frames under pure exploration); rare contexts need a
  generalising progress estimate; and the gains are useless when gated.
- **Stochastic coin outcomes** (sapling): the field improves when looked at, but generation is capped by the
  sampler's bias against minorities. Fixation cannot fix that part.
- **Hidden state** (zombie cooldown): no progress from looking (the rule's own result: unchanged with the eye on
  health in its zombie-adjacent contexts). A progress chooser would correctly stop looking there; the fix is memory.
- **Mobs at about half the recorded count**: unclear. Progress curiosity attends to animate agents (Kim et al.), but
  vanishing mobs may be the sampler averaging out uncertain positions rather than a training-signal deficit. Not
  decomposed for any of our worlds.
- **Gating**: a chooser that is deterministic in its context recreates P1-F's gate, learned instead of hand-placed.
  The literature's remedy is overlap (Suau et al.; IAC's epsilon 0.35; HPM's random fraction); whether overlap alone
  removes the gate in our denoiser is unmeasured.

## 4. Why the hand-placed fixation did not fix it
1. Coverage: five targets, chosen by labels; mining, eating, crafting, mobs and most deaths never had the eye.
2. The covered stochastic targets are limited elsewhere: sapling by the sampler, damage by hidden state.
3. What it taught is gated to the exact window, so nothing reached the periphery under any other gaze, which is where
   failure 3 was measured.
4. Before the decoder fix (M0.5) the looked-at token could not express the digit, so looking taught nothing about
   increments.

## 5. Open design questions (not decided; each needs its own discussion)
1. **Progress of what.** Glimpse G1 progress is measured on LEARNED targets that drift as the encoder trains (old and
   new parameters would score against different targets, and a simplifying encoder looks like progress). Pixel
   targets (TSP's or the reconstruction's error at the crop) are fixed. Transferable progress (the periphery's
   prediction with the eye elsewhere, on fresh samples: Graves's self / target prediction gain, Codex's suggestion)
   directly penalises gated knowledge but is noisier and harder to credit to a window.
2. **Same-sample or fresh.** Prediction gain on the trained sample favours high-variance (noisy) regions (Graves);
   gamma-progress on not-yet-trained samples (Kim) is the cheap fresh-sample estimate (one EMA copy, one extra forward
   pass).
3. **Chooser form and inputs.** A contextual bandit over the 49 windows, from pre-outcome inputs only (belief at t-1
   and action: the timing contract, which also gives matching negatives). Options: region-based (IAC), Exp3.S
   (Graves), a value head with epsilon (Kim), a learned progress predictor (HPM, MAGELLAN). No simulator labels
   (generality), unlike the rule.
4. **Overlap and schedule.** A permanent floor (IAC 0.35, Suau 0.1) and a ramp from random to chosen (HPM). The floor
   that removes the gate is an empirical fact about our model.
5. **Structure vs data for consistency.** Is the physical prediction allowed to condition on the next fixation at
   all (GQN / Williams structure), or is overlap enough? Architecture: a design decision, informed by the measurement
   in 4.
6. **Stability.** Absolute progress against forgetting (Portelas, Kanitscheider); non-stationarity as the encoder
   and world change; early progress lives in the bulk (movement), consequences later, which matches the user's
   curriculum.
7. **P2 / P3.** VoI needs glimpses that change the belief and decisions: on world 1 the belief kept nothing of
   glimpses and the hindsight value of looking was 0.25-0.6% of the loss (fovea probes 10-29); neither is measured on
   P1-F. A gated world lets a VoI gaze suppress imagined consequences without changing the physical action (Codex):
   consistency is a prerequisite. Per LEARNING_SKETCH the world is frozen during actor training, so learning-progress
   fixation has no role in P3 unless the world keeps training there.

## 6. Candidate tests (none launched; each states what it would rule out)
- **A. Failure-3 decomposition on P1-F** (reading, existing checkpoints; 10-29's instruments A / B). Claim: the
  remaining sapling, damage, mining and mob deficits split into field vs sampler vs hidden state. Competing: all field.
  Control: world 1 read the same way. Rules out expecting fixation to fix the sampler-bound and hidden-state parts.
- **B. Would progress be visible?** (reading, world 1's ten checkpoints). Pixel-space error (TSP / reconstruction) at
  each window x context class on frames the random prior happened to fixate, across checkpoints: does a progress
  signal exist at consequence windows, when, and how noisy? Competing: random-gaze training only shows progress under
  random looking (a lower bound). Rules out the premise "exploration will find consequence progress" if none is
  visible.
- **C. Does overlap dissolve the gate?** (intervention: P1-F fine-tuned with the rule at Q < 1 vs Q = 1, same learning
  rate, then Codex's consistency test). Competing: the gate is structural and survives overlap; the fine-tune's
  learning rate alone moves it (the Q = 1 arm controls). Rules out "a floor is enough" if the gate survives. Motivated
  by Codex's result, so it needs your explicit go-ahead under the no-reaction rule.
- **D. VoI substrate on P1-F** (reading; fovea probes F1 / F2). Does the belief keep glimpse evidence now that the
  glimpse carries the digit? Rules out VoI as a near-term stage if null.
- **E. P1-F2 itself**, from scratch, with declared readings: Codex's consistency test as a standing criterion, the
  random-gaze and own-gaze panels, the chooser's statistics (entropy per context, overlap both ways: eye elsewhere on
  attempts, eye on counters without attempts), and the failure-3 decomposition.

## Sources
Read in full or by section: Oudeyer, Kaplan & Hafner 2007 (https://adapt.informatik.hu-berlin.de/pub/papers/ims.pdf);
Graves et al. 2017 (https://arxiv.org/abs/1704.03003); Kim et al. 2020 (https://arxiv.org/abs/2007.07853); Kauvar et
al. 2023 (https://arxiv.org/abs/2306.15934); Suau et al. 2024 (https://arxiv.org/abs/2306.02419); Codex's report
(`../20261031_fovea_shortcut_review/REPORT.md`). Partly read: Bhaskara & Wang 2026 (https://arxiv.org/abs/2604.18701).
Abstract or summary level only (claims above limited accordingly): Lopes et al. 2012 (NeurIPS 25, 206-214); Portelas
et al. 2019 (https://arxiv.org/abs/1910.07224); Kanitscheider et al. 2021 (https://arxiv.org/abs/2106.14876); Gaven et
al. 2025 (https://arxiv.org/abs/2502.07709); Grimes et al. 2023 (TMLR, https://openreview.net/forum?id=9aXKUJEKwV);
Shang & Ryoo 2023 (https://arxiv.org/abs/2306.00975); arXiv 2506.10968; arXiv 2510.08442; Wang et al. 2023
(https://arxiv.org/abs/2304.05919); Shi et al. 2022 (https://arxiv.org/abs/2201.13100); Mindermann et al. 2022
(https://arxiv.org/abs/2206.07137); Eslami et al. 2018 (Science 360:1204); Gottlieb & Oudeyer 2018
(doi:10.1038/s41583-018-0078-0); Poli et al. 2020 (Sci. Adv. 6:eabb5053); Yang, Lengyel & Wolpert 2016 (eLife
5:e12215); Friston et al. 2012 (Front. Psychol. 3:151); Um, Lee & Ye 2024 (https://arxiv.org/abs/2301.12334); Zhao,
Rothkopf & Triesch 2012 (secondary sources).
