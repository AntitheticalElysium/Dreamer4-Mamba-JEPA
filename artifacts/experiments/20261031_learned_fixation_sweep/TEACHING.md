# What should the fovea learn, and how? (research pass, 10-31 night; for discussion, nothing decided)

User, 10-31: the hand-placed fovea is ad hoc and not the final version (no test C); we still need a way to teach the
fovea where to look; fixing the field deficits is a bonus; the shortcut is likely tied to the hand placement. Read
the literature, the brain included. This note builds on A, B and D (RESULTS.md, corrected b95b3b25), Codex's
reports (`../20261031_fovea_shortcut_review/REPORT.md`, `../20261031_grounding_loss_review/followup/REPORT.md`,
`.../sampler_followup/REPORT.md`, `../20261031_learning_foveation_review/REVIEW.md`) and one new reading, F (PLAN F).

## 1. What limits what (A, B, D and Codex together)
| deficit | limited by | evidence | lever |
|---|---|---|---|
| deterministic visible consequences (drink, wood, mining) | the field's training signal (rarity: single-cell consequences were 0.31% of G1, 10-29 H) | A: world 1's field at sigma 6.67 0.03-0.34 against 1.00 recorded | training signal: the fovea is one allocator; others are outside the fovea |
| mobs (zombies, cows, P1-F's skeletons) | sampler-sensitive at fixed weights | A; Codex's factorial: step count and churn each matter (P1-F skeletons 0.051 / 0.107 / 0.162, real 0.133) | sampler (separate design question) |
| sapling's chance, once the field has it | sampler (minority suppression) | A: P1-F own gaze field 0.03-0.04, production 0.005 | sampler |
| zombie hit timing | hidden cooldown | A: field ignores readiness | memory |
| the gate (consequence drawn only under the rule's exact window) | deterministic, outcome-correlated placement + a denoiser free to read the query fixation | Codex: rule crop 83.6% vs another crop containing the counter 7.3%; persists under all six TSP variants; world 1 9-15% everywhere | structure + overlap (section 4) |
| periphery static at night | TSP's mean teacher (Codex) | 6.69x noise variance; beta TSP -13.8%; no generation gain | Codex's overnight run |
| VoI substrate | the periphery already carries the digits (12-19x); belief keeps nothing of glimpses | D | periphery/fovea capacity (section 6) |

B1 concerns the progress signal's targets, not the field. The "denoiser and sampler" reading is half right: the
sampler is a real, separate lever; the denoiser's field deficit is a training-signal deficit, which is exactly what
a learning fovea was meant to address, though not the only lever.

## 2. The shortcut: what Codex established, and why a learned chooser alone would not remove it
Codex (frozen P1-F, 18,432 generated successors): the periphery's wood gain is drawn under the rule's window 83.6%,
under a window one row up that still contains the whole counter 7.3%, under a map window 0% (drink 64.6 / 12.0 / 0);
action still necessary (NOOP 0 of 4,608); past gaze matters much less than the next (query) fixation; the reward
readout is view-dependent even with the real successor re-encoded (wood reward NLL 0.0036 / 0.35 / 0.57). Not
established: the internal pathway (fixation mask, crop token position or features), guidance vs decoder shares.
Its four contracts for any learned chooser: pre-outcome inputs; coverage (revisits); progress on fresh samples
and a common reference; passive-view and action tests as success criteria.

The user's hypothesis holds in this sense: the gate needs placement that is deterministic and correlated with the
outcome, which the hand rule was (world 1, random gaze: no gate). But that property, not the hand, is the cause:
- **Policy confounding is a property of the data distribution, not of who chose it** (Suau et al., RLC 2024: remedies
  "revolve around the idea of broadening the distribution of trajectories").
- **Networks take the simplest sufficient cue and stop learning the rest.** SGD's simplicity bias is extreme: models
  rely exclusively on the simplest predictive feature (Shah et al., NeurIPS 2020); cross-entropy training captures a
  subset of predictive features and the others "fail to be discovered" (gradient starvation, Pezeshki et al.,
  NeurIPS 2021). B2's P1-F sign (the look's signal turning against the gaze-free wood prediction from checkpoint 4)
  and A (P1-F's gaze-free wood 0.04 against world 1's 0.24) are what starvation looks like.
- A learned chooser deterministic in its context hands the denoiser the same kind of free feature: its fixation is a
  function of (belief, action) that summarises "an attempt is happening here". Under the hand rule the feature may
  also have carried simulator information the belief lacks (facing direction from labels); that label leak is
  untested and would not transfer to a learned chooser, but the shortcut would.
- It would break exactly when the gaze policy changes: random-gaze evaluation, and P2 / P3's VoI gaze.

## 3. The brain: how gaze is chosen and what looking teaches
- **Gaze goes where the action will act, before it acts, and is learned.** Grasp-and-move task: "Landmarks at which
  contact events took place were obligatory gaze targets"; "Subjects never fixated the hand or the moving bar";
  gaze leads, and leaves a landmark at the kinematic event there; "The salience of gaze targets arises from the
  functional sensorimotor requirements of the task"; gaze "contributes to the development and maintenance of
  sensorimotor correlation matrices that support predictive motor control" (Johansson, Westling, Backstrom &
  Flanagan, J Neurosci 2001). In natural tasks "almost no fixations fall on irrelevant objects", patterns "must be
  learned", saccades are "pro-active ... in advance of an expected event" (cricket batsmen fixate the bounce point,
  experts ~100 ms earlier), "shaped by learnt internal models of the dynamic properties of the world" (Hayhoe &
  Ballard, TiCS 2005). The fixation that teaches the forward model is at the action's locus, chosen before the
  outcome, and covers successes and failures alike (matching negatives by construction).
- **Information value for the next action is a distinct, measured signal.** LIP neurons encode, before an
  information-sampling saccade, the expected reduction in uncertainty for the subsequent action, separable from
  reward (Foley, Kelly, Mhatre, Lopes & Gottlieb, PNAS 2017). In the Sprague-Ballard model "fixation is allocated to
  the task with the most to lose" (in Hayhoe & Ballard 2005). This is P2 / P3's VoI.
- **Learning progress is supported behaviourally, not yet for saccades.** Humans choosing among activities are best
  fit by learning progress plus competence; LP-driven participants sampled the unlearnable activity "in relative
  moderation" (Ten, Kaushik, Oudeyer & Gottlieb, Nat Commun 2021). Novelty and surprise "do not guarantee that an
  agent will learn" and risk "trapping him in unlearnable tasks" (white noise on a TV); but "a quantitative response
  to learning progress has not been demonstrated" neurally and its relation to eye-movement control "is unknown"
  (Gottlieb, Oudeyer, Lopes & Baranes, TiCS 2013).
- **Bottom-up surprise attracts free viewing** (72% of saccades to above-average Bayesian surprise in TV / game
  footage; Itti & Baldi, Vision Res 2009), but task demands override salience (Johansson; Hayhoe & Ballard).
- **The world model is kept independent of where the eye goes; the fovea's input is predicted from the periphery.**
  A corollary discharge of the saccade command is "the critical" signal for visual stability; receptive fields shift
  to their post-saccadic location before the saccade (predictive remapping; Wurtz, Vision Res 2008). Before a saccade
  the fovea anticipates the target's defining features, fed back from the peripheral target: "a retinotopic
  anticipation of soon-to-be foveated information" (Kroell & Rolfs, eLife 2022). Peripheral-foveal associations are
  learned across saccades: peripheral recognition is biased toward previously associated post-saccadic foveal input,
  tied to making saccades (Herwig & Schneider, JEP:G 2014). The brain predicts what the fovea will see from what the
  periphery holds and where the eye goes: p(glimpse | periphery, target), never the world conditional on the target.

## 4. Machine learning precedents
- **Scene state independent of the next view.** Jayaraman & Grauman (CVPR 2018): the decoder reads only the
  aggregated code, never the next viewpoint; a REINFORCE look-around policy rewarded by the final completion error
  ("per-timestep rewards would induce greedy short-term behavior") improves 1.5-2.2x over random's gain. GQN
  (Eslami et al., Science 2018) is the same structure.
- **A learned "where" map from the periphery.** Daucé, Albiges & Perrinet (J Vis 2020): a dorsal-like network maps
  log-polar peripheral input to the predicted post-saccade foveal accuracy at each target ("should be trained by
  sampling, that is, by trial and error"); saccade = arg-max, stop when the centre is better; an approximate
  information gain. Hard Patches Mining and MAGELLAN are the predicted-loss / predicted-progress versions.
- **Privileged teachers, then removed.** Sidekick policy learning (Ramakrishnan & Grauman, ECCV 2018): helpers with
  full observability during training shape the reward or initialise the policy, then the agent acts without them;
  faster convergence. The hand rule is a sidekick that was never removed (and Q = 1 left no overlap).
- **Sensorimotor reward.** SUGARL (Shang & Ryoo, NeurIPS 2023): the sensory policy is rewarded when its view lets an
  inverse model infer the agent's own action; fixation and tracking of agent-relevant objects emerge. Caveat from
  its ablations: most of the gain appears with reward balancing; a naive positive version is worse than random.
- **Learning progress gaze.** Kim et al. (ICML 2020) gamma-progress gaze controller; IAC; Graves et al. 2017 (uniform
  a strong baseline).

## 5. What to teach (P1), reasoned
A fixation does three things in M0: it allocates a third of the frame's G1 target (training signal), it is evidence
for the belief, and it is the query under which the glimpse is rendered.
1. **Physical predictions must not depend on the next fixation** (Codex's passive-view marginal consistency; the
   brain's remapping and foveal prediction; GQN / Jayaraman's decoder). With that property the gate cannot form,
   whatever the chooser. Its price: the fovea then reaches physical prediction only indirectly: through shared
   weights and the belief (B2's transfer, positive in sign and small), and through the representation (TSP teaches
   the periphery to carry what the fovea sees: P1-F's periphery now carries the digits at 12-19x, the brain's
   transsaccadic learning). The fovea will not be the main fix for the field's dynamics deficit (the user's
   "bonus"); that needs its own lever.
2. **What P1's looking then teaches is the view model at the action's locus**: what the eye will see at a window,
   given the periphery, the belief and the action (the brain's foveal prediction and transsaccadic learning; our
   TSP and glimpse G1). That is what P2 / P3 need: VoI in imagination requires predicting what a look would show,
   and the looks that matter for decisions are where actions act. P1-F's glimpse gains (increments 0.04 -> 0.73)
   were real view learning; only their leak into physical prediction was wrong.
3. **Where to look, as a priority over windows from pre-outcome inputs**:
   - an action-locus prior (Johansson): where the world's prediction of the next percept depends on the action taken
     (label-free; matching negatives by construction: it looks at attempts whatever the outcome). F: available from
     mid-training, finds the acted-on tile, blind to consequences displayed elsewhere (section 7);
   - a learned priority map for the remote consequences, trained with full feedback from the periphery on the
     realized action effect per window (section 7), so the eye learns to go where the effect will appear;
   - modulated by competence and progress of the view prediction there (Ten et al.: LP plus competence; IAC):
     stop looking where the glimpse is already predicted, keep looking where it is improving;
   - with a permanent random floor (Suau 0.1, IAC 0.35, HPM's random fraction): overlap for every context;
   - candidates set aside as primary: pure predictive uncertainty or surprise (noise capture: night static; F
     measures it), pure inverse-model reward (SUGARL's ablation), pure LP per look (B2: small, noisy; needs pooling).
4. **The hand rule, if used at all, as a decayed sidekick** (Ramakrishnan & Grauman), never at Q = 1. It reads
   simulator labels, so the generality rule argues for not using it.

## 6. Later: VoI (P2 / P3)
D found no substrate, and the reason is partly the environment and the capacity split: P1-F's periphery carries the
digits at 12-19x the within-value spread, so in Craftax daylight a glimpse adds little the periphery lacks. VoI is
bounded by the periphery / fovea capacity gap (Codex: beta TSP "does not enforce a useful periphery/fovea capacity
gap"). Whether that gap should be designed in is its own question; night content is where the glimpse's value was
(and fell, 0.26 -> 0.08% of the loss).

## 7. F: does the world itself locate the action's locus? (PLAN F, RESULTS F, e7cc201f)
The frozen world's action contingency (its predicted next periphery under DO minus under NOOP, from pure noise,
belief recomputed per action) peaks on the tile DO acts on from checkpoint 5 onward: wood 0.68, coal 0.64, stone 0.46,
failed stone attempts 0.49 (world 1, ckpt 10; P1-F 0.80-0.90), almost never the player's own cell. Mining is located
(coal 0.93, chance 0.27); consequences displayed only in the HUD inventory are not (drink 0.38, wood 0.25, sapling
0.22, chance 0.12), although the realized change shows them after the fact (drink 0.77, wood 0.42). Nothing is
located at checkpoint 2. Under a uniform query P1-F's gated counter consequences are invisible (wood 0.04).

Three consequences for the design:
- **The brain's contact-landmark prior is available label-free, from mid-training, and gives matching negatives**
  (it fires on failed attempts as on successes): look where the action acts.
- **Craftax displays many consequences away from the locus** (the inventory). The world's own predictions cannot point
  there until it predicts them, which is the deficit (A): circular. But the periphery sees the whole frame at every
  step: the realized action effect at every cell (the realized next periphery minus the world's prediction under
  the null action) is observable after the fact WITHOUT looking. A priority map over windows, from pre-outcome
  inputs, can therefore be trained with full feedback on every window, not bandit feedback on the looked-at one:
  the brain's learned "where the event will be" (cricket bounce point; signs at intersections; an indicator after a
  switch press), from peripheral detection to proactive foveation.
- **Early training has no prior**: random gaze first, the chosen share growing (HPM's schedule), a permanent floor.

## 8. Decisions this leaves, in the order they depend on each other (one at a time)
1. **Physical prediction independent of the next fixation, by structure.** Everything else assumes it. Open: how
   (periphery and lighting tokens not attending to the query fixation and glimpse tokens, the glimpse generated
   conditional on the periphery, as the brain's foveal prediction; exact joint sampling then needs a periphery-first
   order); whether the reward / continuation heads may read the glimpse percept (Codex: P1-F's reward readout is
   view-dependent with the real successor) or only the belief it updates.
2. **What the priority map is trained on.** Candidate: the realized action effect per window from the periphery
   (full feedback), plus the analytic contingency; modulated by the view model's competence / progress there. Open:
   night static in a realized residual (normalise by the null-action prediction's own spread?), token-scale drift
   early (B1), the null-action reference's generality (or the mean over actions: F's marginal variant was weaker).
3. **Schedule and overlap**: random-only early (F: no prior at ckpt 2), chosen share growing, a permanent floor; the
   floor's size is an empirical property of the structure chosen in 1.
4. **The field's dynamics deficit**: its own lever (outside the fovea).
5. **The sampler**: Codex's factorial makes step count and churn a separate, cheap design question.
6. **VoI's substrate**: the periphery / fovea capacity gap (section 6).

## Sources (read for this note)
Johansson et al. 2001 (https://pmc.ncbi.nlm.nih.gov/articles/PMC6763066/, abstract and discussion); Hayhoe & Ballard
2005 (https://www.cs.utexas.edu/~dana/Hayhoe.pdf, full text); Foley et al. 2017
(https://pmc.ncbi.nlm.nih.gov/articles/PMC5402436/, abstract, task, results); Gottlieb et al. 2013
(https://pmc.ncbi.nlm.nih.gov/articles/PMC4193662/); Ten et al. 2021 (https://pmc.ncbi.nlm.nih.gov/articles/PMC8514490/);
Wurtz 2008 (https://pmc.ncbi.nlm.nih.gov/articles/PMC2556215/, first half); Kroell & Rolfs 2022
(https://elifesciences.org/articles/78106, first half); Daucé et al. 2020 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7443118/);
Jayaraman & Grauman 2018 (https://arxiv.org/abs/1709.00507); Shang & Ryoo 2023 (https://arxiv.org/html/2306.00975v2).
Abstract level: Herwig & Schneider 2014 (JEP:G 143:1903); Itti & Baldi 2009 (Vision Res 49:1295); Najemnik & Geisler
2005 (Nature 434:387, not used above); Ramakrishnan & Grauman 2018 (https://arxiv.org/abs/1807.11010); Shah et al. 2020
(https://arxiv.org/abs/2006.07710); Pezeshki et al. 2021 (https://arxiv.org/abs/2011.09468). Earlier notes: SWEEP.md,
PROGRESS.md.

## 9. Revisions after the user's questions (10-31, late)
**Which fixation path is cut (code-checked, saccade/train.py p1_losses).** The fixation reaches the periphery by two
different paths. (a) Perception: TSP reconstructs the CURRENT crop from the CURRENT periphery tokens at the looked-at
cells (`model.tsp.predict(bp, Pc.p, Pc.v, M.window(fix))`, `Pc.p` not detached): where the eye looks is where the
periphery encoder is taught fine detail (P1-F's periphery carries the digits because the rule looked at counters).
This is the brain's transsaccadic learning (Herwig & Schneider 2014) and stays. (b) Dynamics: the denoiser's
prediction of the NEXT periphery also reads the NEXT (query) fixation; its targets are detached (`b.detach()`, prefix
under `no_grad`), so this path never trains the encoder's tokens; G1 reaches the encoder only through the belief.
Section 5.1 proposes cutting (b) only. With the glimpse generated from the predicted periphery (+ belief memory +
fixation), the glimpse's loss trains the predicted periphery to hold what the fovea will need, everywhere, because
the next fixation is unknown to it: more pressure on fine detail, not less.
**Why (b) carries no information.** Looking does not change Craftax, and the periphery sensor does not depend on gaze
(Codex asserts it). The next fixation is chosen from the belief and the action, which the denoiser already has; it
can only add a cue (a learned chooser) or leaked labels (the hand rule). In POMDP terms the fixation is a sensing
action: it enters the observation of the gaze-dependent part (the glimpse), never the transition. The joint
denoiser merges a world model and a sensor model; the factorisation separates them, as the brain does (remapping
and foveal prediction predict the gaze-dependent input from a gaze-invariant representation). Precedent for the
generative form: cascaded diffusion, a super-resolution stage conditioned on the low-resolution sample, with
conditioning augmentation "crucial" against compounding error (Ho et al., JMLR 2022). It also makes VoI computable
in imagination: what a look would reveal is the stochastic part of p(glimpse | periphery, belief, fixation).
**Look first or act first.** Both, in one fixation stream. In skilled manipulation gaze reaches the contact point
before the hand and leaves at the contact event (Johansson 2001): the look follows the decision of what to do and
precedes the outcome, and it both guides the movement and verifies it. While a novel visuomotor mapping is being
learned, "gaze typically pursued the cursor" (the action's visible consequence); as the hit rate rose, "gaze
fixations began to mark predictively desired cursor positions", and later peripheral vision sufficed for refinement
(Sailer, Flanagan & Johansson, J Neurosci 2005). Information-sampling looks precede the decision they inform (Foley
2017). In M0 the fixation for frame t is chosen with a_{t-1}: it lands on a_{t-1}'s outcome (monitoring, learning)
and feeds the belief that chooses a_t (information): P1's and P2 / P3's objectives are two uses of the same stream.
**Learning progress is not set aside: it is the modulation; action contingency is the location.** Brain: the
consequence-monitoring gaze of Sailer et al. ends as the mapping is learned; humans choose activities by progress plus
competence (Ten 2021); infants' looking away is predicted independently by surprise, predictability and learning
progress (Poli et al., Sci Adv 2020, abstract level). ML: IAC's progress is measured on predictions of the sensory
consequences of the agent's own actions in sensorimotor regions (Oudeyer et al. 2007): action-contingent progress is
IAC. Progress alone over 49 windows is statistically hard (B2: small, noisy per look); restricted to where the action
acts it has a structure for its credit, and under the factorisation the view prediction's progress at the looked-at
window is a direct signal (no longer transfer), measurable against fixed pixel targets (the real crop), which no
longer rewards gating because the physical prediction cannot gate.
**Prior art (none found combining them).** LP-driven gaze for world-model learning: Kim et al. ICML 2020. Attention to
the controllable region learned by inverse dynamics, on a 9 x 9 grid, label-free across eight Atari games ("contingency
awareness"): Choi et al. ICLR 2019. Sensory policy rewarded by inverse-model predictability: SUGARL 2023. A self-model
predicting the world model's error drives play (Haber et al. NeurIPS 2018; noise-prone, fixed by progress in Kim 2020).
Learned where-maps: Daucé et al. 2020; HPM 2023.
**The learned map's generality.** Any chooser other than a fixed rule is a learned map (the brain's gaze patterns
"must be learned", Hayhoe & Ballard). What must be general is its target. Revised proposal: the contingency term is
computed from the world itself (F; no extra head; a null action or the mean over actions as reference), the learned
part predicts the view model's progress (bandit feedback from the looked-at window: available online, not only on
recorded data). The full-feedback realized-action-effect head of section 7 is dropped as the default: progress
subsumes its purpose; it stays a fallback if remote consequences are found too slowly. Offline, P1 could evaluate any
window on the stored full-resolution frame; that would not carry over to an agent that only perceives a crop.
**Coherence P1 -> P2 / P3.** One priority map over windows: floor + contingency x (progress, competence) [learning]
+ VoI [decisions; covers exogenous, decision-relevant events]. The active-inference split (novelty vs salience,
Schwartenbeck et al. 2019) is the same decomposition, with novelty declining as the model is learned. Two conditions:
the learning term matters in P2 / P3 only if the world (or the view model) keeps training there (LEARNING_SKETCH froze
it during actor training); exogenous events that are learnable but not decision-relevant get only the floor (in
Craftax the mob deficits are the sampler's, A).
