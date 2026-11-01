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

## 10. Decision 1, proposed (11-01, for review): one denoiser, factorised by structure; the reward head unchanged
**The shortcut and labels.** The hand rule read simulator labels, but the gate does not need them: any useful chooser
makes the next fixation predictive of the outcome (that is its job: look where a consequence is coming), so a
denoiser that reads the query fixation can use it as a cue, whoever chose it. Labels made the cue perfect (Q = 1, the
exact window); randomness only dilutes it. A label leak beyond that (e.g. facing direction) is possible and untested;
the factorisation removes both.
**What changes** (GENERATOR v3 / STEP_SPEC 7 tokens: 81 periphery, 9 glimpse at the fixation's cells, 1 lighting, K
memory; spatial attention within a frame, temporal attention per token index):
- In spatial layers the physical tokens (periphery, lighting, memory) never attend to glimpse tokens; glimpse tokens
  attend to all. Memory tokens are included: through them a glimpse could otherwise reach the periphery within a layer.
  Temporal attention is per token index, so no cross-frame route remains.
- The glimpse group gets its own noise level in training, drawn independently of the frame's, with per-group noise
  conditioning (adaLN per token group instead of per frame). UniDiffuser (Bao et al., ICML 2023, method read): one
  transformer, an independent timestep per modality; "a maximum timestep T means marginalizing it", "a zero timestep
  means conditioning on the corresponding modality", "a tied timestep means sampling two modalities jointly".
- Sampling: periphery and lighting first (3 Euler steps; glimpse tokens omitted, they cannot affect them), then the
  glimpse given the finished periphery at a small conditioning-augmentation level, 3 steps over 9 tokens with the
  physical tokens' keys / values computed once. Cascaded diffusion (Ho et al., JMLR 2022, section read): conditioning
  augmentation "alleviates compounding error from a train-test mismatch" and is "useful precisely when conditioning on
  generated samples"; its level is a post-training choice (there: s = 0-1501, best 501-751).
**What it guarantees.** p(periphery, lighting | history, action) cannot depend on the next fixation (unit-testable: the
gradient of the physical outputs with respect to the glimpse inputs and the fixation is zero). The joint stays
coherent by the chain rule, p(p') p(g' | p', f), an exact factorisation under two environment facts (looking does not
change Craftax; the periphery sensor is gaze-invariant, Codex's `torch.equal` check). GENERATOR v3's reason for joint
denoising ("so they stay correlated") is kept, in a stronger form. It also gives VoI its object: the glimpse stage
answers "what would a look at w show in this imagined frame" for any w, from one sampled periphery.
**What it costs.** (1) One route: a past glimpse reaches the next periphery only through the belief (memory tokens)
and the glimpse stream, no longer through the past frame's periphery tokens. Bound from D: the glimpse's whole
one-step value is 0.08-0.6% of the next-frame loss. The belief's uptake of glimpses becomes a standing reading (the
fovea probes found it kept nothing). (2) Sampling: 3 extra passes over 9 tokens. (3) Code: the mask, per-group
conditioning, a per-group sigma draw in the denoising loss, the sampler's order. (4) From scratch: trained worlds do
not transfer.
**Considered: the mask alone, tied noise, joint sampling.** The periphery marginal stays exact (its flow never reads
the glimpse), but the glimpse is only approximately consistent with it (the joint flow needs E[p | x_p, x_g], the
mask gives E[p | x_p]), and the conditional query VoI needs is not available. Rejected for the system's sake.
**Unchanged.** TSP and reconstruction (the encoder's fine-detail teacher, section 9); the belief reading glimpses.
**Reward and continuation heads: no structural change.** They read the observed percept, glimpse included: a
posterior p(r | evidence). Looking legitimately sharpens a posterior (seeing the counter tick); the failure Codex
measured is different: with the real successor and the counter visible at both crops, P1-F's reward NLL was 0.0036
at the rule's window and 0.35 one row up, so the head keyed on the window's position, not on the digit. That is a
calibration failure from deterministic placement (the same simplicity bias), not dependence on evidence. The
requirement is marginal consistency, E over looks of p(r | p', g', f) = p(r | p') for every way of choosing f,
which a calibrated head on a calibrated glimpse stage satisfies. It is obtained by data (non-deterministic gaze, the
floor) and checked by Codex's crop test as a standing reading. It matters for P3: an actor choosing fixations could
otherwise raise imagined reward by looking (Goodhart through a miscalibrated head). Blinding the head would also cost
the cases where only the fovea sees the reward's cause (night).

## 10a. Decision 1 after Codex's review (11-01; user agreed decision 1; `../20261101_fovea_factorization_review/REVIEW.md`)
Verified against the code, point by point. The decision stands; four claims of section 10 are withdrawn or narrowed.
- **D does not bound the cost of the cut (conceded; my error).** `20261029_fovea_probes/analyse.py` R2 is the
  per-frame best fixation minus the best FIXED fixation, a glimpse present in both: it prices choosing, not having,
  a glimpse, and says nothing about the removed route or rare events. Withdrawn: "the glimpse's whole one-step value
  is 0.08-0.6%". The cost of the route is unmeasured.
- **The route itself: kept cut, as an argued bet, with Codex's test standing.** The mask must hold in every frame
  (diffusion forcing makes each frame both a target and later context, so a past periphery token that read its own
  glimpse would also have read its frame's query). The system argument for routing past evidence through the belief
  only: the actor reads the belief (STEP_SPEC 1, 5). Evidence held only in the denoiser's context reaches a decision
  in imagination solely through what it makes the generator draw within its 8-frame reach, and not at all when the
  agent acts in the real environment, where no denoiser runs and only the belief carries it. Cutting it puts G1's pressure
  (night, occlusion: next-frame periphery the glimpse could inform) on the belief, which is where D found no uptake.
  Standing reading on the first factorised world (Codex): informative past glimpses (night entities seen only in the
  glimpse) improve the next physical predictions through the belief. If the bet fails, the fallback is a strictly-past
  route: in temporal layers, physical tokens of frame t also attend to the glimpse tokens of frames t-4..t-1 (36 extra
  keys per query, causal by construction, cached in streaming); its cost and chunk/stream equality to be checked.
- **"Exact coherent world" (narrowed).** The chain rule makes q(X | H, a) q(G | X, H, a, f) a valid joint for ONE
  query; the learned conditionals, the 3-step sampler and conditioning augmentation are approximate. Several views of
  one imagined frame would need a coupling (CGQN); our interface never takes two looks in one frame, and VoI compares
  looks in separate branches, which needs only the per-query marginals. Successive frames' glimpses stay consistent
  through the committed context and the belief, a learned property to measure.
- **Training contract (conceded: unspecified; my section 9 sentence withdrawn).** In training the glimpse group sees
  the REAL periphery tokens noised at the physical group's sigma (targets detached, as now). Its loss trains the glimpse
  conditional and the denoiser's shared weights, including the hidden features of physical tokens (glimpse queries read
  them: the route B2 found carrying ~90% of the transfer); it does not reach the periphery encoder or the generated
  periphery. Withdrawn: "the glimpse's loss trains the predicted periphery to hold what the fovea will need". TSP stays
  the encoder's fine-detail teacher. Noise: sigma_p and sigma_g drawn independently from the current log-normal, each
  group's adaLN from its own sigma (memory tokens with the physical group), never sigma = 0 (EDM's c_noise = log(sigma)/4
  and the division by c_out): stage 2 conditions on the generated periphery noised at a declared s_aug > 0 with
  c_noise from s_aug. Coverage of that regime under the current schedule: P(sigma <= 0.1) = 0.056, (<= 0.2) = 0.157;
  whether to raise it is an implementation choice read on the real-vs-generated conditioning gap (Codex), not decided.
- **Reward head (narrowed requirement, Codex's).** Pooled calibration under one gaze policy is not enough: for EVERY
  fixed f, the reward / termination distribution integrated over generated glimpses at f must be the same,
  integral q(r, d | H, a, X, g, f) q(g | H, a, X, f) dg independent of f. Standing test: per legal f, K glimpses per
  context, the spread of the per-f expectations against sampling error; plus the real-crop counter check. Overlap makes
  this learnable, not guaranteed. Narrowed: the crop result shows view dependence; "keyed on the window's position" is
  not established (the other crop changes content and position together).
- **Label leaks (narrowed).** The factorisation removes the query from the physical marginal only; the glimpse stage and
  the outcome heads still read f and can learn the logging policy's correlations. Checked by glimpse-periphery
  coherence at every f and the reward test above.
- **Checks before calling it built** (Codex's list): physical outputs equal across all legal next fixations under common
  randomness, on a model with NONZERO output weights (zero-initialised heads pass a broken mask vacuously; the
  fixation is an integer, so "a zero gradient" was the wrong test); no query path through memory, modulation or caches;
  chunked = streaming; cost of stage 2 measured (the physical tokens' per-layer keys / values computed once, 3 passes over
  9 glimpse tokens, one commit per frame, Mamba advanced once).

## 11. Decision 2, draft (11-01; for review; the open choices wait for G, PLAN G)
**What a look can teach once decision 1 holds** (10a's contract). A look at w trains: the glimpse stage at w (forward
from history when the physical group is noisy; super-resolution of the periphery when it is nearly clean), TSP at
w's cells (the periphery encoder's fine detail), the fovea encoder (reconstruction), and it hands the belief evidence;
the reward / continuation heads read it. The gaze-free physical stream is trained on every cell wherever the eye is;
a look reaches it only through shared denoiser features (B2's ~90% route: small, positive in world 1, rarely
significant), TSP's representation and the belief.
**Of what (proposed): the error of the predictions the look trains, against the crop's pixels, measured where the
eye actually looked, before training on it.**
- Gaze-free physical progress is the deficit (failure 3) but is not the chooser's to buy under decision 1; rewarding
  it would reward watching the world learn. It stays a standing reading (B2's transfer), and the field's deficit
  stays decision 4. This answers Codex's "view progress is not automatically progress in gaze-free dynamics": under
  decision 1 it should not be.
- The user's 10-29 principle (look where the model can improve most, where it is starved) survives intact, applied to
  what a look feeds: the fine view and the periphery's detail, where P2 / P3's VoI will read them.
- Pixels, not tokens: token targets move early (B1); pixel targets are fixed and are faithful's training metric.
- Prequential, at the looked-at window: the error before the update is a fresh sample (Graves et al.: same-sample
  gain favours noise), and an agent never sees other windows at full resolution (bandit feedback, not offline
  scoring of every window on stored frames).
- Decision-level progress (imagined reward, termination) is VoI's, P2 / P3.
**How (proposed): a learned competence estimator and its lagged copy.** e_hat(b, a, w): a small head on the predicted
belief, the action and a window embedding, regressed on that prequential error (no gradient into the world:
STEP_SPEC 2); progress = |e_hat_now - e_hat_old|, the old copy an EMA of the head's weights. Precedents, read:
- MAGELLAN (Gaven et al. 2025): an MLP on the agent's own latent predicts competence per goal; absolute progress is
  the difference from a stored past copy of the estimator (100 snapshots, 32 updates apart); goals sampled in
  proportion to it with epsilon annealed 1 -> 0.2; generalises across goals (Online-ALP's generalisation error 0.53
  vs 0.11) and was the only method without expert groupings to master its whole goal space.
- Kim et al. (ICML 2020): gamma-progress, the loss of an EMA "old" model (gamma 0.9995; 0.9995-0.9999 similar) minus
  the current one, on the agent's recent data, as a DQN gaze controller's reward; end performance over random 7.83x
  (mixture world) and 13.79x (noise world), against RND 1.25 / 0.85, disagreement 0.62 / 0.76, error-seeking 0.62 /
  0.59; noise fixation 0 of 8 behaviours (error-seeking 8 of 8); attention to animate agents emerged in 7 of 7, then
  faded ("lose interest") once learned. With an oracle object encoder: our representation-learning case is untested.
- Kanitscheider et al. (2021): progress = fast minus slow EMA of success (time scale 1,250 updates); the absolute
  (bidirectional) version, because the one-sided one caused forgetting cycles; 82 vs ~70 Minecraft items for the
  strongest uniform baseline (one run each).
- Ten et al. (Nat Commun 2021): human free choice best fit by a softmax over w_PC x competence + w_LP x |progress|
  (15-trial windows); progress-driven participants avoided the unlearnable activity and mastered all three learnable
  ones 64% vs 35%.
Noise: a stationary error (night static once its level is learned, sapling's coin once its rate is) gives e_hat_now
= e_hat_old, so no pull: Kim's noise world and Ten's unlearnable activity.
**Location prior: the action contingency (F), additive, never a gate.** Codex: an exogenous learnable event must stay
reachable, so priority = floor + softmax(lambda_C contingency + lambda_LP progress), not a product. Brain: gaze goes to
the contact landmark (Johansson 2001), and after a few action-effect pairings people make anticipatory saccades to
where their action's effect will appear, even 14 degrees away, before it appears, "a comparison of expected and
actual effects" (Pfeuffer et al. 2022 online replication; Pfeuffer, Kiesel & Huestegge 2016 in the lab; 10-month-olds:
Kenward 2010): the remote case is a LEARNED association, which is what the progress estimator would learn while the
view there is improving. Cost: two extra denoiser passes per frame (taken action vs NOOP). Open, for G: whether
progress already concentrates at the action loci (then contingency is redundant as P1's location signal) and at the
remote counter where contingency is blind.
**Floor** (decision 3's size, this decision's requirement): per-fixation reward calibration (10a), the glimpse stage's
coverage for VoI, and policy confounding (Suau et al.) all need overlap in every context: MAGELLAN 0.2, Suau 0.1,
IAC 0.35, Kim's epsilon decayed to 0.025.
**Relation to P3** (Codex's open item). With the world frozen in P3, this chooser is P1's; its product is a glimpse
stage and a periphery trained where the loci and the progress were, plus the floor everywhere, which VoI then reads.
If the world keeps training online (the user, 10-31: keep learning fixation at a reduced rate), the same estimator
continues beside VoI.
**What G decides.** No structured progress after the first intervals: drop the learned part (contingency + floor).
Structured but tracking error: the irreducible correction is needed. Located only where contingency is: progress as
the modulation, contingency as the location. Located at the counter as well, and moving: keep both, progress as the
learned location for remote consequences.

## 11b. Decision 2 after G (11-01; proposed, for the user's call; RESULTS G 092dc4b5)
**What G showed.** Under random looks the progress of what a look trains (the forward glimpse, TSP) is replicable and
not pulled by night static, but: it locates action consequences no better than the world's own action contingency
(world 1 mining, arg-max coverage 0.31-0.61 against F's 0.58-0.95 on the same rows), it does not move toward them
over training (front-loaded with everything else in the first third: views flat from checkpoint 4-5, TSP converged),
and in the second half it reverses sign between consecutive intervals 625 updates apart (post hoc; corrected 11-01:
not gamma-progress's 2,000-update horizon).
This fits decision 1's accounting: a starved consequence is starved in the gaze-free physical stream, which a look
cannot buy; what a look can buy (fine view, periphery detail) is learned early and broadly.
**Proposed decision 2.** P1's chooser = the world's action contingency (F; additive, computed from the physical
stream under decision 1) with a permanent random floor; no learned progress map in P1.
- Why not the progress map: G gives it no location value beyond contingency in the clean world, no movement toward the
  starved events, and late at its natural horizon it would chase reversals; Graves et al.'s "uniform is a surprisingly
  strong benchmark" and Kanitscheider's modest margin over a strong uniform (82 vs ~70) point the same way. It would
  add a head, an EMA copy and a schedule for nothing measurable.
- Why contingency: it locates the action's locus label-free from mid-training, failures included (F), which is where
  P2 / P3's VoI will need the glimpse stage to be good (the brain's contact landmarks, Johansson 2001). Before it is
  informative (F: flat at checkpoint 2), a softmax over it is near-uniform by itself (to be checked when built,
  decision 3).
- What stays of the user's progress idea: as a standing READING (is the glimpse stage still learning, where; G's
  instrument), and as the candidate for later if the world keeps learning online in P2 / P3, where the data stream is
  no longer fixed. Not as P1's driver. The honest limit: G is a frozen landscape under random looks; a chooser that
  follows progress changes the data and could create the positive feedback G cannot show (Kim et al.'s attention
  "bump" then "lose interest"); testing that needs a run, and G gives no reason to expect it to beat contingency.
- Remote consequences (shown away from the locus): contingency is blind there until the world predicts them (F), and
  G could not test progress there cleanly (world 1 digit-blind, P1-F gated). In Craftax the periphery carries the HUD
  (12-19x), so the physical stream gets these consequences without a look; the floor covers the glimpse stage there
  (6 of 49 windows cover a given counter). The general case (a remote effect the periphery cannot resolve) is the
  brain's learned anticipatory saccade (Pfeuffer et al.); its candidate is section 7's realized-effect map, added on
  measured need.
- Costs: contingency needs two extra denoiser evaluations per training frame (taken action vs NOOP) or a cheaper
  amortised form (implementation, decision 3).

## 12. Learning progress revisited (11-01, user: "I'm not sure we should be dropping learning-progress so easily"; papers re-read in full where quoted)
**11b withdrawn as stated.** G tested the weakest form of the idea: change in error between checkpoints, under
uniform looks, credited to no look. None of the validated forms is that.
**What the literature measures, and where it works** (read: Lopes & Montesano's survey section 2.6-2.7 and the
strategic student; Graves et al. 2017; Kim et al. 2020; Mindermann et al. 2022; Evans et al. 2024; Jiang et al. 2021;
Kanitscheider et al. 2021; Gaven et al. 2025; Poli et al. 2020; Dubey & Griffiths 2017):
- **Progress is defined against the samples that cause it.** The empirical measure is "L(T^{-k}; D) - L(T; D): to
  which extent the last k experiences help to learn a better model as evaluated over the complete data" (survey
  2.6.4). In the strategic student the optimal policy probes "the task whose learning curve has an higher derivative",
  the curve against the time allocated TO THAT TASK; the optimum is non-stationary (simple topics first), some
  choices are never studied with little time, and greedy is near-optimal (1 - 1/e) only for diminishing-returns
  problems; "online learning methods introduce bias since the order of the data changes the results". Graves et al.:
  prediction gain on the batch just trained (before vs after that step) was the most consistent signal; uniform
  sampling is strong because "learning is dominated by gradients from the tasks on which the network is making
  fastest progress" (an implicit curriculum, with "the inefficiency of unnecessary samples"). Poli et al.: infants'
  looking is predicted by the information gain of each observation (KL between successive predictive estimates;
  effect 7.0 against surprise 2.4, predictability 1.3).
- **A second family scores learnability against a reference** and is the strongest record for selecting training
  data, which is what P1's fovea does (it chooses which crop of a stored frame becomes a target). RHO-LOSS selects
  "the points that most reduce holdout loss", approximated by training loss minus an irreducible-loss model's
  (trained on held-out data, never updated, may be small): noisy points have high irreducible loss, redundant ones
  low training loss; Clothing-1M reaches uniform's 50-epoch accuracy in 2 epochs; CIFAR-10 with label noise 75% in
  27 vs 62 epochs. JEST (learner loss minus a pretrained reference's): uniform's final performance with 3x fewer
  examples (90% filtering), up to 13x fewer iterations. PLR (value-loss score plus staleness, rho = 0.1): +28%
  normalised test return on Procgen, against +3% for a temporal-progress curriculum (TSCL Window); gains "only occur
  when P_replay considers both level scores and staleness".
- **The rational form** (Dubey & Griffiths 2017): the value of learning stimulus k is dV/dh_k = p_k dc_k/dh_k, the
  need probability (how likely the knowledge will be needed) times the improvement per exposure. RHO's holdout
  objective and Kim et al.'s validation-loss objective are the same quantity on a target distribution.
**What we were missing.**
1. Credit and clock: progress must be measured per look (prediction gain, gamma-progress on the looked-at data,
   learnability), not as a checkpoint derivative of a uniform world. Within a frame G's ranking equals the per-look
   one only to first order (all windows equally sampled), and its checkpoint differences carry everything else the
   world learned and the stream's order (Codex: 2,500 then 625 updates per interval).
2. Allocation feedback: a starved (context, window) gets ~1/49 of rare frames, so its progress against GLOBAL time
   is small however steep its curve against its own samples. Allocation is the point. P1-F is the natural experiment:
   Q = 1 looks on the rule's window drew the counter's view there 0.98 vs 0.14 at another covering window (Codex's
   crop test; gated, but it shows looks determine where the view learns).
3. The learnability family (absent from SWEEP / PROGRESS / 11): robust to noise by construction (the reference
   carries it), free of the lagged-estimator artefact Codex raised (no difference of the estimator's own outputs),
   and natural in offline P1, where all 49 windows of a stored frame can be scored (full feedback). G's arrays: the
   hindsight learnable error (e_k - e_10, same world) is concentrated (top 5 windows hold 27-54% of a row's, uniform
   10%), increasingly redundant (half the windows hold < 5% of the row's maximum by checkpoint 5), and at night it is
   about half of raw error's share in world 1 (dark frames 19-25% of learnable vs 40-49% of error).
4. Need: progress alone answers "can I learn it", not "will it be needed". The contingency map is a need prior
   (where decisions will want the view), not a competitor: priority ~ need x per-look learnability, need = base +
   contingency, so exogenous events keep the base (Codex's additive point is kept by the base).
**Limits of the families for us.** Learnability can only point at what the reference has learned (a reference trained
under the same starvation is blind to the starved); temporal progress can discover them but only with a floor and
is noisy per look (B2); both need need-weighting to stay on what the actor will use.
**Test H** (PLAN H, running): the causal premise both rest on, crop selection speeding look-local learning, on frozen
P1-F inputs with fresh TSP heads (super-resolution and forward), arms uniform / learnability / error / progress.

## 12b. After H (11-01; for discussion with the user, nothing decided)
**What H adds.** The selection premise holds in its learnability form: choosing which crop a look-local predictor
trains on by learnability (current loss minus an irreducible-loss reference, RHO-LOSS) makes it learn faster in the
coverage regime P1 operates in (~10 crops per frame: H's step ~1,250; 16-43% lower loss for super-resolution, 6-9%
for the forward view, 28-52% at the DO consequence windows for super-resolution), and is never worse at the end; the
reference is what keeps it safe on the stochastic forward target, where plain error-seeking trades the consequence
windows (-11%) against everything else (+17%). Temporal progress (gamma, bidirectional, greedy over 49 windows) is
harmful here: 80% of looks on the HUD. Neither learnability nor error finds the action's locus by itself (faced tile
on DO frames 19-20% vs uniform 18%): location stays contingency's (F).
**So the user's idea survives in its learnability form, and composes with contingency as need x learnability:**
- priority(w | frame) ~ need(w) x learnability(w), with a permanent uniform floor; need = base + action contingency
  (Dubey & Griffiths' need probability; the base keeps exogenous events reachable, Codex); learnability = current
  look-local pixel loss minus a reference's (RHO; JEST's "pretrained reference" can be P1-F's own TSP / forward heads,
  in pixel space, independent of the new world's tokens);
- P1 is offline: every window of a stored frame can be scored (full feedback; cheap for pixel heads: H scored 512 x
  49 windows per step); an online agent needs a predictor of the score from pre-outcome inputs, trained by regression
  on P1's full-feedback labels (no lagged self-difference, so Codex's estimator artefact does not arise);
- floor: per-fixation reward calibration and policy confounding still need overlap.
**Open, in order:** (1) whether need-weighting changes what learnability buys (H had no contingency arm); (2) the
joint-system test (decision 1 built; P1-F2 arms on the factorised world: need x learnability + floor against
contingency + floor, uniform if the user wants the baseline); (3) the reference's blind spot (it cannot point at what
it never learned; the floor and contingency are what reach the starved).
