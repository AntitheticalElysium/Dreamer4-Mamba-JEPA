# Progress of what? (reasoning, 10-31; for discussion, nothing decided)

User, 10-31: rewarding learning progress is intuitive; what the progress is OF is the central question, and the
research needed to qualify it has not been done. This note works through it from what the world model is for, then
names what must be measured before a choice (B, redesigned).

## 1. The question made precise
The chooser picks the window w of frame t from pre-outcome inputs. Choosing w does three things: the glimpse of w
becomes evidence (it enters the belief and the denoiser's context for the next frames); it becomes a training target
(9 x 64 of the frame's 1,880 next-percept values, plus TSP's and the reconstruction's pixels at w); and it is the
query under which the denoiser generates frame t. "Learning progress" is a change in some performance P, measured on
some distribution mu, caused by training. Every published signal is an estimate of Kim et al.'s r = L_mu(theta) -
L_mu(theta'): the gain in the TARGET loss on the TARGET distribution. "Progress of what" is therefore four choices:
which performance, on which distribution (of states and of gaze), in which space, and attributed to the choice how.

## 2. Which performance: physical and view predictions
The world is for the actor: imagined percepts update a belief, heads give rewards and terminations, and later the
actor chooses where to look. The world's predictions split in two:
- **physical**: what is on the screen and what happens (periphery tokens, lighting, reward, termination). Looking
  has no physical effect in Craftax, so these must be the same wherever the eye goes;
- **view**: the glimpse rendering at the requested window, legitimately gaze-dependent.
Failure 3 is a physical deficit, measured in the periphery under random gaze. P1-F's gain was a view gain that leaked
into a gaze-conditional physical prediction (Codex: the wood gain 83.6% at the rule's window, 7.3% one row up with
the whole counter in view). A progress signal on the view prediction at the chosen crop would have rewarded exactly
that: the glimpse increments went 0.04 -> 0.73 while the gaze-free prediction went to 0. So progress on the view
alone cannot be the answer. Progress on physical predictions, measured with the eye placed independently of the
chooser, rewards only what holds wherever the eye goes. View quality still matters later (VoI needs faithful
glimpse renderings in imagination), so it is a secondary candidate, not a discard.

## 3. Which distribution
States: the corpus. Gaze: a reference distribution independent of the chooser (the uniform prior world 1 trained
under), so that gaze-conditional knowledge earns nothing. Region: the content the chosen crop covers, in contexts like
the current one (IAC's lesson: progress compared within similar situations, never across activities). The
alternative, the chooser's own gaze, makes the target move with the policy, which is how policy confounding arises
(Suau et al.).

## 4. Which space
- **Token targets move.** Targets are the online encoder's tokens (stop-gradient, not an EMA). Prediction gain
  compares two parameter states; if each is scored on its own targets, an encoder that simplifies or re-allocates its
  tokens looks like progress, or hides it. How large this is in our runs is unmeasured. Precedents for keeping a
  latent target still: BYOL-Explore predicts "an older copy of its own latent representation"; RND uses a fixed random
  target network; gamma-progress's old model is an exponential moving average of all the weights.
- **Pixel targets do not move but cover only the crop** (no full-frame reconstruction, DECISIONS 10-25), and the
  pixel losses we have do not measure dynamics: TSP predicts the crop from the SAME frame's periphery tokens plus the
  belief (super-resolution), the reconstruction autoencodes the crop. A fixed-target view measure does exist: the
  denoiser's predicted glimpse rendered through the pixel decoder against the real crop.
- **Reward and termination are fixed, label-free and decision-relevant**, read by the heads on imagined successors
  generated under the reference gaze. Sparse (unlocks ~3.5% of frames), and blind to consequences without reward
  (drink increments carry none; health changes carry 0.1 per point).
- **A measurement readout** (stop-gradient) from predicted periphery tokens to the 3 x 3 pooled pixels they encode
  would give physical progress fixed targets. It approaches the pixel-equivalent state the design avoids, though only
  as an instrument; flagged, not proposed.

## 5. Attribution to the choice
Regional progress measured over time mixes the effect of looking with everything else the model learns (the bulk
improves every region). The causal quantity is the difference looking at w makes to the target loss compared with a
reference draw. To first order it is the alignment between the gradient the choice ADDS, grad l(x; q = w) - E_w' grad
l(x; q = w'), and the gradient of the target loss on fresh samples: the influence estimate of TracIn (Pruthi et al.
2020) that LESS (Xia et al., ICML 2024) adapts to Adam. Graves's gradient prediction gain is its self-aligned special
case, ||grad||^2, which is why it favours noise: irreducible noise aligns with itself and with nothing else, so
alignment with FRESH samples does not reward it. Stochastic events (sapling) earn progress while their probability is
being learned and none after, as they should. First order sees only routes through the parameters the target
reaches directly (the denoiser and, through memory tokens, the belief); the representation route (pixel grounding
at w changing the encoder, hence the targets) is second order and invisible to it.

## 6. Candidates (not ranked)
1. View progress at the crop (glimpse tokens, or decoded crop pixels): cheap, credit exact; rewards gating; tokens
   drift (pixels do not).
2. Physical progress on the covered content under the reference gaze, on fresh samples of the region, as
   gamma-progress (old = an EMA of the weights): the closest to the goal; depends on token drift; credit by region.
3. First-order counterfactual (gradient alignment of the choice's added signal with the physical target): causal and
   immediate; gradient costs; Adam preconditioning; blind to the representation route.
4. Fixed-target decision progress (reward and termination of imagined successors under the reference gaze): fixed
   and actor-relevant; sparse; misses unrewarded consequences.
5. A combination: physical primary, view (in pixels) secondary.
A possible outcome to face honestly: if looking at a consequence barely moves the gaze-free physical prediction, a
physical-progress chooser has no reason to look there, the fovea's training role cannot reach failure 3, and the
field deficit needs another lever. Graves found uniform sampling "a surprisingly strong benchmark".

## 7. B, redesigned: what to measure before choosing (proposal; needs the user's go-ahead)
Frozen checkpoints 1-10 of world 1 (random gaze) and P1-F (rule gaze), dev frames, GPU, reading only. Consecutive
checkpoints are 1,562 updates apart, close to gamma-progress's horizon (gamma 0.9995: 2,000 updates).
- **B1. Token drift.** Each checkpoint encodes the same frames; per token group (map, HUD, glimpse, lighting), the
  part of checkpoint k+1's tokens that a linear map from checkpoint k's cannot predict, against the held-out G1 change
  over the same interval (logged monitor). Claim: drift is small next to progress after the first checkpoints.
  Competing: re-allocation is comparable to the G1 change. Rules out token-space progress (candidates 1-3 in tokens)
  where drift dominates.
- **B2. Transfer.** At each checkpoint, per consequence class (drink, wood, sapling; stone as an unguided control),
  the alignment between the added training signal of the consequence window (rule window minus a uniform draw, same
  frames, common noise) and the gradient of the gaze-free loss of the consequence cell (uniform query fixation) on
  FRESH frames of the class; specificity: alignment with other classes' targets; noise floor: split halves. Claim:
  looking at a consequence pushes the gaze-free prediction of it (positive, specific alignment), at some stage of
  training. Competing: no transfer (alignment within noise), or transfer only before the gaze cue forms (P1-F's later
  checkpoints). Rules out the premise that a fovea's training signal can reach failure 3 through the denoiser, if
  null; says nothing about the representation route.
Not chaining: B is one reading with two instruments. Nothing about the chooser's form is selected by it.

## Sources
Kim et al. 2020 (https://arxiv.org/abs/2007.07853, sections 4.1-4.2 read); Graves et al. 2017
(https://arxiv.org/abs/1704.03003); Oudeyer, Kaplan & Hafner 2007; Suau et al. 2024; BYOL-Explore (Guo et al.,
NeurIPS 2022, https://arxiv.org/abs/2206.08332, abstract level); RND as described by Kim et al.; TracIn (Pruthi et al.,
NeurIPS 2020) and LESS (Xia et al., ICML 2024, https://arxiv.org/abs/2402.04333, abstract level); shortcut preference
for simple cues (Shah et al., NeurIPS 2020, https://arxiv.org/abs/2006.07710; Geirhos et al., Nature Machine
Intelligence 2020, https://arxiv.org/abs/2004.07780; abstract level); Codex's REPORT.md (fixation gate).
