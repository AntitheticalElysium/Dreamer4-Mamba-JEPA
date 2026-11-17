# Design discussion that led to this gate (2026-11-11)

Kept verbatim so we can refer back to it as results come in. Part 1 is the research pass (Claude, after re-reading the
thread since 11-07); part 2 is the follow-up after the user's questions about the change map, the rollout, the actor
and the shortcut findings. Part 3 records what the user asked us to keep in mind for later. Codex's review of part 1 is
summarised at the end of part 2 (its corrections were accepted).

---

## Part 1. Research pass: is the timer too small to learn?

I've read across world foundation models, language models, robotics, learning theory and the brain. First I re-read our thread from 11-07, including Codex's 11-08 answers to the three questions my usage cut off. The headline: **the timer isn't small; our losses make it small.** That reframing, and one finding that keeps recurring outside Craftax, change how I read every arm since 11-08.

### Is the timer too small to learn?
**In information, no.** A model that doesn't track the four awake counters pays the surprise of every drop. I took the periods from the Craftax-Classic source:

| Counter | Awake period | Cost per awake step if untracked |
|---|---:|---:|
| Food | 26 | 0.163 nats |
| Drink | 21 | 0.191 nats |
| Energy | 31 | 0.143 nats |
| Recovery | 26 | 0.163 nats |

Together that's 0.66 nats, about one bit per awake step. That is not a small signal.

**In our loss, yes.** G1 is a squared error averaged over 1,880 latent numbers per frame. The food token is 16 of them and changes on 1 step in 26, so it makes up about 0.03% of what G1 averages over. Codex measured 0.06% for sleeping damage. Two more of our choices shrink it further:
- **The ramp weighting.** Dreamer 4's ramp, which we copied, gives the pure-noise targets a tenth of the weight of near-clean ones. Those pure-noise targets are the ones where the denoiser can't read the answer off its noisy input and has to consult the belief.
- **The surprise-based losses all see the answer.** Reward, continuation and the change map price events by surprise, but each one reads the real next frame.

Papers do discuss this, mostly outside world models:
- **Language models.** [LongPPL](https://arxiv.org/html/2410.23771v5) (ICLR 2025) shows that averaging the loss hides the few tokens that need long context; under 10% gain much from it. Long-context skill ends up undertrained. [STRING](https://arxiv.org/html/2410.18745v1) finds the same for long distances, which are rare in training data.
- **Scaling theory.** In the [quantization model](https://arxiv.org/abs/2303.13506v3), skills are learned roughly in order of how often they reduce the loss, so rare ones come late.
- **Recurrent-network theory.** The ["curse of memory"](https://arxiv.org/abs/2009.07799) papers (also [Zucchet & Orvieto](https://arxiv.org/abs/2405.21064)) show gradient descent slows sharply as the time lag grows.
- **World models.** [OC-STORM](https://arxiv.org/html/2501.16443v1) notes that a pixel squared error "is dominated by large areas", so small decision-relevant objects get missed.
- **Diffusion.** [Kingma & Gao](https://arxiv.org/abs/2303.00848) show common diffusion losses are weighted likelihoods, and the weighting decides what gets paid for.

None of these says "unlearnable". They say late, slow, and only if the loss pays for it. Our checkpoint audit fits: across 10 checkpoints the best timer fit was R² 0.0007, with no trend.

### A precise local shortcut starves the memory
This is the finding I'd most want us to reconsider from. Separate fields hit our exact setup and its failure:
- **[SWAX](https://arxiv.org/pdf/2509.24552)** (Meta, 2025) studied language models mixing a short attention window with a recurrent memory.
  - Longer windows made long-range recall worse. Their explanation: short windows push the recurrent layers "to receive more supervisory signal".
  - Their fix: each batch, randomly use a short (128) or long (2048) window, long only for the last 10% of training, and long at inference. It matched or beat both fixed windows.
- **[Po et al.](https://arxiv.org/html/2505.20171v1)** (ICCV 2025) is essentially our architecture: a state-space memory plus 10-frame local attention, in a diffusion video world model.
  - They write that redundancy lets models "rely primarily on nearby frames", leaving them "trapped in local minima".
  - Their fix keeps a random clean prefix and noises a long suffix after it. Without that, the model "generates frames without looking at distant context".
- **[Gregor et al.](https://ar5iv.labs.arxiv.org/html/1906.09237)** (DeepMind, 2019): "the immediate past weakens the dependency on belief-state vectors." Belief states formed only when they predicted several steps ahead from the belief alone, through an expressive generator.
- **Robotics.** [Past-token prediction](https://arxiv.org/html/2505.09561v2) (Finn's group, 2025) found that diffusion policies given long histories depend on them 10–100× more weakly than the experts did. Also training them to predict their own past actions raised success from 0.17 to 0.89 on one task and from 0 to 0.75 on another.
- **Theory.** [Bachmann & Nagarajan](https://arxiv.org/html/2403.06963v3) and the [Belief State Transformer](https://arxiv.org/html/2410.23506v3) show two things:
  - once a shortcut fits most of the loss, the leftover signal can be as hard to learn as parity;
  - neither next-step nor multi-step prediction guarantees a real belief state.

None of these removes the shortcut at inference; they remove it during part of training. That answers your worry about amputating the cache.

### What world foundation models do
On hidden state, essentially nothing:
- **[Cosmos](https://arxiv.org/html/2501.03575v3)** conditions on 1 or 9 frames and makes long videos by chaining 9-frame windows. It has no memory, and says all its models "equally struggle with physics adherence".
- **[V-JEPA 2-AC](https://arxiv.org/html/2506.09985)** trains on 16-frame clips with a 2-step rollout loss. The rollout is scored against a frozen encoder's features of the real frames.
- **[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)** keeps about a minute of visual memory, which emerges from its context. Dreamer 4 attends over 192 frames.
- **Memory add-ons** either retrieve past frames ([WorldMem](https://arxiv.org/html/2504.12369v3)) or use a state-space model (Po et al.). Their memory tests are "come back to a place you saw".

So these models teach us about shortcuts and scale, not about hidden state you can't see in any frame. Having a belief is where our design goes beyond them.

### Your multi-step idea
The structure is right: it's Gregor's multi-step overshoot, which did form belief states. The target is what went wrong.
- A blind rollout scored against the teacher belief can only reward what that belief already holds. It holds no timer, and the exact food digit only 34–40% of the time. That's why the gates price forgetting the phase at 0.19–0.29%.
- Every version that worked scored the rollout against real observations: Gregor through a generator, V-JEPA 2 through a frozen encoder, Po et al. through the diffusion loss.

Ours would be a blind Mamba rollout with the cache also blinded, scored by G1 on the real future percepts; Codex suggested the same. But that is still a squared error, and the percept gate prices the timer at 0.0018 there. I'd expect it to improve retention of large content (layout, mobs out of view) more than the timers.

### Where the belief's teachers sit
This is my clearest account of why Mamba learns some things and not others:

| | Priced by surprise | Priced by squared error |
|---|---|---|
| **Reads the outcome** | reward, continuation, change map | TSP |
| **Must anticipate** | **nothing** | G1, competing with the cache and its noisy view of the answer |

- **What the belief learns:** what the top-left teachers need. It keeps health exactly, because the reward needs it, and tracks sleep through the change map.
- **What it doesn't learn:** what only G1 asks for.
- **Our arms never reached the empty cell.** Since 11-08, every arm has added to an occupied one: the HUD arm added a reactive surprise loss, and the multi-step, dense and endpoint arms added anticipatory squared error.
- **Dreamer 4 does fill it.** Its reward and policy heads predict the next 8 steps from the current state, as a training-only objective. In imagination, the same head reads rewards off the generated frames.
  - That answers your 11-08 worry about consistency: the predictive heads only shape training, so there is no second world.

One caveat from our own data: the reset has to register in Mamba's input in the first place. Health changes do (AP 0.999), but food, drink and energy changes barely do (0.27 / 0.19 / 0.06). That input layer is trained only through the belief's teachers, so it's the same pricing problem one layer down. If pricing is the cause, that number should move before any timer appears.

### The actor
- **What the actor's state must do.** The [information-state framework](https://jmlr.org/papers/v23/20-1165.html) says it must predict reward and observations.
- **Dreamer 4 versus us.** Its actor reads tokens that attend to its whole context, which in our terms is belief plus cache. Ours reads Mamba alone.
- **How SWAX's random window helps.** My reading is that it serves both "Mamba as the brain" and the actor: on short-cache batches, the belief has to carry what the cache normally supplies. It won't fix the timers, which the cache never held, but it makes the belief sufficient on its own without amputating the generator.

### For you to decide
1. **The reframing.** Is the problem how anticipation is priced, and which shortcut it can take, rather than how much supervision Mamba gets? If you agree, the next step is a test that changes only the pricing. That means one anticipatory, surprise-priced teacher on the belief, over all cells, with nothing Craftax-specific, and the input-registration check above.
2. **The actor contract.** Dreamer 4-style, where the actor reads belief plus cache? Or a belief that is sufficient alone, trained with a randomly shortened cache?
3. **Your rollout idea.** Keep it in the grounded form (blind rollout, blinded cache, G1), knowing it mainly helps long-range retention of large content rather than timers?

I lean towards the pricing test plus the random cache. They're what the outside work supports most directly, and neither adds a module. Two cautions:
- **Self-reweighting can't help here.** LongPPL-style reweighting can't discover a dependency the model doesn't use yet; for the timers its weights start near 1.
- **The change target needs fixing first.** Codex's gate showed it over-prices a +3/255 brightness shift through its translation comparison, so a surprise-priced observation teacher needs a target that ignores that kind of nuisance.

---

## Part 2. Follow-up: the change map, the rollout, the actor and the shortcut findings

**Yes, it's the change map in reverse, and Codex already priced that exact candidate on 11-11.** But the change map has a different job, so I'd extend it rather than replace it.

### Why we have the change map
It isn't there to teach Mamba. It's the periphery's anti-collapse anchor, decided 10-27 (LOSSES principle 7):
- **G1 never trains the encoder.** Its frame inputs and targets are stop-gradient, because a diffusion loss that flows into its own encoder collapses it. REPA-E, cited in LOSSES, measured spatial variance falling from 17 to 0.02.
- **So every token type needs a fixed target.** TSP grounds the periphery only where we fixate. The change map is the only fixed, pixel-derived target over the whole screen.
- **That job has to be reactive.** You can only check that the tokens encode a change by showing them both frames.

Is it contributing? As grounding, clearly. Changing only its definition, in the HUD arm, moved the belief's input digits (health 0.92 → 1.00, food 0.42 → 0.61), danger, and coal/iron. As a teacher of the belief, it reads the belief and sends it 3–4× G1's gradient on ordinary frames. But that gradient is for interpreting a change it can already see, and we can't say it caused anything there.

### One head, two modes
I wouldn't replace it, because removing it leaves the unfixated periphery without a fixed target. Instead, give the same head a second mode: on some samples, hide the next frame. It then has to forecast the change map from the current frame and the post-action belief, which already includes the action.
- There's no new module and no new target.
- It's training-only, so imagination never uses it and there's no second world.
- It's the standard masked-input trick, as in masked autoencoders or the conditioning dropout used in diffusion.

Two things need settling first:
- **The target fails a nuisance check.** In Codex's gate, a +3/255 brightness step cost 12× as much as erasing the hunger phase.
  - Almost all of that came from the translation comparison. The evidence suggests the brightness step changes which camera shift the target picks.
  - Our 10-25 design said self-motion would be predicted from the action and checked by alignment; the implementation estimates it from pixels alone. That's my first lead, and it's untested.
- **Should the grounding mode read the belief at all?** Its job is the tokens. With the belief as an input, it can lean on Mamba instead of the tokens, and it rewards Mamba for interpretation. We've never tested it without the belief.

### The rollout
You're right that routing a blind rollout through the denoiser mixes two jobs; I'd drop that version. One correction on "faith": diffusion-forcing models routinely learn to trust context according to its noise level (Po et al., Dreamer 4), so the harm isn't automatic. The real problem is that it buys almost nothing. It's still squared error over the whole percept, where the gate prices the timer at 0.0018.

But I'll challenge the other half. Your multi-step arm's separation was the right design; its target just wasn't what we thought.
- **What the target really was.** Codex's dissection found that giving the teacher identical future inputs removes 99.3% of its separation between the two hunger histories. So the "latent target" was really the future observation seen through the belief's own encoding, and that encoding gets the exact food digit only 34–40% of the time.
- **Why it priced the timer so low.** That lossy lens is why it priced the timer at 0.2–0.3%, below G1.

So keep your separation and change the target: a dedicated head that reads the belief and predicts observable changes, priced by surprise. That's the change map's forecast mode. Your multi-step idea and "the change map in reverse" are the same proposal at different horizons.

I'd start at one step:
- **One step already demands the memory.** Forecasting the next food change requires up to 26 steps of history.
- **Averaging over horizons divides the event.** The gate gave 0.00034 averaged over 16 steps, against 0.0028 at the informative step.
- **Gregor's argument doesn't apply to timers.** His case for multi-step prediction was that the next step is nearly determined by recent frames. That's true of what the cache covers, not of the timers.

If we later want multi-step prediction for slow, global content, I'd do it Dreamer 4's way. Its reward head predicts the next 8 rewards from the current state, one output per distance. That avoids running Mamba on mask tokens, a regime it never sees at inference (in imagination it always gets generated frames). It also avoids the fork bugs that came with the blind rollout.

### The actor
Agreed: belief plus cache. This also resolves the "two masters" mismatch I raised on 11-07. The belief can stay the history-dependent part, because the actor gets the cache too. Dreamer 4 is the precedent: its agent tokens attend to every token in its context. And Mamba already informs the cache, since every cached frame carries its memory tokens.

### The shortcut findings
The takeaway is a principle: **a precise route to the answer takes the supervision, and memory gets what's left. Remove that route during training, never at inference.**

Our system has two such routes:
- **The cache.** It isn't what competes with the timers: in the hunger pairs, the last 8 frames are identical, as Codex says. Cache-shortening (SWAX, Po et al.) is worth keeping for long-range spatial memory, such as remembering where water was off-screen, which is what Po et al. tested. With belief plus cache for the actor, we don't need it now.
- **The answer itself.** The reactive heads read the real next frame, and the denoiser reads a noisy copy of it. This is the route that competes with the timers, and the forecast mode removes it.

There's one more instance inside G1. Our ramp weights near-clean targets 10× more than pure-noise ones, so it favours the regime where the answer is visible. Dreamer 4 chose it for image quality. It's worth revisiting, but it changes rendering, so it's a separate decision.

The robotics result (also training the model to predict its own past) stays in reserve, in case forecasting teaches use but retention still fails.

### Codex
Its corrections hold, and I accept them:
- **One bit per step** is an idealised upper bound, not a measurement.
- **0.03%** was a share of coordinates, not of gradient. Codex's measured figure is better:
  - erasing the phase costs 0.2085 in the food token, but only 0.00177 over the whole percept;
  - that's a 118× dilution, exactly the token's share of the percept (16 of 1,880 numbers).
- **My teacher table** shows which teachers exist, not what caused Mamba to keep health or sleep. I withdraw the causal reading.

Its proposed next step matches mine: a one-step forecast over all cells, qualified on three checks before any training.
1. Erasing older evidence must cost real loss, after the averaging over every cell.
2. The same must hold for a consequence that isn't a timer.
3. A nuisance change, such as the brightness step, must cost less than the consequence.

The acceptance test is native generation with the full cache: more drops when they're due, and no rise when they're not.

---

## Part 3. Kept for later (user, 11-11)

The user agreed with the one-head-two-modes reading of the change map, with dropping the denoiser-routed rollout, and
with belief + cache for the actor ("Mamba should inform the cache, or vice-versa, but they shouldn't replace one
another"). Two things to revisit depending on results:

1. **Multi-step prediction as we generalize.** If we want better multi-step prediction, explore Dreamer 4's method:
   per-distance outputs read from the current state (its reward and policy heads predict the next L = 8 steps from
   h_t, one output layer per distance; imagination still reads rewards from the generated trajectory). This avoids
   running Mamba blind on mask tokens, which never happens at inference.
2. **Shortcut / route findings.** The principle above, and these specific remedies, all applied during training only:
   - SWAX: stochastic short/long local window per batch (128 / 2048, p = .5), long-only for the last 10% of training.
   - Po et al.: random clean prefix (> half the sequence, p = .5) with a long independently-noised suffix, so frames
     far from the prefix have no clean neighbour inside the local window and must use the state-space memory.
   - Gregor et al.: grounded overshoot from the belief through an expressive generator.
   - Past-token prediction (robotics): also predict the past, to teach retention.
   - The ramp weighting inside G1 (w = .9 tau + .1) favours the regime where the noisy target shows the answer.

   **The cache findings may become especially relevant in P2**, where sampled scenes must stay coherent over long
   imagined rollouts: the cache supplies short-range coherence, but anything older than 8 frames (off-screen layout,
   entities that left the view) depends on the belief, and that is exactly the content these papers say a local
   route starves. Po et al.'s spatial-retrieval tests (revisit a place after hundreds of frames) are the relevant
   precedent, and their training-time fix is designed for an architecture like ours.
