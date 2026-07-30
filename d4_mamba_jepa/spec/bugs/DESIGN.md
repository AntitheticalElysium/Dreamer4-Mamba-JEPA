Yes. The correct way to restart is to define **two explicit substitutions into stock Dreamer 4**, then inspect every downstream contract they affect.

I am treating your current `ARCHITECTURE.md` only as a record of the existing implementation, not as the design source. 

# The thesis architecture in one sentence

> **Keep Dreamer 4’s causal latent world-model agent and imagination-training pipeline, replace temporal attention with Mamba, and replace generative shortcut-forcing prediction with action-conditioned JEPA prediction.**

That means:

```text
Pixels
  ↓
Causal JEPA encoder
  ↓
state representation z_t
  ↓ action a_t
Action-conditioned Mamba world model
  ↓
predicted representation ẑ_t+1
  ↓
Dreamer 4 policy, reward, continuation and value heads
  ↓
Dreamer 4 imagination training
```

The important point is that **JEPA is not a layer**. It changes how the representation and future predictor are trained. Mamba, by contrast, is an architectural layer replacement.

Stock Dreamer 4 has three phases:

1. Pretrain the causal tokenizer.
2. Pretrain the action-conditioned dynamics, then add policy and reward heads.
3. Freeze the world model and train policy/value heads in imagination. ([arXiv][1])

Below is every major component and the decision required.

---

# A. Inputs and temporal semantics

## 1. Environment interface

### Stock Dreamer 4

The model receives:

```text
observation x_t
action a_t
reward r_t
task q_t
continuation c_t
```

The paper uses high-resolution pixels and low-level actions, but the architecture itself is environment agnostic. ([arXiv][1])

### Change

None from JEPA or Mamba.

### Required decision

Define the transition convention once:

```text
x_t + a_t → x_t+1, r_t+1, c_t+1
```

Recommended:

```text
z_t      = state before taking a_t
policy   = π(a_t | z_t)
ẑ_t+1   = world(z_t, a_t)
reward   = r̂_t+1
continue = ĉ_t+1
```

Never use ambiguous “previous action” conventions internally. Name tensors `action_t`, `next_reward`, and `next_continue`.

---

## 2. Sequence boundaries

### Stock Dreamer 4

Training uses video sequences, with temporal causality and long contexts.

### Change

Mamba introduces a persistent recurrent state. That state must correspond to a real episode history.

### Required decision

A Mamba state must:

* reset at genuine episode starts;
* persist across arbitrary training chunks;
* never carry across two unrelated episodes;
* be copied when branching imagined action trajectories.

Recommended training sampling:

```text
episode
  ├── burn-in context, reconstruct Mamba state
  └── loss segment, compute gradients
```

Do not initialise Mamba from zero at every random replay window and then claim long-term memory.

---

## 3. Observation preprocessing and patchification

### Stock Dreamer 4

Each frame is divided into image-patch tokens. Learned latent tokens attend to those patches and produce the compressed state representation. ([arXiv][1])

### Change

None required.

### Required decision

Select:

* image resolution;
* patch size;
* number of spatial latent tokens;
* latent channel dimension.

These are capacity decisions, not JEPA or Mamba decisions.

Recommended principle:

> Preserve spatial tokens throughout the entire world model. Do not flatten them into a tiny pooled vector and later expand them back.

The world model should predict:

```text
[S spatial tokens, D channels]
        ↓
[S spatial tokens, D channels]
```

not:

```text
[S × D]
   ↓ pool
[D]
   ↓ expand
[S × D]
```

---

# B. Observation encoder and tokenizer

## 4. Causal tokenizer architecture

### Stock Dreamer 4

Dreamer 4 uses a causal encoder and decoder. Within a timestep, image patches and learned latent tokens interact. Across time, the tokenizer is causal, allowing each representation to use current and previous frames. ([arXiv][1])

### Change

Keep the causal structure.

JEPA does not require making the encoder non-causal. Mamba fits naturally because its recurrence is causal.

### Required decision

Should `z_t` represent:

1. only frame `x_t`, or
2. the video history `x_≤t`?

Recommended for a Dreamer 4-derived system:

```text
z_t = E(x_≤t)
```

This preserves Dreamer 4’s temporal compression and allows the representation to encode inventory, motion and partially observed state.

The target encoder must use the same causal convention.

---

## 5. Spatial layers inside the encoder

### Stock Dreamer 4

Dreamer 4 factorises spatial and temporal processing. Most layers are space-only, with temporal mixing occurring less frequently. ([arXiv][1])

### Change

Keep spatial attention.

### Required decision

Do not replace spatial attention with ordinary one-dimensional Mamba. Spatial patches do not have a naturally privileged scan direction.

Recommended encoder block:

```text
Spatial self-attention
Spatial MLP
Temporal Mamba
Spatial self-attention
Spatial MLP
...
```

Mamba handles history. Attention handles relationships between objects and locations inside the current frame.

---

## 6. Temporal layers inside the encoder

### Stock Dreamer 4

Time-only attention is used periodically rather than in every layer. Dreamer 4 reports that temporal attention every four layers improved speed and quality. ([arXiv][1])

### Change

Replace each time-only attention layer with Mamba-2.

Mamba-2 is a recurrent sequence operator that can run in parallel during training and recurrently during inference. ([arXiv][2])

### Required decision

How should spatial tokens map to Mamba streams?

Recommended:

```text
for spatial slot s:
    run one temporal Mamba stream over z_1:T,s
```

Registers and global tokens receive their own streams.

Within-frame spatial attention allows information to move between streams. This is the closest literal replacement for time-only attention.

Do not flatten time and space into one arbitrary raster sequence.

---

## 7. Latent bottleneck geometry

### Stock Dreamer 4

The tokenizer projects latent tokens into a lower-dimensional bottleneck and applies `tanh`, producing a bounded continuous representation. ([arXiv][1])

### Conflict with JEPA

LeJEPA/SIGReg aims to make embeddings follow an isotropic Gaussian distribution. A `tanh` representation is bounded and therefore cannot actually follow an unbounded Gaussian distribution. ([arXiv][3])

This is one place where the two sources cannot simply be glued together.

### Required decision

Choose one geometry for the actual world state.

Recommended:

```text
latent projection → unbounded z_t
```

No final `tanh`. No per-sample L2 normalisation at the world-state interface.

Use:

* gradient clipping;
* careful initialisation;
* SIGReg or EMA stabilisation;
* optional running statistics for diagnostics.

The EMA and SIGReg models must use the **same latent geometry**. Otherwise the anti-collapse comparison is confounded.

---

## 8. Decoder

### Stock Dreamer 4

The tokenizer decoder reconstructs frames from latent representations. It is required for MAE training and allows imagined representations to be visualised. ([arXiv][1])

### Change

A JEPA world model does not require a decoder for control.

### Required decision

Is your thesis about:

1. latent imagination for control, or
2. a full interactive video generator?

Recommended core design:

* decoder is not part of the control model;
* encoder and dynamics operate entirely in representation space;
* train a decoder after freezing the encoder for visualisation and diagnostics;
* do not let decoder reconstruction gradients define the representation.

This gives you Dreamer-style visualisations without silently turning the JEPA objective back into an autoencoder objective.

---

# C. JEPA representation learning

## 9. What exactly “adding JEPA” means

There are two independent JEPA roles:

### Representation JEPA

Learn the observation encoder:

```text
partial/context video → predict target video representation
```

### Action-conditioned JEPA

Learn the dynamics:

```text
past representations + action → future representation
```

These must be treated as two components, even if both use the same general principle.

V-JEPA 2 follows this staged design: first pretrain a representation encoder, then freeze it and train a separate action-conditioned predictor. ([arXiv][4])

### Recommended thesis

Use both, but test them independently:

```text
Experiment A: D4 encoder + JEPA dynamics
Experiment B: JEPA encoder + D4 dynamics
Experiment C: JEPA encoder + JEPA dynamics
Experiment D: C + Mamba
```

That tells you what JEPA is actually contributing.

---

## 10. Encoder views and prediction task

### Stock Dreamer 4

The tokenizer is trained with masked autoencoding. Image patches are masked with probabilities sampled between 0 and 0.9, and the decoder reconstructs pixels using MSE and LPIPS. ([arXiv][1])

### Change

Replace the encoder’s primary training target with latent prediction.

### Required decision

What does one view predict?

Recommended causal video task:

```text
context view:
    visible patches and frames up to t

target view:
    masked patches at t
    and/or representations at t+1 ... t+k
```

The online encoder must never see future frames used as targets.

A practical objective:

```text
Lrepresentation =
    spatial masked-feature prediction
  + temporal future-feature prediction
  + anti-collapse term
```

This preserves both visual detail and temporal state information.

---

## 11. EMA target version

### Design

Maintain:

```text
online encoder Eθ
target encoder Eξ
```

with:

```text
ξ ← τξ + (1 - τ)θ
```

The online context and predictor receive gradients. Target representations are stop-gradient.

### Required decisions

* initial and final EMA momentum;
* whether momentum is constant or scheduled;
* whether prediction is token-wise L2 or L1;
* whether the target encoder receives the full target frame or a causal sequence.

Recommended:

* same causal architecture for both encoders;
* raw token-wise mean-squared prediction;
* one action-conditioned predictor, not an additional SPR-specific projection stack;
* EMA is the anti-collapse mechanism, not a collection of extra architectural differences.

---

## 12. SIGReg version

### Design

Use the same encoder and same predictive task, but remove:

* target encoder;
* stop-gradient;
* EMA schedule.

Add SIGReg to the encoder representations. LeJEPA combines prediction agreement with SIGReg and specifically presents this as an alternative to teacher-student and stop-gradient heuristics. ([arXiv][3])

### Required decisions

Where is SIGReg applied?

Possible units:

* pooled frame representation;
* flattened complete frame state;
* each spatial token separately;
* all tokens mixed as samples.

Recommended starting point:

```text
one sample = flattened latent state z_t ∈ R^(S×D)
```

Apply SIGReg across independent batch-time samples. Continue measuring covariance and rank per spatial token so local collapse is not hidden.

This is a genuine open design decision because LeJEPA is not specifically designed around Dreamer 4’s dense causal spatial state.

---

## 13. Fair EMA versus SIGReg comparison

The two arms must share:

* encoder architecture;
* Mamba architecture;
* latent dimension;
* predictor architecture;
* predictive loss;
* masking;
* optimiser;
* data;
* rollout loss.

The only difference should be:

```text
EMA:
    stop-gradient target encoder

SIGReg:
    gradients through both views + SIGReg
```

Do not give the EMA branch additional projection heads, normalisation rules and cosine losses while giving SIGReg raw latent MSE. That would not isolate anti-collapse.

---

## 14. Encoder freezing

### Stock Dreamer 4

The dynamics model is trained against representations produced by a frozen tokenizer. ([arXiv][1])

### Change

None required. This actually complements JEPA well.

V-JEPA 2-AC also freezes the pretrained encoder before action-conditioned dynamics training. ([arXiv][4])

### Recommended training stages

```text
Stage 1:
    train JEPA encoder

Stage 2:
    freeze encoder
    train action-conditioned Mamba predictor

Stage 3:
    add policy/reward/continuation heads

Stage 4:
    freeze world model
    train policy/value in imagination
```

Only after that works should you test slow joint encoder finetuning.

This separates representation failure from transition-model failure.

---

# D. Action-conditioned world model

## 15. Action representation

### Stock Dreamer 4

Action components are embedded separately. Continuous values use linear projections, while categorical or binary actions use lookup embeddings. The component embeddings are then combined. ([arXiv][1])

### Change

None required.

### Required decision

How does the action reach every spatial prediction stream?

Recommended:

* encode actions as explicit action tokens;
* include them in within-timestep spatial attention;
* let every spatial latent attend to the action before temporal recurrence or prediction;
* do not reduce the action to a tiny additive bias unless tested.

---

## 16. Exact causal order inside a timestep

This must be explicitly designed.

The causal contract is:

```text
z_t
  ↓
agent state h_t
  ↓
policy chooses a_t
  ↓
world predicts z_t+1, r_t+1, c_t+1
```

Therefore:

```text
h_t must not see a_t
transition must see a_t
```

### Recommended block structure

```text
Current-state block:
    z_t
    registers
    task/agent query h_t

Action transition:
    a_t

Outputs:
    ẑ_t+1
    r̂_t+1
    ĉ_t+1
```

This can use either:

1. two model calls per environment step, or
2. one carefully masked token block.

Start with two explicit stages. It is harder to accidentally leak the action being predicted into the policy.

---

## 17. Spatial dynamics

### Stock Dreamer 4

The dynamics linearly projects each representation into spatial tokens, then processes them using space-only and periodic temporal layers. ([arXiv][1])

### Change

Keep spatial attention.

### Required decision

The action-conditioned world model should predict every future spatial token directly.

Recommended:

```text
input:  z_t ∈ R^(S×D)
output: ẑ_t+1 ∈ R^(S×D)
```

No pooled context bottleneck. No separate small MLP responsible for recreating the entire scene.

The full Mamba-spatial backbone is the JEPA predictor.

---

## 18. Temporal dynamics and Mamba

### Change

Replace Dreamer 4’s time-only attention layers with Mamba-2.

### Required decision

What state is recurrent?

Recommended state:

```text
M_t = {
    Mamba state for each spatial stream,
    Mamba state for action streams,
    Mamba state for register streams
}
```

Transition:

```text
(ẑ_t+1, M_t+1) = Pφ(z_t, a_t, M_t)
```

During training, use parallel Mamba scans.

During imagination, use the exact recurrent step implementation. Test parallel-scan and recurrent-step numerical parity before agent training.

---

## 19. Register tokens

### Stock Dreamer 4

Register tokens are added to dynamics blocks. The paper reports no measurable FVD improvement but qualitatively better temporal consistency. ([arXiv][1])

### Change

Keep them initially.

### Mamba interpretation

Registers become useful global recurrent memory streams:

```text
spatial tokens ↔ registers through spatial attention
registers persist through temporal Mamba
```

This is a principled way for information to move between distant spatial regions and across time.

Do not use separate unexplained “agent memory,” “global memory,” and register systems until registers have been tested.

---

## 20. Shortcut signal and step-size tokens

### Stock Dreamer 4

Dynamics receives:

* corrupted latent;
* signal level `τ`;
* shortcut step size `d`;
* action.

It predicts the clean representation using shortcut forcing. ([arXiv][1])

### JEPA change

Remove:

* `τ`;
* `d`;
* noise injection;
* shortcut embeddings;
* denoising loops;
* `K=4`;
* flow bootstrap loss;
* context corruption used to indicate noisy generations.

These exist specifically for the generative flow objective. They have no role in a deterministic action-conditioned JEPA predictor.

Keeping them as constant tokens would preserve code shape, not architecture.

---

## 21. Transition output

### Stock Dreamer 4

The dynamics predicts a clean latent from a corrupted candidate latent.

### JEPA version

The dynamics directly predicts the next target embedding:

```text
ẑ_t+1 = Pφ(z_≤t, a_≤t)
```

V-JEPA 2-AC similarly uses the action-conditioned predictor itself to output the next frozen-encoder feature map. ([arXiv][4])

### Recommended

Use a linear projection from the final spatial hidden states to the encoder latent dimension.

No separate narrow `CDPPredictor`. The sequence model is the predictor.

---

## 22. Prediction loss

There are two sensible source-faithful choices:

* V-JEPA 2-AC uses token/feature L1.
* LeJEPA uses squared prediction agreement plus SIGReg. ([arXiv][4])

### Required decision

Because you want EMA versus SIGReg with minimal confounding, use the same squared error for both:

```text
L1step = mean ||ẑ_t+1 - target(z_t+1)||²
```

Do not globally normalise the entire latent before computing the loss. That discards magnitude and spatial allocation information.

Per-channel standardisation based on frozen target-encoder statistics is defensible, but must be shared across all arms.

---

## 23. Multi-step rollout loss

### Stock Dreamer 4

Diffusion forcing and context corruption train the model to handle imperfect generated history.

### JEPA consequence

Once flow is removed, you must replace that robustness mechanism explicitly.

V-JEPA 2-AC combines teacher-forced prediction with a two-step autoregressive rollout loss. ([arXiv][4])

### Recommended objective

```text
Lworld =
    Lteacher-forced
  + λ2 L2-step
  + λ4 L4-step
  + λ8 L8-step
```

Use predicted representations as subsequent inputs.

The rollout horizon should be selected in relation to the imagination horizon. It must not be inherited from CartPole or selected arbitrarily.

A reasonable curriculum:

```text
early: one-step
then: 2-step
then: 4-step
then: randomly sample 1, 2, 4, 8, 16
```

---

## 24. Synthetic-prefix distribution

This is separate from rollout length.

During imagination, the model may eventually receive a context containing only predicted states. It therefore needs training examples with:

```text
0 synthetic states
1 synthetic state
2 synthetic states
...
fully synthetic context
```

Recommended:

* unroll from a real prefix;
* progressively replace real states with generated states;
* compute future prediction, reward and continuation losses on those states;
* match the maximum synthetic-prefix length to the actual actor-training regime.

Without this, world training and imagination training use different state distributions.

---

## 25. Stochastic futures

### Stock Dreamer 4

Flow prediction is generative. Multiple plausible latent futures can be sampled.

### JEPA version

A plain predictor is deterministic.

### Required decision

Is a deterministic latent future sufficient?

For a first Craftax model, the clean choice is:

```text
deterministic JEPA world model
```

Mamba history should reduce partial observability by retaining previous evidence.

Do not retain half of shortcut forcing merely to provide noise. That creates an undefined hybrid.

If deterministic prediction later proves inadequate, introduce a separately designed stochastic state variable or multi-hypothesis predictor and treat it as another research axis.

---

# E. Agent integration

## 26. Agent/task tokens

### Stock Dreamer 4

Agent tokens receive task embeddings and attend to all world modalities. World tokens cannot attend back to agent tokens. This prevents the requested task from changing the world’s predicted physics. ([arXiv][1])

### Change

Keep this asymmetric information flow.

### Required decision

Where do agent tokens attach in a Mamba system?

Recommended:

* world Mamba state remains task independent;
* after processing `z_t`, agent queries read the current world hidden state through cross-attention or within-step spatial attention;
* their outputs go to policy, reward and value heads;
* agent tokens never update world Mamba state.

This preserves Dreamer 4’s causal-confusion protection.

---

## 27. Policy head

### Stock Dreamer 4

Policy is learned first through behaviour cloning. It uses multi-token prediction with horizon 8 and an action-appropriate categorical or binary distribution. ([arXiv][1])

### Change

None required.

### Required decisions

* whether to preserve MTP horizon 8;
* whether the action space is categorical, factored categorical or continuous;
* whether policy reads one agent token or several.

Recommended:

* keep Dreamer 4 MTP initially;
* predict `a_t` and future actions from pre-action state `h_t`;
* verify that no current action token is visible to the policy output.

---

## 28. Reward head

### Stock Dreamer 4

Reward is predicted from agent/task embeddings using an MLP and symexp two-hot distribution, also with MTP. ([arXiv][1])

### Change

Keep the distributional reward head.

### Required decision

Timing.

Recommended:

```text
r̂_t+1 = reward(post-action transition state)
```

The reward must depend on both `z_t` and `a_t`, particularly for crafting, attacking and interaction actions.

Task conditioning can change which event counts as reward, but it must not alter `ẑ_t+1`.

---

## 29. Continuation or termination head

### Stock Dreamer 4

The return equation uses `c_t` to represent non-terminal states, but the main architecture description explicitly details policy and reward heads more clearly than the origin of continuation predictions. ([arXiv][1])

### Change

Your implementation must make this explicit.

Recommended:

```text
ĉ_t+1 = BernoulliHead(post-action transition state)
```

Train on true environment continuation.

Do not fold timeout truncation and absorbing death into the same target unless that is the intended RL semantics.

---

## 30. Behaviour-cloning and reward finetuning phase

### Stock Dreamer 4

During agent finetuning, Dreamer 4 continues applying video-prediction loss while adding policy and reward losses, preserving world-model capabilities. ([arXiv][1])

### JEPA version

Continue applying:

```text
JEPA transition loss
rollout loss
policy BC loss
reward loss
continuation loss
```

Do not freeze a weak world model and then train heads in isolation merely because head training is cheaper.

Once the whole world and heads pass validation, freeze them before imagination RL.

---

## 31. Behavioural prior

### Stock Dreamer 4

At imagination-training start, the BC policy is copied and frozen as a behavioural prior. The actor is regularised toward this prior. ([arXiv][1])

### Change

None.

This is especially important for an offline model because the actor can otherwise exploit world-model errors by selecting out-of-distribution actions.

---

## 32. Value head

### Stock Dreamer 4

The value head predicts symexp two-hot value distributions and is trained on TD lambda returns. ([arXiv][1])

### Change

None required.

Value input should be the same pre-action agent state used by the policy:

```text
v_t = V(h_t)
```

The value head must not see future predicted rewards directly.

---

# F. Imagination

## 33. Starting states

### Stock Dreamer 4

Imagined trajectories begin from dataset contexts. Dreamer 4 starts one rollout per context to maximise diversity and reduce memory use. ([arXiv][1])

### Change

Keep this.

### Mamba requirement

The replay context must initialise:

```text
encoder Mamba state
world-model Mamba state
current latent z_t
```

Do not encode the context, discard the recurrent state, and then begin imagination from zero memory.

---

## 34. One imagined transition

The complete JEPA-Mamba step should be:

```text
1. Read current world state:
      h_t = agent_readout(z_t, M_t, task)

2. Sample action:
      a_t ~ π(h_t)

3. Predict next world state:
      ẑ_t+1, M_t+1 = P_Mamba(z_t, a_t, M_t)

4. Predict outcomes:
      r̂_t+1 = R(post_transition_state, task)
      ĉ_t+1 = C(post_transition_state)
      v_t    = V(h_t)

5. Feed ẑ_t+1 back for the next step.
```

There is:

* no noise sample;
* no four-step denoising;
* no shortcut schedule;
* no re-encoding of a decoded frame;
* no full-context rescan.

That is the clean Mamba-JEPA imagination loop.

---

## 35. Recurrent state branching

Actor training normally samples one action per rollout state, so one recurrent state continues forward.

For planners or action comparisons:

```text
M_t
 ├── copy → candidate action 1
 ├── copy → candidate action 2
 └── copy → candidate action 3
```

The candidate branches must not mutate the shared prefix state.

This should be a first-class recurrent-state API, not an optimisation added afterward.

---

## 36. Imagination horizon

### Required decision

The imagination horizon must match what the world model has demonstrated.

Do not select 32 merely because Dreamer implementations commonly use long rollouts.

Recommended progression:

```text
prove 4-step accuracy
prove 8-step accuracy
prove 16-step accuracy
then train actor at 16
```

Increase actor horizon only when real-versus-imagined reward, continuation and representation probes remain calibrated at that horizon.

---

## 37. Policy optimisation

### Stock Dreamer 4

Dreamer 4 uses PMPO based on the sign of the advantage, balancing positive and negative examples, plus a reverse KL to the frozen behavioural prior. ([arXiv][1])

### Change

None.

This part is independent of whether the future state came from flow or JEPA.

Keep:

```text
advantage = lambda_return - value
positive/negative PMPO terms
reverse KL to BC prior
```

---

## 38. World-model freezing during RL

### Stock Dreamer 4

The transformer is normally frozen during imagination RL, with only policy and value heads updated. Full transformer finetuning requires retaining the dynamics, policy-prior and reward losses. ([arXiv][1])

### Change

Keep it frozen initially.

Otherwise the actor can modify the representation and transition model to create easier imagined rewards.

Joint actor/world finetuning should be a later experiment with all preservation losses active.

---

# G. Optimisation and training protocol

## 39. Loss composition

Recommended stage-specific losses:

### Encoder pretraining, EMA

```text
Lencoder = Lmasked-prediction + Lfuture-prediction
```

Target encoder is EMA and stop-gradient.

### Encoder pretraining, SIGReg

```text
Lencoder =
    (1 - λ) Lprediction
  + λ LSIGReg
```

### Dynamics training

```text
Ldynamics =
    Lone-step
  + λrollout Lmulti-step
```

### Agent finetuning

```text
Lagent =
    Ldynamics
  + Lpolicy-BC
  + Lreward
  + Lcontinuation
```

### Imagination RL

```text
Lactor = LPMPO + β KL(actor || BC-prior)
Lvalue = LTD-lambda
```

Each stage has a defined purpose. There should not be one permanent bag of losses.

---

## 40. RMS loss normalisation

### Stock Dreamer 4

Dreamer 4 uses running RMS estimates to balance multiple output losses. ([arXiv][1])

### JEPA conflict

LeJEPA’s `λ` explicitly balances prediction and SIGReg. Separately RMS-normalising both terms changes that objective.

### Recommended

First combine the representation objective:

```text
LLeJEPA = (1 - λ)Lprediction + λLSIGReg
```

Then treat `LLeJEPA` as one loss block when balancing against reward or policy losses.

The same applies to EMA JEPA: normalise the complete JEPA loss, not every internal subterm independently.

---

## 41. Sequence-length curriculum

### Stock Dreamer 4

Dreamer 4 alternates short and long sequences, then finetunes on long batches. It notes that training sequences should exceed the context length to avoid dependence on always seeing a sequence beginning. ([arXiv][1])

### Mamba version

Use:

* many short scans for throughput;
* occasional long scans;
* true episode-aligned recurrent states;
* truncated backpropagation through time;
* random burn-in lengths.

The important metric is not merely “sequence length 16.” It is how far useful information remains recoverable in the recurrent state.

---

## 42. Parameter and compute matching

For the Transformer versus Mamba comparison, match:

* total depth;
* hidden dimension;
* number of spatial layers;
* latent geometry;
* training tokens;
* optimisation steps;
* data order;
* approximate temporal-layer parameter count;
* approximate training and recurrent-inference compute.

Report both parameter count and actual throughput. Mamba’s purpose is not just accuracy but efficient recurrent sequence modelling.

---

# What the final architecture should contain

## Kept from Dreamer 4

* patch-based causal visual encoder;
* spatial latent tokens;
* factorised spatial and temporal processing;
* register tokens;
* explicit action tokens;
* agent/task tokens with one-way information flow;
* policy and reward MTP heads;
* symexp two-hot reward and value distributions;
* BC behavioural prior;
* replay-context imagination starts;
* TD lambda value learning;
* PMPO actor learning;
* frozen world model during imagination training.

## Replaced by Mamba

* causal time-attention layers in the encoder;
* causal time-attention layers in the action-conditioned world model;
* KV-cache-based temporal history;
* repeated context rescanning during rollout.

## Replaced by JEPA

* tokenizer MAE as the primary representation objective;
* shortcut-flow future-state objective;
* noisy latent inputs;
* signal-level embeddings;
* shortcut-step embeddings;
* bootstrap flow loss;
* four-step denoising;
* stochastic latent sampling.

## Newly required because of those replacements

* explicit EMA or SIGReg anti-collapse design;
* compatible unbounded latent geometry;
* action-conditioned direct future-representation predictor;
* multi-step autoregressive rollout loss;
* synthetic-prefix training;
* exact Mamba recurrent-state semantics;
* continuation head;
* deterministic-versus-stochastic scope decision;
* diagnostic decoder rather than training-path decoder.

# The clean implementation order

```text
1. Causal Transformer JEPA encoder
2. Causal Mamba JEPA encoder
3. Frozen encoder + Transformer action-JEPA dynamics
4. Frozen encoder + Mamba action-JEPA dynamics
5. Reward and continuation heads
6. BC policy
7. Real-context versus imagined-context parity
8. Dreamer 4 actor/value training
9. EMA versus SIGReg
10. Optional joint encoder finetuning
```

Most importantly, the architecture should have **one world predictor**:

```text
Action-conditioned spatial-attention + temporal-Mamba network
```

It should not have:

```text
encoder
→ dynamics context transformer
→ pooled agent token
→ small CDP MLP
→ expanded future spatial state
→ second dynamics pass
```

That latter chain is precisely the sort of source-shaped glue that obscures which component is responsible for the world transition.

[1]: https://arxiv.org/pdf/2509.24527 "https://arxiv.org/pdf/2509.24527"
[2]: https://arxiv.org/abs/2405.21060 "https://arxiv.org/abs/2405.21060"
[3]: https://arxiv.org/html/2511.08544v3 "https://arxiv.org/html/2511.08544v3"
[4]: https://arxiv.org/html/2506.09985v1 "https://arxiv.org/html/2506.09985v1"
