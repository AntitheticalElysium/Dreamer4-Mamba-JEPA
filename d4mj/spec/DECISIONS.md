# Decisions and function plan

Companion to `ARCHITECTURE.md`. That file says what the system is; this one says
what is settled, what is open, and exactly which functions exist.

The contract: **no new files, and no function that is not listed here.** Private
stateless tensor helpers are permitted and are listed like everything else — the
defence is planning every signature in advance, not forcing arithmetic to be
duplicated inside oversized functions. Anything absent below is a design change,
not an implementation detail. If the plan is wrong, fix the plan.

## Settled

| # | Decision | Basis |
|---|---|---|
| S1 | Patch size 7, no resize, 81 tokens — one token per rendered tile | Craftax-Classic obs is 63×63 with `BLOCK_PIXEL_SIZE_AGENT = 7`; 63 = 9×7. Divisors of 63 are {1,3,7,9,21,63}, of which only {7,21} are tile-aligned and non-degenerate. 7 is the finest. `[R0-CHOICE]` — no genericity claim is made |
| S2 | `Z*` = D4 `tanh` bottleneck, **unnormalized**; direct readout is `tanh`-bounded | Keeps the flow anchor paper-faithful and Stage A genuinely fixed-representation. Matches the readout's codomain to its target's, which is the principle behind V-JEPA's `normalize_reps` for unbounded features. Bounds range only — **does not** address the scale contraction toward the conditional mean, which S35 shows is a proven consequence of determinism rather than an open question |
| S3 | MTP `L = 8`, leads `n = 0..7`; policy lead 0 = outgoing `a_t`, reward lead 0 = incoming `r_t` | §3.3 states `L = 8`. Module defaults agree in both reproductions, but edwhu's eval script runs `L: int = 2` and `nicklashansen/dreamer4` has no MTP heads at all, so the paper is the basis and the reproductions only fail to contradict it. Eq. (9)'s inclusive `Σ_{n=0}^{L}` reads as nine terms and is recorded as ambiguous |
| S4 | Reward and value: 255 bins, symlog support ±20 | D4 §3.3 "Following Dreamer 3"; D3 `rewhead`/`value` `bins: 255` with centres `symexp(linspace(-20,0))` mirrored. Two-hot is linear interpolation between neighbouring centres, so bin count sets grid density, not a quantization floor; the wide support is miscalibration headroom, not wasted resolution |
| S5 | `time_every = 4` | D4 §3.4. Reproductions ship 2 and 1 — not inherited |
| S6 | λ-returns in next-index form, `G_t = r_{t+1} + γc_{t+1}[(1−λ)v_{t+1} + λG_{t+1}]` | Eq. (10) prints same-index reward, continuation and value, which is not self-consistent with §3.3's shared-index annotation. `dreamerv3/agent.py:487` indexes `rew[:, 1:]` and `boot[:, 1:]`, expanding to exactly this form |
| S7 | λ = 0.95 | `[DESIGN]`. D4 is silent; DreamerV3 `lam: 0.95` is implementation precedent only |
| S8 | Temporal state is a per-layer, per-slot pair; the conditioning slot is retained in every arm | `ARCHITECTURE.md` Boxes 3 and 4 |
| S9 | One `World.forward` evaluates any block; `World.predict` turns its features into a latent | The occupying latent and its conditioning are per-call arguments, not state. Flow runs N candidate rungs then one commit; direct runs a predictor head then one commit (S34) |
| S10 | The signal table has exactly `k_max` rows; every context index is clamped to `k_max−1` | Eq. (4)'s grid tops out at `1−d`, so τ = 1 is never trained. MMBench2 sizes `signal_embed` at `k_max+1` and labels uncorrupted context with index `k_max` — a row its own sampler cannot reach; `plan_cem.py:189,545` pass `tau_ctx=0.0`, so its planner runs entirely on that untrained row, and `round(0.9·k_max) = k_max` also fires at `k_max = 4`. Same pathology as the predecessor's unreachable shortcut rows, in the upstream source |
| S16 | τ_ctx = **0.1 noise / 0.9 signal**, snapped to the nearest trained bin | `train_dynamics.py:576` states it verbatim: "tau_ctx is the noise fraction: 0 = fully clean, 1 = fully noisy". The paper's "signal level τ_ctx = 0.1" is a slip — 0.1 *signal* contradicts its own word "slightly". No longer open |
| ~~S17~~ | **Superseded by S34.** The in-block query is untrainable in one pass, which S17 did not see: a position cannot hold the real latent for later blocks to attend to *and* the query to be predicted from. Direct now predicts from committed world features. The firewall objection S17 raised against that is answered by reading spatial and register outputs, which the dynamics mask makes agent-free | |
| S18 | Invariant 5 is `[DESIGN]`, not `[D4-ENTAILED]` | No audited source persists dynamics state across accepted frames — MMBench2 re-draws context noise per frame (`z0_ctx = torch.randn_like(...)` inside the per-frame call) and re-prefills. Persistence is our deviation and the `[MAMBA]` efficiency hypothesis rests on it |
| S11 | Context corruption is applied once at commit, never at read | Fresh noise per read changes the prefix every frame, invalidating any KV cache or SSM state and contradicting invariant 5 |
| S12 | Agent slots exist from Phase 1B, last in the layout, masked both ways until Phase 2 | Fixes `S`, mask shape, stream count and state shapes once, so no Phase-1B gate is invalidated when the agent modality activates |
| S13 | `depth % time_every == 0` and at least two time layers | At `depth = 4` the entire Mamba blast radius would be one module and parameter matching would be dominated by it |
| S14 | Anchor drops GQA and attention-logit soft capping, keeps RoPE and QKNorm | Table 2 shows GQA at FVD 70 → 71, adopted for KV bandwidth at 2B parameters — no benefit at our scale. RoPE and QKNorm have executable precedent in pinned MMBench2 attention; GQA and capping have none anywhere. Anchor is described as paper-constrained *modulo declared omissions* |
| S19 | Flow is **5 backbone passes/frame**, direct is **1** plus a predictor head | Flow's commit is a real extra pass: S11 puts corruption at commit, and the final rung's input is the running Euler iterate, not a fresh corruption of the accepted latent. Direct needs no candidate pass at all under S34. 5-vs-1 is an *evaluation-count* ratio; `diagnostics.cost` measures whether it is a throughput ratio |
| S20 | Causal temporal tokenizer, as D4 §3.1 | The primary model follows Dreamer 4. Both arms share one encoder, so the T-vs-M comparison stays fair even though the encoder carries history; a frame-only encoder is a later ablation, not a fork |
| S21 | `k_max ≥ 8`, declared in `Config` alongside `K = 4` | `round(0.9·k_max)` hits the untrained top row exactly at `k_max = 4`, and S10's clamp then drops τ_ctx to 0.75. Both failures vanish at `k_max ≥ 8` (τ_ctx = 0.875). k_max sets the training noise grid; K is the generation rung count — they are independent and neither was registered |
| S22 | The reward and continuation caused by `a_t` are read at lead 0 of `h_{t+1}`, never `h_t` | S3 defines reward lead 0 as the reward *arriving*. `a_t` is chosen at `h_t`; its consequence arrives with `o_{t+1}`. Reading lead 0 at `h_t` returns the previous action's reward and shifts every return by one step — the predecessor's `reward_logits[:, 0, 0]` is correct only under this reading. Realised in `imagination.imagine`, which reads both from the readout `advance` returns; **no gate asserts it** -- doing so needs the gate to run a rollout, which it does not |
| S23 | `Z*` is defined at MAE probability 0 | Masking is a Phase-1A training mechanism. The predecessor trained with masking silently disabled while advertising 0.9; the mirror failure is emitting cached targets under a random mask, which makes the same frame yield different `Z*` |
| ~~S24~~ | **Superseded by S34.** With no candidate pass there is nothing for a `{candidate, commit}` table to distinguish, so direct's conditioning slot carries a single reachable embedding | |
| S25 | Action table has `n_actions + 1` rows, the extra being BOS. No query token exists (S34) | Reachable at every true episode start; the predecessor shipped 18 rows undocumented. The query-shape half is void under S34, which has no query |
| S26 | Committed and observed blocks carry the finest step index `d_min` | The signal bin is fixed by S16; nothing fixed `d`. MMBench2 labels context blocks with `d_min`; training samples `d` per block, so a committed block needs a declared value rather than an inherited one |
| S38 | `long_only_fraction = 0.25` and `rms_decay = 0.99` are `[DESIGN]` | D4 specifies a final long-only finetune and running-RMS normalisation but neither number. Registered so they are not read as paper values |
| S39 | Separate-image training is **not implemented**, and the claim is withdrawn | D4 uses it to generate start frames without context; every rollout here begins from a committed dataset context, so the mechanism it trains is never exercised. The earlier `_isolate_rows` cleared target flags without changing the temporal context at all -- measured, the loss was bit-identical with and without it, which is worse than an honest omission |
| S40 | MSE on masked patches, LPIPS on the whole predicted frame | Scoring visible patches under MSE rewards copying, which masked autoencoding exists to avoid. That leaves `p = 0` images carrying perceptual signal alone, which is why LPIPS must not be composited. Eq. 5 masks neither term; both halves are declared |
| S41 | A quarter of flow training rows carry the *rollout* prefix: every block but the last at the commit condition | Independent per-block draws make that joint prefix vanishingly rare -- measured, a 48-block prefix uniformly at the commit condition has probability 0.000000, while at rollout it is every prefix. Per block the condition is in distribution; the prefix never was. `commit_prefix_fraction` is `[DESIGN]`: it must be nonzero and small enough not to displace diffusion forcing |
| S42 | The attention temperature is clamped, as the pinned source clamps it | S14 dropped the paper's logit soft capping alone; without the clamp too, nothing bounds the logits at all, and a runaway temperature saturates attention to one-hot with no diagnostic. The two decisions belong together |
| S43 | The 50/50 mixture belongs to **Phases 2 and 3**. Phases 1A and 1B sample the whole corpus and score every row. In Phases 2 and 3, half the rows are drawn uniformly and half are drawn so the window *contains* a task event -- §4.1 names "behavioral cloning, reward modeling, and reinforcement learning", so imagination RL starts from mixture contexts too, not the pretraining distribution; BC scores the relevant half, the continued dynamics loss the uniform half, reward and continuation everything. `relevant` is a **sampling role**, `None` in pretraining. Nothing falls back to the other pool | Applying this to pretraining was a real error, now fixed: D4 pretrains the world model on the full corpus (§3.3, Algorithm 1) and only then finetunes with the mixture. Restricting pretraining to uniform rows discards half the corpus. §4.1, verbatim: "we use data mixture of 50% uniform sequences and 50% relevant sequences that accomplish one of the tasks. The behavioral cloning loss is applied only on the relevant fraction, while the dynamics loss is applied only on the uniform sequences **to avoid optimistic generations**." A dynamics model fitted only on successful play learns that things tend to go well, and imagination inherits it. Reward/continuation are not restricted because the paper restricts the other two by name and says the mixture amplifies signal *for* reward modelling; halving that head's data would be an addition, not a reading. Relevance is per step (`Episode.events`), not per episode: a random window from a successful 2500-step episode mostly shows walking. Verified: pretraining dynamics responds to every row; finetuning dynamics is bitwise unchanged when relevant rows are perturbed, and BC is the exact mirror |
| S44 | Blocks are gradient-checkpointed, and `batch` is **4** | The 6 GB probe, run per-configuration in fresh processes because in-process trials do not release memory and gave non-monotonic garbage. Phase 1A binds. Without checkpointing the declared architecture fits **only batch 1 at the short length** (2.69 GiB) and the long batch never fits -- the previous `batch: 8` was unreachable by 8x, so every phase-1A number to date would have come from a config that cannot run. With checkpointing: 3.00 GiB at batch 8 short, 3.07 GiB at batch 4 long. Batch 4 is the largest that runs *both* lengths, which D4's alternating short/long schedule requires. Checkpointing is exact -- measured bitwise-identical loss and gradients on attention, 1e-10 relative on Mamba from kernel nondeterminism -- so *it* moves cost, not results. **The batch change does not**: halving the batch halves the examples per update and raises gradient variance, so it is an optimization change and the effective batch/update budget must be registered with any result produced under it. No Z*-defining field changed. Confirmed end to end: all four arms, all four phases, 3.20 GiB peak |
| S45 | Both time mixers start from the same weights wherever they share a parameter | `manual_seed` alone does not achieve this. Construction draws in order and the two mixers consume different numbers of values, so every parameter built after the first time layer diverges. Measured: 36 of 99 shared tensors differed between the arms -- everything from block 4 onward -- so the Mamba-vs-attention comparison was also a different-random-init comparison, on the one axis the project exists to measure. Shared weights are now copied from the attention arm by name and shape; 99/99 identical |
| S47 | Where Phase-1A memory goes, measured, and why checkpointing was the right lever rather than a workaround | Profiled per submodule rather than assumed. Nothing is unvectorized and attention is not the problem: `F.scaled_dot_product_attention` already selects a backend costing 36.4 MiB for our shapes, *better* than forced mem-efficient (52.7) or forced math (77.5), and flash is unavailable only because the model runs fp32. The cost is the ordinary transformer intermediates against a 5.1 MiB activation: SwiGLU's gated 4x expansion saves 40.6 + 20.3 + 20.3 + 5.1 = **86.3 MiB per layer, matching the measured 86.3 exactly**, and space attention 52.1. Eight layers of that is the 1.4 GiB encoder forward. So there is no hidden waste to remove -- recomputation is the only lever that does not change the architecture |
| S48 | `Batch.scored` is per block and per row; **S31's single burn-in integer is withdrawn** | A block is faithful once it holds a full receptive field, and unconditionally in a window starting at the episode start, where nothing earlier is missing. One integer masked the first `receptive_field - 1` blocks out of *every* row -- so the states deployment actually begins from were never supervised, while still costing full activation memory. Measured: 65% of encoder memory was spent on blocks producing no loss term; the scored fraction rises from 34.8% to 51.1%. A quarter of uniform rows are drawn at the episode start, because at a uniform start those windows arrive with probability 1/span (~0.5%) and the fix would be inert |
| S49 | The archived replay is loaded, not regenerated, and `train_expert` is withdrawn in favour of `expert.load_archive` | There is no expert to retrain: the archive's manifest carries `params_sha256` and `replay_sha256`, and the corpus supplies both §4.1 sampling roles (S46), so nothing is "missing" in the mixture sense. Broader collection is a *support-coverage* decision -- terminal exposure (S50) and behavioural diversity -- not a requirement the paper imposes. Conversion is exact and lazy -- crop and permute are views, so 320 episodes cost 0.86 GiB RSS against an 8.6 GiB file, and the 696,746 transitions match the manifest. `events` is reconstructed from the per-frame cumulative achievements the archive already stores. The previous pipeline was right where this one had regressed: its `_dead` reads `in_lava | (player_health <= 0)` directly, exactly the fix applied to `env.step` this session |
| S50 | The archive's 2500 cap is **ours, not the environment's**, and its cost is terminal scarcity rather than truncated windows | Craftax's native horizon is 10000 and no archived episode reaches it, so all 252 of 320 cap endings are our own truncation and only 68 episodes died. Against this architecture the cap is harmless for window sampling -- the mean episode is 2177 steps, 34x `sequence_long` and 45x `dynamics_context`, and only 3 episodes are too short for a long window. What it does cost is terminations: 68 terminal transitions in 696,746 is 0.0098%. The per-step figure must be computed under the *actual* sampler -- episode-uniform, then start-uniform -- not by multiplying the global frequency by the block count, which assumes block-uniform sampling and is optimistic. Measured over the real archive: **0.00209 terminal blocks per short batch (one every 479 steps)** and 0.00563 per long batch (one every 178). An earlier figure of "one every 160" used the block-uniform shortcut and is corrected here. Task events are not scarce by comparison (0.95%, 0.61 per step). Terminal exposure is therefore a collection problem for the uniform half, not an argument about the cap, and raising the cap would make it worse by lengthening the surviving episodes |
| S51 | **Active task**: one aggregate Craftax-Classic task. Primary performance is the official 22-achievement geometric-mean score; "any first achievement" is the relevance event | No task tokens exist in this architecture (a declared D4 omission), so there is nothing to condition a per-task policy on. Under one aggregate task the relevance criterion already implemented -- the step at which the achievement count rises -- is coherent rather than arbitrary. If the intended goal were specifically diamond acquisition, the task and reward interface would be insufficient and would have to change *first*; this decision forecloses that reading deliberately |
| S52 | **Evaluation protocol**, frozen before any cell is inspected | Native **10000**-step horizon, not the collector's 2500 cap (S50): scoring at a quarter of the horizon measures the cap. Categorical sampling at temperature 1 is primary, greedy secondary. Primary metric is the official geometric-mean achievement score; secondary are raw return, achievement count, per-achievement rates, termination rate and length -- all reported, none promoted afterwards. Controls are the actor's own frozen BC prior and a random policy. Every policy runs the *same* preregistered seeds, with environment, policy and flow-corruption streams separately derived and recorded, so the arms do not differ by their own randomness. Intervals are 95% percentile from a **paired** bootstrap over seeds that recomputes the nonlinear score from resampled rows; raw episode rows are retained. An arm passes only when the lower bound of its advantage over **both** controls exceeds zero. DEV and FINAL seed sets are disjoint; FINAL is opened once, after all selection |
| S53 | **Parameter matching**: at most **0.5%** deployed-parameter residual, shared dimensions held fixed | `d_state` is the single matching knob (S28). The measured residual at `d_state = 64` is -0.316%, which passes, so 64 is settled by a declared rule rather than by being the value that happened to be there. Tolerance and rule are fixed before any result is read; a later arm that misses it must move `d_state`, not the tolerance |
| S54 | **Imagination horizon is not settled at 8.** `horizon` is a smoke default; the final value is selected on DEV from `horizon_candidates` | Blessing a default because it ran is how an arbitrary constant becomes a result. Selection uses `diagnostics.multistep_error` under a *full* committed context, which is why that diagnostic was corrected from its one-block start -- choosing a horizon from a model with almost no history selects for the wrong regime. Selection happens on DEV, never against executed-control FINAL numbers |
| S55 | **Generated-prefix contract**, closing the open register row | One-step teacher forcing plus a two-step autoregressive rollout through the real `advance` path; the two rollout terms **averaged**, following the pinned V-JEPA 2-AC source, which computes `jloss + sloss` with each a mean. Squared error rather than the source's L1 (`loss_exp: 1.0`), because S35's conditional-mean analysis is stated for squared loss -- a declared deviation. **Incoming recurrent memory is detached in both arms.** Official Mamba-2 mutates its `InferenceParams` cache in place, so its step is not differentiable in the state it receives while attention's is: measured, an attention prefix took gradient 147.92 through carried memory and a Mamba prefix took `None`. Truncating both is a *matching* choice, and it is registered here rather than only explained at the call site because it changes the objective both arms optimise |
| S56 | Uniform sequences are drawn uniformly over eligible **(episode, start)** pairs, and the two sampling strata are explicit | Picking an episode and then a start weights every episode equally regardless of how many windows it contains: measured on this archive, a single window from the shortest eligible episode was hundreds of times likelier than one from a 2500-step episode. That is not "uniform sequences", and it would silently overweight every state of a short failure rollout the moment support data is added. On top sit two declared strata: `episode_start_fraction` of uniform rows begin at the episode start (the only windows whose earliest blocks are scorable, S48), and one row every `support_every` steps is a terminal tail. The `row % 4` trick that previously forced episode starts was measured at 1.5% under the Phase-2 mixture rather than the intended 25%, because no uniform row index satisfied it at batch 4 |
| S57 | `Episode` carries **three independent facts**: `events`, `uniform_eligible`, `bc_eligible` | Relevance-for-BC was previously inferred from `events.any()`, so any rollout that happened to unlock one achievement became eligible for behaviour cloning. Deliberately degraded support data would therefore have entered BC, which is the one thing it must never do. The archive is `bc_eligible=True`; the support corpus is `uniform_eligible=True, bc_eligible=False` while keeping its true events and rewards |
| S58 | A **support corpus** is collected before the reported run: archived PPO with epsilon-greedy noise at 0.1, 0.25, 0.5 and 1.0, every rollout kept, until ~320 genuine terminal episodes | The archive gives the continuation head 68 terminals in 696,746 transitions, so a terminal-containing batch arrives about once in 479 short updates and a constant "continue" head would fit it. Pure random noise is too narrow a failure distribution: the epsilon ladder produces deaths near competent play as well as early ones. Survivors are kept because filtering for death would make the corpus a death distribution rather than a behaviour distribution. Restraint is deliberate -- the predecessor measured that 50% terminal-window sampling inflated terminal targets 955x and took 61.5% of BCE mass -- so oversampling is one row every eight updates, routed to the continuation head alone and to no other loss |
| S59 | Continuation is diagnosed **split by target**, not by its global mean | With terminals at ~0.01% of transitions, a head that always predicts "continue" matches the global mean and the global target rate almost exactly, which is what the previous diagnostic reported. `continuation_separation` -- the gap between the probability on continuing and on terminal states -- is the quantity that collapses, and `terminal_targets` says how many terminals the estimate rests on |
| S60 | The critic gets its own trunk, and Phase 3 sizes its own batch | Policy and value shared `actor_body`, so a value-only backward put gradient 17654 into the body the policy reads: the critic reshaped policy features outside PMPO and outside the prior KL meant to bound how far the actor may move. Measured `None` after the split. D4 calls it "an additional value head" and does not ask for a shared trunk. Phase 3 also inherited Phase 1A's batch of 4 -- a memory ceiling from a phase that runs the tokenizer, imposed on one that never does -- while PMPO's sign-of-advantage estimate is over starting *contexts*; `actor_batch` is 16. Measured together on flow-attention, actor-minus-BC moved from **-1.59 to +0.37** |
| ~~S61~~ | **Withdrawn.** The claim that flow's reward models carry no information rested on a zero baseline measured on the wrong windows: it used *uncached* DEV episodes (burn-in 30, short/long mix of its own) while the model MAE came from the *cached* DEV batches, and it ignored `reward_rows`. On identical cached windows with the correct mask both flow arms beat zero. The baseline is now computed inside `head_calibration` on the same rows it scores, so the two can no longer disagree. What survives is narrower and belongs to S64 | Original text: Phase 3 gated on the reward model beating a zero predictor | Measured on DEV, the zero-predictor MAE is 0.0795. Both flow arms score worse -- 0.098 and 0.086 -- so PMPO was optimising a reward signal carrying no information, while direct scored 0.043-0.049. D4 assumes the learned reward model is useful because Phase 3 optimises it directly; that assumption has to be checked rather than inherited |
| S62 | The S54 horizon rule as first written degenerates and must be restated before it decides anything | "Largest candidate within `horizon_tolerance` of the one-step error, else the smallest" selected 32 for flow-attention and 4 for the other three -- but in those three *no* candidate met the tolerance, so 4 was a fallback rather than a selection. A tolerance relative to the one-step error is tighter in absolute terms for a more accurate arm, which is backwards, and it cannot distinguish "4 is good" from "nothing is good". The mechanism (DEV, declared candidates, full-context roll) stands; the criterion does not |
| S63 | The imagination horizon is the largest declared candidate at which the rollout still beats the **marginal predictor** -- the constant mean latent | S62 rejected the previous criterion; this replaces it. The line is not a tuned threshold: past it the rollout carries less about the future than knowing nothing, so imagining further cannot inform the actor. It is scale-free, which a tolerance relative to the one-step error is not -- that rule was *tighter* for a more accurate arm and degenerated to "no candidate qualified" on three arms of four. It doubles as a collapse test, since a predictor collapsed to the conditional mean scores at the marginal by construction. Reported alongside is Lemma 1 of *On Training in Imagination* (2605.06732), whose return-gap bound holds only while `gamma * L_f (1 + L_pi) < 1` with dynamics-error coefficient `1 / ((1 - gamma)(1 - gamma L_f (1 + L_pi)))`; the measured per-step error growth stands in for `L_f (1 + L_pi)`, so `gamma * growth >= 1` says the bound is vacuous at that horizon |
| S64 | Head output scales follow the pinned DreamerV3 config: `reward` and `value` at 0.0, `policy` at 0.01, `continuation` at 1.0 | `configs.yaml` sets `rewhead: outscale 0.0`, `value: outscale 0.0`, `policy: outscale 0.01`, `conhead: outscale 1.0`. We shipped PyTorch defaults on all four. A value head starting at random emits random advantages on Phase 3's first updates and PMPO reads only their *sign*, so the actor's first moves are noise -- a plausible contributor to the measured post-RL regression. The consequence is worth stating: a zero-initialised reward head is uniform, and a uniform distribution has cross-entropy `log(bins)` against every target, so it begins with no preference at all |
| S65 | Behaviour cloning currently reaches only a fraction of expert behaviour, and that is the largest known defect in the pipeline | Measured on the 76 BC-eligible training episodes: event-centred relevant windows can expose **10.9%** of expert transitions at the short length and **21.4%** at the long one, or **15.5%** under the real 56.3/43.7 short-to-long mix -- so **84.5% of expert behaviour can never become a BC target**. This follows from S51's relevance criterion ("any first achievement") meeting Craftax's one-shot, early-clustering achievements, not from a coding error: the uniform half still shows that behaviour to the *dynamics* loss. It is registered here because it bounds what BC can learn regardless of training length, and because widening the criterion is a change to S51 rather than a fix |
| S46 | The archived Craftax replay is **losslessly convertible and usable for both §4.1 sampling roles**; what it lacks is behavioural breadth, not a "uniform half" | Everything below verified by reading the artifact, not from its manifest alone. `artifacts/expert/craftax_expert_v1.pt` (8.6 GB) holds 320 episodes / 696,746 transitions at `mean_achievements` 20.62 of 22, with all 320 flagged deep-achievement. Conversion is exact: `obs` is `(2501, 3, 64, 64)` uint8 channels-first, zero-padded from 63x63 (row 63 and col 63 measured all-zero), so `[:, :, :63, :63]` then permute to HWC loses nothing. It stores only `continues`, so `terminated = continues == 0` and `truncated` is the 2500 cap. **Correction, 2026-08-01**: the claim that this makes the corpus "100% relevant and 0% uniform" was wrong and is withdrawn. S43 defines `relevant` as a *sampling role*, not a property of an episode, and §4.1's language is sequence-level: a 2500-step successful episode contains many ordinary windows holding no achievement event, so this corpus supplies both roles. The generator also never acceptance-filtered by achievement -- it records every completed rollout and counts achievements afterwards -- so it is already unfiltered *within one PPO policy's behaviour distribution*. The fallback that sentence described no longer exists in the code either. What remains true is narrower and is the actual limitation: one strong policy over 320 episodes is far less diverse than 2541 hours of contractor play, its failure support is tiny (S50), and its expert lacks vendored source lineage. Two further defects compound it: only 68 of 320 episodes terminate (252 hit the cap), so the continuation head sees almost no real terminations, and its expert has no byte-level provenance (`Craftax_Baselines@7ce36fa` is not pinned in `third_party/`). Per S29 it therefore stays a smoke-test corpus. What is missing is not more expert play -- 697k expert transitions is already ample for the relevant half -- but an equal mass of unfiltered rollouts, which also supplies the terminations |
| S27 | Bounded encoder context `W`, part of `C*` — `z_t = Z*(x_{t−W+1..t})` everywhere | Phase 1A windows carry a `W−1` burn-in that is encoded but not scored; once frozen, each episode is scanned once and cached **under the same `W` limit** — an unbounded full-episode scan would produce a different `Z*` from deployment. `W` is not a capacity number: it defines the representation, so changing it changes every `z_t` and it must be frozen before the final encoder trains |
| S28 | Match total **deployed** parameters within a declared tolerance while holding `d_model`, depth, token layout and shared interfaces fixed | Primary knob is Mamba's `d_state`, the only one that moves M-arm parameters without touching the shared backbone. Parameter counts move discretely, so `d_state` alone may not reach tolerance at a sane state size — any additional knob must be declared before training. Report the unmatched residual, FLOPs, memory, recurrent-state size and measured throughput regardless. The predecessor called arms matched in a comment while the temporal module differed by 29.6% |
| S29 | Archived replay is for debugging and smoke tests only; regenerated replay backs every reported number | Its expert has no byte-level provenance and its terminal-window support is 58 windows. Keeping both uses named stops the old set from quietly becoming the final dataset, and keeps `expert.py` exercised |
| S30 | The frozen latent cache lives on `Episode` as `latents` + `latent_digest`; `train_representation` writes it at the Phase-1A boundary and `load_episodes` verifies it | The digest covers exactly `C*`: encoder checkpoint, exported copy, `W`, bottleneck and packing, `p_mae = 0`, patch size. Without it a cache built under a different encoder or `W` is silently reusable, which is the one place a wrong number contaminates every downstream result at once |
| S31 | `Batch` names its regions explicitly: `burn_in` (int prefix), `valid` (per-position target validity). The MAE mask is **not** a loader output — the encoder generates it | "masks" was ambiguous across five different things. Burn-in is always a prefix, so an integer beats a mask. Burn-in frames update encoder memory and score no loss; scored frames follow immediately under that memory. `patches` and `latents` are phase-determined: Phase 1A carries pixels, Phase 1B onward carries cached latents |
| S32 | `Encoder.forward(patches, memory, p_mask, rng) -> (z, memory, patch_mask)` | One signature serves all four uses: Phase-1A window training, frozen episode scanning, recurrent execution, and diagnostics. The caller supplies and receives the bounded-`W` memory, so batched scanning and frame-by-frame execution are identical by construction; `p_mask = 0` on every `Z*` path per S23; burn-in is the caller slicing `z[:, burn_in:]`, which keeps it out of the encoder |
| S33 | `scan_step_parity` also covers the encoder; `alignment` also asserts the `W` horizon and `p_mae = 0` | Batched `W`-context scan ≡ frame-by-frame recurrence ≡ the cached `Z*`; frames older than `W` cannot change `z_t`; burn-in + scored window ≡ episode caching; reset clears encoder memory. Folded into existing gates — no new function |
| S15 | Windows never cross an episode boundary; `evaluate` takes no reset mask | Keeps a reset a fresh construction. Asserted by `gates.alignment`; relaxing it changes the signature |

## Design decisions taken on evidence

Three questions that shape the experiment rather than the code. Each was settled
against `third_party/`, not by preference.

### S34 — Direct predicts from an external head over committed world features

**Rejected:** an in-block query trained in one teacher-forced pass. Position `t`
cannot simultaneously hold the real latent, so later positions can attend to it,
*and* hold the query, so it can be predicted from. Filling every position with a
query resolves that by deleting the task: measured, the prediction is bitwise
identical for two different latent sequences, so the model is asked to predict
`z_t` from the action history alone.

**Also rejected:** a two-pass teacher-forced loss (commit pass to build memory,
query pass against it). Correct, but doubles Direct's training cost for a
mechanism no source uses.

**Taken:** `ẑ_{t+1} = P(world features of committed block t, a_t)`, one pass, the
predictor reading spatial and register outputs. This is what V-JEPA 2-AC does --
its predictor is a separate module over the interleaved `(a_k, s_k, z_k)`
sequence -- and what DINO-WM does. Box 4 rejected it over the §3.3 firewall, but
that objection only bites if the predictor pools the *agent* slot; spatial and
register outputs are agent-free by induction over depth, which `gates.firewall`
already asserts.

Dreamer 4 can share its backbone because a corrupted latent at `τ > 0` still
carries signal, so one block is both context and query. A query token carries
none, so the same trick does not transfer.

### S35 — Deterministic Direct is `K = 1`; the stochastic extension is best-of-K

Craftax is stochastic enough for this to matter. Measured on the installed
environment, 64 draws per `(s, a)` over 72 pairs: **83% branch**, mean 4.2
distinct successors, and top-mode mass as low as 0.19. The distribution is
heavy-tailed -- most states are near-deterministic, a minority are strongly
multimodal, and those are the mob-spawn and combat states that decide episodes.

MoP-JEPA (`2607.05238`, Prop. 1) proves the consequence: under squared loss the
optimal single predictor is `E[z'|c] = Σ w_m μ_m`, error lower-bounded by the
between-mode variance, and for separated modes the optimum lies far from *every*
mode. Under cosine loss with normalised targets the same holds. Prop. 2 shows a
gated weighted-sum mixture does not escape it -- it still emits one vector.

The fix is Prop. 3: best-of-K regression, `L = E[min_k ‖g_k(c) − z'‖²]`, which is
the per-context K-means distortion, so every optimum assigns a head per mode.
Plus a router trained on the winning index and a load-balance term.

**Why this settles the roadmap rather than the code:** at `K = 1` the MoP loss
*reduces exactly to the dense loss*. So deterministic Direct is not a separate
design -- it is the `K = 1` case of the extension. Stage A runs `K = 1`; if it
collapses, `K > 1` is a strict generalisation of the same head, not a new arm.

**Consequence for the plan:** `diagnostics.multistep_error` is a mean error, and
the mean error is *minimised* by the collapsed predictor, so it cannot detect the
failure it exists to detect. It needs MoP-JEPA's measurement alongside: distance
from the prediction to the nearest true successor mode, against distance to their
mean, at high-branching contexts.

### S36 — Dynamics context `C = 3·T_short`, with `T_long = 4·T_short`

Neither cache is currently bounded, and `Config` had no dynamics-context field at
all. Appendix A gives all three of D4's configurations: Minecraft `C=192,
T1=64, T2=256`; SOAR and Epic Kitchens `C=96, T1=32, T2=128`. Every one satisfies
`C = 3·T_short` and `T_long = 4·T_short` -- an invariant across every
configuration the paper reports, not a single data point.

Taking `T_short = 16` gives `C = 48`, `T_long = 64`. This also satisfies §3.4's
requirement that the long batch exceed the context.

## Open, with the functions each one constrains

Nothing here blocks writing the listed signatures. Each blocks a *body* or a
config value, and each must be closed before the phase named.

| Question | Close before | Constrains |
|---|---|---|
| **SIGReg on a projector vs on `z`** — projector keeps `Z*` and gives clean attribution; on `z` forks `Z*` from the anchor and tests the stronger claim | Stage B | `representation.Projector`, `representation.representation_loss` |
| **EMA views / masks / loss; faithful LeJEPA vs anti-collapse ablation** — is the EMA arm learning spatial invariance, temporal predictability, or both? | Stage B | `data.views`, `representation.representation_loss`, `representation.update_target` |
| ~~**Generated-prefix contract**~~ — **closed by S55.** Settled: one-step teacher forcing plus a two-step autoregressive rollout through the real `advance` path, the two rollout terms averaged, squared error, and incoming recurrent memory detached in both arms | ~~Before Stage-A direct training~~ | `transition._direct_loss` |
| **Go/no-go threshold *numbers*** — formulas exist now; scales come from the anchor or a pilot; numbers freeze **before any experimental cell is inspected**. Choosing them after all four cells train is not preregistration, whatever the intent | After the anchor, before inspecting Direct/Mamba cells | `Config` |
| ~~**Capacity**~~ — **closed by S44**. The probe was run on the real worst case (Phase 1A: encoder + decoder + LPIPS + gradients + optimizer state, both sequence lengths). It did not move a single architecture field; it changed `batch` and turned on checkpointing. The remaining untested worst case is the EMA copy, which Stage B introduces | ~~Phase 1A~~ | `Config` |
| ~~**Imagination horizon**~~ — **closed by S54.** The selection *rule* is now fixed: DEV only, from `horizon_candidates`, via `multistep_error` under a full committed context. The resulting number is not yet chosen, and choosing it is a run, not a decision | ~~Phase 3~~ | `Config.horizon` |
| ~~**Matching tolerance and final Mamba dimensions**~~ — **closed by S53.** 0.5% deployed residual, shared dimensions fixed; `d_state = 64` measured at -0.316% and passes | ~~Before building the Stage-A models~~ | `diagnostics.cost`, `Config` |
| ~~**Executed-control metric definition**~~ — **closed by S51 and S52**, and implemented in `execution`: aggregate task, official geometric-mean score, native horizon, sampled primary, BC and random controls, paired seeds, paired bootstrap, and a two-sided pass rule | ~~Before Stage A~~ | `execution.run_episode`, `evaluate`, `score` |
| **Behavioural support of the corpus** — the only data question left open, and deliberately. The archive serves both §4.1 roles (S46), so this is not about a "uniform half": it is whether one PPO policy over 320 episodes, with 68 terminals in 696,746 transitions (S50), is broad enough to attribute a world-model result to. Open: whether to collect more, from what policy mixture, how much, and whether reported runs treat the archive as a hash-pinned artifact whose behavioural lineage cannot be reproduced | Before reported Stage-A training | `expert.collect` |

## Function plan

Modules in architecture order. `Type` means a dataclass or `nn.Module`;
everything else is a function. Every entry maps to a box in `ARCHITECTURE.md`.

### Contracts

**`config.py`** — `Config` (Type, frozen). Every constant, plus
`transition ∈ {flow, direct}` and `time_mixer ∈ {attention, mamba}`. The four
Stage-A arms are four `Config` values; there is no arm factory.

**`sources.py`** — `source_digests(config)`, `verify_sources(recorded, config)`.

**`state.py`** — `WorldState` (Type: latent, memory, step), `RealState` (Type:
encoder_memory, world). No functions: a reset is a fresh construction, and
non-mutation is a property of the time mixer rather than a caller duty.

### Environment and data — Box 1

**`env.py`** — `reset(key)`, `step(state, action)`. `step` returns `terminated`
and `truncated` as separate raw fields; continuation is derived in `agent.py`.

**`expert.py`** — `load_archive(path, config, limit)`, `collect(policy, count, config, limit)`.

**`data.py`** — `Episode` (Type, unshifted storage: `observations`,
`actions_taken`, `rewards`, `terminated`, `truncated`, `relevant`), `Batch` (Type,
block arrays: `patches`, `led_to_action`, targets, masks, `relevant`),
`patchify(frames, patch)`, `mixture_weight(rows)`, `episode_splits(n, seed)`,
`sample_batch(episodes, rng, config)`, `save_episodes(path, episodes)`,
`load_episodes(path)`, `views(batch, rng, config)`.

`relevant` is the S43 mixture label and `mixture_weight` selects one half of it.
It is public and lives here, not privately in each loss, because behaviour cloning
and the dynamics loss select complementary halves and neither owns the rule.

`Episode` and `Batch` are the two index conventions, deliberately named apart.

### Representation — Box 2

**`representation.py`** — `Encoder` (Type), `Decoder` (Type), `Projector` (Type),
`pack(z, n_spatial, k)`, `reconstruction_loss(pred, target, mask)`,
`representation_loss(online, target, projector, views, config)`,
`update_target(online, target, momentum)`.

`Encoder.forward` is all of `C* ∘ E*`: patch projection, MAE replacement, latent
tokens, backbone, bottleneck, `tanh`. `Decoder` is diagnostic-only after freeze.

### Backbone and time mixer — Box 3

**`backbone.py`** — `Layout` (Type: segment order, slices, modality ids),
`space_mask(layout, mode)`, `rope(x, positions)`, `Attention` (Type),
`SwiGLU` (Type), `Block` (Type), `Backbone` (Type).

`mode ∈ {encoder, decoder, dynamics}` is the only place the three masks differ.
`Block` is pre-RMSNorm → space → optional time → MLP, matching the source layer
exactly. `Backbone.forward(x, memory) -> (x, memory)`.

**`time_mixer.py`** — `TimeAttention` (Type), `TimeMamba` (Type),
`time_mixer(config)`.

Both types share `forward(x, memory) -> (y, memory)` and never mutate `memory`
in place. This module is the entire Mamba blast radius.

### Transition — Box 4

**`transition.py`** — `World` (Type), `flow_conditioning(rng, shape, config)`,
`observe(world, encoder, state, led_to_action, patches, rng, config)`,
`advance(world, state, led_to_action, rng, config)`,
`transition_loss(world, batch, rng, config)`.

`World.forward(state, led_to_action, latent, conditioning) -> (latent_out,
agent_out, state_out)` is the generic `evaluate`. `advance` is *N* read-only
candidate evaluations plus exactly one commit evaluation — flow N=4, direct N=1
per S17 — and always reads `h` from the commit pass. `observe` is the one place a real frame becomes `(e_t, z_t, m_t, h_t)`. It takes
`rng` because the flow arm corrupts the committed latent at τ_ctx while the direct
arm commits it clean; it exists because training,
imagination context construction and execution would otherwise each inline
encode-then-commit, which is the accretion this contract forbids.
`transition_loss` covers shortcut-flow and direct feature prediction, including
the rollout schedule, selected by `config.transition`.

### Agent — Box 5

**`agent.py`** — `Heads` (Type: policy, reward, continuation, value),
`twohot(values, centers)`, `head_targets(batch, config)`,
`head_loss(predictions, targets, config)`.

`value` exists from construction but enters no optimizer before Phase 3.
`head_targets` is where MTP lead alignment and the terminal/truncation split are
realised.

### Imagination and improvement — Boxes 6, 7

**`imagination.py`** — `Trajectory` (Type),
`imagine(world, heads, state, agent, rng, policy_rng, config)`.

The caller — `train_actor` — builds the starting state by repeated `observe`;
`imagine` receives it complete and owns no encoder. Box 6's "encode and scan the
context" describes the caller's work, and that boundary is where the
predecessor's context re-slice bug lived.

`Trajectory` carries **soft** `continuation` — the head's probability, not a
boolean — because imagination has no ground-truth termination. DreamerV3 passes
`self.con(inp, 2).prob(1)` and sets `term = 1 - con`, so the probability enters
the discount directly (`agent.py:206,404`). It carries no boundary mask: that
repo sets `last = jnp.zeros_like(con)` for imagination, and the horizon end is
the recursion's initial condition `R_T = v_T`, not a mask. Hard `terminated` and
`truncated` stay in `Batch`, where they are the continuation head's targets.

**`actor_critic.py`** — `lambda_returns(trajectory, config)`,
`actor_loss(logits, actions, returns, values, prior_logits, config)`,
`critic_loss(logits, returns, centers)`.

`actor_loss` is PMPO plus the reverse KL to the frozen prior.

### Execution — Box 8

**`execution.py`** — `Result` (Type: seed, steps, reward, terminated, truncated, achievements), `score(results)`, `evaluate(policies, seeds, config)`, `run_random(seed, config, limit)`,
`run_episode(world, encoder, heads, seed, config)`.

### Orchestration

**`train.py`** — `optimizer(modules, config)`, `train_representation(config)`,
`train_dynamics(config)`, `train_agent(config)`, `train_actor(config)`.

One driver per phase, in phase order. `optimizer` is the only place parameter
groups are built, and the only place upstream `_no_weight_decay` is honoured.

**`checkpoint.py`** — `save(path, config, **state)`, `load(path, config)`.

**`__main__.py`** — `main()`. **`__init__.py`** — exports only.

### Validation — the Stage-A gate list

**`gates.py`** — `alignment(config)`, `scan_step_parity(config)`,
`reset_parity(config)`, `firewall(config)`, `branch_nonmutation(config)`,
`recurrent_carry(config)`.

`scan_step_parity` covers four things: scan versus recurrent step, teacher-forced
forward versus the equivalent reconstructed `evaluate`, encoder scan versus
recurrence, and the windowed branch -- its sequence runs past `dynamics_context`
on purpose, since at a shorter length that branch never executes.
`recurrent_carry` additionally asserts every conditioning row receives gradient
(S10's pathology) and is the only place the flow arm's *history* dependence is
tested, `_observation_dependence` being satisfiable there by the current block
alone. Nothing else exercises `advance`, so without the second half the
most bug-prone function in the system ships untested until Stage-A results are
already contaminated.

`alignment` carries the Box-1 fixtures: length invariants, no-action-leak,
future-observation leakage (with MAE masking disabled or seeded), reward shift,
window-start action identity, the separate-image fraction, that the transition
loss is finite, and that the prediction moves when only its context does. It does
*not* cover S22's imagination read index, which would need a rollout.

**`diagnostics.py`** — `multistep_error(world, batch, config)`,
`latent_stats(world, batch, config)`, `head_calibration(heads, agent, batch, config)`,
`cost(modules, config)`.

`latent_stats` reports range *and* scale, so S2's residual — contraction toward
the conditional mean — is measured rather than assumed away. `cost` reports
deployed parameters, training-only parameters, FLOPs, memory, throughput, and
`e_t`/`m_t` state sizes separately, per invariant 5, which is why it takes the
whole module set rather than the world alone.

## Totals

19 types, 57 public functions and 38 private helpers, across 20 modules.

The private count has grown from the planned 18. That is drift the contract exists
to catch, and it is recorded rather than rounded away: the growth is real, most of
it in `gates.py` and `train.py`, and it should be pushed back down before Stage B
adds more. This sweep moved it in the right direction for the first time -- two
`_device` copies deleted and a duplicated mixture fallback consolidated into one
`data.mixture_weight` -- but consolidating the rest is outstanding work.

## Signatures corrected during implementation

Each was forced by a contract already in this document, and each is a defect the
plan would otherwise have shipped.

| Signature | Why |
|---|---|
| `observe` takes and returns `RealState`, not `WorldState` | It has to carry the encoder's bounded-window memory, which `WorldState` does not hold. Merging them is how a rollout starts from zero memory while looking correct |
| `imagine` takes the starting `agent` readout | Recomputing it would ingest the same latent twice, against the rule that `m_t` already covers block `t` |
| `Trajectory` carries `agent` | The frozen prior and the critic must be evaluated where the actions were chosen, not at the first state alone |
| `diagnostics.cost` takes the module set and the world | It cannot separate deployed from training-only parameters given the world alone |
| `actor_loss(trajectory, returns, prior_logits, config)` | It needs the logits, actions and values together and they already travel as one `Trajectory`; passing them apart invites a mismatched slice |
| `World.forward` returns memory, not a `WorldState` | `WorldState.latent` is defined as the *accepted* latent, which only a commit site can supply; a candidate has no accepted latent to put there |

`representation_loss` raises `NotImplementedError` naming its open decision rather
than guessing a default. `expert.train_expert` did too; S49 withdrew it.

## S37 — `window` bounds state, not receptive field

Measured: with two encoder time layers at `window = 4`, perturbing frames more
than 4 back still moves `z` by 1.8e-3, and beyond 8 back moves it by **exactly
zero**. Influence travels one window per time layer, so the reach is their
product while each layer's cache stays bounded to `window`.

The state bound is the load-bearing half -- it is what makes the cache and the
deployed rollout produce the same latent for the same frame. `Config.receptive_field`
names the other half so no claim confuses them.

Also settled by execution: the tokenizer always mixes time with attention, in
every arm. Mamba's state summarises all history rather than a window, so a Mamba
encoder cannot honour the bound that makes `Z*` well defined -- and keeping the
encoder common is what confines the substitution to the dynamics.

## Verified defects, in fix order

Two independent audits; every entry below reproduced by execution before being
accepted. Nothing here reopens the architecture -- these are the implementation
failing to be what this document already says.

**Before any training run:**

| # | Defect | Evidence |
|---|---|---|
| 1 | Direct's transition loss is observation-free | Prediction bitwise identical for two different latent sequences. Fixed by S34 |
| 2 | Flow is diffusion forcing, not shortcut forcing | Eq. (7)'s bootstrap branch and its two half-step evaluations are absent, so the step token is inert and the arm is the paper's own ablation: Table 2 puts it at FVD 875 against 329 |
| 3 | `tanh` at inference but not in training | `_direct_candidate` squashes, `transition_loss` does not. A unit that learned 0.900 emits 0.716, decaying to 0.431 over six recursive steps toward a fixed point of 0 |
| 4 | Encoder window not enforced | `‖z(t=19 \| 20 frames) − z(t=19 \| 4)‖ = 1.7e-2`. The declared bound is documentation only |
| 5 | Phase 1A never writes the latent cache | `train_dynamics` fails on `batch.latents is None` |
| 6 | Multi-block decode against a cache is non-causal | New blocks attend bidirectionally; batched-vs-sequential differs by 9.2e-2. Attention corrupts silently, Mamba crashes -- divergent failure across the compared axis |
| 7 | Committed content and its label disagree | Mixes at 0.9 signal, labels bin 7/8 = 0.875 |
| 8 | No gate reaches `transition_loss` or `advance` | All six gates pass on the arm defeated by defect 1 |

**All fixed.** Plus, from the same pass: LPIPS at weight 0.2 with `p ~ U(0, 0.9)`
sampled per image; running-RMS loss balancing; the short/long batch curriculum;
seeded construction and device-matched generators; learned agent tokens;
diagnostics that commit before predicting.

Two things the fixes added rather than repaired. `transition.initial` commits a
known latent and returns the state it produces -- gates, diagnostics and Phase 3
all needed it, and inlining it three times is how the uncommitted-start defect
appeared in the first place. `data.unpatchify` is required by the perceptual
term. Both are in the inventory.

Still deferred: `representation_loss` (Stage B), which raises rather than guessing.
`expert.train_expert` is withdrawn -- see S49; there is no expert to retrain.

**Sweep of 2026-08-01.** Four more, each reproduced before being fixed:

1. **The configured batch could not run.** Phase 1A at `batch: 8` OOMs on the 6 GB
   card at every length; only batch 1 short fits. Every capacity claim to date came
   from a config that cannot execute. Fixed by S44 (checkpointing, `batch: 4`), not
   by shrinking the architecture. The first probe was itself wrong -- run in one
   process it reported batch 2 fitting where batch 1 OOMed, because memory is not
   released between trials; per-configuration subprocesses gave monotonic numbers.
2. **The two arms trained from different initialisations** -- 36 of 99 shared
   tensors, everything after the first time layer. Fixed by S45, measured 99/99.
3. **The dynamics loss saw only expert play**, because no mixture existed. Fixed by
   S43, verified bitwise: dynamics ignores relevant rows, BC ignores uniform rows.
4. **`_device` was still duplicated** across `train.py` and `gates.py` -- the exact
   duplication that caused an earlier defect when one copy was fixed and the other
   missed. Both were by then identical passthroughs, so both are deleted and call
   sites read `config.device`; the rationale moved onto the field itself.

Not a defect, and recorded so it is not re-raised: no gate runs in a deployment
dtype other than FP32 **because no other dtype exists** -- there is no autocast,
bf16 or fp16 path anywhere in the system. A dtype gate would be gating a path that
does not exist. It becomes real only if mixed precision is ever introduced.
