# Why the world does not draw consequences: rules, routes, provenance, noise, literature (Stage A)

2026-11-05. User's request: research the consequence failure thoroughly before any intervention, close to the code,
without generalising from papers. The user's three hypotheses: (1) the information is lost at the fusion / mixer /
global stage, (2) Mamba struggles with it, (3) even with the information the world cannot generate the right thing.
The user also asked how heavily Dreamer 4 uses noise and whether our noise could be responsible.

This is reading and arithmetic only: the Craftax-Classic source, our code, the vendored sources (Dreamer 4 PDF and two
reimplementations, JEDI LaTeX, DRAMA), DIAMOND's GitHub config, papers. One CPU script (`noise_profile.py`) re-reads
F2's saved monitor losses; no model was run. Stage B (`PLAN.md`) is the measurement on the model, not yet run.

## 1. What decides each failing consequence (Craftax-Classic source)

`craftax_step` (game_logic.py 1655-1710) runs, in order: crafting, `do_action` (DO), placing, movement, mobs (attacks),
spawning, plants, intrinsics (sleep / wake, food, drink, energy, health regeneration), caps. Reward = number of
achievements newly unlocked + 0.1 x health change. So "first coal" in the actor interface (P(reward > .5)) is the first
coal collection of the episode; later collections change the percept identically but earn no reward.

| Event | Conditions at frame t (state before the step) | What changes at t + 1 | Hidden part | On screen at t |
|---|---|---|---|---|
| collect wood | DO, faced cell TREE, no mob on it, awake | tree -> GRASS; wood +1, HUD (7, 5) | first-unlock flag only | faced block, player sprite (direction), HUD |
| collect stone / coal | DO, faced STONE / COAL, wood pickaxe >= 1 | block -> PATH; HUD (7, 6) / (7, 7) | as above | as above + pickaxe slot (8, 1) |
| collect iron | DO, faced IRON, stone pickaxe >= 1 | block -> PATH; HUD (7, 8) | as above | as above + slot (8, 2) |
| drink | DO, faced WATER | drink +1 (cap 9), HUD (7, 2) | as above | faced water, HUD |
| zombie damage | zombie at L1 distance 1 from the player's POST-move position (zombie's pre-move position), its cooldown <= 0 | health -2 (-7 asleep), HUD (7, 0); cooldown -> 5 | the cooldown (decrements every step: one hit per 6 steps while adjacent) | zombie, action, health; cooldown inferable from the last hit within 5 frames |
| death (risky) | damage >= health | termination | as above | health <= 2 + the above |
| wake up | asleep and energy >= 9 at the start of the step, or hit while asleep | sleep tint removed (whole map) | none | energy HUD (7, 3) (the HUD is drawn without the sleep tint) |
| defeat zombie / eat cow | zombie health 5 / cow 3, damage 1-5 by best sword | mob gone | mob health = hits so far | the hits, if seen |
| place table | PLACE_TABLE, wood >= 2, faced cell not solid | table on faced cell | none | wood count, faced cell |

What follows for the three hypotheses, before any measurement:
- Wood, stone, coal, iron, drink and wake depend only on facts visible in frame t, all decidable from the coarse
  periphery (Codex 11-04: 70,208 / 70,208 HUD values from coarse tokens, zero task-relevant coarse misses on 17,518
  resource frames; 10-29: facing direction 1-NN 1.000 at 3 x 3). No fine information and no memory beyond one frame
  is required. For these, hypotheses 1 and 2 can only matter if the generator gets the facts through the memory
  route and that route drops them.
- Damage and death add one hidden variable, the cooldown, inferable from at most 5 frames back: inside the
  denoiser's 8-frame direct reach, but it needs events integrated across frames.
- Defeat zombie / eat cow need the hit count, which can exceed the direct reach when the fight is interrupted:
  here Mamba's memory genuinely matters.

## 2. Where information can travel in the model (saccade/model.py)

- Memory route: the whole percept (81 x 16 coarse, the 9 fine tokens on an 81 x 64 canvas, the mask, lighting: 6,569
  inputs) through ONE linear map to 256 (`Belief.percept_embed`, model.py 227-232), six Mamba-2 layers interleaved
  with the action token, then 256 -> 4 x 256 memory tokens (full column rank, Codex 11-03) that every denoiser layer
  attends to. In training Mamba always reads clean recorded percepts, so this route is always clean.
- Direct route: the denoiser's temporal attention, at layers 4 and 8 only, each token attending to the SAME token slot
  in the past 4 frames (model.py 336-343); reach 8 frames. A context frame's tokens have already passed three spatial
  layers within their own frame when they are cached, so cross-cell facts can be gathered either there or in layers
  5-7 of the target frame.
- Action: adaLN (one global modulation, every token) and the action token in Mamba. Measured on F2 (M0.5 battery,
  `20261103_p1f2/readings/eval_m05`): recomputing the belief with NOOP at t - 1 while the denoiser's own action input
  stays DO removes the drink / wood consequence entirely (appearances: success 0.00 vs 0.80 / 0.72 with the belief
  recomputed unchanged): the action reaches these consequences through the memory route only.
- Since the factorised denoiser (decision 1, 11-01) physical tokens never attend to glimpse tokens in spatial layers,
  and temporal attention pairs slot with slot, so fine evidence reaches the next periphery ONLY through the memory
  route. Coarse evidence has both routes.

## 3. Why each stage exists, what the source does, where we departed

Choices the user and agents made deliberately are marked as such; departures are facts, not errors.
- Percept -> Mamba: flatten + one linear map (m0.yaml cites DRAMA; STEP_SPEC 7: "position-sensitive and lossy; probes
  show what survives"). Codex's `../20261103_fusion_projection/PROVENANCE.md`: DRAMA projects a reconstruction-trained
  categorical latent into a width-512 Mamba; JEDI conditions its denoiser on spatial latents (4 x 16 x 8 x 8); Po et
  al. 2025 keep spatial tokens with block-wise scans. None of our sources passes a spatial grid through one linear map
  to a 256-D recurrent state: ours is a minimal-build bet.
- Mamba belief: the LeWM-Raw recipe (6 layers, width 256, d_state 64), our past working Mamba world.
- 4 memory tokens: a working value, no source.
- Denoiser: Dreamer 4's space-time transformer, temporal attention every 4 layers (D4: also faster AND better FVD).
  D4 attends over 192 frames of context (Minecraft); ours over 8, with Mamba as the long memory.
- Factorisation (decision 1): to make the world's outcomes independent of the gaze (P1-F's privileged-gaze gate). Side
  effect, not a goal: it closed the denoiser's direct fine route.
- Spatial fusion (F1.6): Codex's 11-02 discussion (no content-dependent coarse / fine interaction before compression).
- Noise, objective and sampler (never audited before today):

| Choice | Ours | Source | Departure |
|---|---|---|---|
| training noise | ln sigma ~ N(-0.4, 1.2), clip [0.002, 20], sigma_data = 1 | DIAMOND: same numbers with sigma_data = 0.5 (agent/default.yaml; trainer.yaml). JEDI: kept DIAMOND's library, changed only "sigma of input data to be 1" (App., hyperparameters) | the absolute draws are identical; sigma_data is a preconditioning constant (Codex). If DIAMOND's 0.5 matches its pixels' spread (EDM's image convention, not measured here) and ours roughly matches our tokens (periphery RMS 0.88), we train at about half DIAMOND's noise relative to the data |
| loss weight | x-space error x (1 + 1 / sigma^2) (EDM / DIAMOND raw-output MSE) | D4: x-space error x (0.9 tau + 0.1), tau uniform (eq. 4, 6, 8) | ours is close to the v-loss weighting (1 / (1 - tau)^2) that D4 replaced (Table 2: x-loss 326 -> 151 FVD, then ramp 151 -> 102) |
| noise on context in training | each frame its own sigma (diffusion forcing, D4) | D4: per-frame uniform tau, context included (reimplementation: every frame); DIAMOND and JEDI: CLEAN context frames (JEDI: randomly its own denoised output) | DIAMOND / JEDI's distribution was built for clean context; we apply it to context too, beside an always-clean memory route D4 does not have |
| context at inference | sigma 0.1 | D4 text: "slightly corrupt ... to signal level tau_ctx = 0.1" (by its own convention tau = 1 is clean, so literally 90% noise); reimplementations: clean (nicklashansen) or signal 0.9 (edwhu) | ours = edwhu's reading; declared in STEP_SPEC 7 as an interpretation |
| sampler | 3 Euler steps, sigma_max 5, rho 7, S_churn 1, sigma_max-scaled start, evaluated at sigma-hat | DIAMOND: same schedule, churn 0, unit-noise start, evaluated at sigma. JEDI: churn 1 for stochastic experiments. D4: 4 shortcut steps at tau = 0, .25, .5, .75 (sigma = inf, 3, 1, 0.33), deterministic, network trained for those steps | ours evaluates at sigma 6.67, 0.38, 0.003 (sigma_data units; DIAMOND 10, 0.57, 0.004): one Euler step covers the whole band 0.4-6.7 |
| loss reduction | mean over the 1,880 scalars of a frame | D4 and DIAMOND: mean over elements | standard; a single cell is 16 / 1,880 = 0.85% of a frame |

## 4. Noise: how heavily Dreamer 4 uses it, and whether ours could be responsible

Dreamer 4 trains on every signal level uniformly, pure noise included (tau = 0 is on the grid of eq. 4; with the
reimplementations' default k_max = 8 it is 1 / 8 of all draws), weights the x-space error by 0.9 tau + 0.1 (more weight
toward clean, but on an x-space error that vanishes as the input becomes clean), and samples in 4 steps evenly spaced
in tau with a network trained (shortcut bootstrap) to take exactly those steps. Its own ablation measures that plain
diffusion forcing cut to 4 steps degrades badly (FVD 306 -> 875) and that the shortcut objective recovers it (329).

Our draws above sigma = 5: 4.7% (D4-style continuous tau: 16.7%; 25% at k_max = 8; D4's bootstrap rows sample coarser
grids, up to 47% pure noise if step sizes are uniform over 1-1/8: Codex). A SURROGATE of the objective (Codex 11-05:
F2's 8 monitor errors under the inference context and one shared sigma, reweighted; extrapolated above sigma 4.2; not
the stochastic-context P1 term, not gradients; the D4 rows are a flow-loss illustration on finest-step rows, not D4's
shortcut mixture). Stage B's B4 measures the actual term. Weighted by each recipe:

| Share of the objective | sigma < 0.3 | 0.3-1 | 1-3 | 3-7 | > 7 |
|---|---|---|---|---|---|
| ours | 56% | 23-24% | 13% | 5% | 2-3% |
| DIAMOND as published (sigma_data 0.5) | 26-28% | 24-25% | 23-25% | 15% | 7-12% |
| Dreamer 4 (uniform tau, ramp) | 6-7% | 22-25% | 34-38% | 18-19% | 12-19% |
| Dreamer 4, k_max = 8 | 3-4% | 17-20% | 30-34% | 17-20% | 16-18% |

F2's x-space error rises from 0.0009 (sigma 0.11) to 0.021 (sigma 4.2): the unpredictable part of a frame is ~2-3% of its
second moment (periphery token RMS 0.88, F1.5 geometry), so at low noise the input itself is the answer and the
residual is pure denoising. Choi et al. 2022 (P2
weighting) call SNR > 1 (here sigma < 1) the "clean-up" stage where "recovering images does not require prior
knowledge", and SNR 10^-2-1 (sigma 1-10) the "content" stage. Ours spends ~80% of its objective in clean-up and ~20%
in content; the Dreamer-4 weighting, on the same error profile, about 30% / 70%.

How this could make the consequences fail (hypotheses; Stage B measures each):
- N1, objective allocation. A consequence must be inferred from the history only in draws where the noise hides it,
  i.e. above some sigma* set by its token-space size. Those draws are a small share of ours, the cell is 0.85% of the
  frame and the event is rare: the drive to learn "DO + facing coal + pickaxe -> coal" from the conditioning is tiny
  (10-29, P1 world 1: single-cell consequences 0.31% of G1, drive gone by checkpoint 10). At low noise the network
  can reproduce the consequence by reading it from the input, which teaches nothing about predicting it. Not the
  whole story as stated: on F2 drink APPEARANCES (token step ~6, M0.5 separation) are generated 67-83% and increments
  (steps ~2-4) 8-25%, although a larger step stays visible to higher noise. A miss costs ~step^2, so the quantity is
  the consequence's share of the objective where it is hidden (Stage B, B0 / B4), not the draw count alone.
- The first sampling call (sigma-hat 6.67, 2.8% of draws, ~2% of the objective) is where the prediction must come from
  the conditioning alone. 11-01 (P1-F / beta, joint denoiser): there the field barely separated must-wake from
  stay-asleep from pure noise (beta .53 vs .49) but did when the real successor leaked through the same noise level
  (.83 vs -.02): the pattern of Lin et al. 2024 (models trained without zero terminal SNR learn to use the leaked
  signal and fail from pure noise). Not yet measured on a factorised world or on coal, iron, damage.
- N2, sampler. With rho = 7, the first update puts 95.8% of the state on the first estimate (x1 = 0.042 x-hat +
  0.958 D from sigma-hat 6.67 to 0.28): an interpolation coefficient, not the share of the outcome decided there
  (Codex); the next call at 0.38 can still change it, for instance by snapping a blended estimate to the nearer branch: 10-29 (exact two-outcome field, real token geometry) 3-step Euler alone turned a 15% branch into
  5%; F1.5 (Codex 11-02): 8 steps without churn raised risky death .010 -> .076 (recorded .145), coal .07 -> .21,
  iron .02 -> .17. Um et al. 2024: Tweedie denoising favours majority features.
- N3, route competition. In training the previous frame is cleaner than sigma 0.2 only 16% of the time (D4: 17%
  cleaner than sigma 0.2, so context noise itself is not unusual), while the memory tokens are always clean. The
  network may get frame-t content through the 256-D memory rather than the noisy direct route. Measured already for
  the ACTION on F2 (memory only, section 2); not for the context content.

What Dreamer 4's evidence is not: FVD on Minecraft video, not consequence fidelity; a cascade (each row of Table 2 on
top of the previous ones, so the x-loss gain is measured with shortcut and x-prediction present); its latents' scale is
not reported (a smaller latent scale would make its training even noisier relative to the data). D4 deliberately
down-weights high noise relative to its own x-loss; "train harder at high noise" is not a free conclusion from it.
Shortcut forcing (user, 11-05): later, not now.

## 5. Literature by link

Each entry: setting; claim; what would have to hold in our model; what in our model would show it.
- Choi et al. 2022, Perception Prioritized Training (CVPR; arXiv 2204.00227). Unconditional DDPM images. Content is
  learned at medium SNR, imperceptible detail at high SNR; down-weighting clean-up improves samples on every dataset
  tested. Ours: the world-model prediction would be the "content". Shown by: the field's consequence present at low
  sigma only (Stage B, profile).
- Lin et al. 2024, Common Diffusion Noise Schedules and Sample Steps are Flawed (WACV; 2305.08891). Stable Diffusion:
  non-zero terminal SNR leaks low-frequency signal; the model learns to rely on it and cannot produce very bright /
  dark images from pure noise. Ours: training at sigma <= 20 with little mass near sigma-hat, sampling from pure
  noise. Shown by: field from pure noise << field from the real successor + the same noise, at the first call.
- Everaert et al. 2024, Exploiting the Signal-Leak Bias (WACV; 2309.15842): the same leak characterised at inference.
- Kingma and Gao 2023, Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation (NeurIPS;
  2303.00848): every common objective is a weighted integral of ELBOs over noise levels; the weighting is a free
  choice. Frames section 4: the weighting decides what the network learns, not whether it is a valid diffusion model.
- Um, Lee and Ye 2024, Minority Guidance (ICLR; 2301.12334): samplers favour majority features (Tweedie). Ours:
  minority consequences (damage 15%, death 12%). Shown by: production below the field's own first estimate.
- Dreamer 4 (2509.24527) Table 2 / Figure 8 / eq. 8: section 4. DIAMOND (2405.12399) section 2.3, Appendix L: a single
  denoising prediction is a conditional mean; 3 vs 1 steps in Boxing (already read 10-29).
- GameNGen (2408.14837): noise augmentation of context frames is needed against autoregressive drift; zero noise at
  inference works. Supports noisy context in training; says nothing about a clean side route.
- Jelassi et al. 2024, Repeat After Me (ICML; 2402.01032): a fixed-size state cannot copy more bits than it holds;
  Mamba loses to transformers at copying / retrieval, is better at tracking state. Zoology (Arora et al. 2023): the same
  for associative recall. Applicability: our Mamba receives one 256-D vector per step and holds 6 x 4 x 64 x 64 state
  values; the facts that decide coal or damage are a few bits from the LAST frame. These papers bear on long memory
  (revisits, interrupted fights), not on the failing primaries. Our own d4mj history: a Mamba fed per-slot inputs
  showed real long memory (E17-E20).
- Po et al. 2025, Long-Context State-Space Video World Models (ICCV; 2505.20171): spatial tokens kept, a separate SSM
  scan per spatial block (more state capacity), local frame attention, and a long-context training scheme without
  which the model ignores distant context. Supports questioning the one-vector interface for MEMORY; not tested on
  one-step consequences.
- Dedieu et al. 2025, Improving Transformer World Models for Data-Efficient RL (ICML; 2502.01591): Craftax-Classic,
  our frames; per-patch nearest-neighbour tokens (fixed codes) and block teacher forcing beat DreamerV3 (69.7% vs
  53.2% reward). A per-patch, no-global-bottleneck world model works on our exact environment. Our denoiser is
  already per-cell; only the Mamba side compresses.
- DIAMOND's thesis ("Visual Details Matter in Atari"): compact discrete latents can drop RL-relevant detail. Ours
  compresses only the memory route; the generator itself works on per-cell tokens.
- Wu et al. 2022, the greedy nature of multi-modal learning (ICML; 2202.05306) and Huang et al. 2022 (Codex 11-04):
  networks lean on the input that is learned fastest and under-fit the other. Ours: always-clean memory vs noisy
  direct context. Shown by: the field's consequence surviving heavy context noise (Stage B, route).

## 6. What this changes

- The measured failing consequences are coarse, one frame back and deterministic (or one hidden cooldown): the fine
  path is not on their critical path. Fine information remains an open question for fine targets (HUD digits in the
  glimpse, night creatures), measured by Codex.
- The noise recipe is the least examined part of the design and departs from every source at once: DIAMOND's
  distribution at JEDI's preconditioning, applied to context (D4's practice) beside a clean memory route (ours), with
  an EDM weighting D4 moved away from and a 3-step schedule D4 measured as bad without shortcut training. Each
  departure had a reason, and neither finite-noise EDM nor departing from D4 is a defect by itself (DIAMOND and JEDI
  work with them: Codex); their combination was never checked against consequences.
- Codex's review of this note and of Stage B: `../20261105_consequence_trace_review/REVIEW.md`; responses in PLAN.md.
- Stage B measures, on F2, where each primary consequence stops: in what the generator receives (H1 / H2), in the
  field's estimate at the decision level (N1, H3), in the sampler (N2), and through which route (N3).
