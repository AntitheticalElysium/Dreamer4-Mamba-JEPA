# Where does the generated distribution go wrong? Diagnosis plan (2026-10-29, nothing run yet)

## What is established
- One-step generated successors from real histories carry almost no deaths and a third of the negative rewards, while
  the same heads on recorded successors are calibrated (Codex 10-28; reproduced 10-29 with the production streaming
  sampler, fresh draws: death 0.367% recorded vs 0.0001% generated; 12.77% vs 0.015% in low-health / adjacent-zombie
  contexts; `../20261029_consequence_reproduction/RESULTS.md`). In those contexts generated futures show negative reward
  in 4.1% but death in 0.015%: partial consequences without the rule-implied completion.
- Generated tokens violate the encoder's output geometry (per-token zero channel mean after inverting the tanh): a
  summary-statistics classifier separates them at 99.5% (Codex). So the S1 two-sample score (0.76) and reader counts on
  generated codes mix geometry with semantics.
- Reader counts depend strongly on the solver (production 3 steps: zombies 0.24 vs real 0.44; 8 deterministic steps
  0.41; 32 steps 0.53), and readers' support on generated entity codes is poor (Codex). Counts are not established.

Not established: whether the bulk of the generated distribution is right (only rare outcomes missing) or broadly
wrong; at which link the rare outcomes are lost.

## The chain and the competing explanations
history -> (1) information in the conditioning (predicted belief + 8 context percepts + action)
        -> (2) the learned conditional p(next percept | conditioning)
        -> (3) the solver that samples it (3 churned Euler steps from sigma_max 5)
        -> (4) generated token geometry -> (5) heads / readers that read it -> measured consequence
A deficit can enter at any link. Our history points two ways: every deterministic world we built copied consequences
because rare events were ~0.3% of a uniform per-element loss (E14, E17-E20, health diagnosis 10-08) -> link (2); and
diffusion needs an iterative solver to pick a mode, its single step being the average (DIAMOND 5.2) -> link (3).
Sources for (2): diffusion denoisers "play favourites" with majority features (Um et al., ICLR 2024); conditional
diffusion on long-tailed data loses tail classes (Qin et al., CVPR 2023). For (3): EDM assumes p(x; sigma_max) is
"practically indistinguishable from pure Gaussian noise" for sigma_max >> sigma_data (section 2; EDM uses 80 / 0.5,
DIAMOND 5 / 0.5, ours 5 / 1, trained only up to sigma 20); a percept's 1,880 scalars give an aggregate SNR of ~75 at
sigma 5, and Codex measured the start far off the data-mean direction (0.15 vs 30.0).

## Tests (frozen world; each substitutes one link; claim / competing / control / what each outcome rules out)

T0. Instruments valid on generated geometry (links 4-5), FIRST, because every later readout depends on it.
- Claim: the heads (and a HUD-health reader) read model-output geometry correctly. Competing: they misread it, so
  deficits are partly an interface artefact. Control: "denoised real" successors D(x_real + sigma eps) at small sigma
  (model-output geometry, known content), and the idempotent projection P(y) = 3 tanh(LN(3 atanh(y / 3)) / 3), which
  leaves every encoder output unchanged and puts any sample on the encoder's output set.
- Death / damage must be recognised on denoised-real terminal successors at the recorded rate; the projected generated
  samples must give the same head outputs as unprojected ones (else geometry, not content, drives the heads).
- Health readout without a learned reader: the HUD health cell (screen cell 63; the HUD is drawn after night shading)
  as nearest real-glyph cluster with a support radius calibrated on real AND denoised-real tokens; plus the token's
  change from the previous frame (bimodal on real data: unchanged / changed).
- Outcomes: valid -> T1-T5 can use heads and the cell reader; invalid -> fix the instrument before reading anything.

T1. Solver substitution (link 3 vs link 2).
- Claim: the learned conditional holds the dangerous branches; the production solver loses them. Competing: the learned
  conditional itself lacks them. Control: the SAME model and histories under a solver ladder: production; deterministic
  Heun at 8 / 16 / 32 / 64 steps; stochastic (churn) at 32 / 64; sigma_max 5 / 10 / 20 (no extrapolation past the trained
  20). Read death, negative reward and the T0 health readout per setting; the converged plateau is the learned model's
  own rate.
- Outcomes: plateau near recorded -> the solver is the cause and the sampler choice is a measured decision, not a
  default; plateau near production -> the learned conditional (go to T2-T3); in between -> both, with the split measured.
- Guard: a proper latent score can favour an under-dispersed solver (Codex: production had the best joint energy score
  while missing deaths); read event rates, never the bulk score alone.

T2. Sampler-free likelihood ratio (link 2, no solver, no head).
- Claim: the model's conditional assigns the dangerous successor a probability far below its recorded frequency.
  Competing: it assigns roughly the right probability (then T1's solver explanation holds). Control: for a context,
  compare the diffusion ELBO of the recorded successor with that of a counterfactual differing ONLY in the HUD health
  cell (the previous frame's token: "no damage"), same conditioning and common noise draws; the denoising loss is a
  weighted integral of ELBOs (Kingma & Gao) and ELBO comparisons across alternatives are the diffusion-classifier method
  (Li et al., ICCV 2023). Average implied log-odds per context class vs recorded log-odds.
- Positive controls: deterministic visible HUD changes (drinking at water, eating a cow: the recorded change must win
  by a wide margin); negative control: ordinary frames (an unprovoked health drop must lose). Fixations covering the
  HUD are excluded (the glimpse would carry the cell too).
- Outcomes: odds far too low in risky contexts while controls pass -> the learned conditional underweights danger;
  odds right -> the loss is downstream (solver or interface).

T3. Information in the conditioning (link 1).
- Claim: the conditioning contains what predicts damage (adjacency, hidden cooldown phase, health). Competing: it does
  not, so even a perfect generator could not place the risk. Control: event-probability probes (the probe's objective is
  the judged quantity: calibrated P(damage at t)) on (a) the predicted belief + action, (b) the 8 raw context percepts +
  actions, (c) both, beside the simulator-state reference (adjacent at L1 1, cooldown <= 0, health: the attack rule).
  Report AP, log loss, calibration; positive control: the reference must be near-deterministic.
- Outcomes: information present but not generated -> link 2 (objective / rarity); absent -> memory / perception.

T4. Consequence catalogue (is it rarity, stochasticity or hidden state?).
- Event classes by rarity x cause: deterministic and visible (drink, eat, collect, place), rule + hidden state (zombie
  damage, death), stochastic (zombie spawns, arrows, cow moves), common (scroll, walking). Recorded vs generated rates
  through T0-valid instruments, under the production and the converged solver, with T2 odds where a cell counterfactual
  exists.
- Separates: rare-but-deterministic events also missing -> rarity under the objective; only hidden-state / stochastic
  missing -> solver or information.

T5. Bulk vs tail (is the distribution broadly wrong, or only its tails?).
- Precision / recall of generated percepts (k-NN manifolds, Kynkaanniemi et al. 2019; density / coverage, Naeem et al.
  2020) in projected token space; two-sample classifier after projection (geometry removed); rank histograms of scalar
  summaries (health change, visible counts, reward) over K = 16 futures per history, stratified by context (Hamill 2001:
  a U shape can be conditional bias, and flat is not proof of calibration).

T6. Replication: P1 world seed 2 (the declared budget, ~9 h), so no conclusion rests on one world (our >= 2 seeds rule).

## Order and cost
T0 -> T1 and T2 (the decisive pair) -> T3 -> T4 / T5; each is a frozen-world read under 1 GPU hour. T6 can run
overnight when the GPU is free (one GPU job at a time).

## What we will not do
No loss, sampler or architecture change until the chain is localised; no reweighting dose (generic doses failed for
measured reasons, E14); no reader result on generated codes without T0 validation; no single-world claim; every probe
carries a positive control (several 10-28 readers failed theirs).

## Separate S1 items, not in this plan
The 256-D percept projection and Mamba output read food / drink less exactly than the raw percept (dedicated readers,
Codex); the auxiliary pixel decoder emits a digit-1 template for HUD fields although the tokens are readable; the
departed-entity probe failed its own control (the 8-step context, which contains the frames, scored like the belief).
