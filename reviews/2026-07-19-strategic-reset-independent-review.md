# Strategic reset: from predictive diagnostics to control evidence

Date: 2026-07-19  
Independent reviewer: Codex companion  
Reviewed HEAD: `f295a7d` (`Record Stage-2G negative outcome`)

## Bottom line

The present wall is not best explained as a Mamba bug, a general failure of
JEPA, or Crafter being unsuitable. The strongest diagnosis is a mismatch
between the system we trained and the system we want to deploy:

1. the compact model is trained primarily as a decision-agnostic predictor on
   a fixed 40k-transition uniformly random replay;
2. it is then expected to support eight-step counterfactual reward planning;
3. no value/TD objective, policy-improved replay, or source-faithful
   posterior/prior state model supplies control pressure;
4. the offline diagnostic gates have never been calibrated against executed
   control.

The gates found real failures. They should not be discarded. But the project
has treated a conservative conjunction of proxy metrics as though it were a
validated control objective. That inference has not been tested.

Stop the Stage-2 sequence of local reward-objective edits. The next
discriminating work should be:

1. calibrate the gates with a bounded, no-learning executed planner comparison
   across the existing checkpoint zoo;
2. run a source-faithful control baseline, with Dreamer-CDP the closest Crafter
   reference;
3. only then decide whether to retain the compact factorization or move JEPA
   prediction into an auxiliary role inside a decision-aware world model.

## What Stage 2G establishes

The committed Stage-2G result is a valid negative result, not an obvious
implementation failure.

- The historical A, C-L, and C-LR predictions reproduce exactly, which rules
  out evaluator drift.
- The actual-action auxiliary probes learn their labels strongly.
- That learning does not produce the counterfactual action geometry needed by
  the planner readouts.
- G-LA and G-LRA both fail their registered operational comparisons.

The most important interpretation is narrower than “task supervision does not
work.” Labels derived from the action actually taken permit state and action
frequency shortcuts. Separating observed reward event/sign under the behavior
distribution is not equivalent to distinguishing unrealized suffixes from an
identical prefix.

The experiment also remains a one-training-seed screen. Its clustered
bootstrap quantifies evaluation-cluster variation, not training-seed
variation. That is acceptable for rejecting these candidates, but not for
general architectural claims.

## Why the project appears to have produced only negatives

It has not produced only negatives. It has established all of the following:

- the corrected Step-1 representation avoids collapse and carries semantic and
  inventory information;
- held-out action conditioning is weak but real and survives shuffled-action
  controls;
- the fork oracle shows large unused branch-discrimination headroom in frozen
  latents;
- rollout supervision improves changed-patch long-horizon discrimination;
- spatial mixing and full-grid recurrence improve action separation;
- the official Mamba-2 cache/API path is correct and fits the target GPU;
- generated-state task supervision can repair ranking and continuation;
- generated latent targets improve latent fidelity;
- reward gradients causally alter shared action geometry;
- the Dreamer-style reward operator causally reduces false reward;
- observed-action event/sign separability is not counterfactual usefulness.

Those are useful findings. The reason they have not accumulated into a useful
agent is that most recent experiments changed one loss inside the same custom
offline system and then required a Pareto improvement across every diagnostic.
The objectives trade off:

- C-L improves latent fidelity and damages action ranking;
- C-LR restores ranking/reward discrimination and creates false reward;
- head adaptation repairs ranking without calibrating magnitude;
- Stage 2F reduces false reward without closing the deployment gap;
- Stage 2G teaches an easier observed-action classification problem.

This pattern is evidence of objective and regime mismatch, not evidence that
every component is defective.

## Are the gates wrong?

### Keep as hard correctness gates

- transition and reward indexing;
- episode-boundary/reset handling;
- no future leakage;
- target encoder freeze/EMA contract;
- recurrent cache reset and sequence/step equivalence;
- canonicalized environment observations and actions;
- source/checkpoint/config compatibility.

### Treat as diagnostics until calibrated against control

- latent cosine error;
- reward AUROC/AP/Pearson/MAE;
- the present absolute zero-suffix false-return threshold;
- the full conjunction of “no worse” K1/K8 metrics;
- small fork-set ranking statistics.

These diagnostics are informative, especially same-prefix suffix ranking and
false-reward behavior. But neither the `A + .02` false-return budget nor the
entire Pareto conjunction has been shown necessary or sufficient for executed
Crafter performance.

The Stage-2 protocol said that an evaluation-only planner harness would be
built during the stage (`reviews/2026-07-18-stage2-ab-protocol.md:53-56`).
No random-shooting/CEM planner or executable runner exists. The generic
`train.py` still contains only a note to build that runner after validation
(`m3_hjwm_compact/train.py:102-107`). This is a missed deliverable and the
largest current measurement gap.

## Component diagnosis

### Mamba: unlikely to be the root cause

GRU and Mamba fail in similar ways under the same supervision. Mamba-2
sequence/step correctness and memory feasibility have been checked, and larger
or longer-context Mamba screens did not reverse the quality result. At the
tested short planning horizon, a GRU is a strong low-overhead baseline and the
selective-SSM long-context advantage is not automatically useful.

Mamba should remain a thesis/backend hypothesis, not the organizing
explanation for the current failure. Hold Mamba-2; do not scale it again until
a decision-aware pipeline produces a useful control signal. Then compare
matched recurrent cores.

### JEPA: not generally incompatible with imagination

Dreamer-CDP is the direct counterexample. At pinned source commit
`a851fa3e3d70b624b094ee1810ad4bb602346092`, it combines a JEPA-style
continuous deterministic prediction loss with:

- a stochastic RSSM posterior/prior and asymmetric KL losses
  (`dreamerv3/rssm.py:129-146`);
- live reward gradients;
- reward, continuation, actor, and value learning;
- policy imagination on every world-model update
  (`dreamerv3/agent.py:217-302`).

This differs fundamentally from using same-frame masked-image JEPA features and
future cosine prediction as the sole organizing representation objective.

The correct conclusion is: the current frozen target and training objective
have not produced a sufficiently decision-aligned imagination state. It is not
“JEPA cannot be used for imagination.”

### Crafter: exposes the regime mismatch; it is not the sole cause

Crafter is sparse-reward, partially observed, and compositional. Uniformly
random data gives poor coverage of achievement chains and weak support for
counterfactual actions. Moving to a more complex environment would worsen that
identification problem.

Crafter is still learnable. The older repository's best rigorously evaluated
checkpoint achieved `3.883 +/- .240` achievements versus random
`2.217 +/- .158`, paired `+1.667 +/- .281`, over 60 episodes
(`runs/jepacnn/ls_eval.log:7-12`). That is not proof of a faithful world model:
later runs collapsed, the action repertoire was narrow, and the old code has
an action-label leakage loss and a reward-imagination indexing defect. It is
evidence that online, policy-shaped data and state-conditioned control can
beat random in this environment.

### Architecture: plausible, custom, and not yet anchored

The compact predictor/temporal split, pooled recurrent readout, deterministic
rollout, and dense bypass are a custom factorization. No inspected source uses
the whole combination. It lacks a stochastic posterior/prior alignment and a
value/TD objective. The frozen encoder and K=2 training versus K=8 deployment
add further assumptions.

That does not prove the factorization is wrong. It means it should remain a
separately runnable experimental control, not be treated as the default thesis
architecture merely because many local ablations have been performed on it.

### Objective and data regime: highest-probability causes

The current data are collected uniformly at random
(`m3_hjwm_compact/verification/ssl_step1.py:40-65`) and the temporal screen is
built around a fixed 40k-transition cache
(`m3_hjwm_compact/verification/step3_temporal.py:31-67`). The loss rewards
predicting average visual futures; controllable and reward-relevant factors can
be a small part of that signal.

This agrees with the primary literature:

- Dreamer-CDP uses prediction as one term inside a stochastic, reward/value,
  online actor-learning system.
- TD-MPC2 uses joint latent prediction, reward, TD value, and a policy prior;
  planning scores predicted rewards plus terminal value.
- “When does Self-Prediction help?” concludes that latent self-prediction is
  most defensible as an auxiliary representation objective rather than an
  isolated decision objective.
- JEPA-WMs and RC-aux separately show that prediction quality and planning
  geometry need not coincide.

## Missing source-faithful controls

The original handoff explicitly required separately runnable random,
Dreamer-CDP, deterministic JEPA, GRU, and Mamba controls
(`m3_hjwm_compact/HANDOFF.md:259-272`). The Dreamer-CDP control was never run.
This is now more important than another custom loss.

One correction is important for planning that control. The official
Dreamer-CDP README command is `--configs crafter --run.train_ratio 32`. The
`crafter` config does **not** inherit `size1m`; absent another size override, it
keeps the large default RSSM (`deter=8192`, 1024-unit heads). `size1m` through
`size400m` are available source-defined configurations, but a `size1m` run is
a mechanism-faithful, scale-divergent feasibility baseline—not a paper-scale
reproduction.

The current machine is an RTX 3060 Laptop GPU with 6144 MiB, and the project
virtual environment does not contain JAX. Dreamer-CDP therefore needs an
isolated environment and an explicit smoke/VRAM ladder. Do not silently alter
the source configuration and call the result a reproduction.

DRAMA at pinned commit
`a50bd54c34e77d1d13e988a031733a47817098e2` is the complementary Mamba
endpoint: Mamba/Mamba-2 recurrence inside an online stochastic world model with
KL, reconstruction, reward/termination, and actor/critic learning. It is not
evidence that replacing a GRU with Mamba in the compact offline model must
improve prediction.

## Next experiments, in order

### 0. Calibrate the gates with bounded executed control

Build one evaluation-only, no-learning receding-horizon planner. Evaluate the
existing A, C-L, C-LR, F-DZ, G-LA, and G-LRA checkpoints on fresh canonical
Crafter seeds.

Contract:

- identical random-shooting or categorical-CEM candidate budget;
- horizon 8;
- common candidate RNG across checkpoints;
- score
  `sum_k gamma^k (product_{j<k} continuation_j) reward_k`;
- execute only the first action and replan;
- matched random and state-independent policy controls;
- no checkpoint selection or tuning on these episodes;
- report paired reward, achievements, action histograms, predicted versus
  realized returns, and planner exploitation signatures.

Primary question: does C-LR's false reward actually induce planner
exploitation, or is much of it prefix-common bias that cancels when ranking
actions? Correlate each offline gate with executed outcomes across the
checkpoint zoo. A negative result is useful: it validates the safety gate. A
positive result is equally useful: it shows which proxy threshold was
over-conservative.

This is not a long online RL run. It is the missing validation instrument for
the gates.

### 1. Establish source-backed reality anchors

Run an isolated Dreamer-CDP feasibility ladder:

1. import/install and CPU smoke;
2. one-update and one-imagination-update correctness;
3. peak VRAM for source `size1m`;
4. try `size12m` only if memory permits;
5. short interaction-budget ladder, preserving posterior/prior KL, CDP,
   reward/value gradients, actor loop, and source learning-rate separation.

Label every scale and budget divergence. Use the same canonical evaluation
seeds and report wall time, environment steps, updates, and VRAM. The paper
configuration likely exceeds 6 GB; failure to fit is a result, not permission
to call a smaller model source-equivalent.

Canonicalize and re-evaluate the older best checkpoint alongside it. A DRAMA
smoke/short run is secondary; Dreamer-CDP is the direct reconstruction-free
Crafter control.

### 2. Isolate replay regime before topology search

Compare the same compact GRU world under:

- 40k uniformly random transitions;
- 40k policy-shaped transitions;
- an online aggregated replay ladder up to the older 150k-transition scale.

Match transition and optimizer-update budgets where possible. This tests the
highest-probability causal difference between the compact and older systems.

### 3. Re-center the thesis only after Tracks 0-2

If a source-shaped control learns:

- retain JEPA/CDP prediction as an auxiliary;
- retain stochastic posterior/prior alignment;
- retain reward, value/TD, and online replay;
- compare GRU and official Mamba-2 as the only changed temporal factor;
- test frozen encoder versus a small task adapter versus two-timescale
  unfreezing as a clean factorial.

The clean thesis candidate is then the intersection of the two verified
endpoints: reconstruction-free Dreamer-CDP-style learning plus a Mamba-2
temporal state model. It should not inherit the compact architecture's custom
factorization by default.

Use one tiny controlled action environment as a one-day diagnostic ladder. If
the current model cannot learn counterfactual reward and planning there, the
problem is implementation/objective. If it passes there and fails on Crafter,
coverage, partial observability, and task alignment move up the causal list.

## Explicit decisions

| Decision | Status | Reason |
|---|---|---|
| Another Stage-2 reward/TACO/AdaLN arm on current DEV | **NO-GO** | spent evaluation set and same local failure family |
| Bounded no-learning planner gate calibration | **GO** | smallest test of whether the gates predict control |
| Long compact online actor/critic run | **NO-GO** | no calibrated gate or source baseline |
| Dreamer-CDP feasibility and short budget ladder | **GO** | required source-faithful reality anchor |
| DRAMA smoke/short control | **GO after Dreamer-CDP setup** | complementary Mamba endpoint |
| Current compact architecture | **KEEP AS CONTROL** | useful evidence, not established thesis default |
| Mamba-2 | **HOLD, DO NOT REJECT** | correct backend; no quality advantage under current objective |
| Larger Mamba search | **NO-GO FOR NOW** | short-horizon control signal is not working yet |
| Frozen encoder forever | **REOPEN AS A FACTORIAL** | useful for attribution, not proven control-optimal |
| JEPA prediction | **KEEP AS AUXILIARY CANDIDATE** | source-backed, but not sufficient alone |
| Predictor mixture / reliability weighting | **NO-GO / SHADOW ONLY** | still uncalibrated |

## Consensus questions

1. Do we agree to stop optimizing the current DEV gate and build the bounded
   executed-control calibration harness next?
2. Do we agree that the omitted Dreamer-CDP baseline is now mandatory before
   further compact architecture search?
3. Do we agree to treat the compact architecture as a control, and to make a
   source-shaped stochastic reward/value system the new reference frame?
4. Do we agree that Mamba remains a matched backend hypothesis rather than the
   presumed cause or cure?

