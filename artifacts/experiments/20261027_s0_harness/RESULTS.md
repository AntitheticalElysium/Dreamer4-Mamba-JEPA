# S0 harness — results (2026-10-27)

TEST_SPEC v1.1, stage S0 (no model). Everything here is measured on simulator labels through the SAME observable
quantities a percept reader will later output (7x9 entity-kind grid, terrain, HUD, camera shift, action, reward,
termination), so the same code runs on generated trajectories.

| step | code | output (`artifacts/eda/s0/`, gitignored) | env |
|---|---|---|---|
| 1 labels | `s0_extract.py` | `s0_dev_labels.npz`, `s0_train400_labels.npz` | project `.venv` (JAX CPU) |
| 2 one-step gates | `s0_gates.py` (also the shared helpers) | `s0_gates.json` | numpy + lightgbm |
| 3 revisits | `s0_revisit.py` | `s0_revisit.json` | numpy |
| 4 multi-step oracle | `s0_oracle.py` | `s0_oracle.json` | project `.venv` (JAX CPU) |
| 5 pixel probe | `s0_probe.py` | `s0_probe.json` | project `.venv` (CUDA) |
| 6 outcome calibration | `s0_calib.py` | `s0_calib.json` | numpy + lightgbm |

## Step 1 — labels
Exact replay of all 1,009 dev episodes (304,898 transitions; eps 0.1 / 0.25 / 0.5 / 1.0: 235 / 253 / 264 / 257) and of
the first 400 train episodes (128,564 transitions; the probe's training set). Checks: rewards recomputed from replayed
achievements and health equal the stored rewards (max |diff| 3.6e-8); the terrain view's orientation matches the stored
frames (per-class mean colours: water blue, grass green, diamond cyan with std ~1 across 10 episodes). 1,004 dev episodes
terminate; 30.7% of actions are exogenous (collector flags regenerated from the seed tree, A8).

## Step 2 — one-step gates, references and power (`s0_gates.json`)
References from dev half A (even episodes); a gate rejects at |B - A| > 3 joint episode-bootstrap SE; 43 gates.
| case | gates rejecting | classifier two-sample test accuracy |
|---|---|---|
| real half B (must pass) | 0 | 0.495 |
| shuffled histories | 38 | 0.922 |
| mismatched actions | 2 (self-motion) | 0.758 |
| duplicated entities (30%) | 11 (appearance, counts) | 0.543 |
| delayed consequences (+2 steps) | 6 (damage timing) | 0.495 |
| frozen time | 34 | 0.812 |
Every corruption is rejected by at least one gate family; the real half passes. The classifier two-sample test alone
is blind to delayed consequences and nearly blind to duplication, so it is never a gate on its own. Re-run on the
re-extracted labels (step 1 added the terrain view): references and rejections identical bit for bit.
Selected references: P(health drop | zombie adjacent) by steps since the last drop 1-5: 0.023 / 0.039 / 0.033 / 0.039 /
0.037, >= 6: 0.353 +- 0.014 (the hidden 5-step attack cooldown, visible at the observable level); no zombie adjacent
0.006. Arrow appears given a skeleton at Manhattan 4-5: 0.170, otherwise 0.014. Camera moved given a movement action
0.785, direction agrees 1.000. Night zombies: edge appearance 0.045 / step, interior 0.003, in-view vanishing 0.006.

## Step 3 — revisits (`s0_revisit.json`)
A revisit = a cell entering the view that was seen before; gap = steps since it was last in view; terrain compared
where no entity covers the cell at either sighting.
| gap | all dev: agreement (n) | revisit cells / transition | eps 1.0: agreement (n) | cells / transition |
|---|---|---|---|---|
| 1-16 | 0.99997 (216,849) | 0.71 | 1.0 (17,069) | 0.45 |
| 17-64 | 0.9999 (144,965) | 0.48 | 1.0 (9,609) | 0.25 |
| 65-256 | 0.99983 (88,055) | 0.29 | 1.0 (2,373) | 0.063 |
| > 256 | 0.99923 (34,913) | 0.11 | 1.0 (8) | 0.0002 |
Terrain is effectively static (the only changes are the player's own edits), so a revisited cell must look as it did.
Power: half B passes in every stratum; a 5% forgetting corruption (revisited terrain redrawn from the terrain marginal)
is rejected in every stratum (-0.035 to -0.037). Against simulator truth this is trivially strong (minimal detectable
drop <= 0.0008), so the BINDING reference for generated revisits is the frozen reader's agreement on REAL revisits
(its own error enters twice), recomputed after P1 together with this power check. Long gaps come from expert-heavy
episodes: S2's random-play windows reach > 64-step gaps only through the real prefix.

## Step 4 — multi-step gates with a simulator oracle (`s0_oracle.json`)
All 2,236 dev windows of 16 consecutive exogenous actions (all from eps = 1.0; 257 episodes). From the replayed start
state, 16 futures with the recorded actions and fresh environment keys: set A (8) plays a perfect model, set B (8) is
the null. The oracle conditions on the FULL simulator state, so its scores are a floor no belief can reach.
Sanity (the sampler reproduces the logged dynamics): randomised PIT of the recorded future among the 16 samples is
uniform (quartile shares 0.24-0.27 for health drops, visible count at the end, displacement; SE ~0.009); recorded vs
oracle persistence and visible count agree within 2.1 SE; set A vs set B: no gate rejects.
References: joint-trajectory energy score (per-step summary vectors, 320 dims scaled by the recorded futures' std,
absorbing terminal convention, fair pairwise term): oracle 5.68, marginal (8 recorded futures of other episodes'
windows, history ignored) 11.56. Entity persistence (same kind within j cells at t+j, k = 1-4): oracle 0.987 / 0.969 /
0.955 / 0.942, recorded 0.987 / 0.967 / 0.951 / 0.937; mean visible entities 1.06 (oracle) / 1.05 (recorded).
Power (corruptions of set A at the entity level, in world coordinates; difference to set B in episode-bootstrap SE):
| corruption | energy score | persistence (k = 1..4) | visible count |
|---|---|---|---|
| flicker (p 0.2 per step) | +0.108 (3.7 SE) | -0.18 ... -0.46 | -0.21 |
| teleport (p 0.1, +-2 cells, stays) | +0.38 (9.8 SE) | -0.09 ... -0.17 | -0.06 |
| phantom drift (a cow every 4 steps) | +4.65 | +0.006 ... +0.033 | +1.66 |
| vanishing drift (p 0.05, for good) | +0.40 (11 SE) | -0.05 ... -0.15 | -0.37 |
| frozen time | +6.77 | +0.013 ... +0.058 | -0.16 |
| delayed consequences (HUD +2 steps) | +0.26 (8.6 SE) | (none) | (none) |
All six are rejected; the energy score is the weakest on flicker (3.7 SE) and the only multi-step family that sees
delayed consequences; persistence and counts are the strongest on entity bookkeeping. Because the oracle sees hidden
state, a model's energy score is reported with the oracle and marginal references beside it, never as pass / fail;
the pass / fail S2 gates are persistence, counts and the step-2 event probabilities applied to rollouts. Sample
counts declared for S2: m = 8 trajectories per window, all 2,236 windows.

## Step 5 — pixel-probe reference for S1.1 (`s0_probe.json`)
Definition (shared with the token probe after P1, which swaps the cell's pixels for its token and the frame mean for
the lighting latent v): one map cell's pixels + the frame's mean map RGB -> MLP 150-256-256 (GELU) -> kind (5) and
terrain (17) heads; cross-entropy at natural frequencies (terrain only where no entity); AdamW 1e-3; trained on 360
train episodes (7.26 M cells) until 40 held-out episodes stop improving for 3 epochs; 2 seeds; read on all dev.
| | presence recall | presence precision | kind accuracy on entities | terrain accuracy |
|---|---|---|---|---|
| day, all rings | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| night, ring 1 | 0.9997 / 1.0000 | 1.0000 / 1.0000 | 0.9997 / 1.0000 | 0.9999 / 0.9999 |
| night, ring 2 | 0.981 / 0.984 | 0.9999 / 0.9997 | 0.981 / 0.984 | 0.996 / 0.997 |
| night, ring 3 | 0.913 / 0.923 | 0.994 / 0.991 | 0.911 / 0.920 | 0.978 / 0.979 |
| night, ring 4 | 0.845 / 0.855 | 0.984 / 0.974 | 0.841 / 0.849 | 0.963 / 0.962 |
| night, all | 0.937 / 0.943 | 0.995 / 0.993 | 0.936 / 0.942 | 0.982 / 0.982 |
(seed 0 / seed 1; episode-bootstrap SE 0.001-0.005.) Night per-kind recall: zombies 0.943 / 0.955, cows 0.962 / 0.953,
skeletons 0.905 / 0.923, arrows 0.838 / 0.855. The renderer adds per-frame random static at night whose intensity grows
toward the screen edge, so a cell-local reader loses entities at rings 3-4 at night: that is a limit of the single
percept, not of the probe. A belief that tracked the entity from closer can exceed it (an S1.2 question). Probe-fit
spread between seeds (~1% at night) exceeds the bootstrap SE: token-vs-pixel comparisons use >= 2 probe seeds and
report the spread.

## Step 6 — reward and continuation evaluator and references (`s0_calib.py`, `s0_calib.json`)
Facts (dev): reward = unlocks + 0.1 x health change, with the unlock part 0 or 1 per step (3.45% of steps); termination
(0.33%) happens exactly when the HUD shows health 0 at t+1, and never otherwise. So given the next percept (the
outcome models' conditioning, STEP_SPEC 2.5) the health part and continuation are READABLE; the unlock part is not
displayed and needs the visible change plus memory of which achievements are unlocked.
Evaluator: reward NLL in nats of the recorded value (each recorded value its own bin of the discretised density);
unlock-event Brier and ECE (10 equal-mass bins); continuation NLL / Brier / ECE; episode-bootstrap SE.
Information references (LightGBM fits are upper bounds on what that information allows: a better learner can only
lower them):
| reference | reward NLL | unlock Brier | unlock ECE |
|---|---|---|---|
| R0 marginal | 0.353 +- 0.005 | 0.0333 | 0.0031 |
| R1 + HUD health part (exact) | 0.150 +- 0.003 | 0.0333 | 0.0031 |
| R2 + visible changes of the percept pair, no memory | 0.0243 +- 0.0006 | 0.0069 | 0.0015 |
| R3 + achievements unlocked before t (perfect memory) | 0.0068 +- 0.0002 | 0.0018 | 0.0012 |
Continuation: marginal NLL 0.0221, Brier 0.0033; reading health 0 from the next percept: 0 (exact). The S1.3 reward
reading is placed on this ladder: below R2 means the belief carries unlock memory.

## Fixes during S0 (bug fixes, documented)
- Probe fit: a fixed 4-epoch budget left the held-out loss falling and the seeds ~8 SE apart on night arrows
  (`s0_probe_4epochs.json`); replaced by train-until-held-out-plateau before any token probe exists.
- Oracle PIT scalars counted health drops and displacement of samples AFTER their termination (the simulator keeps
  stepping a dead state); truncated at termination (energy score and persistence already were); shares moved <= 0.0013.
- Labels re-extracted with the terrain view; all step-2 outputs reproduced exactly.

## Carried into S1 / S2
- Every reference above is recomputed THROUGH the frozen reader on real trajectories once it exists (end of P1), and
  the power checks repeated, so reader error cancels in the comparison and is reported separately.
- The random-feature reference (an untrained encoder of the same architecture) needs the encoder: S1.
- Coverage: 16-step windows exist only in random play.
