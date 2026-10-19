# Matched encoder test: review of Codex's proposal, reconciled with the frozen-state results and the original hypothesis

2026-10-17. PROPOSAL, not registered, not launched (user: "Don't launch it yet").

Inputs:
- Codex `../20261017_encoder_recipe_audit/REPORT.md`, `LEARNING_CURVES.md`, `sampler_recipe.md`;
- the frozen-state control (stage A done; stage B running);
- a new read-only calibration of existing D-class worlds (`d_adapter.py`, `run_d_calibration.sh`; no training).

## 1. The hypothesis, as originally posed

WHITEBOARD §19: *"Our encoder was trained only through its CLS objective. Does an encoder whose patch representations are trained
for action-conditioned prediction give a better predictive world state than frozen Raw?"*

User, 10-17: *the cheapest experiment that actually tells us whether patch training improves spatial dynamics and decisions.*

Operationally this needs:
- one capable downstream learner, held fixed;
- frozen encoders as the only difference;
- matched data, exposure, initialization and readers;
- capability gates that the Raw arm demonstrably passes, so that a null is interpretable.

## 2. What the existing evidence already settles

| question | evidence | settled? |
|---|---|---|
| Static content in the backbone | MoP-s1 backbone-LN zombie AP 0.986 vs Raw 0.987; tiles 0.970 vs 0.977 | yes: no static advantage, no static loss. Not predictive sufficiency |
| Is the **projected** interface a usable world state? | Stage A: a fresh predictor on the frozen full-arm deployed tokens (FD), with Raw's exact init and data stream, erases like the joint predictor (zombie relative decodability 0.089 / 0.071 vs J 0.078 / 0.083; Raw 0.80 / 0.89), from its first checkpoint (0.30 / 0.13 at 1k) | yes, for this learner: the erasure travels with the final projected interface, not with joint history. It is not proof of irreversible loss (Codex's fixed 5NN recovered 0.766). **The projector is excluded from the encoder test** |
| Is the full arm's **backbone-LN** usable by the same small learner? | Stage B (FB), running | pending (§6) |
| Can the Stage 2 learner test the hypothesis? | On Raw it fails every capability gate: screen copy, no motion beyond copy, consequences ≤ the action prior (R64) | no (Codex): a null there is uninformative |
| Can a D-class learner test it, and at what budget? | the calibration below | partly: depends on the capability |

## 3. Calibration: what a D-class learner on frozen Raw achieves, by budget (read-only)

Setup:
- existing scratch-D lineage (seed 7, 6-frame windows, B40, the exact D loss and terminal-enriched pool; snapshots 6k-36k), long D
  (42k, 16 frames) and control D (45k, §17's 0.711);
- scored on the audited gates: `location_dynamics.py` v2 and `consequence_within_root.py`; 4 context frames; 143 diagnosis
  episodes;
- the adapter's alignment was verified: D's map error is 0.15 / 0.12 of copy, vs 0.6-0.9 for the Stage 2 predictors.

| gate (zombie unless noted) | baseline | Stage 2 Raw-pred | D 6k | D 12k | D 18k | D 36k | D 45k (T16) |
|---|---|---|---|---|---|---|---|
| **consequences:** within-root attack AUC, HUD + near-player | action prior 0.62 [0.51, 0.71] | 0.51 / 0.69 | **0.85** [0.76, 0.93] | 0.82 | 0.81 | 0.82 | **0.95** |
| **persistence:** shifted-camera stays AP (cow) | world persistence 0.14 (0.38) | 0.001 (0.002) | **0.18 (0.52)** | 0.19 (0.52) | 0.21 (0.51) | 0.12 (0.40) | 0.16 (0.37) |
| **motion:** arrival AP | screen copy 0.079 | 0.044-0.048 | 0.023 | 0.029 | 0.061 | **0.233** | **0.291** |
| **motion:** stay-vs-vacated AUC | copy 0.66 | 0.65 | 0.48 | 0.51 | 0.59 | **0.75** | **0.80** |
| **coherent samples** | — | not coherent | n/a: D is deterministic | | | | |

Reading. A D-class learner on Raw passes:
- the **consequence** gate from **6k** (+0.11 to +0.37 over the action prior);
- the **camera-aligned persistence** gate from **6k**;
- the **motion** gate only **between 18k and 36k** (6-frame windows).

It cannot be scored on coherent samples. Consequences gain about +0.12 from 16-frame context.

Interval status:
- Consequence intervals are episode-bootstrap.
- Persistence and motion entries vs their baselines are point estimates, except arrival vs world persistence (intervals in
  `locdyn2_summary_dcal.json`).
- The comparator intervals (vs screen copy, vs world persistence) are to be computed at registration.

## 4. Codex's proposal: agreements, and the reconciliation

**Agreed, adopted unchanged:**
- frozen backbone-LN for both arms (never the projector; reinforced by stage A);
- **MoP-s1 vs canonical Raw-s1**: the same initial encoder tensors, so it isolates the patch-training recipe;
- an identical, fresh D-class downstream model, its exact recipe and hashed init, the same window / action tape, optimizer,
  precision, clipping and readers;
- frozen weights and buffers;
- each encoder's decoded-real ceiling reported;
- absolute scores with paired differences;
- **Raw must pass the relevant gate before a MoP negative is interpreted** (and a MoP positive is reported either way);
- no warm-starting the candidate from a Raw-trained world;
- staging decisions on opened data only.

**Reconciled / changed by the calibration:**
1. **Gates are capability-specific with different onset budgets.** A 6k / 12k / 18k schedule can reach consequences and
   persistence (both pass at 6k) but **cannot reach motion**, which Raw passes only by 36k at T6. Codex's caution was right: an 18k
   cut would report a Raw motion-gate failure by construction.
2. **The Raw arm already exists.** The scratch-D lineage *is* a fresh D-class model trained on frozen Raw-s1 tokens:
   - the training code is unchanged (tworld.py sha 2c31cce7… equals the hash embedded in its checkpoints);
   - seed 7; a reproducible pool (seeded windows from the contracted M4 corpus, stored episode ids);
   - snapshots at 6k / 12k / 18k / 24k / 30k / 36k; logged objectives every 500 updates.
   - A matched comparison therefore needs to train **only the MoP arm**: the same code, seed, pool windows and sampler, with the
     pool re-encoded by MoP-s1's backbone-LN. This halves the cost and keeps the Raw arm exactly as it was.
   - Two verifications come before any MoP training:
     - **R0a:** the historical pool tokens equal `PatchState('raw@10000')` on the same frames (Codex: the cache's bridge encoder
       equals Raw joint 10k).
     - **R0b:** rerunning the Raw recipe for 1k updates reproduces the logged objective (0.1654 @ 500, 0.0842 @ 1000) within a
       pre-declared tolerance.
3. **6-frame (T6) windows first, not a fresh T16 / B16 pair.**
   - Raw passes the consequence gate at T6 / 6k. The hypothesis is about spatial representation, and T6 is a valid
     within-context comparison (Codex: "cheaper but narrower").
   - A fresh T16 recipe is unvalidated, needs a fresh Raw arm and a ~40 GB pool cache, and would still not reach motion by 18k.
   - T16 is a later extension, and only if T6 shows a difference worth extending.
4. **Coherent stochastic samples and recursion are outside a D-class test.** D is deterministic. Testing them on matched encoders
   needs the IC / PI lineage (~48k per arm). This experiment does not claim them.

## 5. Proposed design (the cheapest decisive version)

| stage | MoP-arm training | what is compared (Raw snapshot at the same budget) | pre-declared use |
|---|---|---|---|
| R0a / R0b | — (Raw recipe 1k rerun, ~10 min) | pool-token identity; objective reproduction | abort if either fails |
| pool | MoP-s1 backbone-LN encoding of the identical pool windows (≈ 30-33k × 6 frames, ≈ 6 GB, ~15 min GPU) | stored episode ids / windows identical | — |
| **A: 6k** | ~55 min | **consequences** (within-root attack AUC vs action prior); **camera-aligned persistence** (shifted stays vs world persistence); relative decodability; decoded-real ceilings | Raw passes both gates at 6k (calibrated). MoP − Raw paired intervals read here |
| **B: 12k** | +~55 min | the same (confirmation; Raw at 12k) | a decisive negative on both ⇒ stop |
| **C: 36k (conditional)** | +~3.6 h | **motion** (arrivals vs screen copy, vacate AUC), plus all of the above | runs only if B is not decisively negative; Raw passes motion at 36k by point estimate (arrivals 0.233 vs screen copy 0.079) |

**Statistics:**
- paired episode-cluster bootstrap over the 143 diagnosis episodes (identical units in both arms);
- tie-grouped AP; each encoder's own backbone-LN readers fitted by the Tier A procedure;
- the readings, the margins and the "decisively negative" rule are registered before stage A.

**Replication:**
- One downstream seed per arm gives no estimate of downstream-seed variance. No D-loss seed-8 Raw run exists.
- A positive result therefore requires a second downstream seed for **both** arms before any robust claim.
- A general claim about encoder training would also need a second paired encoder-pretraining seed (Codex).

**Cost:**
- the decisive core (R0 + pool + A + B): about 2.5 h GPU;
- with the conditional motion stage C: about 6 h.
- Codex's two-arm T16 design: about 4-6 h for 12-18k with a fresh Raw arm, a ~40 GB cache and no motion reading.

**What results would mean:**
- **MoP > Raw on consequences and / or persistence** (Raw passing): the patch-training recipe improves a D-class world's
  decision-relevant prediction at matched budget. A component result: one encoder seed each, T6, no coherence or recursion claim.
- **MoP ≈ or < Raw with Raw passing:** evidence against the hypothesis for this recipe and learner at this budget.
- **Raw failing a gate:** no encoder reading on that capability. This should not occur at the calibrated stages.

## 6. Dependence on stage B (frozen backbone, running)
- FB tests the full arm's backbone-LN with the small Stage 2 learner. That learner fails the gates on Raw, so FB ≈ Raw is
  uninformative about the hypothesis. **FB > Raw** on a gate would be the first positive sign for patch training, and would be
  reported.
- **FB erasing like FD** would show that the full arm's backbone, not only its projector, is hard for the small learner. That
  raises the prior that MoP-s1's backbone needs the capable learner to show anything, and makes §5 more, not less, necessary.
- FB is on the full arm; §5 uses MoP-s1. A full-arm D-class arm can follow if MoP-s1 is positive.

## 7. Decisions for the user (none taken)
1. Candidate encoder: MoP-s1 (recommended, as Codex) vs full-s1.
2. T6 within-context first (recommended) vs a fresh T16 pair.
3. Reuse the verified historical Raw arm (recommended, after R0a / R0b) vs retraining Raw.
4. The conditional 36k motion stage.
