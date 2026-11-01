# M0 correction verification — 2026-10-27

**Production P1 can start next turn.** The implementation/evaluator defects below are corrected and tested. Production
P1 has not been launched. Model learning, full S1/S2 readings and actor performance remain unmeasured.

```text
frame -> encoder LayerNorm+tanh -> target z -> raw denoising loss
                                                |
                       sampler clean estimate -> range clamp -> belief
                                                preserves a valid z
```

## Corrections

| Audit finding | Final change | Verification |
|---|---|---|
| R1: sampler remapped an already valid target | Encoder map unchanged; both sampler paths share EDM completion plus idempotent [-3, 3] clipping | Exact raw denoiser has loss < 1e-10 and returns encoder targets within 1e-6; full-history cache parity |
| R2: systemd default working directory | Explicit repository working directory and absolute interpreter | Import-only transient unit exits 0, resolves saccade and LightGBM 4.6.0 |
| R3: AP tie order | Shared threshold-grouped AP | Constant balanced scores give .5 regardless of row order; teacher scores below |
| R4: classifier omitted conditioning | Eight complete history frames, every predicted belief, current belief/action/fixation; equal FP16 outcome precision | Sensitivity/no-target-leak tests; corrected classifier positive controls below |
| R5: memory labels required hidden survival | Kind-level observable sighting/absence history; last-visible nearest edge, not hidden exit position | Absence strata, reappearance, reset, player occlusion and hidden-mask invariance tests |
| R6: unlock probability used a different event | Sum reward-bin probabilities for round(reward - .1 * health change) = 1, evaluation-only health labels | All 304,898 dev transitions match the event under exact rewards; old threshold missed 115 of 10,528 unlocks |
| R7: frozen reader was lost | Save exact reader weights/schema/seed/label identity with checkpoint hash; frozen reload | Roundtrip output equality, wrong-checkpoint rejection and real smoke reload |

Also merged the duplicate YAML key; installed/pinned LightGBM 4.6.0 in .venv; checked corpus/target shape; bound resume
to cfg, seed, source/config/manifest/target hashes and torch version; distinguished blocked/unmeasured/passed; extended
the cache test to 15 frames with reset, branching and serialization; corrected JEDI's title and reward-gradient wording.
P2/P3 checks are explicitly deferred until those phases exist. Nuisance drift remains a separate diagnostic.

## Verification and bounded costs

- Full CUDA suite: **33 passed, 0 skipped, 42.92 s** (`tests.log`). The subsequently tightened resume assertion
  requires maximum model difference **exactly 0**; it passes separately in 32.32 s (`resume_exact.log`).
- New diagnostics preserve accumulated gradients exactly, RNG state and model mode, including hook cleanup on error.
- Audit seed 1898: **20 updates / 20,480 frames**, 16 x 64, all real-data losses/optimizer/routing; two checkpoints and
  two four-window monitors; **4.0063 GiB** peak, 51.14 s including initialization/monitors (`smoke_cost.json`).
- Audit seed 1899: **one 4 x 4 x 256 accumulated update**, including diagnostic and two-window monitor; **3.7130 GiB**
  peak, 26.19 s including initialization (`long_chunk_cost.json`). This single update is not a throughput estimate.
- Fresh checkpoint provenance matches current production source/config/data identities (`verification.json`).
- All S1 sections execute with **15 dev / 30 train-400 episodes** on that same 20-update checkpoint. The fitted reader
  is saved and reloads frozen. Binary checkpoints/readers remain local ignored artifacts; scripts and numeric results
  are versioned. There is no production world-seed-1/2 checkpoint from this work.

The initial diagnostic retained the full training graph for repeated backwards and **OOMed at the first 16 x 64
checkpoint** (`smoke_initial_oom.out`). The correction recomputes one stream capped at 64 new frames after training
backward, using saved incoming history and the same pre-update parameters. RNG is restored and .grad is untouched.
The identical bounded run then completed. This changes diagnostics, not the training objective or frame budget.

## Conditional classifier defect found during the smoke

The full-input tree classifier completed but scored .5 on shuffled histories and mismatched actions (`s1_smoke.json`).
It also scored .5 on a separating control: real pairs (x,x), synthetic pairs (x,-x), x balanced in {-1,+1} within every
episode. The two classes have exactly matching coordinate marginals and different dependence. Greedy coordinate
splits have no initial class-separating gain in this construction. The neural probe scores **1.0** on its held-out
episodes, and real-versus-itself scores **exactly .5** in the regression suite (`conditional_classifier.json`).

S1 now reuses its existing MLP 256-256 GELU and fit schedule (seed 0, <=30 epochs, held-out validation early stopping).
Means/std are computed from training examples only, std floor 1e-3. History/current-belief/action/fixation/outcome
groups receive equal expected input energy; all original inputs remain available. Donors and validation/test splits
stay within physical-episode partitions. Features are assembled in batches, avoiding duplicated full FP32 matrices.

| Smoke corruption | Full-input tree | Unbalanced neural input | Final neural input |
|---|---:|---:|---:|
| shuffled histories | .5000 | .8144 | **.6945** |
| mismatched actions | .5000 | .5011 | **.9407** |
| frozen time | .9776 | not run | **.7395** |

These are fixed smoke tuples, **2,818 test domain examples** from five held-out physical episodes, not a full-dev
scientific reading. The unbalanced neural failure and decreased history accuracy after balancing are retained in the
intermediate JSONs. Final context has **17,962** features; outcome **1,882**; group widths [17640, 256, 17, 49, 1882].
Final entry-point output: `s1_neural_smoke.json` (evaluation version 3). Its generated samples are distinguishable at
accuracy **1.0**, and the perception blocker reports **blocked**. Those poor tiny-checkpoint results are retained;
successful evaluator execution is not a trained-world acceptance result. Full P1 readings must recheck power.

## Teacher and data

The persistent-venv refit reproduces the historical AP/precision/recall at every printed digit under the legacy AP
definition. Corrected threshold-grouped AP is **.9430835717422393 day / .6120662290745817 night**, versus historical
.943060400146609 / .6120657088385467 (`teacher/change_teacher.json`). Canonical teacher/targets are unchanged:

- teacher SHA256: `69862f6b98e3ba770395d07bdb516c2cc0e326917c0f848c4a8aedadd109406f`;
- target SHA256: `f1a79ed7dba730db02b5efb537cc8f44fd580406c2da7894005b4e3e84c87b97`.

The new refit's serialized bytes differ: all 400 inference tree structures/splits match, maximum leaf-value difference
4.21e-13; on 10,000 stored cell examples, maximum probability difference 2.87e-14 and **zero FP16 mismatches**
(`teacher_equivalence.json`). This is numerical agreement, not a claim of bitwise refit identity. The production target
file was not regenerated. No final-split observation or outcome examples were inspected.

## Sources and reproduction

Rechecked [JEDI's pinned primary source](https://arxiv.org/html/2605.13013v1) and vendored V-JEPA 2-AC training
(`third_party/vjepa2/app/vjepa_droid/train.py:417-440`): its normalization is applied inside the loss. The sampler
correction follows [EDM Appendix B.3/B.6 and Algorithm 2](https://arxiv.org/abs/2206.00364), rather than treating an
untrained output remapping as equivalent. [C2ST](https://arxiv.org/abs/1610.06545) supports a learned held-out
classifier on the supplied features; it gives no power guarantee for a particular learner or omitted conditioning.

See PROTOCOL.md for the initial scope and both implementation amendments. Main commands, from the repository root:

```sh
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m pytest saccade/tests -q -ra
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m artifacts.saccade_m0.checks.review_fixes.smoke
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m artifacts.saccade_m0.checks.review_fixes.long_chunk
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m saccade.eval.s1 artifacts/saccade_m0/checks/review_fixes/smoke/latest.pt artifacts/saccade_m0/checks/review_fixes/s1_neural_smoke.json 15 30
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m artifacts.saccade_m0.checks.review_fixes.verify
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -m artifacts.saccade_m0.checks.review_fixes.conditional_classifier
```

The smoke scripts resume their local checkpoints if present; remove or relocate only their audit output directories
to reproduce initialization from scratch. Production launch uses CURRICULUM's explicit working directory next turn.
