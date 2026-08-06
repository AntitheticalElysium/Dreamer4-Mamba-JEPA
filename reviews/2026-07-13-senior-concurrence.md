# Senior concurrence with the 2026-07-13 consensus re-audit

Role note: several findings in the re-audit are corrections of **my own**
2026-07-12 work. I verified the load-bearing ones independently before writing
this; where the re-audit is right, the corresponding claims in my earlier
reviews are retracted below. Earlier documents are left unedited as the record.

## Independent verification of the critical findings

1. **False anti-collapse certificate — CONFIRMED by my own reproduction.**
   A synthetic encoder that ignores the observation entirely (fixed per-position
   codebook + 1% noise) satisfies my flattened variance term (loss 0.05) and
   scores flat rank 51.5/64; the corrected streamwise statistic assigns it
   variance loss 0.99. My P1 "pass" certified position diversity, not
   observation sensitivity. The re-audit's long-run measurements (observation
   variance share 42.6%→10.8% while flat rank rose) complete the picture. I also
   note, for the record, that I *observed* the symptom — the unrelated-pair
   "ruler" shrinking 0.149→0.07 across my runs — and failed to chase it.
2. **Backend attribution invalid — CONFIRMED from my own artifact.** In
   `rollout_loss_experiment.json`, relative to each model's own copy drift at
   k=8, GRU = 0.174/0.201 = 0.865 vs Mamba-2 = 0.116/0.122 = 0.949: by the only
   cross-space-comparable ratio, GRU was *better*, opposite to my headline.
   Absolute cosine across independently trained latent spaces is not evidence.
3. **7.5× Mamba speed claim — RETRACTED, mechanism confirmed.** My per-step
   latency averaged 16 steps including the first call; the GRU arm loaded a
   checkpoint and paid its attention-kernel compilation (~164 ms) inside my
   timed window, while Mamba-2 had just trained and was warm. One cold step
   averaged into 16 reproduces my 14.9 ms figure almost exactly. Warm,
   order-stable numbers: GRU 1.44 ms vs Mamba-2 1.91 ms per imagined step.
4. **Unseeded replay — CONFIRMED by inspection** (`np.random` in
   `EpisodeReplay.sample()`); my "identical data/seed" claims across arms were
   overstated: identical episode pools, non-identical sampling.
5. Corrected artifacts, 44-test suite, repaired AMI PDF, PAPERS.lock: all check
   out as reported.

## Retractions of my 2026-07-12 claims

- "Anti-collapse fix confirmed at 13× budget" → **retracted** (false
  certificate; the corrected streamwise variant is the honest arm, and it
  passes only P1/P3).
- "D1 crossed for the first time" → **retracted as a gate result** (measured in
  a latent space holding a false certificate, with endogenous changed-token
  selection, no CI, unmatched sampling). The rollout bridge survives as a
  source-backed, unit-tested **mechanism** — efficacy to be established per the
  matrix, step 3.
- "Mamba-2 validated default (+7.5× recurrent speed)" → **retracted**. Backend
  choice returns to an open question answered only by matrix step 4 (one frozen
  shared representation, matched replay indices, seeds, CI).
- Crafter multimodality "not supported" → **downgraded to provisional** per the
  probe critiques (branch-count-dependent statistic, no reward/termination
  divergence, register-only pooling in the latent probe). The operative
  decision — deterministic predictor by default — stands.

What survives of my work: the collapse diagnosis of the original objective, the
invalid original semantic probe, the copy-baseline analysis, the cross-token
predictor requirement (now source-qualified as an adaptation, which is
accurate), the M1–M5 bridge analysis and the option-1/2/3 reasoning, the MoP
scale argument (now labelled suggestive), and the rollout mechanism.

## Answers to the two standing questions

### Are the "not a faithful port" disclaimers true/intentional?

True, and they now say the right thing. Three tiers must be kept distinct:

1. **Necessary adaptations** — no pinned source implements anti-collapse for
   dense token grids, or a JEPA predictor feeding a separate temporal SSM core;
   this architecture is a hybrid by design, so *every* port is an adaptation.
   These are legitimate when (a) labelled and (b) each load-bearing deviation
   carries its own falsification control (the position-codebook regression is
   the model example).
2. **Intentional, defensible deviations** — cosine invariance (matches our loss
   geometry), no VICReg expander at d=64, online-branch-only statistics,
   learned positions + action/horizon conditioning tokens. Fine, and now
   documented in-code.
3. **Unrecognized deviations** — my sample-axis flattening. I did not flag it
   because I did not see it as a deviation; it inverted the regularizer's
   meaning for dense grids and manufactured a false pass. This tier is the
   dangerous one, and the lesson is procedural: porting a method to a new data
   topology (images → token grids) changes *what the axes mean*, and every axis
   choice must be justified against the source's definition of "sample".

### GRU default vs the Mamba thesis

The re-audit's "GRU remains default correctness backend" should be read as
*scaffolding, not thesis abandonment*, and I concur with it for now:

- The thesis is that a Mamba-core world model is viable/advantageous on 6 GB —
  a claim to be **demonstrated**, not asserted via defaults. Right now backend
  efficacy is unmeasurable (Phase B failing; past arms trained different latent
  spaces). Flipping the default to Mamba-2 today would repeat the old project's
  central error.
- The thesis is alive on the engineering side: Mamba-2's parallel training path
  is *faster* (sequence 1.35 vs 1.73 ms; full world update 30.3 vs 34.3 ms),
  memory fits with headroom, and the exact official cache/step semantics pass
  all tests on this GPU. Its recurrent step is slower at d=64 (0.86 vs 0.11 ms)
  — kernel-overhead-dominated at toy width, and both are ~2 ms/step in
  deployment shape, i.e., immaterial at our scale.
- The decisive test is matrix step 4: frozen shared representation, same
  non-temporal weights, matched replay indices, ≥3 seeds, paired CI on task
  error. If Mamba-2 wins there, it becomes the default *with evidence*, which is
  the only kind of default this project should have. Until then, GRU-as-default
  is a correctness harness, and every gate run should continue to execute both
  backends.

## Process concurrences and one addition

- **Stop decision: concur.** No arm passes the corrected Phase B; more tuning
  of the hybrid destroys attribution.
- **Matrix: concur** with the 6-step order (true I-JEPA/SIGReg image-level
  objective first, isolated from action prediction). One addition for step 1:
  pre-register the pass thresholds numerically before the first run, in the
  report header, as was done for the rollout experiment — the re-audit's
  corrected gates are defined but not yet quantified as pass bars.
- **Process gap to close:** the re-audit's ~1,200 lines landed uncommitted
  (now committed as 763224a). Both agents must commit at every checkpoint;
  reviewer verification depends on diffable history.
- A recurring pattern across both audit rounds, worth writing on the wall:
  **every "pass" so far that later fell was a metric artifact, not a model
  property** (global-map probe, flattened rank, cold-kernel timing, endogenous
  changed tokens, chronological inventory split). The corrected harness now
  embeds the right instincts: untrained-baseline rows, falsification controls,
  fixed exogenous selections, paired CIs, warm timing.
