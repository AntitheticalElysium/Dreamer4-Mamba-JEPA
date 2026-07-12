# Neutral implementation and verification handoff

You are taking over an experimental model-based RL repository. Treat both the
existing repository and the attached compact implementation as **untrusted
research prototypes**. Your job is not to defend the proposed architecture.
Your job is to determine, through source inspection and controlled tests, what
is correct, what is unsupported, and what must change before any expensive run.

Hardware constraint: **one GPU with 6 GB VRAM**.

Files in this handoff:

- `model.py`: JEPA representation encoder, temporal adapter, future predictor,
  reward/continuation heads, and shadow reliability system.
- `agent.py`: actor/critic and strictly indexed imagination loop.
- `data.py`: raw-frame episode replay and thin Crafter adapter.
- `train.py`: 6 GB-safe starting defaults and optimisation steps.
- `smoke_test.py`: CPU-safe shape, indexing, reset, gradient, and optimiser tests.
- This document.

## 1. Do not assume the proposal is correct

Start by writing an audit report. Explicitly look for:

- future leakage;
- action-label leakage;
- action/reward off-by-one errors;
- repeated action conditioning that changes the intended transition semantics;
- stale recurrent caches after world-model updates;
- target encoder components that are not actually EMA/frozen;
- unsupported claims that a signal is uncertainty;
- mode collapse or input-agnostic successor codebooks;
- mismatches between sequence training and recurrent imagination;
- hidden memory costs that make the design unsuitable for 6 GB VRAM.

Do not start a long run until the report is complete and the relevant unit tests
fail before—and pass after—each correction.

## 2. Verify against primary source code

Read the implementation, not only the papers. Record exact commit hashes.

### Core baselines and architecture

1. `fmi-basel/Dreamer-CDP`
   - Inspect the RSSM observation and imagination paths.
   - Inspect continuous deterministic representation prediction.
   - Inspect dynamics/representation KL losses.
   - Confirm exactly which state predicts reward and continuation.

2. `facebookresearch/vjepa2`
   - Inspect context/target encoder construction, EMA updates, masking,
     predictor inputs, dense losses, and action-conditioned V-JEPA 2-AC code.
   - Do not assume a large pretrained V-JEPA model is appropriate for Crafter.

3. `state-spaces/mamba`
   - Inspect the current Mamba-3 and Mamba-2 constructors, sequence forwards,
     `step()` methods, and inference-cache allocation.
   - Pin a commit. Mamba-3 APIs and kernels have changed.
   - Verify support on the actual GPU. Do not fake a recurrent cache.
   - Add a numerical sequence-versus-step equivalence test.

4. `realwenlongwang/drama`
   - Inspect its Mamba world-model sequence layout, sampling strategy,
     actor/critic input, reward timing, replay ratio, and hardware footprint.
   - Treat its hyperparameters as local to DRAMA unless reproduced.

5. `danijar/crafter`
   - Inspect observation dtype/shape, action space, reset/termination semantics,
     episode statistics, and official evaluation score.
   - Confirm the installed package API rather than assuming Gym signatures.

6. Dreamer 4 implementations:
   - `edwhu/dreamer4-jax`
   - `nicklashansen/dreamer4`
   - There may be no canonical official implementation. Clearly distinguish
     paper claims from reproduction-specific choices.
   - Inspect shortcut forcing, tokenizer/world-model interfaces, policy
     objective, task-token causality, and imagination sampling cost.

### Relevant reconstruction-free and JEPA world models

7. `fmi-basel/Dreamer-CDP` and its paper.
8. `facebookresearch/vjepa2` and V-JEPA 2 / 2-AC papers.
9. LeWorldModel (`lucas-maes/le-wm` or the current official repository).
10. JEDI: Joint Embedding Diffusion World Model.
11. MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models.
    - If no official code is available, say so.
    - Reproduce its verification controls before claiming that hard modes model
      transition modes:
        - input-agnostic codebook control;
        - shuffled-context test;
        - router-gated readout;
        - transition precision;
        - route validity.
12. I-JEPA and the original LeCun JEPA/AMI architecture papers.
13. R²-Dreamer and NE-Dreamer if their code is available and relevant.

### Imagination reliability and efficiency

14. “On Training in Imagination”
    - Use it for error/sensitivity diagnostics and sample-allocation reasoning.
    - Do not infer that Mamba delta or any internal activation is uncertainty.
15. `nicklashansen/dreamer4` / MMBench2 hallucination work if available.
16. `leor-c/horizon-imagination`
    - Inspect efficient diffusion-world-model rollout schedules if a flow or
      diffusion predictor is reconsidered.

## 3. Architectural hypotheses to challenge

The current proposal makes these hypotheses; none is established:

1. Dense `8x8` tokens are better than a pooled state for Crafter.
2. A small spatial-attention mixer plus temporal Mamba is a good division.
3. Mamba-3 is beneficial at the short sequences and tiny width affordable on
   6 GB VRAM.
4. A hard-assigned predictor mixture is a sufficiently faithful and cheaper
   stochastic future model than shortcut flow/diffusion.
5. The generated target representation can be fed into the temporal state
   without a distribution mismatch that destabilises imagination.
6. Reward gradients should shape the JEPA representation.
7. Register tokens retain inventory/global state.
8. The four reliability inputs predict actual rollout error.
9. Score-function actor updates are competitive enough for this setting.

Design minimal experiments that can falsify each hypothesis.

## 4. Required changes before a real Mamba run

The compact code intentionally does not implement a guessed Mamba-3 recurrent
cache. After pinning the official package:

- implement `MambaSequenceAdapter.init_state()`;
- implement `MambaSequenceAdapter.step()` by calling the official API;
- test recurrent output against sequence output on random inputs;
- test reset isolation;
- test mixed precision;
- test gradients in sequence training;
- test inference-cache memory;
- test sequence lengths 16, 32, 64, and 128.

If Mamba-3 is unsupported or slower/less stable on this GPU, use Mamba-2 as the
research backend and document that outcome. Do not silently call the GRU result
a Mamba result.

## 5. Six-gigabyte VRAM plan

Start smaller than the proposed maximum:

- token dimension: 48 or 64;
- local grid: `8x8`;
- registers: 2;
- spatial depth: 1;
- temporal depth: 1;
- Mamba state: 16 or 32;
- modes: 2, then 4 only if justified;
- replay on CPU as raw uint8;
- batch size: 2–4;
- sequence length: 16;
- imagination batch: 16–32;
- imagination horizon: 5–8;
- AMP;
- gradient accumulation;
- no simultaneous optimiser graphs;
- freeze world-model parameters during actor/critic updates;
- measure allocated and reserved VRAM around every phase.

Do not optimise around parameter count alone. Record peak activation and cache
memory.

## 6. Mandatory validation sequence

### Phase A — static audit

- Run `smoke_test.py`.
- Compile all files.
- Add tests for no future leakage and no action-label leakage.
- Verify target EMA includes the entire representation encoder.
- Confirm exact transition timing.

### Phase B — representation only

Train only encoder/target/predictor on a small replay dataset.

Report:

- target variance;
- covariance spectrum;
- effective rank;
- spatial probes;
- inventory/health probes if obtainable from Crafter `info`;
- held-out one-step prediction;
- shuffled-target control;
- copy-latent baseline.

### Phase C — successor modes

Compare:

- deterministic predictor;
- 2-mode hard mixture;
- 4-mode hard mixture only if memory permits.

Run all MoP-style controls. Do not interpret mode spread as epistemic
uncertainty.

### Phase D — temporal backend

Using the same representation and predictor:

- GRU correctness control;
- Mamba-2;
- Mamba-3 if functional.

Compare:

- held-out one/multi-step latent error;
- throughput;
- peak VRAM;
- recurrent latency;
- state/reset correctness.

### Phase E — task heads

Verify reward and continuation calibration from real prefixes. Ensure action_t
maps to reward_{t+1}. Test synthetic transitions with known rewards.

### Phase F — shadow reliability

Construct true targets from held-out real continuations:

\[
e_{t,k}=d(\hat Y_{t+k},Y_{t+k}).
\]

Train reliability only on earlier replay and evaluate on held-out seeds and later
policy checkpoints. Report:

- Spearman/Pearson correlation;
- AUROC for top-error quantiles;
- calibration plots/error;
- performance under distribution shift.

Keep `use_reliability_weights=False` until these tests succeed. Then compare
soft weighting against no weighting. Hard truncation is last.

### Phase G — policy imagination

Begin with short horizons and fixed world-model checkpoints. Verify:

- imagined reward against real reward from identical prefixes;
- action entropy;
- action histogram;
- value ensemble calibration;
- policy performance over multiple seeds.

Only then enable alternating online world-model and policy updates.

## 7. Baselines

Keep separately runnable controls rather than gradually mutating one model:

- random policy;
- compact Dreamer-CDP or official configuration feasible on 6 GB;
- deterministic JEPA predictor;
- GRU temporal backend;
- Mamba-2;
- Mamba-3 if supported;
- mixture predictor;
- reliability off versus shadow versus calibrated weighting.

Use identical environment steps, replay, evaluation seeds, and checkpoint rules.

## 8. Evaluation

Do not use only “achievements above random.”

Report:

- official Crafter score if available;
- return;
- number of achievements;
- paired per-seed differences;
- learning curves;
- at least 3 training seeds for screening;
- 5+ for final claims if compute permits;
- world-model throughput;
- actor-imagination throughput;
- peak VRAM;
- wall-clock;
- parameter count.

## 9. Doubts that must remain explicit

- Dreamer 4 code references are reproductions unless an official release is
  found.
- Mamba-3 may not be practical on this GPU or at these sequence lengths.
- MoP-JEPA is extremely recent; code and independent reproduction may be absent.
- A finite set of modes may poorly model continuous stochastic futures.
- Mode routing can become an input-agnostic codebook.
- The JEPA energy and manifold projector may co-adapt and cease to diagnose OOD
  imagination.
- Reliability learned on one policy distribution may fail after the policy
  changes.
- Crafter may not be stochastic enough to reveal benefits from multimodal future
  models.
- The proposed architecture is specified but not mathematically proven to
  outperform Dreamer-CDP, Dreamer 4, or DRAMA.

## 10. Decision authority

You may reject or revise any part of the attached implementation. When doing so:

1. cite the source code or controlled experiment that motivated the change;
2. identify which architectural contract changes;
3. update tests before training;
4. run the smallest discriminating experiment;
5. record negative results.

Do not make narrative-driven fixes. Do not treat an encouraging metric as proof
of its preferred explanation.

## 11. Initial deliverable

Before launching training, return:

1. source commit table;
2. code audit with severity-ranked findings;
3. revised architecture diagram and transition equations;
4. 6 GB memory estimate measured by phase;
5. test results;
6. minimal experiment matrix;
7. clear go/no-go decision for:
   - Mamba-3;
   - hard predictor mixture;
   - reliability weighting;
   - full online policy training.
