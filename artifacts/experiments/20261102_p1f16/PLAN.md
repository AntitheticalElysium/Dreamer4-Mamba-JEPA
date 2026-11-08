# P1-F1.6: local spatial fusion, from scratch

2026-11-02. STATUS: APPROVED, mechanical checks before launch. The user approved
the revised mandatory joint64 route and instructed smoke tests followed by
overnight launch and automatic matching evaluation. The earlier optional16D
correction prototype is superseded; its logs remain historical checks.
P1-F2's learned chooser is deferred. This is one new world
seed1 run compared with P1-F1.5 seed1; not a seed-robust general result.

## Hypothesis and the one architectural change

The frozen projection audit established that useful fine information survives
but is difficult to access in its native location-dependent code. It did not
establish irreversible erasure or train a repaired native model. Test whether
shared nonlinear local integration before compression makes the fine evidence
more useful to belief and native future predictions when acquired FROM SCRATCH.

For the nine observed locations only:

```
f_i = out_map(Linear128->64(GELU(Linear88->128([p_i(16), g_i(64), v(8)]))))
e = existing_projection([original_coarse_grid, fused_fine_canvas, fixation_mask, v])
```

The same19,648-parameter network applies at every looked-at location. Its layers
use ordinary nonzero Linear initialization. The output uses the EXISTING
per-token out_map (channel LayerNorm, same tanh clamp) to avoid shrinking the
fine route merely by adding a randomly initialized MLP. All original weights and
post-construction RNG states remain exactly P1-F1.5's via isolated CPU initialization
(seed611901); initial belief inputs intentionally differ. Hidden128, normalization
and initialization are explicit working choices, not paper-derived
optima. There is no learned position table, label, named object, HUD path, new
noise estimate or independent-sensor assumption. Light is already observed.

Keep the original coarse path, and REPLACE the raw64D fine-to-belief path with
joint64 tokens. No separate raw-fine bypass enters Mamba; fine evidence is not
forced through a16D replacement. Keep256D mixed output and the same Mamba/denoiser. Fusion is internal
to `Belief.percept_embed`, used identically in real and imagined belief/outcome
updates. It does NOT change the observation encoder or prediction targets.
Current-target fine content never enters that frame's predicted belief. TSP
still predicts before reading the crop. Corrected local evidence is not a new
persistent spatial map. Existing outcome-query dependence may change and is read.

## Source review before implementation/launch

Reviewed AFF method3.2/4 and ablations, CF-ViT feature-reuse module/fine-stage
training, and FiLM's conditioning/residual architecture. They support aligned,
content-dependent feature integration; none establishes this recurrent world
model or its unsupervised acquisition claim. Details in RESEARCH.md.

## Matching and budget

Control: `20261101_p1f15/run/ckpt_10.pt` with its actual saved config. The launcher
checks this equals P1-F1.5's recipe from current canonical config. Candidate cfg
differs ONLY by `belief.spatial_fusion={kind:joint_fine,hidden:128,init_seed:611901}`.
No fine-tuning: all baseline weights use seed1's original initialization.

Full25.6M-frame budget;16 streams;64 then256-step chunks with original gradient
accumulation; original loss terms/weights, weak coupling, optimizer, schedule,
uniform fixations, corpus, monitor, sampler and checkpoint rules. Same stream
and training-noise RNG. No learned or hand-picked gaze, loss switch, corpus
change, sampling change or width increase. One GPU research job at a time.

Ten numbered checkpoints plus latest are retained. Checkpoint selection reuses
`saccade.eval.after_p1.select_checkpoint`: final unless3 strict consecutive monitor
increases occur, then global minimum (earliest exact tie). Complete the full
budget regardless. No selection by semantic outcomes and no automatic extension.

## Pre-launch qualification

- Legacy config/state dictionaries and all legacy initial weights/RNG unchanged;
  fusion intentionally changes the input embedding, not any observation target.
- Arbitrary batch/time shapes, all49 legal windows, original percept immutable;
  joint features only at observed cells and shared under a matched spatial shift;
  a dedicated test excludes a raw fine bypass.
- Fine/coarse/light gradients reach the local module; its parameters must
  receive a real future-G1 gradient after the denoiser's
  zero-initialized output heads have begun learning. The first version of this
  test incorrectly required a nonzero upstream G1 gradient at untouched native
  initialization; it failed as those heads block all upstream conditioning
  gradients at initialization. Correct post-warm-up qualification is still open.
- Full native P1 objective trains all modules (no M0.5v2 freezes/G1-only recipe).
- Actual launcher smoke with finite terms;64/256-phase memory/throughput; resume
  versus uninterrupted weights/carry/cache/RNG and identity rejection checks.
- Strict loading/evaluation of the F1.5 control remains functional. New checkpoint
  reload uses its saved fusion cfg automatically, including CPU inference.
- Evaluation dispatch and paths smoke-tested without overwriting F1.5 evidence.

These are implementation checks, not a scientific miniature trained result.
Failure blocks launch until resolved. No short efficacy gate selects this run.

## Automatic post-run evaluation on equal grounds

After successful training, validate completed budget/checkpoint/source identities,
record fixed checkpoint selection, then run the same F1.5 PLAN readings1–6:

1. Full canonical S1, including freshly fitted checkpoint-specific readers.
2. Actor interface under random AND guided gaze, M0.5 evaluator and geometry,
   decoder rendering, fixed re-render/noise/counter panels, CPU illumination/
   real-counter reads, and cascade sampler factorial. F1.5 is the paired control
   for interfaces/M0.5, and its saved matching Codex panels supply comparisons.
3. All-query consistency and real-versus-generated conditioning gap.
4. Past-glimpse uptake, same250 episodes/windows/noise on CPU alongside GPU reads.
5. Existing locus contingency/surprise reading with the same contexts.

All estimators, episode counts, seeds, fixations and primary sampler remain those
of F1.5. Reuse existing script entry points; wrapper edits only redirect output
and world labels. Assert paired contexts where existing code does so. Keep
Euler3/churn1 primary;8/churn0 and32/churn1 secondary, no best-of selection.
Evaluation runs even on scientifically bad readings. No actor/P2 job follows.
Readings have explicit status/failure logs; no silent retrial or infinite polling.

Primary scientific question: does this package improve native future behavior
and useful past-glimpse uptake without material regressions? Report the complete
same battery, including negatives. Improvement in reconstruction alone does not
establish repair. This compares the local-fusion package against unchanged F1.5;
without an extra-capacity nonlinear control it cannot isolate locality from all
other aspects of the extra19.6k parameters. Budget allows one arm tonight, and
that limitation is declared before its outcomes.

## Launch qualification and execution

All pre-launch checks passed; see CHECKS.md/checks.json.63 GPU regression tests,
3 queue tests,200 native64-step updates, full monitor, actual cascade,3 updates
at256 steps, strict reload, exact resumed weights/carry/cache/RNG. Peak4.25GiB
allocated. No scientific gate selected this arm from smoke results.

Launched as `saccade-p1f16.service`, invocation
`bbfeabb74895465a986d3d92ef328471`, one-shot training then evaluation worker.
GPU initially free;818GiB disk available.24GiB host-memory cap, deterministic
Mamba, expandable CUDA segments,6 CPU math threads. No restart loop/timer/P2 job.
All48 pinned sources and a recoverable source archive are saved. Initial live
training record reproduces the smoke's first losses and gradient norm exactly.
