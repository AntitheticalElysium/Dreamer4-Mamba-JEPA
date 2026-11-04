# P1-F2.1 launch qualification

User approved the diversity arm, fixes, commit, launch and automatic evaluation
on 2026-11-04. P1-F2.2 is deferred. No loss, architecture, acuity, fixation-policy,
optimizer or budget change is bundled into this run.

## Data and actual training path

Claude completed assembly, full union verification and the single required GPU
preflight; Codex inspected and adopted their artifacts, rather than running a
duplicate. `targets.json`, `union_checks.json`, `checks.json` are the records.

- All 1,060 shard hashes checked with no mismatch; 24 union reads equal source
  reads. Original train is the union prefix; original monitor/dev/final unchanged.
- New teacher targets cover 8,937,169 emitted train frames. The old target block
  is copied bitwise; the frozen teacher reproduced 12 spread old episodes bitwise.
- Actual train: 22,267 episodes / 8,914,902 transitions. Original monitor alone
  selects checkpoints; 140 coverage and 625 risk monitor episodes are held out.
- GPU preflight: exact model, streams, carry, cache, state, RNG, chooser and gaze
  bytes after resume. Optimizer maximum absolute difference 1.42e-14.
- Fresh-process launcher passed; 200 actual updates at 64 steps and 3 at 256;
  full 2,000-window monitor finite; online head updated in both phases.
- Final-half update means 1.788 s at 64 and 6.511 s at 256; peak allocation
  4.296 GiB. These are short preflight timings, not a whole-run timing promise.
- Preflight trained-frame shares: 27.6% original, 42.8% coverage, 29.6% risk;
  corpus frame shares are 27.3%, 41.6%, 31.1%. No source weighting introduced.

## Repairs and evaluation contract

- `run.py` now dispatches `evaluation.py` after the unchanged selection rule,
  passing the actual selected checkpoint. All 10 numbered checkpoints retained.
- Output-isolated wrappers reuse the usual F2 instruments. Original references
  remain immutable; readers are fitted to this checkpoint where required.
- The actor interface is additionally read under each world's own head, with
  query gaze chosen from generated coarse evidence, on old dev and each new
  held-out source. Old random/guided readings remain, with the guided cue caveat.
- `heldout.py` assembles labels from hashed collection sidecars, never simulator
  replay with old keys. All 765 episodes are excluded from training; sampled
  action/reward/termination sequences agree exactly with the stored episodes.
- Source seals include shared worker/check dependencies, the new evaluation
  wrappers, held-out labels and immutable comparison inputs.
- The plan no longer excludes coal, iron, damage or death from possible diversity
  gains. No-large-train-advantage does not prove more diverse contexts cannot help.

Codex CPU suite: 61 passed, 6 GPU tests skipped while the actual GPU preflight
ran independently. Four new tests cover selected-checkpoint routing, failed
evaluation status, queue output isolation and new held-out membership/alignment.
`eval_checks.json` records the bounded GPU evaluation-wrapper smoke (identical
F2 checkpoints must give zero paired difference on all three distributions).

## Launch

```sh
systemd-run --user --unit=saccade-p1f21 -p MemoryMax=24G \
  --setenv=MAMBA_DETERMINISTIC=1 --setenv=JAX_PLATFORMS=cpu \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --working-directory="$PWD" \
  "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261104_p1f21.run
```

`STATUS.json` distinguishes training, evaluation, completion and failure;
`readings/STATUS.json` names each evaluation command and its log. This does not
launch P2/P3 or the deferred 2x2 arm. Scientific interpretation remains diversity
at fixed 25.6M-frame compute, not increased exposure to every rare outcome or a
test proving fine information is necessary.
