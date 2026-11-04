# P1-F2 preflight

2026-11-03. **Ready for launch review; overnight training NOT launched.**
No production run directory, training service, timer or automatic launch was
created. Disposable smoke outputs live in `checks/` only.

-12 CPU contract/queue tests passed (`unit_tests.log`): exact canonical-loop AST
 after removing declared hooks; original policy/input equivalence; no current
 target in the forecast; episode resets; exact coarse encoding without a crop;
 unchanged prefix observations; isolated gradients/RNG; trace recovery; source/
 cfg matching; paired-evaluator labels/cohorts; one-shot failure dispatch.
-Actual F1.6 checkpoint provenance matches current world source, corpus, targets,
 torch and full world cfg. The control's actual first logged update matches all
 nine recorded numerical loss/gradient/sigma fields in the canonical repeat.
-Canonical P1 versus local P1 with uniform control, **four real updates with the
 auxiliary head genuinely training**: bitwise identical world weights, carried
 belief, cache, stream state, counters and default CPU/CUDA RNG. All losses match.
-Exact-resume learned arm: world weights/carry/cache/data/RNG, head parameters,
 head optimizer/RNG/clock, gaze bytes and chooser-log bytes all match.
-200 actual64-step native updates, finite logged losses; mean final-half1.709s/
 update.200 auxiliary updates. Full unchanged uniform monitor runs successfully.
-Three256-step updates plus a fully logged three-update confirmation: all terms
 and final world parameters finite;12 auxiliary updates per three chunks.
 Warm updates2/3 take6.030/6.040s. No scientific claim from these short trajectories.
-Peak observed allocation4.296GiB. Full-budget timing is an estimate: about11–12h
 training plus monitor/checkpoint overhead; evaluation follows separately.
-Saved world reload, real S1 encoding on two dev episodes, actual cascade sampling
 (81x16 periphery,9x64 fine,8D light), CUDA-autocast coarse equivalence and online-
 head held-out loading/policy all passed. Actual gaze-analysis label schema also
 smoke-tested on an isolated mechanics checkpoint. Long evaluation fits are not
 rerun before training; the unchanged estimators are reused.

One initial strict check failed: world Adam moments differed by1.137e-13 while
weights and all other world state matched exactly. Repeating **canonical P1 by
itself** produced the same1.137e-13 moment discrepancy. The learned-resume moment
maximum was2.274e-13. These are retained in `checks.log`, `checks_attempt2.log`,
`checks/canonical_repeat/` and `checks.json`; moment equality is qualified with a
1e-11 absolute bound, not described as bitwise equality. The head state and emitted
policy trajectory did match bitwise. No learning recipe changed to pass this.

The source/input manifest seals code, fixed inputs and qualification outputs.
`run --dry-run` validates hashes and prints commands without starting a job.
The one-shot worker would train the full unchanged F1.6 budget, select by the
unchanged uniform-monitor rule, then run `queue.sh` once. Standard common-gaze
readings compare with F1.6; extras assess genuine-look uptake, the frozen online
head on fixed dev cohorts, and actual gaze acquisition. No P2/actor successor.

## Proposed launch, only after the user requests it

```sh
systemd-run --user --unit=saccade-p1f2 \
  -p MemoryMax=24G \
  --setenv=MAMBA_DETERMINISTIC=1 \
  --setenv=JAX_PLATFORMS=cpu \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --working-directory="$PWD" \
  "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261103_p1f2.run
```

Check live GPU ownership before any launch. The online learning clock is explicit
in PLAN/chooser.json: one B256 head update per1024 world frames. This is a new
prospective working setting, not a previously validated frozen-head result.
