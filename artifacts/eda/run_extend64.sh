#!/usr/bin/env bash
# Direct64 extension, 20k -> 40k. Both 64-slot arms are still climbing at 20k, so
# 0.261-0.355 is a training point rather than a demonstrated ceiling.
#
# Literal continuation: resume.pt restores modules, optimizer, both RNG streams and
# the draw stream, and neither the LR schedule (step/warmup only) nor this script's
# sampling depends on the total step count -- so 40k is the same run continued, not a
# differently-scheduled one.
#
# Registered before the results: ceiling iff the paired 20k->40k increment in R_delta
# has a bootstrap CI including zero in BOTH seeds; still climbing iff it excludes zero
# in both. The 32-slot arms are deliberately not extended, so the 64-32 paired
# statistic stays anchored at its registered 20k comparison.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

until grep -q "direct path localization finished" triage.log; do sleep 30; done

for suf in s1 s0; do
  [ -f "phase1b_${suf}_n64/world_040000.pt" ] || \
    stage "C1_extend_${suf}_n64" "$PY" train_phase1b_fork.py --n-latents 64 --suffix $suf \
      --lam 2.3761 --steps 40000 --milestones 30000 40000 --out "phase1b_${suf}_n64"
done
for suf in s1 s0; do
  for m in 30000 40000; do
    [ -f "phase1b_delta_${suf}_n64_$(printf %06d $m).json" ] || \
      stage "C2_eval_${suf}_${m}" "$PY" reevaluate_phase1b_delta.py \
        --n-latents 64 --suffix $suf --milestone $m
  done
  stage "C3_traj_${suf}" "$PY" reevaluate_phase1b_delta.py --trajectory --suffix $suf \
    --n-latents 64 --milestones 5000 10000 20000 30000 40000
done
echo "direct64 extension finished $(date -Is)" >> triage.log
