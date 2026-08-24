#!/usr/bin/env bash
# Is 0.261-0.355 a Direct ceiling or the current training point?
# Rescore the 5k and 10k checkpoints with the same corrected dz metrics, then
# bootstrap each arm's R_delta increments. No training.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi; }

for suf in s0 s1; do
  for n in 32 64; do
    for m in 5000 10000; do
      [ -f "phase1b_delta_${suf}_n${n}_$(printf %06d $m).json" ] || \
        stage "A1_eval_${suf}_n${n}_${m}" "$PY" reevaluate_phase1b_delta.py \
          --n-latents $n --suffix $suf --milestone $m
    done
  done
done
for suf in s0 s1; do
  for m in 5000 10000; do
    stage "A2_paired_${suf}_${m}" "$PY" reevaluate_phase1b_delta.py --combine --suffix $suf --milestone $m
  done
  for n in 32 64; do
    stage "A3_traj_${suf}_n${n}" "$PY" reevaluate_phase1b_delta.py --trajectory --suffix $suf --n-latents $n
  done
done
echo "milestone rescore finished $(date -Is)" >> triage.log
