#!/usr/bin/env bash
# Seed-1 confirmation of the one-block mixer repair. Control is abt1, same world seed,
# same streams, same lambda; only the mixer differs.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
[ -f phase1b_abm1_n64/training_report.json ] || \
  stage K1_train_mixer1 "$PY" train_phase1b_fork.py --n-latents 64 --suffix abm1 \
    --lam 1.4758 --steps 20000 --world-seed 1 --mixer --out phase1b_abm1_n64
for m in 5000 10000 20000; do
  [ -f "phase1b_delta_abm1_n64_$(printf %06d $m).json" ] || \
    stage "K2_eval_${m}" "$PY" reevaluate_phase1b_delta.py --n-latents 64 --suffix abm1 \
      --milestone $m --mixer
done
stage K3_traj "$PY" reevaluate_phase1b_delta.py --trajectory --suffix abm1 --n-latents 64 \
  --milestones 5000 10000 20000
stage K4_path "$PY" probe_direct_path.py --n-latents 64 --suffix abm1 --mixer
stage K5_matching "$PY" probe_action_matching.py --suffix abm1 --mixer
echo "mixer seed1 finished $(date -Is)" >> triage.log
