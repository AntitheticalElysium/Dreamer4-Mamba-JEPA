#!/usr/bin/env bash
# Endpoints for the no-tanh A/B. Waits for all four training runs.
#
# Mechanism panel, declared before the results:
#   geometric fidelity  action-effect NSE and cosine
#   semantic transfer   R_delta, with the three fixed rp512 controls
#   safety gates        output magnitude/range and recursive drift under NOOP
# Primary is the paired no-tanh minus tanh difference within each world seed; a claim
# needs both seeds to agree, since root bootstraps quantify evaluation-root
# uncertainty and say nothing about training-seed uncertainty.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

until grep -q "tanh A/B training finished" triage.log; do sleep 30; done

for w in 0 1; do
  for arm in t n; do
    suffix="ab${arm}${w}"
    extra=""; [ "$arm" = "n" ] && extra="--no-tanh"
    for m in 5000 10000 20000; do
      [ -f "phase1b_delta_${suffix}_n64_$(printf %06d $m).json" ] || \
        stage "G1_eval_${suffix}_${m}" "$PY" reevaluate_phase1b_delta.py \
          --n-latents 64 --suffix "$suffix" --milestone $m $extra
    done
    stage "G2_traj_${suffix}" "$PY" reevaluate_phase1b_delta.py --trajectory \
      --suffix "$suffix" --n-latents 64 --milestones 5000 10000 20000
    stage "G3_path_${suffix}" "$PY" probe_direct_path.py --n-latents 64 \
      --suffix "$suffix" $extra
  done
done
echo "tanh A/B eval finished $(date -Is)" >> triage.log
