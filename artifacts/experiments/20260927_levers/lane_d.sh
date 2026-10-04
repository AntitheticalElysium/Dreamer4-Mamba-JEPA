#!/usr/bin/env bash
# Lane D: every non-lane-A GPU job, strictly one at a time, after lane C4. A job that dies of CUDA OOM is retried
# (up to 6 times, 3 min apart) instead of being skipped; any other failure stops the lane.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
D=artifacts/eda/levers_ldad_v1
LOG=${LOG:?}
run() {  # run a command with OOM retries
  for attempt in 1 2 3 4 5 6; do
    tmp=$(mktemp)
    "$@" > "$tmp" 2>&1; code=$?
    cat "$tmp" >> "$LOG"
    if [ $code -eq 0 ]; then rm -f "$tmp"; return 0; fi
    if grep -q "out of memory" "$tmp"; then echo "RETRY_OOM attempt $attempt: $*" >> "$LOG"; rm -f "$tmp"; sleep 180; continue; fi
    echo "FAILED (not OOM): $*" >> "$LOG"; rm -f "$tmp"; exit 1
  done
  echo "FAILED after OOM retries: $*" >> "$LOG"; exit 1
}
screen() { # variant lam
  local out=$D/$1_lam$2_screen2000
  [ -f $out/step-002000.pt ] && return 0
  rm -rf $out
  run $PY $L/ldad_joint.py --variant $1 --lam $2 --steps 2000 || return 1
  [ -f $out/step-002000.pt ] || { echo "FAILED: no checkpoint for $out" >> "$LOG"; exit 1; }
}
screen raw 1
screen raw 0.1
screen tc 1
screen tc 10
JAX_PLATFORMS=cpu run $PY $L/ldad_eval.py --screens
JAX_PLATFORMS=cpu run $PY $L/ldad_eval.py
echo LANE_D_DONE >> "$LOG"
