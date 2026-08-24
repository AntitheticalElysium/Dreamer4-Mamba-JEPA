#!/usr/bin/env bash
# Causal coverage test at fixed data volume. Both arms train the promoted mixer Direct
# on 1,825 fit roots -- half the full set -- differing only in which roots. Random
# against k-center spread in pooled space, selected from fit roots only. Same
# initialization, updates, lambda, streams and evaluator, so any difference is coverage
# and not volume.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
stage N1_random "$PY" train_phase1b_fork.py --n-latents 64 --suffix cvr0 --lam 1.4758 \
  --steps 20000 --world-seed 0 --mixer --fit-subset random --out phase1b_cvr0_n64
stage N2_spread "$PY" train_phase1b_fork.py --n-latents 64 --suffix cvs0 --lam 1.4758 \
  --steps 20000 --world-seed 0 --mixer --fit-subset spread_pooled --out phase1b_cvs0_n64
echo "coverage A/B finished $(date -Is)" >> triage.log
