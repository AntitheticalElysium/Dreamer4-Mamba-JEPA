#!/usr/bin/env bash
# Preregistered no-tanh Direct A/B.
#
# Changed: tanh(pre) versus pre at the output of the Direct readout. Nothing else.
# Identical parameters, initialisation, batches, optimizer, budget and data; lambda is
# held at the parity-calibrated 1.4758 in BOTH arms and is deliberately not
# recalibrated per architecture.
#
# Paired world seeds: --world-seed offsets world initialisation and the commit stream
# only. The numpy batch-draw stream is fixed, so the two arms of a pair see identical
# batches in identical order. Root-bootstrap intervals quantify evaluation-root
# uncertainty, not training-seed uncertainty -- which is exactly why the pairing exists.
#
# `assert_faithful` checks that tanh(no-tanh predict) reproduces production predict on
# shared weights before any no-tanh run starts, so the duplicated four lines cannot
# drift from `World.predict` unnoticed.
#
# If an extension rule fires later it must extend BOTH arms, never only the one that
# is still climbing.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

for w in 0 1; do
  for arm in t n; do
    suffix="ab${arm}${w}"
    extra=""; [ "$arm" = "n" ] && extra="--no-tanh"
    [ -f "phase1b_${suffix}_n64/training_report.json" ] || \
      stage "F1_train_${suffix}" "$PY" train_phase1b_fork.py --n-latents 64 --suffix "$suffix" \
        --lam 1.4758 --steps 20000 --world-seed "$w" $extra --out "phase1b_${suffix}_n64"
  done
done
echo "tanh A/B training finished $(date -Is)" >> triage.log
