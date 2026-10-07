#!/usr/bin/env bash
# Frozen local derivatives, separate from research training; every batch is resumable.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e19_local_gradients_s7 1300 $PY -B $L/e19_gradients.py
job e19_train_mechanism_s7 1300 $PY -B $L/e19_train_diagnose.py
