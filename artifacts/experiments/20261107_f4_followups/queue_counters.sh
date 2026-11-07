#!/bin/sh
# Exact hidden counters (PLAN 11): dev, then train; CPU.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; PY=$R/.venv/bin/python
$PY -u -m artifacts.experiments.20261107_f4_followups.counters dev > $D/counters_dev.log 2>&1
$PY -u -m artifacts.experiments.20261107_f4_followups.counters train > $D/counters_train.log 2>&1
echo done > $D/counters.done
