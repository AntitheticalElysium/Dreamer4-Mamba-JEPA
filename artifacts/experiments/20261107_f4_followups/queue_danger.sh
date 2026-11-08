#!/bin/sh
# Danger M1 (GPU) beside M2's bank build (CPU, JAX), then M2's run (GPU); stops at the first failure.
set -e
D=artifacts/experiments/20261107_f4_followups
M=artifacts.experiments.20261107_f4_followups.danger_m
.venv/bin/python -u -m $M m2build > $D/m2build.log 2>&1 &
B=$!
.venv/bin/python -u -m $M m1 > $D/m1.log 2>&1
wait $B
.venv/bin/python -u -m $M m2run > $D/m2run.log 2>&1
echo done > $D/danger.done
