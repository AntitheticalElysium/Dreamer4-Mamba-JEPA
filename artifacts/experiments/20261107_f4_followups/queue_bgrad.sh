#!/bin/sh
# Belief gradient (PLAN 12): F4, 300 microbatches of 4 streams x 64 steps.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; PY=$R/.venv/bin/python
$PY -u -m artifacts.experiments.20261107_f4_followups.belief_gradient artifacts/experiments/20261106_p1f4/run/ckpt_10.pt $D/readings/danger/belief_gradient_f4.json 300 > $D/bgrad_f4.log 2>&1
echo done > $D/bgrad.done
