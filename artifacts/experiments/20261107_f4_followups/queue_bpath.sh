#!/bin/sh
# Belief-path probe (PLAN 11): F4, after the dev counters exist.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; PY=$R/.venv/bin/python
until grep -q "dev all states equal the labels" $D/counters_dev.log 2>/dev/null; do sleep 10; done
$PY -u -m artifacts.experiments.20261107_f4_followups.belief_path_probe artifacts/experiments/20261106_p1f4/run/ckpt_10.pt $D/readings/danger/belief_path_f4.json > $D/bpath_f4.log 2>&1
echo done > $D/bpath.done
