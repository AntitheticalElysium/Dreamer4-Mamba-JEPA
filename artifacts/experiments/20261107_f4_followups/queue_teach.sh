#!/bin/sh
# Denoiser teaching (PLAN 13): F4, 150 microbatches of 4 x 64, then M4 on dev.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; PY=$R/.venv/bin/python
$PY -u -m artifacts.experiments.20261107_f4_followups.denoiser_teaching artifacts/experiments/20261106_p1f4/run/ckpt_10.pt $D/readings/danger/denoiser_teaching_f4.json 150 > $D/teach_f4.log 2>&1
echo done > $D/teach.done
