#!/bin/sh
# Denoiser probe (PLAN 10): F4 (canonical core).
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; PY=$R/.venv/bin/python; O=$D/readings/danger
$PY -u -m artifacts.experiments.20261107_f4_followups.denoiser_probe artifacts/experiments/20261106_p1f4/run/ckpt_10.pt $O/denoiser_f4.json > $D/dprobe_f4.log 2>&1
echo done > $D/dprobe.done
