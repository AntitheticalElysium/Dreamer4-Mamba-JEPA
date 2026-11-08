#!/bin/sh
# Danger readings R1-R3 (PLAN 8): F4 under the canonical core, then F3 under fa6edd1a's (OLDROOT); stops at the first failure.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; M=artifacts.experiments.20261107_f4_followups.danger_readings; PY=$R/.venv/bin/python; O=$D/readings/danger
F4=artifacts/experiments/20261106_p1f4/run/ckpt_10.pt; F3=artifacts/experiments/20261105_p1f3/run/ckpt_10.pt
$PY -u -m $M attn $F4 4 $O/r2_f4.json > $D/readings_r2_f4.log 2>&1
$PY -u -m $M reliance $F4 4 $O/r3_f4.json > $D/readings_r3_f4.log 2>&1
$PY -u -m $M phase $F4 4 $O/r1_f4.json > $D/readings_r1_f4.log 2>&1
cd /tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/e9c84dff-54c3-49e7-8b48-5c9b055ea788/scratchpad/oldroot
$PY -u -m $M attn $F3 16 $O/r2_f3.json > $D/readings_r2_f3.log 2>&1
$PY -u -m $M reliance $F3 16 $O/r3_f3.json > $D/readings_r3_f3.log 2>&1
$PY -u -m $M phase $F3 3 $O/r1_f3_3.json > $D/readings_r1_f3_3.log 2>&1
$PY -u -m $M phase $F3 16 $O/r1_f3_16.json > $D/readings_r1_f3_16.log 2>&1
echo done > $D/readings.done
