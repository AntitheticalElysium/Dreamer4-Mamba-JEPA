#!/bin/sh
# Danger checks (PLAN 7): D1 / D2 on the CPU beside the GPU jobs; F4 under the canonical core, F3 under fa6edd1a's
# saccade/ (a copy at OLDROOT, with artifacts/ and third_party/ linked to this repository); stops at the first failure.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; M=artifacts.experiments.20261107_f4_followups.danger_checks; PY=$R/.venv/bin/python
OLDROOT=/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/e9c84dff-54c3-49e7-8b48-5c9b055ea788/scratchpad/oldroot
F4=artifacts/experiments/20261106_p1f4/run/ckpt_10.pt; F3=artifacts/experiments/20261105_p1f3/run/ckpt_10.pt
$PY -u -m $M data $D/readings/danger/d1_d2.json > $D/checks_data.log 2>&1 &
B=$!
$PY -u -m $M hud $F4 4 $D/readings/danger/c1_f4.json > $D/checks_c1_f4.log 2>&1
$PY -u -m $M probe $F4 4 $D/readings/danger/c3_f4.json > $D/checks_c3_f4.log 2>&1
cd $OLDROOT
$PY -u -m $M hud $F3 16 $D/readings/danger/c2_f3.json --health > $D/checks_c2_f3.log 2>&1
$PY -u -m $M probe $F3 16 $D/readings/danger/c3_f3.json > $D/checks_c3_f3.log 2>&1
wait $B
echo done > $D/checks.done
