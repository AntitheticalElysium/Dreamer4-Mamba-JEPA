#!/bin/sh
# Belief probe (PLAN 9): F4 (canonical core), then F3 (fa6edd1a's, OLDROOT); stops at the first failure.
set -e
R=$PWD; D=$R/artifacts/experiments/20261107_f4_followups; M=artifacts.experiments.20261107_f4_followups.belief_probe; PY=$R/.venv/bin/python; O=$D/readings/danger
$PY -u -m $M artifacts/experiments/20261106_p1f4/run/ckpt_10.pt $O/belief_f4.json > $D/probe_f4.log 2>&1
cd /tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/e9c84dff-54c3-49e7-8b48-5c9b055ea788/scratchpad/oldroot
$PY -u -m $M artifacts/experiments/20261105_p1f3/run/ckpt_10.pt $O/belief_f3.json > $D/probe_f3.log 2>&1
echo done > $D/probe.done
