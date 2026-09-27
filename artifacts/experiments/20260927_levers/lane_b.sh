#!/usr/bin/env bash
# Lane B (sequential): Delta-JEPA joint runs, context-length continuation, memory use, then evaluations.
# Every step is idempotent (skips finished outputs), so the lane can be relaunched after an interruption.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
D=artifacts/eda/levers_ldad_v1
run_ldad() { # variant lam
  local out=$D/$1_lam$2
  if [ -f $out/step-010000.pt ]; then return; fi
  rm -rf $out                                  # a partial run cannot resume; restart it from the paired init
  $PY $L/ldad_joint.py --variant $1 --lam $2
}
run_ldad raw 10
run_ldad tc 10
run_ldad raw 1
$PY $L/context_cont.py
$PY $L/memory_use.py
JAX_PLATFORMS=cpu $PY $L/ldad_eval.py
echo LANE_B_DONE
