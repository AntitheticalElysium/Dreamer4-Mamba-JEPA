#!/usr/bin/env bash
# Queue B: Delta-JEPA (LDAD) joint runs paired with the canonical Raw and TC runs.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
[ -f artifacts/eda/levers_ldad_v1/raw_lam10/step-010000.pt ] || $PY $L/ldad_joint.py --variant raw --lam 10
[ -f artifacts/eda/levers_ldad_v1/tc_lam10/step-010000.pt ] || $PY $L/ldad_joint.py --variant tc --lam 10
echo QUEUE_B_DONE
