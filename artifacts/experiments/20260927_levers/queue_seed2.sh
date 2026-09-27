#!/usr/bin/env bash
# Queue A2: second training seed for the arms that decide TC vs Raw and the copy path. Idempotent.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
W=artifacts/eda/levers_tworlds_v1
$PY $L/tworld.py --head residual --pool raw --loss suffix --seed 8
$PY $L/tworld.py --head residual --pool tc  --loss suffix --seed 8
$PY $L/tworld.py --head direct   --pool raw --loss suffix --seed 8
$PY $L/teval.py $W/residual_raw_suffix_s8.pt $W/residual_tc_suffix_s8.pt $W/direct_raw_suffix_s8.pt
echo QUEUE_A2_DONE
