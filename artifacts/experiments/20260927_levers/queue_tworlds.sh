#!/usr/bin/env bash
# Queue A: per-tile world arms (E2, E3b, E5), 6,000 updates each, then their evaluations. Idempotent: finished
# arms are skipped by tworld.py.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
W=artifacts/eda/levers_tworlds_v1
CB=artifacts/eda/levers_codebooks_v1/raw_K4096.pt
$PY $L/tworld.py --head residual --pool raw --loss suffix
$PY $L/tworld.py --head direct   --pool raw --loss suffix
$PY $L/tworld.py --head residual --pool tc  --loss suffix
$PY $L/tworld.py --head gated    --pool raw --loss suffix
$PY $L/tworld.py --head corr     --pool raw --loss suffix
$PY $L/tworld.py --head residual --pool raw --loss teacher
$PY $L/tworld.py --head categorical --pool raw --loss teacher --codebook $CB
$PY $L/tworld.py --head direct   --pool tc  --loss suffix
$PY $L/teval.py $W/residual_raw_suffix_s7.pt $W/direct_raw_suffix_s7.pt $W/residual_tc_suffix_s7.pt \
    $W/gated_raw_suffix_s7.pt $W/corr_raw_suffix_s7.pt $W/residual_raw_teacher_s7.pt \
    $W/categorical_raw_teacher_s7_K4096.pt $W/direct_tc_suffix_s7.pt
$PY $L/teval.py $W/residual_raw_suffix_s7.pt $W/residual_raw_teacher_s7.pt $W/direct_raw_suffix_s7.pt --snap $CB
echo QUEUE_A_DONE
