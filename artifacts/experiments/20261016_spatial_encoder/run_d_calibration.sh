#!/usr/bin/env bash
# Read-only calibration (no training): existing D-class worlds on frozen Raw tokens, scored on the audited gates (R64), to measure
# what a D-class learner achieves by budget before any matched encoder comparison is designed.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)"
R="raw@10000"; P="$OUT/spatial_encoder_v1/probes2_raw10000.pt"
LV="$OUT/levers_tworlds_v1"; S1=corrt_raw_teacher_s7_fmamba_di_u36000
W=(D6k=d:$LV/${S1}_at6000.pt D12k=d:$LV/${S1}_at12000.pt D18k=d:$LV/${S1}_at18000.pt D36k=d:$LV/$S1.pt
   D42k=d:$LV/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt
   D45k=d:$OUT/oracle_v1/worlds/control_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt)
export EVAL_DEVICE=cuda OMP_NUM_THREADS=4
$PY location_dynamics.py $(for w in "${W[@]}"; do n=${w%%=*}; c=${w#*=}; echo "$n=$R,$c,$P"; done)
$PY consequence_check.py $(for w in "${W[@]}"; do n=${w%%=*}; c=${w#*=}; echo "$n=$R,$c,$P"; done)
export EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES=
$PY location_dynamics.py --analyse rawpred_s1 rawpred_s2 D6k D12k D18k D36k D42k D45k --tag dcal
$PY consequence_within_root.py rawpred_s1 rawpred_s2 full_s1 full_s2 mop_s1 mop_s2 fd_s1 fd_s2 D6k D12k D18k D36k D42k D45k
echo d_calibration_done
